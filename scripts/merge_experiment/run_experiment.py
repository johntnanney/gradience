#!/usr/bin/env python3
"""
Master orchestrator for the merge execution & validation experiment.

Runs the full pipeline:
  1. Download MNLI + QNLI adapters from HuggingFace
  2. Run merge-audit on the adapter pair
  3. Execute 3 merge strategies (uniform_linear, audit_aware, overlap_ties)
  4. Evaluate all 5 adapters (2 baselines + 3 merged) on both tasks
  5. Collect results into experiment_results.json

Requires: pip install "gradience[bench]"

Usage::

    python scripts/merge_experiment/run_experiment.py \\
        --workspace /workspace/merge_experiment \\
        --base-model mistralai/Mistral-7B-v0.1 \\
        --adapter-a-repo <hf-repo-for-mnli-adapter> \\
        --adapter-b-repo <hf-repo-for-qnli-adapter> \\
        --max-eval-samples 500
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import torch
except ImportError:
    print("Error: torch not installed")
    sys.exit(1)

# Gradience imports (core — no bench deps needed)
from gradience.vnext.merge import (
    merge_audit,
    plan_from_audit,
    execute_merge,
    PLAN_STRATEGIES,
)


# ---------------------------------------------------------------------------
# Phase 1: Download adapters
# ---------------------------------------------------------------------------


def download_adapters(args: argparse.Namespace) -> tuple[Path, Path]:
    """Download adapters from HuggingFace Hub."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("Error: huggingface_hub not installed.")
        print('Install with: pip install "gradience[bench]"')
        sys.exit(1)

    workspace = Path(args.workspace)
    adapters_dir = workspace / "adapters"
    adapters_dir.mkdir(parents=True, exist_ok=True)

    adapter_a_dir = adapters_dir / "adapter_a"
    adapter_b_dir = adapters_dir / "adapter_b"

    if not adapter_a_dir.exists() or not any(adapter_a_dir.iterdir()):
        print(f"\n--- Downloading adapter A: {args.adapter_a_repo} ---")
        snapshot_download(
            repo_id=args.adapter_a_repo,
            local_dir=str(adapter_a_dir),
        )
        print(f"  Downloaded to: {adapter_a_dir}")
    else:
        print(f"  Adapter A already cached: {adapter_a_dir}")

    if not adapter_b_dir.exists() or not any(adapter_b_dir.iterdir()):
        print(f"\n--- Downloading adapter B: {args.adapter_b_repo} ---")
        snapshot_download(
            repo_id=args.adapter_b_repo,
            local_dir=str(adapter_b_dir),
        )
        print(f"  Downloaded to: {adapter_b_dir}")
    else:
        print(f"  Adapter B already cached: {adapter_b_dir}")

    return adapter_a_dir, adapter_b_dir


# ---------------------------------------------------------------------------
# Phase 2: Audit
# ---------------------------------------------------------------------------


def run_audit(
    adapter_a_dir: Path,
    adapter_b_dir: Path,
    output_dir: Path,
) -> Any:
    """Run merge-audit and save report."""
    print("\n--- Phase 2: Running merge audit ---")
    audit_dir = output_dir / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)

    report = merge_audit(
        adapter_a_dir=adapter_a_dir,
        adapter_b_dir=adapter_b_dir,
        output_dir=audit_dir,
        verbose=True,
    )

    print(f"  Audit report: {audit_dir / 'merge_audit.json'}")
    print(f"  Overall verdict: {report.aggregate['overall_verdict']}")
    return report


# ---------------------------------------------------------------------------
# Phase 3: Execute merges
# ---------------------------------------------------------------------------


def run_merges(
    report: Any,
    adapter_a_dir: Path,
    adapter_b_dir: Path,
    output_dir: Path,
    output_rank: int = 8,
    output_alpha: float = 16.0,
) -> Dict[str, Path]:
    """Execute all 3 merge strategies."""
    print("\n--- Phase 3: Executing merges ---")
    strategies = ["uniform_linear", "audit_aware", "overlap_ties"]
    merged_dirs: Dict[str, Path] = {}

    for strategy in strategies:
        print(f"\n  Strategy: {strategy}")
        merge_dir = output_dir / f"merged_{strategy}"
        merge_dir.mkdir(parents=True, exist_ok=True)

        plan = plan_from_audit(
            strategy,
            report,
            str(adapter_a_dir),
            str(adapter_b_dir),
            output_rank=output_rank,
            output_alpha=output_alpha,
        )

        result = execute_merge(plan, merge_dir, verbose=True)

        print(
            f"  -> Mean recon error: {result.mean_reconstruction_error:.4f}"
            f"  Max: {result.max_reconstruction_error:.4f}"
            f"  Time: {result.total_time_seconds:.1f}s"
        )

        merged_dirs[strategy] = merge_dir

    return merged_dirs


# ---------------------------------------------------------------------------
# Phase 4: Evaluate
# ---------------------------------------------------------------------------


def run_evaluations(
    base_model: str,
    adapter_a_dir: Path,
    adapter_b_dir: Path,
    merged_dirs: Dict[str, Path],
    output_dir: Path,
    max_samples: int = 500,
    device: str = "cuda",
) -> List[Dict[str, Any]]:
    """Evaluate all adapters on both MNLI and QNLI."""
    print("\n--- Phase 4: Evaluating adapters ---")

    # Import evaluate module from this directory
    sys.path.insert(0, str(Path(__file__).parent))
    from evaluate import evaluate_nli

    eval_dir = output_dir / "evaluations"
    eval_dir.mkdir(parents=True, exist_ok=True)

    # Define evaluation matrix: (adapter_name, adapter_dir_or_none)
    adapters = [
        ("baseline_base_model", None),
        ("baseline_adapter_a", str(adapter_a_dir)),
        ("baseline_adapter_b", str(adapter_b_dir)),
    ]
    for strategy, merge_dir in merged_dirs.items():
        adapters.append((f"merged_{strategy}", str(merge_dir)))

    datasets = ["mnli", "qnli"]
    all_results: List[Dict[str, Any]] = []

    for adapter_name, adapter_dir in adapters:
        for dataset in datasets:
            print(f"\n  Evaluating {adapter_name} on {dataset}...")
            result = evaluate_nli(
                base_model_name=base_model,
                adapter_dir=adapter_dir,
                dataset_name=dataset,
                max_samples=max_samples,
                device=device,
            )
            result["adapter_name"] = adapter_name

            # Save individual result
            result_path = eval_dir / f"{adapter_name}_{dataset}.json"
            with open(result_path, "w") as f:
                # Don't save per-example predictions in the summary
                summary = {k: v for k, v in result.items() if k != "predictions"}
                json.dump(summary, f, indent=2)

            all_results.append(result)

    return all_results


# ---------------------------------------------------------------------------
# Phase 5: Collect results
# ---------------------------------------------------------------------------


def collect_results(
    all_results: List[Dict[str, Any]],
    output_dir: Path,
) -> None:
    """Collect all results into a summary JSON."""
    print("\n--- Phase 5: Collecting results ---")

    # Build summary table
    summary_rows = []
    for r in all_results:
        summary_rows.append({
            "adapter": r["adapter_name"],
            "dataset": r["dataset"],
            "accuracy": r["accuracy"],
            "n_samples": r["n_samples"],
            "time_seconds": r.get("total_time_seconds", 0),
        })

    summary = {
        "schema_version": "gradience.merge_experiment/v1",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "results": summary_rows,
    }

    summary_path = output_dir / "experiment_results.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Print summary table
    print("\n" + "=" * 70)
    print(f"  {'Adapter':<30s} {'Dataset':<8s} {'Accuracy':>10s}")
    print("  " + "-" * 60)
    for row in summary_rows:
        print(
            f"  {row['adapter']:<30s} "
            f"{row['dataset']:<8s} "
            f"{row['accuracy']:10.4f}"
        )
    print("=" * 70)
    print(f"\nFull results: {summary_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Merge Execution & Validation Experiment"
    )
    parser.add_argument(
        "--workspace",
        type=str,
        default="/workspace/merge_experiment",
        help="Working directory for all experiment artifacts",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="mistralai/Mistral-7B-v0.1",
        help="Base model HuggingFace identifier",
    )
    parser.add_argument(
        "--adapter-a-repo",
        type=str,
        required=True,
        help="HuggingFace repo for MNLI adapter",
    )
    parser.add_argument(
        "--adapter-b-repo",
        type=str,
        required=True,
        help="HuggingFace repo for QNLI adapter",
    )
    parser.add_argument(
        "--output-rank",
        type=int,
        default=8,
        help="Target rank for merged adapters (default: 8)",
    )
    parser.add_argument(
        "--output-alpha",
        type=float,
        default=16.0,
        help="LoRA alpha for merged adapters (default: 16.0)",
    )
    parser.add_argument(
        "--max-eval-samples",
        type=int,
        default=500,
        help="Max evaluation samples per dataset (default: 500)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device for evaluation (default: cuda)",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip evaluation phase (only audit + merge)",
    )

    args = parser.parse_args()

    workspace = Path(args.workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    output_dir = workspace / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    start = time.monotonic()
    print("=" * 70)
    print("  Merge Execution & Validation Experiment")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Workspace: {workspace}")
    print("=" * 70)

    # Phase 1: Download
    adapter_a_dir, adapter_b_dir = download_adapters(args)

    # Phase 2: Audit
    report = run_audit(adapter_a_dir, adapter_b_dir, output_dir)

    # Phase 3: Merge
    merged_dirs = run_merges(
        report, adapter_a_dir, adapter_b_dir, output_dir,
        output_rank=args.output_rank,
        output_alpha=args.output_alpha,
    )

    # Phase 4: Evaluate (optional)
    if not args.skip_eval:
        all_results = run_evaluations(
            base_model=args.base_model,
            adapter_a_dir=adapter_a_dir,
            adapter_b_dir=adapter_b_dir,
            merged_dirs=merged_dirs,
            output_dir=output_dir,
            max_samples=args.max_eval_samples,
            device=args.device,
        )

        # Phase 5: Collect
        collect_results(all_results, output_dir)
    else:
        print("\n--- Skipping evaluation (--skip-eval) ---")

    elapsed = time.monotonic() - start
    print(f"\nTotal experiment time: {elapsed / 60:.1f} minutes")
    print("Done!")


if __name__ == "__main__":
    main()
