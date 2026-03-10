#!/usr/bin/env python3
"""
Parameter sweep orchestrator for merge experiments.

Runs a grid of merge configurations through the audit -> merge -> evaluate
pipeline.  Reuses audit reports across configs that share the same adapter
pair.  Evaluates in two phases: quick-screen (100 samples) then full eval
on the top-k configs.

Usage::

    python scripts/merge_experiment/run_sweep.py \\
        --workspace /workspace/merge_sweep \\
        --base-model mistralai/Mistral-7B-v0.1 \\
        --trained-a-dir /workspace/merge_experiment/trained_adapters/mnli \\
        --trained-b-dir /workspace/merge_experiment/trained_adapters/qnli \\
        --predibase-a-dir /workspace/merge_experiment/adapters/adapter_a \\
        --predibase-b-dir /workspace/merge_experiment/adapters/adapter_b \\
        --quick-samples 100 \\
        --full-samples 500 \\
        --top-k 5

Requires: pip install "gradience[bench]"
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import torch  # noqa: F401
except ImportError:
    print("Error: torch not installed")
    sys.exit(1)

# Gradience core imports
from gradience.vnext.merge import (
    execute_merge,
    merge_audit,
    plan_from_audit,
)

# Sibling imports
sys.path.insert(0, str(Path(__file__).parent))
from evaluate import evaluate_nli  # noqa: E402
from sweep_configs import SWEEP_GRID, SweepConfig, get_experiment_names  # noqa: E402

# ---------------------------------------------------------------------------
# Phase 1: Audit (shared per adapter source)
# ---------------------------------------------------------------------------


def run_audits(
    adapter_dirs: dict[str, tuple[Path, Path]],
    output_dir: Path,
) -> dict[str, Any]:
    """Run merge-audit for each adapter source, caching results.

    Parameters
    ----------
    adapter_dirs : mapping from source name to (adapter_a_dir, adapter_b_dir)
    output_dir : base directory for audit outputs

    Returns
    -------
    Dict mapping source name to MergeAuditReport.
    """
    print("\n" + "=" * 70)
    print("  Phase 1: Running audits")
    print("=" * 70)

    audits = {}
    for source, (a_dir, b_dir) in adapter_dirs.items():
        print(f"\n  Auditing {source} adapter pair...")
        audit_dir = output_dir / f"audit_{source}"
        audit_dir.mkdir(parents=True, exist_ok=True)

        report = merge_audit(
            adapter_a_dir=a_dir,
            adapter_b_dir=b_dir,
            output_dir=audit_dir,
            verbose=True,
        )

        verdict = report.aggregate.get("overall_verdict", "unknown")
        score = report.aggregate.get("compatibility_score", 0.0)
        print(f"  {source}: verdict={verdict}, score={score:.3f}")
        audits[source] = report

    return audits


# ---------------------------------------------------------------------------
# Phase 2: Execute merges
# ---------------------------------------------------------------------------


def run_single_merge(
    config: SweepConfig,
    audit_report: Any,
    adapter_a_dir: Path,
    adapter_b_dir: Path,
    output_dir: Path,
) -> tuple[Path, dict[str, Any]]:
    """Execute a single merge config.

    Returns
    -------
    (merged_dir, merge_metrics) tuple.
    """
    merge_dir = output_dir / f"merged_{config.name}"
    merge_dir.mkdir(parents=True, exist_ok=True)

    kwargs = config.plan_kwargs()
    plan = plan_from_audit(
        config.strategy,
        audit_report,
        str(adapter_a_dir),
        str(adapter_b_dir),
        **kwargs,
    )
    result = execute_merge(plan, merge_dir, verbose=False)

    metrics = {
        "mean_reconstruction_error": round(result.mean_reconstruction_error, 6),
        "max_reconstruction_error": round(result.max_reconstruction_error, 6),
        "time_seconds": round(result.total_time_seconds, 1),
    }
    return merge_dir, metrics


def run_all_merges(
    configs: list[SweepConfig],
    audits: dict[str, Any],
    adapter_dirs: dict[str, tuple[Path, Path]],
    output_dir: Path,
) -> dict[str, dict[str, Any]]:
    """Execute merges for all configs.

    Returns
    -------
    Dict mapping config name to {dir, merge_metrics, config}.
    """
    print("\n" + "=" * 70)
    print(f"  Phase 2: Executing {len(configs)} merges")
    print("=" * 70)

    results = {}
    start = time.monotonic()

    for i, config in enumerate(configs, 1):
        report = audits[config.adapter_source]
        a_dir, b_dir = adapter_dirs[config.adapter_source]

        merge_dir, metrics = run_single_merge(
            config,
            report,
            a_dir,
            b_dir,
            output_dir,
        )

        results[config.name] = {
            "dir": merge_dir,
            "merge_metrics": metrics,
            "config": config,
        }

        elapsed = time.monotonic() - start
        recon = metrics["mean_reconstruction_error"]
        print(f"  [{i}/{len(configs)}] {config.name}: recon={recon:.4f}, {elapsed:.0f}s elapsed")

    total = time.monotonic() - start
    print(f"\n  All merges completed in {total:.0f}s")
    return results


# ---------------------------------------------------------------------------
# Phase 3: Quick-screen evaluation
# ---------------------------------------------------------------------------


def run_quick_screen(
    merged_results: dict[str, dict[str, Any]],
    base_model: str,
    n_samples: int,
    device: str,
    time_budget_seconds: float | None = None,
) -> dict[tuple[str, str], float]:
    """Evaluate all merged configs with a small sample count.

    Returns
    -------
    Dict mapping (config_name, dataset) to accuracy.
    """
    print("\n" + "=" * 70)
    print(f"  Phase 3: Quick-screen evaluation ({n_samples} samples)")
    print("=" * 70)

    quick_results: dict[tuple[str, str], float] = {}
    datasets = ["mnli", "qnli"]
    total_evals = len(merged_results) * len(datasets)
    start = time.monotonic()
    eval_count = 0

    for name, info in merged_results.items():
        info["config"]
        # Use custom templates for all evaluations (fair comparison)
        template_mode = "custom"

        for dataset in datasets:
            eval_count += 1

            # Time budget check
            if time_budget_seconds is not None:
                elapsed = time.monotonic() - start
                if elapsed > time_budget_seconds:
                    print(f"\n  TIME BUDGET EXCEEDED ({elapsed:.0f}s). Completed {eval_count - 1}/{total_evals} evals.")
                    return quick_results

            print(f"\n  [{eval_count}/{total_evals}] {name} on {dataset}...")

            result = evaluate_nli(
                base_model_name=base_model,
                adapter_dir=str(info["dir"]),
                dataset_name=dataset,
                max_samples=n_samples,
                device=device,
                template_mode=template_mode,
            )

            acc = result["accuracy"]
            quick_results[(name, dataset)] = acc

            elapsed = time.monotonic() - start
            eta_per = elapsed / eval_count
            remaining = (total_evals - eval_count) * eta_per
            print(f"    acc={acc:.4f} | {elapsed:.0f}s elapsed, ~{remaining / 60:.0f}min remaining")

    total = time.monotonic() - start
    print(f"\n  Quick-screen completed in {total / 60:.1f} minutes")
    return quick_results


# ---------------------------------------------------------------------------
# Phase 4: Rank and select top-k
# ---------------------------------------------------------------------------


def rank_configs(
    quick_results: dict[tuple[str, str], float],
    merged_results: dict[str, dict[str, Any]],
    top_k: int,
) -> list[str]:
    """Rank configs by combined MNLI+QNLI accuracy, return top-k names."""
    print("\n" + "=" * 70)
    print("  Phase 4: Ranking configs")
    print("=" * 70)

    combined = {}
    for name in merged_results:
        mnli_acc = quick_results.get((name, "mnli"), 0.0)
        qnli_acc = quick_results.get((name, "qnli"), 0.0)
        combined[name] = mnli_acc + qnli_acc

    ranked = sorted(combined, key=lambda n: combined[n], reverse=True)

    print(f"\n  {'Rank':<6s} {'Config':<30s} {'MNLI':>8s} {'QNLI':>8s} {'Combined':>10s}")
    print("  " + "-" * 65)
    for i, name in enumerate(ranked, 1):
        mnli = quick_results.get((name, "mnli"), 0.0)
        qnli = quick_results.get((name, "qnli"), 0.0)
        marker = " ***" if i <= top_k else ""
        print(f"  {i:<6d} {name:<30s} {mnli:8.4f} {qnli:8.4f} {combined[name]:10.4f}{marker}")

    top_names = ranked[:top_k]
    print(f"\n  Selected top-{top_k}: {top_names}")
    return top_names


# ---------------------------------------------------------------------------
# Phase 5: Full evaluation on top-k
# ---------------------------------------------------------------------------


def run_full_eval(
    top_names: list[str],
    merged_results: dict[str, dict[str, Any]],
    base_model: str,
    n_samples: int,
    device: str,
) -> dict[tuple[str, str], dict[str, Any]]:
    """Full evaluation on top-k configs with more samples.

    Returns
    -------
    Dict mapping (config_name, dataset) to full evaluation result dict.
    """
    print("\n" + "=" * 70)
    print(f"  Phase 5: Full evaluation ({n_samples} samples) on top-{len(top_names)}")
    print("=" * 70)

    full_results: dict[tuple[str, str], dict[str, Any]] = {}
    datasets = ["mnli", "qnli"]
    total_evals = len(top_names) * len(datasets)
    start = time.monotonic()
    eval_count = 0

    for name in top_names:
        info = merged_results[name]
        template_mode = "custom"

        for dataset in datasets:
            eval_count += 1
            print(f"\n  [{eval_count}/{total_evals}] {name} on {dataset}...")

            result = evaluate_nli(
                base_model_name=base_model,
                adapter_dir=str(info["dir"]),
                dataset_name=dataset,
                max_samples=n_samples,
                device=device,
                template_mode=template_mode,
            )

            full_results[(name, dataset)] = result

            elapsed = time.monotonic() - start
            acc = result["accuracy"]
            print(f"    acc={acc:.4f} | {elapsed / 60:.1f}min elapsed")

    total = time.monotonic() - start
    print(f"\n  Full evaluation completed in {total / 60:.1f} minutes")
    return full_results


# ---------------------------------------------------------------------------
# Phase 6: Collect results
# ---------------------------------------------------------------------------


def collect_results(
    configs: list[SweepConfig],
    merged_results: dict[str, dict[str, Any]],
    quick_results: dict[tuple[str, str], float],
    full_results: dict[tuple[str, str], dict[str, Any]],
    top_names: list[str],
    quick_samples: int,
    full_samples: int,
    output_dir: Path,
) -> None:
    """Collect all sweep results into summary JSON and markdown."""
    print("\n" + "=" * 70)
    print("  Phase 6: Collecting results")
    print("=" * 70)

    # Quick-screen table
    quick_rows = []
    for config in configs:
        name = config.name
        mnli_acc = quick_results.get((name, "mnli"))
        qnli_acc = quick_results.get((name, "qnli"))
        merge_metrics = merged_results.get(name, {}).get("merge_metrics", {})

        quick_rows.append(
            {
                "config": config.to_dict(),
                "mnli_accuracy": mnli_acc,
                "qnli_accuracy": qnli_acc,
                "combined_accuracy": ((mnli_acc or 0) + (qnli_acc or 0) if mnli_acc is not None else None),
                "mean_reconstruction_error": merge_metrics.get("mean_reconstruction_error"),
                "max_reconstruction_error": merge_metrics.get("max_reconstruction_error"),
            }
        )

    # Full eval table
    full_rows = []
    for name in top_names:
        for dataset in ["mnli", "qnli"]:
            result = full_results.get((name, dataset), {})
            full_rows.append(
                {
                    "config_name": name,
                    "dataset": dataset,
                    "accuracy": result.get("accuracy"),
                    "n_samples": result.get("n_samples"),
                    "time_seconds": result.get("total_time_seconds"),
                }
            )

    summary = {
        "schema_version": "gradience.merge_sweep/v1",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "quick_screen": {
            "n_samples": quick_samples,
            "results": quick_rows,
        },
        "full_eval": {
            "n_samples": full_samples,
            "top_k_configs": top_names,
            "results": full_rows,
        },
    }

    results_dir = output_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Write JSON
    json_path = results_dir / "sweep_results.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  JSON: {json_path}")

    # Write markdown
    md_path = results_dir / "sweep_results.md"
    with open(md_path, "w") as f:
        f.write("# Merge Parameter Sweep Results\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # Quick-screen table
        f.write(f"## Quick Screen ({quick_samples} samples)\n\n")
        f.write(
            f"| {'Config':<30s} | {'Experiment':<20s} | {'Strategy':<18s} | "
            f"{'MNLI':>6s} | {'QNLI':>6s} | {'Combined':>9s} | {'Recon Err':>9s} |\n"
        )
        f.write(f"|{'-' * 31}|{'-' * 21}|{'-' * 19}|{'-' * 8}|{'-' * 8}|{'-' * 11}|{'-' * 11}|\n")

        # Sort by combined accuracy descending
        sorted_rows = sorted(
            quick_rows,
            key=lambda r: r["combined_accuracy"] or 0,
            reverse=True,
        )
        for row in sorted_rows:
            cfg = row["config"]
            mnli = f"{row['mnli_accuracy']:.4f}" if row["mnli_accuracy"] is not None else "  N/A "
            qnli = f"{row['qnli_accuracy']:.4f}" if row["qnli_accuracy"] is not None else "  N/A "
            combined = f"{row['combined_accuracy']:.4f}" if row["combined_accuracy"] is not None else "    N/A  "
            recon = (
                f"{row['mean_reconstruction_error']:.4f}"
                if row["mean_reconstruction_error"] is not None
                else "    N/A  "
            )
            top_marker = " *" if cfg["name"] in top_names else ""
            f.write(
                f"| {cfg['name']:<30s} | {cfg['experiment']:<20s} | "
                f"{cfg['strategy']:<18s} | {mnli} | {qnli} | {combined} | {recon} |"
                f"{top_marker}\n"
            )

        # Full eval table
        if full_rows:
            f.write(f"\n## Full Evaluation ({full_samples} samples) — Top {len(top_names)}\n\n")
            f.write(
                f"| {'Config':<30s} | {'Dataset':<8s} | {'Accuracy':>10s} | {'N Samples':>10s} | {'Time (s)':>10s} |\n"
            )
            f.write(f"|{'-' * 31}|{'-' * 9}|{'-' * 12}|{'-' * 12}|{'-' * 12}|\n")
            for row in full_rows:
                acc = f"{row['accuracy']:.4f}" if row["accuracy"] is not None else "   N/A   "
                ns = f"{row['n_samples']}" if row["n_samples"] is not None else "  N/A"
                ts = f"{row['time_seconds']:.1f}" if row["time_seconds"] is not None else "  N/A"
                f.write(f"| {row['config_name']:<30s} | {row['dataset']:<8s} | {acc:>10s} | {ns:>10s} | {ts:>10s} |\n")

        # Per-experiment summary
        f.write("\n## Per-Experiment Summary\n\n")
        experiments = {}
        for row in sorted_rows:
            exp = row["config"]["experiment"]
            if exp not in experiments:
                experiments[exp] = []
            experiments[exp].append(row)

        for exp_name, rows in experiments.items():
            best = max(rows, key=lambda r: r["combined_accuracy"] or 0)
            worst = min(rows, key=lambda r: r["combined_accuracy"] or 0)
            f.write(f"### {exp_name}\n\n")
            f.write(f"- **Best**: {best['config']['name']} (combined={best['combined_accuracy']:.4f})\n")
            f.write(f"- **Worst**: {worst['config']['name']} (combined={worst['combined_accuracy']:.4f})\n\n")

    print(f"  Markdown: {md_path}")

    # Print quick summary to stdout
    print("\n" + "=" * 70)
    print(f"  {'Config':<30s} {'MNLI':>8s} {'QNLI':>8s} {'Combined':>10s}")
    print("  " + "-" * 60)
    for row in sorted_rows[:10]:
        cfg = row["config"]
        mnli = row["mnli_accuracy"] or 0
        qnli = row["qnli_accuracy"] or 0
        combined = row["combined_accuracy"] or 0
        marker = " ***" if cfg["name"] in top_names else ""
        print(f"  {cfg['name']:<30s} {mnli:8.4f} {qnli:8.4f} {combined:10.4f}{marker}")
    print("=" * 70)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Parameter sweep for merge experiments")
    parser.add_argument(
        "--workspace",
        type=str,
        default="/workspace/merge_sweep",
        help="Working directory for sweep artifacts",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="mistralai/Mistral-7B-v0.1",
        help="Base model HuggingFace identifier",
    )
    parser.add_argument(
        "--trained-a-dir",
        type=str,
        default=None,
        help="Path to self-trained MNLI adapter",
    )
    parser.add_argument(
        "--trained-b-dir",
        type=str,
        default=None,
        help="Path to self-trained QNLI adapter",
    )
    parser.add_argument(
        "--predibase-a-dir",
        type=str,
        default=None,
        help="Path to Predibase MNLI adapter",
    )
    parser.add_argument(
        "--predibase-b-dir",
        type=str,
        default=None,
        help="Path to Predibase QNLI adapter",
    )
    parser.add_argument(
        "--quick-samples",
        type=int,
        default=100,
        help="Samples per eval in quick-screen phase (default: 100)",
    )
    parser.add_argument(
        "--full-samples",
        type=int,
        default=500,
        help="Samples per eval in full evaluation phase (default: 500)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top configs for full evaluation (default: 5)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device for evaluation (default: cuda)",
    )
    parser.add_argument(
        "--time-budget",
        type=int,
        default=0,
        help="Time budget in minutes for quick-screen phase (0 = unlimited)",
    )
    parser.add_argument(
        "--experiments",
        type=str,
        nargs="*",
        default=None,
        help=f"Subset of experiments to run (default: all). Available: {get_experiment_names()}",
    )
    parser.add_argument(
        "--skip-full-eval",
        action="store_true",
        help="Skip the full evaluation phase (quick-screen only)",
    )

    args = parser.parse_args()

    workspace = Path(args.workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    output_dir = workspace / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build adapter dirs mapping
    adapter_dirs: dict[str, tuple[Path, Path]] = {}
    if args.trained_a_dir and args.trained_b_dir:
        adapter_dirs["trained"] = (
            Path(args.trained_a_dir),
            Path(args.trained_b_dir),
        )
    if args.predibase_a_dir and args.predibase_b_dir:
        adapter_dirs["predibase"] = (
            Path(args.predibase_a_dir),
            Path(args.predibase_b_dir),
        )

    if not adapter_dirs:
        parser.error(
            "Provide at least one adapter pair: "
            "--trained-a-dir/--trained-b-dir "
            "and/or --predibase-a-dir/--predibase-b-dir"
        )

    # Filter grid
    configs = list(SWEEP_GRID)
    if args.experiments:
        configs = [c for c in configs if c.experiment in args.experiments]
        print(f"Filtered to {len(configs)} configs from experiments: {args.experiments}")

    # Filter to available adapter sources
    configs = [c for c in configs if c.adapter_source in adapter_dirs]
    if not configs:
        parser.error(f"No configs match available adapter sources. Available: {list(adapter_dirs.keys())}")

    # Banner
    total_start = time.monotonic()
    print("=" * 70)
    print("  Merge Parameter Sweep")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Workspace: {workspace}")
    print(f"  Configs: {len(configs)}")
    print(f"  Quick-screen: {args.quick_samples} samples")
    print(f"  Full eval: {args.full_samples} samples (top-{args.top_k})")
    print(f"  Adapter sources: {list(adapter_dirs.keys())}")
    print("=" * 70)

    # Phase 1: Audits
    audits = run_audits(adapter_dirs, output_dir)

    # Phase 2: Merges
    merged_results = run_all_merges(configs, audits, adapter_dirs, output_dir)

    # Phase 3: Quick-screen
    time_budget_s = args.time_budget * 60 if args.time_budget > 0 else None
    quick_results = run_quick_screen(
        merged_results,
        args.base_model,
        args.quick_samples,
        args.device,
        time_budget_s,
    )

    # Phase 4: Rank
    top_names = rank_configs(quick_results, merged_results, args.top_k)

    # Phase 5: Full eval (optional)
    full_results: dict[tuple[str, str], dict[str, Any]] = {}
    if not args.skip_full_eval:
        full_results = run_full_eval(
            top_names,
            merged_results,
            args.base_model,
            args.full_samples,
            args.device,
        )
    else:
        print("\n  Skipping full evaluation (--skip-full-eval)")

    # Phase 6: Collect
    collect_results(
        configs,
        merged_results,
        quick_results,
        full_results,
        top_names,
        args.quick_samples,
        args.full_samples,
        output_dir,
    )

    total_time = time.monotonic() - total_start
    print(f"\nTotal sweep time: {total_time / 60:.1f} minutes")
    print("Done!")


if __name__ == "__main__":
    main()
