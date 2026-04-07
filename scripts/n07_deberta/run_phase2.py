#!/usr/bin/env python3
"""N07 Phase 2: Audit individual adapters + merge-audit all 28 cross-task pairs.

Usage:
    python3 scripts/n07_deberta/run_phase2.py --base-dir /workspace/n07

Steps:
  1. Run `gradience audit-adapter` on each of the 8 adapters → QA artifacts
  2. Run `gradience merge-audit` on all 28 pairs → merge reports
  3. Run `gradience summarize-inventory` → inventory summary
"""

from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
import time
from pathlib import Path


def run_cmd(cmd: list[str], label: str) -> subprocess.CompletedProcess[str]:
    """Run a command, print status, and return result."""
    print(f"  → {label}")
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"    ✗ FAILED ({elapsed:.1f}s)")
        print(f"    stderr: {result.stderr[:500]}")
        return result
    print(f"    ✓ OK ({elapsed:.1f}s)")
    return result


def audit_adapters(base_dir: Path) -> list[Path]:
    """Run gradience audit-adapter on each adapter, return list of QA artifact paths."""
    adapters_dir = base_dir / "adapters"
    qa_dir = base_dir / "qa"
    qa_dir.mkdir(exist_ok=True)

    adapter_dirs = sorted(adapters_dir.iterdir())
    qa_paths = []

    print(f"\n{'=' * 60}")
    print(f"  Phase 2a: Audit {len(adapter_dirs)} adapters")
    print(f"{'=' * 60}\n")

    for adapter_dir in adapter_dirs:
        if not adapter_dir.is_dir():
            continue

        adapter_id = adapter_dir.name
        qa_path = qa_dir / f"{adapter_id}_qa.json"

        # Load source score for behavioral evidence
        score_file = adapter_dir / "source_score.json"
        if not score_file.exists():
            print(f"  SKIP {adapter_id}: no source_score.json")
            continue

        score_data = json.loads(score_file.read_text())
        accuracy = score_data["accuracy"]
        task = score_data["task"]

        # Map task to eval dataset name
        eval_dataset_map = {
            "qnli": "glue/qnli",
            "rte": "glue/rte",
            "mrpc": "glue/mrpc",
            "sst2": "glue/sst2",
        }

        cmd = [
            sys.executable, "-m", "gradience", "audit-adapter",
            "--peft-dir", str(adapter_dir),
            "--base-model", "microsoft/deberta-v3-base",
            "--eval-dataset", eval_dataset_map.get(task, task),
            "--metric-name", "accuracy",
            "--adapter-score", str(accuracy),
            "--base-score", "0.5",  # Random baseline for binary/NLI classification
            "--higher-is-better",
            "--out", str(qa_path),
        ]

        result = run_cmd(cmd, f"audit-adapter {adapter_id} (acc={accuracy:.4f})")
        if result.returncode == 0:
            qa_paths.append(qa_path)

    return qa_paths


def merge_audit_pairs(base_dir: Path, qa_paths: list[Path]) -> list[Path]:
    """Run gradience merge-audit on all 28 adapter pairs."""
    adapters_dir = base_dir / "adapters"
    reports_dir = base_dir / "reports"
    reports_dir.mkdir(exist_ok=True)

    adapter_dirs = sorted([d for d in adapters_dir.iterdir() if d.is_dir()])

    # Build QA path lookup
    qa_lookup: dict[str, Path] = {}
    for qp in qa_paths:
        # qa file named like deberta_qnli_s42_qa.json
        adapter_id = qp.stem.replace("_qa", "")
        qa_lookup[adapter_id] = qp

    pairs = list(itertools.combinations(adapter_dirs, 2))
    report_paths = []

    print(f"\n{'=' * 60}")
    print(f"  Phase 2b: Merge-audit {len(pairs)} pairs")
    print(f"{'=' * 60}\n")

    for i, (adapter_a, adapter_b) in enumerate(pairs, 1):
        a_id = adapter_a.name
        b_id = adapter_b.name
        pair_label = f"{a_id}_vs_{b_id}"
        pair_dir = reports_dir / pair_label
        pair_dir.mkdir(exist_ok=True)
        report_path = reports_dir / f"{pair_label}.json"

        cmd = [
            sys.executable, "-m", "gradience", "merge-audit",
            "--adapter-a", str(adapter_a),
            "--adapter-b", str(adapter_b),
            "--output-dir", str(pair_dir),
            "--emit-report", str(report_path),
            "--verbose",
        ]

        # Add source QA if available
        if a_id in qa_lookup:
            cmd.extend(["--source-a-qa", str(qa_lookup[a_id])])
        if b_id in qa_lookup:
            cmd.extend(["--source-b-qa", str(qa_lookup[b_id])])

        result = run_cmd(cmd, f"[{i}/{len(pairs)}] {pair_label}")
        if result.returncode == 0:
            report_paths.append(report_path)

    return report_paths


def summarize_inventory(base_dir: Path) -> None:
    """Run gradience summarize-inventory over all QA + reports."""
    qa_dir = base_dir / "qa"
    reports_dir = base_dir / "reports"
    inventory_dir = base_dir / "inventory"
    inventory_dir.mkdir(exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"  Phase 2c: Summarize inventory")
    print(f"{'=' * 60}\n")

    cmd = [
        sys.executable, "-m", "gradience", "summarize-inventory",
        "--qa-dir", str(qa_dir),
        "--report-dir", str(reports_dir),
        "--emit-report", str(inventory_dir / "summary.json"),
        "--emit-bundle", str(inventory_dir / "bundle"),
    ]

    run_cmd(cmd, "summarize-inventory")


def print_summary(base_dir: Path) -> None:
    """Print a summary table of all merge-audit results."""
    reports_dir = base_dir / "reports"

    print(f"\n{'=' * 60}")
    print(f"  Phase 2 Summary")
    print(f"{'=' * 60}\n")

    # Collect results
    results = []
    for report_file in sorted(reports_dir.glob("*.json")):
        if report_file.name.startswith("."):
            continue
        try:
            data = json.loads(report_file.read_text())
            # Extract key fields from merge QA report
            pair_risk = data.get("pair_risk", "?")
            strategy = data.get("recommended_strategy", "?")
            dominant = data.get("dominant_issue", "?")
            results.append({
                "pair": report_file.stem,
                "risk": pair_risk,
                "strategy": strategy,
                "issue": dominant,
            })
        except (json.JSONDecodeError, KeyError):
            continue

    if results:
        # Print table
        print(f"  {'Pair':<50} {'Risk':<12} {'Strategy':<18} {'Issue'}")
        print(f"  {'-'*50} {'-'*12} {'-'*18} {'-'*20}")
        for r in results:
            print(f"  {r['pair']:<50} {r['risk']:<12} {r['strategy']:<18} {r['issue']}")

    # Count by risk level
    risk_counts: dict[str, int] = {}
    for r in results:
        risk = str(r["risk"])
        risk_counts[risk] = risk_counts.get(risk, 0) + 1
    print(f"\n  Risk distribution: {dict(risk_counts)}")
    print(f"  Total pairs: {len(results)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="N07 Phase 2: Audit + Merge-Audit")
    parser.add_argument("--base-dir", type=Path, required=True, help="N07 output directory")
    parser.add_argument("--skip-audit", action="store_true", help="Skip individual adapter audits")
    parser.add_argument("--skip-merge", action="store_true", help="Skip merge-audit pairs")
    parser.add_argument("--skip-inventory", action="store_true", help="Skip inventory summary")
    args = parser.parse_args()

    base_dir = args.base_dir
    if not base_dir.exists():
        print(f"ERROR: {base_dir} does not exist")
        sys.exit(1)

    t0 = time.time()

    # Step 1: Audit individual adapters
    qa_paths: list[Path] = []
    if not args.skip_audit:
        qa_paths = audit_adapters(base_dir)
        print(f"\n  Audited {len(qa_paths)} adapters")
    else:
        # Load existing QA paths
        qa_dir = base_dir / "qa"
        if qa_dir.exists():
            qa_paths = sorted(qa_dir.glob("*_qa.json"))

    # Step 2: Merge-audit all pairs
    if not args.skip_merge:
        report_paths = merge_audit_pairs(base_dir, qa_paths)
        print(f"\n  Merge-audited {len(report_paths)} pairs")

    # Step 3: Inventory summary
    if not args.skip_inventory:
        summarize_inventory(base_dir)

    # Print summary
    print_summary(base_dir)

    elapsed = time.time() - t0
    print(f"\n  Phase 2 total time: {elapsed:.0f}s ({elapsed/60:.1f}m)")


if __name__ == "__main__":
    main()
