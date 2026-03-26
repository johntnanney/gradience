#!/usr/bin/env python3
"""Run a complete Gradience preflight pass and emit a standard run bundle.

Usage:
    python3 scripts/run_preflight.py \\
        --inventory-id my_inventory \\
        --adapters adapter1_dir qa1.json adapter2_dir qa2.json ... \\
        [--base-model distilbert-base-uncased] \\
        [--output-root results/preflight_runs]

Each adapter is specified as a pair: adapter_dir qa_file.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def find_previous_run(inventory_root: Path, current_run_id: str) -> Path | None:
    """Find the most recent previous run directory."""
    if not inventory_root.exists():
        return None
    runs = sorted(
        [d for d in inventory_root.iterdir()
         if d.is_dir() and d.name != "latest" and d.name != current_run_id
         and (d / "preflight_summary.json").exists()],
        key=lambda d: d.name,
    )
    return runs[-1] if runs else None


def main():
    parser = argparse.ArgumentParser(description="Run Gradience preflight and emit a standard run bundle")
    parser.add_argument("--inventory-id", required=True, help="Inventory identifier")
    parser.add_argument("--adapters", nargs="+", required=True,
                        help="Alternating adapter_dir qa_file pairs")
    parser.add_argument("--base-model", default=None, help="Base model name")
    parser.add_argument("--output-root", default="results/preflight_runs",
                        help="Root directory for preflight runs")
    args = parser.parse_args()

    # Parse adapter/QA pairs
    if len(args.adapters) % 2 != 0:
        print("Error: --adapters must have an even number of arguments (adapter_dir qa_file pairs)")
        sys.exit(1)

    adapter_pairs = []
    for i in range(0, len(args.adapters), 2):
        adapter_dir = Path(args.adapters[i])
        qa_file = Path(args.adapters[i + 1])
        if not adapter_dir.is_dir():
            print(f"Error: adapter directory not found: {adapter_dir}")
            sys.exit(1)
        if not qa_file.is_file():
            print(f"Error: QA file not found: {qa_file}")
            sys.exit(1)
        adapter_pairs.append((adapter_dir, qa_file))

    # Setup paths
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    inventory_root = Path(args.output_root) / args.inventory_id
    run_dir = inventory_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    qa_dir = run_dir / "qa"
    pair_dir = run_dir / "pair_reports"
    inv_dir = run_dir / "inventory"
    nbr_dir = run_dir / "neighborhoods"

    qa_dir.mkdir(exist_ok=True)
    pair_dir.mkdir(exist_ok=True)
    inv_dir.mkdir(exist_ok=True)
    nbr_dir.mkdir(exist_ok=True)

    # Copy QA artifacts
    print(f"Inventory: {args.inventory_id}")
    print(f"Run: {run_id}")
    print(f"Adapters: {len(adapter_pairs)}")

    for adapter_dir, qa_file in adapter_pairs:
        name = adapter_dir.name
        dest = qa_dir / f"{name}_qa.json"
        dest.write_text(qa_file.read_text())
        print(f"  {name}")

    # Run pairwise merge reports
    print("\nRunning pairwise merge reports...")
    for i, (a_dir, a_qa) in enumerate(adapter_pairs):
        for j in range(i + 1, len(adapter_pairs)):
            b_dir, b_qa = adapter_pairs[j]
            a_name = a_dir.name
            b_name = b_dir.name
            pair_id = f"{a_name}_vs_{b_name}"
            out_json = pair_dir / f"{pair_id}.json"

            print(f"  {pair_id}")
            subprocess.run(
                [
                    sys.executable, "-m", "gradience", "merge-audit",
                    "--adapter-a", str(a_dir),
                    "--adapter-b", str(b_dir),
                    "--source-a-qa", str(qa_dir / f"{a_name}_qa.json"),
                    "--source-b-qa", str(qa_dir / f"{b_name}_qa.json"),
                    "--qa-report",
                    "--emit-report", str(out_json),
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
            )

    # Run inventory summary
    print("\nRunning inventory summary...")
    subprocess.run(
        [
            sys.executable, "-m", "gradience", "summarize-inventory",
            "--qa-dir", str(qa_dir),
            "--report-dir", str(pair_dir),
            "--emit-report", str(inv_dir / "inventory_summary.json"),
        ],
        stdout=(inv_dir / "inventory_summary.terminal.txt").open("w"),
        stderr=subprocess.DEVNULL,
        check=True,
    )

    # Run neighborhoods
    print("Running neighborhoods...")
    subprocess.run(
        [
            sys.executable, "-m", "gradience", "suggest-neighborhoods",
            "--qa-dir", str(qa_dir),
            "--report-dir", str(pair_dir),
            "--emit-report", str(nbr_dir / "neighborhoods.json"),
        ],
        stdout=(nbr_dir / "neighborhoods.terminal.txt").open("w"),
        stderr=subprocess.DEVNULL,
        check=True,
    )

    # Build action plan and run bundle
    print("Building run bundle...")
    from gradience.api import _load_schema_artifacts
    from gradience.vnext.audit.qa_artifact import AdapterQAArtifact
    from gradience.vnext.inventory.run_bundle import emit_run_bundle, update_latest_pointer
    from gradience.vnext.inventory.summary import build_action_plan
    from gradience.vnext.merge.qa_report import MergeQAReport

    qa_artifacts = _load_schema_artifacts(
        files=sorted(qa_dir.glob("*.json")),
        schema_prefix="gradience.adapter_qa/",
        from_dict=AdapterQAArtifact.from_dict,
        strict_input=False,
        malformed_label="QA artifact",
    )
    merge_reports = _load_schema_artifacts(
        files=sorted(pair_dir.glob("*.json")),
        schema_prefix="gradience.merge_qa_report/",
        from_dict=MergeQAReport.from_dict,
        strict_input=False,
        malformed_label="merge report",
    )

    action_plan = build_action_plan(qa_artifacts, merge_reports)

    # Find previous run for comparison
    previous_run = find_previous_run(inventory_root, run_id)

    emit_run_bundle(
        inventory_id=args.inventory_id,
        run_id=run_id,
        run_dir=run_dir,
        qa_artifacts=qa_artifacts,
        merge_reports=merge_reports,
        action_plan=action_plan,
        base_model=args.base_model,
        previous_run_dir=previous_run,
    )

    # Update latest pointer
    update_latest_pointer(inventory_root, run_dir)

    print(f"\nPreflight complete.")
    print(f"  Bundle: {run_dir}")
    print(f"  Latest: {inventory_root / 'latest'}")
    print(f"  Open first: {run_dir / 'preflight_summary.md'}")
    if previous_run:
        print(f"  Comparison: {run_dir / 'compare_to_previous.md'}")


if __name__ == "__main__":
    main()
