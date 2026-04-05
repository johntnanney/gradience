#!/usr/bin/env python3
"""Run decoder triage inventory pipeline on a trained adapter cohort.

Pipeline:
1) adapter QA artifacts (audit-adapter)
2) pairwise merge audit reports (merge-audit)
3) inventory + neighborhoods summaries
4) canonical preflight bundle + structured action plan JSON
"""

from __future__ import annotations

import argparse
import itertools
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from gradience.api import _load_schema_artifacts
from gradience.vnext.audit.qa_artifact import AdapterQAArtifact
from gradience.vnext.inventory.run_bundle import emit_run_bundle
from gradience.vnext.inventory.summary import build_action_plan, format_action_plan
from gradience.vnext.merge.qa_report import MergeQAReport
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from _common import (
    CohortItem,
    adapter_dir,
    load_cohort_manifest,
    load_json,
    training_manifest_path,
    write_json,
)


def _run(cmd: list[str], *, quiet: bool = False) -> None:
    if quiet:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        subprocess.run(cmd, check=True)


def _load_training_records(adapter_root: Path, items: list[CohortItem]) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for item in items:
        path = training_manifest_path(adapter_root, item.adapter_id)
        if not path.exists():
            raise FileNotFoundError(f"Missing training manifest for {item.adapter_id}: {path}")
        rec = load_json(path)
        if not rec.get("training_complete", False):
            raise RuntimeError(f"Incomplete training manifest for {item.adapter_id}: {path}")
        records[item.adapter_id] = rec
    return records


def _write_structured_action_plan(action_plan: Any, path: Path) -> None:
    payload = {
        "schema": "gradience.inventory_action_plan/v1",
        "exclude": list(action_plan.exclude),
        "same_task_priority": list(action_plan.same_task_priority),
        "cross_task_caution": list(action_plan.cross_task_caution),
        "evaluate_first": list(action_plan.evaluate_first),
        "summary_line": action_plan.summary_line,
        "total_pairs": action_plan.total_pairs,
        "retained_count": action_plan.retained_count,
        "cross_task_count": action_plan.cross_task_count,
        "behavioral_evidence_count": action_plan.behavioral_evidence_count,
        "total_source_count": action_plan.total_source_count,
        "retained_pair_detail": [list(x) for x in action_plan.retained_pair_detail],
        "near_miss_candidates": list(action_plan.near_miss_candidates),
        "near_miss_detail": [list(x) for x in action_plan.near_miss_detail],
        "near_miss_severity": [list(x) for x in action_plan.near_miss_severity],
    }
    write_json(path, payload)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run decoder triage inventory pipeline.")
    parser.add_argument(
        "--manifest",
        default="scripts/decoder_triage_study/cohort_manifest.json",
        help="Path to cohort manifest JSON",
    )
    parser.add_argument(
        "--adapter-dir",
        required=True,
        help="Path to trained adapter root (contains <adapter_id>/training_manifest.json)",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for pipeline artifacts (qa, pair_reports, inventory, neighborhoods)",
    )
    parser.add_argument(
        "--inventory-id",
        default="decoder_merge_triage",
        help="Inventory identifier for preflight bundle metadata",
    )
    parser.add_argument(
        "--strict-qa",
        action="store_true",
        help="Forward strict QA gate to merge-audit",
    )
    parser.add_argument(
        "--quiet-subprocess",
        action="store_true",
        help="Suppress subprocess stdout/stderr for CLI steps",
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    adapter_root = Path(args.adapter_dir).resolve()
    out_root = Path(args.output_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    qa_dir = out_root / "qa"
    report_dir = out_root / "pair_reports"
    inventory_dir = out_root / "inventory"
    neighborhoods_dir = out_root / "neighborhoods"
    for d in (qa_dir, report_dir, inventory_dir, neighborhoods_dir):
        d.mkdir(parents=True, exist_ok=True)

    items = load_cohort_manifest(manifest_path)
    item_by_id = {x.adapter_id: x for x in items}
    training = _load_training_records(adapter_root, items)

    # Step 1: QA artifacts
    for item in items:
        rec = training[item.adapter_id]
        qa_path = qa_dir / f"{item.adapter_id}_qa.json"
        if qa_path.exists():
            continue
        eval_dataset_label = f"{item.dataset}:{item.dataset_subset or 'default'}"
        cmd = [
            "python3",
            "-m",
            "gradience",
            "audit-adapter",
            "--peft-dir",
            str(adapter_dir(adapter_root, item.adapter_id)),
            "--eval-dataset",
            eval_dataset_label,
            "--metric-name",
            str(rec["metric_name"]),
            "--adapter-score",
            str(rec["adapter_score"]),
            "--base-score",
            str(rec["base_score"]),
            "--margin",
            "0.0",
            "--out",
            str(qa_path),
        ]
        if bool(rec.get("higher_is_better", False)):
            cmd.append("--higher-is-better")
        else:
            cmd.append("--lower-is-better")
        _run(cmd, quiet=args.quiet_subprocess)
        print(f"[pipeline] QA: {item.adapter_id}")

    # Step 2: pairwise merge reports
    for a, b in itertools.combinations(items, 2):
        out_json = report_dir / f"{a.adapter_id}_vs_{b.adapter_id}.json"
        if out_json.exists():
            continue
        cmd = [
            "python3",
            "-m",
            "gradience",
            "merge-audit",
            "--adapter-a",
            str(adapter_dir(adapter_root, a.adapter_id)),
            "--adapter-b",
            str(adapter_dir(adapter_root, b.adapter_id)),
            "--source-a-qa",
            str(qa_dir / f"{a.adapter_id}_qa.json"),
            "--source-b-qa",
            str(qa_dir / f"{b.adapter_id}_qa.json"),
            "--qa-report",
            "--emit-report",
            str(out_json),
        ]
        if args.strict_qa:
            cmd.append("--strict-qa")
        _run(cmd, quiet=args.quiet_subprocess)
        print(f"[pipeline] pair: {a.adapter_id} × {b.adapter_id}")

    # Step 3: inventory + neighborhoods
    inv_json = inventory_dir / "inventory_summary.json"
    if not inv_json.exists():
        _run(
            [
                "python3",
                "-m",
                "gradience",
                "summarize-inventory",
                "--qa-dir",
                str(qa_dir),
                "--report-dir",
                str(report_dir),
                "--emit-report",
                str(inv_json),
            ],
            quiet=args.quiet_subprocess,
        )
        print("[pipeline] inventory summary")

    neighborhoods_json = neighborhoods_dir / "neighborhoods.json"
    if not neighborhoods_json.exists():
        _run(
            [
                "python3",
                "-m",
                "gradience",
                "suggest-neighborhoods",
                "--qa-dir",
                str(qa_dir),
                "--report-dir",
                str(report_dir),
                "--emit-report",
                str(neighborhoods_json),
            ],
            quiet=args.quiet_subprocess,
        )
        print("[pipeline] neighborhoods")

    # Step 4: canonical action plan + run bundle
    qa_artifacts = _load_schema_artifacts(
        files=sorted(qa_dir.glob("*.json")),
        schema_prefix="gradience.adapter_qa/",
        from_dict=AdapterQAArtifact.from_dict,
        strict_input=False,
        malformed_label="QA artifact",
    )
    merge_reports = _load_schema_artifacts(
        files=sorted(report_dir.glob("*.json")),
        schema_prefix="gradience.merge_qa_report/",
        from_dict=MergeQAReport.from_dict,
        strict_input=False,
        malformed_label="merge report",
    )
    action_plan = build_action_plan(qa_artifacts, merge_reports)

    run_id = f"decoder_run_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    emit_run_bundle(
        inventory_id=args.inventory_id,
        run_id=run_id,
        run_dir=out_root,
        qa_artifacts=qa_artifacts,
        merge_reports=merge_reports,
        action_plan=action_plan,
        base_model=(items[0].base_model if items else None),
        previous_run_dir=None,
    )
    _write_structured_action_plan(action_plan, out_root / "inventory_action_plan.json")
    (out_root / "inventory_action_plan.txt").write_text(format_action_plan(action_plan))

    # Also write a compact lookup table for downstream scripts.
    adapter_table = []
    for adapter_id, rec in training.items():
        item = item_by_id[adapter_id]
        adapter_table.append(
            {
                "adapter_id": adapter_id,
                "task_type": item.task_type,
                "task_family": item.task_family,
                "task_key": item.task_key,
                "dataset": item.dataset,
                "dataset_subset": item.dataset_subset,
                "metric_name": rec["metric_name"],
                "higher_is_better": rec["higher_is_better"],
                "adapter_score": rec["adapter_score"],
                "base_score": rec["base_score"],
                "adapter_dir": rec["adapter_dir"],
            }
        )
    write_json(out_root / "adapter_table.json", {"adapters": adapter_table})

    print(f"\nPipeline complete: {out_root}")
    print(f"Action plan JSON: {out_root / 'inventory_action_plan.json'}")


if __name__ == "__main__":
    main()
