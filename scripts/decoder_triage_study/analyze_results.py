#!/usr/bin/env python3
"""Post-run analysis for controlled decoder merge triage study."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from _common import load_json, load_cohort_manifest, pair_label, write_json


ENCODER_RETENTION_BASELINE = (0.07, 0.10)  # 7-10%


def _load_pair_reports(report_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for p in sorted(report_dir.glob("*.json")):
        r = load_json(p)
        a_name = Path(r["adapter_a"]["path"]).name
        b_name = Path(r["adapter_b"]["path"]).name
        rows.append(
            {
                "pair_label": pair_label(a_name, b_name),
                "adapter_a": a_name,
                "adapter_b": b_name,
                "pair_risk": r.get("pair_risk", "unknown"),
                "recommended_strategy": r.get("recommended_strategy"),
                "task_relationship": r.get("task_relationship"),
                "advisory_active": r.get("task_relationship_advisory") is not None,
                "verdict_distribution": r.get("verdict_distribution", {}),
            }
        )
    return rows


def _category_means(merge_results: dict[str, Any]) -> dict[str, float]:
    means: dict[str, float] = {}
    for cat in ("retained", "near_miss", "control"):
        rows = [r for r in merge_results["results"] if r.get("category") == cat]
        if rows:
            vals = [float(r["delta_vs_best_source_directional"]) for r in rows]
            means[cat] = float(sum(vals) / len(vals))
    return means


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze controlled decoder merge triage run outputs.")
    parser.add_argument("--study-dir", required=True, help="Study root directory")
    parser.add_argument(
        "--manifest",
        default="scripts/decoder_triage_study/cohort_manifest.json",
        help="Cohort manifest path",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output memo path (default: <study-dir>/study_memo.md)",
    )
    args = parser.parse_args()

    study_dir = Path(args.study_dir).resolve()
    analysis_dir = study_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_cohort_manifest(Path(args.manifest).resolve())
    item_by_id = {x.adapter_id: x for x in manifest}

    pipeline_dir = study_dir / "pipeline_output"
    merge_eval_dir = study_dir / "merge_evaluation"
    if not pipeline_dir.exists():
        # fallback naming
        pipeline_dir = study_dir / "inventory"
    if not merge_eval_dir.exists():
        merge_eval_dir = study_dir / "merges"

    action_plan = load_json(pipeline_dir / "inventory_action_plan.json")
    merge_results = load_json(merge_eval_dir / "merge_results.json")
    pair_reports = _load_pair_reports(pipeline_dir / "pair_reports")

    total_pairs = int(action_plan.get("total_pairs", 0))
    retained = int(action_plan.get("retained_count", 0))
    retention_rate = (retained / total_pairs) if total_pairs else 0.0
    narrow = {
        "total_pairs": total_pairs,
        "retained_count": retained,
        "retention_rate": retention_rate,
        "elimination_rate": 1.0 - retention_rate if total_pairs else 0.0,
        "encoder_baseline_retention_range": list(ENCODER_RETENTION_BASELINE),
        "within_encoder_baseline_band": ENCODER_RETENTION_BASELINE[0] <= retention_rate <= ENCODER_RETENTION_BASELINE[1],
    }
    write_json(analysis_dir / "narrowing_rate.json", narrow)

    means = _category_means(merge_results)
    ordering = {
        "means": means,
        "retained_gt_control": means.get("retained", -999.0) > means.get("control", -999.0),
        "near_miss_gt_control": means.get("near_miss", -999.0) > means.get("control", -999.0),
        "retained_gt_near_miss": means.get("retained", -999.0) >= means.get("near_miss", -999.0),
        "expected_ordering_holds": (
            ("retained" in means)
            and ("control" in means)
            and means.get("retained", -999.0) > means.get("control", -999.0)
            and (
                ("near_miss" not in means)
                or means.get("near_miss", -999.0) > means.get("control", -999.0)
            )
        ),
    }
    write_json(analysis_dir / "prioritization_test.json", ordering)

    # Task-boundary checks
    same_task_total = 0
    cross_task_total = 0
    false_pos = 0
    false_neg = 0
    for row in pair_reports:
        a = row["adapter_a"]
        b = row["adapter_b"]
        if a not in item_by_id or b not in item_by_id:
            continue
        same_task = item_by_id[a].task_key == item_by_id[b].task_key
        advisory = bool(row["advisory_active"])
        if same_task:
            same_task_total += 1
            if advisory:
                false_pos += 1
        else:
            cross_task_total += 1
            if not advisory:
                false_neg += 1

    task_boundary = {
        "same_task_pairs": same_task_total,
        "cross_task_pairs": cross_task_total,
        "false_positives": false_pos,
        "false_negatives": false_neg,
        "same_task_false_positive_rate": (false_pos / same_task_total) if same_task_total else 0.0,
        "cross_task_false_negative_rate": (false_neg / cross_task_total) if cross_task_total else 0.0,
    }
    write_json(analysis_dir / "task_boundary_accuracy.json", task_boundary)

    # Verdict distribution
    verdict_totals: dict[str, int] = {}
    for row in pair_reports:
        for k, v in row.get("verdict_distribution", {}).items():
            verdict_totals[k] = verdict_totals.get(k, 0) + int(v)
    verdict_payload = {
        "total_pairs": len(pair_reports),
        "verdict_totals": verdict_totals,
        "dominant_verdict": (max(verdict_totals, key=verdict_totals.get) if verdict_totals else None),
    }
    write_json(analysis_dir / "verdict_distribution.json", verdict_payload)

    encoder_comparison = {
        "encoder_retention_baseline_range": list(ENCODER_RETENTION_BASELINE),
        "decoder_retention_rate": retention_rate,
        "retention_shift_vs_encoder_midpoint": retention_rate - sum(ENCODER_RETENTION_BASELINE) / 2,
        "notes": [
            "Positive shift means more retained pairs than encoder baseline.",
            "Negative shift means stronger narrowing than encoder baseline.",
        ],
    }
    write_json(analysis_dir / "encoder_comparison.json", encoder_comparison)

    # Overall outcome classification
    holds = bool(ordering["expected_ordering_holds"])
    good_boundary = (task_boundary["false_positives"] == 0)
    nondegenerate = retained > 0 and retained < total_pairs
    if nondegenerate and holds and good_boundary:
        outcome = "success"
    elif nondegenerate:
        outcome = "partial_success"
    else:
        outcome = "negative"

    memo_path = Path(args.output).resolve() if args.output else (study_dir / "study_memo.md")
    lines = []
    lines.append("# Controlled Decoder Merge Triage — Study Memo")
    lines.append("")
    lines.append(f"Outcome: **{outcome}**")
    lines.append("")
    lines.append("## Narrowing rate")
    lines.append("")
    lines.append(
        f"- Retained: {retained}/{total_pairs} "
        f"({retention_rate*100:.1f}% retained, {narrow['elimination_rate']*100:.1f}% eliminated)"
    )
    lines.append(
        f"- Encoder baseline band (retention): "
        f"{ENCODER_RETENTION_BASELINE[0]*100:.1f}%–{ENCODER_RETENTION_BASELINE[1]*100:.1f}%"
    )
    lines.append("")
    lines.append("## Prioritization test")
    lines.append("")
    for cat in ("retained", "near_miss", "control"):
        if cat in means:
            lines.append(f"- {cat}: mean directional Δ vs best source = {means[cat]:+.4f}")
    lines.append(f"- Expected ordering holds: {ordering['expected_ordering_holds']}")
    lines.append("")
    lines.append("## Task-boundary detection")
    lines.append("")
    lines.append(f"- Same-task pairs: {same_task_total}, false positives: {false_pos}")
    lines.append(f"- Cross-task pairs: {cross_task_total}, false negatives: {false_neg}")
    lines.append("")
    lines.append("## Verdict profile")
    lines.append("")
    if verdict_totals:
        for k in sorted(verdict_totals):
            lines.append(f"- {k}: {verdict_totals[k]}")
    else:
        lines.append("- No verdict distribution data found.")
    lines.append("")
    lines.append("## Output artifacts")
    lines.append("")
    lines.append(f"- [narrowing_rate.json]({analysis_dir / 'narrowing_rate.json'})")
    lines.append(f"- [prioritization_test.json]({analysis_dir / 'prioritization_test.json'})")
    lines.append(f"- [task_boundary_accuracy.json]({analysis_dir / 'task_boundary_accuracy.json'})")
    lines.append(f"- [verdict_distribution.json]({analysis_dir / 'verdict_distribution.json'})")
    lines.append(f"- [encoder_comparison.json]({analysis_dir / 'encoder_comparison.json'})")
    memo_path.write_text("\n".join(lines))

    print(f"Wrote analysis memo: {memo_path}")


if __name__ == "__main__":
    main()
