#!/usr/bin/env python3
"""Analyze per-example behavior across the Output Example Semantics panel.

Computes all five spec metrics:
  1. Source preservation rate
  2. Joint-source breakage
  3. Neither-source behavior
  4. Confidence/margin change
  5. Error concentration (feeds taxonomy)

Produces:
  - example_behavior_summary.json
  - preservation_breakage_table.json
  - Per-case taxonomy classification for each example
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

PREDICTIONS_DIR = Path(__file__).resolve().parent.parent / "results" / "example_semantics" / "predictions"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "results" / "example_semantics"


def load_case(case_id: str) -> dict:
    path = PREDICTIONS_DIR / f"{case_id}.json"
    with open(path) as f:
        return json.load(f)


def analyze_case(data: dict) -> dict:
    """Compute all five metrics for a single case."""
    case_id = data["case_id"]
    labels = data["labels"]
    preds_a = data["predictions_source_a"]
    preds_b = data["predictions_source_b"]
    preds_m = data["predictions_merged"]
    probs_a = data["probabilities_source_a"]
    probs_b = data["probabilities_source_b"]
    probs_m = data["probabilities_merged"]
    n = data["num_examples"]
    source_b_available = data.get("source_b_available", True)

    # Per-example classification
    categories = []
    for i in range(n):
        l = labels[i]
        pa, pb, pm = preds_a[i], preds_b[i], preds_m[i]
        a_correct = (pa == l)
        b_correct = (pb == l) if source_b_available else None
        m_correct = (pm == l)

        if source_b_available:
            both_correct = a_correct and b_correct
            either_correct = a_correct or b_correct
            sources_agree = (pa == pb)
            sources_disagree = not sources_agree

            if both_correct and m_correct:
                cat = "preserved_consensus"
            elif both_correct and not m_correct:
                cat = "consensus_breakage"
            elif a_correct and not b_correct and m_correct:
                cat = "better_source_preserved"
            elif not a_correct and b_correct and m_correct:
                cat = "better_source_preserved"
            elif a_correct and not b_correct and not m_correct:
                cat = "better_source_loss"
            elif not a_correct and b_correct and not m_correct:
                cat = "better_source_loss"
            elif sources_disagree and not a_correct and not b_correct and not m_correct:
                cat = "shared_failure"
            elif sources_agree and not a_correct and not b_correct and m_correct:
                cat = "merge_recovery"
            elif sources_agree and not a_correct and not b_correct and not m_correct:
                cat = "shared_failure"
            else:
                cat = "other"
        else:
            # Cross-task: only source A is available
            if a_correct and m_correct:
                cat = "preserved_consensus"
            elif a_correct and not m_correct:
                cat = "source_a_loss"
            elif not a_correct and m_correct:
                cat = "merge_recovery"
            elif not a_correct and not m_correct:
                cat = "shared_failure"
            else:
                cat = "other"

        # Confidence analysis
        conf_m = max(probs_m[i]) if probs_m[i] else 0.0
        conf_a = max(probs_a[i]) if probs_a[i] else 0.0
        margin_m = sorted(probs_m[i], reverse=True)
        margin_gap = margin_m[0] - margin_m[1] if len(margin_m) > 1 else margin_m[0]

        # Neither-source behavior (merged pred matches neither source)
        if source_b_available:
            matches_a = (pm == pa)
            matches_b = (pm == pb)
            neither_source = not matches_a and not matches_b
        else:
            matches_a = (pm == pa)
            neither_source = not matches_a  # can only check against source A

        categories.append({
            "idx": i,
            "label": l,
            "pred_a": pa,
            "pred_b": pb if source_b_available else None,
            "pred_m": pm,
            "category": cat,
            "confidence_merged": round(conf_m, 4),
            "confidence_source_a": round(conf_a, 4),
            "margin_gap": round(margin_gap, 4),
            "neither_source": neither_source,
        })

    # Aggregate metrics
    cat_counts = {}
    for c in categories:
        cat_counts[c["category"]] = cat_counts.get(c["category"], 0) + 1

    # Metric 1: Source preservation rate
    if source_b_available:
        either_correct_count = sum(
            1 for i in range(n)
            if preds_a[i] == labels[i] or preds_b[i] == labels[i]
        )
        preserved_count = sum(
            1 for i in range(n)
            if (preds_a[i] == labels[i] or preds_b[i] == labels[i])
            and preds_m[i] == labels[i]
        )
    else:
        either_correct_count = sum(1 for i in range(n) if preds_a[i] == labels[i])
        preserved_count = sum(
            1 for i in range(n)
            if preds_a[i] == labels[i] and preds_m[i] == labels[i]
        )

    preservation_rate = preserved_count / either_correct_count if either_correct_count > 0 else None

    # Metric 2: Joint-source breakage
    if source_b_available:
        both_correct_count = sum(
            1 for i in range(n)
            if preds_a[i] == labels[i] and preds_b[i] == labels[i]
        )
        joint_breakage_count = sum(
            1 for i in range(n)
            if preds_a[i] == labels[i] and preds_b[i] == labels[i]
            and preds_m[i] != labels[i]
        )
    else:
        both_correct_count = sum(1 for i in range(n) if preds_a[i] == labels[i])
        joint_breakage_count = sum(
            1 for i in range(n)
            if preds_a[i] == labels[i] and preds_m[i] != labels[i]
        )
    joint_breakage_rate = joint_breakage_count / both_correct_count if both_correct_count > 0 else None

    # Metric 3: Neither-source behavior rate
    neither_count = sum(1 for c in categories if c["neither_source"])
    neither_rate = neither_count / n

    # Metric 4: Confidence analysis
    conf_merged_mean = np.mean([c["confidence_merged"] for c in categories])
    conf_a_mean = np.mean([c["confidence_source_a"] for c in categories])
    margin_mean = np.mean([c["margin_gap"] for c in categories])

    # Confidence collapse: merged confidence < 0.4 where source A was > 0.6
    conf_collapse_count = sum(
        1 for c in categories
        if c["confidence_merged"] < 0.4 and c["confidence_source_a"] > 0.6
    )
    # High-confidence wrong: merged is wrong with confidence > 0.8
    high_conf_wrong = sum(
        1 for c in categories
        if preds_m[c["idx"]] != labels[c["idx"]] and c["confidence_merged"] > 0.8
    )

    # Metric 5: Error concentration
    error_cats = {}
    for c in categories:
        if preds_m[c["idx"]] != labels[c["idx"]]:
            error_cats[c["category"]] = error_cats.get(c["category"], 0) + 1

    return {
        "case_id": case_id,
        "case_class": data["case_class"],
        "num_examples": n,
        "source_b_available": source_b_available,
        "accuracy_a": data["accuracy_source_a"],
        "accuracy_b": data.get("accuracy_source_b"),
        "accuracy_m": data["accuracy_merged"],
        "metrics": {
            "preservation_rate": round(preservation_rate, 4) if preservation_rate is not None else None,
            "either_correct_count": either_correct_count,
            "preserved_count": preserved_count,
            "joint_breakage_rate": round(joint_breakage_rate, 4) if joint_breakage_rate is not None else None,
            "both_correct_count": both_correct_count,
            "joint_breakage_count": joint_breakage_count,
            "neither_source_rate": round(neither_rate, 4),
            "neither_source_count": neither_count,
            "confidence_merged_mean": round(float(conf_merged_mean), 4),
            "confidence_source_a_mean": round(float(conf_a_mean), 4),
            "margin_gap_mean": round(float(margin_mean), 4),
            "confidence_collapse_count": conf_collapse_count,
            "high_confidence_wrong_count": high_conf_wrong,
        },
        "category_counts": cat_counts,
        "error_concentration": error_cats,
        "per_example": categories,
    }


def main():
    cases = ["SR-01", "SR-02", "FR-01", "FR-02", "CT-01", "NM-01", "NM-02", "AN-01"]

    all_results = []
    summary_rows = []

    for case_id in cases:
        data = load_case(case_id)
        result = analyze_case(data)
        all_results.append(result)

        # Summary row (without per-example detail)
        row = {k: v for k, v in result.items() if k != "per_example"}
        summary_rows.append(row)

        m = result["metrics"]
        print(f"{case_id:6s} ({result['case_class']:15s}) | "
              f"pres={m['preservation_rate']:.3f}  "
              f"brk={m['joint_breakage_rate']:.3f}  "
              f"neither={m['neither_source_rate']:.3f}  "
              f"conf_coll={m['confidence_collapse_count']:3d}  "
              f"hi_conf_wrong={m['high_confidence_wrong_count']:3d}")

    # Save summary
    with open(OUTPUT_DIR / "example_behavior_summary.json", "w") as f:
        json.dump(summary_rows, f, indent=2)

    # Save preservation/breakage table
    pbt = []
    for r in summary_rows:
        pbt.append({
            "case_id": r["case_id"],
            "case_class": r["case_class"],
            "preservation_rate": r["metrics"]["preservation_rate"],
            "joint_breakage_rate": r["metrics"]["joint_breakage_rate"],
            "neither_source_rate": r["metrics"]["neither_source_rate"],
            "confidence_collapse": r["metrics"]["confidence_collapse_count"],
            "category_counts": r["category_counts"],
        })
    with open(OUTPUT_DIR / "preservation_breakage_table.json", "w") as f:
        json.dump(pbt, f, indent=2)

    # Save full per-example results for taxonomy construction
    for result in all_results:
        with open(OUTPUT_DIR / f"analyzed_{result['case_id']}.json", "w") as f:
            json.dump(result, f, indent=2)

    print(f"\nSaved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
