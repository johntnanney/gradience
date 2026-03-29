#!/usr/bin/env python3
"""
Compile adjudication results from existing Gradience studies into the
three-backbone comparison, seed stability, and backbone shift tables
needed for Study S01.

This script works entirely from existing adjudication JSON files.
No GPU or model training required.

Usage:
    python sidecar/benchmarks/compile_adjudication.py

Outputs go to sidecar/results/s01/
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from pathlib import Path

# Resolve paths relative to repo root
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = REPO_ROOT / "results"
SIDECAR_RESULTS = REPO_ROOT / "sidecar" / "results" / "s01"

# Existing adjudication files
DISTILBERT_PATH = RESULTS_DIR / "cross_task_subtype_study_01" / "pairs" / "adjudication_results.json"
ROBERTA_PATH = (
    RESULTS_DIR
    / "task_pair_severity_generalization_study_01"
    / "roberta"
    / "pairs"
    / "adjudication_results.json"
)

# Severity thresholds from Panel P01
THRESHOLDS = {
    "catastrophic": 0.15,
    "severe": 0.10,
    "broad_degradation": 0.05,
    "mild": 0.0,
}


def classify_severity(max_delta: float) -> str:
    """Classify a pair's severity based on worst-case delta."""
    if max_delta > THRESHOLDS["catastrophic"]:
        return "catastrophic"
    elif max_delta > THRESHOLDS["severe"]:
        return "severe"
    elif max_delta > THRESHOLDS["broad_degradation"]:
        return "broad_degradation"
    else:
        return "mild"


def normalize_task_pair(task_a: str, task_b: str) -> tuple[str, str]:
    """Canonical ordering for task pairs."""
    canonical_order = ["qnli", "rte", "mrpc", "sst2"]
    ia = canonical_order.index(task_a) if task_a in canonical_order else 99
    ib = canonical_order.index(task_b) if task_b in canonical_order else 99
    if ia <= ib:
        return (task_a, task_b)
    return (task_b, task_a)


def load_adjudication(path: Path) -> list[dict]:
    """Load and return adjudication results."""
    with open(path) as f:
        return json.load(f)


def extract_task_pair(entry: dict) -> str:
    """Extract canonical task pair key from an adjudication entry."""
    task_a = entry.get("task_a", "")
    task_b = entry.get("task_b", "")
    t1, t2 = normalize_task_pair(task_a, task_b)
    return f"{t1} × {t2}"


def compute_max_delta(entry: dict) -> float:
    """Compute the worst-case delta for a pair (positive = degradation)."""
    da = entry.get("delta_task_a", 0.0)
    db = entry.get("delta_task_b", 0.0)
    return max(da, db)


def extract_seed_variant(entry: dict) -> str:
    """Extract the seed variant label from source names."""
    sa = entry.get("source_a", "")
    sb = entry.get("source_b", "")
    seed_a = sa.split("_s")[-1] if "_s" in sa else "?"
    seed_b = sb.split("_s")[-1] if "_s" in sb else "?"
    return f"s{seed_a}×s{seed_b}"


def is_cross_task(entry: dict) -> bool:
    """Check if a pair is cross-task."""
    dist = entry.get("distance", "")
    if dist:
        return dist != "same_task"
    return entry.get("task_a", "") != entry.get("task_b", "")


def analyze_backbone(entries: list[dict], backbone_name: str) -> dict:
    """Analyze all pairs for a single backbone."""
    cross_task = [e for e in entries if is_cross_task(e)]
    same_task = [e for e in entries if not is_cross_task(e)]

    # Per task-pair aggregation
    pair_data: dict[str, list[dict]] = defaultdict(list)
    for entry in cross_task:
        key = extract_task_pair(entry)
        max_d = compute_max_delta(entry)
        pair_data[key].append({
            "pair_id": entry.get("pair_id", ""),
            "seed_variant": extract_seed_variant(entry),
            "delta_task_a": entry.get("delta_task_a", 0.0),
            "delta_task_b": entry.get("delta_task_b", 0.0),
            "max_delta": max_d,
            "classification": classify_severity(max_d),
            "pair_risk": entry.get("pair_risk", ""),
            "dominant_issue": entry.get("dominant_issue", ""),
        })

    # Per task-pair summary
    pair_summaries = {}
    for task_pair, variants in sorted(pair_data.items()):
        deltas = [v["max_delta"] for v in variants]
        worst = max(deltas)
        best = min(deltas)
        mean = sum(deltas) / len(deltas)
        worst_variant = [v for v in variants if v["max_delta"] == worst][0]

        pair_summaries[task_pair] = {
            "task_pair": task_pair,
            "backbone": backbone_name,
            "n_seed_variants": len(variants),
            "worst_delta": round(worst, 4),
            "best_delta": round(best, 4),
            "mean_delta": round(mean, 4),
            "range": round(worst - best, 4),
            "worst_classification": classify_severity(worst),
            "worst_seed_variant": worst_variant["seed_variant"],
            "all_classifications": [v["classification"] for v in variants],
            "seed_stable": len(set(classify_severity(d) for d in deltas)) == 1,
            "variants": variants,
        }

    # Same-task summary
    same_task_deltas = [compute_max_delta(e) for e in same_task]
    same_task_summary = {
        "mean_delta": round(sum(same_task_deltas) / len(same_task_deltas), 4) if same_task_deltas else 0.0,
        "max_delta": round(max(same_task_deltas), 4) if same_task_deltas else 0.0,
        "n_pairs": len(same_task),
    }

    return {
        "backbone": backbone_name,
        "n_cross_task_pairs": len(cross_task),
        "n_same_task_pairs": len(same_task),
        "same_task_control": same_task_summary,
        "pair_summaries": pair_summaries,
    }


def build_three_backbone_comparison(
    distilbert: dict, roberta: dict, deberta: dict | None = None
) -> list[dict]:
    """Build the comparison table across backbones."""
    all_pairs = sorted(
        set(list(distilbert["pair_summaries"].keys()) + list(roberta["pair_summaries"].keys()))
    )

    rows = []
    for pair in all_pairs:
        row = {"task_pair": pair}
        for name, data in [("distilbert", distilbert), ("roberta", roberta)]:
            if pair in data["pair_summaries"]:
                s = data["pair_summaries"][pair]
                row[f"{name}_worst_delta"] = s["worst_delta"]
                row[f"{name}_classification"] = s["worst_classification"]
                row[f"{name}_worst_seed"] = s["worst_seed_variant"]
                row[f"{name}_range"] = s["range"]
                row[f"{name}_seed_stable"] = s["seed_stable"]
            else:
                row[f"{name}_worst_delta"] = None
                row[f"{name}_classification"] = None

        if deberta and pair in deberta["pair_summaries"]:
            s = deberta["pair_summaries"][pair]
            row["deberta_worst_delta"] = s["worst_delta"]
            row["deberta_classification"] = s["worst_classification"]
            row["deberta_worst_seed"] = s["worst_seed_variant"]
            row["deberta_range"] = s["range"]
            row["deberta_seed_stable"] = s["seed_stable"]

        # Compute shift characterization
        d_class = row.get("distilbert_classification")
        r_class = row.get("roberta_classification")
        d_delta = row.get("distilbert_worst_delta", 0)
        r_delta = row.get("roberta_worst_delta", 0)

        if d_class and r_class:
            if d_class == r_class:
                row["shift_type"] = "stable"
            elif d_delta > r_delta:
                row["shift_type"] = "collapses_on_roberta"
            else:
                row["shift_type"] = "escalates_on_roberta"

            row["delta_ratio"] = round(r_delta / d_delta, 2) if d_delta > 0 else None

        rows.append(row)

    return rows


def build_seed_stability(backbone_analysis: dict) -> list[dict]:
    """Assess seed stability for each task pair on a backbone."""
    rows = []
    for pair, summary in sorted(backbone_analysis["pair_summaries"].items()):
        deltas = [v["max_delta"] for v in summary["variants"]]
        classifications = [v["classification"] for v in summary["variants"]]

        # Coefficient of variation
        mean = sum(deltas) / len(deltas) if deltas else 0
        variance = sum((d - mean) ** 2 for d in deltas) / len(deltas) if len(deltas) > 1 else 0
        std = variance ** 0.5
        cv = round(std / mean, 3) if mean > 0 else 0

        # Threshold-crossing: does the pair cross a P01 threshold across seeds?
        unique_classes = set(classifications)
        crosses_threshold = len(unique_classes) > 1

        rows.append({
            "task_pair": pair,
            "backbone": backbone_analysis["backbone"],
            "n_variants": len(deltas),
            "worst_delta": round(max(deltas), 4),
            "best_delta": round(min(deltas), 4),
            "mean_delta": round(mean, 4),
            "std_delta": round(std, 4),
            "cv": cv,
            "range": round(max(deltas) - min(deltas), 4),
            "seed_stable": not crosses_threshold,
            "classifications": classifications,
            "unique_classifications": sorted(unique_classes),
            "crosses_threshold": crosses_threshold,
        })

    return rows


def main():
    # Create output directory
    SIDECAR_RESULTS.mkdir(parents=True, exist_ok=True)

    # Load existing data
    print("Loading DistilBERT adjudication data...")
    distilbert_raw = load_adjudication(DISTILBERT_PATH)
    print(f"  {len(distilbert_raw)} pairs loaded")

    print("Loading RoBERTa adjudication data...")
    roberta_raw = load_adjudication(ROBERTA_PATH)
    print(f"  {len(roberta_raw)} pairs loaded")

    # Analyze each backbone
    print("\nAnalyzing DistilBERT...")
    distilbert = analyze_backbone(distilbert_raw, "distilbert-base-uncased")

    print("Analyzing RoBERTa...")
    roberta = analyze_backbone(roberta_raw, "roberta-base")

    # Build three-backbone comparison (two backbones for now)
    print("\nBuilding backbone comparison...")
    comparison = build_three_backbone_comparison(distilbert, roberta)

    comparison_path = SIDECAR_RESULTS / "three_backbone_comparison.json"
    with open(comparison_path, "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"  Written to {comparison_path}")

    # Build seed stability tables
    print("\nBuilding seed stability analysis...")
    seed_stability = {
        "distilbert": build_seed_stability(distilbert),
        "roberta": build_seed_stability(roberta),
    }

    seed_path = SIDECAR_RESULTS / "seed_stability.json"
    with open(seed_path, "w") as f:
        json.dump(seed_stability, f, indent=2)
    print(f"  Written to {seed_path}")

    # Save full backbone analyses
    for name, data in [("distilbert", distilbert), ("roberta", roberta)]:
        # Convert defaultdict for serialization
        serializable = json.loads(json.dumps(data, default=str))
        analysis_path = SIDECAR_RESULTS / f"{name}_analysis.json"
        with open(analysis_path, "w") as f:
            json.dump(serializable, f, indent=2)
        print(f"  {name} analysis written to {analysis_path}")

    # Print summary to stdout
    print("\n" + "=" * 80)
    print("BACKBONE COMPARISON SUMMARY")
    print("=" * 80)
    print(f"\n{'Task pair':<20} {'DistilBERT':<25} {'RoBERTa':<25} {'Shift'}")
    print("-" * 80)
    for row in comparison:
        d_str = f"{row.get('distilbert_classification', '?'):>15} ({row.get('distilbert_worst_delta', 0):5.1%})"
        r_str = f"{row.get('roberta_classification', '?'):>15} ({row.get('roberta_worst_delta', 0):5.1%})"
        shift = row.get("shift_type", "?")
        print(f"{row['task_pair']:<20} {d_str:<25} {r_str:<25} {shift}")

    print(f"\n{'Same-task controls:'}")
    print(f"  DistilBERT: mean Δ = {distilbert['same_task_control']['mean_delta']:.1%}, "
          f"max Δ = {distilbert['same_task_control']['max_delta']:.1%}")
    print(f"  RoBERTa:    mean Δ = {roberta['same_task_control']['mean_delta']:.1%}, "
          f"max Δ = {roberta['same_task_control']['max_delta']:.1%}")

    print("\n" + "=" * 80)
    print("SEED STABILITY SUMMARY")
    print("=" * 80)
    for backbone_name in ["distilbert", "roberta"]:
        print(f"\n{backbone_name}:")
        for row in seed_stability[backbone_name]:
            stability = "STABLE" if row["seed_stable"] else "CROSSES THRESHOLD"
            print(
                f"  {row['task_pair']:<20} "
                f"range={row['range']:5.1%}  "
                f"CV={row['cv']:.2f}  "
                f"{stability:>20}  "
                f"classes={row['unique_classifications']}"
            )

    print("\nDone.")


if __name__ == "__main__":
    main()
