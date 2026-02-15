#!/usr/bin/env python3
"""
Phase 5: Correlation analysis + report generation.

Loads all audit JSONs + eval results and computes:
  1. Pearson/Spearman correlation between spectral metrics and merge quality
  2. Linear regression: merge_quality ~ overlap + rank_ratio + scale_ratio
  3. Binary classification: predict "bad merge" (>5% degradation)
  4. Per-module-type breakdown
  5. Per-method comparison

Output: correlation_report.json + correlation_report.md

Usage:
    python scripts/m1_experiment/phase5_analyze.py \\
        --config scripts/m1_experiment/m1_config.yaml
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_audit_metrics(audits_dir: Path, pair_name: str, seed: int) -> dict | None:
    """Load aggregate spectral metrics from a merge audit."""
    audit_path = audits_dir / pair_name / f"seed_{seed}" / "merge_audit.json"
    if not audit_path.exists():
        return None
    with open(audit_path) as f:
        data = json.load(f)

    # Extract key metrics from aggregate and layer-level data
    aggregate = data.get("aggregate", {})
    layer_verdicts = data.get("layer_verdicts", [])

    # Compute mean metrics across layers
    overlaps = [lv["metrics"]["mean_overlap"] for lv in layer_verdicts if "metrics" in lv]
    dir_agreements = [
        lv["metrics"].get("directional_agreement", 0.0)
        for lv in layer_verdicts if "metrics" in lv
    ]
    mag_ratios = [
        lv["metrics"].get("magnitude_ratio", 1.0)
        for lv in layer_verdicts if "metrics" in lv
    ]
    stable_rank_ratios = [
        lv["metrics"].get("stable_rank_ratio", 1.0)
        for lv in layer_verdicts if "metrics" in lv
    ]

    def safe_mean(xs):
        return sum(xs) / len(xs) if xs else 0.0

    return {
        "pair": pair_name,
        "seed": seed,
        "overall_verdict": aggregate.get("overall_verdict", "unknown"),
        "compatibility_score": aggregate.get("compatibility_score", 0.0),
        "mean_overlap": safe_mean(overlaps),
        "mean_directional_agreement": safe_mean(dir_agreements),
        "mean_magnitude_ratio": safe_mean(mag_ratios),
        "mean_stable_rank_ratio": safe_mean(stable_rank_ratios),
        "layer_verdicts": layer_verdicts,
    }


def load_eval_result(evals_dir: Path, filename: str) -> dict | None:
    """Load an evaluation result JSON."""
    path = evals_dir / filename
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def extract_accuracy(eval_result: dict | None) -> float | None:
    """Extract the primary accuracy metric from an eval result."""
    if eval_result is None:
        return None
    if eval_result.get("error"):
        return None

    # lm-eval-harness result format
    results = eval_result.get("results", {})
    for task_name, task_results in results.items():
        # Try common metric keys
        for key in ["acc,none", "acc_norm,none", "exact_match,strict-match", "pass@1"]:
            if key in task_results:
                return task_results[key]

    return None


def compute_degradation(
    merged_acc: float | None,
    baseline_a_acc: float | None,
    baseline_b_acc: float | None,
) -> float | None:
    """Compute worst-case degradation from the better baseline."""
    if any(x is None for x in [merged_acc, baseline_a_acc, baseline_b_acc]):
        return None
    best_baseline = max(baseline_a_acc, baseline_b_acc)
    if best_baseline == 0:
        return None
    return (best_baseline - merged_acc) / best_baseline


def main():
    parser = argparse.ArgumentParser(description="M1 Phase 5: Correlation analysis")
    parser.add_argument("--config", required=True, help="Path to m1_config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    workspace = Path(config["runtime"]["workspace"])
    audits_dir = workspace / "audits"
    evals_dir = workspace / "evals"
    analysis_dir = workspace / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    seeds = config["experiment"]["seeds"]
    task_names = list(config["adapters"].keys())
    methods = config["merge"]["methods"]

    pairs = list(itertools.combinations(task_names, 2))

    total_start = time.monotonic()
    print("Phase 5: Correlation analysis")

    # --- Collect all data points ---
    data_points: list[dict[str, Any]] = []

    for task_a, task_b in pairs:
        pair_name = f"{task_a}_{task_b}"
        eval_task_a = config["adapters"][task_a]["eval_task"]
        eval_task_b = config["adapters"][task_b]["eval_task"]

        for seed in seeds:
            audit = load_audit_metrics(audits_dir, pair_name, seed)
            if audit is None:
                continue

            # Load baseline eval results
            baseline_a_result = load_eval_result(
                evals_dir / "individual",
                f"{task_a}_seed_{seed}_{eval_task_a}.json",
            )
            baseline_b_result = load_eval_result(
                evals_dir / "individual",
                f"{task_b}_seed_{seed}_{eval_task_b}.json",
            )
            baseline_a_acc = extract_accuracy(baseline_a_result)
            baseline_b_acc = extract_accuracy(baseline_b_result)

            for method in methods:
                # Load merged eval on task A
                merged_a_result = load_eval_result(
                    evals_dir / "merged",
                    f"{pair_name}_seed_{seed}_{method}_{eval_task_a}.json",
                )
                merged_b_result = load_eval_result(
                    evals_dir / "merged",
                    f"{pair_name}_seed_{seed}_{method}_{eval_task_b}.json",
                )

                merged_a_acc = extract_accuracy(merged_a_result)
                merged_b_acc = extract_accuracy(merged_b_result)

                degradation_a = compute_degradation(merged_a_acc, baseline_a_acc, baseline_a_acc)
                degradation_b = compute_degradation(merged_b_acc, baseline_b_acc, baseline_b_acc)

                # Worst degradation across both tasks
                degradations = [d for d in [degradation_a, degradation_b] if d is not None]
                worst_degradation = max(degradations) if degradations else None

                data_points.append({
                    "pair": pair_name,
                    "seed": seed,
                    "method": method,
                    "mean_overlap": audit["mean_overlap"],
                    "mean_directional_agreement": audit["mean_directional_agreement"],
                    "mean_magnitude_ratio": audit["mean_magnitude_ratio"],
                    "mean_stable_rank_ratio": audit["mean_stable_rank_ratio"],
                    "compatibility_score": audit["compatibility_score"],
                    "merged_acc_task_a": merged_a_acc,
                    "merged_acc_task_b": merged_b_acc,
                    "baseline_acc_a": baseline_a_acc,
                    "baseline_acc_b": baseline_b_acc,
                    "degradation_a": degradation_a,
                    "degradation_b": degradation_b,
                    "worst_degradation": worst_degradation,
                    "is_bad_merge": worst_degradation > 0.05 if worst_degradation is not None else None,
                })

    print(f"  Collected {len(data_points)} data points")

    # --- Statistical analysis ---
    # Filter to valid points
    valid = [p for p in data_points if p["worst_degradation"] is not None]
    print(f"  Valid data points (with eval results): {len(valid)}")

    report: dict[str, Any] = {
        "schema_version": "gradience.m1_analysis/v1",
        "n_total_points": len(data_points),
        "n_valid_points": len(valid),
    }

    if len(valid) >= 5:
        try:
            from scipy import stats
            import numpy as np

            # Extract arrays
            overlaps = np.array([p["mean_overlap"] for p in valid])
            dir_agree = np.array([p["mean_directional_agreement"] for p in valid])
            mag_ratio = np.array([p["mean_magnitude_ratio"] for p in valid])
            rank_ratio = np.array([p["mean_stable_rank_ratio"] for p in valid])
            degradation = np.array([p["worst_degradation"] for p in valid])

            # 1. Correlation analysis
            correlations = {}
            for name, values in [
                ("mean_overlap", overlaps),
                ("directional_agreement", dir_agree),
                ("magnitude_ratio", mag_ratio),
                ("stable_rank_ratio", rank_ratio),
            ]:
                pearson_r, pearson_p = stats.pearsonr(values, degradation)
                spearman_r, spearman_p = stats.spearmanr(values, degradation)
                correlations[name] = {
                    "pearson_r": float(pearson_r),
                    "pearson_p": float(pearson_p),
                    "spearman_r": float(spearman_r),
                    "spearman_p": float(spearman_p),
                }
            report["correlations"] = correlations

            # 2. Linear regression
            from sklearn.linear_model import LinearRegression

            X = np.column_stack([overlaps, rank_ratio, mag_ratio])
            y = degradation
            reg = LinearRegression().fit(X, y)
            r_squared = reg.score(X, y)
            report["linear_regression"] = {
                "r_squared": float(r_squared),
                "coefficients": {
                    "mean_overlap": float(reg.coef_[0]),
                    "stable_rank_ratio": float(reg.coef_[1]),
                    "magnitude_ratio": float(reg.coef_[2]),
                },
                "intercept": float(reg.intercept_),
            }

            # 3. Binary classification: predict bad merge
            bad_labels = np.array([p["is_bad_merge"] for p in valid], dtype=float)
            if bad_labels.sum() > 0 and bad_labels.sum() < len(bad_labels):
                from sklearn.linear_model import LogisticRegression
                from sklearn.metrics import accuracy_score, precision_score, recall_score

                clf = LogisticRegression().fit(X, bad_labels)
                predictions = clf.predict(X)
                report["binary_classification"] = {
                    "accuracy": float(accuracy_score(bad_labels, predictions)),
                    "precision": float(precision_score(bad_labels, predictions, zero_division=0)),
                    "recall": float(recall_score(bad_labels, predictions, zero_division=0)),
                    "n_bad_merges": int(bad_labels.sum()),
                    "n_good_merges": int(len(bad_labels) - bad_labels.sum()),
                }
            else:
                report["binary_classification"] = {
                    "note": "All merges same class -- cannot fit classifier",
                    "n_bad_merges": int(bad_labels.sum()),
                    "n_good_merges": int(len(bad_labels) - bad_labels.sum()),
                }

            # 4. Per-method comparison
            method_stats = {}
            for method in methods:
                method_points = [p for p in valid if p["method"] == method]
                if method_points:
                    degs = [p["worst_degradation"] for p in method_points]
                    n_bad = sum(1 for d in degs if d > 0.05)
                    method_stats[method] = {
                        "n_merges": len(method_points),
                        "mean_degradation": float(np.mean(degs)),
                        "std_degradation": float(np.std(degs, ddof=1)) if len(degs) > 1 else 0.0,
                        "max_degradation": float(max(degs)),
                        "n_bad_merges": n_bad,
                        "bad_merge_rate": n_bad / len(method_points),
                    }
            report["per_method"] = method_stats

        except ImportError as e:
            report["error"] = f"Missing dependency: {e}. Install scipy and scikit-learn."
    else:
        report["error"] = f"Insufficient valid data points ({len(valid)}) for analysis."

    # --- Success criteria evaluation ---
    report["success_criteria"] = {}
    if "linear_regression" in report:
        r2 = report["linear_regression"]["r_squared"]
        report["success_criteria"]["variance_explained"] = {
            "value": r2,
            "threshold": 0.50,
            "met": r2 >= 0.50,
        }
    if "binary_classification" in report and "recall" in report["binary_classification"]:
        recall = report["binary_classification"]["recall"]
        report["success_criteria"]["bad_merge_detection"] = {
            "value": recall,
            "threshold": 0.80,
            "met": recall >= 0.80,
        }

    # Save data points
    report["data_points"] = data_points

    # --- Write outputs ---
    report_json_path = analysis_dir / "correlation_report.json"
    with open(report_json_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Saved: {report_json_path}")

    # Generate markdown report
    md = _generate_markdown(report, config)
    report_md_path = analysis_dir / "correlation_report.md"
    with open(report_md_path, "w") as f:
        f.write(md)
    print(f"  Saved: {report_md_path}")

    elapsed = time.monotonic() - total_start
    print(f"\nPhase 5 complete in {elapsed:.1f}s")

    # Print summary
    if "success_criteria" in report:
        print("\n--- Success Criteria ---")
        for name, criterion in report["success_criteria"].items():
            status = "PASS" if criterion["met"] else "FAIL"
            print(f"  {name}: {criterion['value']:.3f} vs {criterion['threshold']:.2f} [{status}]")


def _generate_markdown(report: dict, config: dict) -> str:
    """Generate human-readable markdown report."""
    lines = [
        f"# M1 Controlled Interference Experiment -- Results",
        "",
        f"**Experiment**: {config['experiment']['name']}",
        f"**Base Model**: {config['experiment']['base_model']}",
        f"**Data Points**: {report['n_valid_points']} valid / {report['n_total_points']} total",
        "",
    ]

    if "correlations" in report:
        lines.extend([
            "## Correlation Analysis",
            "",
            "| Metric | Pearson r | p-value | Spearman r | p-value |",
            "|--------|-----------|---------|------------|---------|",
        ])
        for name, corr in report["correlations"].items():
            lines.append(
                f"| {name} | {corr['pearson_r']:.3f} | {corr['pearson_p']:.4f} "
                f"| {corr['spearman_r']:.3f} | {corr['spearman_p']:.4f} |"
            )
        lines.append("")

    if "linear_regression" in report:
        reg = report["linear_regression"]
        lines.extend([
            "## Linear Regression",
            "",
            f"**R-squared**: {reg['r_squared']:.3f}",
            "",
            "| Feature | Coefficient |",
            "|---------|-------------|",
        ])
        for name, coef in reg["coefficients"].items():
            lines.append(f"| {name} | {coef:.4f} |")
        lines.append(f"| intercept | {reg['intercept']:.4f} |")
        lines.append("")

    if "binary_classification" in report:
        bc = report["binary_classification"]
        if "accuracy" in bc:
            lines.extend([
                "## Bad Merge Detection",
                "",
                f"- **Accuracy**: {bc['accuracy']:.3f}",
                f"- **Precision**: {bc['precision']:.3f}",
                f"- **Recall**: {bc['recall']:.3f}",
                f"- Bad merges: {bc['n_bad_merges']} / {bc['n_bad_merges'] + bc['n_good_merges']}",
                "",
            ])

    if "per_method" in report:
        lines.extend([
            "## Per-Method Comparison",
            "",
            "| Method | Mean Deg. | Std | Max Deg. | Bad Merges |",
            "|--------|-----------|-----|----------|------------|",
        ])
        for method, stats in report["per_method"].items():
            lines.append(
                f"| {method} | {stats['mean_degradation']:.3f} "
                f"| {stats['std_degradation']:.3f} "
                f"| {stats['max_degradation']:.3f} "
                f"| {stats['n_bad_merges']}/{stats['n_merges']} |"
            )
        lines.append("")

    if "success_criteria" in report:
        lines.extend([
            "## Success Criteria",
            "",
        ])
        for name, criterion in report["success_criteria"].items():
            status = "PASS" if criterion["met"] else "FAIL"
            lines.append(
                f"- **{name}**: {criterion['value']:.3f} "
                f"(threshold: {criterion['threshold']:.2f}) -- **{status}**"
            )
        lines.append("")

    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
