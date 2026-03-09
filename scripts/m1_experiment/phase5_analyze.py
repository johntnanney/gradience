#!/usr/bin/env python3
"""
Phase 5: Correlation analysis + report generation (revised M1 protocol).

Loads all audit JSONs + eval results and computes:
  1. Pearson/Spearman correlation between spectral metrics and Q_min
  2. Merge outcome prediction evaluation (AUC-ROC, AP, R^2 on Q_min)
  3. Null control analysis (randomized subspace + layer shuffle)
  4. Calibration analysis (same-task vs cross-task overlap)
  5. Per-weight comparison (linear only, weight grid)

Output: correlation_report.json + correlation_report.md

Usage:
    python3 scripts/m1_experiment/phase5_analyze.py \\
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
    layer_verdicts = data.get("per_layer", data.get("layer_verdicts", []))

    # Compute mean metrics across layers
    overlaps = [lv["metrics"]["mean_overlap"] for lv in layer_verdicts if "metrics" in lv]
    dir_agreements = [lv["metrics"].get("directional_agreement", 0.0) for lv in layer_verdicts if "metrics" in lv]
    mag_ratios = [lv["metrics"].get("magnitude_ratio", 1.0) for lv in layer_verdicts if "metrics" in lv]
    stable_rank_ratios = [lv["metrics"].get("stable_rank_ratio", 1.0) for lv in layer_verdicts if "metrics" in lv]
    scale_bounded_ratios = [lv["metrics"].get("scale_bounded_ratio", 0.0) for lv in layer_verdicts if "metrics" in lv]
    frob_bounded_ratios = [lv["metrics"].get("frob_bounded_ratio", 0.0) for lv in layer_verdicts if "metrics" in lv]

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
        "mean_scale_bounded_ratio": safe_mean(scale_bounded_ratios),
        "mean_frob_bounded_ratio": safe_mean(frob_bounded_ratios),
        "layer_verdicts": layer_verdicts,
    }


def load_calibration_audit_metrics(
    audits_dir: Path,
    task: str,
    seed_a: int,
    seed_b: int,
) -> dict | None:
    """Load aggregate spectral metrics from a calibration audit."""
    cal_name = f"{task}_seeds_{seed_a}_{seed_b}"
    audit_path = audits_dir / "calibration" / cal_name / "merge_audit.json"
    if not audit_path.exists():
        return None
    with open(audit_path) as f:
        data = json.load(f)

    aggregate = data.get("aggregate", {})
    layer_verdicts = data.get("per_layer", data.get("layer_verdicts", []))

    overlaps = [lv["metrics"]["mean_overlap"] for lv in layer_verdicts if "metrics" in lv]
    dir_agreements = [lv["metrics"].get("directional_agreement", 0.0) for lv in layer_verdicts if "metrics" in lv]

    def safe_mean(xs):
        return sum(xs) / len(xs) if xs else 0.0

    return {
        "pair": cal_name,
        "task": task,
        "seed_a": seed_a,
        "seed_b": seed_b,
        "pair_type": "calibration",
        "compatibility_score": aggregate.get("compatibility_score", 0.0),
        "mean_overlap": safe_mean(overlaps),
        "mean_directional_agreement": safe_mean(dir_agreements),
        "layer_verdicts": layer_verdicts,
    }


def load_eval_result(evals_dir: Path, filename: str) -> dict | None:
    """Load an evaluation result JSON."""
    path = evals_dir / filename
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def extract_accuracy(eval_result: dict | None, task_name: str | None = None) -> float | None:
    """Extract the primary accuracy metric from an eval result.

    Args:
        eval_result: Parsed lm-eval results JSON.
        task_name: The lm-eval task name (e.g., "mmlu", "gsm8k", "mbpp").
            If provided, looks for the aggregate key first, which is important
            for tasks like MMLU that return 60+ subtask keys alongside one
            aggregate key.
    """
    if eval_result is None:
        return None
    if eval_result.get("error"):
        return None

    _METRIC_KEYS = [
        "acc,none",
        "acc_norm,none",
        "exact_match,strict-match",
        "exact_match,none",
        "pass_at_1,none",
        "pass@1,none",
        "pass@1",
    ]

    results = eval_result.get("results", {})

    # Prefer the aggregate task key if provided (e.g., "mmlu" over "mmlu_abstract_algebra")
    if task_name and task_name in results:
        task_results = results[task_name]
        for key in _METRIC_KEYS:
            if key in task_results:
                return task_results[key]

    # Fallback: iterate all result keys
    for tname, task_results in results.items():
        for key in _METRIC_KEYS:
            if key in task_results:
                return task_results[key]

        # Last resort: search for any key containing acc/pass
        for key, val in task_results.items():
            if isinstance(val, (int, float)) and (
                "acc" in key or "pass@" in key or "pass_at" in key or "exact_match" in key
            ):
                return val

    return None


def _weight_label(weights: list[float]) -> str:
    """Format weight pair as directory-safe label."""
    return f"linear_w{weights[0]}_{weights[1]}"


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
    weight_grid = config["merge"]["weight_grid"]
    criteria = config.get("success_criteria", {})
    calibration_enabled = config["experiment"].get("calibration_pairs", False)

    pairs = list(itertools.combinations(task_names, 2))
    seed_pairs = list(itertools.combinations(seeds, 2))

    total_start = time.monotonic()
    print("Phase 5: Correlation analysis (revised M1 protocol)")

    # ------------------------------------------------------------------
    # 1. Collect cross-task data points
    # ------------------------------------------------------------------
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
            baseline_a_acc = extract_accuracy(baseline_a_result, eval_task_a)
            baseline_b_acc = extract_accuracy(baseline_b_result, eval_task_b)

            for weights in weight_grid:
                wlabel = _weight_label(weights)

                # Load merged eval on task A and task B
                merged_a_result = load_eval_result(
                    evals_dir / "merged",
                    f"{pair_name}_seed_{seed}_{wlabel}_{eval_task_a}.json",
                )
                merged_b_result = load_eval_result(
                    evals_dir / "merged",
                    f"{pair_name}_seed_{seed}_{wlabel}_{eval_task_b}.json",
                )

                merged_a_acc = extract_accuracy(merged_a_result, eval_task_a)
                merged_b_acc = extract_accuracy(merged_b_result, eval_task_b)

                # Compute outcomes using the outcomes module
                outcomes: dict[str, Any] | None = None
                if all(x is not None for x in [merged_a_acc, merged_b_acc, baseline_a_acc, baseline_b_acc]):
                    from gradience.vnext.merge.outcomes import compute_merge_outcomes

                    outcomes = compute_merge_outcomes(
                        score_A_solo=baseline_a_acc,
                        score_B_solo=baseline_b_acc,
                        score_A_merged=merged_a_acc,
                        score_B_merged=merged_b_acc,
                    )

                data_points.append(
                    {
                        "pair": pair_name,
                        "pair_type": "cross_task",
                        "seed": seed,
                        "weights": weights,
                        "weight_label": wlabel,
                        "mean_overlap": audit["mean_overlap"],
                        "mean_directional_agreement": audit["mean_directional_agreement"],
                        "mean_magnitude_ratio": audit["mean_magnitude_ratio"],
                        "mean_stable_rank_ratio": audit["mean_stable_rank_ratio"],
                        "mean_scale_bounded_ratio": audit["mean_scale_bounded_ratio"],
                        "mean_frob_bounded_ratio": audit["mean_frob_bounded_ratio"],
                        "compatibility_score": audit["compatibility_score"],
                        "merged_acc_task_a": merged_a_acc,
                        "merged_acc_task_b": merged_b_acc,
                        "baseline_acc_a": baseline_a_acc,
                        "baseline_acc_b": baseline_b_acc,
                        "Q_min": outcomes["Q_min"] if outcomes else None,
                        "D": outcomes["D"] if outcomes else None,
                        "retention_A": outcomes["retention_A"] if outcomes else None,
                        "retention_B": outcomes["retention_B"] if outcomes else None,
                        "is_bad_merge": outcomes["is_bad_merge"] if outcomes else None,
                    }
                )

    # ------------------------------------------------------------------
    # 2. Collect calibration data points
    # ------------------------------------------------------------------
    calibration_points: list[dict[str, Any]] = []

    if calibration_enabled:
        for task in task_names:
            eval_task = config["adapters"][task]["eval_task"]
            for seed_a, seed_b in seed_pairs:
                cal_audit = load_calibration_audit_metrics(audits_dir, task, seed_a, seed_b)
                if cal_audit is None:
                    continue

                for weights in weight_grid:
                    wlabel = _weight_label(weights)
                    cal_dir_name = f"{task}_seeds_{seed_a}_{seed_b}"

                    # Baselines: each seed's solo performance on same task
                    baseline_a_result = load_eval_result(
                        evals_dir / "individual",
                        f"{task}_seed_{seed_a}_{eval_task}.json",
                    )
                    baseline_b_result = load_eval_result(
                        evals_dir / "individual",
                        f"{task}_seed_{seed_b}_{eval_task}.json",
                    )
                    baseline_a_acc = extract_accuracy(baseline_a_result, eval_task)
                    baseline_b_acc = extract_accuracy(baseline_b_result, eval_task)

                    merged_a_result = load_eval_result(
                        evals_dir / "merged",
                        f"calibration_{cal_dir_name}_{wlabel}_{eval_task}.json",
                    )
                    merged_acc = extract_accuracy(merged_a_result, eval_task)

                    outcomes = None
                    if all(x is not None for x in [merged_acc, baseline_a_acc, baseline_b_acc]):
                        from gradience.vnext.merge.outcomes import compute_merge_outcomes

                        outcomes = compute_merge_outcomes(
                            score_A_solo=baseline_a_acc,
                            score_B_solo=baseline_b_acc,
                            score_A_merged=merged_acc,
                            score_B_merged=merged_acc,
                        )

                    calibration_points.append(
                        {
                            "pair": cal_dir_name,
                            "pair_type": "calibration",
                            "task": task,
                            "seed_a": seed_a,
                            "seed_b": seed_b,
                            "weights": weights,
                            "weight_label": wlabel,
                            "mean_overlap": cal_audit["mean_overlap"],
                            "mean_directional_agreement": cal_audit["mean_directional_agreement"],
                            "compatibility_score": cal_audit["compatibility_score"],
                            "merged_acc": merged_acc,
                            "baseline_acc_a": baseline_a_acc,
                            "baseline_acc_b": baseline_b_acc,
                            "Q_min": outcomes["Q_min"] if outcomes else None,
                            "D": outcomes["D"] if outcomes else None,
                            "is_bad_merge": outcomes["is_bad_merge"] if outcomes else None,
                        }
                    )

    print(f"  Collected {len(data_points)} cross-task data points")
    print(f"  Collected {len(calibration_points)} calibration data points")

    # ------------------------------------------------------------------
    # 3. Statistical analysis
    # ------------------------------------------------------------------
    valid = [p for p in data_points if p["Q_min"] is not None]
    print(f"  Valid data points (with eval results): {len(valid)}")

    report: dict[str, Any] = {
        "schema_version": "gradience.m1_analysis/v2",
        "n_total_points": len(data_points),
        "n_valid_points": len(valid),
        "n_calibration_points": len(calibration_points),
    }

    if len(valid) >= 5:
        try:
            import numpy as np
            from scipy import stats

            # Extract arrays
            overlaps = np.array([p["mean_overlap"] for p in valid])
            dir_agree = np.array([p["mean_directional_agreement"] for p in valid])
            mag_ratio = np.array([p["mean_magnitude_ratio"] for p in valid])
            rank_ratio = np.array([p["mean_stable_rank_ratio"] for p in valid])
            scale_bounded = np.array([p["mean_scale_bounded_ratio"] for p in valid])
            frob_bounded = np.array([p["mean_frob_bounded_ratio"] for p in valid])
            compat_score = np.array([p["compatibility_score"] for p in valid])
            Q_min_arr = np.array([p["Q_min"] for p in valid])
            D_arr = np.array([p["D"] for p in valid])

            # --- Correlation analysis: spectral metrics vs Q_min ---
            correlations = {}
            for name, values in [
                ("mean_overlap", overlaps),
                ("directional_agreement", dir_agree),
                ("magnitude_ratio", mag_ratio),
                ("stable_rank_ratio", rank_ratio),
                ("scale_bounded_ratio", scale_bounded),
                ("frob_bounded_ratio", frob_bounded),
                ("compatibility_score", compat_score),
            ]:
                pearson_r, pearson_p = stats.pearsonr(values, Q_min_arr)
                spearman_r, spearman_p = stats.spearmanr(values, Q_min_arr)
                correlations[name] = {
                    "vs_Q_min": {
                        "pearson_r": float(pearson_r),
                        "pearson_p": float(pearson_p),
                        "spearman_r": float(spearman_r),
                        "spearman_p": float(spearman_p),
                    },
                }
                # Also correlate against D
                pearson_r_d, pearson_p_d = stats.pearsonr(values, D_arr)
                spearman_r_d, spearman_p_d = stats.spearmanr(values, D_arr)
                correlations[name]["vs_D"] = {
                    "pearson_r": float(pearson_r_d),
                    "pearson_p": float(pearson_p_d),
                    "spearman_r": float(spearman_r_d),
                    "spearman_p": float(spearman_p_d),
                }
            report["correlations"] = correlations

            # --- Merge prediction evaluation ---
            from gradience.vnext.merge.evaluation import merge_prediction_evaluation

            predicted_risk = [1.0 - p["compatibility_score"] for p in valid]
            actual_bad = [p["is_bad_merge"] for p in valid]
            actual_Q_min = [p["Q_min"] for p in valid]

            pe = merge_prediction_evaluation(
                predicted_risk=predicted_risk,
                actual_bad_merge=actual_bad,
                actual_Q_min=actual_Q_min,
                n_calibration_bins=5,
            )
            report["prediction_evaluation"] = pe

            # --- Per-weight comparison ---
            weight_stats = {}
            for weights in weight_grid:
                wlabel = _weight_label(weights)
                weight_points = [p for p in valid if p["weight_label"] == wlabel]
                if weight_points:
                    q_vals = [p["Q_min"] for p in weight_points]
                    d_vals = [p["D"] for p in weight_points]
                    n_bad = sum(1 for p in weight_points if p["is_bad_merge"])
                    weight_stats[wlabel] = {
                        "n_merges": len(weight_points),
                        "mean_Q_min": float(np.mean(q_vals)),
                        "std_Q_min": float(np.std(q_vals, ddof=1)) if len(q_vals) > 1 else 0.0,
                        "min_Q_min": float(min(q_vals)),
                        "mean_D": float(np.mean(d_vals)),
                        "std_D": float(np.std(d_vals, ddof=1)) if len(d_vals) > 1 else 0.0,
                        "n_bad_merges": n_bad,
                        "bad_merge_rate": n_bad / len(weight_points),
                    }
            report["per_weight"] = weight_stats

            # --- Null control analysis ---
            # Attempt to run null controls on a sample of audit data.
            # This requires per-layer SVD data stored in the audit JSONs.
            null_control_results = _run_null_controls(audits_dir, pairs, seeds)
            if null_control_results:
                report["null_controls"] = null_control_results

            # --- Calibration analysis ---
            if calibration_points:
                cal_analysis = _calibration_analysis(
                    cross_task_points=valid,
                    calibration_points=calibration_points,
                )
                report["calibration_analysis"] = cal_analysis

        except ImportError as e:
            report["error"] = f"Missing dependency: {e}. Install scipy and scikit-learn."
    else:
        report["error"] = f"Insufficient valid data points ({len(valid)}) for analysis."

    # --- Success criteria evaluation ---
    report["success_criteria"] = {}
    if "prediction_evaluation" in report:
        pe = report["prediction_evaluation"]
        for metric, threshold in [
            ("auc_roc", criteria.get("auc_roc", 0.75)),
            ("average_precision", criteria.get("average_precision", 0.60)),
            ("r_squared_Q_min", criteria.get("r_squared_Q_min", 0.30)),
        ]:
            val = pe.get(metric, 0.0)
            if val is None:
                val = 0.0
            report["success_criteria"][metric] = {
                "value": val,
                "threshold": threshold,
                "met": val >= threshold,
            }

    # Save data points
    report["data_points"] = data_points
    report["calibration_data_points"] = calibration_points

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
    if report.get("success_criteria"):
        print("\n--- Success Criteria ---")
        for name, criterion in report["success_criteria"].items():
            status = "PASS" if criterion["met"] else "FAIL"
            print(f"  {name}: {criterion['value']:.3f} vs {criterion['threshold']:.2f} [{status}]")


# ----------------------------------------------------------------------
# Null control analysis
# ----------------------------------------------------------------------


def _run_null_controls(
    audits_dir: Path,
    pairs: list[tuple[str, str]],
    seeds: list[int],
) -> dict[str, Any] | None:
    """Run null controls on audit data to validate geometric metrics.

    For each audit, loads the per-layer SVD data and runs
    randomized_subspace_control. Compares real overlaps to null
    distribution (real should be significantly higher for cross-task pairs
    and even higher for same-task calibration pairs).

    Returns None if the necessary per-layer data is not available.
    """
    try:
        import numpy as np
        import torch

        from gradience.vnext.merge.null_controls import randomized_subspace_control
    except ImportError:
        return None

    # Sample a subset of audits for efficiency
    sampled: list[dict[str, Any]] = []
    for task_a, task_b in pairs[:3]:  # up to 3 pairs
        for seed in seeds[:1]:  # 1 seed per pair
            pair_name = f"{task_a}_{task_b}"
            audit_path = audits_dir / pair_name / f"seed_{seed}" / "merge_audit.json"
            if not audit_path.exists():
                continue
            with open(audit_path) as f:
                data = json.load(f)

            layer_verdicts = data.get("per_layer", data.get("layer_verdicts", []))
            real_overlaps = []
            null_means = []

            for lv in layer_verdicts:
                metrics = lv.get("metrics", {})
                svd_data = lv.get("svd_data", {})

                real_overlap = metrics.get("mean_overlap")
                if real_overlap is None:
                    continue

                # Need per-layer singular vectors to run null control
                U_A_list = svd_data.get("U_A")
                U_B_list = svd_data.get("U_B")
                if U_A_list is None or U_B_list is None:
                    continue

                U_A = torch.tensor(U_A_list, dtype=torch.float64)
                U_B = torch.tensor(U_B_list, dtype=torch.float64)

                null_result = randomized_subspace_control(U_A, U_B, n_random=10)
                real_overlaps.append(real_overlap)
                null_means.append(null_result["null_mean_of_means"])

            if real_overlaps:
                sampled.append(
                    {
                        "pair": pair_name,
                        "seed": seed,
                        "n_layers_tested": len(real_overlaps),
                        "mean_real_overlap": float(np.mean(real_overlaps)),
                        "mean_null_overlap": float(np.mean(null_means)),
                        "above_null": float(np.mean(real_overlaps)) > float(np.mean(null_means)),
                    }
                )

    if not sampled:
        return None

    return {
        "method": "randomized_subspace_control",
        "n_audits_tested": len(sampled),
        "results": sampled,
        "all_above_null": all(s["above_null"] for s in sampled),
    }


# ----------------------------------------------------------------------
# Calibration analysis (same-task vs cross-task)
# ----------------------------------------------------------------------


def _calibration_analysis(
    cross_task_points: list[dict[str, Any]],
    calibration_points: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compare overlap metrics for same-task pairs vs cross-task pairs.

    Same-task (calibration) pairs should have higher overlap since they
    are trained on the same data distribution with different seeds.
    """
    import numpy as np

    cross_overlaps = [p["mean_overlap"] for p in cross_task_points]
    cal_overlaps = [p["mean_overlap"] for p in calibration_points if p.get("mean_overlap") is not None]

    cross_compat = [p["compatibility_score"] for p in cross_task_points]
    cal_compat = [p["compatibility_score"] for p in calibration_points if p.get("compatibility_score") is not None]

    result: dict[str, Any] = {
        "cross_task_n": len(cross_overlaps),
        "calibration_n": len(cal_overlaps),
    }

    if cross_overlaps:
        result["cross_task_mean_overlap"] = float(np.mean(cross_overlaps))
        result["cross_task_std_overlap"] = float(np.std(cross_overlaps, ddof=1)) if len(cross_overlaps) > 1 else 0.0
    if cal_overlaps:
        result["calibration_mean_overlap"] = float(np.mean(cal_overlaps))
        result["calibration_std_overlap"] = float(np.std(cal_overlaps, ddof=1)) if len(cal_overlaps) > 1 else 0.0

    if cross_compat:
        result["cross_task_mean_compat"] = float(np.mean(cross_compat))
    if cal_compat:
        result["calibration_mean_compat"] = float(np.mean(cal_compat))

    # Statistical test: calibration overlap should be significantly higher
    if len(cross_overlaps) >= 2 and len(cal_overlaps) >= 2:
        from scipy import stats

        t_stat, p_value = stats.ttest_ind(cal_overlaps, cross_overlaps, alternative="greater")
        result["ttest_t_stat"] = float(t_stat)
        result["ttest_p_value"] = float(p_value)
        result["calibration_higher"] = float(np.mean(cal_overlaps)) > float(np.mean(cross_overlaps))

    return result


# ----------------------------------------------------------------------
# Markdown report
# ----------------------------------------------------------------------


def _generate_markdown(report: dict, config: dict) -> str:
    """Generate human-readable markdown report."""
    lines = [
        "# M1 Controlled Interference Experiment -- Results (Revised Protocol)",
        "",
        f"**Experiment**: {config['experiment']['name']}",
        f"**Base Model**: {config['experiment']['base_model']}",
        f"**Merge Method**: linear only (weight grid: {config['merge']['weight_grid']})",
        f"**Data Points**: {report['n_valid_points']} valid / {report['n_total_points']} total cross-task",
        f"**Calibration Points**: {report.get('n_calibration_points', 0)}",
        "",
    ]

    # --- Correlation Analysis ---
    if "correlations" in report:
        lines.extend(
            [
                "## Correlation Analysis (Spectral Metrics vs Q_min)",
                "",
                "| Metric | Pearson r | p-value | Spearman r | p-value |",
                "|--------|-----------|---------|------------|---------|",
            ]
        )
        for name, corr in report["correlations"].items():
            q = corr["vs_Q_min"]
            lines.append(
                f"| {name} | {q['pearson_r']:.3f} | {q['pearson_p']:.4f} "
                f"| {q['spearman_r']:.3f} | {q['spearman_p']:.4f} |"
            )
        lines.append("")

        lines.extend(
            [
                "### Correlation with D (Dominance Index)",
                "",
                "| Metric | Pearson r | p-value | Spearman r | p-value |",
                "|--------|-----------|---------|------------|---------|",
            ]
        )
        for name, corr in report["correlations"].items():
            d = corr["vs_D"]
            lines.append(
                f"| {name} | {d['pearson_r']:.3f} | {d['pearson_p']:.4f} "
                f"| {d['spearman_r']:.3f} | {d['spearman_p']:.4f} |"
            )
        lines.append("")

    # --- Prediction Evaluation ---
    if "prediction_evaluation" in report:
        pe = report["prediction_evaluation"]
        lines.extend(
            [
                "## Merge Outcome Prediction",
                "",
                f"- **AUC-ROC**: {pe['auc_roc']:.3f}"
                if pe.get("auc_roc") is not None
                else "- **AUC-ROC**: N/A (single class)",
                f"- **Average Precision**: {pe['average_precision']:.3f}"
                if pe.get("average_precision") is not None
                else "- **Average Precision**: N/A (single class)",
                f"- **R-squared (Q_min)**: {pe['r_squared_Q_min']:.3f}"
                if pe.get("r_squared_Q_min") is not None
                else "- **R-squared (Q_min)**: N/A",
                f"- Bad merges: {pe['n_bad']} / {pe['n_samples']}",
                "",
            ]
        )

        if pe.get("calibration") and pe["calibration"].get("bin_edges"):
            cal = pe["calibration"]
            lines.extend(
                [
                    "### Calibration (Predicted Risk vs Observed Bad-Merge Rate)",
                    "",
                    "| Bin | Predicted | Observed | Count |",
                    "|----|-----------|----------|-------|",
                ]
            )
            for i, (edges, pred, obs, count) in enumerate(
                zip(
                    cal["bin_edges"],
                    cal["mean_predicted"],
                    cal["mean_observed"],
                    cal["bin_counts"],
                )
            ):
                pred_str = f"{pred:.3f}" if pred is not None else "-"
                obs_str = f"{obs:.3f}" if obs is not None else "-"
                lines.append(f"| [{edges[0]:.2f}, {edges[1]:.2f}) | {pred_str} | {obs_str} | {count} |")
            lines.append("")

    # --- Per-Weight Comparison ---
    if "per_weight" in report:
        lines.extend(
            [
                "## Per-Weight Comparison",
                "",
                "| Weights | Mean Q_min | Std | Min Q_min | Mean D | Bad Merges |",
                "|---------|-----------|-----|-----------|--------|------------|",
            ]
        )
        for wlabel, wstats in report["per_weight"].items():
            lines.append(
                f"| {wlabel} | {wstats['mean_Q_min']:.3f} "
                f"| {wstats['std_Q_min']:.3f} "
                f"| {wstats['min_Q_min']:.3f} "
                f"| {wstats['mean_D']:.3f} "
                f"| {wstats['n_bad_merges']}/{wstats['n_merges']} |"
            )
        lines.append("")

    # --- Null Control Analysis ---
    if "null_controls" in report:
        nc = report["null_controls"]
        lines.extend(
            [
                "## Null Control Analysis",
                "",
                f"- **Method**: {nc['method']}",
                f"- **Audits tested**: {nc['n_audits_tested']}",
                f"- **All above null**: {nc['all_above_null']}",
                "",
                "| Pair | Seed | Layers | Real Overlap | Null Overlap | Above Null |",
                "|------|------|--------|-------------|-------------|------------|",
            ]
        )
        for r in nc["results"]:
            lines.append(
                f"| {r['pair']} | {r['seed']} | {r['n_layers_tested']} "
                f"| {r['mean_real_overlap']:.3f} | {r['mean_null_overlap']:.3f} "
                f"| {r['above_null']} |"
            )
        lines.append("")

    # --- Calibration Analysis ---
    if "calibration_analysis" in report:
        ca = report["calibration_analysis"]
        lines.extend(
            [
                "## Calibration Analysis (Same-Task vs Cross-Task)",
                "",
            ]
        )
        if "cross_task_mean_overlap" in ca:
            lines.append(f"- **Cross-task mean overlap**: {ca['cross_task_mean_overlap']:.3f} (n={ca['cross_task_n']})")
        if "calibration_mean_overlap" in ca:
            lines.append(
                f"- **Same-task (calibration) mean overlap**: {ca['calibration_mean_overlap']:.3f} (n={ca['calibration_n']})"
            )
        if "calibration_higher" in ca:
            lines.append(f"- **Calibration higher**: {ca['calibration_higher']}")
        if "ttest_p_value" in ca:
            lines.append(f"- **t-test (one-sided)**: t={ca['ttest_t_stat']:.3f}, p={ca['ttest_p_value']:.4f}")
        lines.append("")

    # --- Success Criteria ---
    if report.get("success_criteria"):
        lines.extend(
            [
                "## Success Criteria",
                "",
            ]
        )
        for name, criterion in report["success_criteria"].items():
            status = "PASS" if criterion["met"] else "FAIL"
            lines.append(
                f"- **{name}**: {criterion['value']:.3f} (threshold: {criterion['threshold']:.2f}) -- **{status}**"
            )
        lines.append("")

    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
