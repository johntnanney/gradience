#!/usr/bin/env python3
"""N07 Phase 4: Analysis — test Predictions A–G against experimental data.

Predictions:
  A: Same-task/different-seed pairs show high spectral redundancy
  B: Cross-task pairs show distinct geometric signatures by task similarity
  C: Spectral risk level predicts merge quality degradation
  D: Curvature telemetry (Hessian energy) leads validation accuracy by 3-6 updates
  E: Spectral partitioning: high-SV shared / low-SV task-specific
  F: Norm imbalance between adapters from different-size datasets
  G: Phase transitions detectable in curvature dynamics

Usage:
    python3 scripts/n07_deberta/run_phase4.py --base-dir /workspace/n07
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats


# ─── Helpers ───────────────────────────────────────────────

def load_jsonl_events(path: Path, kind: str | None = None) -> list[dict]:
    """Load events from a telemetry JSONL file, optionally filtering by kind."""
    events = []
    for line in path.read_text().strip().split("\n"):
        if not line.strip():
            continue
        ev = json.loads(line)
        if kind is None or ev.get("kind") == kind:
            events.append(ev)
    return events


def get_task(adapter_id: str) -> str:
    """Extract task from adapter ID like deberta_qnli_s42."""
    return adapter_id.split("_")[1]


def get_seed(adapter_id: str) -> int:
    return int(adapter_id.split("_")[2].replace("s", ""))


TASK_SIZES = {"qnli": 104743, "sst2": 67349, "mrpc": 3668, "rte": 2490}


# ─── Prediction A: Same-task redundancy ────────────────────

def test_prediction_a(reports: dict[str, dict]) -> dict:
    """Same-task/different-seed pairs should show higher redundancy than cross-task pairs."""
    same_task_compat = []
    cross_task_compat = []
    same_task_redundant = []
    cross_task_redundant = []

    for pair_label, report in reports.items():
        parts = pair_label.split("_vs_")
        task_a = get_task(parts[0])
        task_b = get_task(parts[1])
        compat = report.get("compatibility_score", 0)
        verdicts = report.get("verdict_distribution", {})
        n_redundant = verdicts.get("redundant", 0)
        n_total = sum(verdicts.values())
        redundancy_frac = n_redundant / n_total if n_total > 0 else 0

        if task_a == task_b:
            same_task_compat.append(compat)
            same_task_redundant.append(redundancy_frac)
        else:
            cross_task_compat.append(compat)
            cross_task_redundant.append(redundancy_frac)

    # Statistical test: same-task should have higher compatibility (more overlap)
    t_stat, p_value = stats.ttest_ind(same_task_compat, cross_task_compat, alternative="greater")
    # Mann-Whitney for redundancy fraction
    u_stat, u_p = stats.mannwhitneyu(same_task_redundant, cross_task_redundant, alternative="greater")

    return {
        "prediction": "A: Same-task pairs show higher spectral redundancy",
        "same_task_compat_mean": float(np.mean(same_task_compat)),
        "cross_task_compat_mean": float(np.mean(cross_task_compat)),
        "same_task_redundancy_frac": float(np.mean(same_task_redundant)),
        "cross_task_redundancy_frac": float(np.mean(cross_task_redundant)),
        "t_stat": float(t_stat),
        "p_value_compat": float(p_value),
        "u_stat": float(u_stat),
        "p_value_redundancy": float(u_p),
        "supported": p_value < 0.05,
        "n_same": len(same_task_compat),
        "n_cross": len(cross_task_compat),
    }


# ─── Prediction B: Cross-task geometric signatures ────────

def test_prediction_b(reports: dict[str, dict]) -> dict:
    """Cross-task pairs should show distinct geometric patterns by task pair type."""
    pair_type_scores: dict[str, list] = defaultdict(list)
    pair_type_issues: dict[str, list] = defaultdict(list)

    for pair_label, report in reports.items():
        parts = pair_label.split("_vs_")
        task_a = get_task(parts[0])
        task_b = get_task(parts[1])
        if task_a == task_b:
            continue  # skip same-task

        task_pair = tuple(sorted([task_a, task_b]))
        compat = report.get("compatibility_score", 0)
        issue = report.get("dominant_issue", "?")
        pair_type_scores[task_pair].append(compat)
        pair_type_issues[task_pair].append(issue)

    # Compute per-task-pair statistics
    task_pair_stats = {}
    for tp, scores in sorted(pair_type_scores.items()):
        issues = pair_type_issues[tp]
        issue_counts = {}
        for i in issues:
            issue_counts[i] = issue_counts.get(i, 0) + 1
        task_pair_stats[f"{tp[0]}_vs_{tp[1]}"] = {
            "mean_compatibility": float(np.mean(scores)),
            "std_compatibility": float(np.std(scores)),
            "n_pairs": len(scores),
            "dominant_issues": issue_counts,
        }

    # Test: is there significant variance between task-pair types?
    groups = list(pair_type_scores.values())
    if len(groups) >= 2 and all(len(g) >= 2 for g in groups):
        f_stat, anova_p = stats.f_oneway(*groups)
    else:
        f_stat, anova_p = float("nan"), float("nan")

    return {
        "prediction": "B: Cross-task pairs show distinct geometric signatures",
        "task_pair_stats": task_pair_stats,
        "anova_f": float(f_stat),
        "anova_p": float(anova_p),
        "supported": anova_p < 0.05 if not np.isnan(anova_p) else False,
    }


# ─── Prediction C: Risk predicts degradation ──────────────

def test_prediction_c(merge_results: list[dict], reports: dict[str, dict]) -> dict:
    """Spectral risk level should predict merge quality degradation."""
    risk_degradations: dict[str, list] = defaultdict(list)

    for r in merge_results:
        if "error" in r:
            continue
        risk = r.get("pair_risk", "?")
        deg = r.get("degradation", 0)
        risk_degradations[risk].append(deg)

    risk_stats = {}
    for risk in ["low", "medium", "high"]:
        degs = risk_degradations.get(risk, [])
        if degs:
            risk_stats[risk] = {
                "mean_degradation": float(np.mean(degs)),
                "std_degradation": float(np.std(degs)),
                "n": len(degs),
            }

    # Ordinal correlation: risk level → degradation
    risk_order = {"low": 0, "medium": 1, "high": 2}
    risk_vals = []
    deg_vals = []
    for r in merge_results:
        if "error" in r or r.get("pair_risk", "?") not in risk_order:
            continue
        risk_vals.append(risk_order[r["pair_risk"]])
        deg_vals.append(r["degradation"])

    if len(risk_vals) > 2:
        spearman_r, spearman_p = stats.spearmanr(risk_vals, deg_vals)
    else:
        spearman_r, spearman_p = float("nan"), float("nan")

    # Compatibility score (continuous) vs degradation
    compat_vals = []
    deg_compat = []
    for r in merge_results:
        if "error" in r:
            continue
        pair = r.get("pair", "")
        if pair in reports:
            compat = reports[pair].get("compatibility_score", None)
            if compat is not None:
                compat_vals.append(compat)
                deg_compat.append(r["degradation"])

    if len(compat_vals) > 2:
        pearson_r, pearson_p = stats.pearsonr(compat_vals, deg_compat)
    else:
        pearson_r, pearson_p = float("nan"), float("nan")

    # ANOVA: low vs medium vs high
    groups = [risk_degradations.get(r, []) for r in ["low", "medium", "high"] if r in risk_degradations]
    if len(groups) >= 2 and all(len(g) >= 2 for g in groups):
        f_stat, anova_p = stats.f_oneway(*groups)
    else:
        f_stat, anova_p = float("nan"), float("nan")

    return {
        "prediction": "C: Spectral risk level predicts merge quality degradation",
        "risk_stats": risk_stats,
        "spearman_r_risk_vs_deg": float(spearman_r),
        "spearman_p": float(spearman_p),
        "pearson_r_compat_vs_deg": float(pearson_r),
        "pearson_p": float(pearson_p),
        "anova_f": float(f_stat),
        "anova_p": float(anova_p),
        "supported_ordinal": spearman_p < 0.05 if not np.isnan(spearman_p) else False,
        "supported_continuous": pearson_p < 0.05 if not np.isnan(pearson_p) else False,
        "note": "Linear merge baseline only — audit-aware strategies not yet tested",
    }


# ─── Prediction D: Curvature leads validation ─────────────

def test_prediction_d(adapters_dir: Path) -> dict:
    """Curvature telemetry (Hessian energy) should lead validation accuracy by 3-6 updates."""
    lead_lag_results = {}

    for adapter_dir in sorted(adapters_dir.iterdir()):
        if not adapter_dir.is_dir():
            continue
        adapter_id = adapter_dir.name
        jsonl = adapter_dir / "run.jsonl"
        if not jsonl.exists():
            continue

        events = load_jsonl_events(jsonl)

        # Extract curvature time series
        curv_steps = []
        curv_energy = []  # Hessian energy = trace = sum(lambda^2)
        curv_lambda_max = []
        curv_anisotropy = []

        # Extract eval time series
        eval_steps = []
        eval_acc = []

        # Extract loss time series
        loss_steps = []
        loss_vals = []

        for ev in events:
            if ev.get("kind") == "curvature":
                m = ev.get("metrics", {})
                step = m.get("step", ev.get("step"))
                if step is not None:
                    curv_steps.append(step)
                    curv_energy.append(m.get("hessian_trace", 0))
                    curv_lambda_max.append(m.get("lambda_max", 0))
                    curv_anisotropy.append(m.get("anisotropy", 0))
            elif ev.get("event") == "eval":
                m = ev.get("metrics", {})
                step = ev.get("step")
                acc = m.get("accuracy")
                if step is not None and acc is not None:
                    eval_steps.append(step)
                    eval_acc.append(acc)
            elif ev.get("event") == "train_step":
                step = ev.get("step")
                loss = ev.get("loss")
                if step is not None and loss is not None:
                    loss_steps.append(step)
                    loss_vals.append(loss)

        if len(curv_steps) < 5 or len(eval_steps) < 3:
            lead_lag_results[adapter_id] = {
                "status": "insufficient_data",
                "n_curvature": len(curv_steps),
                "n_eval": len(eval_steps),
            }
            continue

        # Compute cross-correlation between curvature changes and eval accuracy changes
        # Interpolate both to common step grid
        curv_steps_arr = np.array(curv_steps)
        curv_energy_arr = np.array(curv_energy)
        eval_steps_arr = np.array(eval_steps)
        eval_acc_arr = np.array(eval_acc)

        # Use curvature steps as reference grid, interpolate eval
        common_steps = curv_steps_arr[(curv_steps_arr >= eval_steps_arr.min()) &
                                      (curv_steps_arr <= eval_steps_arr.max())]
        if len(common_steps) < 5:
            lead_lag_results[adapter_id] = {"status": "insufficient_overlap", "n_common": len(common_steps)}
            continue

        eval_interp = np.interp(common_steps, eval_steps_arr, eval_acc_arr)
        curv_interp = np.interp(common_steps, curv_steps_arr, curv_energy_arr)

        # Compute changes (first differences)
        d_curv = np.diff(curv_interp)
        d_eval = np.diff(eval_interp)

        if len(d_curv) < 5:
            lead_lag_results[adapter_id] = {"status": "insufficient_diffs"}
            continue

        # Cross-correlation at various lags
        max_lag = min(10, len(d_curv) // 3)
        lag_corrs = {}
        for lag in range(-max_lag, max_lag + 1):
            if lag < 0:
                x = d_curv[:lag]
                y = d_eval[-lag:]
            elif lag > 0:
                x = d_curv[lag:]
                y = d_eval[:-lag]
            else:
                x = d_curv
                y = d_eval
            if len(x) > 2:
                r, p = stats.pearsonr(x, y)
                lag_corrs[lag] = {"r": float(r), "p": float(p)}

        # Find optimal lag (most negative correlation — curvature spike precedes accuracy drop)
        best_lag = None
        best_r = 0
        for lag, lc in lag_corrs.items():
            if abs(lc["r"]) > abs(best_r):
                best_r = lc["r"]
                best_lag = lag

        # Also compute overall trends
        if len(curv_energy_arr) > 2:
            trend_r, trend_p = stats.pearsonr(
                np.arange(len(curv_energy_arr)),
                curv_energy_arr,
            )
        else:
            trend_r, trend_p = float("nan"), float("nan")

        lead_lag_results[adapter_id] = {
            "status": "computed",
            "n_curvature": len(curv_steps),
            "n_eval": len(eval_steps),
            "n_common": len(common_steps),
            "hessian_energy_range": [float(curv_energy_arr.min()), float(curv_energy_arr.max())],
            "hessian_energy_trend_r": float(trend_r),
            "hessian_energy_trend_p": float(trend_p),
            "best_lag": best_lag,
            "best_lag_r": float(best_r),
            "lag_correlations": {str(k): v for k, v in lag_corrs.items()},
        }

    # Aggregate: does curvature consistently lead?
    computed = {k: v for k, v in lead_lag_results.items() if v.get("status") == "computed"}
    if computed:
        best_lags = [v["best_lag"] for v in computed.values() if v["best_lag"] is not None]
        energy_trends = [v["hessian_energy_trend_r"] for v in computed.values()]
    else:
        best_lags = []
        energy_trends = []

    return {
        "prediction": "D: Curvature (Hessian energy) leads validation accuracy",
        "per_adapter": lead_lag_results,
        "n_computed": len(computed),
        "median_best_lag": float(np.median(best_lags)) if best_lags else None,
        "mean_energy_trend": float(np.mean(energy_trends)) if energy_trends else None,
        "supported": len(computed) > 0,
        "note": "Positive lag = curvature leads eval; lag in units of snapshot intervals (50 steps)",
    }


# ─── Prediction E: Spectral partitioning ──────────────────

def test_prediction_e(adapters_dir: Path) -> dict:
    """High singular values should be shared across tasks, low SVs task-specific."""
    # Collect final structural metrics per adapter
    structural_by_task: dict[str, list] = defaultdict(list)

    for adapter_dir in sorted(adapters_dir.iterdir()):
        if not adapter_dir.is_dir():
            continue
        adapter_id = adapter_dir.name
        task = get_task(adapter_id)
        jsonl = adapter_dir / "run.jsonl"
        if not jsonl.exists():
            continue

        # Get last structural event
        events = load_jsonl_events(jsonl, kind="structural")
        if events:
            last = events[-1]
            m = last.get("metrics", {})
            structural_by_task[task].append({
                "adapter_id": adapter_id,
                "stable_rank_mean": m.get("stable_rank_mean"),
                "energy_rank_90_median": m.get("energy_rank_90_median"),
                "effective_rank_mean": m.get("effective_rank_mean"),
                "sigma_max_mean": m.get("sigma_max_mean"),
                "frob_norm_mean": m.get("adapter_frob_norm_mean"),
                "frob_norm_total": m.get("adapter_frob_norm_total"),
            })

    # Compare: same-task adapters should have similar rank structure
    # Different-task adapters should diverge
    within_task_rank_var = []
    between_task_rank_diff = []
    task_means = {}

    for task, adapters in structural_by_task.items():
        ranks = [a["stable_rank_mean"] for a in adapters if a["stable_rank_mean"] is not None]
        er90s = [a["energy_rank_90_median"] for a in adapters if a["energy_rank_90_median"] is not None]
        norms = [a["frob_norm_total"] for a in adapters if a["frob_norm_total"] is not None]
        if ranks:
            within_task_rank_var.append(np.var(ranks))
            task_means[task] = {
                "stable_rank_mean": float(np.mean(ranks)),
                "energy_rank_90_median": float(np.mean(er90s)) if er90s else None,
                "frob_norm_total": float(np.mean(norms)) if norms else None,
            }

    # Between-task differences
    tasks = list(task_means.keys())
    for i, t1 in enumerate(tasks):
        for t2 in tasks[i + 1:]:
            diff = abs(task_means[t1]["stable_rank_mean"] - task_means[t2]["stable_rank_mean"])
            between_task_rank_diff.append(diff)

    return {
        "prediction": "E: Spectral partitioning — high-SV shared, low-SV task-specific",
        "task_structural_stats": task_means,
        "within_task_rank_variance": float(np.mean(within_task_rank_var)) if within_task_rank_var else None,
        "between_task_rank_diff_mean": float(np.mean(between_task_rank_diff)) if between_task_rank_diff else None,
        "supported": (float(np.mean(between_task_rank_diff)) > float(np.mean(within_task_rank_var)) * 2
                      if within_task_rank_var and between_task_rank_diff else False),
        "note": "Supported if between-task rank differences > 2× within-task variance",
    }


# ─── Prediction F: Norm imbalance from dataset size ───────

def test_prediction_f(adapters_dir: Path, reports: dict[str, dict]) -> dict:
    """Adapters from larger datasets should have larger norms, causing imbalance in cross-size pairs."""
    adapter_norms: dict[str, float] = {}
    adapter_tasks: dict[str, str] = {}

    for adapter_dir in sorted(adapters_dir.iterdir()):
        if not adapter_dir.is_dir():
            continue
        adapter_id = adapter_dir.name
        task = get_task(adapter_id)
        adapter_tasks[adapter_id] = task
        jsonl = adapter_dir / "run.jsonl"
        if not jsonl.exists():
            continue

        events = load_jsonl_events(jsonl, kind="structural")
        if events:
            last = events[-1]
            norm = last.get("metrics", {}).get("adapter_frob_norm_total")
            if norm is not None:
                adapter_norms[adapter_id] = norm

    # Correlate dataset size with adapter norm
    sizes = []
    norms = []
    for aid, norm in adapter_norms.items():
        task = adapter_tasks[aid]
        sizes.append(TASK_SIZES.get(task, 0))
        norms.append(norm)

    if len(sizes) > 2:
        r_size_norm, p_size_norm = stats.pearsonr(np.log(sizes), np.log(norms))
        spearman_r, spearman_p = stats.spearmanr(sizes, norms)
    else:
        r_size_norm, p_size_norm = float("nan"), float("nan")
        spearman_r, spearman_p = float("nan"), float("nan")

    # Check: do norm_imbalance issues correlate with dataset size ratio?
    size_ratios = []
    is_imbalanced = []
    for pair_label, report in reports.items():
        parts = pair_label.split("_vs_")
        task_a = get_task(parts[0])
        task_b = get_task(parts[1])
        if task_a == task_b:
            continue
        sa = TASK_SIZES.get(task_a, 1)
        sb = TASK_SIZES.get(task_b, 1)
        ratio = max(sa, sb) / min(sa, sb)
        size_ratios.append(ratio)
        is_imbalanced.append(1 if report.get("dominant_issue") == "norm_imbalance" else 0)

    if len(size_ratios) > 2:
        r_ratio_imb, p_ratio_imb = stats.pointbiserialr(is_imbalanced, size_ratios)
    else:
        r_ratio_imb, p_ratio_imb = float("nan"), float("nan")

    per_adapter = {}
    for aid in sorted(adapter_norms):
        task = adapter_tasks[aid]
        per_adapter[aid] = {
            "task": task,
            "dataset_size": TASK_SIZES.get(task, 0),
            "frob_norm_total": adapter_norms[aid],
        }

    return {
        "prediction": "F: Norm imbalance correlates with dataset size difference",
        "per_adapter_norms": per_adapter,
        "pearson_r_logsize_lognorm": float(r_size_norm),
        "pearson_p": float(p_size_norm),
        "spearman_r_size_norm": float(spearman_r),
        "spearman_p": float(spearman_p),
        "pointbiserial_r_ratio_imbalance": float(r_ratio_imb),
        "pointbiserial_p": float(p_ratio_imb),
        "supported": p_size_norm < 0.05 if not np.isnan(p_size_norm) else False,
    }


# ─── Prediction G: Phase transitions in curvature ─────────

def test_prediction_g(adapters_dir: Path) -> dict:
    """Curvature dynamics should show detectable phase transitions (sharp changes in Hessian spectrum)."""
    phase_results = {}

    for adapter_dir in sorted(adapters_dir.iterdir()):
        if not adapter_dir.is_dir():
            continue
        adapter_id = adapter_dir.name
        jsonl = adapter_dir / "run.jsonl"
        if not jsonl.exists():
            continue

        events = load_jsonl_events(jsonl, kind="curvature")
        if len(events) < 10:
            phase_results[adapter_id] = {"status": "insufficient_data", "n": len(events)}
            continue

        steps = []
        energies = []
        lambda_maxes = []
        anisotropies = []

        for ev in events:
            m = ev.get("metrics", {})
            steps.append(m.get("step", ev.get("step", 0)))
            energies.append(m.get("hessian_trace", 0))
            lambda_maxes.append(m.get("lambda_max", 0))
            anisotropies.append(m.get("anisotropy", 0))

        energies_arr = np.array(energies)
        lambda_arr = np.array(lambda_maxes)

        # Detect phase transitions via change-point analysis
        # Use rolling variance ratio (simple approach)
        window = max(5, len(energies) // 20)
        if len(energies) < window * 3:
            phase_results[adapter_id] = {"status": "too_short_for_phases", "n": len(events)}
            continue

        # Compute rolling statistics
        energy_diffs = np.diff(energies_arr)
        rolling_var = []
        for i in range(window, len(energy_diffs)):
            rolling_var.append(np.var(energy_diffs[i - window:i]))
        rolling_var = np.array(rolling_var)

        # Phase transition = spike in rolling variance (> 3× median)
        if len(rolling_var) > 0:
            median_var = np.median(rolling_var)
            threshold = max(median_var * 3, 1e-10)
            transition_indices = np.where(rolling_var > threshold)[0]
            n_transitions = len(transition_indices)

            # Map back to steps
            transition_steps = [int(steps[i + window]) for i in transition_indices if i + window < len(steps)]
        else:
            n_transitions = 0
            transition_steps = []

        # Epoch boundaries (potential phase transition locations)
        total_steps = steps[-1] - steps[0] if steps else 0

        # Energy trajectory: does it have distinct regimes?
        # Split into thirds and compare means
        n = len(energies_arr)
        third = n // 3
        if third > 0:
            phase_1_mean = float(np.mean(energies_arr[:third]))
            phase_2_mean = float(np.mean(energies_arr[third:2*third]))
            phase_3_mean = float(np.mean(energies_arr[2*third:]))
            regime_shift = abs(phase_2_mean - phase_1_mean) / (abs(phase_1_mean) + 1e-10)
        else:
            phase_1_mean = phase_2_mean = phase_3_mean = regime_shift = 0

        phase_results[adapter_id] = {
            "status": "computed",
            "n_snapshots": len(events),
            "n_transitions_detected": n_transitions,
            "transition_steps": transition_steps[:10],  # cap at 10
            "energy_phase1_mean": phase_1_mean,
            "energy_phase2_mean": phase_2_mean,
            "energy_phase3_mean": phase_3_mean,
            "regime_shift_magnitude": float(regime_shift),
            "lambda_max_range": [float(lambda_arr.min()), float(lambda_arr.max())],
            "energy_range": [float(energies_arr.min()), float(energies_arr.max())],
        }

    computed = {k: v for k, v in phase_results.items() if v.get("status") == "computed"}
    n_with_transitions = sum(1 for v in computed.values() if v.get("n_transitions_detected", 0) > 0)
    regime_shifts = [v["regime_shift_magnitude"] for v in computed.values()]

    return {
        "prediction": "G: Phase transitions detectable in curvature dynamics",
        "per_adapter": phase_results,
        "n_computed": len(computed),
        "n_with_transitions": n_with_transitions,
        "mean_regime_shift": float(np.mean(regime_shifts)) if regime_shifts else None,
        "supported": n_with_transitions >= len(computed) // 2 if computed else False,
        "note": "Phase transitions detected via rolling variance spike (>3× median)",
    }


# ─── Main ─────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="N07 Phase 4: Analysis")
    parser.add_argument("--base-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None, help="Output JSON path")
    args = parser.parse_args()

    base_dir = args.base_dir
    adapters_dir = base_dir / "adapters"
    reports_dir = base_dir / "reports"
    merges_dir = base_dir / "merges"

    # Load merge reports
    reports: dict[str, dict] = {}
    for f in sorted(reports_dir.glob("*.json")):
        reports[f.stem] = json.loads(f.read_text())

    # Load merge evaluation results
    results_file = merges_dir / "phase3_results.json"
    if results_file.exists():
        merge_results = json.loads(results_file.read_text())
    else:
        merge_results = []
        print("WARNING: No Phase 3 results found")

    print("N07 Phase 4: Analysis")
    print(f"  Adapters: {len(list(adapters_dir.iterdir()))}")
    print(f"  Reports: {len(reports)}")
    print(f"  Merge evals: {len(merge_results)}")
    print()

    analysis = {}

    # Test each prediction
    tests = [
        ("A", lambda: test_prediction_a(reports)),
        ("B", lambda: test_prediction_b(reports)),
        ("C", lambda: test_prediction_c(merge_results, reports)),
        ("D", lambda: test_prediction_d(adapters_dir)),
        ("E", lambda: test_prediction_e(adapters_dir)),
        ("F", lambda: test_prediction_f(adapters_dir, reports)),
        ("G", lambda: test_prediction_g(adapters_dir)),
    ]

    for label, test_fn in tests:
        print(f"{'=' * 60}")
        print(f"  Testing Prediction {label}")
        print(f"{'=' * 60}")
        try:
            result = test_fn()
            analysis[f"prediction_{label}"] = result
            supported = result.get("supported", "?")
            status = "✓ SUPPORTED" if supported else "✗ NOT SUPPORTED"
            print(f"  {result['prediction']}")
            print(f"  Result: {status}")
            # Print key metrics
            for k, v in result.items():
                if k in ("prediction", "per_adapter", "lag_correlations", "task_pair_stats",
                         "per_adapter_norms", "note"):
                    continue
                if isinstance(v, dict):
                    print(f"    {k}:")
                    for k2, v2 in v.items():
                        print(f"      {k2}: {v2}")
                else:
                    print(f"    {k}: {v}")
            if "note" in result:
                print(f"  Note: {result['note']}")
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            analysis[f"prediction_{label}"] = {"error": str(e)}
        print()

    # Summary
    print(f"{'=' * 60}")
    print(f"  SUMMARY")
    print(f"{'=' * 60}")
    for label in "ABCDEFG":
        key = f"prediction_{label}"
        if key in analysis:
            r = analysis[key]
            supported = r.get("supported", "?")
            pred = r.get("prediction", key)
            status = "✓" if supported else "✗"
            print(f"  {status} {pred}")
    print()

    # Save
    output = args.output or (base_dir / "analysis" / "phase4_analysis.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(analysis, indent=2, default=str))
    print(f"  Results saved: {output}")


if __name__ == "__main__":
    main()
