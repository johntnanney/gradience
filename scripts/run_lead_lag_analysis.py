#!/usr/bin/env python3
"""
Run the complete lead-lag reanalysis on all available bench_runs telemetry.

Outputs results to results/lead_lag/ as JSON + summary markdown.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from gradience.analysis.extract_timeseries import extract_all_runs, save_aligned
from gradience.analysis.lead_lag import (
    aggregate_ccfs,
    compute_ccf,
    granger_causality_test,
    ridge_forecast,
    run_lead_lag_analysis,
    surrogate_null_test,
)
from gradience.analysis.early_stopping import (
    aggregate_stopping_results,
    make_stopping_rules,
    simulate_early_stopping,
)

BENCH_RUNS = PROJECT_ROOT / "bench_runs"
RESULTS_DIR = PROJECT_ROOT / "results" / "lead_lag"

# Tier runs by quality
TIER1_PREFIXES = ["safety_uniform_r16_extended"]  # 10 evals, 111 steps
TIER2_PREFIXES = ["cert_v0.1_seed"]  # 5 evals, 56 steps, 3 seeds
TIER3_PREFIXES = ["multiseed_seed", "uniform_r16_seed", "uniform_r8_seed"]  # 4 evals, 3 seeds


def _run_label(run) -> str:
    """Create a short label from the source path."""
    p = Path(run.meta.source_path)
    parts = p.parts
    # Get the last 3 meaningful parts
    bench_idx = next((i for i, x in enumerate(parts) if x == "bench_runs"), -1)
    if bench_idx >= 0:
        relevant = parts[bench_idx + 1 :]
        return "/".join(relevant).replace("/run.jsonl", "")
    return p.stem


def _tier(label: str) -> int:
    for prefix in TIER1_PREFIXES:
        if label.startswith(prefix):
            return 1
    for prefix in TIER2_PREFIXES:
        if label.startswith(prefix):
            return 2
    for prefix in TIER3_PREFIXES:
        if label.startswith(prefix):
            return 3
    return 4


def _serializable(obj):
    """Make objects JSON-serializable."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    if hasattr(obj, "__dataclass_fields__"):
        d = {}
        for k, v in asdict(obj).items():
            d[k] = _serializable(v)
        return d
    if isinstance(obj, dict):
        return {k: _serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_serializable(v) for v in obj]
    return obj


def main():
    print("=" * 70)
    print("LEAD-LAG REANALYSIS PIPELINE")
    print("=" * 70)

    # --- Phase 0: Extract and align ---
    print("\n[Phase 0] Extracting time series from bench_runs...")
    runs = extract_all_runs(BENCH_RUNS, min_eval_events=4, min_train_steps=10)
    print(f"  Found {len(runs)} runs meeting minimum requirements")

    # Tag with labels and tiers
    run_info = []
    for run in runs:
        label = _run_label(run)
        tier = _tier(label)
        run_info.append((run, label, tier))

    # Summary by tier
    for t in [1, 2, 3, 4]:
        tier_runs = [(r, l) for r, l, ti in run_info if ti == t]
        if tier_runs:
            print(f"  Tier {t}: {len(tier_runs)} runs")
            for r, l in tier_runs[:5]:
                print(f"    {l}: {r.meta.total_eval_events} evals, {r.meta.total_train_steps} train steps, seed={r.meta.seed}, r={r.meta.lora_r}")
            if len(tier_runs) > 5:
                print(f"    ... and {len(tier_runs) - 5} more")

    # Save aligned parquets
    aligned_dir = RESULTS_DIR / "aligned"
    save_aligned(runs, aligned_dir)
    print(f"  Saved aligned DataFrames to {aligned_dir}")

    # --- Phase 1 & 2 & 3: Lead-lag analysis per run ---
    print("\n[Phase 1-3] Running lead-lag analysis per run...")

    geometric_features = ["grad_norm_mean", "grad_norm_max", "grad_norm_std", "grad_norm_last"]
    all_results = {}
    all_ccfs_by_feature: dict[str, list] = {}

    for run, label, tier in run_info:
        if run.aligned.empty or len(run.aligned) < 4:
            print(f"  SKIP {label}: too few aligned points ({len(run.aligned)})")
            continue

        results = run_lead_lag_analysis(
            run.aligned,
            geometric_features=geometric_features,
            target="eval_loss",
            max_ccf_lag=min(4, len(run.aligned) // 2 - 1),
            run_label=label,
        )

        all_results[label] = results

        # Collect CCFs by feature for aggregation
        for ccf_r in results["ccf"]:
            key = ccf_r.feature_name
            if key not in all_ccfs_by_feature:
                all_ccfs_by_feature[key] = []
            all_ccfs_by_feature[key].append(ccf_r)

        # Print per-run summary
        ccf_summary = ""
        for ccf_r in results["ccf"]:
            if not np.isnan(ccf_r.peak_corr):
                ccf_summary += f"  {ccf_r.feature_name}: peak_lag={ccf_r.peak_lag}, r={ccf_r.peak_corr:.3f}"

        granger_summary = ""
        for gr in results["granger"]:
            if not np.isnan(gr.p_value):
                granger_summary += f"  {gr.feature_name}: p={gr.p_value:.3f}{'*' if gr.reject_null else ''}"

        forecast_summary = ""
        for flabel, fr in results["forecast"].items():
            if not np.isnan(fr.rmse_reduction_pct):
                forecast_summary += f"  {flabel}: RMSE_red={fr.rmse_reduction_pct:.1f}%"

        print(f"\n  [{label}] (tier {tier}, {len(run.aligned)} eval pts)")
        if ccf_summary:
            print(f"    CCF:{ccf_summary}")
        if granger_summary:
            print(f"    Granger:{granger_summary}")
        if forecast_summary:
            print(f"    Forecast:{forecast_summary}")

    # --- Aggregate CCFs ---
    print("\n[Phase 1 Aggregate] CCF aggregation across runs...")
    agg_ccfs = {}
    for feat_name, ccf_list in all_ccfs_by_feature.items():
        agg = aggregate_ccfs(ccf_list)
        agg_ccfs[feat_name] = agg
        if agg:
            print(
                f"  {feat_name}: peak_lag={agg['peak_lag_mean']:.1f}±{agg['peak_lag_std']:.1f}, "
                f"leads={agg['leads_count']}/{agg['total_runs']}, "
                f"mean_peak_corr={agg['peak_corr_mean']:.3f}"
            )

    # --- Phase 3.5: Surrogate tests on best runs ---
    print("\n[Phase 3.5] Surrogate null tests on tier-1 runs...")
    surrogate_results = {}
    tier1_runs = [(r, l) for r, l, t in run_info if t == 1 and l in all_results]

    for run, label in tier1_runs:
        base_features = [f for f in geometric_features if f in run.aligned.columns]
        if not base_features or len(run.aligned) < 5:
            continue

        surr = surrogate_null_test(
            run.aligned,
            geometric_features=base_features,
            target="eval_loss",
            horizon=1,
            n_surrogates=200,
            method="circular_rotation",
        )
        surrogate_results[label] = surr
        print(
            f"  {label}: actual_red={surr.actual_rmse_reduction:.1f}%, "
            f"p={surr.p_value:.3f}, z={surr.z_score:.2f}"
        )

    # If no tier-1, try tier-2
    if not surrogate_results:
        print("  No tier-1 runs, running surrogates on tier-2...")
        tier2_runs = [(r, l) for r, l, t in run_info if t == 2 and l in all_results]
        for run, label in tier2_runs[:3]:  # limit to 3
            base_features = [f for f in geometric_features if f in run.aligned.columns]
            if not base_features or len(run.aligned) < 4:
                continue
            surr = surrogate_null_test(
                run.aligned,
                geometric_features=base_features,
                target="eval_loss",
                horizon=1,
                n_surrogates=100,
                method="circular_rotation",
            )
            surrogate_results[label] = surr
            print(
                f"  {label}: actual_red={surr.actual_rmse_reduction:.1f}%, "
                f"p={surr.p_value:.3f}, z={surr.z_score:.2f}"
            )

    # --- Phase 4: Early-stopping simulation ---
    print("\n[Phase 4] Retrospective early-stopping simulation...")
    rules = make_stopping_rules(
        delta_thresholds=[0.02, 0.05, 0.10],
        patience_values=[2, 3],
    )

    all_stopping = {}
    for run, label, tier in run_info:
        if run.aligned.empty or len(run.aligned) < 4:
            continue
        stopping = simulate_early_stopping(
            run.aligned,
            rules=rules,
            final_eval_metric="eval_accuracy",
            fallback_metric="eval_loss",
        )
        if stopping:
            all_stopping[label] = stopping

    if all_stopping:
        stopping_summary = aggregate_stopping_results(all_stopping)
        print("\n  Early-Stopping Summary (sorted by mean % saved):")
        sorted_df = stopping_summary.sort_values("mean_pct_saved", ascending=False)
        for _, row in sorted_df.head(15).iterrows():
            print(
                f"    {row['rule']:<50s} saved={row['mean_pct_saved']:5.1f}% ± {row['std_pct_saved']:4.1f}%  "
                f"Δmetric={row['mean_metric_delta']:+.4f}  triggered={row['triggered']}"
            )

    # --- Save all results ---
    print("\n[Phase 5] Saving results...")

    # CCF results
    ccf_dir = RESULTS_DIR / "ccf"
    ccf_dir.mkdir(parents=True, exist_ok=True)
    with open(ccf_dir / "aggregate_ccf.json", "w") as f:
        json.dump(_serializable(agg_ccfs), f, indent=2)

    # Per-run results
    per_run_dir = RESULTS_DIR / "per_run"
    per_run_dir.mkdir(parents=True, exist_ok=True)
    for label, results in all_results.items():
        safe_label = label.replace("/", "__")
        out = {
            "run_label": label,
            "n_eval_events": results["n_eval_events"],
            "ccf": [_serializable(asdict(c)) for c in results["ccf"]],
            "granger": [_serializable(asdict(g)) for g in results["granger"]],
            "forecast": {k: _serializable(asdict(v)) for k, v in results["forecast"].items()},
        }
        with open(per_run_dir / f"{safe_label}.json", "w") as f:
            json.dump(out, f, indent=2)

    # Granger summary
    granger_dir = RESULTS_DIR / "granger"
    granger_dir.mkdir(parents=True, exist_ok=True)
    granger_summary_data = []
    for label, results in all_results.items():
        for gr in results["granger"]:
            granger_summary_data.append({
                "run": label,
                "feature": gr.feature_name,
                "target": gr.target_name,
                "lag": gr.selected_lag,
                "f_stat": gr.f_statistic,
                "p_value": gr.p_value,
                "reject": gr.reject_null,
                "delta_r2": gr.delta_r_squared,
                "n_obs": gr.n_obs,
            })
    with open(granger_dir / "summary.json", "w") as f:
        json.dump(granger_summary_data, f, indent=2)

    # Forecast summary
    forecast_dir = RESULTS_DIR / "forecast"
    forecast_dir.mkdir(parents=True, exist_ok=True)
    forecast_summary_data = []
    for label, results in all_results.items():
        for flabel, fr in results["forecast"].items():
            forecast_summary_data.append({
                "run": label,
                "variant": flabel,
                "rmse_model": fr.rmse_model,
                "rmse_persistence": fr.rmse_persistence,
                "rmse_reduction_pct": fr.rmse_reduction_pct,
                "r2_oos": fr.r_squared_oos,
                "n_predictions": fr.n_predictions,
                "horizon": fr.horizon,
                "include_ar": fr.include_ar,
            })
    with open(forecast_dir / "summary.json", "w") as f:
        json.dump(_serializable(forecast_summary_data), f, indent=2)

    # Surrogate results
    surr_dir = RESULTS_DIR / "surrogates"
    surr_dir.mkdir(parents=True, exist_ok=True)
    for label, surr in surrogate_results.items():
        safe_label = label.replace("/", "__")
        with open(surr_dir / f"{safe_label}.json", "w") as f:
            json.dump(_serializable(asdict(surr)), f, indent=2)

    # Early stopping summary
    es_dir = RESULTS_DIR / "early_stopping"
    es_dir.mkdir(parents=True, exist_ok=True)
    if all_stopping:
        stopping_summary.to_json(es_dir / "summary.json", orient="records", indent=2)

    # --- Generate report ---
    print("\n[Phase 5] Generating report...")
    report = _generate_report(
        runs, run_info, all_results, agg_ccfs, surrogate_results, all_stopping,
        stopping_summary if all_stopping else None
    )
    report_path = RESULTS_DIR / "LEAD_LAG_REPORT.md"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"  Report written to {report_path}")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


def _generate_report(runs, run_info, all_results, agg_ccfs, surrogate_results, all_stopping, stopping_summary):
    """Generate the markdown summary report."""
    lines = ["# Lead-Lag Reanalysis Report", ""]
    lines.append("## 1. Data Inventory")
    lines.append("")
    lines.append(f"**Total runs analyzed:** {len(all_results)}")
    lines.append(f"**Total runs extracted:** {len(runs)}")
    lines.append("")
    lines.append("| Tier | Runs | Eval Events | Train Steps | Description |")
    lines.append("|------|------|-------------|-------------|-------------|")
    for t in [1, 2, 3, 4]:
        tier_runs = [(r, l) for r, l, ti in run_info if ti == t]
        if tier_runs:
            evals = [r.meta.total_eval_events for r, l in tier_runs]
            steps = [r.meta.total_train_steps for r, l in tier_runs]
            desc = {1: "Extended runs (best)", 2: "Cert runs (3 seeds)", 3: "Multi-seed/uniform", 4: "Other"}[t]
            lines.append(f"| {t} | {len(tier_runs)} | {min(evals)}-{max(evals)} | {min(steps)}-{max(steps)} | {desc} |")

    lines.append("")
    lines.append("**Geometric features available:** grad_norm (aggregated as mean, max, min, std, last, delta, acceleration)")
    lines.append("")
    lines.append("**NOT available in telemetry:** adapter_norm, stable_rank, energy_rank_90 (logged only in post-hoc audits, not during training)")
    lines.append("")
    lines.append("**Validation metrics:** eval_loss, eval_accuracy")
    lines.append("")

    # CCF summary
    lines.append("## 2. Cross-Correlation Analysis (CCF)")
    lines.append("")
    lines.append("Convention: negative lag = geometric feature LEADS validation metric (the hypothesis)")
    lines.append("")
    lines.append("| Feature | Peak Lag (mean +/- std) | Peak Corr (mean) | Leads (n/total) |")
    lines.append("|---------|------------------------|------------------|-----------------|")
    for feat, agg in sorted(agg_ccfs.items()):
        if agg:
            lines.append(
                f"| {feat} | {agg['peak_lag_mean']:.1f} +/- {agg['peak_lag_std']:.1f} | "
                f"{agg['peak_corr_mean']:.3f} | {agg['leads_count']}/{agg['total_runs']} |"
            )
    lines.append("")

    # Granger summary
    lines.append("## 3. Granger Causality Tests")
    lines.append("")
    granger_features: dict[str, list] = {}
    for label, results in all_results.items():
        for gr in results["granger"]:
            if gr.feature_name not in granger_features:
                granger_features[gr.feature_name] = []
            granger_features[gr.feature_name].append(gr)

    lines.append("| Feature | Runs p < 0.05 | Mean F-stat | Mean delta-R2 |")
    lines.append("|---------|---------------|-------------|---------------|")
    for feat, grs in sorted(granger_features.items()):
        valid = [g for g in grs if not np.isnan(g.p_value)]
        if valid:
            reject_count = sum(1 for g in valid if g.reject_null)
            mean_f = np.mean([g.f_statistic for g in valid if not np.isnan(g.f_statistic)])
            mean_dr2 = np.mean([g.delta_r_squared for g in valid if not np.isnan(g.delta_r_squared)])
            lines.append(f"| {feat} | {reject_count}/{len(valid)} | {mean_f:.2f} | {mean_dr2:.4f} |")
    lines.append("")

    # Forecast summary
    lines.append("## 4. Ridge Forecasting")
    lines.append("")
    forecast_variants: dict[str, list] = {}
    for label, results in all_results.items():
        for flabel, fr in results["forecast"].items():
            if flabel not in forecast_variants:
                forecast_variants[flabel] = []
            forecast_variants[flabel].append((label, fr))

    lines.append("| Variant | Mean RMSE Reduction | Runs with > 0% | Mean R2 OOS |")
    lines.append("|---------|--------------------|--------------------|-------------|")
    for variant, frs in sorted(forecast_variants.items()):
        valid = [(l, f) for l, f in frs if not np.isnan(f.rmse_reduction_pct) and f.n_predictions > 0]
        if valid:
            reds = [f.rmse_reduction_pct for _, f in valid]
            positive = sum(1 for r in reds if r > 0)
            r2s = [f.r_squared_oos for _, f in valid if not np.isnan(f.r_squared_oos)]
            lines.append(
                f"| {variant} | {np.mean(reds):.1f}% +/- {np.std(reds):.1f}% | "
                f"{positive}/{len(valid)} | {np.mean(r2s):.3f} |"
            )
    lines.append("")

    # Surrogate results
    if surrogate_results:
        lines.append("## 5. Surrogate Null Tests")
        lines.append("")
        lines.append("| Run | Actual RMSE Red | p-value | z-score |")
        lines.append("|-----|-----------------|---------|---------|")
        for label, surr in surrogate_results.items():
            lines.append(f"| {label} | {surr.actual_rmse_reduction:.1f}% | {surr.p_value:.3f} | {surr.z_score:.2f} |")
        lines.append("")

    # Early stopping
    if stopping_summary is not None:
        lines.append("## 6. Early-Stopping Simulation")
        lines.append("")
        lines.append("| Rule | Triggered | Mean % Saved | Mean Metric Delta |")
        lines.append("|------|-----------|-------------|-------------------|")
        sorted_df = stopping_summary.sort_values("mean_pct_saved", ascending=False)
        for _, row in sorted_df.head(15).iterrows():
            lines.append(
                f"| {row['rule']} | {row['triggered']} | {row['mean_pct_saved']:.1f}% | {row['mean_metric_delta']:+.4f} |"
            )
        lines.append("")

    # Limitations
    lines.append("## 7. Known Limitations")
    lines.append("")
    lines.append("1. **Single geometric feature**: Only `grad_norm` is logged during training. adapter_norm, stable_rank, energy_rank_90 are not available in telemetry (computed only in post-hoc audits).")
    lines.append("2. **Short series**: Most runs have 4-10 eval events, limiting statistical power for CCF and Granger tests.")
    lines.append("3. **Single task**: All runs are SST-2 on DistilBERT. No cross-task replication.")
    lines.append("4. **Train-step granularity**: grad_norm is logged every 10 steps; eval every 100 steps. The 10:1 ratio limits lead-lag resolution.")
    lines.append("")

    # Recommendations
    lines.append("## 8. Recommendations for Enhanced Telemetry")
    lines.append("")
    lines.append("To enable a full lead-lag study, the GradienceCallback should log these additional fields at each train_step:")
    lines.append("")
    lines.append("```python")
    lines.append("# In GradienceCallback.on_log():")
    lines.append("adapter_norm = compute_adapter_frobenius_norm(model)")
    lines.append("stable_rank = compute_stable_rank(adapter_BA)")
    lines.append("writer.train_step(step, loss=loss, grad_norm=grad_norm,")
    lines.append("                  adapter_norm=adapter_norm, stable_rank=stable_rank)")
    lines.append("```")
    lines.append("")
    lines.append("Additionally, eval frequency should be increased (every 25-50 steps instead of 100) for runs intended for lead-lag analysis.")
    lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    main()
