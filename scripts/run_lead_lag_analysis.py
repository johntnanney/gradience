#!/usr/bin/env python3
"""
Run the complete lead-lag reanalysis on M1 experiment telemetry.

Targets the telemetry-enhanced training runs (chat task, 3 seeds)
with both grad_norm and structural features (stable_rank, effective_rank,
energy_rank_90, adapter_frob_norm, sigma_max).

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

from gradience.analysis.early_stopping import (
    aggregate_stopping_results,
    make_stopping_rules,
    simulate_early_stopping,
)
from gradience.analysis.extract_timeseries import extract_all_runs, save_aligned
from gradience.analysis.lead_lag import (
    aggregate_ccfs,
    compute_ccf,
    granger_causality_test,
    ridge_forecast,
    run_lead_lag_analysis,
    surrogate_null_test,
)

# M1 telemetry data (chat task, 3 seeds with eval every 25 steps + structural every 50 steps)
M1_ADAPTERS_DIR = PROJECT_ROOT / "results" / "lead_lag" / "workspace" / "m1" / "adapters"
RESULTS_DIR = PROJECT_ROOT / "results" / "lead_lag"

# Full feature set: grad_norm aggregates + structural metrics
GEOMETRIC_FEATURES = [
    # grad_norm aggregates (from train_step events, aggregated per eval interval)
    "grad_norm_mean",
    "grad_norm_max",
    "grad_norm_std",
    "grad_norm_last",
    # structural metrics (from StructuralMetricsCallback, forward-filled to eval grid)
    "stable_rank_mean",
    "effective_rank_mean",
    "energy_rank_90_mean",
    "adapter_frob_norm_mean",
    "sigma_max_mean",
]


def _run_label(run) -> str:
    """Create a short label from the source path."""
    p = Path(run.meta.source_path)
    parts = p.parts
    # Extract task/seed from path like .../adapters/chat/seed_42/telemetry/run.jsonl
    try:
        adapters_idx = next(i for i, x in enumerate(parts) if x == "adapters")
        task = parts[adapters_idx + 1]
        seed_dir = parts[adapters_idx + 2]
        return f"{task}/{seed_dir}"
    except (StopIteration, IndexError):
        return p.stem


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
    print("LEAD-LAG REANALYSIS PIPELINE (M1 Experiment)")
    print("=" * 70)

    # --- Phase 0: Extract and align ---
    print(f"\n[Phase 0] Extracting time series from {M1_ADAPTERS_DIR}...")
    runs = extract_all_runs(M1_ADAPTERS_DIR, min_eval_events=4, min_train_steps=10)
    print(f"  Found {len(runs)} runs meeting minimum requirements")

    for run in runs:
        label = _run_label(run)
        print(
            f"  {label}: {run.meta.total_eval_events} evals, "
            f"{run.meta.total_train_steps} train steps, "
            f"{run.meta.total_structural_events} structural snapshots, "
            f"eval_interval={run.meta.eval_interval}, "
            f"structural_interval={run.meta.structural_interval}"
        )
        # Show available columns
        struct_cols = [c for c in run.aligned.columns if c.startswith("stable_rank") or c.startswith("effective_rank")]
        if struct_cols:
            print(f"    Structural columns present: {struct_cols[:5]}...")

    # Save aligned parquets
    aligned_dir = RESULTS_DIR / "aligned"
    save_aligned(runs, aligned_dir)
    print(f"  Saved aligned DataFrames to {aligned_dir}")

    # --- Phase 1-3: Lead-lag analysis per run ---
    print("\n[Phase 1-3] Running lead-lag analysis per run...")
    print(f"  Features: {GEOMETRIC_FEATURES}")

    all_results = {}
    all_ccfs_by_feature: dict[str, list] = {}

    for run in runs:
        label = _run_label(run)
        if run.aligned.empty or len(run.aligned) < 4:
            print(f"  SKIP {label}: too few aligned points ({len(run.aligned)})")
            continue

        # Filter features to those actually present
        available_features = [f for f in GEOMETRIC_FEATURES if f in run.aligned.columns]
        print(f"\n  [{label}] ({len(run.aligned)} eval pts, {len(available_features)} features)")

        results = run_lead_lag_analysis(
            run.aligned,
            geometric_features=available_features,
            target="eval_loss",
            max_ccf_lag=min(10, len(run.aligned) // 2 - 1),
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
        print("    CCF (peak_lag, peak_corr):")
        for ccf_r in results["ccf"]:
            if not np.isnan(ccf_r.peak_corr):
                sig = "*" if ccf_r.significant_lags else ""
                direction = "LEADS" if ccf_r.peak_lag < 0 else ("LAGS" if ccf_r.peak_lag > 0 else "SYNC")
                print(
                    f"      {ccf_r.feature_name:<30s} lag={ccf_r.peak_lag:+d} r={ccf_r.peak_corr:+.3f} {direction}{sig}"
                )

        print("    Granger (p-value):")
        for gr in results["granger"]:
            if not np.isnan(gr.p_value):
                sig = (
                    "***" if gr.p_value < 0.01 else ("**" if gr.p_value < 0.05 else ("*" if gr.p_value < 0.10 else ""))
                )
                print(
                    f"      {gr.feature_name:<30s} p={gr.p_value:.4f} F={gr.f_statistic:.2f} dR2={gr.delta_r_squared:.4f} {sig}"
                )

        print("    Forecast (RMSE reduction vs persistence):")
        for flabel, fr in results["forecast"].items():
            if not np.isnan(fr.rmse_reduction_pct):
                print(
                    f"      {flabel:<20s} RMSE_red={fr.rmse_reduction_pct:+.1f}% R2_oos={fr.r_squared_oos:.3f} (n={fr.n_predictions})"
                )

    # --- Aggregate CCFs ---
    print("\n" + "=" * 70)
    print("[Aggregate] CCF aggregation across runs...")
    agg_ccfs = {}
    for feat_name, ccf_list in sorted(all_ccfs_by_feature.items()):
        agg = aggregate_ccfs(ccf_list)
        agg_ccfs[feat_name] = agg
        if agg:
            direction = "LEADS" if agg["peak_lag_mean"] < 0 else "LAGS"
            print(
                f"  {feat_name:<35s} peak_lag={agg['peak_lag_mean']:+.1f} +/- {agg['peak_lag_std']:.1f}  "
                f"leads={agg['leads_count']}/{agg['total_runs']}  "
                f"mean_r={agg['peak_corr_mean']:+.3f}  {direction}"
            )

    # --- Surrogate null tests (on all runs since we only have 3) ---
    print("\n[Phase 3.5] Surrogate null tests...")
    surrogate_results = {}

    for run in runs:
        label = _run_label(run)
        if label not in all_results:
            continue
        base_features = [f for f in GEOMETRIC_FEATURES if f in run.aligned.columns]
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
        sig = "***" if surr.p_value < 0.01 else ("**" if surr.p_value < 0.05 else ("*" if surr.p_value < 0.10 else ""))
        print(
            f"  {label}: actual_red={surr.actual_rmse_reduction:+.1f}%, "
            f"p={surr.p_value:.3f}, z={surr.z_score:+.2f} {sig}"
        )

    # --- Phase 4: Early-stopping simulation ---
    print("\n[Phase 4] Retrospective early-stopping simulation...")
    rules = make_stopping_rules(
        delta_thresholds=[0.02, 0.05, 0.10],
        patience_values=[2, 3],
    )

    all_stopping = {}
    for run in runs:
        label = _run_label(run)
        if run.aligned.empty or len(run.aligned) < 4:
            continue
        stopping = simulate_early_stopping(
            run.aligned,
            rules=rules,
            final_eval_metric="eval_loss",  # M1 only has eval_loss (no eval_accuracy for LM)
            fallback_metric="eval_loss",
        )
        if stopping:
            all_stopping[label] = stopping

    stopping_summary = None
    if all_stopping:
        stopping_summary = aggregate_stopping_results(all_stopping)
        print("\n  Early-Stopping Summary (sorted by mean % saved):")
        sorted_df = stopping_summary.sort_values("mean_pct_saved", ascending=False)
        for _, row in sorted_df.head(15).iterrows():
            print(
                f"    {row['rule']:<50s} saved={row['mean_pct_saved']:5.1f}% +/- {row['std_pct_saved']:4.1f}%  "
                f"delta_metric={row['mean_metric_delta']:+.4f}  triggered={row['triggered']}"
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
            granger_summary_data.append(
                {
                    "run": label,
                    "feature": gr.feature_name,
                    "target": gr.target_name,
                    "lag": gr.selected_lag,
                    "f_stat": gr.f_statistic,
                    "p_value": gr.p_value,
                    "reject": gr.reject_null,
                    "delta_r2": gr.delta_r_squared,
                    "n_obs": gr.n_obs,
                }
            )
    with open(granger_dir / "summary.json", "w") as f:
        json.dump(granger_summary_data, f, indent=2)

    # Forecast summary
    forecast_dir = RESULTS_DIR / "forecast"
    forecast_dir.mkdir(parents=True, exist_ok=True)
    forecast_summary_data = []
    for label, results in all_results.items():
        for flabel, fr in results["forecast"].items():
            forecast_summary_data.append(
                {
                    "run": label,
                    "variant": flabel,
                    "rmse_model": fr.rmse_model,
                    "rmse_persistence": fr.rmse_persistence,
                    "rmse_reduction_pct": fr.rmse_reduction_pct,
                    "r2_oos": fr.r_squared_oos,
                    "n_predictions": fr.n_predictions,
                    "horizon": fr.horizon,
                    "include_ar": fr.include_ar,
                }
            )
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
    if stopping_summary is not None:
        stopping_summary.to_json(es_dir / "summary.json", orient="records", indent=2)

    # --- Generate report ---
    print("\n[Phase 6] Generating report...")
    report = _generate_report(runs, all_results, agg_ccfs, surrogate_results, all_stopping, stopping_summary)
    report_path = RESULTS_DIR / "LEAD_LAG_REPORT.md"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"  Report written to {report_path}")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


def _generate_report(runs, all_results, agg_ccfs, surrogate_results, all_stopping, stopping_summary):
    """Generate the markdown summary report."""
    lines = ["# Lead-Lag Reanalysis Report (M1 Experiment)", ""]
    lines.append("## 1. Data Inventory")
    lines.append("")
    lines.append(f"**Total runs analyzed:** {len(all_results)}")
    lines.append("**Task:** chat (Alpaca instruction-following)")
    lines.append("**Model:** Mistral-7B + LoRA r=32")
    lines.append("**Training:** 1200 steps, eval every 25 steps, structural SVD every 50 steps")
    lines.append("")
    lines.append("| Run | Eval Events | Train Steps | Structural Snapshots | Seed |")
    lines.append("|-----|-------------|-------------|---------------------|------|")
    for run in runs:
        label = _run_label(run)
        lines.append(
            f"| {label} | {run.meta.total_eval_events} | {run.meta.total_train_steps} | "
            f"{run.meta.total_structural_events} | {run.meta.seed} |"
        )
    lines.append("")

    # Features
    lines.append("### Features Used")
    lines.append("")
    lines.append("**Gradient features** (aggregated per eval interval from train_step events):")
    lines.append("- `grad_norm_mean`, `grad_norm_max`, `grad_norm_std`, `grad_norm_last` + deltas + acceleration")
    lines.append("")
    lines.append("**Structural features** (from periodic SVD on LoRA A/B pairs, forward-filled to eval grid):")
    lines.append("- `stable_rank_mean`: ratio of squared Frobenius to squared spectral norm")
    lines.append("- `effective_rank_mean`: Shannon entropy of normalized singular values")
    lines.append("- `energy_rank_90_mean`: number of singular values capturing 90% energy")
    lines.append("- `adapter_frob_norm_mean`: Frobenius norm of adapter weight (BA product)")
    lines.append("- `sigma_max_mean`: largest singular value across layers")
    lines.append("")

    # CCF summary
    lines.append("## 2. Cross-Correlation Analysis (CCF)")
    lines.append("")
    lines.append("Convention: negative lag = geometric feature LEADS eval_loss (the hypothesis)")
    lines.append("")
    lines.append("| Feature | Peak Lag (mean +/- std) | Peak Corr (mean) | Leads (n/total) | Direction |")
    lines.append("|---------|------------------------|------------------|-----------------|-----------|")
    for feat, agg in sorted(agg_ccfs.items()):
        if agg:
            direction = "LEADS" if agg["peak_lag_mean"] < -0.5 else ("LAGS" if agg["peak_lag_mean"] > 0.5 else "SYNC")
            lines.append(
                f"| {feat} | {agg['peak_lag_mean']:+.1f} +/- {agg['peak_lag_std']:.1f} | "
                f"{agg['peak_corr_mean']:+.3f} | {agg['leads_count']}/{agg['total_runs']} | {direction} |"
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

    lines.append("| Feature | Runs p < 0.05 | Mean F-stat | Mean p-value | Mean delta-R2 |")
    lines.append("|---------|---------------|-------------|--------------|---------------|")
    for feat, grs in sorted(granger_features.items()):
        valid = [g for g in grs if not np.isnan(g.p_value)]
        if valid:
            reject_count = sum(1 for g in valid if g.reject_null)
            mean_f = np.mean([g.f_statistic for g in valid if not np.isnan(g.f_statistic)])
            mean_p = np.mean([g.p_value for g in valid])
            mean_dr2 = np.mean([g.delta_r_squared for g in valid if not np.isnan(g.delta_r_squared)])
            lines.append(f"| {feat} | {reject_count}/{len(valid)} | {mean_f:.2f} | {mean_p:.4f} | {mean_dr2:+.4f} |")
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
                f"| {variant} | {np.mean(reds):+.1f}% +/- {np.std(reds):.1f}% | "
                f"{positive}/{len(valid)} | {np.mean(r2s):.3f} |"
            )
    lines.append("")

    # Surrogate results
    if surrogate_results:
        lines.append("## 5. Surrogate Null Tests")
        lines.append("")
        lines.append(
            "Method: circular rotation, 200 surrogates. Tests whether ridge forecast improvement is significant."
        )
        lines.append("")
        lines.append("| Run | Actual RMSE Red | p-value | z-score | Significant? |")
        lines.append("|-----|-----------------|---------|---------|-------------|")
        for label, surr in surrogate_results.items():
            sig = "Yes" if surr.p_value < 0.05 else "No"
            lines.append(
                f"| {label} | {surr.actual_rmse_reduction:+.1f}% | {surr.p_value:.3f} | {surr.z_score:+.2f} | {sig} |"
            )
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

    # Interpretation
    lines.append("## 7. Interpretation Notes")
    lines.append("")
    lines.append(
        "- **3 seeds** provides limited statistical power for cross-run aggregation; treat as directional evidence."
    )
    lines.append(
        "- **48 eval events** per run (every 25 steps over 1200 total) gives much better CCF/Granger resolution than the pilot study (4-10 events)."
    )
    lines.append(
        "- **Structural metrics** are forward-filled from 50-step snapshots to 25-step eval grid, so they change every other eval step."
    )
    lines.append(
        "- **First-differencing** is applied for detrending; monotonic structural features (stable_rank, effective_rank) may lose signal if the trend is the signal."
    )
    lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    main()
