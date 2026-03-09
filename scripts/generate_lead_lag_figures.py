#!/usr/bin/env python3
"""Generate publication-quality figures for the lead-lag analysis."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

RESULTS_DIR = PROJECT_ROOT / "results" / "lead_lag"
FIGURES_DIR = RESULTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Clean style
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "figure.dpi": 150,
        "savefig.bbox": "tight",
        "savefig.dpi": 150,
    }
)

COLORS = {
    "primary": "#2C5F7C",
    "secondary": "#B55A30",
    "accent": "#4A9B6F",
    "muted": "#8C8C8C",
    "ci_band": "#D4E5F0",
}


def load_aggregate_ccf():
    path = RESULTS_DIR / "ccf" / "aggregate_ccf.json"
    with open(path) as f:
        return json.load(f)


def load_forecast_summary():
    path = RESULTS_DIR / "forecast" / "summary.json"
    with open(path) as f:
        return json.load(f)


def load_granger_summary():
    path = RESULTS_DIR / "granger" / "summary.json"
    with open(path) as f:
        return json.load(f)


def load_early_stopping():
    path = RESULTS_DIR / "early_stopping" / "summary.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def fig1_ccf_aggregate():
    """CCF aggregate plot: one subplot per key geometric feature."""
    data = load_aggregate_ccf()

    # Select key features (exclude delta/accel which are tier-1 only)
    key_features = ["grad_norm_mean", "grad_norm_max", "grad_norm_std", "grad_norm_last"]
    features = {k: v for k, v in data.items() if k in key_features and v}

    if not features:
        print("  No CCF data to plot")
        return

    n = len(features)
    fig, axes = plt.subplots(1, n, figsize=(3.5 * n, 3), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, (feat, agg) in zip(axes, features.items()):
        lags = np.array(agg["common_lags"])
        mean = np.array(agg["mean_ccf"])
        std = np.array(agg["std_ccf"])

        ax.fill_between(lags, mean - std, mean + std, alpha=0.2, color=COLORS["ci_band"])
        ax.plot(lags, mean, color=COLORS["primary"], linewidth=2)

        # Significance bands
        n_runs = agg["n_runs"]
        # Approximate CI for mean of n runs
        ci = 1.96 / np.sqrt(max(1, lags.max() - lags.min()))
        ax.axhline(ci, color=COLORS["muted"], linestyle="--", linewidth=0.8, alpha=0.5)
        ax.axhline(-ci, color=COLORS["muted"], linestyle="--", linewidth=0.8, alpha=0.5)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.axvline(0, color="black", linewidth=0.5, alpha=0.3)

        # Mark peak
        peak_idx = np.argmax(np.abs(mean))
        ax.scatter(lags[peak_idx], mean[peak_idx], color=COLORS["secondary"], s=60, zorder=5)

        short_name = feat.replace("grad_norm_", "gn_")
        ax.set_title(short_name, fontsize=10)
        ax.set_xlabel("Lag (eval intervals)")
        if ax == axes[0]:
            ax.set_ylabel("Cross-correlation")

        # Annotate lead count
        leads = agg["leads_count"]
        total = agg["total_runs"]
        ax.text(
            0.95,
            0.95,
            f"leads: {leads}/{total}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8,
            color=COLORS["muted"],
        )

    fig.suptitle("Cross-Correlation: Geometric Features vs Eval Loss\n(negative lag = geometry leads)", fontsize=11)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "ccf_aggregate.png")
    plt.close(fig)
    print("  Saved ccf_aggregate.png")


def fig2_lead_lag_heatmap():
    """Lead-lag heatmap: rows = features, columns = lags, color = mean CCF."""
    data = load_aggregate_ccf()

    features_order = ["grad_norm_mean", "grad_norm_max", "grad_norm_std", "grad_norm_last"]
    features = {k: v for k, v in data.items() if k in features_order and v}

    if not features:
        print("  No CCF data for heatmap")
        return

    # Build matrix
    all_lags = None
    for feat, agg in features.items():
        if all_lags is None:
            all_lags = np.array(agg["common_lags"])

    if all_lags is None:
        return

    matrix = []
    labels = []
    for feat in features_order:
        if feat in features:
            agg = features[feat]
            lags = np.array(agg["common_lags"])
            mean = np.array(agg["mean_ccf"])
            # Align to common lags
            row = []
            for l in all_lags:
                idx = np.where(lags == l)[0]
                row.append(mean[idx[0]] if len(idx) > 0 else np.nan)
            matrix.append(row)
            labels.append(feat.replace("grad_norm_", "gn_"))

    matrix = np.array(matrix)

    fig, ax = plt.subplots(figsize=(8, 3))
    vmax = np.nanmax(np.abs(matrix))
    im = ax.imshow(matrix, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)

    ax.set_xticks(range(len(all_lags)))
    ax.set_xticklabels(all_lags, fontsize=8)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Lag (eval intervals)")
    ax.set_title("Lead-Lag Heatmap: Mean Cross-Correlation")

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Mean CCF")

    # Annotate vertical line at lag=0
    zero_idx = np.where(all_lags == 0)[0]
    if len(zero_idx) > 0:
        ax.axvline(zero_idx[0], color="black", linewidth=1, alpha=0.3)

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "lead_lag_heatmap.png")
    plt.close(fig)
    print("  Saved lead_lag_heatmap.png")


def fig3_forecast_comparison():
    """Forecast comparison: bars for persistence vs model RMSE."""
    data = load_forecast_summary()

    # Group by variant
    variants = {}
    for row in data:
        v = row["variant"]
        if v not in variants:
            variants[v] = {"rmse_model": [], "rmse_persist": [], "reduction": []}
        if row["n_predictions"] > 0 and not (np.isnan(row["rmse_model"]) or np.isnan(row["rmse_persistence"])):
            variants[v]["rmse_model"].append(row["rmse_model"])
            variants[v]["rmse_persist"].append(row["rmse_persistence"])
            variants[v]["reduction"].append(row["rmse_reduction_pct"])

    if not variants:
        print("  No forecast data to plot")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Left: RMSE comparison
    x = np.arange(len(variants))
    width = 0.35
    labels = list(variants.keys())

    persist_means = [np.mean(v["rmse_persist"]) for v in variants.values()]
    model_means = [np.mean(v["rmse_model"]) for v in variants.values()]
    persist_stds = [np.std(v["rmse_persist"]) for v in variants.values()]
    model_stds = [np.std(v["rmse_model"]) for v in variants.values()]

    ax1.bar(
        x - width / 2, persist_means, width, yerr=persist_stds, color=COLORS["muted"], label="Persistence", capsize=3
    )
    ax1.bar(x + width / 2, model_means, width, yerr=model_stds, color=COLORS["primary"], label="Ridge (geo)", capsize=3)

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=8)
    ax1.set_ylabel("RMSE")
    ax1.set_title("RMSE: Persistence vs Ridge")
    ax1.legend(fontsize=8)

    # Right: RMSE reduction distribution
    for i, (label, v) in enumerate(variants.items()):
        reds = v["reduction"]
        ax2.hist(reds, bins=15, alpha=0.6, label=label, color=[COLORS["primary"], COLORS["secondary"]][i % 2])

    ax2.axvline(0, color="black", linewidth=1)
    ax2.set_xlabel("RMSE Reduction (%)")
    ax2.set_ylabel("Count")
    ax2.set_title("Forecast RMSE Reduction Distribution")
    ax2.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "forecast_comparison.png")
    plt.close(fig)
    print("  Saved forecast_comparison.png")


def fig4_early_stopping():
    """Early-stopping Pareto: % saved vs metric delta."""
    data = load_early_stopping()
    if not data:
        print("  No early-stopping data to plot")
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    # Separate geometric vs loss-based rules
    geo_rules = [r for r in data if not r["rule"].startswith("loss_")]
    loss_rules = [r for r in data if r["rule"].startswith("loss_")]

    # Plot geometric rules
    for r in geo_rules:
        saved = r["mean_pct_saved"]
        delta = r["mean_metric_delta"]
        if saved > 0:
            ax.scatter(delta, saved, color=COLORS["primary"], s=50, zorder=5)
            # Label the interesting ones
            if saved > 5:
                short = r["rule"].replace("_plateau_", "\n").replace("grad_norm_", "gn_").replace("train_loss_", "tl_")
                ax.annotate(
                    short,
                    (delta, saved),
                    fontsize=6,
                    xytext=(5, 5),
                    textcoords="offset points",
                    color=COLORS["primary"],
                )

    # Plot loss-based rules
    for r in loss_rules:
        saved = r["mean_pct_saved"]
        delta = r["mean_metric_delta"]
        ax.scatter(delta, saved, color=COLORS["secondary"], s=50, marker="^", zorder=5)

    ax.axvline(0, color="black", linewidth=0.5, alpha=0.3)
    ax.axhline(0, color="black", linewidth=0.5, alpha=0.3)

    # Label quadrants
    ax.text(
        0.02,
        0.98,
        "Ideal: high savings,\nno accuracy cost",
        transform=ax.transAxes,
        fontsize=8,
        va="top",
        color=COLORS["accent"],
    )

    ax.set_xlabel("Mean Metric Delta (negative = worse)")
    ax.set_ylabel("Mean % Training Saved")
    ax.set_title("Early-Stopping Rule Pareto: Savings vs Cost")

    # Legend
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D(
            [0], [0], marker="o", color="w", markerfacecolor=COLORS["primary"], markersize=8, label="Geometric rules"
        ),
        Line2D(
            [0], [0], marker="^", color="w", markerfacecolor=COLORS["secondary"], markersize=8, label="Loss-based rules"
        ),
    ]
    ax.legend(handles=legend_elements, fontsize=8)

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "early_stopping_pareto.png")
    plt.close(fig)
    print("  Saved early_stopping_pareto.png")


def main():
    print("Generating figures...")
    fig1_ccf_aggregate()
    fig2_lead_lag_heatmap()
    fig3_forecast_comparison()
    fig4_early_stopping()
    print("Done.")


if __name__ == "__main__":
    main()
