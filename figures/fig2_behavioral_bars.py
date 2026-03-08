#!/usr/bin/env python3
"""
Figure 2 — Behavioral gains are selective rather than universal.

Small-multiple grouped bar chart showing perplexity across naive,
norm-equalized, and recommended merges for each Study 16 adapter pair.

Usage:
    python figures/fig2_behavioral_bars.py
    # Produces: figures/fig2_behavioral_bars.pdf
    #           figures/fig2_behavioral_bars.png
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Data — perplexity (lower is better)
#
# Each pair has: { eval_set: (naive, norm_eq, recommended) }
# Base-model perplexity is constant per eval set.
# ---------------------------------------------------------------------------

BASE_PPL = {
    "GSM8K":  5.89,
    "MBPP":   5.89,
    "oasst2": 5.89,
}

PAIRS = {
    "Pair 01\nmetamath × openwebmath": {
        "evals": {"GSM8K": (6.82, 5.41, 5.18)},
        "annotation": None,
    },
    "Pair 02\nmetamath × magicoder": {
        "evals": {
            "GSM8K": (5.31, 5.24, 5.21),
            "MBPP":  (5.48, 5.42, 5.39),
        },
        "annotation": "no real change",
    },
    "Pair 03\nmagicoder × btgenbot": {
        "evals": {
            "MBPP":  (7.24, 5.81, 5.52),
            "oasst2": (5.14, 5.58, 5.71),
        },
        "annotation": "tradeoff case",
    },
    "Pair 04\nopenwebmath-r64 × btgenbot": {
        "evals": {
            "GSM8K":  (7.51, 5.93, 5.62),
            "oasst2": (5.02, 5.54, 5.78),
        },
        "annotation": "tradeoff case",
    },
    "Pair 06\ncatsubcat × btgenbot": {
        "evals": {"oasst2": (8.14, 6.48, 6.21)},
        "annotation": "best merge still\nworse than base",
    },
}

# ---------------------------------------------------------------------------
# Colors
# ---------------------------------------------------------------------------

COL_NAIVE  = "#c44e52"
COL_NORMEQ = "#e8a838"
COL_REC    = "#2a7b9b"
COL_BASE   = "#555555"
COL_ANNOT  = "#777777"

BAR_COLORS = [COL_NAIVE, COL_NORMEQ, COL_REC]
BAR_LABELS = ["Naive", "Norm-equalized", "Recommended"]

# ---------------------------------------------------------------------------
# Layout — one panel per pair
# ---------------------------------------------------------------------------

pair_names = list(PAIRS.keys())
n_pairs = len(pair_names)

fig, axes = plt.subplots(1, n_pairs, figsize=(14, 4.2), sharey=True)

bar_width = 0.22

for panel_idx, (pair_name, pair_data) in enumerate(PAIRS.items()):
    ax = axes[panel_idx]
    evals = pair_data["evals"]
    eval_names = list(evals.keys())
    n_evals = len(eval_names)

    x_centers = np.arange(n_evals)

    for bar_idx in range(3):
        offsets = x_centers + (bar_idx - 1) * bar_width
        heights = [evals[ev][bar_idx] for ev in eval_names]
        bars = ax.bar(
            offsets, heights, bar_width * 0.90,
            color=BAR_COLORS[bar_idx],
            edgecolor="white", linewidth=0.4,
            label=BAR_LABELS[bar_idx] if panel_idx == 0 else None,
            zorder=3,
        )

    # Base-model dashed lines (one per eval set)
    for ev_idx, ev_name in enumerate(eval_names):
        base_val = BASE_PPL[ev_name]
        x_lo = ev_idx - 0.42
        x_hi = ev_idx + 0.42
        ax.plot(
            [x_lo, x_hi], [base_val, base_val],
            color=COL_BASE, linewidth=1.2, linestyle="--", alpha=0.7,
            zorder=4,
            label="Base model" if (panel_idx == 0 and ev_idx == 0) else None,
        )

    # Panel title
    ax.set_title(pair_name, fontsize=8.5, fontweight="bold", linespacing=1.2, pad=8)

    # X ticks — eval set names
    ax.set_xticks(x_centers)
    ax.set_xticklabels(eval_names, fontsize=7.5)

    # Annotation
    annot = pair_data["annotation"]
    if annot:
        ax.text(
            0.5, 0.97, annot,
            transform=ax.transAxes,
            fontsize=7, fontstyle="italic", color=COL_ANNOT,
            ha="center", va="top",
        )

    # Formatting
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.5)
    ax.spines["bottom"].set_linewidth(0.5)
    ax.tick_params(axis="both", which="both", length=3, width=0.5)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color="#eee", linewidth=0.5)

# Shared y-axis
axes[0].set_ylabel("Perplexity  (lower is better)", fontsize=10)
axes[0].set_ylim(4.0, 9.0)

# Legend — single row at bottom
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles, labels,
    loc="lower center", ncol=4, fontsize=8.5,
    framealpha=0.95, edgecolor="#ddd", handletextpad=0.4,
    bbox_to_anchor=(0.5, -0.02),
)

# Suptitle
fig.suptitle(
    "Figure 2.  Behavioral gains are selective rather than universal",
    fontsize=11, fontweight="bold", y=1.02,
)

plt.tight_layout(rect=[0, 0.06, 1, 0.96])

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

for ext in ("pdf", "png"):
    path = f"figures/fig2_behavioral_bars.{ext}"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"Saved: {path}")

plt.close(fig)
