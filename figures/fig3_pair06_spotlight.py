#!/usr/bin/env python3
"""
Figure 3 — Pair 06: the engine improves the merge, but the candidate set is still bad.

Single-panel vertical bar chart spotlighting Pair 06 (catsubcat × btgenbot)
on oasst2 perplexity. Shows base model, both solo adapters, and all three
merge conditions.

Usage:
    python figures/fig3_pair06_spotlight.py
    # Produces: figures/fig3_pair06_spotlight.pdf
    #           figures/fig3_pair06_spotlight.png
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Data — oasst2 perplexity (lower is better)
# ---------------------------------------------------------------------------

categories = [
    "Base\nmodel",
    "btgenbot\nalone",
    "catsubcat\nalone",
    "Naive\nmerge",
    "Norm-eq.\nmerge",
    "Recommended\nmerge",
]

values = [5.89, 6.47, 6.81, 8.14, 6.48, 6.21]

# ---------------------------------------------------------------------------
# Colors — base gets its own neutral color, solo adapters are muted,
# merge conditions use the palette from figs 1–2
# ---------------------------------------------------------------------------

COL_BASE = "#555555"
COL_SOLO = "#a0a0a0"
COL_NAIVE = "#c44e52"
COL_NORMEQ = "#e8a838"
COL_REC = "#2a7b9b"

colors = [COL_BASE, COL_SOLO, COL_SOLO, COL_NAIVE, COL_NORMEQ, COL_REC]

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(8, 5))

x = np.arange(len(categories))
bars = ax.bar(
    x,
    values,
    width=0.58,
    color=colors,
    edgecolor="white",
    linewidth=0.6,
    zorder=3,
)

# ---------------------------------------------------------------------------
# Value labels above bars
# ---------------------------------------------------------------------------

for i, (xi, v) in enumerate(zip(x, values)):
    ax.text(
        xi,
        v + 0.08,
        f"{v:.2f}",
        ha="center",
        va="bottom",
        fontsize=9.5,
        fontweight="bold" if i == 0 or i == 5 else "normal",
        color=colors[i],
    )

# ---------------------------------------------------------------------------
# Annotation 1: bracket over merge bars — "Recommended is best merge condition"
# ---------------------------------------------------------------------------

merge_x = x[3:6]
bracket_y = max(values[3:6]) + 0.55
bracket_top = bracket_y + 0.12

# Bracket arms
ax.plot(
    [merge_x[0], merge_x[0], merge_x[-1], merge_x[-1]],
    [bracket_y - 0.08, bracket_top, bracket_top, bracket_y - 0.08],
    color=COL_REC,
    linewidth=1.2,
    solid_capstyle="round",
    clip_on=False,
)
ax.text(
    merge_x[1],
    bracket_top + 0.08,
    "recommended is best merge condition",
    ha="center",
    va="bottom",
    fontsize=8.5,
    fontweight="bold",
    color=COL_REC,
)

# ---------------------------------------------------------------------------
# Annotation 2: "But base is still best overall"
# ---------------------------------------------------------------------------

ax.annotate(
    "but base is still\nbest overall",
    xy=(x[0], values[0] + 0.05),
    xytext=(x[0] + 1.1, values[0] - 0.95),
    fontsize=8.5,
    fontweight="bold",
    color=COL_BASE,
    ha="center",
    va="top",
    arrowprops=dict(
        arrowstyle="-|>",
        color=COL_BASE,
        lw=1.2,
        connectionstyle="arc3,rad=-0.15",
    ),
)

# ---------------------------------------------------------------------------
# Base reference line
# ---------------------------------------------------------------------------

ax.axhline(y=values[0], color=COL_BASE, linewidth=0.8, linestyle=":", alpha=0.5, zorder=2)

# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=9)
ax.set_ylabel("Perplexity on oasst2  (lower is better)", fontsize=10.5)
ax.set_ylim(4.5, 9.5)
ax.set_xlim(-0.6, len(categories) - 0.4)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_linewidth(0.5)
ax.spines["bottom"].set_linewidth(0.5)
ax.tick_params(axis="both", which="both", length=3, width=0.5)

ax.set_axisbelow(True)
ax.yaxis.grid(True, color="#eee", linewidth=0.5)

# Title
fig.suptitle(
    "Figure 3.  Pair 06: the engine improves the merge,\nbut the candidate set is still bad",
    fontsize=11,
    fontweight="bold",
    y=1.01,
    linespacing=1.3,
)

plt.tight_layout(rect=[0, 0, 1, 0.94])

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

for ext in ("pdf", "png"):
    path = f"figures/fig3_pair06_spotlight.{ext}"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"Saved: {path}")

plt.close(fig)
