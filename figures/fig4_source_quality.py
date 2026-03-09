#!/usr/bin/env python3
"""
Figure 4 — Source-adapter quality is a gating condition, not a footnote.

Horizontal bar chart showing perplexity delta vs base for each source
adapter. Bars extending left = better than base, right = worse than base.

Usage:
    python figures/fig4_source_quality.py
    # Produces: figures/fig4_source_quality.pdf
    #           figures/fig4_source_quality.png
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Data — perplexity delta vs base (adapter_ppl - base_ppl)
#   negative = adapter is better, positive = adapter is worse
# ---------------------------------------------------------------------------

adapters = [
    ("metamath-r16", -1.38, "GSM8K"),
    ("openwebmath-r16", -0.12, "GSM8K"),
    ("openwebmath-r64", 0.09, "GSM8K"),
    ("magicoder-r16", 0.24, "MBPP"),
    ("btgenbot-r8", 0.58, "oasst2"),
    ("catsubcat-r16", 0.92, "oasst2"),
]

labels = [a[0] for a in adapters]
deltas = [a[1] for a in adapters]
benchmarks = [a[2] for a in adapters]

y_pos = np.arange(len(adapters))

# ---------------------------------------------------------------------------
# Colors — green-ish for better, red-ish for worse, muted for near-zero
# ---------------------------------------------------------------------------

COL_BETTER = "#2a7b9b"
COL_WORSE = "#c44e52"
COL_NEAR = "#b0b0b0"


def bar_color(d: float) -> str:
    if d < -0.3:
        return COL_BETTER
    if d > 0.3:
        return COL_WORSE
    return COL_NEAR


bar_colors = [bar_color(d) for d in deltas]

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(8, 4))

bars = ax.barh(
    y_pos,
    deltas,
    height=0.55,
    color=bar_colors,
    edgecolor="white",
    linewidth=0.5,
    zorder=3,
)

# Zero reference line
ax.axvline(x=0, color="#444", linewidth=0.9, zorder=2)

# ---------------------------------------------------------------------------
# Value labels + benchmark tags
# ---------------------------------------------------------------------------

for i, (yi, d, bench) in enumerate(zip(y_pos, deltas, benchmarks)):
    # Value label
    offset = -0.04 if d < 0 else 0.04
    ha = "right" if d < 0 else "left"
    ax.text(
        d + offset,
        yi,
        f"{d:+.2f}",
        ha=ha,
        va="center",
        fontsize=9,
        fontweight="bold" if abs(d) > 0.3 else "normal",
        color=bar_colors[i],
    )
    # Benchmark tag (small, right-aligned after the label)
    tag_x = max(deltas) + 0.28
    ax.text(
        tag_x,
        yi,
        bench,
        ha="left",
        va="center",
        fontsize=7.5,
        color="#999",
        fontstyle="italic",
    )

# ---------------------------------------------------------------------------
# Zone labels
# ---------------------------------------------------------------------------

ax.text(
    -0.75,
    -0.65,
    "better than base",
    ha="center",
    va="center",
    fontsize=8,
    color=COL_BETTER,
    fontweight="bold",
    alpha=0.7,
)
ax.text(
    0.75,
    -0.65,
    "worse than base",
    ha="center",
    va="center",
    fontsize=8,
    color=COL_WORSE,
    fontweight="bold",
    alpha=0.7,
)

# Small arrows
ax.annotate(
    "",
    xy=(-1.2, -0.65),
    xytext=(-0.38, -0.65),
    arrowprops=dict(arrowstyle="-|>", color=COL_BETTER, lw=1.0, alpha=0.4),
)
ax.annotate(
    "",
    xy=(1.2, -0.65),
    xytext=(0.38, -0.65),
    arrowprops=dict(arrowstyle="-|>", color=COL_WORSE, lw=1.0, alpha=0.4),
)

# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

ax.set_yticks(y_pos)
ax.set_yticklabels(labels, fontsize=9.5)
ax.set_xlabel("Perplexity difference vs base  (negative = better)", fontsize=10)
ax.set_xlim(-1.7, 1.7)
ax.set_ylim(len(adapters) - 0.5, -1.0)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_linewidth(0.5)
ax.spines["bottom"].set_linewidth(0.5)
ax.tick_params(axis="both", which="both", length=3, width=0.5)

ax.set_axisbelow(True)
ax.xaxis.grid(True, color="#eee", linewidth=0.5)

# Title
fig.suptitle(
    "Figure 4.  Source-adapter quality is a gating condition, not a footnote",
    fontsize=11,
    fontweight="bold",
    y=1.0,
)

plt.tight_layout(rect=[0, 0, 1, 0.95])

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

for ext in ("pdf", "png"):
    path = f"figures/fig4_source_quality.{ext}"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"Saved: {path}")

plt.close(fig)
