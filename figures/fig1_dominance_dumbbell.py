#!/usr/bin/env python3
"""
Figure 1 — Structural domination drops sharply in the imbalanced pairs.

Horizontal dumbbell plot showing dominance index D for naive vs recommended
merge across all adapter pairs in the Study 16 public-adapter test set.

Usage:
    python figures/fig1_dominance_dumbbell.py
    # Produces: figures/fig1_dominance_dumbbell.pdf
    #           figures/fig1_dominance_dumbbell.png
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Data — Study 16 adapter pairs
#   (label, D_naive, D_recommended)
# ---------------------------------------------------------------------------

pairs = [
    ("Pair 06\ncatsubcat \u00d7 btgenbot", 0.74, 0.18),
    ("Pair 04\nopenwebmath-r64 \u00d7 btgenbot", 0.68, 0.15),
    ("Pair 03\nmagicoder \u00d7 btgenbot", 0.61, 0.12),
    ("Pair 01\nmetamath \u00d7 openwebmath", 0.22, 0.11),
    ("Pair 02\nmetamath \u00d7 magicoder", 0.09, 0.08),
]

labels = [p[0] for p in pairs]
d_naive = [p[1] for p in pairs]
d_rec = [p[2] for p in pairs]

y_pos = np.arange(len(pairs))

# ---------------------------------------------------------------------------
# Colors
# ---------------------------------------------------------------------------

COL_NAIVE = "#c44e52"
COL_REC = "#2a7b9b"
COL_LINE = "#c0c0c0"
COL_THRESH = "#d4a84a"
COL_ANNOT = "#888888"

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(9, 4.2))

# Connecting lines — thicker for the big movers
for i in range(len(pairs)):
    delta = d_naive[i] - d_rec[i]
    lw = 2.8 if delta > 0.3 else 1.8
    ax.plot(
        [d_naive[i], d_rec[i]],
        [y_pos[i], y_pos[i]],
        color=COL_LINE,
        linewidth=lw,
        zorder=1,
        solid_capstyle="round",
    )

# Naive dots
ax.scatter(
    d_naive,
    y_pos,
    s=100,
    color=COL_NAIVE,
    edgecolors="white",
    linewidths=0.9,
    zorder=3,
    label="Naive merge",
)

# Recommended dots
ax.scatter(
    d_rec,
    y_pos,
    s=100,
    color=COL_REC,
    edgecolors="white",
    linewidths=0.9,
    zorder=3,
    label="Recommended merge",
)

# ---------------------------------------------------------------------------
# ΔD annotations — staggered vertically to avoid overlap
# ---------------------------------------------------------------------------

for i in range(3):
    delta = d_naive[i] - d_rec[i]
    midx = (d_naive[i] + d_rec[i]) / 2
    ax.text(
        midx,
        y_pos[i] - 0.28,
        f"\u0394D = {delta:.2f}",
        fontsize=8,
        fontweight="bold",
        color=COL_REC,
        ha="center",
        va="center",
    )

# Pair 02 — balanced control annotation
ax.annotate(
    "balanced control: unchanged",
    xy=((d_naive[4] + d_rec[4]) / 2, y_pos[4]),
    xytext=(0.42, y_pos[4] + 0.02),
    fontsize=7.5,
    fontstyle="italic",
    color=COL_ANNOT,
    va="center",
    arrowprops=dict(arrowstyle="-", color="#d0d0d0", lw=0.6),
)

# ---------------------------------------------------------------------------
# Threshold line at D = 0.2
# ---------------------------------------------------------------------------

ax.axvline(x=0.2, color=COL_THRESH, linewidth=1.0, linestyle="--", alpha=0.6, zorder=0)
ax.text(
    0.21,
    len(pairs) - 0.65,
    "D = 0.2",
    fontsize=7,
    color=COL_THRESH,
    fontstyle="italic",
    va="top",
)

# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

ax.set_yticks(y_pos)
ax.set_yticklabels(labels, fontsize=9, linespacing=1.15)
ax.set_xlabel("Dominance index  D   (lower is better)", fontsize=10.5)
ax.set_xlim(-0.05, 0.85)
ax.set_ylim(len(pairs) - 0.5, -0.7)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_linewidth(0.5)
ax.spines["bottom"].set_linewidth(0.5)
ax.tick_params(axis="both", which="both", length=3, width=0.5)

# Light horizontal grid
ax.set_axisbelow(True)
ax.xaxis.grid(True, color="#eee", linewidth=0.5)

# Legend
ax.legend(
    loc="lower right",
    fontsize=8.5,
    framealpha=0.95,
    edgecolor="#ddd",
    handletextpad=0.5,
)

# Title
fig.suptitle(
    "Figure 1.  Structural domination drops sharply in the imbalanced pairs",
    fontsize=11,
    fontweight="bold",
    y=0.99,
)

plt.tight_layout(rect=[0, 0, 1, 0.94])

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

for ext in ("pdf", "png"):
    path = f"figures/fig1_dominance_dumbbell.{ext}"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"Saved: {path}")

plt.close(fig)
