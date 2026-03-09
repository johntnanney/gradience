#!/usr/bin/env python3
"""
Figure 5 — The workflow implied by Study 16.

Left-to-right pipeline diagram showing the staged QA architecture:
single-adapter audit → eligibility screening → pairwise merge-risk audit
→ guarded intervention → downstream eval.

Usage:
    python figures/fig5_workflow.py
    # Produces: figures/fig5_workflow.pdf
    #           figures/fig5_workflow.png
"""

from __future__ import annotations

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# ---------------------------------------------------------------------------
# Stage definitions
# ---------------------------------------------------------------------------

stages = [
    {
        "title": "Single-adapter\naudit",
        "bullets": ["utilization", "rank waste", "optional behavioral eval"],
        "warning": None,
        "footnote": None,
    },
    {
        "title": "Source eligibility\nscreening",
        "bullets": ["eligible", "uncertain", "flagged weak"],
        "warning": True,
        "footnote": None,
    },
    {
        "title": "Pairwise\nmerge-risk audit",
        "bullets": ["norm imbalance", "overlap / conflict", "domination risk"],
        "warning": None,
        "footnote": None,
    },
    {
        "title": "Guarded\nintervention",
        "bullets": ["no-op", "norm-equalized", "layerwise rebalance"],
        "warning": None,
        "footnote": "structural recommendation,\nnot guaranteed quality",
    },
    {
        "title": "Downstream\neval",
        "bullets": ["final quality check"],
        "warning": None,
        "footnote": None,
    },
]

# ---------------------------------------------------------------------------
# Layout constants
# ---------------------------------------------------------------------------

N = len(stages)
BOX_W = 1.55
BOX_H = 1.65
GAP = 0.55
TOTAL_W = N * BOX_W + (N - 1) * GAP
X_START = 0.3
Y_CENTER = 1.1

# ---------------------------------------------------------------------------
# Colors
# ---------------------------------------------------------------------------

COL_BOX_FACE = "#f5f7fa"
COL_BOX_EDGE = "#8899aa"
COL_TITLE = "#2a3a4a"
COL_BULLET = "#556677"
COL_ARROW = "#99aabb"
COL_WARNING = "#c44e52"
COL_FOOTNOTE = "#888888"
COL_SCREEN = "#2a7b9b"

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

fig_w = TOTAL_W + 1.2
fig_h = 3.2
fig, ax = plt.subplots(figsize=(fig_w, fig_h))
ax.set_xlim(0, fig_w)
ax.set_ylim(0, fig_h)
ax.set_aspect("equal")
ax.axis("off")

box_positions = []

for i, stage in enumerate(stages):
    x = X_START + i * (BOX_W + GAP)
    y = Y_CENTER - BOX_H / 2

    box_positions.append((x, y))

    # Highlight box 2 (eligibility screening) with a subtle accent border
    edge_color = COL_SCREEN if stage["warning"] else COL_BOX_EDGE
    edge_lw = 2.0 if stage["warning"] else 1.0

    box = FancyBboxPatch(
        (x, y),
        BOX_W,
        BOX_H,
        boxstyle="round,pad=0.08",
        facecolor=COL_BOX_FACE,
        edgecolor=edge_color,
        linewidth=edge_lw,
        zorder=3,
    )
    ax.add_patch(box)

    # Stage number
    ax.text(
        x + BOX_W / 2,
        y + BOX_H - 0.15,
        str(i + 1),
        ha="center",
        va="center",
        fontsize=8.5,
        color=edge_color,
        fontweight="bold",
        alpha=0.5,
    )

    # Title
    ax.text(
        x + BOX_W / 2,
        y + BOX_H - 0.48,
        stage["title"],
        ha="center",
        va="center",
        fontsize=8.5,
        color=COL_TITLE,
        fontweight="bold",
        linespacing=1.25,
    )

    # Bullets
    bullet_start_y = y + BOX_H - 0.88
    for j, bullet in enumerate(stage["bullets"]):
        by = bullet_start_y - j * 0.22
        ax.text(
            x + 0.18,
            by,
            f"\u2022  {bullet}",
            ha="left",
            va="center",
            fontsize=7,
            color=COL_BULLET,
        )

    # Warning icon for eligibility screening
    if stage["warning"]:
        ax.text(
            x + BOX_W - 0.12,
            y + BOX_H - 0.12,
            "\u26a0",
            ha="center",
            va="center",
            fontsize=12,
            color=COL_WARNING,
            zorder=4,
        )

    # Footnote below box
    if stage["footnote"]:
        ax.text(
            x + BOX_W / 2,
            y - 0.15,
            stage["footnote"],
            ha="center",
            va="top",
            fontsize=6.5,
            color=COL_FOOTNOTE,
            fontstyle="italic",
            linespacing=1.2,
        )

# ---------------------------------------------------------------------------
# Arrows between boxes
# ---------------------------------------------------------------------------

for i in range(N - 1):
    x_from = box_positions[i][0] + BOX_W + 0.04
    x_to = box_positions[i + 1][0] - 0.04
    y_mid = Y_CENTER

    ax.annotate(
        "",
        xy=(x_to, y_mid),
        xytext=(x_from, y_mid),
        arrowprops=dict(
            arrowstyle="-|>",
            color=COL_ARROW,
            lw=1.8,
            mutation_scale=14,
        ),
        zorder=2,
    )

# ---------------------------------------------------------------------------
# Title
# ---------------------------------------------------------------------------

ax.text(
    fig_w / 2,
    fig_h - 0.15,
    "Figure 5.  The workflow implied by Study 16",
    ha="center",
    va="top",
    fontsize=11,
    fontweight="bold",
    color=COL_TITLE,
)

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

for ext in ("pdf", "png"):
    path = f"figures/fig5_workflow.{ext}"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"Saved: {path}")

plt.close(fig)
