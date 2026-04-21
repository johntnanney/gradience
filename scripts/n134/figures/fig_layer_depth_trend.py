"""F3 (paper-numbered) — layer-depth trend in same/cross alignment ratio.

Line plot of the per-layer same/cross alignment ratio on Mistral-7B
under N134 (layers 0-31), with linear fit annotated. All numbers
(trend slope, intercept, r, p, per-layer ratios) are extracted from
sidecar/results/n134/analysis_secondary.json; nothing is hardcoded.

The figure's rhetorical role (per draft section 7 discussion): the
depth trend is a property of the aggregate same/cross separation
geometry, not a predictor of per-pair outcomes. The caption in
figure_captions.md preempts the reviewer who sees r = 0.919 and asks
why this is not evidence for the depth-weighted H1 score working:
the depth trend is about the population-level alignment gap widening
with depth, while H1 is about whether per-pair alignment predicts
per-pair merge outcome. The former holds robustly; the latter does
not. Both findings are in the paper; this figure shows the former,
Figure 1 shows the latter.

Run:
    python scripts/n134/figures/fig_layer_depth_trend.py

Output:
    papers/n134_workshop/figures/layer_depth_trend.{pdf,png}
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mpl_style import (  # noqa: E402
    ANNOT_GREY,
    COL_SINGLE,
    N134_RESULTS,
    PALETTE,
    REF_COLOR,
    apply_style,
    save_figure,
)


def _load_secondary() -> dict:
    path = N134_RESULTS / "analysis_secondary.json"
    return json.loads(path.read_text())


def build_figure() -> plt.Figure:
    data = _load_secondary()
    dt = data["depth_trend"]

    layers = np.array([e["layer"] for e in dt["per_layer"]], dtype=float)
    ratios = np.array([e["ratio"] for e in dt["per_layer"]], dtype=float)

    slope = float(dt["trend_slope"])
    intercept = float(dt["trend_intercept"])
    r_val = float(dt["trend_r"])
    p_val = float(dt["trend_p"])
    ratio_min, ratio_max = dt["ratio_range"]

    fit_line = intercept + slope * layers

    apply_style()

    fig, ax = plt.subplots(figsize=(COL_SINGLE, COL_SINGLE * 0.85))

    # Ratio = 1 reference line (no same/cross separation).
    ax.axhline(
        1.0,
        color=REF_COLOR,
        linewidth=0.6,
        linestyle=":",
        zorder=1,
    )

    # Points: per-layer ratios.
    ax.plot(
        layers,
        ratios,
        marker="o",
        markersize=3.2,
        linestyle="",
        color=PALETTE["blue"],
        markeredgecolor="white",
        markeredgewidth=0.4,
        zorder=3,
        label="per-layer ratio",
    )

    # Linear fit line.
    ax.plot(
        layers,
        fit_line,
        linestyle="-",
        linewidth=1.2,
        color=PALETTE["red"],
        label=rf"linear fit ($r = {r_val:.3f}$)",
        zorder=2,
    )

    ax.set_xlabel("Layer index")
    ax.set_ylabel(r"Same/cross alignment ratio")
    ax.set_xlim(-1, 32)
    # Small padding around the data range.
    ax.set_ylim(
        max(0.9, float(ratio_min) - 0.2),
        float(ratio_max) + 0.2,
    )

    # Annotation box with trend numbers.
    annot_text = (
        rf"slope $= {slope:+.3f}$ per layer"
        "\n"
        rf"$r = {r_val:.3f}$"
        "\n"
        rf"$p = {p_val:.1e}$"
        "\n"
        rf"layer 0 ratio: ${ratios[0]:.2f}$"
        "\n"
        rf"layer 31 ratio: ${ratios[-1]:.2f}$"
    )
    ax.text(
        0.03,
        0.97,
        annot_text,
        transform=ax.transAxes,
        fontsize=7.5,
        va="top",
        ha="left",
        color=ANNOT_GREY,
        bbox=dict(
            facecolor="white",
            edgecolor=PALETTE["grey"],
            linewidth=0.5,
            boxstyle="round,pad=0.3",
        ),
        zorder=4,
    )

    ax.legend(loc="lower right", frameon=False, handlelength=2.2)
    ax.set_title("Layer-depth trend in same/cross ratio", pad=6)

    return fig


def main() -> None:
    fig = build_figure()
    pdf, png = save_figure(fig, "layer_depth_trend")
    print(f"wrote {pdf}")
    print(f"wrote {png}")


if __name__ == "__main__":
    main()
