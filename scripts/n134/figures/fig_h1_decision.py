"""F1 — H1 decision plot.

Scatter of S_H1 vs max_degradation for all 45 cross-task pairs. Two
regression lines: raw (dashed) and family-residualized (solid). All
annotation numbers — raw rho, partial rho with p and bootstrap CI,
delta-R2 — are extracted from
sidecar/results/n134/analysis_h1.json; nothing is hardcoded.

The plot visually makes the point stated in draft section 4: the
partial correlation is sizable and statistically significant but
opposite in sign to the pre-registered prediction, and therefore
fails the sign constraint of the H1 decision rule. The visual shows
the residualized trend going the wrong way. The caption (in
papers/n134_workshop/figure_captions.md) makes the decision-rule
consequence explicit; this script produces the figure.

Point coloring: family_pair label mapped through a 28-entry cycling
palette. No legend — 28 entries is unreadable at 3.3" width. The
informal purpose of color is to make the family-level clustering
visible to the eye, which is the 88 percent family-R2 finding shown
visually. The clustering itself matters, not the identity of any
individual family pair.

Run:
    python scripts/n134/figures/fig_h1_decision.py

Output:
    papers/n134_workshop/figures/h1_decision.pdf
    papers/n134_workshop/figures/h1_decision.png

Paper-numbering note: this figure is referenced as Figure 1 in the
paper's LaTeX, but the output filename is descriptive (no numeric
prefix). Paper-facing numbering comes from \begin{figure} ordering,
not from filename. Dropping or reordering figures does not force a
rename of the remaining ones.
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


def _load_analysis() -> dict:
    path = N134_RESULTS / "analysis_h1.json"
    return json.loads(path.read_text())


def _ols_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Return (slope, intercept) of OLS fit y ~ x + const."""
    A = np.column_stack([x, np.ones_like(x)])
    sol, *_ = np.linalg.lstsq(A, y, rcond=None)
    return float(sol[0]), float(sol[1])


def _ols_residuals_against_family(
    values: np.ndarray, family_labels: list[str]
) -> np.ndarray:
    """OLS-residualize `values` on FAMILY_B dummies.

    Matches the residualization in 06_analysis_h1.py (used for the
    partial-Spearman computation). Returns residuals aligned to the
    input order.
    """
    unique = sorted(set(family_labels))
    # Dummies with intercept; drop no columns (ill-conditioned matrix is
    # handled by lstsq).
    X = np.zeros((len(values), len(unique) + 1))
    X[:, 0] = 1.0
    for j, lab in enumerate(unique):
        X[:, j + 1] = np.array([1.0 if fl == lab else 0.0 for fl in family_labels])
    sol, *_ = np.linalg.lstsq(X, values, rcond=None)
    fitted = X @ sol
    return values - fitted


def build_figure() -> plt.Figure:
    data = _load_analysis()
    per_pair = data["per_pair"]

    s_h1 = np.array([p["s_h1"] for p in per_pair], dtype=float)
    max_deg = np.array([p["max_degradation"] for p in per_pair], dtype=float)
    fam = [p["family_pair"] for p in per_pair]

    # Colors: cycle through a 20-color palette over 28 unique family-pair
    # labels. Some labels share colors; this is expected — see docstring.
    unique_fam = sorted(set(fam))
    cmap = plt.cm.tab20
    fam_color = {
        lab: cmap(i % cmap.N / cmap.N) for i, lab in enumerate(unique_fam)
    }
    point_colors = [fam_color[f] for f in fam]

    # Annotation numbers — all extracted from JSON.
    c1 = data["h1_decision_rule"]["criterion_1_partial_spearman"]
    c2 = data["h1_decision_rule"]["criterion_2_delta_r2"]
    c3 = data["h1_decision_rule"]["criterion_3_sign"]
    boot = data["bootstrap"]

    rho_raw = c3["rho_raw"]
    rho_partial = c1["rho_partial"]
    p_partial = c1["p_partial"]
    delta_r2 = c2["delta_r2"]
    rho_ci_lo = boot["rho_ci_025"]
    rho_ci_hi = boot["rho_ci_975"]

    # Raw regression line.
    slope_raw, intercept_raw = _ols_fit(s_h1, max_deg)

    # Partial regression line: residualize both on FAMILY_B, fit the residuals,
    # then project back into the original (s_h1, max_deg) coordinate frame by
    # translating the residual-space slope through the joint means.
    resid_x = _ols_residuals_against_family(s_h1, fam)
    resid_y = _ols_residuals_against_family(max_deg, fam)
    slope_partial, _ = _ols_fit(resid_x, resid_y)
    intercept_partial = max_deg.mean() - slope_partial * s_h1.mean()

    # Apply style.
    apply_style()

    fig, ax = plt.subplots(figsize=(COL_SINGLE, COL_SINGLE * 0.85))

    # Scatter.
    ax.scatter(
        s_h1,
        max_deg * 100.0,  # percent
        c=point_colors,
        s=20,
        edgecolors="white",
        linewidth=0.4,
        alpha=0.95,
        zorder=3,
    )

    # Regression lines. Use the x-range of the data.
    xs = np.linspace(s_h1.min(), s_h1.max(), 100)

    raw_line = (intercept_raw + slope_raw * xs) * 100.0
    ax.plot(
        xs,
        raw_line,
        linestyle=(0, (4, 2)),
        linewidth=1.1,
        color=PALETTE["grey"],
        label=rf"raw $\rho = {rho_raw:+.3f}$",
        zorder=2,
    )

    partial_line = (intercept_partial + slope_partial * xs) * 100.0
    ax.plot(
        xs,
        partial_line,
        linestyle="-",
        linewidth=1.6,
        color=PALETTE["red"],
        label=r"partial $\rho$ | FAMILY$_B$",
        zorder=2,
    )

    # Zero-degradation reference (visual anchor — degradations can be
    # negative if the merge improves on source on one task).
    ax.axhline(0.0, color=REF_COLOR, linewidth=0.6, linestyle=":", zorder=1)

    # Labels.
    ax.set_xlabel(r"$S_{\mathrm{H1}}$ (O-module depth-weighted alignment)")
    ax.set_ylabel("Max degradation (%)")

    # Legend: regression lines only; family-pair colors are unlabeled by
    # design (28 entries would be unreadable).
    ax.legend(loc="lower left", frameon=False, handlelength=2.2)

    # Annotation box with decision numbers.
    annot_text = (
        rf"partial $\rho = {rho_partial:+.3f}$"
        "\n"
        rf"$p = {p_partial:.1e}$"
        "\n"
        rf"$\Delta R^2 = {delta_r2:+.3f}$"
        "\n"
        "bootstrap 95% CI for partial $\\rho$:"
        "\n"
        rf"$[{rho_ci_lo:+.3f},\ {rho_ci_hi:+.3f}]$"
    )
    ax.text(
        0.97,
        0.97,
        annot_text,
        transform=ax.transAxes,
        fontsize=7.5,
        va="top",
        ha="right",
        color=ANNOT_GREY,
        bbox=dict(
            facecolor="white",
            edgecolor=PALETTE["grey"],
            linewidth=0.5,
            boxstyle="round,pad=0.3",
        ),
        zorder=4,
    )

    ax.set_title("H1 decision: null under pre-registered rule", pad=6)

    return fig


def main() -> None:
    fig = build_figure()
    pdf, png = save_figure(fig, "h1_decision")
    print(f"wrote {pdf}")
    print(f"wrote {png}")


if __name__ == "__main__":
    main()
