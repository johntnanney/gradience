"""F2 (paper-numbered) — four-method forest plot.

Horizontal forest plot of retained-set mean max_degradation for the
four methods compared in N134 Phase 5: Gradience S_H1, KnOTS, TSV, SVC.
Whiskers are the family-pair-blocked bootstrap 95% CI (5000 resamples).
Vertical dashed reference line at the random-baseline mean max_deg
(3.14% across all 45 cross-task pairs).

Per review (2026-04-21):
- x-axis: retained-set mean max_degradation in percent. Lower is
  better (consistent with the paper's max_degradation definition).
- y-axis: methods ordered by the MAGNITUDE of their Spearman rho
  (not alphabetical). With this ordering, the method with the
  strongest rank-correlation evidence (regardless of sign) sits at
  the top, and readers can see the rank-correlation story alongside
  the retained-degradation story.
- Inline per-method annotation on the right margin shows rho and p.

All numbers are extracted from sidecar/results/n134/method_comparison.json.
No hardcoded values.

Run:
    python scripts/n134/figures/fig_four_method_forest.py

Output:
    papers/n134_workshop/figures/four_method_forest.{pdf,png}
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

# Pretty method labels (matches paper prose).
METHOD_LABELS = {
    "gradience": r"Gradience $S_{\mathrm{H1}}$",
    "knots": "KnOTS",
    "tsv": "TSV",
    "svc": "SVC",
}

# Per-method plotted color; picked from the palette for perceptual
# separation across four categories.
METHOD_COLORS = {
    "gradience": PALETTE["blue"],
    "knots": PALETTE["green"],
    "tsv": PALETTE["yellow"],
    "svc": PALETTE["purple"],
}


def _load_methods() -> dict:
    path = N134_RESULTS / "method_comparison.json"
    return json.loads(path.read_text())


def build_figure() -> plt.Figure:
    data = _load_methods()
    methods = data["methods"]

    # Extract per-method quantities, converting from fraction to percent.
    entries = []
    for name in ("gradience", "knots", "tsv", "svc"):
        m = methods[name]
        triage = m["triage"]
        boot = m["bootstrap"]
        entries.append({
            "name": name,
            "label": METHOD_LABELS[name],
            "color": METHOD_COLORS[name],
            "rho": float(m["spearman_rho"]),
            "p": float(m["spearman_p"]),
            "retained_deg": float(triage["mean_degradation_retained"]) * 100.0,
            "ci_lo": float(boot["mean_deg_ci_025"]) * 100.0,
            "ci_hi": float(boot["mean_deg_ci_975"]) * 100.0,
        })

    # Order by MAGNITUDE of rho (descending). Strongest evidence at top.
    entries.sort(key=lambda e: abs(e["rho"]), reverse=True)

    # Random-baseline reference (mean max_deg across all 45 cross-task pairs).
    random_baseline = float(
        methods["gradience"]["triage"]["mean_degradation_all"]
    ) * 100.0

    apply_style()

    # Slightly taller than square to accommodate the right-margin annotation.
    fig, ax = plt.subplots(figsize=(COL_SINGLE, COL_SINGLE * 0.95))

    y_positions = np.arange(len(entries))[::-1]  # top = highest |rho|

    # Random-baseline reference line.
    ax.axvline(
        random_baseline,
        color=REF_COLOR,
        linestyle=(0, (4, 2)),
        linewidth=0.9,
        zorder=1,
    )
    # Label the reference line at the top of the plot.
    ax.text(
        random_baseline,
        len(entries) - 0.35,
        f" random ({random_baseline:.2f}%)",
        fontsize=7.5,
        color=REF_COLOR,
        va="bottom",
        ha="left",
        zorder=1,
    )

    # Bars (as horizontal error bars around the point estimate).
    for e, yp in zip(entries, y_positions):
        err_lo = e["retained_deg"] - e["ci_lo"]
        err_hi = e["ci_hi"] - e["retained_deg"]
        ax.errorbar(
            e["retained_deg"],
            yp,
            xerr=[[err_lo], [err_hi]],
            fmt="o",
            markersize=6,
            markerfacecolor=e["color"],
            markeredgecolor="white",
            markeredgewidth=0.6,
            ecolor=e["color"],
            elinewidth=1.4,
            capsize=3,
            capthick=1.0,
            zorder=3,
        )

    # Y-axis labels: method names.
    ax.set_yticks(y_positions)
    ax.set_yticklabels([e["label"] for e in entries])
    ax.set_ylim(-0.6, len(entries) - 0.05)

    ax.set_xlabel("Retained-set mean max degradation (%, lower is better)")

    ax.set_title("Four-method triage comparison (retain 22 of 45 pairs)", pad=6)

    # Inline right-margin annotation: rho and p per method.
    # Place just to the right of the highest CI upper bound.
    x_right_edge = max(e["ci_hi"] for e in entries) + 0.9
    for e, yp in zip(entries, y_positions):
        ax.text(
            x_right_edge,
            yp,
            rf"$\rho = {e['rho']:+.3f}$,  $p = {e['p']:.3f}$",
            fontsize=7.5,
            color=ANNOT_GREY,
            va="center",
            ha="left",
            zorder=4,
        )

    # Extend x-limit so the annotation has room but the CI bars still
    # dominate the visual.
    x_text_right = x_right_edge + 4.2
    ax.set_xlim(None, x_text_right)

    return fig


def main() -> None:
    fig = build_figure()
    pdf, png = save_figure(fig, "four_method_forest")
    print(f"wrote {pdf}")
    print(f"wrote {png}")


if __name__ == "__main__":
    main()
