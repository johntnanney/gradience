"""Generate the OVmix_depth raw-vs-partial sign-flip figure for FINDINGS §28.

Two-panel figure demonstrating that the OV-mixture depth-weighted alignment
score has a positive raw correlation with max_degradation (left panel) but
a strong negative correlation after FAMILY_B task-family residualization
(right panel) on the N133 12-pair evaluation sample.

OVmix_depth (0.7 * O_depth + 0.3 * V_depth) is a composite constructed
during the N133 follow-up sweep — i.e. it was tuned on the confounded
N133 data rather than anchored to any theoretical mechanism. The sign flip
(+0.074 → -0.658 under FAMILY_B residualization) is the textbook
demonstration of "metric hand-tuned on confounded sample learns the
confound, inverts under family control."

Inputs:
  sidecar/data/n133/family_residualized_baseline.json

Output:
  sidecar/figures/n133/ovmix_sign_flip.png (300 DPI)

The figure reads ρ, ρ_partial, and ΔR² from the locked JSON; it does not
re-compute any statistics. The slope-flip invariant is asserted at runtime
so any upstream change that breaks it halts figure generation before a
factually-wrong figure lands on disk.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "sidecar" / "data" / "n133" / "family_residualized_baseline.json"
OUT = ROOT / "sidecar" / "figures" / "n133" / "ovmix_sign_flip.png"

# The variant the figure targets. Must be a key present in
# family_B_four_way.variants and family_B_four_way.per_pair.
TARGET_VARIANT = "OVmix_depth"
TARGET_LABEL = "OVmix_depth (0.7·O_depth + 0.3·V_depth)"


def _variant_row(rows: list[dict], name: str) -> dict:
    for r in rows:
        if r["variant"] == name:
            return r
    raise KeyError(f"variant {name!r} not found")


def main() -> None:
    blob = json.loads(DATA.read_text())
    fam_B = blob["family_B_four_way"]
    per_pair = fam_B["per_pair"]

    row = _variant_row(fam_B["variants"], TARGET_VARIANT)
    rho_raw = row["spearman_raw"]
    rho_partial = row["spearman_partial"]
    delta_r2 = row["delta_r2"]
    r_raw = row["pearson_raw"]
    raw_r2 = float(r_raw) ** 2

    x_raw = np.array([p[TARGET_VARIANT] for p in per_pair])
    y_raw = np.array([p["max_deg"] for p in per_pair])
    x_par = np.array([p[f"{TARGET_VARIANT}_resid"] for p in per_pair])
    y_par = np.array([p["max_deg_resid"] for p in per_pair])
    fam = [p["family_pair"] for p in per_pair]

    # Sanity-check invariant: the whole rhetorical point of this figure is
    # that OVmix_depth has a positive raw slope and a negative residualized
    # slope. If that is not what the data show, halt and investigate rather
    # than ship a factually wrong figure.
    slope_raw = float(np.polyfit(x_raw, y_raw, 1)[0])
    slope_par = float(np.polyfit(x_par, y_par, 1)[0])
    if not (slope_raw > 0 and slope_par < 0):
        raise RuntimeError(
            f"slope-flip invariant violated for {TARGET_VARIANT}: "
            f"slope_raw={slope_raw:+.4f}, slope_par={slope_par:+.4f}. "
            f"Something upstream has changed; stop and investigate before "
            f"regenerating figure."
        )

    uniq = sorted(set(fam))
    cmap = plt.get_cmap("tab10", len(uniq))
    color_of = {k: cmap(i) for i, k in enumerate(uniq)}
    colors = [color_of[k] for k in fam]

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(10, 5))

    # Left panel: raw
    axL.scatter(x_raw, y_raw, c=colors, s=70, edgecolor="black", linewidth=0.5)
    xs = np.linspace(x_raw.min(), x_raw.max(), 50)
    m, b = np.polyfit(x_raw, y_raw, 1)
    axL.plot(xs, m * xs + b, color="black", linewidth=1.5)
    axL.set_title(
        f"Raw: ρ = {rho_raw:+.3f}, r² = {raw_r2:.3f}",
        fontsize=11,
    )
    axL.set_xlabel(f"{TARGET_LABEL}")
    axL.set_ylabel("max_degradation")
    axL.grid(True, alpha=0.3)
    axL.annotate(
        "positive slope → appears to rank pairs by risk",
        xy=(0.02, 0.97), xycoords="axes fraction",
        ha="left", va="top", fontsize=9, style="italic",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85,
                  edgecolor="gray"),
    )

    # Right panel: partial
    axR.scatter(x_par, y_par, c=colors, s=70, edgecolor="black", linewidth=0.5)
    xs = np.linspace(x_par.min(), x_par.max(), 50)
    m, b = np.polyfit(x_par, y_par, 1)
    axR.plot(xs, m * xs + b, color="black", linewidth=1.5)
    axR.set_title(
        f"Partial (FAMILY_B): ρ = {rho_partial:+.3f}, "
        f"ΔR² = {delta_r2:+.3f}",
        fontsize=11,
    )
    axR.set_xlabel(f"{TARGET_LABEL} (FAMILY_B residualized)")
    axR.set_ylabel("max_degradation (FAMILY_B residualized)")
    axR.grid(True, alpha=0.3)
    axR.annotate(
        "negative slope → metric was tracking family, not geometry",
        xy=(0.02, 0.97), xycoords="axes fraction",
        ha="left", va="top", fontsize=9, style="italic",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85,
                  edgecolor="gray"),
    )

    handles = [
        plt.Line2D(
            [0], [0], marker="o", linestyle="",
            color=color_of[k], label=k,
            markersize=8, markeredgecolor="black",
        )
        for k in uniq
    ]
    fig.legend(
        handles=handles, loc="lower center",
        ncol=min(len(uniq), 5),
        bbox_to_anchor=(0.5, -0.02), frameon=False,
        fontsize=9,
    )

    fig.suptitle(
        "OVmix_depth on N133: raw vs. FAMILY_B-residualized\n"
        "(N = 12 evaluated cross-task pairs, Mistral-7B)",
        y=1.03, fontsize=12,
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {OUT}")


if __name__ == "__main__":
    main()
