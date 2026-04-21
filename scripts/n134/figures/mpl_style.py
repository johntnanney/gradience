"""Shared matplotlib style config for N134 workshop-paper figures.

Call apply_style() at the top of every figure script. All figures in
papers/n134_workshop/figures/ are produced under this config.

Conventions fixed here:
- Font: serif (STIX Two Text if available; fallback to the system serif).
- Font sizes: 9pt body, 10pt axis labels, 11pt titles.
- Widths: single column 3.3in, full column 7.0in. Use COL_* constants.
- Palette: paul-tol-style colorblind-safe set of 8 explicit hex codes,
  plus a background grey and an annotation-grey. No seaborn dependency.
- Output: save_figure(fig, name) writes both PDF (vector, primary) and
  PNG at 300 dpi (fallback artifact for web / preprint preview) to
  papers/n134_workshop/figures/.

Deterministic stochasticity: any figure script that resamples, jitters,
or shuffles for visualization uses a local numpy.random.Generator seeded
with RNG_SEED. No global seed state; no reliance on reproducibility of
matplotlib's internal RNG.

Every figure extracts its annotation values from the source JSON in
sidecar/results/n134/; never hardcoded.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[3]
N134_RESULTS = REPO_ROOT / "sidecar" / "results" / "n134"
FIGURES_OUT = REPO_ROOT / "papers" / "n134_workshop" / "figures"

# Fixed RNG seed for any stochastic visualization element (jitter, etc.).
RNG_SEED = 20260420

# Page widths (inches) — NeurIPS-style.
COL_SINGLE = 3.3
COL_FULL = 7.0

# Paul-Tol "bright" palette (colorblind-safe). Deliberately chosen over
# matplotlib's tab10 default: Paul-Tol's schemes are designed for
# colorblind-safe scientific visualization and have better perceptual
# separation in small figures. Explicit hex codes so regeneration is
# stable regardless of matplotlib default-palette changes.
# Source: https://personal.sron.nl/~pault/
PALETTE = {
    "blue":   "#4477AA",
    "cyan":   "#66CCEE",
    "green":  "#228833",
    "yellow": "#CCBB44",
    "red":    "#EE6677",
    "purple": "#AA3377",
    "grey":   "#BBBBBB",
    "black":  "#000000",
}

# Annotation / axis greys.
AXIS_GREY = "#555555"
ANNOT_GREY = "#333333"
BG_GREY = "#F2F2F2"

# Reference lines (e.g., H1 threshold, random baseline).
REF_COLOR = "#888888"


def apply_style() -> None:
    """Apply the N134 figure style to the current matplotlib session."""
    mpl.rcdefaults()

    # Font: serif preferred. STIX Two Text if available, else fallback.
    mpl.rcParams["font.family"] = "serif"
    mpl.rcParams["font.serif"] = [
        "STIX Two Text",
        "STIX",
        "Times New Roman",
        "Times",
        "DejaVu Serif",
        "serif",
    ]

    mpl.rcParams["font.size"] = 9
    mpl.rcParams["axes.labelsize"] = 10
    mpl.rcParams["axes.titlesize"] = 11
    mpl.rcParams["legend.fontsize"] = 8
    mpl.rcParams["xtick.labelsize"] = 9
    mpl.rcParams["ytick.labelsize"] = 9

    mpl.rcParams["axes.edgecolor"] = AXIS_GREY
    mpl.rcParams["axes.labelcolor"] = "black"
    mpl.rcParams["xtick.color"] = AXIS_GREY
    mpl.rcParams["ytick.color"] = AXIS_GREY

    mpl.rcParams["axes.linewidth"] = 0.8
    mpl.rcParams["xtick.major.width"] = 0.8
    mpl.rcParams["ytick.major.width"] = 0.8

    mpl.rcParams["axes.grid"] = True
    mpl.rcParams["grid.color"] = BG_GREY
    mpl.rcParams["grid.linewidth"] = 0.6

    mpl.rcParams["axes.spines.top"] = False
    mpl.rcParams["axes.spines.right"] = False

    mpl.rcParams["figure.dpi"] = 100  # on-screen preview
    mpl.rcParams["savefig.dpi"] = 300  # PNG fallback; PDF is vector anyway
    mpl.rcParams["savefig.bbox"] = "tight"
    mpl.rcParams["savefig.pad_inches"] = 0.02

    mpl.rcParams["pdf.fonttype"] = 42  # TrueType; editable text in PDF
    mpl.rcParams["ps.fonttype"] = 42


def save_figure(fig, name: str) -> tuple[Path, Path]:
    """Save figure to papers/n134_workshop/figures/ as both PDF and PNG.

    Returns the (pdf_path, png_path) tuple. Idempotent: overwrites
    existing files silently, which is the correct behavior for
    regeneration.
    """
    FIGURES_OUT.mkdir(parents=True, exist_ok=True)
    pdf = FIGURES_OUT / f"{name}.pdf"
    png = FIGURES_OUT / f"{name}.png"
    fig.savefig(pdf)
    fig.savefig(png)
    plt.close(fig)
    return pdf, png
