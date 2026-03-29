#!/usr/bin/env python3
"""
Generate the backbone shift figure for Study S01.

Produces a paired bar chart showing how each task pair's worst-case delta
changes between DistilBERT and RoBERTa, with severity threshold bands
and shift annotations.

Usage:
    python sidecar/benchmarks/gen_backbone_shift_figure.py

Outputs:
    sidecar/figures/s01_backbone_shift.svg
    sidecar/figures/s01_backbone_shift.png
"""
from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
COMPARISON_PATH = REPO_ROOT / "sidecar" / "results" / "s01" / "three_backbone_comparison.json"
FIGURES_DIR = REPO_ROOT / "sidecar" / "figures"

# Load data
with open(COMPARISON_PATH) as f:
    comparison = json.load(f)

# Sort by DistilBERT worst delta descending
comparison.sort(key=lambda r: r.get("distilbert_worst_delta", 0), reverse=True)

# Figure config
W = 1200
H = 700
MARGIN_L = 180
MARGIN_R = 60
MARGIN_T = 80
MARGIN_B = 80
PLOT_W = W - MARGIN_L - MARGIN_R
PLOT_H = H - MARGIN_T - MARGIN_B

N = len(comparison)
BAR_GROUP_H = PLOT_H / N
BAR_H = 16
BAR_GAP = 4

# Colors
DISTILBERT_COLOR = "#5B8DBE"
ROBERTA_COLOR = "#C27C5E"
BG = "#FAFAFA"
GRID = "#E8E8E8"
TEXT_DARK = "#2C2C2C"
TEXT_MED = "#5A5A5A"
TEXT_LIGHT = "#888888"
THRESHOLD_COLORS = {
    "catastrophic": "#FFE0E0",
    "severe": "#FFF0DD",
    "broad": "#FFFFF0",
}

# Scale: max delta
max_val = 0.45
x_scale = PLOT_W / max_val

svg = []


def add(s):
    svg.append(s)


add(f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">
<style>
  text {{ font-family: -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }}
</style>
<rect width="{W}" height="{H}" fill="{BG}"/>
''')

# Title
add(f'<text x="{W//2}" y="32" text-anchor="middle" font-size="18" font-weight="600" fill="{TEXT_DARK}">'
    f'Backbone Shift: Worst-Case Merge Delta by Task Pair</text>')
add(f'<text x="{W//2}" y="52" text-anchor="middle" font-size="12" fill="{TEXT_LIGHT}">'
    f'Study S01 — Two-backbone comparison (Panel P01 thresholds shown)</text>')

# Plot area
px = MARGIN_L
py = MARGIN_T

# Threshold bands
thresholds = [
    (0.15, max_val, THRESHOLD_COLORS["catastrophic"], "Catastrophic (>15%)"),
    (0.10, 0.15, THRESHOLD_COLORS["severe"], "Severe (10–15%)"),
    (0.05, 0.10, THRESHOLD_COLORS["broad"], "Broad (5–10%)"),
]
for lo, hi, color, label in thresholds:
    x1 = px + lo * x_scale
    x2 = px + hi * x_scale
    add(f'<rect x="{x1:.0f}" y="{py}" width="{x2-x1:.0f}" height="{PLOT_H}" fill="{color}" opacity="0.5"/>')

# Threshold lines
for val in [0.05, 0.10, 0.15]:
    x = px + val * x_scale
    add(f'<line x1="{x:.0f}" y1="{py}" x2="{x:.0f}" y2="{py + PLOT_H}" '
        f'stroke="#CCCCCC" stroke-width="1" stroke-dasharray="4,3"/>')
    add(f'<text x="{x:.0f}" y="{py + PLOT_H + 16}" text-anchor="middle" '
        f'font-size="10" fill="{TEXT_LIGHT}">{val:.0%}</text>')

# X-axis labels
for val in [0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45]:
    x = px + val * x_scale
    if val not in [0.05, 0.10, 0.15]:
        add(f'<text x="{x:.0f}" y="{py + PLOT_H + 16}" text-anchor="middle" '
            f'font-size="10" fill="{TEXT_LIGHT}">{val:.0%}</text>')

add(f'<text x="{px + PLOT_W/2}" y="{py + PLOT_H + 40}" text-anchor="middle" '
    f'font-size="12" fill="{TEXT_MED}">Worst-case delta (max of Δ_task_a, Δ_task_b)</text>')

# Grid line at 0
add(f'<line x1="{px}" y1="{py}" x2="{px}" y2="{py + PLOT_H}" stroke="#AAAAAA" stroke-width="1"/>')

# Bars
for i, row in enumerate(comparison):
    cy = py + i * BAR_GROUP_H + BAR_GROUP_H / 2
    pair_label = row["task_pair"].upper()

    # Label
    add(f'<text x="{px - 10}" y="{cy + 2}" text-anchor="end" font-size="12" '
        f'font-weight="500" fill="{TEXT_DARK}">{pair_label}</text>')

    # Horizontal guide
    add(f'<line x1="{px}" y1="{cy}" x2="{px + PLOT_W}" y2="{cy}" '
        f'stroke="{GRID}" stroke-width="0.5"/>')

    # DistilBERT bar (top)
    d_delta = row.get("distilbert_worst_delta", 0)
    d_w = d_delta * x_scale
    bar_y1 = cy - BAR_H - BAR_GAP / 2
    add(f'<rect x="{px}" y="{bar_y1:.0f}" width="{d_w:.0f}" height="{BAR_H}" '
        f'fill="{DISTILBERT_COLOR}" rx="2" opacity="0.85"/>')
    if d_delta > 0.03:
        add(f'<text x="{px + d_w + 4:.0f}" y="{bar_y1 + BAR_H - 3:.0f}" '
            f'font-size="10" fill="{DISTILBERT_COLOR}" font-weight="500">{d_delta:.1%}</text>')

    # RoBERTa bar (bottom)
    r_delta = row.get("roberta_worst_delta", 0)
    r_w = r_delta * x_scale
    bar_y2 = cy + BAR_GAP / 2
    add(f'<rect x="{px}" y="{bar_y2:.0f}" width="{r_w:.0f}" height="{BAR_H}" '
        f'fill="{ROBERTA_COLOR}" rx="2" opacity="0.85"/>')
    if r_delta > 0.03:
        add(f'<text x="{px + r_w + 4:.0f}" y="{bar_y2 + BAR_H - 3:.0f}" '
            f'font-size="10" fill="{ROBERTA_COLOR}" font-weight="500">{r_delta:.1%}</text>')

    # Shift annotation
    shift = row.get("shift_type", "")
    if shift == "collapses_on_roberta":
        label = "collapses"
        color = "#2B7A3E"
    elif shift == "escalates_on_roberta":
        label = "escalates"
        color = "#C04040"
    else:
        label = "stable"
        color = TEXT_LIGHT

    ann_x = px + max(d_w, r_w) * 1 + 55
    if ann_x > W - MARGIN_R - 60:
        ann_x = W - MARGIN_R - 10
    add(f'<text x="{ann_x:.0f}" y="{cy + 4:.0f}" text-anchor="end" '
        f'font-size="9" fill="{color}" font-style="italic">{label}</text>')

# Legend
lx = px + PLOT_W - 240
ly = py - 15
add(f'<rect x="{lx}" y="{ly - 10}" width="{14}" height="{14}" fill="{DISTILBERT_COLOR}" rx="2" opacity="0.85"/>')
add(f'<text x="{lx + 20}" y="{ly + 2}" font-size="11" fill="{TEXT_MED}">DistilBERT (6 layers)</text>')
add(f'<rect x="{lx + 150}" y="{ly - 10}" width="{14}" height="{14}" fill="{ROBERTA_COLOR}" rx="2" opacity="0.85"/>')
add(f'<text x="{lx + 170}" y="{ly + 2}" font-size="11" fill="{TEXT_MED}">RoBERTa (12 layers)</text>')

# Footer
add(f'<text x="{W//2}" y="{H - 15}" text-anchor="middle" font-size="11" fill="{TEXT_LIGHT}">'
    f'Each bar shows the worst-case seed variant. '
    f'Shift labels indicate how severity changes from DistilBERT → RoBERTa.</text>')

add("</svg>")

# Write
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
svg_content = "\n".join(svg)

svg_path = FIGURES_DIR / "s01_backbone_shift.svg"
with open(svg_path, "w") as f:
    f.write(svg_content)
print(f"SVG written to {svg_path}")

# PNG export
try:
    import cairosvg
    png_path = FIGURES_DIR / "s01_backbone_shift.png"
    cairosvg.svg2png(url=str(svg_path), write_to=str(png_path), output_width=1200, output_height=700)
    print(f"PNG written to {png_path}")
except ImportError:
    print("cairosvg not available — skipping PNG export")
