#!/usr/bin/env python3
"""
Generate seed stability figure for Study S01.

For each task pair on each backbone, shows the range of max_delta across
seed variants as a range plot (dot = mean, whisker = min–max).

Usage:
    python sidecar/benchmarks/gen_seed_stability_figure.py
"""
from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SEED_PATH = REPO_ROOT / "sidecar" / "results" / "s01" / "seed_stability.json"
FIGURES_DIR = REPO_ROOT / "sidecar" / "figures"

with open(SEED_PATH) as f:
    seed_data = json.load(f)

# Figure config
W = 1200
H = 580
MARGIN_L = 180
MARGIN_R = 80
MARGIN_T = 70
MARGIN_B = 70
PLOT_W = W - MARGIN_L - MARGIN_R
PLOT_H = H - MARGIN_T - MARGIN_B

# Canonical task pair order
PAIR_ORDER = ["qnli × mrpc", "qnli × sst2", "mrpc × sst2", "rte × sst2", "rte × mrpc", "qnli × rte"]
N = len(PAIR_ORDER)
ROW_H = PLOT_H / N

# Colors
DISTILBERT_COLOR = "#5B8DBE"
ROBERTA_COLOR = "#C27C5E"
BG = "#FAFAFA"
TEXT_DARK = "#2C2C2C"
TEXT_MED = "#5A5A5A"
TEXT_LIGHT = "#888888"

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
add(f'<text x="{W//2}" y="28" text-anchor="middle" font-size="18" font-weight="600" fill="{TEXT_DARK}">'
    f'Seed Stability: Range of Worst-Case Delta Across Seed Variants</text>')
add(f'<text x="{W//2}" y="48" text-anchor="middle" font-size="12" fill="{TEXT_LIGHT}">'
    f'Study S01 — Whiskers show min→max across seed combos; dot = mean. '
    f'Threshold bands from Panel P01.</text>')

px = MARGIN_L
py = MARGIN_T

# Threshold bands
threshold_bands = [
    (0.15, max_val, "#FFE0E0", 0.4),
    (0.10, 0.15, "#FFF0DD", 0.4),
    (0.05, 0.10, "#FFFFF0", 0.4),
]
for lo, hi, color, op in threshold_bands:
    x1 = px + lo * x_scale
    x2 = px + hi * x_scale
    add(f'<rect x="{x1:.0f}" y="{py}" width="{x2-x1:.0f}" height="{PLOT_H}" fill="{color}" opacity="{op}"/>')

# Threshold lines
for val in [0.05, 0.10, 0.15]:
    x = px + val * x_scale
    add(f'<line x1="{x:.0f}" y1="{py}" x2="{x:.0f}" y2="{py + PLOT_H}" '
        f'stroke="#CCCCCC" stroke-width="1" stroke-dasharray="4,3"/>')

# X-axis
for val in [0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45]:
    x = px + val * x_scale
    add(f'<text x="{x:.0f}" y="{py + PLOT_H + 16}" text-anchor="middle" '
        f'font-size="10" fill="{TEXT_LIGHT}">{val:.0%}</text>')

add(f'<text x="{px + PLOT_W/2}" y="{py + PLOT_H + 38}" text-anchor="middle" '
    f'font-size="12" fill="{TEXT_MED}">Worst-case delta per seed variant</text>')

# Zero line
add(f'<line x1="{px}" y1="{py}" x2="{px}" y2="{py + PLOT_H}" stroke="#AAAAAA" stroke-width="1"/>')

# Build lookup
distilbert_lookup = {r["task_pair"]: r for r in seed_data["distilbert"]}
roberta_lookup = {r["task_pair"]: r for r in seed_data["roberta"]}

for i, pair in enumerate(PAIR_ORDER):
    cy = py + i * ROW_H + ROW_H / 2
    label = pair.upper()

    # Row label
    add(f'<text x="{px - 10}" y="{cy + 2}" text-anchor="end" font-size="12" '
        f'font-weight="500" fill="{TEXT_DARK}">{label}</text>')

    # Horizontal guide
    add(f'<line x1="{px}" y1="{cy}" x2="{px + PLOT_W}" y2="{cy}" '
        f'stroke="#E8E8E8" stroke-width="0.5"/>')

    offset = -10  # DistilBERT slightly above center
    for backbone_name, lookup, color, y_off in [
        ("distilbert", distilbert_lookup, DISTILBERT_COLOR, -10),
        ("roberta", roberta_lookup, ROBERTA_COLOR, 10),
    ]:
        if pair not in lookup:
            continue
        r = lookup[pair]
        lo = r["best_delta"]
        hi = r["worst_delta"]
        mean = r["mean_delta"]

        x_lo = px + lo * x_scale
        x_hi = px + hi * x_scale
        x_mean = px + mean * x_scale
        y = cy + y_off

        # Whisker line
        add(f'<line x1="{x_lo:.0f}" y1="{y}" x2="{x_hi:.0f}" y2="{y}" '
            f'stroke="{color}" stroke-width="2" opacity="0.7"/>')
        # End caps
        add(f'<line x1="{x_lo:.0f}" y1="{y-4}" x2="{x_lo:.0f}" y2="{y+4}" '
            f'stroke="{color}" stroke-width="1.5" opacity="0.7"/>')
        add(f'<line x1="{x_hi:.0f}" y1="{y-4}" x2="{x_hi:.0f}" y2="{y+4}" '
            f'stroke="{color}" stroke-width="1.5" opacity="0.7"/>')
        # Mean dot
        add(f'<circle cx="{x_mean:.0f}" cy="{y}" r="4" fill="{color}" opacity="0.9"/>')

        # Range annotation
        range_val = hi - lo
        if range_val > 0.02:
            add(f'<text x="{x_hi + 6:.0f}" y="{y + 4}" font-size="9" fill="{color}">'
                f'range={range_val:.1%}</text>')

# Legend
lx = px + PLOT_W - 240
ly = py - 12
add(f'<circle cx="{lx}" cy="{ly}" r="4" fill="{DISTILBERT_COLOR}" opacity="0.9"/>')
add(f'<text x="{lx + 10}" y="{ly + 4}" font-size="11" fill="{TEXT_MED}">DistilBERT (6L)</text>')
add(f'<circle cx="{lx + 130}" cy="{ly}" r="4" fill="{ROBERTA_COLOR}" opacity="0.9"/>')
add(f'<text x="{lx + 140}" y="{ly + 4}" font-size="11" fill="{TEXT_MED}">RoBERTa (12L)</text>')

add("</svg>")

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
svg_content = "\n".join(svg)

svg_path = FIGURES_DIR / "s01_seed_stability.svg"
with open(svg_path, "w") as f:
    f.write(svg_content)
print(f"SVG written to {svg_path}")

try:
    import cairosvg
    png_path = FIGURES_DIR / "s01_seed_stability.png"
    cairosvg.svg2png(url=str(svg_path), write_to=str(png_path), output_width=1200, output_height=580)
    print(f"PNG written to {png_path}")
except ImportError:
    print("cairosvg not available — skipping PNG export")
