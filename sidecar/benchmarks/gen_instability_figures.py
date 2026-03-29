#!/usr/bin/env python3
"""
Generate instability analysis figures for Study S01.

Figure 1: Taxonomy scatter (severity vs instability)
Figure 2: Regime contrast (all evidence regimes side-by-side)

Usage:
    python sidecar/benchmarks/gen_instability_figures.py
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
OUT = REPO / "sidecar" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

profiles = json.load(open(REPO / "sidecar" / "results" / "s01" / "instability_profiles.json"))
regimes = json.load(open(REPO / "sidecar" / "results" / "s01" / "regime_summaries.json"))
same_task = json.load(open(REPO / "sidecar" / "results" / "s01" / "same_task_contrast.json"))

# ── Figure 1: Taxonomy Scatter ──────────────────────────────────────
# X = max severity (worst delta across both backbones)
# Y = instability score
# Color = taxonomy class
# Size = backbone shift magnitude

W, H = 900, 650
ML, MR, MT, MB = 80, 180, 70, 70
PW = W - ML - MR
PH = H - MT - MB

TAXONOMY_COLORS = {
    "backbone_reversal": "#C04040",
    "unstable_severe": "#E8A838",
    "stable_asymmetric": "#5B8DBE",
    "stable_mild": "#6BAF7A",
}
TAXONOMY_LABELS = {
    "backbone_reversal": "Backbone reversal",
    "unstable_severe": "Unstable severe",
    "stable_asymmetric": "Stable asymmetric",
    "stable_mild": "Stable mild",
}

BG = "#FAFAFA"
TEXT_DARK = "#2C2C2C"
TEXT_MED = "#5A5A5A"
TEXT_LIGHT = "#888888"
GRID = "#E0E0E0"

svg = []


def add(s):
    svg.append(s)


add(f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">
<style>text {{ font-family: -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }}</style>
<rect width="{W}" height="{H}" fill="{BG}"/>''')

# Title
add(f'<text x="{ML + PW//2}" y="28" text-anchor="middle" font-size="17" font-weight="600" '
    f'fill="{TEXT_DARK}">Cross-Task Pair Taxonomy: Severity vs Instability</text>')
add(f'<text x="{ML + PW//2}" y="48" text-anchor="middle" font-size="11" fill="{TEXT_LIGHT}">'
    f'Each dot is one cross-task pair. X = worst delta across both backbones. '
    f'Y = composite instability score.</text>')

# Axes
max_sev = 0.45
max_inst = 1.0

# Grid
for v in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45]:
    x = ML + (v / max_sev) * PW
    add(f'<line x1="{x:.0f}" y1="{MT}" x2="{x:.0f}" y2="{MT+PH}" stroke="{GRID}" stroke-width="0.5"/>')
    add(f'<text x="{x:.0f}" y="{MT+PH+16}" text-anchor="middle" font-size="9" fill="{TEXT_LIGHT}">{v:.0%}</text>')

for v in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
    y = MT + PH - (v / max_inst) * PH
    add(f'<line x1="{ML}" y1="{y:.0f}" x2="{ML+PW}" y2="{y:.0f}" stroke="{GRID}" stroke-width="0.5"/>')
    add(f'<text x="{ML-8}" y="{y+4:.0f}" text-anchor="end" font-size="9" fill="{TEXT_LIGHT}">{v:.1f}</text>')

# Threshold lines
for thresh, label in [(0.05, "5%"), (0.10, "10%"), (0.15, "15%")]:
    x = ML + (thresh / max_sev) * PW
    add(f'<line x1="{x:.0f}" y1="{MT}" x2="{x:.0f}" y2="{MT+PH}" '
        f'stroke="#CCCCCC" stroke-width="1" stroke-dasharray="4,3"/>')

# Axis labels
add(f'<text x="{ML + PW//2}" y="{MT+PH+40}" text-anchor="middle" font-size="12" fill="{TEXT_MED}">'
    f'Max severity (worst delta across both backbones)</text>')
add(f'<text x="18" y="{MT + PH//2}" text-anchor="middle" font-size="12" fill="{TEXT_MED}" '
    f'transform="rotate(-90, 18, {MT + PH//2})">Instability score</text>')

# Axes border
add(f'<rect x="{ML}" y="{MT}" width="{PW}" height="{PH}" fill="none" stroke="#AAAAAA" stroke-width="1"/>')

# Plot points
for pp in profiles:
    d_worst = pp["distilbert_stats"]["worst"]
    r_worst = pp["roberta_stats"]["worst"]
    sev = max(d_worst, r_worst)
    inst = pp["instability_score"]
    tax = pp["taxonomy"]
    color = TAXONOMY_COLORS[tax]
    backbone_shift = pp["cross_backbone"]["backbone_shift"]

    x = ML + (sev / max_sev) * PW
    y = MT + PH - (inst / max_inst) * PH
    r = 8 + backbone_shift * 30  # radius scales with backbone shift

    add(f'<circle cx="{x:.0f}" cy="{y:.0f}" r="{r:.0f}" fill="{color}" opacity="0.7" '
        f'stroke="{color}" stroke-width="1.5"/>')

    # Label
    label = pp["task_pair"].upper().replace(" × ", "×")
    lx = x + r + 5
    ly = y + 4
    add(f'<text x="{lx:.0f}" y="{ly:.0f}" font-size="10" font-weight="500" fill="{TEXT_DARK}">{label}</text>')

# Legend
lx = W - MR + 15
ly = MT + 10
add(f'<text x="{lx}" y="{ly}" font-size="11" font-weight="600" fill="{TEXT_DARK}">Taxonomy</text>')
for i, (key, label) in enumerate(TAXONOMY_LABELS.items()):
    cy = ly + 22 + i * 22
    color = TAXONOMY_COLORS[key]
    add(f'<circle cx="{lx + 7}" cy="{cy - 3}" r="6" fill="{color}" opacity="0.7"/>')
    add(f'<text x="{lx + 18}" y="{cy + 1}" font-size="10" fill="{TEXT_MED}">{label}</text>')

# Size legend
sy = ly + 130
add(f'<text x="{lx}" y="{sy}" font-size="10" font-weight="500" fill="{TEXT_DARK}">Dot size =</text>')
add(f'<text x="{lx}" y="{sy+14}" font-size="10" fill="{TEXT_MED}">backbone shift</text>')

add("</svg>")

svg_path = OUT / "s01_taxonomy_scatter.svg"
with open(svg_path, "w") as f:
    f.write("\n".join(svg))
print(f"Written: {svg_path}")

try:
    import cairosvg
    png_path = OUT / "s01_taxonomy_scatter.png"
    cairosvg.svg2png(url=str(svg_path), write_to=str(png_path), output_width=900, output_height=650)
    print(f"Written: {png_path}")
except ImportError:
    pass


# ── Figure 2: Regime Contrast ───────────────────────────────────────
# Strip plot showing every pair's delta, grouped by regime

svg2 = []
W2, H2 = 1100, 550
ML2, MR2, MT2, MB2 = 200, 40, 60, 60
PW2 = W2 - ML2 - MR2
PH2 = H2 - MT2 - MB2


def add2(s):
    svg2.append(s)


add2(f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W2} {H2}" width="{W2}" height="{H2}">
<style>text {{ font-family: -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }}</style>
<rect width="{W2}" height="{H2}" fill="{BG}"/>''')

add2(f'<text x="{ML2 + PW2//2}" y="28" text-anchor="middle" font-size="17" font-weight="600" '
     f'fill="{TEXT_DARK}">Regime Contrast: Delta Distributions Across All Evidence</text>')
add2(f'<text x="{ML2 + PW2//2}" y="48" text-anchor="middle" font-size="11" fill="{TEXT_LIGHT}">'
     f'Each dot is one pair. Same-task regimes cluster near zero; cross-task regimes spread widely.</text>')

# Build regime data
regime_data = [
    ("Same-task (DistilBERT)", "#6BAF7A",
     [p["delta"] for p in same_task if p["backbone"] == "distilbert" and p["regime"] == "same_task_cross_study"]),
    ("Same-task (RoBERTa)", "#6BAF7A",
     [p["delta"] for p in same_task if p["backbone"] == "roberta" and p["regime"] == "same_task_cross_study"]),
    ("Domain shift", "#9B7DBF",
     [p["delta"] for p in same_task if p["regime"] == "domain_shift"]),
    ("Source strength", "#5B8DBE",
     [p["delta"] for p in same_task if p["regime"] == "source_strength"]),
    ("Training style", "#5B8DBE",
     [p["delta"] for p in same_task if p["regime"] == "training_style"]),
    ("Cross-task (DistilBERT)", "#C04040",
     [max(e.get("delta_task_a", 0), e.get("delta_task_b", 0)) for e in
      json.load(open(REPO / "results" / "cross_task_subtype_study_01" / "pairs" / "adjudication_results.json"))
      if e.get("distance", "") != "same_task"]),
    ("Cross-task (RoBERTa)", "#E8A838",
     [max(e.get("delta_task_a", 0), e.get("delta_task_b", 0)) for e in
      json.load(open(REPO / "results" / "task_pair_severity_generalization_study_01" / "roberta" / "pairs" / "adjudication_results.json"))
      if e.get("task_a", "") != e.get("task_b", "")]),
]

N_REGIMES = len(regime_data)
ROW_H = PH2 / N_REGIMES
max_x = 0.45
x_scale = PW2 / max_x

# Threshold bands
for lo, hi, color in [(0.15, max_x, "#FFE0E0"), (0.10, 0.15, "#FFF0DD"), (0.05, 0.10, "#FFFFF0")]:
    x1 = ML2 + lo * x_scale
    x2 = ML2 + hi * x_scale
    add2(f'<rect x="{x1:.0f}" y="{MT2}" width="{x2-x1:.0f}" height="{PH2}" fill="{color}" opacity="0.4"/>')

# Grid
for v in [0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45]:
    x = ML2 + v * x_scale
    add2(f'<line x1="{x:.0f}" y1="{MT2}" x2="{x:.0f}" y2="{MT2+PH2}" stroke="{GRID}" stroke-width="0.5"/>')
    add2(f'<text x="{x:.0f}" y="{MT2+PH2+16}" text-anchor="middle" font-size="9" fill="{TEXT_LIGHT}">{v:.0%}</text>')

add2(f'<text x="{ML2 + PW2//2}" y="{MT2+PH2+38}" text-anchor="middle" font-size="12" fill="{TEXT_MED}">'
     f'Worst-case delta per pair</text>')

import random
random.seed(42)

for i, (label, color, deltas) in enumerate(regime_data):
    cy = MT2 + i * ROW_H + ROW_H / 2

    # Label
    add2(f'<text x="{ML2 - 10}" y="{cy + 4}" text-anchor="end" font-size="11" font-weight="500" '
         f'fill="{TEXT_DARK}">{label}</text>')

    # Horizontal guide
    add2(f'<line x1="{ML2}" y1="{cy}" x2="{ML2+PW2}" y2="{cy}" stroke="{GRID}" stroke-width="0.5"/>')

    # Dots (jittered vertically)
    for d in deltas:
        d_clamped = max(d, 0)
        x = ML2 + d_clamped * x_scale
        jitter = random.uniform(-ROW_H * 0.25, ROW_H * 0.25)
        y = cy + jitter
        add2(f'<circle cx="{x:.0f}" cy="{y:.0f}" r="3.5" fill="{color}" opacity="0.6"/>')

# Frame
add2(f'<rect x="{ML2}" y="{MT2}" width="{PW2}" height="{PH2}" fill="none" stroke="#AAAAAA" stroke-width="1"/>')

add2("</svg>")

svg2_path = OUT / "s01_regime_contrast.svg"
with open(svg2_path, "w") as f:
    f.write("\n".join(svg2))
print(f"Written: {svg2_path}")

try:
    import cairosvg
    png2_path = OUT / "s01_regime_contrast.png"
    cairosvg.svg2png(url=str(svg2_path), write_to=str(png2_path), output_width=1100, output_height=550)
    print(f"Written: {png2_path}")
except ImportError:
    pass
