#!/usr/bin/env python3
"""
Generate a composite summary panel for the sidecar's S01 findings.

Three sections, vertically stacked:
  A. Taxonomy scatter (severity vs instability) — compact version
  B. Instability case table (top 6 rows, one per pair)
  C. Regime contrast strip plot — compact version

Output: sidecar/figures/s01_summary_panel.svg + .png (1200 × 1600)
"""
from __future__ import annotations

import json
import random
from pathlib import Path

random.seed(42)

REPO = Path(__file__).resolve().parent.parent.parent
OUT = REPO / "sidecar" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

profiles = json.load(open(REPO / "sidecar" / "results" / "s01" / "instability_profiles.json"))
same_task = json.load(open(REPO / "sidecar" / "results" / "s01" / "same_task_contrast.json"))

# ── Colors & style ─────────────────────────────────────────────────

BG = "#FAFAFA"
TEXT_DARK = "#2C2C2C"
TEXT_MED = "#5A5A5A"
TEXT_LIGHT = "#888888"
GRID = "#E0E0E0"

TAX_COLORS = {
    "backbone_reversal": "#C04040",
    "stable_asymmetric": "#5B8DBE",
}
TAX_LABELS = {
    "backbone_reversal": "Backbone reversal",
    "stable_asymmetric": "Stable asymmetric",
}

FONT = '-apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif'

W = 1200
SECTION_GAP = 50

# ── Section A: Taxonomy Scatter ────────────────────────────────────

HA = 420
ML_A, MR_A, MT_A, MB_A = 90, 180, 65, 50
PW_A = W - ML_A - MR_A
PH_A = HA - MT_A - MB_A

max_sev = 0.45
max_inst = 1.0

svg_a = []


def a(s):
    svg_a.append(s)


# Title
a(f'<text x="{W//2}" y="24" text-anchor="middle" font-size="18" font-weight="700" '
  f'fill="{TEXT_DARK}">S01 Summary: Instability as Working Concept</text>')
a(f'<text x="{W//2}" y="42" text-anchor="middle" font-size="11" fill="{TEXT_LIGHT}">'
  f'Two-backbone phase (DistilBERT + RoBERTa) — DeBERTa-v3 adjudication pending</text>')

# Section label
a(f'<text x="{ML_A}" y="{MT_A - 8}" font-size="13" font-weight="600" fill="{TEXT_DARK}">'
  f'A. Taxonomy Scatter — Severity vs Instability</text>')

# Grid
for v in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45]:
    x = ML_A + (v / max_sev) * PW_A
    a(f'<line x1="{x:.0f}" y1="{MT_A}" x2="{x:.0f}" y2="{MT_A+PH_A}" stroke="{GRID}" stroke-width="0.5"/>')
    a(f'<text x="{x:.0f}" y="{MT_A+PH_A+14}" text-anchor="middle" font-size="8" fill="{TEXT_LIGHT}">{v:.0%}</text>')

for v in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
    y = MT_A + PH_A - (v / max_inst) * PH_A
    a(f'<line x1="{ML_A}" y1="{y:.0f}" x2="{ML_A+PW_A}" y2="{y:.0f}" stroke="{GRID}" stroke-width="0.5"/>')
    a(f'<text x="{ML_A-6}" y="{y+3:.0f}" text-anchor="end" font-size="8" fill="{TEXT_LIGHT}">{v:.1f}</text>')

# Instability gap band
gap_lo = 0.30
gap_hi = 0.74
y_lo = MT_A + PH_A - (gap_lo / max_inst) * PH_A
y_hi = MT_A + PH_A - (gap_hi / max_inst) * PH_A
a(f'<rect x="{ML_A}" y="{y_hi:.0f}" width="{PW_A}" height="{y_lo - y_hi:.0f}" '
  f'fill="#F5F0FF" opacity="0.6"/>')
a(f'<text x="{ML_A + PW_A + 6}" y="{(y_lo + y_hi)/2 + 4:.0f}" font-size="9" fill="#9B7DBF" '
  f'font-style="italic">instability gap</text>')

# Threshold lines
for thresh in [0.05, 0.10, 0.15]:
    x = ML_A + (thresh / max_sev) * PW_A
    a(f'<line x1="{x:.0f}" y1="{MT_A}" x2="{x:.0f}" y2="{MT_A+PH_A}" '
      f'stroke="#CCCCCC" stroke-width="1" stroke-dasharray="4,3"/>')

# Axis labels
a(f'<text x="{ML_A + PW_A//2}" y="{MT_A+PH_A+32}" text-anchor="middle" font-size="11" fill="{TEXT_MED}">'
  f'Max severity (worst delta across both backbones)</text>')
a(f'<text x="18" y="{MT_A + PH_A//2}" text-anchor="middle" font-size="11" fill="{TEXT_MED}" '
  f'transform="rotate(-90, 18, {MT_A + PH_A//2})">Instability score</text>')

# Frame
a(f'<rect x="{ML_A}" y="{MT_A}" width="{PW_A}" height="{PH_A}" fill="none" stroke="#AAAAAA" stroke-width="1"/>')

# Points
for pp in profiles:
    d_worst = pp["distilbert_stats"]["worst"]
    r_worst = pp["roberta_stats"]["worst"]
    sev = max(d_worst, r_worst)
    inst = pp["instability_score"]
    tax = pp["taxonomy"]
    color = TAX_COLORS[tax]
    bb_shift = pp["cross_backbone"]["backbone_shift"]

    x = ML_A + (sev / max_sev) * PW_A
    y = MT_A + PH_A - (inst / max_inst) * PH_A
    r = 8 + bb_shift * 28

    a(f'<circle cx="{x:.0f}" cy="{y:.0f}" r="{r:.0f}" fill="{color}" opacity="0.7" '
      f'stroke="{color}" stroke-width="1.5"/>')

    label = pp["task_pair"].upper().replace(" × ", "×")
    lx = x + r + 5
    ly = y + 4
    a(f'<text x="{lx:.0f}" y="{ly:.0f}" font-size="9" font-weight="500" fill="{TEXT_DARK}">{label}</text>')

# Legend
lx = W - MR_A + 15
ly = MT_A + 10
a(f'<text x="{lx}" y="{ly}" font-size="10" font-weight="600" fill="{TEXT_DARK}">Taxonomy</text>')
for i, (key, label) in enumerate(TAX_LABELS.items()):
    cy = ly + 20 + i * 20
    color = TAX_COLORS[key]
    a(f'<circle cx="{lx + 7}" cy="{cy - 3}" r="5" fill="{color}" opacity="0.7"/>')
    a(f'<text x="{lx + 17}" y="{cy + 1}" font-size="9" fill="{TEXT_MED}">{label}</text>')

sy = ly + 75
a(f'<text x="{lx}" y="{sy}" font-size="9" font-weight="500" fill="{TEXT_DARK}">Dot size =</text>')
a(f'<text x="{lx}" y="{sy+13}" font-size="9" fill="{TEXT_MED}">backbone shift</text>')


# ── Section B: Case Table ──────────────────────────────────────────

# Build sorted pair list
pairs_sorted = sorted(profiles, key=lambda p: -p["instability_score"])

HB_ROW = 24
HB_HEADER = 30
HB_TITLE = 28
HB = HB_TITLE + HB_HEADER + len(pairs_sorted) * HB_ROW + 20
Y_B = HA + SECTION_GAP

# Section label
svg_b = []


def b(s):
    svg_b.append(s)


b(f'<text x="{ML_A}" y="{Y_B + 16}" font-size="13" font-weight="600" fill="{TEXT_DARK}">'
  f'B. Instability Case Table — One Row Per Pair</text>')

# Table layout
COL_X = [ML_A, ML_A + 130, ML_A + 260, ML_A + 370, ML_A + 470, ML_A + 580, ML_A + 690]
COL_LABELS = ["Pair", "Taxonomy", "Instability", "Max Severity", "Seed Range", "BB Shift", "CV"]

y_header = Y_B + HB_TITLE + 14

# Header background
b(f'<rect x="{ML_A - 5}" y="{y_header - 14}" width="{W - ML_A - 40 + 5}" height="{HB_HEADER - 6}" '
  f'fill="#EEEEF4" rx="3"/>')

for i, (cx, label) in enumerate(zip(COL_X, COL_LABELS)):
    b(f'<text x="{cx}" y="{y_header}" font-size="10" font-weight="600" fill="{TEXT_DARK}">{label}</text>')

# Rows
for ri, pp in enumerate(pairs_sorted):
    y_row = y_header + HB_HEADER + ri * HB_ROW
    pair_label = pp["task_pair"].upper().replace(" × ", " × ")
    tax = pp["taxonomy"].replace("_", " ")
    inst = pp["instability_score"]
    d_worst = pp["distilbert_stats"]["worst"]
    r_worst = pp["roberta_stats"]["worst"]
    sev = max(d_worst, r_worst)
    bb_shift = pp["cross_backbone"]["backbone_shift"]
    max_range = pp["cross_backbone"]["max_seed_range"]
    max_cv = pp["cross_backbone"]["max_cv"]

    color = TAX_COLORS[pp["taxonomy"]]

    # Alternating row shading
    if ri % 2 == 0:
        b(f'<rect x="{ML_A - 5}" y="{y_row - 14}" width="{W - ML_A - 40 + 5}" height="{HB_ROW}" '
          f'fill="#F5F5F8" rx="2"/>')

    # Color bar
    b(f'<rect x="{ML_A - 5}" y="{y_row - 12}" width="3" height="{HB_ROW - 4}" fill="{color}" rx="1"/>')

    vals = [
        pair_label,
        tax,
        f"{inst:.2f}",
        f"{sev:.1%}",
        f"{max_range:.1%}",
        f"{bb_shift:.1%}",
        f"{max_cv:.2f}",
    ]
    for ci, (cx, v) in enumerate(zip(COL_X, vals)):
        weight = "500" if ci == 0 else "400"
        b(f'<text x="{cx}" y="{y_row}" font-size="10" font-weight="{weight}" fill="{TEXT_DARK}">{v}</text>')

# Gap annotation
gap_y = y_header + HB_HEADER + 2 * HB_ROW + 4
b(f'<line x1="{COL_X[2]}" y1="{gap_y}" x2="{COL_X[2] + 80}" y2="{gap_y}" '
  f'stroke="#9B7DBF" stroke-width="1.5" stroke-dasharray="4,2"/>')
b(f'<text x="{COL_X[2] + 85}" y="{gap_y + 4}" font-size="9" fill="#9B7DBF" font-style="italic">'
  f'instability gap (0.74 → 0.30)</text>')


# ── Section C: Regime Contrast ─────────────────────────────────────

HC = 380
Y_C = Y_B + HB + SECTION_GAP
ML_C, MR_C, MT_C, MB_C = 200, 40, 35, 45
PW_C = W - ML_C - MR_C
PH_C = HC - MT_C - MB_C

svg_c = []


def c(s):
    svg_c.append(s)


c(f'<text x="{ML_A}" y="{Y_C + 16}" font-size="13" font-weight="600" fill="{TEXT_DARK}">'
  f'C. Regime Contrast — All Evidence Regimes</text>')

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
]

# Cross-task data from main results
try:
    ct_distilbert = json.load(
        open(REPO / "results" / "cross_task_subtype_study_01" / "pairs" / "adjudication_results.json"))
    regime_data.append(("Cross-task (DistilBERT)", "#C04040",
                        [max(e.get("delta_task_a", 0), e.get("delta_task_b", 0)) for e in ct_distilbert
                         if e.get("distance", "") != "same_task"]))
except FileNotFoundError:
    pass

try:
    ct_roberta = json.load(
        open(REPO / "results" / "task_pair_severity_generalization_study_01" / "roberta" / "pairs" / "adjudication_results.json"))
    regime_data.append(("Cross-task (RoBERTa)", "#E8A838",
                        [max(e.get("delta_task_a", 0), e.get("delta_task_b", 0)) for e in ct_roberta
                         if e.get("task_a", "") != e.get("task_b", "")]))
except FileNotFoundError:
    pass

N_REGIMES = len(regime_data)
ROW_H = PH_C / max(N_REGIMES, 1)
max_x = 0.45
x_scale = PW_C / max_x

Y_C_PLOT = Y_C + MT_C + 20

# Threshold bands
for lo, hi, color in [(0.15, max_x, "#FFE0E0"), (0.10, 0.15, "#FFF0DD"), (0.05, 0.10, "#FFFFF0")]:
    x1 = ML_C + lo * x_scale
    x2 = ML_C + hi * x_scale
    c(f'<rect x="{x1:.0f}" y="{Y_C_PLOT}" width="{x2-x1:.0f}" height="{PH_C}" fill="{color}" opacity="0.35"/>')

# Grid
for v in [0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45]:
    x = ML_C + v * x_scale
    c(f'<line x1="{x:.0f}" y1="{Y_C_PLOT}" x2="{x:.0f}" y2="{Y_C_PLOT+PH_C}" stroke="{GRID}" stroke-width="0.5"/>')
    c(f'<text x="{x:.0f}" y="{Y_C_PLOT+PH_C+14}" text-anchor="middle" font-size="8" fill="{TEXT_LIGHT}">{v:.0%}</text>')

c(f'<text x="{ML_C + PW_C//2}" y="{Y_C_PLOT+PH_C+32}" text-anchor="middle" font-size="11" fill="{TEXT_MED}">'
  f'Worst-case delta per pair</text>')

# Rows
for i, (label, color, deltas) in enumerate(regime_data):
    cy = Y_C_PLOT + i * ROW_H + ROW_H / 2

    c(f'<text x="{ML_C - 10}" y="{cy + 4}" text-anchor="end" font-size="10" font-weight="500" '
      f'fill="{TEXT_DARK}">{label}</text>')
    c(f'<line x1="{ML_C}" y1="{cy}" x2="{ML_C+PW_C}" y2="{cy}" stroke="{GRID}" stroke-width="0.5"/>')

    for d in deltas:
        d_clamped = max(d, 0)
        x = ML_C + d_clamped * x_scale
        jitter = random.uniform(-ROW_H * 0.25, ROW_H * 0.25)
        y = cy + jitter
        c(f'<circle cx="{x:.0f}" cy="{y:.0f}" r="3" fill="{color}" opacity="0.55"/>')

# Frame
c(f'<rect x="{ML_C}" y="{Y_C_PLOT}" width="{PW_C}" height="{PH_C}" fill="none" stroke="#AAAAAA" stroke-width="1"/>')

# Threshold labels
for thresh, label in [(0.05, "broad"), (0.10, "severe"), (0.15, "catastrophic")]:
    x = ML_C + thresh * x_scale
    c(f'<text x="{x + 3}" y="{Y_C_PLOT + 12}" font-size="8" fill="{TEXT_LIGHT}" font-style="italic">{label}</text>')


# ── Assemble ───────────────────────────────────────────────────────

TOTAL_H = Y_C + HC + 20

header = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {TOTAL_H}" width="{W}" height="{TOTAL_H}">
<style>text {{ font-family: {FONT}; }}</style>
<rect width="{W}" height="{TOTAL_H}" fill="{BG}"/>'''

footer = "</svg>"

full_svg = "\n".join([header] + svg_a + svg_b + svg_c + [footer])

svg_path = OUT / "s01_summary_panel.svg"
with open(svg_path, "w") as f:
    f.write(full_svg)
print(f"Written: {svg_path}")

try:
    import cairosvg
    png_path = OUT / "s01_summary_panel.png"
    cairosvg.svg2png(url=str(svg_path), write_to=str(png_path), output_width=W * 2, output_height=TOTAL_H * 2)
    print(f"Written: {png_path}")
except ImportError:
    print("cairosvg not available — PNG not generated")
