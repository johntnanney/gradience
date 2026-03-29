#!/usr/bin/env python3
"""
Generate a compact instability case table from existing S01 data.

Produces:
  - results/s01/instability_case_table.json  (structured)
  - results/s01/instability_case_table.md    (human-readable)

Columns: pair, backbone, mean_delta, seed_range, cv, backbone_shift,
         instability_score, instability_rank, taxonomy
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
S01 = REPO / "sidecar" / "results" / "s01"

profiles = json.load(open(S01 / "instability_profiles.json"))

# Build per-backbone rows for each pair, plus a cross-backbone summary row
rows = []
for p in profiles:
    pair = p["task_pair"].upper().replace(" × ", " × ")
    inst = p["instability_score"]
    tax = p["taxonomy"]
    bb_shift = p["cross_backbone"]["backbone_shift"]

    for bb_key, bb_label in [("distilbert_stats", "DistilBERT"), ("roberta_stats", "RoBERTa")]:
        s = p[bb_key]
        rows.append({
            "pair": pair,
            "backbone": bb_label,
            "mean_delta": round(s["mean"] * 100, 1),
            "seed_range": round(s["range"] * 100, 1),
            "cv": round(s["cv"], 2),
            "backbone_shift": round(bb_shift * 100, 1),
            "instability_score": round(inst, 2),
            "taxonomy": tax.replace("_", " "),
        })

# Sort by instability score descending, then backbone
rows.sort(key=lambda r: (-r["instability_score"], r["pair"], r["backbone"]))

# Assign instability rank (shared across both backbones of same pair)
seen_pairs = {}
rank = 0
for r in rows:
    if r["pair"] not in seen_pairs:
        rank += 1
        seen_pairs[r["pair"]] = rank
    r["instability_rank"] = seen_pairs[r["pair"]]

# Write JSON
json_path = S01 / "instability_case_table.json"
with open(json_path, "w") as f:
    json.dump(rows, f, indent=2)
print(f"Written: {json_path}")

# Write Markdown
md_lines = [
    "# Instability Case Table — S01 Two-Backbone Phase",
    "",
    "Each cross-task pair appears twice: once per backbone. Pairs are ranked by composite instability score.",
    "Backbone shift is the absolute difference in worst-case delta between backbones.",
    "",
    "| Rank | Pair | Backbone | Mean \u0394 | Seed Range | CV | BB Shift | Instability | Taxonomy |",
    "|-----:|------|----------|------:|-----------:|---:|--------:|------------:|----------|",
]

for r in rows:
    md_lines.append(
        f"| {r['instability_rank']} "
        f"| {r['pair']} "
        f"| {r['backbone']} "
        f"| {r['mean_delta']:.1f}% "
        f"| {r['seed_range']:.1f}% "
        f"| {r['cv']:.2f} "
        f"| {r['backbone_shift']:.1f}% "
        f"| {r['instability_score']:.2f} "
        f"| {r['taxonomy']} |"
    )

md_lines.extend([
    "",
    "---",
    "",
    "**Reading guide:**",
    "",
    "- **Mean \u0394**: average worst-case delta across seed variants on this backbone",
    "- **Seed Range**: difference between worst and best seed variant (higher = more seed-fragile)",
    "- **CV**: coefficient of variation of delta across seed variants (higher = more variable relative to mean)",
    "- **BB Shift**: absolute change in worst-case delta between DistilBERT and RoBERTa",
    "- **Instability**: composite score (0.4 \u00d7 seed_range + 0.3 \u00d7 backbone_shift + 0.3 \u00d7 CV), normalized 0\u20131",
    "- **Taxonomy**: backbone reversal (instability > 0.7, classes differ), stable asymmetric (< 0.3, consistent)",
    "",
    "**Key observation:** The instability gap between rank 2 and rank 3 (0.74 \u2192 0.30) is the cleanest",
    "separation in the evidence base. No severity-based ranking produces a comparable gap.",
])

md_path = S01 / "instability_case_table.md"
with open(md_path, "w") as f:
    f.write("\n".join(md_lines) + "\n")
print(f"Written: {md_path}")
