#!/usr/bin/env python3
"""Generate figures for the Output Example Semantics program.

Figure 1: Preservation / breakage comparison (grouped bar)
Figure 2: Failure taxonomy composition (stacked bar)
Figure 3: Confidence / ambiguity comparison
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "results" / "example_semantics"
FIGURE_DIR = Path(__file__).resolve().parent.parent / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

# Load data
with open(OUTPUT_DIR / "example_behavior_summary.json") as f:
    summary = json.load(f)

# Order: safe, near_miss, fragile, control, anchor
ORDER = ["SR-01", "SR-02", "NM-01", "NM-02", "FR-01", "FR-02", "CT-01", "AN-01"]
data_by_id = {r["case_id"]: r for r in summary}
ordered = [data_by_id[cid] for cid in ORDER]

CLASS_COLORS = {
    "safe_retained": "#2ecc71",
    "near_miss": "#3498db",
    "fragile": "#e67e22",
    "control": "#e74c3c",
    "anchor": "#9b59b6",
}

# ─────────────────────────────────────────────
# Figure 1: Preservation / breakage comparison
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 5))

labels = [r["case_id"] for r in ordered]
colors = [CLASS_COLORS[r["case_class"]] for r in ordered]

# Preservation rate
vals = [r["metrics"]["preservation_rate"] for r in ordered]
axes[0].barh(labels, vals, color=colors)
axes[0].set_xlim(0, 1.05)
axes[0].set_xlabel("Source Preservation Rate")
axes[0].set_title("Preservation Rate")
axes[0].axvline(x=0.8, color="gray", linestyle="--", alpha=0.5)

# Joint breakage rate
vals = [r["metrics"]["joint_breakage_rate"] if r["metrics"]["joint_breakage_rate"] is not None else 0 for r in ordered]
axes[1].barh(labels, vals, color=colors)
axes[1].set_xlim(0, max(vals) * 1.2 + 0.01)
axes[1].set_xlabel("Joint-Source Breakage Rate")
axes[1].set_title("Consensus Breakage")

# Neither-source rate
vals = [r["metrics"]["neither_source_rate"] for r in ordered]
axes[2].barh(labels, vals, color=colors)
axes[2].set_xlim(0, max(vals) * 1.2 + 0.01)
axes[2].set_xlabel("Neither-Source Behavior Rate")
axes[2].set_title("Neither-Source Predictions")

plt.tight_layout()
fig.savefig(FIGURE_DIR / "example_semantics_preservation_breakage.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: example_semantics_preservation_breakage.png")

# ─────────────────────────────────────────────
# Figure 2: Failure taxonomy composition
# ─────────────────────────────────────────────
taxonomy_cats = [
    "preserved_consensus", "better_source_preserved",
    "consensus_breakage", "better_source_loss",
    "shared_failure", "merge_recovery", "source_a_loss", "other"
]
cat_colors = {
    "preserved_consensus": "#27ae60",
    "better_source_preserved": "#2ecc71",
    "consensus_breakage": "#e74c3c",
    "better_source_loss": "#e67e22",
    "shared_failure": "#95a5a6",
    "merge_recovery": "#3498db",
    "source_a_loss": "#c0392b",
    "other": "#bdc3c7",
}

fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(ordered))
bottom = np.zeros(len(ordered))

for cat in taxonomy_cats:
    vals = []
    for r in ordered:
        count = r["category_counts"].get(cat, 0)
        vals.append(count / r["num_examples"] * 100)
    vals = np.array(vals)
    if vals.sum() > 0:
        ax.bar(x, vals, bottom=bottom, label=cat.replace("_", " "),
               color=cat_colors.get(cat, "#bdc3c7"))
        bottom += vals

ax.set_xticks(x)
ax.set_xticklabels([r["case_id"] for r in ordered], rotation=45, ha="right")
ax.set_ylabel("Percentage of examples")
ax.set_title("Failure Taxonomy Composition by Case")
ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
plt.tight_layout()
fig.savefig(FIGURE_DIR / "example_semantics_taxonomy_composition.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: example_semantics_taxonomy_composition.png")

# ─────────────────────────────────────────────
# Figure 3: Confidence / ambiguity comparison
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 5))

# Mean merged confidence
vals = [r["metrics"]["confidence_merged_mean"] for r in ordered]
axes[0].barh(labels, vals, color=colors)
axes[0].set_xlim(0, 1.05)
axes[0].set_xlabel("Mean Merged Confidence")
axes[0].set_title("Merged Model Confidence")

# Confidence collapse count
vals = [r["metrics"]["confidence_collapse_count"] for r in ordered]
axes[1].barh(labels, vals, color=colors)
axes[1].set_xlabel("Confidence Collapse Count")
axes[1].set_title("Confidence Collapse\n(merged < 0.4, source A > 0.6)")

# High-confidence wrong count
vals = [r["metrics"]["high_confidence_wrong_count"] for r in ordered]
axes[2].barh(labels, vals, color=colors)
axes[2].set_xlabel("High-Confidence Wrong Count")
axes[2].set_title("High-Confidence Errors\n(wrong, confidence > 0.8)")

plt.tight_layout()
fig.savefig(FIGURE_DIR / "example_semantics_confidence.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: example_semantics_confidence.png")

print("All figures generated.")
