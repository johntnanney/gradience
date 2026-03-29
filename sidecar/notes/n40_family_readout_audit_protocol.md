# n40 — Family-Level Readout Audit Protocol

**Type:** protocol
**Date:** 2026-03-28
**Depends on:** n39 (attractor panel definition), n36 (coupling findings)
**Status:** Stage B protocol for the attractor mapping lab.

---

## Objective

Map the readout-axis geometry of each family in the attractor panel
(n39) and determine whether it looks single-attractor or multi-attractor.
The central question for each family is:

> Does this family show one dominant readout basin or multiple valid basins?

---

## Input data

All readout and upstream metrics are drawn from the seed-contingent
readout program (n36). No new weight extraction is required. The
analysis script consumes:

- `sidecar/results/seed_readout/seed_panel_table.json` — 17 pairs × full metrics
- `sidecar/results/seed_readout/coupling_metrics.json` — readout classifications, cross-task linkage
- `sidecar/results/attractor_mapping/attractor_panel_table.json` — family inventory from Stage A

---

## Metrics

### Metric 1 — Pairwise readout-axis cosine

For each seed pair in a family: the decision_axis_cos from n36.

**Classification rule (deterministic):**
- |cos| < 0.1 → **orthogonal** (multi-attractor signal)
- |cos| > 0.95 → **aligned** (single-attractor signal)
- 0.1 ≤ |cos| ≤ 0.95 → **intermediate** (ambiguous — not observed in current data)

This rule matches the bimodal distribution established in n36: all 17
pairs fall cleanly into orthogonal or aligned with no intermediate values.

### Metric 2 — Distribution shape classification

For each family (aggregated across backbones where applicable):

- **Tightly aligned:** All seed pairs show |cos| > 0.95.
- **Consistently orthogonal:** All seed pairs show |cos| < 0.1.
- **Backbone-split:** Aligned on one backbone, orthogonal on another.
- **Convergence-split:** Aligned at some training depths, orthogonal at others.
- **Diffuse:** Intermediate values present, no clear basin structure.

**Note:** With only 2 seeds per family×backbone, "distribution" is a
single data point. The classification reflects the pattern across
conditions (backbones, convergence bands), not within-condition
variability. This is a known limitation.

### Metric 3 — Prediction overlap

**Not computable.** No stored predictions or logits are available for
any family in the panel. This metric is deferred to a future stage
that would require GPU inference.

### Metric 4 — Merge safety context

For each family: the merge_delta from n36. This contextualizes whether
multi-attractor families remain safe under merge.

**Classification rule:**
- Δ ≤ 3.4% → **safe** (within same-task safety envelope from P01)
- 3.4% < Δ ≤ 15% → **moderate**
- Δ > 15% → **catastrophic**

---

## Family-level attractor classification

Each family receives a final attractor classification based on the
combined evidence from Metrics 1, 2, and 4:

| Classification | Definition |
|---------------|------------|
| **single_attractor** | All seed pairs aligned, all backbones consistent |
| **multi_attractor** | All seed pairs orthogonal on all available backbones |
| **backbone_contingent** | Attractor class changes across backbones |
| **convergence_contingent** | Attractor class changes across training depth |
| **training_contingent** | Attractor class changes across training regime (domain shift) |

A family can receive multiple labels if it satisfies multiple conditions
(e.g., QNLI is both multi_attractor and convergence_contingent when
the strength contrast is considered).

---

## Contrasts

### Contrast 1 — Single vs multi-attractor families
Compare readout geometry, upstream V-module health, and merge safety
across the two attractor classes. Question: do multi-attractor families
differ in any upstream property from single-attractor families?

### Contrast 2 — Backbone contingency (MRPC)
Compare MRPC on DistilBERT (orthogonal) vs RoBERTa (aligned).
Question: what changes between backbones that could explain the
attractor-structure shift?

### Contrast 3 — Convergence contingency (QNLI strength)
Compare Strong QNLI (orthogonal) vs Medium/Weak QNLI (aligned).
Question: does the transition from single to multi-attractor
correlate with any upstream metric?

### Contrast 4 — Domain contrast (sentiment block)
Compare SST-2 (domain, orthogonal) vs Yelp (aligned) vs Amazon (aligned).
Question: why does SST-2 under domain shift become multi-attractor
while Yelp and Amazon remain single-attractor?

### Contrast 5 — Cross-domain readout alignment
Examine the three cross-domain pairs: SST-2×Yelp (orthogonal),
SST-2×Amazon (orthogonal), Yelp×Amazon (aligned).
Question: does cross-family readout alignment track within-family
attractor structure?

---

## Required figures

### Figure 1 — Family attractor map
Per-family bar or dot chart showing readout cosine by family, colored
by attractor classification. Grouped by panel group (A, B, C, D).

### Figure 2 — Backbone contrast panel
Side-by-side comparison of readout cosine for the 4 core tasks on
DistilBERT vs RoBERTa, highlighting MRPC's backbone contingency.

### Figure 3 — Convergence contrast
Strong / Medium / Weak QNLI readout cosine as a three-point series,
showing the transition from multi-attractor to single-attractor.

### Figure 4 — Domain contrast block
SST-2 (domain) / Yelp / Amazon readout cosines with cross-domain
pair cosines overlaid.

---

## Deliverables

| Deliverable | Path |
|------------|------|
| Protocol (this note) | `sidecar/notes/n40_family_readout_audit_protocol.md` |
| Analysis script | `sidecar/scripts/per_layer/attractor_mapping_audit.py` |
| Family readout metrics | `sidecar/results/attractor_mapping/family_readout_metrics.json` |
| Attractor classifications | `sidecar/results/attractor_mapping/attractor_classifications.json` |
| Figures 1–4 | `sidecar/figures/attractor_mapping_*.svg` |
| Findings note | `sidecar/notes/n41_family_readout_audit_findings.md` |

---

## Interpretation rules

1. **Multi-attractor does not imply fragile.** All multi-attractor
   same-task families in the current panel merge safely (Δ ≤ 2.2%).
   Multi-attractor is a property of the readout landscape, not a
   risk indicator.

2. **Single-attractor does not imply safe for cross-task merge.**
   A single-attractor task can still be catastrophic when merged
   cross-task if V-module pathology is present.

3. **Backbone contingency is informative, not diagnostic.** MRPC's
   backbone-contingent attractor structure tells us about the
   representation geometry of each backbone, not about MRPC's
   intrinsic risk.

4. **Convergence contingency suggests attractor selection is
   training-depth-dependent.** The Strong/Medium/Weak QNLI
   contrast shows that longer training opens access to a second
   attractor basin.
