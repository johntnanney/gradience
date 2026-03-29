# n34 — Seed-Contingent Readout-Axis Selection: Panel Definition

**Type:** panel definition
**Program:** Seed-Contingent Readout-Axis Selection
**Stage:** A
**Date:** 2026-03-27
**Depends on:** n32 (output-space findings), n33 (conjunctive synthesis)

---

## 1. Motivation

The conjunctive mechanism synthesis (n33) identified seed-contingent
readout-axis selection as the best remaining CPU-feasible question.
The CA-02 contrast showed that the same task pair (QNLI × SST-2 on
RoBERTa) produces either orthogonal or aligned readout depending on
which seeds are paired. This raises the question: how much does readout
geometry vary *within* same-task seed families, and does that variation
track upstream representation differences?

---

## 2. Available same-task seed families

### Group 1 — Core task families (4 tasks × 2 backbones × 2 seeds)

| Family | Backbone | Seeds | Source weights | Classifier heads | Merge Δ |
|--------|----------|-------|---------------|-----------------|---------|
| QNLI | DistilBERT | s42, s7 | yes | yes | 2.2% |
| MRPC | DistilBERT | s42, s7 | yes | yes | 0.25% |
| RTE | DistilBERT | s42, s7 | yes | yes | 0.36% |
| SST-2 | DistilBERT | s42, s7 | yes | yes | 0.40% |
| QNLI | RoBERTa | s42, s7 | yes | yes | 0.0% |
| MRPC | RoBERTa | s42, s7 | yes | yes | 0.98% |
| RTE | RoBERTa | s42, s7 | yes | yes | 0.0% |
| SST-2 | RoBERTa | s42, s7 | yes | yes | 0.0% |

All 8 families have 2 seeds, full LoRA weights, and classifier heads
in safetensors. Same-task merge degradation is minimal (0–2.2%).

**Analyzable level:** upstream_readout_plus_merge for all.

### Group 2 — Domain-shift sentiment family (DistilBERT)

| Family | Seeds | Source weights | Classifier heads | Same-task merge Δ |
|--------|-------|---------------|-----------------|-------------------|
| SST-2 | s42, s7 | yes | yes | 1.0% |
| Yelp | s42, s7 | yes | yes | 0.2% |
| Amazon | s42, s7 | yes | yes | 0.6% |

Three sentiment-classification variants on DistilBERT. These share
the same binary classification structure (positive/negative) but differ
in domain. Cross-domain merge Δ ranges 0.0–2.2%, all safe.

**Analyzable level:** upstream_readout_plus_merge.
**Special value:** These are "adjacent tasks" — same classification
target, different data distribution. If readout axes differ between
SST-2 and Yelp, that's domain-driven readout divergence; if they
align, sentiment classification has a single natural readout direction
regardless of domain.

### Group 3 — Source-strength QNLI family (DistilBERT)

| Family | Seeds | Source weights | Classifier heads | Same-task merge Δ |
|--------|-------|---------------|-----------------|-------------------|
| Strong | s42, s7 | yes | yes | 0.4% |
| Medium | s42, s7 | yes | yes | 1.4% |
| Weak | s42, s7 | yes | yes | 0.2% |

Three quality bands of QNLI on DistilBERT (same task, different
training duration/convergence). Cross-band merge Δ ranges 0.0–2.4%.

**Analyzable level:** upstream_readout_plus_merge.
**Special value:** Readout variation here would reflect
convergence-dependent axis selection: do strong and weak adapters
land on the same decision axis, or does training quality affect
which features the classifier exploits?

### Group 4 — Known fragile cross-task linkage

The core task families already contain the adapters involved in
catastrophic cross-task failure:

| Pair | Backbone | Cross-task Δ range | Key adapters |
|------|----------|-------------------|--------------|
| QNLI × MRPC | DistilBERT | 12.7–41.7% | qnli_s42, qnli_s7, mrpc_s7 |
| QNLI × SST-2 | RoBERTa | 1.0–27.2% | qnli_s42, qnli_s7, sst2_s42, sst2_s7 |

These are not separate families — they use the same adapters as
Group 1. But the cross-task merge outcomes provide the fragility
variation that makes same-task readout geometry meaningful: if
qnli_s42 and qnli_s7 have different readout axes, that may explain
why one produces catastrophic cross-task merges and the other does not.

---

## 3. Panel design

### Primary panel: 16 same-task seed pairs

Each core task family contributes one same-task seed pair
(s42 vs s7). Total: 8 families × 1 pair = 8 primary pairs.

Each domain-shift family contributes one pair. Total: 3 pairs.

Each source-strength family contributes one pair. Total: 3 pairs.

**Plus cross-family pairs** within domain-shift (6 pairs: SST-2/Yelp,
SST-2/Amazon, Yelp/Amazon × 2 seed combos) and source-strength (6
pairs: strong/medium, strong/weak, medium/weak × 2 seed combos).

### Metrics to compute per pair

**Readout metrics** (from classifier heads):
1. Decision-axis cosine (same as Sidecar B Metric 1)
2. Pre-classifier subspace overlap (same as Sidecar B Metric 2)
3. Per-class weight cosine (class_0 and class_1 separately)

**Upstream metrics** (from LoRA V-module weights):
4. V-module dimensionality ratio (same as Sidecar A, n21)
5. V-module top-direction overlap
6. V-module principal angle mean cosine

**Coupling metrics** (derived):
7. Readout-upstream rank correlation (does within-family readout
   variation track upstream variation?)
8. Coupling classification: aligned/mismatched (do pairs that are
   similar upstream also have similar readout, or can they diverge?)

---

## 4. Key questions this panel can answer

### RQ1 — Within-family readout variation
How much do decision axes vary between s42 and s7 for the same task
on the same backbone? The Sidecar B data already showed that
cross-task readout can be orthogonal (~0) or aligned (~1). If
same-task readout is always aligned (~1), readout selection is
task-determined. If it varies, readout selection is genuinely
seed-contingent even within a task.

### RQ2 — Upstream-readout coupling
When seeds differ in readout geometry, do they also differ in V-module
geometry? If readout and upstream vary in parallel, the coupling is
strong (readout follows upstream). If readout varies while upstream is
stable (or vice versa), the coupling is weak and readout axis
selection is partially independent.

### RQ3 — Task-family differences
Do some tasks show more seed-contingent readout than others? The
Sidecar B data suggests QNLI × SST-2 is variable (seed-contingent)
while QNLI × MRPC is always orthogonal (structurally determined).
Same-task data can test whether QNLI or MRPC or SST-2 themselves
vary more in readout direction.

### RQ4 — Domain vs seed readout variation
In the sentiment family, is cross-domain readout variation (SST-2 vs
Yelp) larger or smaller than cross-seed variation (SST-2_s42 vs
SST-2_s7)? If domain matters more, readout selection tracks data
distribution. If seed matters more, readout selection is training-
contingent even for functionally identical tasks.

---

## 5. Success criteria

Stage A is successful if:
- [x] At least one same-task family has upstream + readout artifacts
  for comparison *(all 14 families qualify)*
- [x] At least one family includes interpretable merge variation
  *(Group 4 provides cross-task fragility linkage)*
- [x] A small but useful panel is defined *(14 families, 14+12
  pairs, 8 metrics)*

---

## 6. Files to produce

| File | Description |
|------|-------------|
| `sidecar/results/seed_readout/seed_panel_table.json` | Machine-readable panel |
| `sidecar/results/seed_readout/seed_panel_table.md` | Human-readable summary |
| This note (n34) | Panel definition and rationale |
