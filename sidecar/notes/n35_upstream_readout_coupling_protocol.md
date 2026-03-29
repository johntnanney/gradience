# n35 — Upstream-Readout Coupling: Protocol

**Type:** protocol
**Program:** Seed-Contingent Readout-Axis Selection
**Stage:** B
**Date:** 2026-03-27
**Depends on:** n34 (panel definition)

---

## Central question

> When seeds of the same task differ in readout geometry, does
> upstream representation geometry vary in parallel, or does readout
> move independently?

---

## 1. Panel

14 same-task seed pairs (8 core + 3 domain-shift + 3 source-strength)
plus 3 adjacent-task pairs from the domain-shift family. See n34 for
the full panel definition.

## 2. Metrics

### Readout metrics (from classifier heads)
1. **Decision-axis cosine:** cos(d_a, d_b) where d = W[0] − W[1].
2. **Per-class weight cosine:** class_0 and class_1 separately.
3. **Pre-classifier subspace overlap:** top-4 SVD subspace overlap.
4. **Frobenius relative difference:** pre-classifier weight difference.

### Upstream V-module metrics (from LoRA V weights, all layers)
5. **Dim ratio mean:** mean min(ER_a, ER_b) / max(ER_a, ER_b) across layers.
6. **Top-direction overlap mean:** mean |u_a[0] · u_b[0]| across layers.
7. **Principal-angle mean cosine:** mean top-4 PA cosine across layers.

### Coupling metrics (derived)
8. **Readout classification:** aligned (cos > 0.95), moderate (0.5–0.95), weak (0.1–0.5), orthogonal (< 0.1).
9. **Decoupling test:** do pairs exist with similar upstream V-module geometry but different readout classification?

## 3. Required contrasts

### Contrast 1 — Within same-task families
Quantify readout and upstream variation for each task × backbone.

### Contrast 2 — Cross-family comparison
Test whether some tasks are more readout-contingent than others.

### Contrast 3 — Same-task vs adjacent-task (domain-shift)
Compare within-task (SST-2 s42 vs s7) with cross-domain (SST-2 vs Yelp).

### Contrast 4 — Source-strength comparison
Compare Strong (orthogonal) vs Medium/Weak (aligned) QNLI readout.

## 4. Interpretation rules

**Positive (coupled):** Families with orthogonal readout also show
divergent upstream geometry; aligned readout tracks similar upstream.

**Mixed (partially coupled):** Some families show coupling, others
show decoupling (same upstream, different readout).

**Negative (decoupled):** Readout varies independently of upstream —
the classifier head is not tracking representation-space structure.

## 5. Script

`sidecar/scripts/per_layer/seed_readout_coupling.py` — combined
Stages A+B analysis.
