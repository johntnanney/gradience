# n31 — Output-Space Readout Geometry and Margin Audit Protocol

**Type:** protocol
**Program:** Sidecar B — Output-Space Compatibility
**Stage:** B (CPU-only readout geometry and margin audit)
**Date:** 2026-03-27
**Depends on:** n30 (panel definition)

---

## 1. Central question

> Do catastrophic merges produce distinctive readout-side incompatibility
> that is not visible from upstream geometry alone?

---

## 2. Panel

Use the 11-case panel defined in n30. All cases have classifier head
weights available. DistilBERT cases (7/11) support full inference;
RoBERTa cases (4/11) support weight-level analysis only.

---

## 3. Metrics

### Metric 1 — Readout vector alignment

**What:** Cosine similarity between source A and source B final readout
vectors (the 2×768 classifier weight matrix rows).

**Computation:**
For each adapter pair, extract the 2×768 readout weight matrix.
Each row is a class-direction vector in representation space.

```
readout_cos = cosine_similarity(W_a_readout, W_b_readout)
```

Specifically, compute:
- `class_0_cos`: cosine of source A class-0 vector vs source B class-0 vector
- `class_1_cos`: cosine of source A class-1 vector vs source B class-1 vector
- `decision_axis_cos`: cosine of source A decision axis (class_1 - class_0)
  vs source B decision axis

**Hypothesis:** Catastrophic anchors show lower `decision_axis_cos` than
safe controls, meaning their task-specific decision boundaries point in
more incompatible directions.

### Metric 2 — Pre-classifier projection alignment

**What:** Cosine similarity and spectral overlap of the 768×768
pre-classifier (DistilBERT) / classifier.dense (RoBERTa) projection
matrices.

**Computation:**
- SVD of each pre-classifier matrix
- Top-k subspace overlap between source A and source B
- Frobenius norm of the difference

**Hypothesis:** Catastrophic pairs may show pre-classifier divergence
that amplifies readout incompatibility.

### Metric 3 — Merged readout geometry

**What:** Properties of the linearly merged readout weights.

**Computation:**
```
W_merged = 0.5 * W_a + 0.5 * W_b
```

Then measure:
- `merged_decision_norm`: L2 norm of merged decision axis (class_1 - class_0)
- `merged_margin_proxy`: ratio of merged decision axis norm to average
  source decision axis norm
- `merged_alignment_a`: cosine of merged decision axis vs source A
- `merged_alignment_b`: cosine of merged decision axis vs source B
- `neither_task_score`: 1 - max(|merged_alignment_a|, |merged_alignment_b|)

**Hypothesis:** Catastrophic anchors show:
- lower `merged_margin_proxy` (decision axis shrinks when incompatible)
- higher `neither_task_score` (merged axis aligned with neither source)

### Metric 4 — Decision boundary angular distance

**What:** Angle between source A and source B decision hyperplanes.

**Computation:**
The decision hyperplane normal is `w_1 - w_0` (class 1 row minus class 0 row
of the classifier weight matrix). Compute the angle between source A's
hyperplane normal and source B's hyperplane normal.

**Hypothesis:** Catastrophic anchors have larger angular distance between
decision hyperplanes than safe controls.

### Metric 5 — Readout sensitivity to LoRA output space

**What:** How much of the LoRA output-module perturbation falls along vs
orthogonal to the classifier readout direction.

**Computation:**
For each adapter, compute the LoRA product W = B × A for the final-layer
output module. Project this onto the classifier readout direction
(decision axis). The fraction of the LoRA perturbation aligned with
readout tells us how much the LoRA modification directly affects the
classification decision.

**Hypothesis:** In catastrophic pairs, the two LoRA output modules push
the representation in directions that have high but opposite projections
onto the decision axis — creating constructive interference toward
"neither-task" output.

---

## 4. Contrasts

### Contrast 1 — CA-01 (catastrophic) vs SC-QMRB (safe)
Same pair (QNLI × MRPC), different backbone.
Tests: does readout geometry differ when the same pair produces
catastrophic vs safe outcomes?

### Contrast 2 — CA-01 (catastrophic) vs NC-RMDB, NC-RSDB (mild)
Different pairs on same backbone (DistilBERT).
Tests: does readout incompatibility separate catastrophic from mild?

### Contrast 3 — CA-01-catastrophic vs CA-01-mild
Same pair, same backbone, same MRPC adapter. Only QNLI seed differs.
Tests: does readout geometry explain the 29pp seed gap?

### Contrast 4 — CA-02-toxic vs CA-02-benign
Same pair (QNLI × SST-2) on RoBERTa. QNLI seed differs.
Tests: is the toxic adapter's readout signature distinctive?

### Contrast 5 — SC-MSRB (safe, high collision) vs CA-01 (catastrophic)
Tests: does compatible readout + high collision → safe, while
incompatible readout + collision → catastrophic?

---

## 5. Output files

### Required
- `sidecar/results/output_space/readout_metrics.json` —
  per-case metric values for all 5 metrics
- `sidecar/results/output_space/margin_audit.json` —
  merged readout geometry details per case
- `sidecar/results/output_space/example_behavior_summary.json` —
  contrast-level comparisons and effect sizes

### Figures
- `sidecar/figures/output_space_readout_alignment.svg` —
  decision axis cosine by case group
- `sidecar/figures/output_space_margin_compression.svg` —
  merged margin proxy by case
- `sidecar/figures/output_space_ca01_seed_contrast.svg` —
  seed variant readout comparison
- `sidecar/figures/output_space_neither_task.svg` —
  neither-task score by case group

---

## 6. Script

Analysis script: `sidecar/scripts/per_layer/output_space_readout.py`

The script should:
1. Load classifier head weights from all panel adapters
2. Compute all 5 metrics for each panel case
3. Compute contrasts and effect sizes
4. Write JSON outputs
5. Generate figures

---

## 7. Interpretation rules

### If readout alignment separates catastrophic from safe/mild
(decision_axis_cos for catastrophic anchors is notably lower
than for controls, and merged_margin_proxy is compressed):
→ **Positive signal.** Proceed to Stage C error-dossier analysis.
Readout incompatibility is a real explanatory rung.

### If signal is mixed
(some metrics discriminate, others don't; or effect is
backbone-conditional):
→ **Mixed signal.** Document which metrics work and which don't.
Consider whether DistilBERT vs RoBERTa architecture differences
explain the heterogeneity.

### If no readout metric separates groups
→ **Negative signal.** The downstream amplification hypothesis
is not supported by readout geometry alone. Record as narrowing
evidence.

---

## 8. Relationship to Sidecar A

This protocol tests the open Rung 3 in the multiscale mechanism
ladder (n25). Sidecar A established:
- Rung 1: V-module dim ratio discriminates catastrophic from safe (d=3.36)
- Rung 2: Head-level V geometry explains seed-sensitive modulation

This protocol asks: does the readout layer explain **why** some
V-module configurations produce catastrophic outcomes while others
produce mild outcomes?

The key novelty is that we now know classifier heads are available
in the adapter files. This was not assumed during Sidecar A design.
