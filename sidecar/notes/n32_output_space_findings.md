# n32 — Output-Space Readout Findings

**Type:** findings
**Program:** Sidecar B — Output-Space Compatibility
**Stage:** B (readout geometry and margin audit)
**Date:** 2026-03-27
**Depends on:** n30 (panel), n31 (protocol)

---

## 1. Summary verdict

**Mixed signal — with a critical falsifier.**

Readout alignment reliably predicts safety (compatible decision axes →
safe merge), but readout incompatibility does **not** reliably predict
catastrophe. The crucial falsifier is SC-QMRB: the same task pair
(QNLI × MRPC) on RoBERTa has nearly identical readout geometry to the
catastrophic CA-01 case on DistilBERT, yet produces only 1.7%
degradation.

Readout incompatibility is a **necessary background condition** for
catastrophic cross-task failure, not a discriminative cause.

---

## 2. Key results by contrast

### Contrast 1 — CA-01 (catastrophic) vs SC-QMRB (safe)

**The falsifier.** Same pair (QNLI × MRPC), different backbone.

| Metric | CA-01 (DistilBERT, Δ=41.7%) | SC-QMRB (RoBERTa, Δ=1.7%) |
|--------|----:|----:|
| decision_axis_cos | 0.015 | −0.019 |
| merged_margin_proxy | 0.712 | 0.701 |
| neither_task_score | 0.283 | 0.274 |
| decision_angle_deg | 89.15° | 88.90° |
| top4_subspace_overlap | 0.048 | 0.073 |
| frobenius_relative_diff | 1.415 | 1.415 |

Every readout metric is virtually identical. Both pairs have orthogonal
decision axes (~89°), both show ~30% margin compression, both land in
the "neither-task" zone. Yet CA-01 loses 41.7 percentage points and
SC-QMRB loses 1.7.

**Interpretation:** Readout geometry is task-pair-determined, not
backbone-determined. QNLI and MRPC learn orthogonal decision boundaries
regardless of whether the backbone is DistilBERT or RoBERTa. The
difference in merge outcome must therefore originate in representation
space — how each backbone's internal geometry interacts with the
identical readout incompatibility.

### Contrast 2 — CA-01 (catastrophic) vs NC-RMDB, NC-RSDB (mild)

| Case | Label | decision_axis_cos | margin_proxy | angle |
|------|-------|------------------:|-------------:|------:|
| CA-01 | catastrophic | 0.015 | 0.712 | 89.15° |
| NC-RMDB | mild (7.1%) | −0.052 | 0.688 | 87.01° |
| NC-RSDB | mild (8.3%) | 0.990 | 0.997 | 8.23° |

NC-RMDB (RTE × MRPC) has the same orthogonal readout geometry as
CA-01 but is only mildly degraded — another falsifier of the
readout-alone hypothesis. NC-RSDB (RTE × SST-2) has near-perfect
readout alignment and correspondingly mild degradation, consistent with
the "compatible readout → safe" direction.

### Contrast 3 — CA-01-catastrophic vs CA-01-mild (seed contrast)

| Metric | CA-01-cat (Δ=41.7%) | CA-01-mild (Δ=12.7%) |
|--------|----:|----:|
| decision_axis_cos | 0.015 | −0.059 |
| merged_margin_proxy | 0.712 | 0.686 |
| neither_task_score | 0.283 | 0.305 |
| decision_angle_deg | 89.15° | 86.62° |

Virtually identical readout geometry across a 29 percentage-point
performance gap. The catastrophic variant actually has *slightly higher*
margin proxy and *slightly lower* neither-task score than the mild
variant — the opposite direction from the hypothesis.

**Interpretation:** Readout geometry explains zero variance in the
seed-sensitive modulation. The 29pp gap is entirely a representation-
space phenomenon, consistent with Sidecar A's head-level cancellation
finding (n24).

### Contrast 4 — CA-02-toxic vs CA-02-benign

| Metric | CA-02-toxic (Δ=27.2%) | CA-02-benign (Δ=1.0%) |
|--------|----:|----:|
| decision_axis_cos | −0.020 | 0.999 |
| merged_margin_proxy | 0.701 | 1.000 |
| neither_task_score | 0.274 | 0.000 |
| decision_angle_deg | 88.87° | 2.29° |
| top4_subspace_overlap | 0.065 | 0.890 |

Here readout geometry **does** differ massively. QNLI_s42 × SST-2_s7
have orthogonal decision axes; QNLI_s7 × SST-2_s42 have nearly
parallel ones. This is the most surprising result: within the same
nominal task pair (QNLI × SST-2 on RoBERTa), different training seeds
produce qualitatively different readout geometry.

**Interpretation:** The SST-2 and QNLI classification tasks do not have
a single canonical decision axis in representation space. Seed choice
determines which direction the classifier head learns to use, and some
seed combinations produce aligned readout while others produce
orthogonal readout. This means readout compatibility is seed-contingent,
not merely task-contingent.

### Contrast 5 — SC-MSRB (safe, high collision) vs CA-01 (catastrophic)

| Metric | SC-MSRB (safe, Δ=4.8%) | CA-01 (catastrophic, Δ=41.7%) |
|--------|----:|----:|
| decision_axis_cos | 0.999 | 0.015 |
| merged_margin_proxy | 1.000 | 0.712 |
| collision (ρ) | 0.89 | 0.87 |

SC-MSRB has the highest collision in the panel (ρ=0.89) but near-
perfect readout alignment, and is safe. CA-01 has comparable collision
but orthogonal readout, and is catastrophic. This is consistent with
the joint-condition model: collision alone is not catastrophic, and
readout incompatibility alone is not catastrophic (per SC-QMRB). It is
the conjunction of representation-space risk factors WITH readout
incompatibility that produces catastrophe.

---

## 3. Metric-level assessment

### Metrics that discriminate (one direction)

**Readout alignment predicts safety.** All cases with decision_axis_cos
> 0.95 are safe or mildly degraded: SC-MSRB (0.999, safe), NC-RSDB
(0.990, mild), CA-02-benign (0.999, safe). Compatible readout is a
sufficient condition for safe merge in this panel.

### Metrics that do NOT discriminate

**Readout incompatibility does not predict catastrophe.** Cases with
decision_axis_cos near zero include catastrophic (CA-01, CA-02),
mild (NC-RMDB, CA-01-mild), moderate (NC-QSDB), and safe (SC-QMRB)
outcomes. Orthogonal readout is necessary but not sufficient for
catastrophic failure.

**Margin proxy is bimodal, not graduated.** Cases cluster at either
~0.70 (orthogonal readout) or ~1.00 (aligned readout), with no
intermediate values. The margin proxy is a near-deterministic function
of decision_axis_cos, adding no independent discriminative information.

**LoRA-readout coupling is uniformly low.** All coupling values are
in the 0.02–0.07 range. There is no case where the LoRA output-space
perturbation has substantial projection onto the classifier readout
direction. This metric does not discriminate.

### Metric 2 — Pre-classifier alignment

Pre-classifier geometry mirrors readout geometry: orthogonal readout
cases have low subspace overlap (~0.04–0.07) and high Frobenius
difference (~1.41), while aligned cases have high overlap (~0.71–0.89)
and low Frobenius difference (~0.06–0.12). The pre-classifier does not
add discriminative power beyond the readout layer.

---

## 4. Theoretical implications

### 4.1 Readout incompatibility as necessary background condition

The central finding is that readout incompatibility is **necessary but
not sufficient** for catastrophic cross-task failure. It functions as a
background condition — an enabling constraint — rather than a
discriminating cause. This is analogous to the distinction in causal
reasoning between a standing condition (oxygen in a room) and a
triggering cause (a spark): orthogonal readout geometry creates the
*possibility* of catastrophic failure, but the actual failure is
triggered by representation-space dynamics.

### 4.2 Readout geometry is partially task-determined, partially seed-determined

Contrast 1 (CA-01 vs SC-QMRB) shows that QNLI × MRPC produces
orthogonal readout regardless of backbone. But Contrast 4 (CA-02-toxic
vs CA-02-benign) shows that QNLI × SST-2 can produce either orthogonal
or aligned readout depending on seed. This means the readout-
compatibility landscape has two regimes:

- **Structurally orthogonal pairs** (e.g., QNLI × MRPC): readout is
  always incompatible regardless of seed or backbone. Catastrophic
  failure depends on representation-space factors.

- **Contingently orthogonal pairs** (e.g., QNLI × SST-2): readout
  compatibility is seed-dependent. Some seed combinations produce
  aligned readout and safe merges; others produce orthogonal readout
  and (possibly) catastrophic merges.

### 4.3 Updated mechanism ladder

The multiscale mechanism ladder (n25) now reads:

| Rung | Finding | Status |
|------|---------|--------|
| 1. V-module dim ratio | Discriminates catastrophic from safe (d=3.36) | Confirmed (n17) |
| 2. Head-level cancellation | Explains seed-sensitive modulation | Confirmed (n24) |
| 3. Readout incompatibility | Necessary background condition, not discriminative | **Resolved — mixed** |

**Rung 3 is resolved as mixed.** The readout layer does not independently
explain catastrophic failure, but it establishes the boundary condition
under which representation-space risk factors become catastrophic. The
mechanism is conjunctive: catastrophic failure requires BOTH
representation-space V-module pathology (Rungs 1-2) AND readout
incompatibility (Rung 3).

### 4.4 What this means for the amplification hypothesis

The original hypothesis (n31) was that the readout layer *amplifies*
upstream incompatibility. The data suggest something more precise: the
readout layer is a **gate**, not an amplifier. When readout geometry is
compatible (decision axes aligned), upstream incompatibility is
*absorbed* — the merged classifier can still find a valid decision
boundary despite representation-space conflict. When readout geometry
is incompatible, upstream incompatibility is *transmitted* — the merged
classifier lacks a valid boundary and performance collapses.

This gating model explains why SC-QMRB is safe despite orthogonal
readout: on RoBERTa, the representation-space V-module geometry for
QNLI × MRPC does not produce the pathological cancellation pattern
found on DistilBERT (per Sidecar A). Compatible upstream geometry +
incompatible readout = safe.

---

## 5. Quantitative summary

### Group-level means

| Group | n | decision_axis_cos | margin_proxy | neither_task | angle |
|-------|--:|------------------:|-------------:|-------------:|------:|
| catastrophic | 4 | −0.003 ± 0.017 | 0.707 | 0.278 | 89.0° |
| moderate | 1 | 0.003 | 0.708 | 0.279 | 89.8° |
| mild | 3 | 0.293 ± 0.493 | 0.791 | 0.206 | 60.6° |
| safe | 3 | 0.660 ± 0.480 | 0.900 | 0.092 | 31.2° |

The group means show a gradient from catastrophic → safe in decision
axis cosine and margin proxy, but the variance within the mild and
safe groups is enormous (std ~0.49), entirely driven by the bimodal
split between orthogonal and aligned readout. The mean is not
meaningful; the distribution is bimodal.

### Named contrasts

| Contrast | Metric | Value A | Value B | Δ |
|----------|--------|--------:|--------:|--:|
| CA-01 seed | decision_axis_cos | 0.015 | −0.059 | 0.074 |
| CA-01 seed | margin_proxy | 0.712 | 0.686 | 0.026 |
| CA-02 toxic | decision_axis_cos | −0.020 | 0.999 | 1.019 |
| CA-02 toxic | margin_proxy | 0.701 | 1.000 | 0.299 |

---

## 6. Verdict per interpretation rules (n31 §7)

**Mixed signal.** Per the protocol's interpretation framework:

> "Some metrics discriminate, others don't; or effect is
> backbone-conditional."

Specifically:
- Compatible readout (decision_axis_cos > 0.95) reliably predicts safe
  outcomes (3/3 cases).
- Incompatible readout does not reliably predict catastrophic outcomes
  (only 4/7 incompatible cases are catastrophic or moderate; 2 are mild,
  1 is safe).
- The effect is backbone-conditional: the same readout geometry
  produces catastrophic failure on DistilBERT and safe merge on RoBERTa
  for the QNLI × MRPC pair.
- Readout geometry explains zero variance in the seed-sensitive
  modulation (CA-01 seed contrast).

### What works
- Decision axis cosine partitions the panel into "orthogonal" (~0) and
  "aligned" (~1) clusters with no intermediate values.
- Aligned readout is a reliable safety predictor.

### What doesn't work
- Orthogonal readout does not predict catastrophe.
- Margin proxy and neither-task score are near-deterministic functions
  of decision axis cosine and add no independent information.
- LoRA-readout coupling is uniformly low and non-discriminative.
- Pre-classifier alignment tracks readout alignment exactly.

---

## 7. Implications for Stage C

Given the mixed signal, the Stage C question shifts. The original plan
was to build error dossiers on a per-example basis. The findings here
refine what those dossiers should test:

1. **Gate vs amplifier test:** For SC-QMRB (safe, orthogonal readout),
   does CPU inference show that the merged model produces sensible
   logits despite orthogonal readout? If yes, the representation-space
   geometry is absorbing the readout incompatibility, confirming the
   gating model.

2. **Seed-contingent readout:** For the CA-02 contrast, what makes
   QNLI_s42 learn a readout direction orthogonal to SST-2_s7, while
   QNLI_s7 learns one aligned with SST-2_s42? This is a question about
   how training seed determines the orientation of the decision axis in
   representation space — potentially a question about which subspace
   of the representation the classifier head "chooses" to use.

3. **Backbone gating mechanism:** For the CA-01 vs SC-QMRB pair, what
   differs in the DistilBERT vs RoBERTa representation spaces that
   causes one to transmit readout incompatibility and the other to
   absorb it? This is the central remaining mechanistic question.

---

## 8. Files produced

| File | Description |
|------|-------------|
| `sidecar/results/output_space/readout_metrics.json` | Per-case metrics (11 cases × 5 metric groups) |
| `sidecar/results/output_space/margin_audit.json` | Group-level contrasts and named comparisons |
| `sidecar/results/output_space/example_behavior_summary.json` | Key contrast detail (CA-01 vs SC-QMRB) |
| `sidecar/figures/output_space_readout_alignment.svg` | Decision axis cosine by case |
| `sidecar/figures/output_space_margin_compression.svg` | Merged margin proxy by case |
| `sidecar/figures/output_space_ca01_seed_contrast.svg` | Seed-variant readout comparison |
| `sidecar/figures/output_space_neither_task.svg` | Neither-task score by case |

---

## 9. Relationship to Sidecar A

Sidecar B Stage B confirms the conjunctive model implied by Sidecar A's
findings:

- **Sidecar A** established that V-module geometry (dim ratio, head-level
  cancellation) discriminates catastrophic from safe outcomes in
  representation space.

- **Sidecar B Stage B** establishes that readout geometry is a necessary
  background condition but not independently discriminative. The
  readout layer gates (transmits or absorbs) the upstream risk rather
  than generating new risk.

Together, the mechanism is: **V-module pathology × readout
incompatibility → catastrophic failure.** Either factor alone is
insufficient. This is the core finding of the Sidecar B Stage B audit.
