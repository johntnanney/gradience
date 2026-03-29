# n38 — What Is Now Ruled Out

**Type:** negative-results summary
**Date:** 2026-03-27
**Status:** Current as of the completion of the seed-contingent readout program (n36–n37).

---

## Purpose

This note collects the sidecar's negative results in one place.
Each entry names a hypothesis that was tested and rejected, states
the evidence that rejected it, and says what replaced it. The
entries are ordered roughly by when they were ruled out.

Negative results are not failures. Each one constrains the space
of viable explanations and prevented a line of work that would
have produced misleading conclusions.

---

## 1. Severity as portable descriptor

**Hypothesis:** The magnitude of cross-task merge degradation is a
stable property of the task pair — some pairs are inherently more
dangerous than others.

**Ruled out by:** Backbone comparison (S01). QNLI × MRPC degrades
41.7% on DistilBERT and 1.7% on RoBERTa. Severity rankings
reverse across backbones. Six candidate severity signals (task-pair
identity, core-space shared basis, pair-risk label, format
similarity, source-strength gap, reconstruction error) all failed
to predict severity portably.

**Replaced by:** Instability — the variability of severity across
conditions — which does rank consistently across backbones.

---

## 2. Task-pair identity as catastrophe lookup

**Hypothesis:** Certain task pairs are catastrophic and others are
not, as a fixed property of the pair.

**Ruled out by:** The backbone-reversal pattern (S01, n05).
QNLI × MRPC is catastrophic on DistilBERT but mild on RoBERTa.
QNLI × SST-2 is catastrophic on RoBERTa but moderate on
DistilBERT. No pair is catastrophic on both backbones. The unit
of catastrophe is the (task pair × backbone × seed) triple, not
the pair.

**Replaced by:** The conjunctive model (n33, n37) — catastrophe
requires both V-module pathology and readout incompatibility, each
of which is backbone- and seed-dependent.

---

## 3. Aggregate within-layer geometry as threshold variable

**Hypothesis:** Catastrophic pairs would show distinctive subspace
geometry when adapter weight matrices are compared layer by layer,
using the concatenated Q/K/V/O product.

**Ruled out by:** Within-layer geometry pilot (n18). When backbone
is controlled, catastrophic cases (CA-02) are indistinguishable
from safe collision controls on all four metrics (principal angles,
top-direction overlap, dimensionality ratio, directional conflict).
The separation visible in raw data was a backbone confound:
DistilBERT's 6-layer compression forces tighter alignment than
RoBERTa's 12 layers.

**Replaced by:** Per-module decomposition (n21). The concatenation
was diluting the signal. Separating Q/K/V/O recovers a clean
V-module dim-ratio signal (d = 3.36) that the aggregate analysis
had averaged away.

---

## 4. Collision as sufficient condition for catastrophe

**Hypothesis:** High per-layer alignment (collision) between two
adapters' norm-mass profiles is sufficient to trigger catastrophic
interference.

**Ruled out by:** Collision subset analysis (n16). MRPC × SST-2
on RoBERTa has the highest cross-task alignment (ρ = 0.89) but is
stable (instability = 0.21). Multiple high-alignment pairs are
non-catastrophic.

**Replaced by:** Collision as a necessary precondition that gates
entry to the mechanism ladder, not a sufficient trigger. The
trigger is V-module dimensionality mismatch within the collision
regime.

---

## 5. Readout orthogonality as risk marker

**Hypothesis:** Near-orthogonal decision axes (cos ≈ 0) between
two adapters' classifier heads indicates dangerous readout-level
incompatibility.

**Ruled out by:** Two independent lines of evidence.
First, the SC-QMRB falsifier (n32): identical readout geometry to
catastrophic CA-01 but safe (Δ = 1.7%).
Second, same-task seed analysis (n36): 5 of 14 same-task seed
pairs show orthogonal readout yet all merge safely (Δ ≤ 2.2%).
A stand-alone readout-cosine metric would false-alarm on roughly
40% of same-task merges.

**Replaced by:** Readout incompatibility as a gate condition in
the conjunctive model. Orthogonal readout is common and harmless;
it becomes dangerous only when combined with upstream V-module
pathology.

---

## 6. Readout-upstream coupling as mechanism

**Hypothesis:** When two adapters have incompatible readout, this
reflects incompatible upstream representations — the readout
divergence is a downstream symptom of a representation-space
problem.

**Ruled out by:** Seed-contingent readout analysis (n36). All
same-task seed pairs have healthy V-module geometry (dim ratio
> 0.78) regardless of whether their readout is aligned or
orthogonal. The two conditions are independently determined by
different mechanisms.

**Replaced by:** The attractor model (n37). Readout axis selection
depends on which attractor the classifier head converges to during
training — a function of task identity, backbone architecture,
and training convergence — not on upstream LoRA geometry.

---

## 7. Readout geometry as seed-sensitivity explanation

**Hypothesis:** The large seed-sensitivity effects (CA-01's 29pp
gap, CA-02's 26pp gap) might reflect seed-dependent differences
in readout geometry.

**Ruled out by:** Output-space readout audit (n32). CA-01's
catastrophic and mild seed variants have virtually identical
readout geometry (decision_axis_cos: 0.015 vs −0.059). Readout
explains zero variance in the seed-sensitive modulation.

**Replaced by:** Head-level V-module cancellation (n24). The 29pp
gap localizes to 7 attention heads with |Δ_DR| ≥ 0.15, where
opposite-sign deltas cancel at module level.

---

## 8. Downstream amplification as Rung 3 mechanism

**Hypothesis:** The O module and classification head jointly
amplify upstream incompatibility — orthogonal decision axes
magnify small representation-space distortions into large
classification errors.

**Ruled out by:** The SC-QMRB falsifier (n32). If readout were
an amplifier, SC-QMRB should have amplified whatever small
upstream distortion exists on RoBERTa for the QNLI × MRPC pair.
It did not (Δ = 1.7%).

**Replaced by:** Readout gating (n32, n37). The readout layer
transmits or absorbs upstream pathology rather than amplifying it.
Compatible readout absorbs; incompatible readout transmits.

---

## What survives

After eight ruled-out hypotheses, the surviving explanatory
framework is:

- **Boundary detection** (core, confirmed) — same-task vs
  cross-task.
- **Instability** (sidecar, promising) — portable variability
  descriptor. Awaiting DeBERTa.
- **V-module dimensionality ratio** (sidecar, strongest signal) —
  d = 3.36, zero overlap. Awaiting DeBERTa.
- **Head-level cancellation** (sidecar, resolved) — explains
  seed sensitivity within a risk class.
- **Readout gating** (sidecar, resolved) — the gate condition
  in the conjunctive model. Common, harmless alone,
  independently determined.
- **Conjunctive model** (sidecar, current best) — V-module
  pathology × readout incompatibility → catastrophe.

Everything else has been tested and set aside.
