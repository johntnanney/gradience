# n45 — The Conjunctive Model with Mechanisms

**Type:** synthesis
**Date:** 2026-03-28
**Depends on:** n37 (conjunctive model update), n42 (readout topology), n44 (decision-axis findings)
**Status:** Current best statement of the complete sidecar mechanism model. Supersedes n37 §4 and n33 Rung 3.

---

## The model as it stood before n44

The conjunctive model (n37) states that catastrophic merge failure
requires the conjunction of two independently determined conditions:

1. **Upstream V-module pathology** — the two adapters' V-module
   representations are incompatible (dimensionality ratio below
   threshold, d=3.36 separation).
2. **Readout incompatibility** — the two adapters' classifier heads
   use different decision axes (readout gate is open).

Either condition alone is insufficient. Compatible readout absorbs
upstream pathology (gate closed → safe). Incompatible readout with
healthy upstream geometry is benign (gate open, nothing to transmit →
safe). Only the conjunction is catastrophic.

The topology note (n42) added that the readout condition has internal
structure — single-attractor vs multi-attractor vs contingent families —
but did not specify the *mechanism* by which the gate opens.

---

## What n44 adds

The decision-axis analysis identifies two distinct mechanisms by which
the readout gate opens. The gate is no longer a black box with a
binary state. It has a mechanism, and that mechanism is
backbone-dependent.

### Mechanism 1 — Rotational degeneracy

**Where it operates:** DistilBERT multi-attractor families (QNLI,
MRPC, SST-2 domain-shift, Strong QNLI).

**What happens:** The task's relevant features span a low-rank
subspace (effective dimension 1–2 in the pre-classifier's top PCs)
within which the decision axis is free to rotate without affecting
classification accuracy. The classifier head's random initialization
determines the orientation. Different seeds land at different angles
within the same subspace.

**Signature:** Orthogonal readout axes (cos ≈ 0), but high PC energy
overlap (> 0.93) and shared principal components. The seeds use the
same features but combine them differently.

**Gate state:** Open, but the two adapters' decision pathways pass
through the same low-dimensional subspace of the representation.
Their disagreement is rotational, not structural.

### Mechanism 2 — Feature-set switching

**Where it operates:** QNLI on RoBERTa (the only confirmed instance).

**What happens:** The pretrained model's representation space encodes
multiple independent feature sets sufficient for the task. Different
seeds lock onto different feature sets. The decision axes occupy
non-overlapping PC subspaces.

**Signature:** Orthogonal readout axes (cos ≈ 0), low PC energy
overlap (0.255), zero shared top-3 PCs. The seeds use genuinely
different features.

**Evidence for cross-task feature exploitation:** QNLI/rb/s7 has
effective axis cosine +0.86 with RTE/rb/s7. One QNLI seed uses a
decision direction that resembles an entirely different task's
direction. The pretrained representation's feature inventory is
shared across tasks.

**Gate state:** Open, and the two adapters' decision pathways pass
through different subspaces entirely. Their disagreement is
structural, not rotational.

---

## The updated model

The conjunctive model now has three levels of specification:

### Level 1 — The conjunction (unchanged)

Catastrophe = upstream V-module pathology **AND** readout
incompatibility.

This remains the top-level structure. The conjunction is the
sidecar's central empirical finding. Nothing in n44 changes it.

### Level 2 — The gate taxonomy (new from n41–n42)

The readout gate has three states determined by the task family's
attractor topology:

| Gate state | Condition | Prevalence |
|-----------|-----------|------------|
| **Closed** | Single-attractor family: seeds converge to same axis | 6/10 families |
| **Open (degeneracy)** | Multi-attractor, Mechanism 1: seeds find different orientations in shared subspace | 3/10 families (DistilBERT) |
| **Open (switching)** | Multi-attractor, Mechanism 2: seeds lock onto different feature sets | 1/10 families (QNLI/RoBERTa) |

The closed gate is the majority condition. When it is closed,
upstream pathology is absorbed and the merge is safe regardless of
representation-space damage.

### Level 3 — The mechanism determinants (new from n44)

Which gate state obtains is determined by:

1. **Task identity** (primary). Some tasks have rotationally
   degenerate feature subspaces (QNLI, MRPC on DistilBERT); others
   have one dominant direction (RTE, SST-2).

2. **Backbone depth** (secondary). Deeper backbones (RoBERTa, 12
   layers) can either collapse degeneracy into a single attractor
   (MRPC) or provide access to genuinely distinct feature sets
   (QNLI). Shallower backbones (DistilBERT, 6 layers) tend toward
   rotational degeneracy when multi-attractor.

3. **Training depth** (tertiary). Longer training can open access
   to secondary attractor basins (Strong QNLI) that shorter training
   does not reach (Medium/Weak QNLI).

---

## What the two mechanisms imply for catastrophe

The distinction between rotational degeneracy and feature-set
switching is not just descriptive. It may have different consequences
for how catastrophe manifests when the conjunction is satisfied.

### Rotational degeneracy + upstream pathology

The two adapters disagree about the decision axis but agree about
which features are relevant. When upstream pathology distorts the
shared feature subspace, both adapters are affected — the distortion
projects onto both adapters' decision axes because both axes live in
the same subspace. The merged classifier receives a corrupted version
of a representation it was trained to read.

**Prediction:** Catastrophe under Mechanism 1 should manifest as
*incoherent confidence* — the merged model makes the same errors
as the individual models but with distorted logit margins, because
the averaging of two rotated axes within a corrupted subspace
produces a direction that neither model was optimized for.

### Feature-set switching + upstream pathology

The two adapters disagree about which features to use entirely. When
upstream pathology selectively damages one feature set (the one
adapter A uses) while leaving the other intact (the one adapter B
uses), the merged classifier averages a corrupted signal with an
intact signal from a different feature space.

**Prediction:** Catastrophe under Mechanism 2 should manifest as
*systematic misclassification* — the merged model's effective
decision axis is pulled toward a direction that neither task uses
productively, creating a novel failure mode not present in either
source adapter.

These predictions are not yet testable (they require GPU inference
on merged models). But they suggest that the two mechanisms may
produce qualitatively different catastrophe profiles, which would
be visible in the existing merge outcome data if the right analysis
were applied.

---

## The complete mechanism ladder (updated)

| Rung | Scale | What it explains | Signal | Status |
|------|-------|-----------------|--------|--------|
| 1 | Module | Which pairs are catastrophe-capable | V-module dim ratio (d=3.36) | confirmed, 2 backbones |
| 2 | Head | Why severity varies across seeds | Head-level cancellation (7 heads, max Δ_DR=0.229) | confirmed, CA-01 |
| 3a | Readout (degeneracy) | Gate condition on DistilBERT | Rotational freedom in shared PC subspace | confirmed, 4 families |
| 3b | Readout (switching) | Gate condition on RoBERTa | Feature-set switching across PC subspaces | confirmed, 1 family |

Rung 3 is no longer a single mechanism. It bifurcates by backbone.
This is consistent with the sidecar's recurring finding that backbone
architecture is a modulating factor at every level of the mechanism
ladder.

---

## What this changes about commensurability

The glossary definition of commensurability (the conjunction of
upstream and readout compatibility) is unchanged. But the readout
component now decomposes further:

**Readout commensurability** =
- Single-attractor family → automatically satisfied (gate closed)
- Multi-attractor, Mechanism 1 → depends on which rotation each
  adapter selected within the degenerate subspace
- Multi-attractor, Mechanism 2 → depends on which feature set each
  adapter locked onto

The practical consequence: a future readout-commensurability check
would need to distinguish between these cases. For Mechanism 1, the
check is whether the two axes' angular separation within the shared
subspace exceeds some threshold. For Mechanism 2, the check is
whether the two axes occupy overlapping PC subspaces at all. These
are different computations with different failure semantics.

---

## What remains open

1. **Do the two mechanisms produce different catastrophe profiles?**
   Testable with existing merge outcomes if the analysis separates
   DistilBERT and RoBERTa catastrophic cases. (CPU-feasible but
   requires a new contrast design.)

2. **What property of a task determines whether its feature subspace
   is rotationally degenerate?** This is the deepest remaining
   question. It requires probing the pretrained model's representation
   structure for each task's input distribution — GPU inference
   territory.

3. **Does the mechanism bifurcation survive on DeBERTa?** DeBERTa's
   disentangled attention may create a third mechanism variant.
   This is testable once DeBERTa adapters are trained (n07).

4. **Is the QNLI/RoBERTa → RTE alignment a coincidence or a
   structural fact?** If it is structural — if QNLI genuinely has
   an RTE-like feature set available in RoBERTa's representation —
   then the cross-task feature exploitation is a deep property of
   pretrained models, not an artifact of this panel. Testable with
   more seeds.

---

## Relationship to prior notes

| Note | Status after n45 |
|------|-----------------|
| n33 (conjunctive synthesis) | Rung 3 superseded by n45's bifurcated Rungs 3a/3b |
| n37 (conjunctive model update) | Gate diagram superseded; upstream condition unchanged |
| n38 (ruled out) | New ruled-out: simple feature plurality as universal multi-attractor mechanism |
| n42 (readout topology) | Three-class taxonomy confirmed; mechanism content now specified |
| n44 (decision-axis findings) | Source data for the mechanism bifurcation |
