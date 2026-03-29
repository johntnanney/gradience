# n49 — Mechanism and Commensurability Synthesis

**Type:** synthesis
**Date:** 2026-03-28
**Depends on:** n48 (determinant findings), n46 (mechanism classification), n45 (conjunctive model with mechanisms), n42 (readout topology)
**Status:** Stage C of the Attractor Mechanism Determinants program. Connects attractor mechanism to commensurability and benign diversity.

---

## The question

What does it mean, for merge interpretation, that a family expresses
multi-attractor structure through rotational degeneracy versus
feature-set switching? The previous stages mapped and classified the
mechanisms. This note asks what those classifications imply for the
sidecar's broader theoretical structure — specifically, for the concept
of commensurability.

---

## 1. Three kinds of benign diversity

The sidecar now distinguishes three qualitatively different kinds of
readout diversity, each with a distinct geometric signature and a
distinct relationship to merge behavior.

### Single-attractor stability

The readout solution space has one dominant basin. Different seeds
converge to the same decision axis (effective axis cosine > 0.94).
The readout layer imposes no variation that merging must accommodate.

**Commensurability implication:** Readout commensurability is
automatically satisfied. The only risk factor is upstream V-module
pathology. If two adapters from a single-attractor family show
incompatible V-module geometry, the readout layer will not gate the
pathology — it will transmit it. But single-attractor families do not
*add* readout incompatibility to the problem.

**Prevalence:** 10 of 14 family×backbone conditions. The majority
condition.

### Rotationally degenerate diversity

The readout solution space has a continuous degenerate manifold: a
low-dimensional subspace (typically 1–2 effective dimensions) within
which the decision axis is free to rotate without affecting
classification accuracy. Different seeds settle at different
orientations. The axes are orthogonal in representation space but pass
through the same low-dimensional PC subspace.

**What the diversity *is*:** Two adapters that use the same features
but combine them in different proportions. They agree about which
dimensions of the representation are task-relevant; they disagree about
the relative weighting. The disagreement is angular, not structural.

**Commensurability implication:** Readout diversity arises from the
geometry of the solution space, not from a difference in what the
adapters have learned about the task. Linear merging of two
rotationally degenerate adapters averages their decision axes within
the shared subspace, producing a direction that neither seed selected
but that still lies within the task-relevant subspace. Whether this
averaged direction is a *good* classifier depends on the relationship
between the two original axes and the class boundaries — but the
averaged direction is at least task-relevant.

The readout gate (n45, Level 2) is **open** under rotational degeneracy:
the two adapters' decision pathways do not agree, so upstream pathology
is not absorbed. But the gate opens onto a shared subspace. If
upstream pathology damages this shared subspace, both adapters are
affected symmetrically. The catastrophe, if it occurs, would manifest
as incoherent confidence (n45 §Rotational degeneracy + upstream
pathology): the merged model attends to the right features but with a
distorted decision boundary.

**Prevalence:** 4 of 14 conditions. All on DistilBERT.

### Feature-set switching diversity

The readout solution space has multiple discrete basins that occupy
different PC subspaces. Different seeds lock onto genuinely different
feature sets in the pretrained representation. The axes are orthogonal
and the subspaces do not overlap.

**What the diversity *is*:** Two adapters that solve the same task by
attending to different information. They disagree about which dimensions
of the representation are task-relevant. The disagreement is structural,
not merely angular.

**Commensurability implication:** Linear merging of two
feature-switching adapters averages a decision axis in one subspace
with a decision axis in a completely different subspace. The averaged
direction lies in neither subspace and may not correspond to any
task-relevant direction. The merged classifier receives a decision
direction that neither adapter was optimized for and that does not
necessarily align with any feature set the pretrained model supports.

The readout gate is **open** under feature-set switching, and it opens
onto non-overlapping subspaces. If upstream pathology selectively
damages one adapter's feature subspace while leaving the other intact,
the merged classifier averages a corrupted signal with an intact signal
from a different subspace. The catastrophe, if it occurs, would
manifest as systematic misclassification (n45 §Feature-set switching +
upstream pathology): the merged decision axis is pulled into a
direction that is productively related to neither source task.

**Prevalence:** 1 of 14 conditions. QNLI on RoBERTa only.

---

## 2. Commensurability, refined

The sidecar's concept of **commensurability** has evolved through
three refinements:

**Version 1 (n33):** Commensurability = upstream V-module compatibility
AND readout compatibility. Two conditions, both necessary, either
insufficient alone.

**Version 2 (n37, n42):** Readout compatibility decomposes into
attractor topology: single-attractor families satisfy it automatically;
multi-attractor families may or may not depending on which basin each
adapter landed in.

**Version 3 (this note):** Readout compatibility further decomposes
by *mechanism*. Readout incompatibility has different semantic content
depending on whether it arises from rotational degeneracy or
feature-set switching:

| Readout state | Mechanism | What the incompatibility means | Merge consequence (if upstream pathology present) |
|--------------|-----------|-------------------------------|------------------------------------------------|
| Compatible | — | Same decision axis | Safe: gate closed |
| Incompatible | rotational degeneracy | Different orientation in shared features | Incoherent confidence: right features, wrong boundary |
| Incompatible | feature-set switching | Different features entirely | Systematic misclassification: novel failure mode |

This refinement does not change the top-level conjunctive model
(catastrophe still requires both conditions). It changes what the
readout condition *means*. Readout incompatibility from rotational
degeneracy is a milder form of openness than readout incompatibility
from feature-set switching — the gate is open in both cases, but the
downstream consequences differ because the geometry of the
incompatibility differs.

---

## 3. What this does NOT imply

### It does not predict which merges will fail

The mechanism classification tells you the *kind* of readout diversity
present, not whether upstream pathology exists. Without upstream
V-module pathology, both kinds of readout diversity are benign — all
multi-attractor same-task families merge safely (max Δ = 2.2%),
regardless of mechanism. The mechanism matters only in the conjunction:
**if** upstream pathology is also present, the mechanism determines
**how** catastrophe manifests, not **whether** it occurs.

### It does not make feature-set switching dangerous

Feature-set switching is not a risk factor. It is a description of the
solution space's topology. QNLI/RoBERTa shows feature-set switching
and merges safely in all tested conditions. The only way feature-set
switching could contribute to catastrophe is in conjunction with
upstream V-module pathology — and even then, the mechanism's
contribution would be to shape the *form* of failure, not to create
risk where none existed.

### It does not make rotational degeneracy safe

Rotational degeneracy is the more common multi-attractor mechanism
in the current panel, and all current instances merge safely. But this
is because no same-task pair in the panel has upstream V-module
pathology. If a cross-task pair with V-module pathology also had
rotationally degenerate readout, the conjunctive model predicts
catastrophe — just with a different failure profile than feature-set
switching would produce.

### It does not upgrade the sidecar to a diagnostic tool

These findings remain descriptive and explanatory. The mechanism
classification enriches understanding of why benign diversity exists
and how it is organized. It does not produce a metric that could be
added to core Gradience without further validation on additional
backbones and with GPU inference.

---

## 4. Integration with the mechanism ladder

The updated mechanism ladder after this program:

| Rung | Scale | What it explains | Signal | Mechanism detail | Status |
|------|-------|-----------------|--------|-----------------|--------|
| 1 | Module | Which pairs are catastrophe-capable | V-module dim ratio (d=3.36) | — | confirmed, 2 backbones |
| 2 | Head | Why severity varies across seeds | Head-level cancellation | — | confirmed, CA-01 |
| 3a | Readout (degeneracy) | Gate condition on compressed backbones | Rotational freedom in shared PC subspace | Low eff rank, high energy overlap | confirmed, 4 families |
| 3b | Readout (switching) | Gate condition on rich backbones | Feature-set switching across PC subspaces | High eff rank, low energy overlap | confirmed, 1 family |
| — | Determinants | Why 3a vs 3b | Backbone representation depth | Task → backbone → convergence hierarchy | confirmed with confound |

Rung 3 now has mechanism content: the gate is not just open or closed,
it opens in structurally different ways depending on the backbone. The
determinant hierarchy provides the explanation for which way it opens.

---

## 5. The "benign diversity" concept

This program gives the sidecar a positive theory of benign diversity —
not just "multi-attractor is not dangerous" (negative finding) but
"multi-attractor structure has specific mechanisms, each with
identifiable geometric signatures and interpretable consequences"
(positive finding).

Benign diversity is not one thing. It has at least two qualitatively
distinct forms, determined by an identifiable hierarchy of factors.
This makes the solution-space topology less mysterious: it is not
arbitrary that some families have multiple attractors. The multiplicity
arises from specific properties of the task–backbone–training
interaction, and the *form* of that multiplicity reveals something
about the representation geometry of the pretrained model.

The philosophical move here is from taxonomy to explanation: the
attractor mapping lab (n41) gave us a classification; the attractor
origin program (n44) gave us mechanisms; this determinant program
(n46–n48) gives us a hierarchy of causes. The three programs together
transform "some families have multiple attractors" from an observation
into a structured explanatory account.

---

## 6. Commensurability context table

The following table summarizes the commensurability context for each
mechanism class:

| Mechanism class | Readout gate | Diversity type | Merge consequence (no upstream pathology) | Merge consequence (with upstream pathology) | Check required |
|----------------|-------------|---------------|------------------------------------------|-------------------------------------------|---------------|
| single_attractor | closed | none | safe | upstream damage absorbed by gate | V-module only |
| rotational_degeneracy | open | angular (shared subspace) | safe (averaged axis still task-relevant) | incoherent confidence (right features, wrong boundary) | V-module + angular separation |
| feature_set_switching | open | structural (different subspaces) | safe (averaged axis may not be task-relevant, but no pathology to transmit) | systematic misclassification (novel failure mode) | V-module + subspace overlap |

The "check required" column indicates what a future commensurability
assessment would need to measure for each mechanism class. For
single-attractor families, only the upstream V-module condition matters.
For rotational degeneracy, the angular separation within the shared
subspace is the relevant readout variable. For feature-set switching,
the relevant variable is whether the two adapters' PC subspaces overlap
at all.

These are different computations with different failure semantics. A
future commensurability metric would need to detect which mechanism
class applies before selecting the appropriate readout check.

---

## 7. Deliverables produced

| Deliverable | Path |
|------------|------|
| This synthesis note | `sidecar/notes/n49_mechanism_and_commensurability_synthesis.md` |
| Commensurability context table | `sidecar/results/attractor_mechanisms/commensurability_context_table.json` |

---

## 8. Success criteria assessment

> **Sharpens the meaning of benign diversity.**

Met. Three qualitatively distinct forms of readout diversity are now
defined with geometric signatures and interpretable consequences.

> **Distinguishes types of attractor multiplicity.**

Met. Rotational degeneracy and feature-set switching are distinguished
at the mechanism level, with different geometric signatures, different
prevalence, and different predicted consequences under conjunctive
failure.

> **Strengthens the concept of commensurability.**

Met. Commensurability version 3 decomposes the readout condition by
mechanism, specifying different computational checks for each class.

> **Avoids overclaiming predictive power.**

Met. The synthesis explicitly states that mechanism classification
does not predict merge outcomes, does not make feature-set switching
dangerous, and does not upgrade the sidecar to a diagnostic tool.

---

## 9. What remains open

1. **Do the two mechanisms produce observably different catastrophe
   profiles?** Testable with existing cross-task merge outcomes if the
   analysis separates DistilBERT and RoBERTa catastrophic cases.

2. **Does the backbone confound dissolve on a third backbone?** The
   DeBERTa adjudication (n07) will show whether disentangled attention
   produces rotational degeneracy, feature-set switching, or a novel
   third mechanism.

3. **Is the angular separation metric for rotational degeneracy
   predictive?** Within the degenerate subspace, some angular
   separations may be more problematic than others for merging. This
   requires GPU inference to test.

4. **Can the mechanism classification be automated?** The classification
   rules in n46 are operational but require pre-classifier weight
   extraction and SVD. A fully automated pipeline would need to handle
   architectural differences across backbones.
