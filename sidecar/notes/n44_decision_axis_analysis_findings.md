# n44 — Decision-Axis Analysis Findings

**Type:** findings
**Date:** 2026-03-28
**Depends on:** n43 (attractor origin program), n41 (attractor mapping findings), n42 (topology synthesis)
**Status:** Stage A of the attractor origin program. Partially falsifies the simple feature plurality hypothesis; establishes a refined replacement.

---

## Summary verdict

**MIXED-POSITIVE — the feature plurality hypothesis is partially
falsified in its simple form and replaced by a sharper one.**

The simple version predicted that multi-attractor families' seeds
would load onto *different* principal components of the classifier
pathway, while single-attractor families' seeds would load onto the
*same* PCs. The data shows something more interesting: most
multi-attractor families' seeds use the *same* PCs but combine them
in orthogonal directions. The axes are orthogonal in representation
space while occupying the same low-dimensional PC subspace.

The exception is QNLI on RoBERTa, where the two seeds genuinely use
different PCs (shared top-3 = 0, energy overlap = 0.255). And the
cross-family alignment analysis reveals that one QNLI/RoBERTa seed
uses a decision direction aligned with RTE (cos = 0.86) — a feature
direction associated with an entirely different task. This is direct
evidence for feature plurality on at least one backbone.

The refined hypothesis: **multi-attractor structure arises from
rotational freedom within a shared low-rank subspace** (most cases)
or from **genuine feature-set switching** (QNLI on RoBERTa). The
distinction between these two mechanisms is itself backbone-dependent.

---

## 1. Two levels of analysis

The script measures decision-axis geometry at two levels:

**Pre-classifier PC space.** The decision axis d = W_cls[0] − W_cls[1]
(the direction in the pre-classifier's output space that determines
classification) is projected onto the pre-classifier's singular
vectors. This tells us which internal directions of the classifier
pathway each seed's decision axis loads onto.

**[CLS] representation space.** The effective decision direction
d_cls = W_pre^T @ d (the linear approximation of which direction in
the pretrained representation the full classifier pathway attends to)
is projected onto the pre-classifier's input singular vectors. This
tells us which directions in the pretrained model's natural coordinate
system each seed exploits.

---

## 2. Key metric: effective axis cosine

The most discriminative metric is the cosine between the two seeds'
effective decision directions in [CLS] space (d_cls). This is the
composed linear direction through the full classifier pathway.

| Group | Mean | Range |
|-------|------|-------|
| Single-attractor (n=8) | +0.971 | [+0.942, +0.985] |
| Multi-attractor (n=4) | +0.504 | [+0.088, +0.759] |
| Backbone-contingent (n=2) | +0.781 | [+0.602, +0.960] |

**The effective axis cosine cleanly separates single-attractor from
multi-attractor families.** Single-attractor pairs always have
eff_cos > 0.94. Multi-attractor pairs always have eff_cos < 0.76.
The groups do not overlap.

This confirms that the orthogonality observed in the raw readout
cosine (n36) propagates through the pre-classifier: multi-attractor
families' seeds use genuinely different directions in the pretrained
representation space, not just different projections of the same
direction.

---

## 3. The PC loading surprise

The simple feature plurality hypothesis predicts that multi-attractor
seeds should load onto different PCs. The data partly contradicts this:

| Pair | Attractor class | Energy overlap | Shared top-3 PCs |
|------|----------------|----------------|-----------------|
| QNLI/dist | multi | 0.937 | 1 |
| QNLI/robe | multi | 0.255 | 0 |
| SST-2(dom)/dist | multi | 0.985 | 2 |
| Strong QNLI/dist | multi | 0.934 | 1 |
| SST-2/dist | single | 0.991 | 3 |
| Yelp/dist | single | 0.987 | 2 |
| Amazon/dist | single | 0.983 | 2 |
| RTE/robe | single | 0.754 | 2 |

**Interpretation:** Most multi-attractor families (QNLI/DistilBERT,
SST-2 domain, Strong QNLI) have high energy overlap (> 0.93). Their
seeds load onto the *same* PCs. Yet their effective axes are
orthogonal. This means the seeds find orthogonal *combinations* of
the same principal components — rotational freedom within a shared
subspace, not a switch to a different subspace.

**QNLI/RoBERTa is the exception.** Energy overlap = 0.255, shared
top-3 = 0. Here the two seeds genuinely use different PCs. This is
the only clear case of the simple feature-plurality mechanism in the
current panel.

---

## 4. Two mechanisms for multi-attractor structure

The data reveals two distinct mechanisms:

### Mechanism 1 — Rotational degeneracy

The classifier pathway has a low-rank subspace (typically 1–2
effective dimensions in the decision-axis PC profile) within which
the decision axis is free to rotate without affecting classification
accuracy. Different seeds settle into different orientations within
this subspace. The axes are orthogonal but the subspace is shared.

**Signature:** High energy overlap, low shared top-3, low PC
effective rank (1–2), orthogonal effective axes.

**Confirmed instances:** QNLI/DistilBERT, SST-2 (domain)/DistilBERT,
Strong QNLI/DistilBERT, MRPC/DistilBERT.

**Interpretation:** The task's relevant features span a subspace of
dimension ≥ 2 in the pre-classifier's top PCs, and any unit vector
within that subspace is an equally valid decision boundary. The
optimization landscape has a continuous family of equally good
solutions (a degenerate manifold, not discrete basins). The bimodal
readout cosine we observe (≈0 or ≈1) may reflect the geometry of
random initialization within this subspace rather than discrete
basin hopping.

### Mechanism 2 — Feature-set switching

The two seeds genuinely use different principal components of the
pre-classifier, corresponding to different feature sets in the
pretrained representation.

**Signature:** Low energy overlap, zero shared top-3 PCs, high PC
effective rank (7+), orthogonal effective axes.

**Confirmed instance:** QNLI/RoBERTa.

**Interpretation:** RoBERTa's richer 12-layer representation encodes
multiple independent feature directions relevant to QNLI, and the
classifier can lock onto different ones depending on initialization.
This is the original feature plurality hypothesis, but it appears to
operate only on the richer backbone.

### The backbone distinction

The two mechanisms correlate with backbone depth:

- DistilBERT (6 layers): Mechanism 1 (rotational degeneracy).
  Decision axes have low effective rank (1.1–1.7 for multi-attractor
  families). The compressed representation provides one dominant
  subspace with rotational freedom.
- RoBERTa (12 layers): Mechanism 2 (feature-set switching) for QNLI.
  Decision axes have high effective rank (7.8–12.8). The richer
  representation provides genuinely distinct feature directions.

This explains the MRPC backbone contingency: MRPC is multi-attractor
on DistilBERT (Mechanism 1, rotational degeneracy in a compressed
space) but single-attractor on RoBERTa (the richer representation
collapses the degeneracy into one dominant direction, effective rank
2.7–3.5, and both seeds find it).

---

## 5. Cross-family alignment

The cross-family analysis (Stage C from n43, pulled forward) reveals
striking structure:

### QNLI/RoBERTa seed B aligns with RTE/RoBERTa

QNLI/rb/s7 has effective axis cosine +0.858 with RTE/rb/s7. One
QNLI seed, on RoBERTa, has found a decision direction that closely
resembles RTE's decision direction. This is **direct evidence that
multi-attractor families can exploit feature directions associated
with other tasks** — the strongest confirmation of the feature
plurality hypothesis at the cross-family level.

The other QNLI/RoBERTa seed (s42) has no strong cross-family
alignment (max cos = 0.109), suggesting it found a QNLI-specific
direction that no other task in the panel uses.

### MRPC/RoBERTa aligns with RTE/RoBERTa

Both MRPC/rb seeds align with RTE/rb (cos ≈ 0.69). Despite MRPC being
single-attractor on RoBERTa (both seeds aligned with each other), the
direction they converge to is one shared with RTE. MRPC and RTE on
RoBERTa may be using overlapping feature sets for their respective
classification tasks.

### SST-2 (domain) aligns with core SST-2

SST-2(dom) seeds align with core SST-2 seeds (cos ≈ 0.76). The
domain-shift adapters use a direction related to but distinct from
the core SST-2 direction. This is consistent with domain-shift
training pushing the classifier to a rotated variant of the same
feature.

### StrongQNLI/db/s7 = QNLI/db/s7 exactly

cos = 1.000. These adapters have identical effective decision
directions. This makes sense: they are the same task on the same
backbone with the same seed — the only difference is the training
set size/filtering for the "source strength" study. This validates
the analysis pipeline.

---

## 6. PC effective rank as a representation signature

An unexpected finding: the PC effective rank of the decision axis
is a strong signature of backbone architecture, largely independent
of attractor class.

| Backbone | Decision-axis PC effective rank (range) |
|----------|----------------------------------------|
| DistilBERT | 1.1 – 3.3 (most < 2.0) |
| RoBERTa | 1.9 – 12.8 (most > 7.0) |

**Exception:** Medium and Weak QNLI on DistilBERT have anomalously
high PC effective ranks (9–10) despite being DistilBERT adapters.
These are the only DistilBERT cases where the decision axis is spread
across many PCs. Both are single-attractor. This may reflect the
shorter training producing a decision axis that has not yet
concentrated into the dominant subspace.

RoBERTa adapters generally have higher effective ranks, meaning their
decision axes spread across more PCs. This is consistent with
RoBERTa's deeper representation providing more accessible feature
directions. But higher effective rank does not predict multi-attractor
structure: RTE/RoBERTa has effective rank 10.6–10.9 yet is firmly
single-attractor. The effective rank measures how *many* PCs the
decision axis engages; the attractor class measures whether two seeds
engage them in *compatible* ways.

---

## 7. Assessment of n43 hypotheses

### Feature plurality hypothesis — partially confirmed, refined

The simple form (multi-attractor seeds use different PCs) is confirmed
for QNLI/RoBERTa and falsified for DistilBERT multi-attractor families.
The refined version: multi-attractor structure arises from two
mechanisms depending on backbone depth.

### Alternative hypothesis A1 (dataset size) — falsified

Medium QNLI uses the same large dataset as Strong QNLI but is
single-attractor. SST-2 is single-attractor despite its large dataset.

### Alternative hypothesis A3 (random initialization) — partially confirmed

Mechanism 1 (rotational degeneracy) is essentially a random
initialization effect: the classifier head's random initialization
determines the orientation within a degenerate subspace. But it is
not *universally* random — it only operates in families where the
relevant feature subspace has dimension ≥ 2 (i.e., is rotationally
degenerate). Single-attractor families constrain the initialization
to converge.

### Alternative hypothesis A4 (anisotropy) — relevant

The pre-classifier SV spectra are similar across all adapters within
a backbone (effective rank ≈ 461–466 for both), so the pre-classifier
itself does not distinguish attractor classes. But the decision axis's
projection onto the pre-classifier PCs *does* differ: multi-attractor
families on DistilBERT have concentrated loadings (rank 1.2–1.7) while
single-attractor families on DistilBERT are slightly more diffuse
(rank 1.2–3.3). The difference is small and the ranges overlap.

---

## 8. Stage A success criteria assessment

From n43:

> Multi-attractor families' two seeds should load onto non-overlapping
> PC subsets, while single-attractor families' seeds should load onto
> the same PCs.

**Result:** Partially met. QNLI/RoBERTa satisfies this criterion
cleanly (energy overlap 0.255, shared top-3 = 0). DistilBERT
multi-attractor families do not — they load onto the same PCs but
combine them differently. The success criterion was too simple; the
data reveals a more interesting structure.

**Revised criterion:** The discriminative metric is the effective
axis cosine in [CLS] space, which cleanly separates attractor classes
(single > 0.94, multi < 0.76). The PC loading profile separates only
QNLI/RoBERTa from the rest. The mechanism is backbone-dependent.

---

## 9. Deliverables produced

| Deliverable | Path |
|------------|------|
| Analysis script | `sidecar/scripts/per_layer/decision_axis_analysis.py` |
| Decision axis projections | `sidecar/results/attractor_origin/decision_axis_projections.json` |
| PC loading profiles | `sidecar/results/attractor_origin/pc_loading_profiles.json` |
| Cross-family alignment | `sidecar/results/attractor_origin/cross_family_axis_alignment.json` |
| PC loadings figure | `sidecar/figures/attractor_origin_pc_loadings.svg` |
| Effective axes figure | `sidecar/figures/attractor_origin_effective_axes.svg` |
| SV spectra figure | `sidecar/figures/attractor_origin_sv_spectra.svg` |
| Cross-family heatmap | `sidecar/figures/attractor_origin_cross_family_heatmap.svg` |

---

## 10. What comes next

### Stage B (representation geometry audit) is now partially addressed

The SV spectra and effective ranks computed here provide the
representation geometry that Stage B would have measured. The key
finding — that the pre-classifier SV spectra are similar across
adapters and do not distinguish attractor classes — means a full
Stage B is unlikely to add new discriminative information. The
remaining Stage B question is whether the *composed* decision
subspace (classifier × pre-classifier) has a different dimensionality
for multi-attractor vs single-attractor families; the effective rank
data suggests the answer is backbone-dependent rather than
attractor-class-dependent.

### Stage C (cross-family alignment) is complete

The cross-family heatmap and alignment analysis were computed as part
of this Stage A run. The key result — QNLI/rb/s7 aligning with
RTE/rb/s7 — is the most important cross-family finding. A dedicated
Stage C note is optional.

### The remaining open question

The deepest open question is now: **what property of a task determines
whether its feature subspace is rotationally degenerate (admitting
Mechanism 1) or contains only one viable direction?** This is not
answerable from weight matrices alone — it would require probing the
pretrained model's representation structure for each task's input
distribution, which requires GPU inference.
