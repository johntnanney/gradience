# n46 — Attractor Mechanism Classification Audit

**Type:** classification audit
**Date:** 2026-03-28
**Depends on:** n44 (decision-axis analysis findings), n41 (attractor mapping findings), n42 (readout topology)
**Status:** Stage A of the Attractor Mechanism Determinants program. Assigns each analyzable family×backbone condition to a mechanism class.

---

## Summary verdict

**POSITIVE — all currently analyzable multi-attractor families are
classifiable, with clear exemplars of both mechanisms and no forced
classifications.**

Four multi-attractor conditions are classified as **rotational
degeneracy**. One is classified as **feature-set switching**. One
backbone-contingent family (MRPC) is classified as rotational degeneracy
on DistilBERT and single-attractor on RoBERTa. No conditions require
the mixed_or_unresolved label — the current panel resolves cleanly,
though the single confirmed instance of feature-set switching means
that mechanism's generality remains undertested.

---

## 1. Classification rules

The following operational rules assign each condition to a mechanism
class. They are derived from the n44 findings and formalized here for
reproducibility.

### `single_attractor`

Apply when:
- Effective axis cosine between seeds > 0.90
- Both seeds converge to the same decision direction

No mechanism classification is needed — the family has one readout basin.

### `rotational_degeneracy`

Apply when:
- Effective axis cosine between seeds < 0.80 (multi-attractor)
- **AND** energy overlap > 0.85
- **AND** PC effective rank of decision axes is low (< 4.0 for both seeds)

Interpretation: the seeds find orthogonal orientations within a shared
low-rank PC subspace. The decision axis has rotational freedom within
that subspace.

### `feature_set_switching`

Apply when:
- Effective axis cosine between seeds < 0.80 (multi-attractor)
- **AND** energy overlap < 0.40
- **AND** shared top-3 PCs = 0
- **AND** PC effective rank is high for at least one seed (> 5.0)

Interpretation: the seeds lock onto genuinely different principal
components of the pretrained representation, corresponding to different
feature sets.

### `mixed_or_unresolved`

Apply when:
- Evidence is ambiguous (e.g., moderate energy overlap, some shared PCs
  but low effective axis cosine)
- Or the condition is insufficiently sampled to distinguish mechanisms

### Threshold notes

The energy overlap boundary between rotational degeneracy (> 0.85) and
feature-set switching (< 0.40) has a wide gap (0.40–0.85) that no
current multi-attractor family occupies. The boundary is therefore not
sensitive to exact threshold placement. If future families fall in this
gap, the mixed_or_unresolved label should be applied.

The effective axis cosine boundary of 0.80 for multi-attractor
classification is slightly above the empirical maximum of 0.76 observed
in the current panel (n44 §2). The 0.90 boundary for single-attractor
is slightly below the empirical minimum of 0.94. This leaves a
0.80–0.90 gap for backbone-contingent or borderline cases.

---

## 2. The classification table

| # | Family | Backbone | Attractor class | Mechanism class | Eff axis cos | Energy overlap | Shared top-3 | PC eff rank (s42/s7) | Confidence | Evidence basis |
|---|--------|----------|----------------|-----------------|-------------|---------------|-------------|---------------------|------------|----------------|
| 1 | QNLI | DistilBERT | multi_attractor | rotational_degeneracy | 0.597 | 0.937 | 1 | 1.36 / 1.55 | high | High energy overlap despite orthogonal effective axes. Both seeds load primarily on PC0. Low effective rank confirms shared subspace. |
| 2 | QNLI | RoBERTa | multi_attractor | feature_set_switching | 0.087 | 0.255 | 0 | 12.78 / 7.77 | high | Lowest energy overlap in panel. Zero shared top-3 PCs. High effective rank for both seeds. QNLI/rb/s7 aligns with RTE (cos=0.86), confirming cross-task feature exploitation. |
| 3 | SST-2 (domain) | DistilBERT | multi_attractor | rotational_degeneracy | 0.759 | 0.985 | 2 | 1.19 / 1.14 | high | Highest energy overlap among multi-attractor families. 2 shared top-3 PCs. Lowest effective rank in panel. Textbook rotational degeneracy. |
| 4 | Strong QNLI | DistilBERT | multi_attractor | rotational_degeneracy | 0.574 | 0.934 | 1 | 1.66 / 1.55 | high | Near-identical profile to QNLI/DistilBERT (same task, same backbone, same seeds, different training set). StrongQNLI/db/s7 = QNLI/db/s7 exactly (cos=1.000). |
| 5 | MRPC | DistilBERT | backbone_contingent (multi on db) | rotational_degeneracy | 0.602 | 0.948 | 1 | 1.43 / 1.30 | high | Multi-attractor on DistilBERT with high energy overlap and low effective rank. Same mechanism signature as QNLI/DistilBERT. |
| 6 | MRPC | RoBERTa | backbone_contingent (single on rb) | single_attractor | 0.960 | 0.868 | 3 | 3.50 / 2.72 | high | Aligned effective axes on RoBERTa. All 3 top-3 PCs shared. The richer backbone collapses the degeneracy. |
| 7 | RTE | DistilBERT | single_attractor | single_attractor | 0.972 | 0.851 | 1 | 2.09 / 3.33 | high | — |
| 8 | RTE | RoBERTa | single_attractor | single_attractor | 0.942 | 0.754 | 2 | 10.92 / 10.65 | high | — |
| 9 | SST-2 | DistilBERT | single_attractor | single_attractor | 0.979 | 0.991 | 3 | 1.20 / 1.21 | high | — |
| 10 | SST-2 | RoBERTa | single_attractor | single_attractor | 0.947 | 0.976 | 3 | 1.98 / 1.90 | high | — |
| 11 | Yelp | DistilBERT | single_attractor | single_attractor | 0.982 | 0.987 | 2 | 1.25 / 1.24 | high | — |
| 12 | Amazon | DistilBERT | single_attractor | single_attractor | 0.984 | 0.983 | 2 | 1.30 / 1.31 | high | — |
| 13 | Medium QNLI | DistilBERT | single_attractor | single_attractor | 0.976 | 0.604 | 1 | 10.08 / 9.18 | high | Single-attractor despite anomalously high PC effective rank (9–10). Low energy overlap for a single-attractor case. See §4 for interpretation. |
| 14 | Weak QNLI | DistilBERT | single_attractor | single_attractor | 0.985 | 0.908 | 2 | 2.49 / 2.84 | high | — |

---

## 3. Mechanism class summary

| Mechanism class | Count | Families |
|----------------|-------|---------|
| single_attractor | 10 | RTE (both), SST-2 (both), Yelp, Amazon, Medium QNLI, Weak QNLI, MRPC/RoBERTa |
| rotational_degeneracy | 4 | QNLI/DistilBERT, SST-2(dom)/DistilBERT, Strong QNLI/DistilBERT, MRPC/DistilBERT |
| feature_set_switching | 1 | QNLI/RoBERTa |
| mixed_or_unresolved | 0 | — |

### Distribution observation

All four rotational degeneracy cases are on DistilBERT. The single
feature-set switching case is on RoBERTa. No DistilBERT family shows
feature-set switching; no RoBERTa family shows rotational degeneracy.
The mechanism split is perfectly confounded with backbone in the
current panel. This is the central observation that Stage B must address.

---

## 4. Notable cases and edge conditions

### QNLI across backbones: the cleanest mechanism contrast

QNLI is multi-attractor on both backbones but expresses it through
different mechanisms. On DistilBERT, the two seeds share a PC subspace
(energy overlap = 0.937) and find orthogonal orientations within it.
On RoBERTa, the two seeds occupy entirely different PC subspaces
(energy overlap = 0.255, shared top-3 = 0). Same task, different
mechanism, different backbone. This is the strongest evidence that
backbone architecture determines mechanism choice, not task identity
alone.

### MRPC as backbone-contingent: mechanism tracks attractor class

MRPC is multi-attractor on DistilBERT and single-attractor on
RoBERTa. On DistilBERT, it shows rotational degeneracy (energy
overlap = 0.948, low effective rank). On RoBERTa, the degeneracy
collapses and both seeds converge. The backbone transition changes
not just the attractor count but the mechanism that would generate
multi-attractor structure.

### Medium QNLI: single-attractor with anomalous geometry

Medium QNLI is single-attractor (eff_axis_cos = 0.976) but has
anomalously high PC effective rank (9–10) and low energy overlap
(0.604). The high effective rank is comparable to RoBERTa adapters,
not typical DistilBERT. Interpretation: the shorter training (Medium
vs Strong) produces a decision axis that has not concentrated into the
dominant subspace. The axis is diffuse across PCs but the two seeds'
diffuse axes point in the same direction. Longer training (Strong QNLI)
concentrates the axis into a low-rank subspace (effective rank 1.5–1.7)
and opens rotational degeneracy — the concentration *enables* the
multiplicity by creating a low-dimensional manifold with rotational
freedom.

This is an important observation: **training convergence can change the
effective dimensionality of the decision axis, and that change can
open or close multi-attractor structure.** The causal path is:
training depth → axis concentration → rotational degeneracy → multiple
attractors.

### Strong QNLI = QNLI in decision-axis space

StrongQNLI/db/s7 has effective axis cosine = 1.000 with QNLI/db/s7.
These adapters share a seed, backbone, and task; the only difference
is training-set filtering. The identity confirms that the Strong QNLI
panel is informationally equivalent to the QNLI panel at the
decision-axis level.

### QNLI/rb/s7 → RTE alignment: cross-task feature exploitation

QNLI/rb/s7 has effective axis cosine +0.858 with RTE/rb/s7 (from the
cross-family alignment matrix). One of QNLI's seeds, on RoBERTa, has
found a decision direction resembling a different task's direction.
The other QNLI/RoBERTa seed (s42) shows no strong cross-family
alignment (max cos = 0.109). This confirms that feature-set switching
on RoBERTa allows access to directions that other tasks also use —
the pretrained representation's feature inventory is shared across
tasks.

---

## 5. Assessment of success criteria

From the spec:

> **All currently analyzable multi-attractor families are classified.**

Met. All 5 multi-attractor conditions (4 rotational degeneracy + 1
feature-set switching) are classified with high confidence. No
conditions require the mixed_or_unresolved label.

> **At least one family clearly exemplifies rotational degeneracy.**

Met. QNLI/DistilBERT, SST-2(dom)/DistilBERT, Strong QNLI/DistilBERT,
and MRPC/DistilBERT all clearly exemplify rotational degeneracy.
SST-2(dom)/DistilBERT is the most textbook case (energy overlap =
0.985, 2 shared top-3 PCs, lowest effective rank in panel).

> **At least one family clearly exemplifies feature-set switching, or
> the absence of this class is recorded honestly.**

Met. QNLI/RoBERTa clearly exemplifies feature-set switching (energy
overlap = 0.255, 0 shared top-3 PCs, highest effective rank for s42,
cross-task alignment with RTE). This is the only confirmed instance.

> **Unresolved cases are explicitly labeled rather than forced.**

Met trivially — no cases required the unresolved label. However, the
panel's resolution may partly reflect the wide energy-overlap gap
between the two mechanism clusters (0.255 vs 0.934+). Future families
falling in the gap might require the unresolved label.

---

## 6. Relationship to prior notes

| Note | Status after n46 |
|------|--------------------|
| n44 (decision-axis findings) | Source data; mechanism labels now formalized here |
| n45 (conjunctive model with mechanisms) | Unchanged; n46 provides the formal classification that n45 references informally |
| n42 (readout topology) | Three-class topology confirmed; mechanism content now mapped per family |
| n41 (attractor mapping findings) | Attractor classes unchanged; mechanism dimension added |

---

## 7. What comes next

Stage B (determinant analysis) should test why the mechanism split
correlates perfectly with backbone in the current panel. The key
question: **is backbone the cause, or is the backbone confound an
artifact of the panel's limited coverage?**

The QNLI cross-backbone contrast (rotational degeneracy on DistilBERT,
feature-set switching on RoBERTa) is the strongest evidence for a
causal backbone effect. The MRPC cross-backbone contrast (rotational
degeneracy on DistilBERT, single-attractor on RoBERTa) shows that
backbone can change attractor count as well as mechanism. But without
a family that shows feature-set switching on DistilBERT or rotational
degeneracy on RoBERTa, the backbone confound cannot be disentangled
from a task×backbone interaction.
