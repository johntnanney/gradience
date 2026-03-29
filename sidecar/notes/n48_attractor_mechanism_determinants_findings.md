# n48 — Attractor Mechanism Determinants: Findings

**Type:** findings
**Date:** 2026-03-28
**Depends on:** n47 (determinant protocol), n46 (mechanism classification), n44 (decision-axis findings)
**Status:** Stage B of the Attractor Mechanism Determinants program. Identifies the determinant hierarchy for mechanism choice.

---

## Summary verdict

**MIXED-POSITIVE — a structured determinant hierarchy emerges, but with
a critical backbone confound that the current panel cannot fully
disentangle.**

The hierarchy: **task identity** (primary) → **backbone architecture**
(secondary) → **training convergence** (tertiary) → **domain
structure** (weak). This ordering is consistent with the earlier
attractor topology findings (n42) and extends them by adding mechanism
selection as a distinct dimension. The backbone confound — all
rotational degeneracy on DistilBERT, all feature-set switching on
RoBERTa — is the main limitation.

---

## 1. Contrast results

### Contrast 1 — Backbone contrast

| Task | DistilBERT mechanism | RoBERTa mechanism | Mechanism changes? |
|------|---------------------|-------------------|-------------------|
| QNLI | rotational_degeneracy | feature_set_switching | **Yes — mechanism flips** |
| MRPC | rotational_degeneracy | single_attractor | **Yes — attractor count changes** |
| RTE | single_attractor | single_attractor | No |
| SST-2 | single_attractor | single_attractor | No |

**Finding:** Backbone architecture changes mechanism class in 2 of 4
testable families. In both cases, the change goes in the same
direction: DistilBERT multi-attractor → RoBERTa either different
mechanism (QNLI) or single-attractor (MRPC). Single-attractor families
are stable across backbones.

**Interpretation:** Backbone architecture is a **mechanism selector**
for families that are already multi-attractor by task identity. It does
not create multi-attractor structure in families that are intrinsically
single-attractor. The deeper backbone (RoBERTa, 12 layers) either
opens access to genuinely distinct feature sets (QNLI) or collapses
rotational degeneracy into a dominant direction (MRPC). The shallower
backbone (DistilBERT, 6 layers) compresses representations into
low-rank subspaces where rotational freedom is the only available
mechanism for multi-attractor structure.

### Contrast 2 — Convergence contrast

| Condition | Attractor class | Mechanism | Mean PC eff rank | Energy overlap |
|-----------|----------------|-----------|-----------------|---------------|
| Weak QNLI | single | single_attractor | 2.67 | 0.908 |
| Medium QNLI | single | single_attractor | 9.63 | 0.604 |
| Strong QNLI | multi | rotational_degeneracy | 1.61 | 0.934 |
| QNLI (core) | multi | rotational_degeneracy | 1.46 | 0.937 |

**Finding:** Training depth changes attractor count (Weak/Medium →
single, Strong/core → multi) but does **not** change mechanism class.
All convergence-opened multi-attractor cases show rotational
degeneracy. The mechanism is determined by the backbone (DistilBERT),
not by the training depth.

**The Medium QNLI anomaly:** Medium QNLI has anomalously high PC
effective rank (9.63) despite being single-attractor. This is the
opposite of the expected pattern: the other DistilBERT multi-attractor
families have *low* effective rank (1.2–1.7). Interpretation: Medium
training produces a decision axis that has not yet concentrated into
the dominant subspace. The axis is diffuse across many PCs, but both
seeds' diffuse axes point in the same direction — hence single-attractor.
Stronger training concentrates the axis into a low-rank subspace
(effective rank 1.5–1.7), and this concentration creates a
low-dimensional manifold with rotational freedom, enabling degeneracy.

**Causal path:** training depth → axis concentration → subspace
dimensionality reduction → rotational freedom within subspace →
multiple attractors. Convergence is a **gate on attractor count**, not
a selector of mechanism.

### Contrast 3 — Domain / label contrast

| Condition | Attractor class | Mechanism | Notes |
|-----------|----------------|-----------|-------|
| SST-2 core (DistilBERT) | single | single_attractor | Standard sentiment training |
| SST-2 domain (DistilBERT) | multi | rotational_degeneracy | Domain-shifted sentiment training |
| Yelp (DistilBERT) | single | single_attractor | Same task type, different corpus |
| Amazon (DistilBERT) | single | single_attractor | Same task type, different corpus |

**Finding:** Domain shift opens multi-attractor structure in SST-2, but
label similarity does not predict attractor class. Yelp and Amazon
(same sentiment task, different corpora) are single-attractor. The
domain effect operates through the training *distribution*, not the
label structure.

**Limitation:** This is a single contrast (n=1 domain-shift vs n=1
core). The domain effect is real but its generality is uncertain.

### Contrast 4 — Task-family contrast (backbone held constant)

**On DistilBERT:**

| Task | Mechanism | Notes |
|------|-----------|-------|
| QNLI | rotational_degeneracy | Always multi-attractor |
| MRPC | rotational_degeneracy | Multi on DistilBERT only |
| SST-2 (domain) | rotational_degeneracy | Only with domain shift |
| RTE | single_attractor | Always single |
| SST-2 | single_attractor | Under standard training |
| Yelp/Amazon | single_attractor | Always single |

**On RoBERTa:**

| Task | Mechanism | Notes |
|------|-----------|-------|
| QNLI | feature_set_switching | Only confirmed FSS case |
| MRPC | single_attractor | Degeneracy collapsed |
| RTE | single_attractor | Always single |
| SST-2 | single_attractor | Always single |

**Finding:** Task identity determines **whether multi-attractor
structure is possible** (QNLI yes, RTE no). Backbone determines
**which mechanism realizes it** when it appears. The task is the
necessary condition; the backbone is the sufficient condition for a
specific mechanism.

---

## 2. The determinant hierarchy

The five candidate factors arrange into a clear hierarchy:

### Tier 1 — Task family (primary determinant of attractor count)

Task identity is the strongest predictor of whether a family is
single-attractor or multi-attractor. QNLI is multi-attractor in every
condition where sufficient training is provided (both backbones, both
seed pairs). RTE is single-attractor in every condition (both backbones,
both seed pairs). This is the same result found in n42, now confirmed to
hold at the mechanism level.

**Strength:** Supported by all 14 entries. No exceptions.

### Tier 2 — Backbone architecture (primary determinant of mechanism choice)

Backbone architecture determines *which mechanism* realizes
multi-attractor structure. On DistilBERT (6 layers), multi-attractor
families always show rotational degeneracy. On RoBERTa (12 layers),
QNLI shows feature-set switching. The mechanism and backbone are
perfectly confounded in the current panel.

The working explanation: DistilBERT's compressed 6-layer representation
creates a lower-dimensional feature space where task-relevant
information concentrates into a low-rank subspace (PC effective rank
1.1–1.7 for multi-attractor families). Within this subspace, multiple
orientations are equally valid — rotational degeneracy. RoBERTa's
12-layer representation provides enough dimensions that genuinely
distinct feature sets can coexist, enabling feature-set switching.

**Strength:** Supported by 6 entries where cross-backbone data exists.
The QNLI contrast is the strongest evidence: same task, same seeds,
different mechanism by backbone alone. But the perfect confound (no
feature-set switching on DistilBERT, no rotational degeneracy on
RoBERTa) means the claim is suggestive, not decisive.

### Tier 3 — Training convergence (modulates attractor count, not mechanism)

Training depth changes attractor count (Weak/Medium → single, Strong →
multi) but does not change mechanism class. All convergence-opened
cases show the same mechanism as the backbone's default. Convergence is
a gate on attractor count, not a selector of mechanism.

The convergence effect operates through axis concentration: stronger
training concentrates the decision axis into a lower-rank subspace,
creating the geometric conditions for rotational degeneracy.

**Strength:** Supported by 4 entries in the QNLI strength series. Only
one task tested. The generality of the concentration → degeneracy
pathway is uncertain.

### Tier 4 — Domain structure (weak, single contrast)

Domain-shifted training can open multi-attractor structure in a
family that is single-attractor under standard training (SST-2). But
this is a single data point. Domain effects likely operate through the
same training-distribution pathway as convergence effects — changing
the optimization landscape, not the representation geometry.

**Strength:** Supported by 2 entries (SST-2 core vs domain). Weakly
evidenced. The domain effect is real but its relationship to the
mechanism hierarchy is unclear.

### Representation richness (correlated, not independent)

PC effective rank correlates with mechanism class (low rank ↔ rotational
degeneracy, high rank ↔ feature-set switching). But this proxy is
downstream of backbone architecture: DistilBERT naturally produces
low-rank decision axes, RoBERTa naturally produces high-rank ones. The
proxy is informative but not independently testable from backbone.

**Exception:** Medium QNLI on DistilBERT has high effective rank (9.63)
despite being on DistilBERT. This shows that the backbone → rank
association is not deterministic; training depth also affects rank. But
Medium QNLI is single-attractor, not multi-attractor — high rank alone
does not produce multi-attractor structure.

---

## 3. Hypothesis assessment

### H1 — Rotational degeneracy favored by compressed representations

**Confirmed for the current panel.** All four rotational degeneracy
cases are on DistilBERT, and all have low PC effective rank (1.1–1.7).
The compressed 6-layer representation creates exactly the kind of
low-rank subspace where rotational freedom is the available mechanism.

### H2 — Feature-set switching favored by richer representations

**Confirmed for one case.** QNLI on RoBERTa is the only feature-set
switching instance, and RoBERTa's richer representation is the best
available explanation. But with n=1, this could be a QNLI-specific
property rather than a general backbone effect.

### H3 — Training convergence affects attractor count and mechanism

**Partially confirmed.** Training depth affects attractor count
(confirmed) but does **not** affect mechanism choice (contra H3).
Mechanism is determined by backbone, not convergence. The hypothesis
is half right.

### H4 — Task family structure influences mechanism choice

**Partially confirmed.** Task identity determines whether
multi-attractor structure is possible, which is a necessary precondition
for mechanism choice. But the task does not directly select which
mechanism appears — that is determined by the backbone. H4's stronger
prediction (that tasks with "many semantically valid solution
decompositions" would favor feature-set switching) is not clearly
supported: QNLI shows feature-set switching on RoBERTa but rotational
degeneracy on DistilBERT, so the task alone does not determine the
mechanism.

### H5 — Domain similarity and label structure do not predict mechanism

**Confirmed.** Yelp, Amazon, and SST-2 are all binary sentiment tasks
but their attractor structures differ (Yelp/Amazon single-attractor,
SST-2 domain multi-attractor). MRPC and RTE are both sentence-pair
tasks but differ in attractor class. Label structure is not predictive.

---

## 4. The backbone confound

The most important limitation of this analysis: **mechanism class and
backbone are perfectly confounded in the current panel.** Every
rotational degeneracy case is DistilBERT. The only feature-set
switching case is RoBERTa. There is no case of feature-set switching
on DistilBERT, and no case of rotational degeneracy on RoBERTa.

This means the determinant hierarchy cannot distinguish between:

**Interpretation A (backbone causes mechanism):** DistilBERT's
compressed representation *forces* rotational degeneracy because
feature-set switching is geometrically impossible in a 6-layer
representation. RoBERTa's richer representation *enables* feature-set
switching.

**Interpretation B (task×backbone interaction causes mechanism):**
QNLI specifically has multiple independent feature sets available in
RoBERTa's representation, while other tasks do not. The mechanism split
is a QNLI property, not a backbone property.

**Interpretation C (n=1 artifact):** Feature-set switching is rare
regardless of backbone. The single QNLI/RoBERTa instance could be an
outlier.

Distinguishing these interpretations requires either: (a) a
multi-attractor family on RoBERTa other than QNLI, to test whether
feature-set switching generalizes; or (b) a family that shows
feature-set switching on DistilBERT, to falsify the backbone-as-cause
hypothesis. Neither is available in the current panel. The DeBERTa
adjudication (n07) will help: if DeBERTa shows feature-set switching,
that supports Interpretation A (deeper backbones enable FSS); if it
shows rotational degeneracy, that supports a more complex interaction.

---

## 5. The emerging causal model

Despite the backbone confound, a tentative causal model is warranted:

```
Task identity
    │
    ├── single-attractor tasks (RTE, SST-2 core, Yelp, Amazon)
    │   └── always single_attractor regardless of backbone or training
    │
    └── multi-attractor-capable tasks (QNLI, MRPC, SST-2 under domain shift)
        │
        ├── Training depth gate
        │   └── insufficient training → single_attractor
        │       (Weak/Medium QNLI)
        │
        └── Backbone mechanism selector
            ├── compressed representation (DistilBERT)
            │   └── rotational_degeneracy
            │       (low-rank subspace → rotational freedom)
            │
            └── rich representation (RoBERTa)
                ├── feature-set switching
                │   (multiple independent PC subspaces → QNLI)
                │
                └── single_attractor (degeneracy collapse)
                    (dominant direction absorbs alternatives → MRPC)
```

This model makes a testable prediction: **on a deeper backbone like
DeBERTa, multi-attractor families should show either feature-set
switching or single-attractor, but not rotational degeneracy.** If
DeBERTa's disentangled attention creates a third mechanism variant,
the model needs revision.

---

## 6. Deliverables produced

| Deliverable | Path |
|------------|------|
| Determinant protocol | `sidecar/notes/n47_attractor_mechanism_determinants_protocol.md` |
| Determinant matrix (JSON) | `sidecar/results/attractor_mechanisms/determinant_matrix.json` |
| Family factor table (JSON) | `sidecar/results/attractor_mechanisms/family_factor_table.json` |
| Mechanism map figure | `sidecar/figures/attractor_mechanism_map.svg` |
| Determinant matrix figure | `sidecar/figures/attractor_mechanism_determinant_matrix.svg` |
| Convergence panel figure | `sidecar/figures/attractor_mechanism_convergence_panel.svg` |

---

## 7. Success criteria assessment

> **Positive outcome:** A structured determinant pattern emerges.

**Met.** The hierarchy task → backbone → convergence → domain is
structured and internally consistent. The hierarchy has clear
explanatory power for 12 of 14 panel entries. The two entries it
cannot fully explain (Medium QNLI's anomalous geometry, and the
QNLI/RoBERTa feature-set switching as potentially QNLI-specific)
are identified as limitations, not forced into the framework.

The backbone confound is the main caveat. The determinant hierarchy is
the best current explanation but it cannot distinguish backbone-as-cause
from task×backbone interaction until the panel is expanded.

---

## 8. What comes next

Stage C (mechanism-to-commensurability synthesis) should connect
these mechanism classes to the sidecar's broader theory of benign
diversity and merge interpretation. The key question: **does it matter
for commensurability whether diversity arises through rotational
degeneracy or feature-set switching?**

Stage D (GPU-conditional) should specifically target the backbone
confound by training DeBERTa adapters and classifying their mechanism.
