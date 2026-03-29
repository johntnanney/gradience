# n39 — Attractor Mapping Lab: Panel Definition

**Type:** panel definition
**Date:** 2026-03-28
**Depends on:** n34 (seed readout panel), n36 (coupling findings), n37 (conjunctive model update)
**Status:** Stage A deliverable. Defines the analyzable panel for the attractor mapping program.

---

## Purpose

This note defines the panel of task families available for readout attractor
mapping. The attractor mapping lab asks which task families admit multiple
benign readout attractors, which have one dominant attractor, and what
factors govern attractor selection.

The panel is derived from the seed-contingent readout program (n34–n36),
which already measured pairwise readout geometry for 14 same-task seed
pairs and 3 adjacent-task pairs. This note reorganizes that data from a
pair-level inventory into a family-level inventory suitable for attractor
classification.

---

## Panel composition

### Group A — Same-task multi-seed families (core)

These are the primary families for attractor classification. Each has
two seeds on at least one backbone, with full LoRA weights and classifier
heads available.

| Family | Backbone    | Seeds     | Pairs | Readout cos | Classification |
|--------|-------------|-----------|-------|-------------|----------------|
| QNLI   | DistilBERT | s42, s7   | 1     | 0.000       | orthogonal     |
| QNLI   | RoBERTa    | s42, s7   | 1     | −0.019      | orthogonal     |
| MRPC   | DistilBERT | s42, s7   | 1     | −0.059      | orthogonal     |
| MRPC   | RoBERTa    | s42, s7   | 1     | 0.9998      | aligned        |
| RTE    | DistilBERT | s42, s7   | 1     | 0.9978      | aligned        |
| RTE    | RoBERTa    | s42, s7   | 1     | 0.9998      | aligned        |
| SST-2  | DistilBERT | s42, s7   | 1     | 0.9975      | aligned        |
| SST-2  | RoBERTa    | s42, s7   | 1     | 0.9998      | aligned        |

**8 family×backbone entries, 8 seed pairs.**

### Group B — Same-label / adjacent-domain families

These families test whether shared label structure (binary sentiment)
implies shared attractor structure. All on DistilBERT.

| Family         | Seeds   | Pairs | Readout cos | Classification |
|----------------|---------|-------|-------------|----------------|
| SST-2 (domain) | s42, s7 | 1     | 0.040       | orthogonal     |
| Yelp           | s42, s7 | 1     | 0.9978      | aligned        |
| Amazon         | s42, s7 | 1     | 0.9979      | aligned        |

**Cross-domain pairs (adjacent-task, not same-task):**

| Pair            | Readout cos | Classification | Merge Δ |
|-----------------|-------------|----------------|---------|
| SST-2 × Yelp   | 0.012       | orthogonal     | 1.4%    |
| SST-2 × Amazon  | 0.009       | orthogonal     | 1.0%    |
| Yelp × Amazon   | 0.995       | aligned        | 0.0%    |

**3 family entries, 3 seed pairs + 3 cross-domain pairs.**

### Group C — Backbone contrast families

These families appear in Group A on both backbones, enabling direct
comparison of attractor structure across architectures.

| Family | DistilBERT cos | RoBERTa cos | Structure change     |
|--------|----------------|-------------|----------------------|
| QNLI   | 0.000          | −0.019      | orthogonal → orthogonal (stable multi-attractor) |
| MRPC   | −0.059         | 0.9998      | orthogonal → aligned (backbone-contingent)       |
| RTE    | 0.9978         | 0.9998      | aligned → aligned (stable single-attractor)      |
| SST-2  | 0.9975         | 0.9998      | aligned → aligned (stable single-attractor)      |

**4 families with backbone contrast. MRPC is the backbone-contingent case.**

### Group D — Convergence / strength contrast

Three QNLI bands trained on the same backbone (DistilBERT) with the
same seeds (s42, s7) but different training duration/strength.

| Family       | Readout cos | Classification | Merge Δ |
|--------------|-------------|----------------|---------|
| Strong QNLI  | 0.052       | orthogonal     | 0.4%    |
| Medium QNLI  | 0.999       | aligned        | 1.4%    |
| Weak QNLI    | 0.999       | aligned        | 0.2%    |

**3 family entries. Strong band is multi-attractor; Medium and Weak
are single-attractor. This is the convergence-contingent case.**

---

## Family-level artifact inventory

| Family          | Backbone    | Group | Seeds | Source adapters | Readout weights | Predictions | Merge outcomes | Strength metadata | Analyzable class              |
|-----------------|-------------|-------|-------|-----------------|-----------------|-------------|----------------|-------------------|-------------------------------|
| QNLI            | DistilBERT  | A, C  | 2     | yes             | yes             | no          | yes            | no                | multi_backbone_family         |
| QNLI            | RoBERTa     | A, C  | 2     | yes             | yes             | no          | yes            | no                | multi_backbone_family         |
| MRPC            | DistilBERT  | A, C  | 2     | yes             | yes             | no          | yes            | no                | multi_backbone_family         |
| MRPC            | RoBERTa     | A, C  | 2     | yes             | yes             | no          | yes            | no                | multi_backbone_family         |
| RTE             | DistilBERT  | A, C  | 2     | yes             | yes             | no          | yes            | no                | multi_backbone_family         |
| RTE             | RoBERTa     | A, C  | 2     | yes             | yes             | no          | yes            | no                | multi_backbone_family         |
| SST-2           | DistilBERT  | A, C  | 2     | yes             | yes             | no          | yes            | no                | multi_backbone_family         |
| SST-2           | RoBERTa     | A, C  | 2     | yes             | yes             | no          | yes            | no                | multi_backbone_family         |
| SST-2 (domain)  | DistilBERT  | B     | 2     | yes             | yes             | no          | yes            | no                | domain_contrast_family        |
| Yelp            | DistilBERT  | B     | 2     | yes             | yes             | no          | yes            | no                | domain_contrast_family        |
| Amazon          | DistilBERT  | B     | 2     | yes             | yes             | no          | yes            | no                | domain_contrast_family        |
| Strong QNLI    | DistilBERT  | D     | 2     | yes             | yes             | no          | yes            | yes               | strength_contrast_family      |
| Medium QNLI    | DistilBERT  | D     | 2     | yes             | yes             | no          | yes            | yes               | strength_contrast_family      |
| Weak QNLI      | DistilBERT  | D     | 2     | yes             | yes             | no          | yes            | yes               | strength_contrast_family      |

**14 family×backbone entries. All analyzable at readout_plus_merge level.
None blocked. No predictions/logits available (Metric 3 from the spec
is not computable; noted as a limitation).**

---

## Success criteria assessment

The spec requires:

1. **At least 3 same-task seed families analyzable:** Yes — QNLI, MRPC, RTE,
   SST-2 on two backbones each (8 family×backbone entries). ✓
2. **At least 1 backbone contrast family analyzable:** Yes — all 4 core tasks
   appear on both backbones. MRPC is the backbone-contingent case. ✓
3. **At least 1 convergence/strength contrast family analyzable:** Yes —
   Strong/Medium/Weak QNLI. ✓
4. **At least 1 same-label domain contrast family analyzable:** Yes —
   SST-2 (domain) / Yelp / Amazon block. ✓

**All four success criteria are met. Stage A is complete.**

---

## Preliminary attractor classification

Based on the readout cosine data already available from n36, the
following preliminary family-level attractor classifications can be
stated. These will be confirmed or revised in Stage B.

### Single-attractor families

| Family | Evidence | Confidence |
|--------|----------|------------|
| RTE (both backbones) | cos > 0.997 on both | high |
| SST-2 core (both backbones) | cos > 0.997 on both | high |
| MRPC on RoBERTa | cos = 0.9998 | high |
| Yelp | cos = 0.998 | high |
| Amazon | cos = 0.998 | high |
| Medium QNLI | cos = 0.999 | high |
| Weak QNLI | cos = 0.999 | high |

### Multi-attractor families

| Family | Evidence | Confidence |
|--------|----------|------------|
| QNLI (both backbones) | cos ≈ 0 on both | high |
| MRPC on DistilBERT | cos = −0.059 | high |
| SST-2 (domain shift) | cos = 0.040 | high |
| Strong QNLI | cos = 0.052 | high |

### Key patterns visible before Stage B

**Pattern 1 — Task identity is the primary determinant.** QNLI is
multi-attractor everywhere it appears. RTE and core SST-2 are
single-attractor everywhere.

**Pattern 2 — Backbone contingency is real but rare.** Only MRPC
changes attractor class across backbones (orthogonal on DistilBERT,
aligned on RoBERTa). Three of four core tasks are backbone-stable.

**Pattern 3 — Convergence contingency is sharp.** Strong QNLI is
multi-attractor; Medium and Weak are single-attractor. Same task,
same backbone, same seeds. The transition is associated with training
depth, not task identity.

**Pattern 4 — Domain shift creates multi-attractor structure in
otherwise single-attractor tasks.** Core SST-2 is single-attractor
on DistilBERT. Domain-shift SST-2 on the same backbone with the
same seeds is multi-attractor. The training distribution matters.

**Pattern 5 — Adjacent-domain readout tracks attractor structure,
not task label.** SST-2 (domain) is orthogonal to Yelp and Amazon
despite shared binary sentiment labels. Yelp and Amazon are aligned
with each other. Attractor structure is not determined by label
format alone.

---

## Scope and limitations

1. **Two seeds per family.** With only one pairwise cosine per
   family×backbone, "single-attractor" vs "multi-attractor" is
   inferred from a single data point. Three or more seeds would
   allow distribution analysis. This is a fundamental limitation
   of the current evidence base.

2. **No prediction overlap.** Metric 3 (example-level agreement
   across seeds) is not computable — no stored predictions/logits.
   This means the lab cannot assess whether different attractor
   basins produce behaviorally similar or different outputs.

3. **No intermediate cosine values.** The bimodal distribution
   (all values cluster at ≈0 or ≈1) means the classification is
   unambiguous but also means there are no borderline cases to
   study. The attractor landscape appears discrete, not continuous.

4. **DistilBERT only for Groups B and D.** Domain-shift and
   source-strength families exist only on DistilBERT. Backbone
   contrast for these groups is not available.

---

## Relationship to prior notes

| Note | Relationship |
|------|-------------|
| n34  | Panel definition for seed-contingent readout; this note reorganizes n34's pairs into families |
| n36  | Source data — all readout cosines and V-module metrics used here come from n36 |
| n37  | Conjunctive model update; this lab maps the readout-attractor side of the conjunction |
| n38  | Ruled-out summary; attractor mapping builds on the falsification of readout-as-risk-marker |
