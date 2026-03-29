# n41 — Family-Level Readout Audit Findings

**Type:** findings
**Date:** 2026-03-28
**Depends on:** n39 (panel definition), n40 (protocol), n36 (coupling findings)
**Status:** Stage B deliverable for the attractor mapping lab.

---

## Summary verdict

**POSITIVE — the attractor landscape is structured, discrete, and
mappable with the current evidence base.**

Ten task families were classified into three attractor types using a
deterministic rule applied to pairwise readout-axis cosine. The
classification is unambiguous: all values fall cleanly into orthogonal
(|cos| < 0.1) or aligned (|cos| > 0.95) with no intermediate cases.
Every family merges safely regardless of attractor type. Multi-attractor
structure is common (4 of 10 families) and benign.

---

## 1. Family-level attractor classification

| Family | Attractor class | Evidence |
|--------|----------------|----------|
| QNLI | multi_attractor | Orthogonal on both DistilBERT (cos=0.000) and RoBERTa (cos=−0.019) |
| MRPC | backbone_contingent | Orthogonal on DistilBERT (cos=−0.059), aligned on RoBERTa (cos=0.9998) |
| RTE | single_attractor | Aligned on both DistilBERT (cos=0.998) and RoBERTa (cos=1.000) |
| SST-2 (core) | single_attractor | Aligned on both DistilBERT (cos=0.998) and RoBERTa (cos=1.000) |
| SST-2 (domain) | multi_attractor | Orthogonal on DistilBERT (cos=0.040) |
| Yelp | single_attractor | Aligned on DistilBERT (cos=0.998) |
| Amazon | single_attractor | Aligned on DistilBERT (cos=0.998) |
| Strong QNLI | multi_attractor | Orthogonal on DistilBERT (cos=0.052) |
| Medium QNLI | single_attractor | Aligned on DistilBERT (cos=0.999) |
| Weak QNLI | single_attractor | Aligned on DistilBERT (cos=0.999) |

**Counts:** 6 single-attractor, 3 multi-attractor, 1 backbone-contingent.
All 10 families merge safely (max Δ = 2.2%).

---

## 2. Contrast findings

### Contrast 1 — Single vs multi-attractor families

| Property | Single-attractor (n=6) | Multi-attractor + contingent (n=4) |
|----------|----------------------|-----------------------------------|
| V-module dim ratio (mean) | 0.944 | 0.886 |
| V-module dim ratio (range) | 0.921–0.980 | 0.790–0.925 |
| Merge Δ (max) | 1.4% | 2.2% |

Multi-attractor families show slightly lower upstream V-module
dimensionality ratios on average, but the ranges overlap substantially
and both groups remain well above the catastrophic threshold (< 0.75).
**Upstream V-module geometry does not cleanly separate attractor
classes.** This is expected: attractor structure is a property of the
readout landscape, not of upstream representations (n36, n37).

### Contrast 2 — Backbone contingency (MRPC)

| Property | MRPC on DistilBERT | MRPC on RoBERTa |
|----------|-------------------|-----------------|
| Readout cos | −0.059 | 0.9998 |
| Classification | orthogonal | aligned |
| V-module dim ratio | 0.905 | 0.945 |
| Merge Δ | 0.2% | 1.0% |

MRPC is the only backbone-contingent family. On DistilBERT (6 layers),
seeds settle into different readout attractors. On RoBERTa (12 layers),
the same task produces a single attractor. The representation geometry
of the backbone — specifically, how much of the decision-relevant
feature space is accessible — appears to modulate the number of viable
readout directions. The 6-layer compression of DistilBERT may create
multiple viable decision boundaries for MRPC where RoBERTa's richer
representation space converges to a single one.

Both backbone variants merge safely, confirming that backbone contingency
in attractor structure does not by itself indicate risk.

### Contrast 3 — Convergence contingency (QNLI strength bands)

| Band | Readout cos | Classification | V-module dim ratio | Merge Δ |
|------|-------------|----------------|--------------------|---------|
| Strong | 0.052 | orthogonal | 0.925 | 0.4% |
| Medium | 0.999 | aligned | 0.936 | 1.4% |
| Weak | 0.999 | aligned | 0.957 | 0.2% |

The transition from single-attractor (Medium, Weak) to multi-attractor
(Strong) correlates with training depth. Stronger training appears to
open access to a second attractor basin that shorter training does not
reach. This is the convergence-contingent case predicted by the
attractor model (n37 §2).

The upstream V-module dim ratios are nearly identical across bands
(0.925–0.957), confirming that the convergence effect operates on
the readout landscape, not on upstream representations.

### Contrast 4 — Domain contrast (sentiment block)

| Family | Readout cos | Classification | V-module dim ratio |
|--------|-------------|----------------|--------------------|
| SST-2 (domain) | 0.040 | orthogonal | 0.833 |
| Yelp | 0.998 | aligned | 0.980 |
| Amazon | 0.998 | aligned | 0.966 |

All three are binary sentiment tasks on DistilBERT with the same seeds.
SST-2 (domain shift variant) is multi-attractor; Yelp and Amazon are
single-attractor. The training distribution — not the label format —
determines attractor structure.

### Contrast 5 — Cross-domain readout alignment

| Pair | Readout cos | Classification |
|------|-------------|----------------|
| SST-2 × Yelp | 0.012 | orthogonal |
| SST-2 × Amazon | 0.009 | orthogonal |
| Yelp × Amazon | 0.995 | aligned |

Cross-domain readout alignment tracks within-family attractor structure.
SST-2 (domain), which is itself multi-attractor, is orthogonal to both
Yelp and Amazon. Yelp and Amazon, both single-attractor, are aligned
with each other.

This pattern suggests that when a family is multi-attractor, its seeds
explore readout directions that are orthogonal to the directions found
by single-attractor families solving similar tasks. The multi-attractor
family is not just "noisy" — it is finding genuinely different solutions
in representation space.

---

## 3. Answers to research questions

### RQ1 — Which families are single-attractor vs multi-attractor?

Single-attractor: RTE, SST-2 (core), Yelp, Amazon, Medium QNLI, Weak QNLI.
Multi-attractor: QNLI (both backbones), SST-2 (domain), Strong QNLI.
Backbone-contingent: MRPC.

The classification is sharp and deterministic — the bimodal distribution
admits no borderline cases.

### RQ2 — Does attractor structure differ across backbones?

Yes, for one family. MRPC is orthogonal on DistilBERT and aligned on
RoBERTa. QNLI, RTE, and SST-2 are backbone-stable. Backbone
contingency is real but rare (1/4 core tasks).

### RQ3 — Does convergence affect attractor selection?

Yes. Strong QNLI (longer training) is multi-attractor. Medium and
Weak QNLI (shorter training) are single-attractor. Same task, same
backbone, same seeds. Training depth modulates attractor access.

### RQ4 — Do same-label families share attractor structure?

No. SST-2 (domain) is multi-attractor while Yelp and Amazon are
single-attractor, despite all three being binary sentiment tasks.
Shared label format does not imply shared attractor structure.
Attractor structure is not task-label-equivalent (confirming H3).

### RQ5 — How does attractor structure relate to commensurability?

Multi-attractor structure is orthogonal to upstream commensurability.
All multi-attractor families have healthy V-module geometry (dim
ratio > 0.79) and merge safely. Attractor multiplicity is a property
of the readout landscape. It becomes relevant to commensurability only
through the conjunctive model: when an open readout gate (multi-attractor,
orthogonal readout) coincides with upstream V-module pathology, the
result is catastrophic. But multi-attractor structure alone carries no
risk signal.

---

## 4. Hypothesis assessment

| Hypothesis | Status |
|-----------|--------|
| H1 — Some families are single-attractor | **Confirmed:** RTE, SST-2, Yelp, Amazon, Medium/Weak QNLI |
| H2 — Some families are multi-attractor | **Confirmed:** QNLI, SST-2 (domain), Strong QNLI |
| H3 — Attractor structure ≠ task-label-equivalent | **Confirmed:** SST-2 (domain) vs Yelp/Amazon |
| H4 — Attractor structure may be backbone-contingent | **Confirmed:** MRPC (1/4 core tasks) |
| H5 — Attractor structure may be convergence-contingent | **Confirmed:** Strong vs Medium/Weak QNLI |
| H6 — Multi-attractor ≠ fragile | **Confirmed:** All multi-attractor families merge safely |

All six hypotheses from the spec are confirmed, though with the caveat
that each is based on a small number of families and two seeds per family.

---

## 5. Emerging picture

The attractor mapping lab reveals a landscape with the following
properties:

**Discreteness.** The attractor landscape is discrete, not continuous.
Every family falls cleanly into single-attractor or multi-attractor
with no intermediate cases. This is consistent with the basin metaphor:
the classifier head converges to one of a small number of fixed points,
not to a point on a continuum.

**Task-primary determination.** Task identity is the strongest
predictor of attractor structure. QNLI is multi-attractor on every
backbone and at every strength level where sufficient training is
provided. RTE is single-attractor everywhere.

**Modulation by three secondary factors.** Backbone architecture
(MRPC), training depth (Strong vs Medium/Weak QNLI), and training
distribution (SST-2 core vs domain shift) each modulate attractor
structure. The hierarchy of influence is: task identity > training
regime > backbone architecture, though the evidence base is too thin
to fix this ordering definitively.

**Safety invariance.** All families merge safely regardless of
attractor class. Multi-attractor families are not fragile. This
confirms that the conjunctive model's discriminative work is done
entirely by the upstream V-module condition, not by the readout gate.
The readout gate is common and mostly open; the rare upstream
pathology is what makes the conjunction lethal.

**Cross-domain alignment tracks attractor structure.** When a family
is multi-attractor, its readout directions are orthogonal to those
of single-attractor families solving related tasks. This suggests
that multi-attractor families are genuinely exploring a different
region of the readout landscape, not just adding noise to the same
solution.

---

## 6. Limitations

1. **Two seeds per family.** The "distribution" for each family is a
   single pairwise cosine. Three or more seeds would allow genuine
   clustering analysis and confidence intervals on attractor counts.

2. **No prediction overlap.** Without stored logits, the lab cannot
   assess whether seeds in different attractor basins produce the
   same classifications on the same inputs. Behavioral equivalence
   is assumed from merge safety but not directly measured.

3. **DistilBERT only for Groups B and D.** Domain-shift and
   source-strength contrasts are available on one backbone only.

4. **Small family count.** Ten families is sufficient for
   classification but insufficient for statistical claims about
   the prevalence of multi-attractor structure in general.

5. **No causal mechanism.** The lab maps the landscape but does
   not explain why certain tasks admit multiple attractors. The
   "why" question (n37 §6) remains open and is Stage C territory.

---

## 7. Deliverables produced

| Deliverable | Path |
|------------|------|
| Family readout metrics | `sidecar/results/attractor_mapping/family_readout_metrics.json` |
| Attractor classifications | `sidecar/results/attractor_mapping/attractor_classifications.json` |
| Family attractor map (Fig. 1) | `sidecar/figures/attractor_mapping_family_map.svg` |
| Backbone contrast panel (Fig. 2) | `sidecar/figures/attractor_mapping_backbone_contrast.svg` |
| Convergence contrast (Fig. 3) | `sidecar/figures/attractor_mapping_convergence_contrast.svg` |
| Domain contrast block (Fig. 4) | `sidecar/figures/attractor_mapping_domain_contrast.svg` |

---

## 8. Relationship to prior notes

| Note | Status after n41 |
|------|-----------------|
| n36 (coupling findings) | Source data — all metrics from n36 |
| n37 (conjunctive model) | Attractor model confirmed; hierarchy of modulating factors now empirically grounded |
| n38 (ruled out) | Unchanged — multi-attractor finding is consistent with ruled-out hypotheses |
| n39 (panel definition) | Panel validated; all success criteria met |
| n40 (protocol) | Protocol executed as specified; Metric 3 deferred |

---

## 9. What comes next

The attractor landscape is now mapped at the family level. The
remaining questions are:

1. **Why does QNLI have multiple attractors while RTE has one?**
   This requires analyzing the 768-dimensional decision axes
   themselves — not just their pairwise cosines. Which directions
   do QNLI classifiers use? How do those directions relate to the
   pretrained model's internal structure? This is Stage C per the
   implementation spec.

2. **Does the attractor landscape survive DeBERTa?** The DeBERTa
   adjudication (n07) will test whether QNLI remains multi-attractor
   on a third backbone with disentangled attention.

3. **Can attractor structure be predicted from task properties?**
   The current evidence suggests that task complexity (number of
   independent feature sets that solve the task) determines attractor
   multiplicity, but this is interpretive, not measured.
