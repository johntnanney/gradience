# n43 — Attractor Origin Program: Why Do Some Tasks Admit Multiple Readout Attractors?

**Type:** research program statement
**Date:** 2026-03-28
**Depends on:** n41 (attractor mapping findings), n42 (readout solution topology)
**Status:** Defines a CPU-feasible research program. Not yet executed.

---

## The question

The attractor mapping lab (n41) established that some task families
converge to one readout solution (single-attractor) while others admit
multiple orthogonal solutions (multi-attractor). The topology note (n42)
organized this into a descriptive framework. What remains unexplained
is *why*.

Four specific sub-questions:

1. **What about QNLI makes it multi-attractor?** QNLI is the most
   consistently multi-attractor family — orthogonal on both DistilBERT
   and RoBERTa, at every training depth where sufficient convergence
   occurs.

2. **What about RTE makes it single-attractor?** RTE seeds converge
   to the same decision axis on both backbones. What constrains the
   solution space to one basin?

3. **Why does stronger training open attractor multiplicity in QNLI?**
   Strong QNLI is multi-attractor while Medium/Weak are single-attractor.
   Same task, same backbone, same seeds. Training depth alone changes
   the topology.

4. **Why does MRPC switch attractor class by backbone?** Multi-attractor
   on DistilBERT, single-attractor on RoBERTa. What property of the
   backbone's representation geometry determines the number of viable
   decision boundaries?

---

## Working hypothesis

**Feature plurality hypothesis.** Multi-attractor tasks are tasks for
which the pretrained model's representation space encodes multiple
independent feature sets, each sufficient for classification.
Single-attractor tasks have one dominant feature set.

The hypothesis predicts:

- QNLI (question–sentence entailment) should have multiple separable
  feature clusters in pretrained space — perhaps lexical overlap
  features, semantic similarity features, and syntactic structure
  features, any one of which suffices for a reasonable classifier.

- RTE (textual entailment, small dataset, 2.5k examples) should have
  a more constrained optimization landscape — perhaps the small dataset
  size forces the classifier to exploit only the most accessible
  feature direction, or perhaps textual entailment on short premise–
  hypothesis pairs has genuinely fewer independent feature sets in
  the pretrained representation.

- Stronger QNLI training opens multiplicity because longer training
  allows the classifier to explore beyond the most accessible feature
  direction and settle into a secondary basin that shorter training
  never reaches.

- MRPC switches by backbone because DistilBERT's compressed 6-layer
  representation creates a lower-dimensional manifold where multiple
  viable decision boundaries are geometrically distinct, while
  RoBERTa's 12-layer representation has enough dimensions that the
  multiple feature sets are more easily aligned or subsumed into one
  dominant direction.

---

## Alternative hypotheses

### A1 — Dataset size hypothesis
Multi-attractor structure correlates with dataset size. QNLI (~105k
examples) is multi-attractor; RTE (~2.5k) is single-attractor.
Larger datasets may provide more optimization paths.

**Problem:** SST-2 (~67k) is single-attractor despite being large.
Yelp and Amazon are single-attractor with large datasets. Dataset
size is neither necessary nor sufficient.

### A2 — Class balance hypothesis
Multi-attractor structure correlates with label distribution
imbalance or label ambiguity.

**Problem:** QNLI is roughly balanced (50/50). RTE is also roughly
balanced. Class balance does not obviously differ between the groups.

### A3 — Random initialization hypothesis
Attractor multiplicity is a consequence of the classifier head's
random initialization, not of the task's feature structure. Different
random initializations of the linear classifier push gradient descent
into different basins regardless of the underlying feature landscape.

**Problem:** If this were the case, we would expect multi-attractor
structure to be ubiquitous (since all tasks use random classifier
initialization). But most tasks are single-attractor. The hypothesis
does not explain the task-specificity.

### A4 — Pretrained representation anisotropy hypothesis
Multi-attractor tasks are those whose relevant features align with
multiple high-variance directions in the pretrained representation.
Single-attractor tasks have features concentrated along one principal
component.

**Testable prediction:** The principal components of the pretrained
model's [CLS] representations, projected onto the task's training
distribution, should show a more uniform spectrum for multi-attractor
tasks than for single-attractor tasks.

---

## Research design

### Stage A — Decision-axis analysis

**Objective:** Extract and compare the actual 768-dimensional decision
axes from each family's adapters. Determine whether multi-attractor
families' decision axes lie in different subspaces of the pretrained
representation.

**Method:**

For each adapter in the panel, extract the learned decision axis:
the difference vector between class-0 and class-1 rows of the
classifier weight matrix (for 2-class tasks, this is the direction
that determines classification). For adapters with a pre-classifier
layer, compose the pre-classifier and classifier to get the effective
decision direction in the penultimate representation space.

For each family:
- Compute the decision axes for both seeds on each backbone.
- Project each decision axis onto the top-k principal components of
  the pretrained model's representation at the [CLS] position (the
  pretrained model's natural coordinate system). The top-k PCs can be
  estimated from the pre-classifier weight matrix or from the backbone's
  final-layer LayerNorm parameters as a proxy.
- Measure which PCs each seed's decision axis loads onto.
- For multi-attractor families: do the two seeds' decision axes load
  onto *different* PCs? If so, the seeds are exploiting different
  feature directions in the pretrained space.
- For single-attractor families: do the two seeds load onto the
  *same* PCs? If so, the pretrained space offers only one viable
  direction.

**Deliverables:**
- `sidecar/scripts/per_layer/decision_axis_analysis.py`
- `sidecar/results/attractor_origin/decision_axis_projections.json`
- `sidecar/results/attractor_origin/pc_loading_profiles.json`
- `sidecar/figures/attractor_origin_pc_loadings.svg`
- `sidecar/notes/n44_decision_axis_analysis_findings.md`

**CPU-feasible?** Yes. Requires only the classifier and pre-classifier
weight matrices (already in safetensors) and a spectral decomposition
of those matrices. No model inference needed.

### Stage B — Representation geometry audit

**Objective:** Characterize the dimensionality and structure of the
pretrained representation space as seen by each task's classifiers.

**Method:**

The pre-classifier weight matrix (where it exists, i.e. DistilBERT
adapters) is a linear map from 768-dimensional [CLS] space to a
lower-dimensional pre-classification space. Its singular values
reveal how much of the pretrained representation's variance the
classifier pathway uses.

For each adapter:
- SVD of the pre-classifier weight matrix → singular value spectrum
- SVD of the classifier weight matrix → effective rank
- Composed product (classifier × pre-classifier) → effective
  decision subspace dimensionality

Compare:
- Multi-attractor families vs single-attractor families: do they
  differ in effective decision subspace dimensionality?
- MRPC on DistilBERT vs RoBERTa: does the effective dimensionality
  differ in a way that explains the attractor-structure shift?
- Strong vs Medium/Weak QNLI: does stronger training change the
  effective decision subspace?

**Deliverables:**
- `sidecar/results/attractor_origin/representation_geometry.json`
- `sidecar/figures/attractor_origin_sv_spectra.svg`
- `sidecar/notes/n45_representation_geometry_findings.md`

**CPU-feasible?** Yes. All operations are on weight matrices already
available in safetensors. No inference required.

### Stage C — Cross-family decision-axis alignment

**Objective:** Test whether multi-attractor families' alternative
basins correspond to feature directions used by *other* tasks'
classifiers.

**Method:**

Take the decision axes from all adapters across all families. Compute
an all-pairs cosine similarity matrix of decision axes. Ask:

- When QNLI seed A and QNLI seed B are orthogonal, is QNLI seed B's
  decision axis aligned with any *other* task's axis? (e.g., is one
  QNLI seed using a feature direction that resembles an RTE-like or
  MRPC-like decision boundary?)
- Do the alternative basins of multi-attractor families correspond to
  different task "strategies" that the pretrained model supports?

This directly tests the feature plurality hypothesis: if multi-attractor
families have seeds exploiting directions used by other tasks, then
the pretrained representation genuinely contains multiple task-relevant
feature sets.

**Deliverables:**
- `sidecar/results/attractor_origin/cross_family_axis_alignment.json`
- `sidecar/figures/attractor_origin_cross_family_heatmap.svg`
- `sidecar/notes/n46_cross_family_alignment_findings.md`

**CPU-feasible?** Yes. Same weight-matrix-only approach.

---

## Success criteria

### Stage A success
The decision-axis projection reveals a structural difference between
multi-attractor and single-attractor families in how their axes
distribute across the pretrained representation's principal components.
Specifically: multi-attractor families' two seeds should load onto
non-overlapping PC subsets, while single-attractor families' seeds
should load onto the same PCs.

### Stage B success
The representation geometry audit shows a measurable difference in
effective decision subspace dimensionality between multi-attractor
and single-attractor families, or between MRPC's two backbone
conditions.

### Stage C success
At least one multi-attractor family has a secondary basin whose
decision axis aligns with another task family's primary axis, or
with a recognizable semantic direction in the pretrained space.

### Program-level success
The program succeeds if it can state, with evidence:

> Multi-attractor tasks are multi-attractor *because* their pretrained
> representations encode multiple [specific, characterized] feature
> sets for classification, and gradient descent can reliably find
> more than one of them.

Or, if the feature plurality hypothesis is falsified:

> Multi-attractor structure arises from [identified alternative
> mechanism], not from feature plurality in the pretrained
> representation.

Either outcome advances the sidecar.

---

## Falsification conditions

The feature plurality hypothesis is falsified if:

1. Multi-attractor families' decision axes load onto the *same* PCs
   (just with different signs or magnitudes), suggesting rotational
   degeneracy rather than distinct feature exploitation.

2. The effective decision subspace dimensionality is the same for
   single-attractor and multi-attractor families.

3. Cross-family alignment shows no structure — alternative basins
   do not correspond to any recognizable feature direction.

If all three falsification conditions are met, the feature plurality
hypothesis is wrong and the alternative hypotheses (particularly A4,
pretrained representation anisotropy) should be pursued instead.

---

## Scope and constraints

- **CPU-only.** All stages operate on weight matrices from safetensors.
  No model inference, no forward passes, no GPU required.
- **No new adapter training.** The program uses the existing panel of
  14 family×backbone entries from the attractor mapping lab (n39).
- **DistilBERT pre-classifier available; RoBERTa pre-classifier
  structure differs.** DistilBERT has an explicit pre-classifier linear
  layer. RoBERTa's `classifier` head has `dense` → `dropout` →
  `out_proj`. Both provide the same kind of linear map. The script must
  handle the architectural difference.
- **Two seeds per family.** As with the attractor mapping lab, the
  fundamental limitation is coverage. The analysis can characterize
  the axes but cannot estimate basin density or attractor count beyond
  the binary aligned/orthogonal classification.

---

## Execution order

1. **Stage A first.** The decision-axis projections are the most
   directly informative and the simplest to compute. If they show
   clean separation, Stages B and C become confirmatory. If they
   show no separation, Stages B and C become the primary hypothesis
   test.

2. **Stage B second.** The representation geometry audit provides
   context for interpreting the Stage A results and directly tests
   the MRPC backbone-contingency and QNLI convergence-contingency
   sub-questions.

3. **Stage C third.** The cross-family alignment is the most
   interpretively rich but requires the Stage A axes as input.

---

## Relationship to prior notes

| Note | Relationship |
|------|-------------|
| n41 | Source data: attractor classifications and family-level metrics |
| n42 | Theoretical framework: three topological classes, hierarchy of influence |
| n37 | Conjunctive model: this program addresses the readout gate's internal logic |
| n36 | Raw readout cosines and upstream metrics |
| n38 | Ruled-out hypotheses: this program builds on what survived |
