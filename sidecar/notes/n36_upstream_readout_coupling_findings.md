# n36 — Upstream-Readout Coupling: Findings

**Type:** findings
**Program:** Seed-Contingent Readout-Axis Selection
**Stage:** B
**Date:** 2026-03-27
**Depends on:** n34 (panel), n35 (protocol)

---

## 1. Summary verdict

**Mixed signal — readout axis selection is strongly decoupled from
upstream V-module geometry.**

Same-task seed pairs show the same bimodal readout pattern seen in
cross-task pairs: some have near-perfectly aligned decision axes
(cos ≈ 1), others are near-orthogonal (cos ≈ 0). Critically, this
bimodal split occurs **independently of upstream V-module similarity**.
All same-task seed pairs have healthy upstream geometry (dim ratio >
0.78), yet 5 of 14 show orthogonal readout. The readout axis is not
following upstream structure — it is selected by a different process.

---

## 2. The bimodal readout finding

### 2.1 Same-task seeds show orthogonal readout

This is the most unexpected result. Five same-task seed families
produce orthogonal decision axes (cos < 0.1) between seeds of the
same task:

| Family | Backbone | DecAxis Cos | Angle | Merge Δ |
|--------|----------|------------:|------:|--------:|
| QNLI | DistilBERT | 0.000 | 90.0° | 2.2% |
| MRPC | DistilBERT | −0.059 | 86.6° | 0.2% |
| QNLI | RoBERTa | −0.019 | 88.9° | 0.0% |
| SST-2 (domain) | DistilBERT | 0.040 | 87.7° | 1.0% |
| Strong QNLI | DistilBERT | 0.052 | 87.0° | 0.4% |

Every one of these pairs merges safely (Δ ≤ 2.2%). Orthogonal
readout geometry in same-task merges produces negligible degradation.

### 2.2 The bimodal distribution

Across all 14 same-task seed pairs, the distribution is sharply
bimodal:

- **Orthogonal cluster** (5 pairs): cos in [−0.06, 0.05], angles
  86.6°–90.0°
- **Aligned cluster** (9 pairs): cos in [0.997, 1.000], angles
  1.1°–4.1°
- **No intermediate values.** The gap between 0.05 and 0.997 is
  completely empty.

This matches the cross-task pattern from Sidecar B (n32) exactly.
Decision-axis cosine is not a continuous variable — it clusters at
either ~0 or ~1. The bimodality is a property of how classifier heads
learn in representation space, not a property of task compatibility.

### 2.3 Same task, both patterns

The most striking cases are tasks that show different readout patterns
on different backbones:

| Task | DistilBERT | RoBERTa |
|------|-----------|---------|
| QNLI | orthogonal (0.000) | orthogonal (−0.019) |
| MRPC | orthogonal (−0.059) | aligned (0.9998) |
| RTE | aligned (0.9978) | aligned (0.9998) |
| SST-2 | aligned (0.9975) | aligned (0.9998) |

QNLI is always orthogonal across seeds, on both backbones. MRPC is
orthogonal on DistilBERT but aligned on RoBERTa. RTE and SST-2 are
always aligned. This means readout-axis stability is task-dependent
and backbone-modulated.

---

## 3. Decoupling from upstream geometry

### 3.1 Upstream V-module metrics are uniformly healthy

Every same-task seed pair has V-module dim ratio > 0.78:

| Family | Backbone | Readout class | V dim ratio | V top overlap |
|--------|----------|--------------|------------:|-------------:|
| QNLI | DistilBERT | orthogonal | 0.920 | 0.828 |
| MRPC | DistilBERT | orthogonal | 0.905 | 0.795 |
| RTE | DistilBERT | aligned | 0.913 | 0.747 |
| SST-2 | DistilBERT | aligned | 0.933 | 0.936 |
| QNLI | RoBERTa | orthogonal | 0.790 | 0.206 |
| MRPC | RoBERTa | aligned | 0.945 | 0.713 |
| RTE | RoBERTa | aligned | 0.921 | 0.481 |
| SST-2 | RoBERTa | aligned | 0.947 | 0.821 |

Orthogonal-readout pairs (QNLI, MRPC on DB) have V dim ratios
comparable to aligned-readout pairs (RTE, SST-2). The upstream
geometry provides no signal about readout axis compatibility.

### 3.2 Pre-classifier alignment tracks readout exactly

The pre-classifier subspace overlap perfectly mirrors the readout
classification: orthogonal-readout pairs have overlap ~0.003, aligned
pairs have overlap > 0.9. This means the pre-classifier and the
final readout layer co-rotate together — they are not independent.
The entire classification head (pre-classifier + readout) either
aligns across seeds or does not, as a unit.

### 3.3 Coupling verdict: decoupled

Readout axis selection is **not** tracking upstream V-module geometry.
The decision about which representational direction the classifier
exploits appears to be made by the classifier head itself during
fine-tuning, independently of the LoRA representation-space
modifications. This is consistent with the conjunctive model (n33)
but adds a refinement: the two conditions for catastrophe (V-module
pathology and readout incompatibility) are not just jointly necessary
— they are **independently determined**.

---

## 4. Contrast results

### Contrast 1 — Cross-family comparison

**Task-specific readout stability:**

| Task | Always orthogonal | Always aligned | Varies by backbone |
|------|:-----------------:|:--------------:|:------------------:|
| QNLI | ✓ (both backbones) | | |
| MRPC | | | ✓ (DB: ⊥, RB: ∥) |
| RTE | | ✓ (both backbones) | |
| SST-2 | | ✓ (both backbones) | |

QNLI consistently produces orthogonal readout across seeds regardless
of backbone. RTE and SST-2 consistently produce aligned readout. MRPC
is the only task that varies by backbone — orthogonal on DistilBERT,
aligned on RoBERTa.

**Interpretation:** Some tasks have a single natural decision axis
that seeds converge to (RTE, SST-2). Others have multiple viable axes
that seeds sample from (QNLI). MRPC is intermediate — the number of
viable axes may depend on the backbone's representation structure.

### Contrast 2 — Domain-shift: seed vs domain

| Comparison | DecAxis Cos | Category |
|------------|------------:|----------|
| SST-2 s42 vs s7 (same domain) | 0.040 | orthogonal |
| Yelp s42 vs s7 (same domain) | 0.998 | aligned |
| Amazon s42 vs s7 (same domain) | 0.998 | aligned |
| SST-2 vs Yelp (cross domain) | 0.012 | orthogonal |
| SST-2 vs Amazon (cross domain) | 0.009 | orthogonal |
| Yelp vs Amazon (cross domain) | 0.995 | aligned |

**Pattern:** Yelp and Amazon adapters converge to the same readout
axis across seeds and across domains. SST-2 readout is orthogonal to
both Yelp and Amazon — and also orthogonal to itself across seeds.
This means SST-2 (in the domain-shift study) is using a fundamentally
different readout direction than Yelp/Amazon, even though all three
are binary sentiment classifiers.

**Note:** SST-2 in the core study (cos = 0.998) is aligned, while
SST-2 in the domain-shift study (cos = 0.040) is orthogonal. These
are different adapters trained in different experimental contexts,
suggesting that experimental setup (not just task identity) influences
readout axis selection.

### Contrast 3 — Source-strength QNLI

| Band | DecAxis Cos | Category |
|------|------------:|----------|
| Strong s42 vs s7 | 0.052 | orthogonal |
| Medium s42 vs s7 | 0.999 | aligned |
| Weak s42 vs s7 | 0.999 | aligned |

**Pattern:** Strong-band QNLI adapters produce orthogonal readout
across seeds, while Medium and Weak bands produce aligned readout.
All three bands train on the same task with the same backbone and
same seeds — the only variable is training duration/convergence.

**Interpretation:** Training convergence affects readout axis
selection. Longer training (strong band) may allow the classifier to
explore and settle into a local optimum that differs across seeds,
while shorter training (medium/weak) may stay closer to initialization
and thus converge to similar directions. Alternatively, more capable
models may exploit more diverse representational features, while
weaker models are forced to use a more constrained feature set.

---

## 5. Theoretical implications

### 5.1 Readout orthogonality is a normal feature of LoRA fine-tuning

The most important implication: orthogonal decision axes between seeds
of the same task is **not pathological**. It is a routine outcome of
LoRA fine-tuning for at least some tasks (QNLI consistently, MRPC on
DistilBERT). Same-task pairs with orthogonal readout merge with
negligible degradation (≤ 2.2%).

This means orthogonal readout is far more common than Sidecar B
suggested — it appears even in the safest possible merge context
(same task, same backbone, different seed). The readout gate (n33
Rung 3) is "open" (incompatible readout) for many same-task pairs,
yet no pathology passes through because the upstream V-module
geometry is healthy.

### 5.2 Two regimes of classifier head learning

The bimodal distribution suggests that 768-dimensional representation
space has two qualitatively different regimes for linear classifiers:

**Single-attractor tasks** (RTE, SST-2, Yelp, Amazon, Medium/Weak
QNLI): There is effectively one viable decision direction. All seeds
converge to it regardless of initialization. The classifier head has
no meaningful choice.

**Multi-attractor tasks** (QNLI, MRPC on DistilBERT, Strong QNLI,
SST-2 in domain-shift): Multiple orthogonal directions are viable for
classification. Different seeds land on different attractors. The
classifier head's "choice" is contingent on the training trajectory.

The absence of intermediate values (no pairs with cos ≈ 0.5) supports
the attractor model over a continuous-variation model: the classifier
either finds the same direction or a completely different one.

### 5.3 The conjunctive model is strengthened

The decoupling finding strengthens the conjunctive model (n33) in a
specific way. The two conditions for catastrophe are not just jointly
necessary — they are independently determined by different mechanisms:

- **Condition 1 (V-module pathology):** determined by the interaction
  between two adapters' representation-space modifications. This is a
  pair-level property of the LoRA weight products.

- **Condition 2 (readout incompatibility):** determined by which
  attractor the classifier head converges to during training. This is
  an individual-adapter property that becomes a pair-level property
  only when two adapters are compared.

Because the conditions are independently determined, they can be
independently measured — and a future predictive system can check
them separately.

### 5.4 Why same-task orthogonal readout is harmless

When two seeds of the same task produce orthogonal decision axes but
the same-task merge is still safe, what absorbs the readout
incompatibility? Two possible mechanisms:

**Hypothesis A — Redundant margin.** The merged representation is
good enough for the task that even a rotated decision boundary finds
a valid separation. Same-task LoRA modifications are compatible in
representation space (high V dim ratio), so the merged
representations cluster well even though the classifier is confused
about which direction to use.

**Hypothesis B — Averaging covers both axes.** The merged classifier
(0.5 * W_a + 0.5 * W_b) inherits projections onto both orthogonal
directions, creating a new direction that still captures task-relevant
features even though it differs from either source.

These are testable with CPU inference on the merged model, which
would be Stage C work.

---

## 6. Updated mechanism ladder status

| Rung | Finding | Updated status |
|------|---------|---------------|
| 1. V-module dim ratio | Group discriminator (d=3.36) | Unchanged |
| 2. Head-level cancellation | Seed modulation | Unchanged |
| 3. Readout gating | Gate, not amplifier | **Refined: readout axis selection is decoupled from upstream geometry; bimodal across same-task seeds** |

New addition: the **attractor structure** of readout axis selection is
now characterized. Some tasks have one attractor (always-aligned
readout), others have multiple (seed-contingent readout). This is a
property of the task × backbone × training-convergence interaction,
not of the LoRA representation space.

---

## 7. Answers to research questions

### RQ1 — Within-family readout variation
**Answer:** Dramatic. 5 of 14 same-task seed pairs show orthogonal
readout (cos < 0.1). The variation is bimodal — either aligned or
orthogonal, nothing in between.

### RQ2 — Upstream-readout coupling
**Answer:** Decoupled. All same-task seed pairs have healthy upstream
V-module geometry (dim ratio > 0.78) regardless of readout
classification. Readout axis selection is not tracking upstream
representation structure.

### RQ3 — Task-family differences
**Answer:** Yes. QNLI always produces orthogonal readout across seeds.
RTE and SST-2 always produce aligned readout. MRPC varies by backbone.

### RQ4 — Domain vs seed readout variation
**Answer:** Task-specific. SST-2 (domain-shift) is orthogonal across
seeds, while Yelp and Amazon are aligned across seeds AND across
domains. Cross-domain variation (SST-2 vs Yelp) is orthogonal,
matching within-task seed variation for SST-2. Domain change does not
automatically produce readout divergence — it depends on the task.

---

## 8. Implications for Stage C

The findings sharpen the Stage C question. The original framing was
"does coupling explain merge fragility?" The data show that coupling
is weak — readout is decoupled from upstream. So the Stage C question
becomes:

> For cross-task pairs, does the **combination** of readout
> orthogonality (common and harmless on its own) with V-module
> pathology (the Sidecar A signal) explain fragility better than
> either alone?

This is essentially a re-test of the conjunctive model (n33) from
the seed-variation perspective. The new data adds: orthogonal readout
alone is clearly harmless (5 same-task cases prove this), so any
cross-task fragility attributed to readout must be contingent on
upstream pathology being present.

Additionally, the Strong vs Medium/Weak QNLI contrast opens a new
angle: does training convergence affect both V-module geometry AND
readout geometry in ways that jointly modulate merge risk? This would
be a three-way interaction: task × backbone × convergence.

---

## 9. Files produced

| File | Description |
|------|-------------|
| `sidecar/results/seed_readout/seed_panel_table.json` | Full panel (17 pairs × all metrics) |
| `sidecar/results/seed_readout/seed_panel_table.md` | Human-readable summary |
| `sidecar/results/seed_readout/coupling_metrics.json` | Coupling analysis and classifications |
| `sidecar/results/seed_readout/family_summary_table.json` | Per-family metric summary |
| `sidecar/figures/seed_readout_decision_axis.svg` | Same-task readout variation |
| `sidecar/figures/seed_readout_coupling_scatter.svg` | Upstream vs readout scatter |
| `sidecar/figures/seed_readout_cross_task_linkage.svg` | Cross-task fragility linkage |
