# n33 — Conjunctive Mechanism Synthesis

**Type:** synthesis
**Date:** 2026-03-27
**Related notes:** n25 (multiscale mechanism synthesis), n32 (output-space findings)
**Supersedes:** n25 §3 Rung 3 (which was hypothesized; now resolved)
**Status:** Current best statement of the complete mechanism model.

---

## Purpose

n25 integrated Sidecar A's three-scale findings into a mechanism ladder
with two established rungs and one hypothesized rung. Sidecar B (n32)
has now resolved the third rung — and the answer was not what the
hypothesis predicted. This note integrates the full picture: what the
conjunctive model is, why it takes the form it does, what it explains,
and what single question it leaves open.

---

## 1. The conjunctive model

Catastrophic cross-task interference requires the conjunction of two
independently measurable conditions:

**Condition 1 — Representation-space pathology (Sidecar A).** The
V-module dimensionality ratio between paired adapters falls below
~0.75, indicating that one adapter uses a substantially higher-
dimensional subspace of the value projection than the other. This
creates asymmetric effective rank in the merged V weights, distorting
the attention value channel. Cohen's d = 3.36 between catastrophic and
safe collision pairs, zero range overlap on backbone-controlled
comparison. Established on DistilBERT and RoBERTa (n21).

**Condition 2 — Readout incompatibility (Sidecar B).** The classifier
heads learned by the two adapters have near-orthogonal decision axes
(decision_axis_cos ≈ 0, angle ≈ 89°). The merged classifier's decision
boundary is rotated into a "neither-task" region of representation
space, unable to cleanly classify inputs for either original task.
Detectable from classifier weight cosine similarity alone. Established
on the 11-case output-space panel (n32).

**The conjunction:** Either condition alone is insufficient.

- Condition 1 without Condition 2: the merged model has upstream
  pathology but the compatible readout absorbs it. The classifier
  finds a valid decision boundary despite distorted representations.
  Example: no such case exists in the current panel, but the logical
  structure follows from SC-QMRB (incompatible readout, no upstream
  pathology → safe).

- Condition 2 without Condition 1: the merged model has incompatible
  readout but no upstream pathology. The representation space is
  well-behaved enough that the classifier's confusion does not produce
  catastrophic errors — or the merged representations fall in a region
  where the rotated boundary still makes reasonable decisions.
  Example: **SC-QMRB** (QNLI × MRPC on RoBERTa, decision_axis_cos =
  −0.019, angle = 88.9°, yet Δ = 1.7%).

- Both conditions satisfied: the merged model has distorted
  representations AND no valid readout direction. Performance collapses.
  Example: **CA-01** (QNLI × MRPC on DistilBERT, V dim ratio in
  catastrophic range, decision_axis_cos = 0.015, Δ = 41.7%).

---

## 2. What each sidecar contributed

### Sidecar A: the upstream risk signal

Sidecar A's contribution was identifying *where* in the network the
catastrophe-specific signal lives. The path through aggregate (n18,
negative), per-module (n21, positive), and per-head (n24, mixed-
positive) analyses established:

- The V-module dimensionality ratio is the strongest group-level
  discriminator (Rung 1, d = 3.36).
- Head-level cancellation explains seed sensitivity within a risk class
  (Rung 2, resolving CA-01's 29pp gap).
- The mechanism is localized to specific attention components (V, K),
  not distributed across the aggregate subspace.

Sidecar A's picture was incomplete because it could not explain why
SC-QMRB — the same pair as CA-01 on a different backbone — was safe.
The module-level metrics on RoBERTa presumably lack the V-module
pathology (the dim ratio falls in the safe range), but this had not been
explicitly tested against the output-space hypothesis.

### Sidecar B: the readout gate

Sidecar B's contribution was testing the downstream end of the causal
chain. The original hypothesis (n31) was that the readout layer
*amplifies* upstream incompatibility — that orthogonal decision axes
would magnify small representation-space distortions into large
classification errors.

The data falsified amplification and supported gating:

- **SC-QMRB falsifier.** Identical readout geometry to CA-01
  (orthogonal axes, ~0.70 margin proxy, ~89° angle), yet safe. If the
  readout were an amplifier, it should have amplified whatever small
  upstream distortion exists on RoBERTa. It did not.

- **Compatible readout predicts safety.** All 3 cases with aligned
  decision axes (cos > 0.95) are safe/mild, regardless of upstream
  conditions. Compatible readout appears to absorb upstream risk.

- **Seed contrast shows zero readout variance.** CA-01-catastrophic
  and CA-01-mild have virtually identical readout geometry despite a
  29pp performance gap. The readout layer contributes nothing to the
  seed-sensitive modulation — that is entirely a Rung 2 phenomenon.

The readout layer is a **gate**: it determines whether upstream
pathology is *transmitted* (incompatible readout, no valid boundary to
catch the distortion) or *absorbed* (compatible readout, the classifier
finds a boundary that works despite upstream distortion).

---

## 3. The complete mechanism ladder (updated)

```
Rung   What it measures            Status        Key number
─────────────────────────────────────────────────────────────
 1     V-module dim ratio          ESTABLISHED   d = 3.36
       (group discrimination)      (2 backbones)

 2     Head-level dim ratio        ESTABLISHED   7 hot heads,
       distribution                (2 backbones) max |Δ_DR| = 0.229
       (seed modulation)

 3     Readout decision-axis       RESOLVED      gate, not amplifier;
       compatibility               (MIXED)       cos > 0.95 → safe
       (output gating)
```

**How the rungs interact:**

Rung 1 is a necessary precondition: if V-module geometry is in the safe
range, the pair does not catastrophically fail regardless of readout
compatibility. Rung 2 modulates whether a Rung-1-positive pair actually
reaches the catastrophic threshold in a given seed configuration. Rung 3
gates whether the upstream pathology manifests as classification errors:
compatible readout absorbs the pathology, incompatible readout transmits
it.

The previous hypothesis (n25 §3) placed the O-module and classification
head together under "downstream amplification." The data now split them:
the classification head (readout) is a gate (Rung 3, resolved), while
the O-module's role in head-level weighting (which heads' incompati-
bilities dominate the output) remains open and is properly a sub-
mechanism of Rung 2, not a separate rung.

---

## 4. The remaining open question

One question emerged from Sidecar B that was not anticipated by the
original mechanism ladder: **readout compatibility is seed-contingent**.

The CA-02 contrast demonstrates this sharply:

| Variant | Seeds | decision_axis_cos | Outcome |
|---------|-------|------------------:|---------|
| CA-02-toxic | qnli_s42 × sst2_s7 | −0.020 | catastrophic (27.2%) |
| CA-02-benign | qnli_s7 × sst2_s42 | 0.999 | safe (1.0%) |

Same nominal task pair (QNLI × SST-2), same backbone (RoBERTa),
completely different readout geometry. The classifier head learned by
qnli_s42 happens to use a representation direction orthogonal to the
one used by sst2_s7. But qnli_s7 and sst2_s42 converge on nearly
parallel directions.

This means the decision axis is not a fixed property of the
classification task. It is a contingent outcome of training — which
region of the representation space the classifier head "chooses" to
exploit. Different random seeds land in different subspaces, and some
subspace combinations are compatible while others are orthogonal.

This has a direct structural implication for the conjunctive model:
**Condition 2 (readout incompatibility) is not deterministically
predictable from the task pair.** It is a stochastic property of the
training run. For some task pairs (QNLI × MRPC), readout is structurally
orthogonal regardless of seed or backbone — these are "always-gated"
pairs where catastrophe depends solely on Condition 1. For others
(QNLI × SST-2), readout compatibility varies with seed — these are
"conditionally-gated" pairs where Condition 2 itself is stochastic.

### What determines the decision axis?

This is the best remaining CPU-feasible question. The classifier head
is a linear projection (or affine + ReLU + linear for DistilBERT) from
representation space to class logits. The decision axis is the
difference between class weight vectors: `d = W[0] - W[1]`. The
orientation of `d` in 768-dimensional space depends on which features
the classifier found useful during fine-tuning.

Two hypotheses:

1. **Representation-space attractors.** The pretrained backbone has a
   small number of "natural" directions for binary classification tasks.
   Some seeds converge to one attractor, others to another. Compatible
   seeds find the same attractor; incompatible seeds find different
   ones. This would predict that readout compatibility clusters
   discretely (bimodal: either ~0 or ~1), which is exactly what the
   data shows.

2. **Task structure interaction.** Some tasks have inherently more
   constrained decision geometry (fewer viable axes), while others have
   redundant viable directions. QNLI × MRPC is always orthogonal
   because QNLI and MRPC genuinely require different representational
   features. QNLI × SST-2 is variable because both tasks can be solved
   by multiple feature sets, and seed determines which feature set the
   classifier exploits.

These are testable with existing adapter weights. Clustering the
decision axes of all 16 source adapters (4 tasks × 2 seeds × 2
backbones) in representation space would reveal whether the
"attractor" or "task structure" model better explains the bimodal
compatibility pattern. This analysis requires only the classifier
weight matrices already available in the safetensors files.

---

## 5. What the conjunctive model means for Gradience

### For the current stable stack

Nothing changes operationally. The task-relationship advisory already
separates same-task (safe) from cross-task (caution). The conjunctive
model provides a mechanistic explanation for why this binary is reliable
— same-task pairs share readout geometry and lack V-module pathology —
but does not alter the user-facing recommendation.

### For a future predictive system

A pair-level risk predictor would need to check both conditions:

1. **V-module dim ratio** (Condition 1): computable from adapter LoRA
   weights without merging or inference. Already the strongest signal
   in the sidecar (d = 3.36). Awaiting DeBERTa confirmation.

2. **Decision-axis cosine** (Condition 2): computable from classifier
   head weights (the `modules_to_save` component of LoRA adapters).
   Trivially cheap — just a cosine between two vectors. But the
   seed-contingency finding means this metric is volatile: the same
   task pair can be compatible or incompatible depending on training
   seed.

The conjunctive structure means a two-check system would have very
high specificity: both conditions must be met for a catastrophe
prediction. The seed-contingency of Condition 2 means that pair-level
risk assessment must be adapter-specific, not merely task-pair-specific
— you cannot say "QNLI × SST-2 is dangerous" without checking the
specific adapters' decision axes.

### For the DeBERTa adjudication

The DeBERTa protocol (n07) already tests Condition 1 (Prediction D:
V-module dim ratio portability). The conjunctive model adds a natural
extension: check whether DeBERTa adapters' readout geometry follows the
same bimodal pattern (structurally orthogonal vs. contingently
orthogonal). If DeBERTa's disentangled attention architecture produces
systematically different readout geometry, that would be informative
about whether the gating mechanism is architecturally generic.

---

## 6. Updated summary table

| Question | Where answered | Status |
|:---------|:---------------|:-------|
| Which pairs are catastrophic? | n21 (V dim ratio) | Resolved — Condition 1 |
| Why does CA-01 show 29pp seed gap? | n24 (head cancellation) | Resolved — Rung 2 |
| Does readout amplify upstream risk? | n32 (SC-QMRB falsifier) | **No** — gate, not amplifier |
| Is readout incompatibility sufficient? | n32 (SC-QMRB) | **No** — necessary, not sufficient |
| Is readout compatibility protective? | n32 (3/3 aligned = safe) | **Yes** — sufficient for safety |
| Why do same-pair backbones differ? | This note | Conjunctive model: different C1 status |
| Is readout task-determined? | n32 (CA-02 contrast) | **Partially** — seed-contingent for some pairs |
| What determines decision axis orientation? | Open | Best remaining CPU question |
| Does the model survive DeBERTa? | n07 protocol | Blocked on GPU |

---

## 7. Relationship to n25

This note supersedes n25's Rung 3 ("downstream amplification,
hypothesized") with a resolved empirical finding ("readout gating,
mixed signal"). The rest of n25 remains valid:

- §1.1 (aggregate within-layer, negative): unchanged.
- §1.2 (per-module, positive): unchanged, now Condition 1.
- §1.3 (per-head, mixed-positive): unchanged, now Rung 2.
- §2 (integrated picture): valid but now incomplete — does not include
  the readout gate.
- §3 (mechanism ladder): Rungs 1–2 unchanged. Rung 3 is now resolved
  differently from the hypothesis.
- §4.1 (amplification mechanism): reframed. The O-module head-weighting
  question remains open but is now understood as part of Rung 2 (which
  heads' incompatibilities dominate), not a separate Rung 3.
- §4.2–4.5: unchanged.

The primary reader should now use n25 for the Rung 1–2 detail and this
note (n33) for the complete model including Rung 3 and the conjunctive
structure.
