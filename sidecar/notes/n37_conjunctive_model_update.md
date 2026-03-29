# n37 — Conjunctive Model Update: After Same-Task Readout Analysis

**Type:** synthesis (update)
**Date:** 2026-03-27
**Supersedes:** n33 §5.2 (attractor hypothesis — now confirmed)
**Depends on:** n33 (conjunctive model), n36 (coupling findings)
**Status:** Current best statement of the mechanism model, incorporating same-task evidence.

---

## What changed

n33 proposed the conjunctive model based on cross-task evidence:
catastrophe requires V-module pathology × readout incompatibility.
n36 tested the readout side of that conjunction against same-task
seed pairs and produced a result that both falsifies a simpler story
and replaces it with a sharper one.

---

## 1. Orthogonal readout is common and often safe

The simpler story was: orthogonal decision axes are a cross-task
pathology that reflects genuinely incompatible classification
objectives. The data falsify this. Five of fourteen same-task seed
pairs — seeds trained on the *identical* task — show orthogonal
readout (decision_axis_cos < 0.1, angle ≈ 87–90°). Every one of
them merges safely (Δ ≤ 2.2%).

This means orthogonal readout is not a marker of task incompatibility.
It is a routine outcome of classifier-head optimization in LoRA
fine-tuning. Two seeds solving exactly the same classification
problem can converge to orthogonal decision axes and still produce a
merged model that works.

**What this rules out:** Any framework that treats readout
orthogonality as intrinsically dangerous, or that uses decision-axis
cosine as a stand-alone risk metric.

**What this preserves:** The conjunctive model. Orthogonal readout is
the gate condition — it enables catastrophe when upstream pathology
is present, but it does not create risk on its own. A gate that is
routinely open is still a gate; it just means the discriminative work
is done by the other condition.

---

## 2. Readout attractors are task-, backbone-, and training-contingent

n33 §5.2 hypothesized two possible explanations for the bimodal
readout pattern: "representation-space attractors" or "task structure
interaction." The same-task data resolve this in favor of the
attractor model, with three modulating factors:

**Task identity:** QNLI always produces orthogonal readout across
seeds (both backbones). RTE and SST-2 always produce aligned readout.
Some tasks have one attractor, others have multiple.

**Backbone architecture:** MRPC is orthogonal on DistilBERT but
aligned on RoBERTa. The number of viable readout attractors for a
given task depends on the backbone's representation geometry.

**Training convergence:** Strong-band QNLI (longer training) is
orthogonal; Medium and Weak bands (shorter training) are aligned.
Same task, same backbone, same seeds. The degree of convergence
affects which attractor the classifier head settles into.

The absence of intermediate values (no decision-axis cosine between
0.05 and 0.997 across 17 pairs) confirms that the landscape is
discretely multimodal, not continuous. The classifier head either
finds the same direction or a genuinely different one — there is no
"partially different" regime.

---

## 3. Upstream V-module health and readout choice are decoupled

Every same-task seed pair has healthy upstream V-module geometry
(dim ratio > 0.78), regardless of whether its readout is aligned or
orthogonal. The readout axis is not following upstream structure.

This has a precise structural implication: the two conditions for
catastrophe operate through **independent mechanisms**.

- **Condition 1 (V-module pathology)** is a property of the LoRA
  weight products in representation space. It measures how the two
  adapters' learned perturbations interact geometrically in the value
  projection. Same-task pairs never trigger this condition because
  they learn similar representation-space modifications.

- **Condition 2 (readout incompatibility)** is a property of the
  classifier head's optimization trajectory. It measures which
  attractor the classifier settled into during fine-tuning. Same-task
  pairs can trigger this condition (QNLI regularly does) without
  consequence because Condition 1 is absent.

The independence means the conditions can be measured and reasoned
about separately. A future risk system does not need to model their
interaction — it needs to check both, and flag only when both are
present.

---

## 4. Catastrophe is conjunctive: the updated picture

The full model, incorporating both cross-task (n32, n33) and
same-task (n36) evidence:

```
                        Readout gate
                     ┌─────────────────┐
                     │                 │
                     │  aligned (cos≈1)│──→ SAFE regardless of upstream
                     │                 │    (gate closed, pathology absorbed)
                     │  orthogonal     │
                     │  (cos≈0)        │──→ gate open: outcome depends
                     │                 │    on upstream condition
                     └─────────────────┘
                              │
                    ┌─────────┴─────────┐
                    │                   │
              V-module healthy     V-module pathological
              (dim ratio > 0.75)  (dim ratio < 0.75)
                    │                   │
                  SAFE               CATASTROPHIC
           (5 same-task cases,    (CA-01, CA-02)
            SC-QMRB, NC-RMDB)
```

The gate metaphor is strengthened by the same-task evidence: the gate
is open in at least a third of all seed pairs (5/14 same-task + 4/7
cross-task with orthogonal readout), yet catastrophe occurs only when
the upstream condition is also satisfied. The gate is common, the
upstream pathology is rare, and the conjunction is what matters.

---

## 5. This is a major negative-positive result

The sidecar's empirical record now includes a clean example of the
most productive kind of finding: a result that falsifies a simpler
story and replaces it with a better one.

**What was falsified:**
- Readout orthogonality as a risk marker (n36: 5/14 same-task pairs
  are orthogonal yet safe)
- Readout-upstream coupling as a mechanism (n36: no correlation
  between V-module health and readout classification)
- The implicit assumption that orthogonal readout reflects task
  incompatibility (n36: same task, orthogonal readout)

**What replaced it:**
- A bimodal attractor model of readout-axis selection, with
  task/backbone/convergence as modulating factors
- A strengthened conjunctive model where the two conditions are
  demonstrably independent
- A sharpened next question: what determines the number and
  orientation of readout attractors for a given task?

**Why this is valuable:** Each falsification constrains the space of
viable mechanistic explanations. Before n36, one could have built a
risk system based on readout cosine alone. That system would have
flagged ~40% of all same-task seed pairs as "at risk" — a false-
positive rate that would make it useless. The conjunctive model avoids
this by requiring both conditions, and the independence finding means
each condition can be cheaply checked.

---

## 6. The sharpened next question

The remaining open question from n33 — "what determines readout axis
orientation?" — now has a partial answer (task identity, backbone,
convergence) and a sharpened residual:

> **Why does QNLI have multiple readout attractors while RTE has only
> one?**

This is a question about the geometry of the classification task in
representation space. QNLI (natural language inference with
entailment/not-entailment classes) may be solvable via multiple
independent feature sets, each corresponding to a different readout
direction. RTE (also NLI but with smaller, more constrained data) may
force all seeds toward a single viable feature set.

Testing this requires no new data — it requires analyzing the
structure of the 768-dimensional decision axes themselves across all
adapters. Which directions do QNLI classifiers use? Do they cluster
into a small number of discrete orientations? How do those
orientations relate to the pretrained model's internal structure?

This is Stage C territory per the implementation spec.

---

## 7. Relationship to prior notes

| Note | Status after n37 |
|------|-----------------|
| n25 (multiscale synthesis) | Rung 3 superseded by n33→n37 |
| n32 (readout findings) | Unchanged — cross-task evidence still valid |
| n33 (conjunctive model) | §1–4 valid; §5.2 attractor hypothesis now confirmed; §5.4 seed-contingency question partially answered |
| n36 (coupling findings) | Primary evidence for this update |

This note (n37) is now the current best statement of the complete
mechanism model. Read n33 for the original formulation, n36 for the
same-task evidence, and this note for the integration.
