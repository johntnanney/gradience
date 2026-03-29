# n42 — On the Topology of Readout Solution Spaces

**Type:** synthesis
**Date:** 2026-03-28
**Depends on:** n41 (attractor mapping findings), n37 (conjunctive model update), n38 (ruled out)
**Status:** Interpretive synthesis. Bridges the attractor mapping lab and the sidecar's larger theoretical structure.

---

## The claim

The readout layer of a fine-tuned classifier does not learn a single
canonical solution. It learns one of several possible solutions drawn
from a structured space whose topology varies by task, backbone, and
training regime. This topology is now empirically characterized for
10 task families. It is descriptive — it maps what the optimization
landscape looks like — not predictive of merge risk. But it sharpens
the sidecar's central concept of commensurability by decomposing it
into two independently characterizable conditions.

---

## 1. Three classes of readout solution space

The attractor mapping lab (n39–n41) classifies each task family's
readout solution space into one of three topological types.

### Single-attractor families

The optimization landscape has one dominant basin. Different seeds,
given the same task and backbone, converge to essentially the same
decision axis (cosine > 0.95). The readout solution is reproducible.

**Confirmed instances:** RTE (both backbones), SST-2 core (both
backbones), Yelp, Amazon, Medium QNLI, Weak QNLI.

The intuitive picture: the task has one natural decision boundary
in the pretrained model's representation space, and gradient descent
finds it reliably regardless of initialization. There is no ambiguity
about which features to use for classification.

### Multi-attractor families

The landscape has multiple distinct basins separated by large angular
distances (cosine ≈ 0, i.e. ~90°). Different seeds land in different
basins. Both basins produce classifiers that perform well on the task —
the solutions are functionally equivalent but geometrically orthogonal.

**Confirmed instances:** QNLI (both backbones), SST-2 (domain-shift
variant), Strong QNLI.

The intuitive picture: the task admits more than one set of features
that are independently sufficient for classification. The pretrained
model encodes both feature sets, and which one a given training run
exploits depends on the random initialization of the classifier head
and the early trajectory of gradient descent. The resulting classifiers
agree on most inputs but arrive at their answers via different
representational routes.

The critical empirical finding: **multi-attractor structure is benign.**
All multi-attractor families merge safely (max Δ = 2.2%). Orthogonal
readout between seeds is not pathology — it is the normal signature of
a rich solution space.

### Contingent families

The topological type itself changes across conditions. A family may
be single-attractor on one backbone and multi-attractor on another,
or single-attractor at one training depth and multi-attractor at
another.

**Confirmed instances:**

- MRPC: single-attractor on RoBERTa, multi-attractor on DistilBERT
  (**backbone-contingent**).
- QNLI across strength bands: single-attractor at Medium/Weak training,
  multi-attractor at Strong training (**convergence-contingent**).
- SST-2 across training regimes: single-attractor under standard
  training, multi-attractor under domain-shift training
  (**regime-contingent**).

Contingency is the most theoretically interesting class. It says that
the number of viable solutions is not fixed by the task alone — it
depends on the interaction between the task, the backbone's
representational geometry, and the training process. The topology of
the solution space is itself a variable, not a constant.

---

## 2. What determines attractor structure

The evidence supports a hierarchy of influence:

**Task identity is primary.** QNLI is multi-attractor on every
backbone and at every strength band where sufficient training is
provided. RTE is single-attractor everywhere. The task's intrinsic
structure — how many independent feature sets are sufficient for
classification — sets the baseline attractor count.

**Training regime is secondary.** SST-2 is single-attractor under
standard training but multi-attractor under domain-shift training.
Strong QNLI is multi-attractor where Medium/Weak are not. Longer or
more varied training can open access to attractor basins that shorter
or more constrained training does not reach.

**Backbone architecture is tertiary.** MRPC changes attractor class
across backbones — multi-attractor on the 6-layer DistilBERT,
single-attractor on the 12-layer RoBERTa. The richer representation
space of the larger model may collapse multiple viable decision
boundaries into one, or make one so much more accessible that gradient
descent reliably finds it.

This hierarchy is tentative — it is derived from 10 families, not 100 —
but the ordering is consistent across all available contrasts.

---

## 3. Why this is descriptive, not predictive

The attractor topology tells you what the readout landscape looks like.
It does not tell you whether a merge will fail.

This is not a limitation of the current evidence; it is a structural
property of the conjunctive model. The readout gate is one of two
independently necessary conditions for catastrophe. Whether the gate
is open (multi-attractor, orthogonal readout between the adapters
being merged) or closed (single-attractor, aligned readout) determines
whether upstream pathology is transmitted to the output. But:

- An open gate with healthy upstream geometry produces a safe merge.
  (All multi-attractor same-task families demonstrate this.)
- A closed gate with upstream pathology also produces a safe merge.
  (The gate absorbs the damage.)
- Only an open gate *with* upstream pathology produces catastrophe.

So knowing the topology tells you the state of one gate. It does not
tell you the state of the other condition. This is why attractor
structure is descriptive — it characterizes the readout landscape —
but not predictive — it cannot by itself forecast merge outcomes.

A standalone "readout compatibility metric" would false-alarm on
roughly 40% of same-task merges (the multi-attractor families). That
is why such a metric was ruled out in n38.

---

## 4. How this sharpens commensurability

The sidecar's concept of **commensurability** (glossary) is the
conjunction of upstream V-module compatibility and readout
compatibility. Before the attractor mapping lab, commensurability was
a conjunction of two conditions whose individual structure was
understood asymmetrically: the upstream condition had been carefully
analyzed (V-module dimensionality ratio, head-level modulation), but
the readout condition was characterized only negatively — we knew what
it was not (not a risk marker, not coupled to upstream geometry), not
what it was.

The attractor topology gives the readout condition positive content.
Commensurability now decomposes as:

**Upstream condition:** Do the two adapters' V-module representations
occupy compatible subspaces? Measured by dimensionality ratio (d=3.36
separation between catastrophic and safe). This is the condition that
carries the discriminative weight.

**Readout condition:** Do the two adapters' classifier heads use
the same decision axis? Determined by the topology of the readout
solution space: single-attractor families always satisfy this
condition; multi-attractor families may or may not, depending on
which basin each adapter landed in. This is the gate condition — it
does not generate risk, but it determines whether upstream risk is
transmitted.

The sharpening is this: the readout condition is no longer a black
box. It has internal structure. Whether two adapters' readout heads
are compatible is not random — it is a function of the task family's
attractor topology, which is itself a function of task identity,
training regime, and backbone architecture, in that order of
influence. The gate has a logic.

---

## 5. Implications for future work

### For the sidecar

The attractor topology opens Stage C of the mapping lab: analyzing
the 768-dimensional decision axes themselves to understand *why*
certain tasks admit multiple attractors. The hypothesis is that
multi-attractor tasks have multiple independent feature sets that
are each sufficient for classification, and the pretrained model
encodes all of them. This is testable by examining which principal
components of the pretrained representation the different attractors
project onto.

### For core Gradience (eventual)

If promoted, the attractor topology would enter the Gradience stack
not as a risk predictor but as a **context signal** — analogous to
how boundary detection provides context without predicting severity.
A user merging two QNLI adapters would see: "This task family admits
multiple readout attractors. Your adapters may use different decision
axes. This is normal and does not indicate risk unless upstream
representation incompatibility is also present."

That framing — descriptive, contextual, explicitly not alarming —
is the right way to surface this information. Anything stronger would
produce false alarms.

### For the conjunctive model

The topology completes the descriptive side of the model. The two
conditions for catastrophe are now both positively characterized:

- Upstream: V-module dimensionality ratio < threshold (Rung 1).
- Readout: multi-attractor family with adapters in different basins
  (Rung 3, gate condition).

What remains uncharacterized is the *interaction* — why certain
combinations of open gate + upstream pathology are catastrophic while
others are merely moderate. The answer likely involves the specific
angular relationship between the readout axis difference and the
direction of upstream pathology, but this is Stage C/D territory
and may require GPU inference to test.

---

## 6. What this note does not claim

- It does not claim that attractor topology causes merge outcomes.
  Topology is a description of the solution space, not a causal force.
- It does not claim that multi-attractor families are more dangerous
  than single-attractor families. The evidence shows the opposite
  tendency (all multi-attractor same-task families are safe).
- It does not claim that the topology is fully characterized. Two
  seeds per family gives one pairwise cosine — enough for the
  bimodal classification but not enough for basin counting or density
  estimation.
- It does not claim that the hierarchy of influence (task > regime >
  backbone) is fixed. Ten families is sufficient for the classification
  but not for the ranking.
