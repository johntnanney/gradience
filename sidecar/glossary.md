# Glossary — Canonical Terms

These definitions are frozen as of March 2026. Use these exact terms in studies, notes, and analysis scripts. Do not introduce synonyms or casual variants.

---

## boundary

The binary classification of an adapter pair as **same-task** or **cross-task**, based on the task identities of the two adapters.

Boundary detection is a core Gradience capability. It is reliably detected with 0 false positives across 2 backbones and 53 same-task pairs. Same-task pairs are safe to merge (max degradation 3.4%); cross-task pairs require further assessment.

Boundary is the outermost layer of the sidecar's three-level framework. It is settled and not under investigation.

**Use for:** the same-task / cross-task distinction.
**Do not use for:** any finer-grained risk classification within the cross-task regime.

---

## severity

The magnitude of performance degradation observed after merging two adapters, measured as the worst-case delta (maximum absolute accuracy drop) across the tasks involved.

Severity is a valid measurement within a single evaluation condition — one backbone, one seed combination. It is **not** a property of the task pair itself: severity rankings reverse across backbones (QNLI×MRPC: 41.7% on DistilBERT, 1.7% on RoBERTa).

Severity is backbone-local. It is the second layer of the three-level framework — useful for characterizing a specific experimental result, but not portable.

**Use for:** reporting the outcome of a specific merge evaluation.
**Do not use for:** ranking or comparing pairs across different backbones or seed conditions.
**Do not conflate with:** instability (which measures the variability of severity).

---

## instability

The variability of a pair's severity across seeds (within-backbone) and across backbones (between-backbone). Operationalized as a composite score:

    instability = 0.4 × normalized_seed_range + 0.3 × normalized_backbone_shift + 0.3 × normalized_cv

where each component is normalized to [0, 1] across the cross-task pair set.

Instability is the sidecar's working concept. It is the first candidate for a **portable descriptor** — a signal that generalizes across backbones. The instability ranking separates into two clusters (backbone-reversal > 0.7, stable-asymmetric < 0.3) with a clean gap that no pair currently occupies.

Instability is not yet confirmed as portable. It requires DeBERTa-v3 adjudication (S01).

**Use for:** ranking pairs by cross-condition variability; stratifying study designs by stability class.
**Do not use for:** predicting the severity of a specific merge (instability measures variability, not magnitude).

---

## catastrophic anchor

A **(task pair × backbone)** combination that produces a worst-case delta exceeding 15% (the catastrophic threshold defined in P01).

A catastrophic anchor is not a task pair — it is a task pair on a specific backbone. QNLI×MRPC is a catastrophic anchor on DistilBERT but not on RoBERTa. QNLI×SST-2 is a catastrophic anchor on RoBERTa but not on DistilBERT.

The "anchor" metaphor reflects the role these cases play in the sidecar's experimental design: they are the fixed reference points against which other pairs and conditions are compared.

**Use for:** referring to specific (pair × backbone) cases that exhibit catastrophic degradation.
**Do not use:** "catastrophic pair" without specifying the backbone, except when the backbone is unambiguous from context.

---

## portable descriptor

A merge-risk signal whose ranking generalizes across backbones.

No signal tested to date is a confirmed portable descriptor. Boundary detection is trivially portable (it depends only on task identity). Severity is explicitly not portable (rankings reverse). Instability is the first *candidate* portable descriptor — its ranking is consistent across both tested backbones, but confirmation on a third backbone is pending.

**Use for:** the property a signal must have to be promotable from sidecar to core.
**Do not use:** as a synonym for "useful signal" — a signal can be useful within one backbone without being portable.

---

## thresholded subspace interference

The working hypothesis that catastrophic merge failures require specific geometric conditions in the learned adapter subspaces to trigger, rather than being a smooth function of task dissimilarity.

Evidence: the co-occurrence of high severity and high instability in the same pairs. Unstable pairs show seed ranges of 25–29 percentage points — a single seed change can move a pair from 1% degradation to 42%. This is consistent with a threshold mechanism: below a critical subspace alignment, interference is moderate and predictable; above it, interference becomes catastrophic.

The best current operationalization of "specific geometric conditions" is **V-module dimensionality mismatch** (see below). Per-layer collision (n15) provides the necessary precondition; V-module geometry (n21) provides the strongest correlate of the catastrophic threshold itself. Head-level analysis (n24) further reveals that the threshold involves the *distribution* of dimensionality across attention heads: different seeds produce different head-level configurations that average to the same module-level value but have different functional impacts, depending on which heads' incompatibilities are amplified by the O module.

**Use for:** the mechanistic explanation the sidecar is pursuing.
**Do not use:** as a confirmed finding. It is a hypothesis, not a result.
**Do not shorten to:** "subspace interference" when the threshold character is the operative claim.

---

## multiscale mechanism ladder

*Added March 2026. Integrates n21, n24, and n25.*

The empirically grounded hierarchy of geometric conditions behind catastrophic merge failure, organized by the spatial scale at which each condition operates:

**Rung 1 — Module-level risk (n21).** V-module dimensionality ratio separates catastrophic from safe collision pairs (d=3.36, zero overlap). This is a *group discriminator*: it identifies which pairs are at risk. Operates on the full (768, 768) V-module product matrix. Aggregation across 12 heads *improves* discrimination by averaging out head-level noise while preserving a consistent directional signal.

**Rung 2 — Head-level modulation (n24).** The same module-level dim ratio can produce catastrophe or not, depending on how mismatch distributes across attention heads. Different training seeds produce different head-level configurations that average to the same module value. Opposite-sign head-level deltas cancel at module level, explaining why seed sensitivity (e.g., CA-01's 29-point gap) is invisible at Rung 1 but visible at Rung 2. Head-level geometry is a *modulator of severity within a risk class*, not a discriminator between classes.

**Rung 3 — Readout gating (n32, n36; resolved as mixed).** The classifier readout layer functions as a **gate**, not an amplifier. When readout geometry is compatible (decision axes aligned, cos > 0.95), upstream V-module pathology is absorbed — the merged classifier finds a valid decision boundary despite representation-space conflict. When readout is incompatible (orthogonal, cos ≈ 0), upstream pathology is transmitted. Readout orthogonality is common (5/14 same-task seed pairs, ~4/7 cross-task pairs) and harmless on its own — it becomes dangerous only when combined with Rung 1 V-module pathology. The two conditions are independently determined: readout axis selection is decoupled from upstream V-module geometry (n36). See also: **readout attractor**.

The ladder is nested, not competing: Rung 1 sets the precondition, Rung 2 modulates the outcome, Rung 3 gates whether the upstream pathology manifests as catastrophic classification errors. The mechanism is **conjunctive**: catastrophe requires both V-module pathology (Rung 1) and readout incompatibility (Rung 3).

**Use for:** the sidecar's integrated mechanistic picture; framing the DeBERTa adjudication (each rung is a separate testable claim).
**Do not use:** as a confirmed causal chain. Rungs 1–2 are correlational; Rung 3 is empirically resolved but on two backbones only.
**Do not conflate rungs:** module-level is for classification, head-level is for seed sensitivity explanation, readout gating is for determining whether upstream pathology manifests. Applying the wrong rung to the wrong question produces misleading conclusions.

---

## V-module dimensionality mismatch

*Frozen as of March 2026. Pending DeBERTa-v3 confirmation before promotion.*

**Canonical finding:** In collision-prone cross-task pairs, catastrophic outcomes are associated with strong dimensionality mismatch in V-module geometry, even when aggregate within-layer geometry is non-discriminating.

**Canonical interpretation:** Catastrophic merges involve structurally incommensurable value-projection perturbations: one adapter's is concentrated, the other's diffuse, and the linear merge smears features that remain commensurable in safe collision pairs.

**Operational definition:** The V-module dimensionality ratio is min(eff_rank_a, eff_rank_b) / max(eff_rank_a, eff_rank_b), computed on the per-module LoRA product W = lora_B @ lora_A for the value projection at critical layers (those carrying ≥ 60% of combined norm mass).

**Evidence:** On the backbone-controlled comparison (CA-02 vs. safe collision controls, all RoBERTa), V-module dimensionality ratio separates the groups with Cohen's d = 3.36 and zero range overlap. Catastrophic range: 0.64–0.74. Safe collision range: 0.79–0.89. No other metric in the sidecar program achieves this separation. (See n21 §2, §4.)

**Head-level refinement (n24):** The module-level dimensionality ratio is the optimal aggregation for group discrimination (d=3.36). Head-level decomposition weakens discrimination (d=1.25) because it introduces variance. However, head-level analysis resolves the CA-01 seed sensitivity mystery: opposite-sign head-level deltas (up to |Δ_DR| = 0.229) cancel when averaged, producing near-zero module-level deltas despite large individual head differences. The module-level signal reflects cross-head consistency; seed sensitivity reflects cross-head reconfiguration.

**Scope and caveats:**

1. Two-backbone result only. The signal must survive DeBERTa-v3 replication before any promotion claim.
2. Derived from one catastrophic case on RoBERTa (CA-02). May not generalize to all catastrophic collision pairs.
3. Distinguishes catastrophic collision from safe collision. Does not distinguish catastrophic from non-collision (Group 3 also shows low dimensionality ratios).
4. Correlational, not causal. The causal chain from V-module geometry to performance degradation is interpretive.
5. Module-level is the right granularity for discrimination; head-level is the right granularity for explaining seed sensitivity. Do not expect head-level metrics to outperform module-level for group classification.

**Use for:** the strongest current correlate of catastrophic threshold within the collision regime; the primary signal to test in DeBERTa replication.
**Do not use:** as a confirmed predictive metric. It is a two-backbone correlational finding, not a validated predictor.
**Do not shorten to:** "dimensionality mismatch" without specifying the V module — other modules show weaker or non-significant versions of the same pattern.

---

## readout incompatibility

*Updated March 2026. Refined by n32, n36, n37.*

The condition where two adapters' classifier heads have near-orthogonal decision axes (decision_axis_cos < 0.1, angle > 85°). The decision axis is `d = W[0] − W[1]`, the difference between the two class weight vectors in the final classification layer.

Readout incompatibility is **not a risk marker on its own**. It is common in LoRA fine-tuning: 5 of 14 same-task seed pairs show orthogonal decision axes, yet all merge safely (Δ ≤ 2.2%). It also appears in cross-task pairs that are safe (SC-QMRB, NC-RMDB) and mild (CA-01-mild). Readout incompatibility becomes dangerous only in conjunction with V-module pathology (Rung 1 of the mechanism ladder) — it is the gate condition in the conjunctive model (n33, n37).

The distribution of decision-axis cosine is **bimodal**: values cluster at ≈0 (orthogonal) or ≈1 (aligned) with no intermediate values across 17 tested pairs. There is no "partially incompatible" regime.

**Use for:** the gate condition in the conjunctive model; the Rung 3 component of the mechanism ladder.
**Do not use as:** a stand-alone risk metric. A system that flags readout incompatibility alone would produce false positives on ~40% of same-task merges.
**Do not conflate with:** task incompatibility. Two seeds of the same task can have incompatible readout.

---

## readout attractor

*Added March 2026. Based on n36, n37.*

A stable direction in representation space toward which the classifier head converges during LoRA fine-tuning. Some tasks have a single attractor (all seeds converge to the same decision axis); others have multiple orthogonal attractors (different seeds converge to different axes).

**Empirical characterization:**

- **Single-attractor tasks** (RTE, SST-2, Yelp, Amazon, Medium/Weak QNLI): All seeds produce aligned readout (cos > 0.99). The classifier has no meaningful choice.
- **Multi-attractor tasks** (QNLI, MRPC on DistilBERT, Strong QNLI, SST-2 in domain-shift): Seeds produce orthogonal readout (cos ≈ 0). The classifier settles into one of several viable directions.
- **Backbone-contingent tasks** (MRPC): Single attractor on RoBERTa, multiple on DistilBERT. The backbone's representation geometry influences how many viable readout directions exist.

The attractor landscape is modulated by three factors: **task identity** (primary), **backbone architecture** (secondary), and **training convergence** (tertiary — longer training may explore more of the landscape). The bimodal distribution (no intermediate cosine values) supports a discrete attractor model over continuous variation.

**Use for:** explaining why readout incompatibility is seed-contingent; characterizing the readout-axis-selection landscape for a given task.
**Do not use as:** a predictor of merge risk. Attractor multiplicity is a property of the readout landscape, not of upstream risk.

---

## commensurability

*Added March 2026. Conceptual term for the full program.*

The degree to which two adapters' learned internal solutions — both upstream representations and downstream readout — are compatible under linear merge. Two adapters are commensurable if their merged output performs comparably to either source; incommensurable if merging produces catastrophic degradation.

Commensurability is the conjunction of two independently measurable conditions:

1. **Upstream commensurability:** The adapters' V-module perturbations have compatible dimensionality structure (dim ratio > 0.75). Measured from LoRA weight products.
2. **Readout commensurability:** The adapters' classifier heads use compatible decision axes (cos > 0.95). Measured from classifier weight vectors.

Same-task pairs are always upstream-commensurable (dim ratio > 0.78) but may be readout-incommensurable (5/14 cases). This is harmless. Cross-task pairs may be upstream-incommensurable (V-module pathology), and if they are also readout-incommensurable, the result is catastrophic.

**Use for:** the overarching concept that integrates the conjunctive model; framing the question of whether a given pair of adapters can be safely merged.
**Do not use as:** a single number. Commensurability is a conjunction of independently measured conditions, not a scalar.
**Do not conflate with:** task similarity. Two adapters solving the same task can be readout-incommensurable. Two adapters solving different tasks can be fully commensurable.

---

## rotational degeneracy

*Added March 2026. Based on n44, n46.*

The mechanism by which multi-attractor structure arises when the task's relevant features span a low-rank subspace within which the decision axis is free to rotate without affecting classification accuracy. Different seeds settle into different orientations within the same shared subspace. The axes are orthogonal in representation space but pass through the same low-dimensional PC subspace.

**Geometric signature:** Effective axis cosine < 0.80, energy overlap > 0.85, PC effective rank < 4.0 for both seeds.

**Confirmed instances:** QNLI/DistilBERT, SST-2(domain)/DistilBERT, Strong QNLI/DistilBERT, MRPC/DistilBERT. All on DistilBERT (6-layer compressed representation).

**Use for:** classifying the mechanism behind multi-attractor readout diversity; the Rung 3a gate condition.
**Do not use as:** a synonym for "harmless." Rotational degeneracy is benign alone but can participate in catastrophic failure via the conjunctive model.

---

## feature-set switching

*Added March 2026. Based on n44, n46.*

The mechanism by which multi-attractor structure arises when the pretrained model's representation space encodes multiple independent feature sets sufficient for classification, and different seeds lock onto different feature sets. The decision axes occupy non-overlapping PC subspaces.

**Geometric signature:** Effective axis cosine < 0.80, energy overlap < 0.40, shared top-3 PCs = 0, PC effective rank > 5.0 for at least one seed.

**Confirmed instance:** QNLI/RoBERTa only. One QNLI/RoBERTa seed (s7) uses a decision direction aligned with RTE (cos = 0.86) — cross-task feature exploitation.

**Use for:** classifying the mechanism behind multi-attractor readout diversity; the Rung 3b gate condition.
**Do not use as:** a synonym for "dangerous." Feature-set switching is benign alone (QNLI/RoBERTa merges safely).

---

## mechanism determinant hierarchy

*Added March 2026. Based on n46–n49.*

The ordering of factors governing which attractor mechanism a multi-attractor family expresses: **task identity** (primary, determines whether multi-attractor structure is possible) → **backbone architecture** (secondary, selects which mechanism realizes multiplicity) → **training convergence** (tertiary, gates attractor count but not mechanism) → **domain structure** (weak, single contrast only).

Critically confounded in the current panel: all rotational degeneracy cases are on DistilBERT, all feature-set switching on RoBERTa. The DeBERTa adjudication will test whether this confound is causal or artifactual.

**Use for:** explaining why different families express different mechanisms; framing DeBERTa predictions.
**Do not use as:** a confirmed causal model. The hierarchy is the best current explanation but rests on a confounded two-backbone panel.
