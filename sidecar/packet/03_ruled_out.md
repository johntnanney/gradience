# n68 — Ruled-Out Mechanisms

**Type:** negative-results packet
**Date:** 2026-03-28
**Supersedes:** n38 (ruled-out hypotheses), n52 (ruled-out mechanisms packet)
**Status:** Definitive. Covers the full sidecar evidence base through the Output Example Semantics program.

---

## Purpose

This packet collects every hypothesis the sidecar program tested and rejected. Each entry names the proposal, the evidence that killed it, and what replaced it. The entries are ordered by theoretical importance — the most consequential eliminations first.

A research program that has ruled things out with evidence is more trustworthy than one that has merely proposed things. These are not failed experiments; they are constraints that forced the theoretical picture into its current shape. Everything that survived did so because everything listed here was eliminated.

---

## 1. Portable severity score

**Proposal:** Assign each task pair a scalar severity number that generalizes across backbones and seeds. Use it to rank merge risk.

**Evidence against:** Severity rankings reverse completely across backbones. QNLI×MRPC degrades 41.7% on DistilBERT and 1.7% on RoBERTa. QNLI×SST-2 degrades 30.3% on RoBERTa and 0.3% on DistilBERT. Six candidate severity signals were tested and all failed: task-pair identity, core-space shared basis, pair-risk label, format similarity, source-strength gap, reconstruction error. None predicts severity portably.

**Why it matters:** This was the obvious first move — if severity were portable, the entire mechanism investigation would have been unnecessary. Its failure forced the shift from "how bad" to "how fragile," which restructured the entire research program.

**Replaced by:** Instability — the variability of severity across conditions. Instability rankings are consistent across two backbones, with a clean gap between the unstable and stable clusters.

**Source:** S01, n03, n05, n06.

---

## 2. Task-pair catastrophe lookup

**Proposal:** Certain task pairs are inherently catastrophic. Build a lookup table mapping task pairs to risk levels.

**Evidence against:** No task pair is catastrophic on both tested backbones. QNLI×MRPC is catastrophic on DistilBERT but mild on RoBERTa. QNLI×SST-2 is catastrophic on RoBERTa but moderate on DistilBERT. The unit of catastrophe is not the task pair but the (task pair × backbone × seed) triple.

**Why it matters:** A lookup table would have been the simplest possible product contribution from the sidecar — a static list saying "don't merge these." Its failure means that merge risk cannot be resolved at the task level. It must be resolved at the representation level, which is geometrically richer and architecturally contingent.

**Replaced by:** The conjunctive model. Catastrophe requires specific geometric conditions (V-module pathology × readout incompatibility) that are backbone- and seed-dependent. No fixed property of the task pair determines the outcome.

**Source:** S01, n05.

---

## 3. Aggregate within-layer threshold variable

**Proposal:** Compare adapter weight matrices layer-by-layer using the concatenated Q/K/V/O product. Catastrophic pairs should show distinctive aggregate subspace geometry — different principal angles, different dimensionality, different directional alignment.

**Evidence against:** When backbone is controlled, catastrophic cases are indistinguishable from safe collision controls on all four metrics (principal angles, top-direction overlap, dimensionality ratio, directional conflict). The separation visible in raw data was a backbone confound: DistilBERT's 6-layer architecture forces tighter subspace alignment than RoBERTa's 12-layer architecture. Directional conflict was actually *reversed* — catastrophic pairs showed lower conflict, consistent with a "similarity without identity" mechanism rather than "opposition."

**Why it matters:** This was the first attempt to find the geometric threshold for catastrophe. Its failure taught the critical methodological lesson: the Q/K/V/O concatenation dilutes module-specific signals. The V-module carries the catastrophe-discriminating signal; Q and O carry noise. Averaging across all four modules erases the very signal that matters.

**Replaced by:** Per-module decomposition. Separating Q, K, V, O into independent analyses recovers a clean V-module dimensionality ratio signal (d = 3.36, zero range overlap) that the aggregate analysis had averaged away.

**Source:** n17, n18, n19, n20, n21.

---

## 4. Readout orthogonality as risk marker

**Proposal:** Near-orthogonal decision axes (cos ≈ 0) between two adapters' classifier heads indicate dangerous incompatibility. Use decision-axis cosine as a merge-risk signal.

**Evidence against:** Two independent falsifications. First, the SC-QMRB control: QNLI×MRPC on RoBERTa has identical readout geometry to catastrophic CA-01 on DistilBERT (~89° orthogonal, ~0.70 margin proxy) but is safe (Δ = 1.7%). Second, same-task seed analysis: 5 of 14 same-task seed pairs show orthogonal readout yet all merge safely (max Δ = 2.2%). A stand-alone readout-cosine metric would false-alarm on approximately 40% of same-task merges.

**Why it matters:** Decision-axis cosine is cheap to compute and intuitive — it seems like it *should* be a risk signal. Its failure is counterintuitive and forces a subtler understanding: readout orthogonality is common, bimodally distributed, decoupled from upstream geometry, and harmless in isolation. It is a structural property of the solution space, not a symptom of pathology.

**Replaced by:** Readout incompatibility as a gate condition in the conjunctive model. Orthogonal readout opens the gate that determines whether upstream V-module pathology reaches the output. Without upstream pathology, the open gate transmits a healthy signal.

**Source:** n32, n36, n37.

---

## 5. Readout-alone explanation of catastrophe

**Proposal:** Catastrophic failure originates in the readout layer. Incompatible classifier heads amplify small representation-space distortions into large classification errors. The readout layer is an amplifier.

**Evidence against:** The SC-QMRB falsifier is decisive. If readout were an amplifier, SC-QMRB should have amplified whatever small upstream distortion exists on RoBERTa for the QNLI×MRPC pair into a large classification error. It did not — Δ = 1.7%. The amplifier model also cannot explain why CA-01's catastrophic and mild seed variants have virtually identical readout geometry (decision_axis_cos: 0.015 vs -0.059) yet differ by 29 points of severity.

**Why it matters:** The amplifier model is the natural successor to the risk-marker model — if orthogonal readout is dangerous, presumably it is because it amplifies errors. Rejecting this model forces the reconceptualization of Rung 3 as a gate rather than an amplifier. The readout layer transmits or absorbs upstream pathology; it does not create pathology where none exists.

**Replaced by:** Readout gating (n32, n37). The readout is a filter, not a generator. Compatible readout absorbs upstream pathology (the gate is closed). Incompatible readout transmits it (the gate is open). The amplification, if any, occurs upstream in the V-module.

**Source:** n32, n36.

---

## 6. Feature plurality as universal attractor origin

**Proposal:** Multi-attractor readout structure arises because the pretrained representation offers multiple task-relevant feature sets. Different training seeds lock onto different features, producing orthogonal decision axes. This is the universal mechanism for readout diversity.

**Evidence against:** The simple feature-plurality hypothesis is partially falsified (n44). Most multi-attractor families' seeds use the *same* principal components but combine them in orthogonal angular orientations. They share a low-rank subspace and rotate freely within it — they do not lock onto different feature sets. Shared top-3 PCs and energy overlap are high for DistilBERT multi-attractor families, inconsistent with feature-set switching.

The exception is QNLI on RoBERTa, where the two seeds genuinely use different PCs (shared top-3 = 0, energy overlap = 0.255). One seed's decision direction aligns with RTE (cos = 0.86), confirming cross-task feature exploitation. This is real feature-set switching — but it is the only observed instance, and it is backbone-specific.

**Why it matters:** A universal feature-plurality mechanism would have implied a single check for multi-attractor structure: do the seeds load onto different principal components? Its partial falsification forced the recognition of two qualitatively distinct mechanisms with different geometric signatures and different failure semantics under conjunctive pathology. The structure of readout diversity is richer than a single hypothesis can capture.

**Replaced by:** Two distinct mechanisms, determined by a structured hierarchy. *Rotational degeneracy* (same features, different angular combinations; all observed instances on DistilBERT) and *feature-set switching* (genuinely different feature sets; QNLI on RoBERTa only). The mechanism that is expressed follows the determinant hierarchy: task identity → backbone architecture → training convergence → domain structure. Commensurability version 3 decomposes the readout condition by mechanism class, because the two mechanisms have different failure profiles — incoherent confidence for degeneracy, systematic misclassification for switching.

**Source:** n43, n44, n46, n48, n49.

---

## Ancillary eliminations

These are narrower hypotheses that were eliminated along the way. Each constrained the search space without requiring a full entry.

### Collision as sufficient condition

High per-layer alignment (collision) between two adapters is necessary but not sufficient for catastrophe. MRPC×SST-2 on RoBERTa has the highest cross-task alignment (ρ = 0.89) but is stable (instability = 0.21). Multiple high-alignment pairs are non-catastrophic. Collision gates entry to the mechanism ladder; V-module dimensionality mismatch provides the trigger within the collision regime. (n16)

### Readout-upstream coupling

Hypothesis: readout divergence is a downstream symptom of upstream representation-space incompatibility — the two should be correlated. Falsified: all same-task seed pairs have healthy V-module geometry (dim ratio > 0.78) regardless of whether their readout is aligned or orthogonal. The two conditions are independently determined by different mechanisms. The attractor model (n37) explains readout axis selection as a function of task identity, backbone architecture, and training convergence — not upstream LoRA geometry. (n36)

### Readout as seed-sensitivity explanation

Hypothesis: the large seed-sensitivity effects (CA-01's 29pp gap) reflect seed-dependent differences in readout geometry. Falsified: CA-01's catastrophic and mild seed variants have virtually identical readout geometry (decision_axis_cos: 0.015 vs -0.059). Readout explains zero variance in the seed-sensitive modulation. Head-level V-module cancellation (n24) explains the gap instead — 7 heads show |Δ_DR| ≥ 0.15 between seed variants, with opposite-sign deltas that cancel at module level. (n24, n32)

### Training depth as primary mechanism determinant

Hypothesis: training depth (convergence level) is the primary factor governing whether a family shows single-attractor or multi-attractor structure. Subordinated: training depth modulates attractor *count* (Strong QNLI is multi-attractor, Medium/Weak QNLI are single-attractor) but does not determine attractor *mechanism*. Task identity and backbone architecture are higher in the determinant hierarchy. (n48)

### Domain structure as primary mechanism determinant

Hypothesis: domain-shift families show different attractor structure because their training distributions differ from the pretraining domain. Subordinated: domain structure is the weakest factor in the hierarchy. SST-2 (domain shift) is multi-attractor on DistilBERT, but this is better explained by the task×backbone interaction than by domain mismatch per se. Domain is a modulator, not a determinant. (n48)

---

## What survives

After ten primary eliminations and five ancillary ones, the surviving explanatory framework is:

| Concept | Role | Evidence status |
|---------|------|----------------|
| Boundary detection (same-task / cross-task) | Core classification, settled | 0 false positives, 5 inventories, 53+ pairs |
| Instability | Portable descriptive organizer | Consistent on 2 backbones; awaiting DeBERTa |
| V-module dimensionality ratio | Strongest upstream risk correlate | d = 3.36, zero overlap; 2 backbones |
| Head-level cancellation | Explains seed sensitivity within a risk class | Resolved for CA-01; CA-02 shows concentrated variant |
| Readout attractor topology | Describes the solution space; not a risk signal | 10 families mapped, 2 mechanisms classified |
| Readout gating | Conjunctive gate condition | Confirmed: common, harmless alone, independently determined |
| Conjunctive model | V-module pathology × readout incompatibility → catastrophe | Four independent evidence lines; behavioral confirmation |
| Failure taxonomy | Downstream behavioral signatures | 5 categories; double dissociation between failure modes |

Everything else — portable severity, task-pair lookup, aggregate thresholds, readout-as-risk, readout-as-amplifier, universal feature plurality, collision-as-sufficient, readout-upstream coupling, readout-as-seed-explanation, training depth and domain as primary determinants — has been tested and set aside. The surviving framework is what remains after the alternatives were eliminated.

---

## The epistemic structure

The eliminations have a logic. They progress from the simplest possible explanations (severity is portable, task pairs are inherently catastrophic) through intermediate mechanisms (readout is the risk, readout is the amplifier, features are the universal origin) to the current model, which requires the conjunction of two independently measurable geometric conditions and admits two distinct readout mechanisms.

At each step, the program did not simply propose a replacement — it *tested* the incumbent hypothesis against specific falsifying evidence. The SC-QMRB control falsified readout-alone; the same-task seed analysis falsified readout-as-risk; the per-module decomposition falsified aggregate thresholds; the decision-axis analysis falsified universal feature plurality. Each falsifier is a specific empirical observation, not a theoretical preference.

This epistemic discipline is the sidecar's strongest methodological contribution. The theoretical picture is credible not because it is elegant but because it survived while ten alternatives did not.
