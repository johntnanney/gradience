# n52 — Ruled-Out Mechanisms Packet

**Type:** negative-results packet
**Date:** 2026-03-28
**Status:** Current. Extends n38 with the full set of eliminated mechanisms.

---

## Purpose

This packet collects every hypothesis the sidecar tested and rejected. Each entry is compact: what was proposed, what killed it, and what replaced it. The entries are ordered by the strength of the evidence against them.

A disciplined research program is defined as much by what it has ruled out as by what it has established. These are the dead ends that constrained the search space.

---

## 1. Portable severity score

**Proposal:** Assign each task pair a severity number that generalizes across backbones. Use it to rank merge risk.

**Evidence against:** QNLI×MRPC degrades 41.7% on DistilBERT and 1.7% on RoBERTa. QNLI×SST-2 degrades 30.3% on RoBERTa and 0.3% on DistilBERT. Six candidate severity signals (task-pair identity, core-space shared basis, pair-risk label, format similarity, source-strength gap, reconstruction error) all failed to predict severity portably. Rankings reverse completely across backbones.

**Replaced by:** Instability — the variability of severity across conditions — which ranks consistently. The unit of analysis shifted from "how bad" to "how fragile."

**Source:** S01, n03, n05, n06.

---

## 2. Task-pair catastrophe lookup

**Proposal:** Certain task pairs are catastrophic as a fixed property. Build a lookup table.

**Evidence against:** No pair is catastrophic on both tested backbones. QNLI×MRPC is catastrophic on DistilBERT but mild on RoBERTa. QNLI×SST-2 is catastrophic on RoBERTa but moderate on DistilBERT. The unit of catastrophe is the (task pair × backbone × seed) triple, not the pair.

**Replaced by:** The conjunctive model. Catastrophe requires specific geometric conditions that are backbone- and seed-dependent.

**Source:** S01, n05.

---

## 3. Aggregate within-layer threshold variable

**Proposal:** Compare adapter weight matrices layer-by-layer using the concatenated Q/K/V/O product. Catastrophic pairs should show distinctive aggregate subspace geometry.

**Evidence against:** When backbone is controlled, catastrophic cases are indistinguishable from safe collision controls on all four metrics (principal angles, top-direction overlap, dimensionality ratio, directional conflict). The apparent separation in raw data was a backbone confound: DistilBERT's 6-layer compression forces tighter alignment than RoBERTa's 12 layers. The concatenation was diluting the signal.

**Replaced by:** Per-module decomposition. Separating Q/K/V/O recovers a clean V-module dim-ratio signal (d=3.36) that the aggregate analysis had averaged away.

**Source:** n17, n18, n19, n20, n21.

---

## 4. Readout orthogonality as risk marker

**Proposal:** Near-orthogonal decision axes (cos ≈ 0) between two adapters' classifier heads indicates dangerous incompatibility. Use it as a merge-risk signal.

**Evidence against:** Two independent falsifications. First, the SC-QMRB control: identical readout geometry to catastrophic CA-01 but safe (Δ=1.7%). Second, same-task seed analysis: 5 of 14 same-task seed pairs show orthogonal readout yet all merge safely (max Δ=2.2%). A stand-alone readout-cosine metric would false-alarm on ~40% of same-task merges.

**Replaced by:** Readout incompatibility as a gate condition in the conjunctive model. Orthogonal readout is common, bimodally distributed, decoupled from upstream geometry, and harmless alone. It becomes dangerous only when combined with V-module pathology.

**Source:** n32, n36, n37.

---

## 5. Readout-alone explanation of catastrophe

**Proposal:** Catastrophic failure originates in the readout layer. Incompatible classifier heads amplify small representation-space distortions into large classification errors. The readout layer is an amplifier.

**Evidence against:** The SC-QMRB falsifier. If readout were an amplifier, SC-QMRB should have amplified whatever small upstream distortion exists on RoBERTa for QNLI×MRPC. It did not (Δ=1.7%). Additionally, the seed-contingent readout program showed that readout axis selection is decoupled from upstream V-module geometry — the two conditions are independently determined.

**Replaced by:** Readout gating. The readout layer transmits or absorbs upstream pathology rather than amplifying it. Compatible readout absorbs; incompatible readout transmits. The amplification, if any, occurs upstream in the V-module.

**Source:** n32, n36.

---

## Ancillary eliminations

### Collision as sufficient condition

High per-layer alignment (collision) between two adapters is necessary but not sufficient. MRPC×SST-2 on RoBERTa has the highest cross-task alignment (ρ=0.89) but is stable. Collision gates entry to the mechanism ladder; V-module dimensionality mismatch provides the trigger within the collision regime. (n16)

### Readout-upstream coupling

Hypothesis: readout divergence is a downstream symptom of upstream representation-space incompatibility. Falsified: all same-task seed pairs have healthy V-module geometry (dim ratio >0.78) regardless of readout alignment. The two conditions are independently determined. (n36)

### Readout as seed-sensitivity explanation

Hypothesis: the large seed-sensitivity effects (CA-01's 29pp gap) reflect seed-dependent readout differences. Falsified: CA-01's catastrophic and mild seed variants have virtually identical readout geometry (decision_axis_cos: 0.015 vs -0.059). Head-level V-module cancellation explains the gap instead. (n24, n32)

---

## What survives

After eliminating portable severity, task-pair lookup, aggregate within-layer thresholds, readout-as-risk-marker, readout-as-amplifier, collision-as-sufficient, readout-upstream coupling, and readout-as-seed-explanation, the surviving framework is:

- **Boundary detection** — same-task vs cross-task (core, confirmed)
- **Instability** — portable variability descriptor (sidecar, promising, awaiting DeBERTa)
- **V-module dimensionality ratio** — strongest correlate of catastrophic threshold (sidecar, d=3.36, 2 backbones)
- **Head-level cancellation** — explains seed sensitivity within a risk class (sidecar, resolved)
- **Readout gating** — conjunctive gate condition, common and harmless alone (sidecar, resolved)
- **Conjunctive model** — V-module pathology × readout incompatibility → catastrophe (sidecar, current best)

Everything else has been tested and set aside.
