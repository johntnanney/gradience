# Note: The Instability Research Program

## Metadata

- **Type:** synthesis
- **Date:** 2026-03-26
- **Related studies:** S01
- **Related panels:** P01
- **Supersedes:** n03 (instability vs severity) — detail note; n05 (instability as working concept) — conceptual argument
- **Project:** Phase 3, Project F — Instability Program Consolidation

---

## Purpose

This note formalizes instability as a **research program**, not just a working concept. A working concept says "look at this variable instead." A research program specifies what is being investigated, what counts as progress, what the program's commitments are, what would falsify it, and what its relationship to other programs is. This document is intended to be the single re-entry point for anyone asking: *what is the sidecar actually doing, and why?*

---

## 1. The Program's Central Claim

**Instability — the variability of a cross-task pair's severity across seeds and backbones — is a more portable and more scientifically tractable descriptor of cross-task merge risk than absolute severity.**

This is a claim about the *right level of description* for cross-task interference, not a claim about a specific mechanism. Mechanism is downstream (§6). The primary commitment is that instability, as a descriptive variable, will survive the DeBERTa adjudication test and continue to organize the evidence better than any severity-level alternative.

---

## 2. What the Program Is Built On

### Empirical base (two-backbone phase)

The program rests on five established findings, each grounded in the S01 two-backbone analysis:

1. **Severity reversal.** The most severe pair on DistilBERT (QNLI×MRPC, 41.7%) is among the mildest on RoBERTa (1.7%). Severity rankings are backbone-local. (S01 Preliminary Results, n01 Finding 1.)

2. **Instability clustering.** The six cross-task pairs separate into two non-overlapping clusters: backbone-reversal pairs (instability > 0.7) and stable-asymmetric pairs (instability < 0.3). The gap between clusters is 0.44 units. No pair occupies the interval [0.30, 0.74]. (instability_profiles.json, n05 §2.)

3. **Seed-fragility concentration.** The two backbone-reversal pairs have seed ranges of 25–29 percentage points; the four stable-asymmetric pairs have seed ranges of 0–8 percentage points. Catastrophic outcomes and high seed sensitivity co-occur. (seed_stability.json, n03 §Results.)

4. **Cross-backbone consistency of instability ranking.** The two most unstable pairs are the most unstable on *both* backbones, even though their severity profiles are inverted. (instability_case_table.md, n05 §2.)

5. **SST-2 escalation pattern.** All four SST-2-involving pairs escalate on RoBERTa relative to DistilBERT; the two non-SST-2 pairs are backbone-stable. This is a backbone-local regularity, not an instability finding, but it constrains mechanistic interpretation. (backbone_shift_table.md, n01 Finding 2.)

### Conceptual base

The program's conceptual commitments:

- **The unit of catastrophic interference is the (task pair × backbone × seed) triple**, not the task pair. Severity is conditioned on the full triple; instability integrates over the variable dimensions.
- **The right question is "stable vs. unstable?", not "how severe?"** Instability reframes the sidecar from a failed prediction problem (severity grading) into a tractable classification problem (stability class assignment).
- **Threshold character, not smooth distribution.** Catastrophic outcomes are not the tail of a smooth severity curve. They appear to require specific geometric conditions to trigger. The bimodal clustering and the 0.44-unit gap support this.

---

## 3. What the Program Predicts

### DeBERTa predictions (pre-registered in S01 §DeBERTa Success Criterion)

**Prediction A:** QNLI×MRPC and QNLI×SST-2 will have the highest seed ranges on DeBERTa-v3, with seed ranges at least 2× the median of the other four pairs.

**Prediction B:** The four stable-asymmetric pairs will remain below 10% seed range, with no backbone-reversal behavior.

**Prediction C:** The instability gap (no pair in the 0.30–0.70 range) will persist.

### Extended predictions

**Prediction D (V-module — now pre-registered in n07):** In collision-prone cross-task pairs, catastrophic outcomes will show V-module dimensionality ratio < 0.75 at critical layers, while safe collision controls will show V-module dimensionality ratio > 0.78 — on DeBERTa, an architecture with disentangled attention. **This is the sharpest prediction in the program.** It was added after n21 identified V-module dimensionality mismatch as the strongest signal in the sidecar evidence base (d=3.36, zero range overlap on the backbone-controlled RoBERTa comparison). The earlier version of Prediction D (structural, per-layer alignment) has been confirmed and superseded — per-layer alignment is a precondition, not the threshold variable; V-module dimensionality mismatch is the threshold-level candidate.

**Prediction E (mechanistic):** The threshold character of catastrophic interference implies that a structural predictor of instability class (not magnitude) exists in the adapter subspaces — something computable from the weight matrices alone, without requiring merge + evaluation. **Candidate now available:** V-module dimensionality ratio (n21). If Prediction D PASSES on DeBERTa, Prediction E is substantively confirmed — V-module dimensionality ratio would be a structural predictor computable from two adapters' weights without merging or evaluating them.

**Prediction F (taxonomy):** The DeBERTa data will populate at most one new taxonomy class (e.g., a pair that is unstable-severe but not backbone-reversing). The current four classes (backbone reversal, unstable severe, stable asymmetric, stable mild) are sufficient for six cross-task pairs on three backbones.

---

## 4. What Counts as Progress

A contribution advances the instability program if it satisfies at least one of:

1. **Portability evidence.** Demonstrates that instability rankings are preserved on a new backbone or a new set of task pairs. The DeBERTa adjudication is the immediate test.

2. **Structural predictor.** Identifies a signal computable from adapter weights that predicts instability class (backbone-reversal vs. stable-asymmetric) without requiring merge evaluation. This is the decisive step toward core promotion.

3. **Mechanistic explanation.** Provides evidence for or against thresholded subspace interference — specifically, whether catastrophic outcomes arise from shared-layer loading (collision) combined with within-layer subspace incompatibility. Per-layer analysis (n15) confirmed the collision precondition. Per-module analysis (n21) identified V-module dimensionality mismatch as the strongest correlate of the catastrophic threshold (d=3.36, zero range overlap). The remaining mechanistic question is whether V-module dimensionality mismatch is portable across backbones and whether it explains CA-01 seed sensitivity (which it does not yet — see n21 §3.1).

4. **Taxonomy refinement.** Demonstrates that the current four-class taxonomy needs expansion, merger, or redrawing based on new data — but only if the refinement improves classification accuracy or interpretive clarity.

A contribution does **not** advance the program if it:

- Improves severity prediction on a single backbone (severity is not the target variable)
- Adds complexity without improving classification or portability
- Conflates instability with severity (they are distinct concepts; see glossary)
- Requires new infrastructure without a driving question

---

## 5. Falsification Conditions

### Program-level falsification

The instability program should be abandoned or fundamentally restructured if:

1. **DeBERTa Prediction A fails.** If a currently stable pair becomes the most unstable on DeBERTa, instability rankings are backbone-dependent and the concept loses its claim to portability. This is the hardest falsification condition.

2. **The instability gap closes.** If DeBERTa data places multiple pairs in the [0.30, 0.70] range, the bimodal structure is a two-backbone artifact rather than a stable property of the task pairs. The taxonomy would need replacement.

3. **A severity signal generalizes.** If a signal is found that predicts absolute severity portably across three backbones, then severity — not instability — would be the right target variable. The instability program would become a secondary concern.

### Partial falsification (triggers recalibration, not abandonment)

- **Prediction B fails** (one or more stable pairs become unstable on DeBERTa): the composite formula needs recalibration, but the concept may survive.
- **Prediction C fails** (gap narrows but doesn't close): the threshold character is softer than expected, but the two-cluster structure may still hold.
- **Prediction D partially confirmed, V-module signal identified** (n15 found collision pattern at layer level; n18 ruled out aggregate within-layer geometry; n21 identified V-module dimensionality mismatch as the threshold-level signal, d=3.36): the threshold mechanism is module-specific within the collision regime, concentrated in value projections. CA-01 seed sensitivity remains below per-module resolution.

---

## 6. Relationship to Mechanism

The instability program is primarily *descriptive*, not *mechanistic*. Its first-order commitment is that instability is a better descriptive variable than severity. Mechanism is a second-order goal: explaining *why* some pairs are unstable.

The leading mechanistic hypothesis is **thresholded subspace interference** (glossary): catastrophic merge failures require shared-layer loading (both adapters concentrating norm mass in the same layers) combined with within-layer subspace incompatibility (the adapter subspaces overlapping in ways that produce destructive interference under linear merge). When either condition is absent — adapters load different layers, or same-layer subspaces are compatible — the same task pair merges without catastrophe. The threshold hypothesis explains:

- Why instability and severity co-vary (the collision precondition — shared-layer loading — is pair-level, but the within-layer incompatibility that triggers high severity is seed-dependent)
- Why catastrophic pairs are seed-fragile (per-layer alignment is consistent across seeds, but per-module subspace geometry varies with initialization — the threshold is in V-module dimensionality balance, not layer profiles)
- Why the gap exists (the collision precondition is either present or not; when absent, degradation stays in the moderate, predictable range regardless of seed)

Per-layer analysis (n15) confirmed the collision precondition: catastrophic pairs show higher layer-level alignment (ρ=0.76) and lower divergence (JS=0.007) than stable cross-task pairs (ρ=0.62, JS=0.014). Aggregate within-layer analysis (n18) ruled out concatenated subspace geometry as the threshold variable. Per-module analysis (n21) then identified **V-module dimensionality mismatch** as the strongest correlate: in collision-prone cross-task pairs, catastrophic outcomes are associated with strong dimensionality mismatch in V-module geometry, even when aggregate within-layer geometry is non-discriminating (d=3.36, zero range overlap). The interpretation is that catastrophic merges involve structurally incommensurable value-projection perturbations — one adapter's concentrated, the other's diffuse — and the linear merge smears features that remain commensurable in safe collision pairs. (See glossary for frozen definitions.)

The descriptive program does not depend on the threshold hypothesis being correct. Even if the mechanism turns out to be different, the instability ranking's portability is an empirical question independent of mechanism. Portability can be confirmed or denied without understanding why it holds.

---

## 7. Program Architecture

### Current components

| Component | Status | Location |
|-----------|--------|----------|
| Instability concept note | Complete | n05 |
| Case table (two-backbone) | Complete | results/s01/instability_case_table.md |
| Instability profiles (JSON) | Complete | results/s01/instability_profiles.json |
| Taxonomy | Complete | results/s01/taxonomy.json |
| Figures (4) | Complete | figures/s01_*.svg |
| DeBERTa success criteria | Complete | S01 §DeBERTa Success Criterion |
| This program statement | Complete | n06 |
| DeBERTa adjudication protocol | Complete | n07 |

### Planned additions (CPU-only)

| Component | Project | Purpose |
|-----------|---------|---------|
| Expanded anchor dossiers | H | Per-anchor reference cases with testable questions |
| Backbone-local interpretation notes | I | Separate local regularities from portable ones |
| Local artifact mining | G | Per-layer structural analysis from saved weights (**complete** — n13–n15, MIXED outcome: collision pattern found) |
| DeBERTa-ready case table | F | Pre-registered predictions per pair for DeBERTa |

### Blocked on GPU

| Component | Requirement |
|-----------|-------------|
| DeBERTa adapter training | 8 adapters (r=16, all 4 attention modules), ~few hours on consumer GPU |
| DeBERTa merge evaluation | 28 pairs, linear merge |
| DeBERTa per-module geometry | Per-module analysis on DeBERTa adapters — tests Prediction D (V-module signal) |
| Three-backbone instability update | Recompute composite scores with DeBERTa data |
| Promotion assessment | Instability ranking (A–C) and V-module signal (D) both require DeBERTa data |

---

## 8. Relationship to Core Gradience

Core Gradience's design is validated by the instability findings. Core stops at boundary detection and does not attempt severity grading — correctly so, because severity does not generalize.

Instability is **not promotable until**:

1. Three-backbone evidence exists (DeBERTa adjudication)
2. A structural predictor is identified (computable without merge evaluation) — *candidate: V-module dimensionality ratio (n21), pending DeBERTa*
3. Decision value is demonstrated (instability must improve workflow outcomes beyond boundary detection)
4. False-negative risk is assessed (instability-based guidance must not cause worse outcomes)
5. The signal can be expressed simply and conservatively in core's output

If conditions 1–5 are met and Prediction D also PASSES, the core output could change from:

> "This is a cross-task pair. Exercise caution."

to:

> "This cross-task pair shows V-module dimensionality mismatch at critical layers. Budget extra evaluation before merging."

That is a qualitative advance over the current binary boundary — it identifies *which* cross-task pairs are highest-risk, using a signal computable from the adapter weights alone. It is premature until DeBERTa adjudicates both the instability ranking and the V-module signal.

---

## 9. What This Document Replaces

- **n03** (instability vs severity): Introduced instability as a phenomenon. Superseded by n05 and now by this program statement. Retained for detail reference.
- **n05** (instability as working concept): Promoted instability from finding to concept. This document promotes it from concept to research program. n05 remains the best single-document argument for *why* instability matters. This document specifies *what we are doing about it*.

The relationship is: n03 introduced the phenomenon → n05 argued it should be the central concept → n06 structures it as a research program.

---

## Decision

The instability research program is now the sidecar's organizing structure. All Phase 3 work (Projects F, G, H, I) and the per-module geometry program should be understood as contributions to this program. The DeBERTa adjudication is the next decisive empirical step — now with two adjudication targets: instability ranking portability (Predictions A–C) and V-module dimensionality mismatch portability (Prediction D). CPU-only work has sharpened the adjudication from a broad "does instability survive?" to the more specific and more falsifiable "does the V-module dimensionality-ratio signal survive on a backbone with disentangled attention?"
