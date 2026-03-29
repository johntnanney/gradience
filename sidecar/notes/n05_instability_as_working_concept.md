# Note: Instability as the Sidecar's Working Concept

## Metadata

- **Type:** synthesis
- **Date:** 2026-03-26
- **Related studies:** S01
- **Related panels:** P01
- **Supersedes:** n03 (instability vs severity) — which introduced instability as a first-class phenomenon; this note elevates it to the sidecar's central organizing concept

---

## Summary

This note records a conceptual pivot. The sidecar began with the question *what determines cross-task severity?* The two-backbone analysis has shown that this question is malformed: severity is not stable enough across conditions to serve as the target variable. The sidecar's working concept is now **instability** — the variability of a pair's severity across seeds and backbones — and the central question becomes *what determines whether a cross-task pair is stable or unstable?*

This is not merely a terminological change. It restructures what the sidecar measures, what it predicts, and what it would need to demonstrate in order to promote a finding to core.

---

## The Argument in Five Steps

### 1. Severity is outcome-local

Severity — the magnitude of performance degradation after merge — is a property of a specific evaluation: one pair, one backbone, one seed combination. The two-backbone analysis shows that severity rankings do not transfer. QNLI×MRPC is the most severe pair on DistilBERT (41.7%) and among the mildest on RoBERTa (1.7%). A severity model trained on DistilBERT data would produce catastrophically wrong recommendations on RoBERTa.

This is not a measurement problem. It is a conceptual problem. Severity is conditioned on too many variables (backbone architecture, learned subspace geometry, random initialization) to function as a portable descriptor of a task pair.

### 2. Instability is descriptively cleaner

Instability — defined as the composite variability of a pair's severity across seeds (within-backbone) and across backbones — produces a ranking that is consistent regardless of which backbone's perspective you adopt. The two backbone-reversal pairs (QNLI×MRPC, QNLI×SST-2) are the most unstable pairs on *both* backbones, even though their severity profiles are inverted. The four stable-asymmetric pairs are the least unstable on both backbones.

The instability ranking separates into two clean clusters with a gap at 0.30–0.74 that no pair currently occupies. No severity-based ranking achieves this. This gap is the sidecar's strongest empirical finding to date.

### 3. Task-pair identity is too coarse

The failed signals (core-space shared-basis, pair-risk label, format similarity, source-strength gap, reconstruction error) all operated at the level of task-pair identity. They asked: *given that this pair is QNLI×MRPC, how severe will the merge be?*

The evidence shows that task-pair identity does not determine merge outcome. The same pair can be catastrophic, severe, or mild depending on backbone and seed. The unit of analysis for catastrophic interference is the **(task pair × backbone × seed) triple**, not the task pair. Any signal that operates only at the task-pair level will necessarily fail to predict outcomes that depend on the triple.

Instability captures this by measuring *how much the outcome varies* across the dimensions that task-pair identity ignores.

### 4. The working hypothesis: thresholded subspace interference

The co-occurrence of high severity and high instability is not a tautology — it could in principle be otherwise. A pair could degrade consistently and severely (high severity, low instability) or erratically and moderately (low severity, high instability). The data shows neither pattern. Instead, severity and instability co-vary, with a threshold character:

- **Unstable pairs** (instability > 0.7) exhibit catastrophic degradation under some conditions and mild degradation under others. The seed ranges are 25–29 percentage points. A single seed change can move a pair from 1% degradation to 42%.
- **Stable pairs** (instability < 0.3) degrade 5–15% consistently across seeds and backbones. Their seed ranges are 4–8 percentage points.

This pattern is consistent with a threshold mechanism: catastrophic interference requires specific geometric conditions — shared-layer loading combined with within-layer subspace incompatibility — to trigger. When those conditions are not met, the same task pair shows only moderate, predictable degradation. The subspace geometry — not the task identity — is the operative variable.

Preliminary per-layer analysis (n15) supports a **collision model**: catastrophic pairs' adapters concentrate their norm mass in the *same* layers (high alignment, low divergence), creating the precondition for destructive interference. Stable cross-task pairs distribute across *different* layers, giving the merge room to accommodate both. However, per-layer alignment alone does not explain seed sensitivity — the seed-dependent variable operates within layers, not across them. The hypothesis is therefore: **thresholded interference inside aligned layer profiles**, where shared-layer loading is necessary but within-layer subspace incompatibility is the sufficient condition.

The per-module geometry analysis (n21) has identified the strongest current operationalization of "within-layer subspace incompatibility": **V-module dimensionality mismatch**. In collision-prone cross-task pairs, catastrophic outcomes are associated with strong dimensionality mismatch in V-module geometry, even when aggregate within-layer geometry is non-discriminating. Catastrophic merges involve structurally incommensurable value-projection perturbations: one adapter's is concentrated, the other's diffuse, and the linear merge smears features that remain commensurable in safe collision pairs. (Cohen's d = 3.36, zero range overlap on the backbone-controlled comparison; see n21 §2, §4, and glossary.)

### 5. DeBERTa-v3 is the adjudication test

The instability ranking is currently derived from two backbones. Two data points are suggestive but not decisive. DeBERTa-v3-base provides the cleanest possible adjudication because:

- It is architecturally distinct from both existing backbones (disentangled attention vs. standard scaled dot-product).
- It matches RoBERTa's depth (12 layers) but not its attention mechanism, disentangling the depth variable from the architecture variable.
- It uses a different tokenizer and pretraining objective (replaced token detection vs. masked language modeling), so LoRA subspaces will be learned in a different representational context.

**The DeBERTa success criterion is not about severity.** The question is not "which pair is catastrophic on DeBERTa?" The question is: **do the same pairs remain the most unstable, regardless of which pair happens to be catastrophic?**

Specifically, the instability hypothesis predicts:

1. QNLI×MRPC and QNLI×SST-2 will have the highest seed ranges on DeBERTa (whether or not either is catastrophic).
2. The four stable-asymmetric pairs will remain in the low-instability cluster (instability < 0.3 or equivalent).
3. The instability gap between the two clusters will persist.

If all three hold, instability is the first portable cross-backbone merge descriptor, and it becomes a serious candidate for eventual promotion to core.

If (1) holds but (2) or (3) fails, instability is real but the specific composite score needs recalibration.

If (1) fails — if a currently stable pair becomes the most unstable on DeBERTa — then instability is backbone-dependent too, and the sidecar needs to rethink its framework.

---

## What Changes

### Organizing framework

The sidecar's taxonomy is now the primary conceptual tool:

| Class | Instability | Behavior | N pairs (current) |
|-------|------------|----------|------------------:|
| **Backbone reversal** | > 0.7 | Catastrophic on one backbone, mild on another. Highest seed ranges. | 2 |
| **Unstable severe** | > 0.5 | High severity and high variance, but may not reverse across backbones. | 0 (empty; may populate with DeBERTa data) |
| **Stable asymmetric** | < 0.3 | Degrades 5–15% consistently. Low seed variance. Predictable across backbones. | 4 |
| **Stable mild** | < 0.1 | Minimal degradation under all conditions. | 0 (no cross-task pair is stably mild) |

### Central question

**Old:** What determines cross-task severity?
**New:** What determines whether a cross-task pair is stable or unstable?

### What counts as progress

A finding advances the sidecar if it:

1. Identifies a structural predictor of instability class (not severity magnitude), OR
2. Provides mechanistic explanation for why unstable pairs have threshold character, OR
3. Demonstrates that instability rankings are portable to a new backbone.

A finding that only improves severity prediction on a single backbone is not progress — it would not generalize.

### Study design implications

Future studies should stratify by stability class, not severity class. The relevant contrast for Workstream B is backbone-reversal vs. stable-asymmetric pairs, not "catastrophic vs. mild." The per-module analysis (n21) has sharpened the question further: the operative contrast is now *"does V-module dimensionality mismatch at critical layers explain why a collision-prone pair crosses the catastrophic threshold?"* The per-layer collision is the precondition; V-module geometry is the strongest current candidate for the threshold variable itself.

---

## Relationship to Core Gradience

Core Gradience's current design is validated by these findings. Core stops at boundary detection (same-task vs. cross-task) and does not attempt severity grading — which is correct, because severity does not generalize.

Instability is **not yet promotable to core**. Promotion requires:

1. Three-backbone evidence (DeBERTa-v3 replication) — *pending*
2. A structural predictor (computable without merge evaluation) — *candidate available: V-module dimensionality mismatch (n21), pending DeBERTa confirmation*
3. Demonstrated decision value (instability must improve workflow outcomes) — *not yet tested*
4. False-negative risk assessed (instability-based guidance must not cause worse outcomes) — *not yet assessed*
5. Simple and conservative expression in core's output — *not yet designed*

(These criteria are refined in n06 §8, which is the authoritative list.)

If all five are met, instability would change core's output from:

> *"This is a cross-task pair. Exercise caution."*

to:

> *"This is an unstable cross-task pair. Budget extra evaluation before merging."*

That is a meaningful improvement. But it is premature until DeBERTa adjudicates.

---

## Decision

Adopt instability as the sidecar's working concept. All subsequent study designs, analysis scripts, and interpretation notes should treat instability as the primary variable of interest and severity as a secondary, condition-dependent measurement. The DeBERTa leg of S01 is the highest-priority empirical task because it adjudicates whether instability is portable.
