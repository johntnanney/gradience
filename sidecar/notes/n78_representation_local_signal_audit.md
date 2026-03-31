# n78 -- Representation-Local Signal Audit

**Type:** findings note
**Date:** 2026-03-31
**Program:** Cross-Artifact Compatibility Research (Route 2)
**Stage:** C
**Depends on:** n76 (panel), n77 (invariant audit)
**Status:** complete

---

## Question

What compatibility signals are local to a representation family and should not be mistaken for substrate-level truths?

---

## Method

For each structural or metric signal used in the compatibility pipeline, assess:
- Which artifact classes produce it?
- Why does it appear there?
- Could it be computed in other classes?
- Is it genuinely portable, or does it only exist because of representation form?

---

## Local signal types identified

### Type A -- Factor-geometry-specific (2 signals)

**Factor subspace overlap.** Principal angles and subspace overlap between A/B factor pairs across adapters. This is the basis for the sidecar's strongest finding (V-module dimensionality ratio, d=3.36). Present in LoRA and shimmed LoHa. Absent from checkpoint delta because Representation C stores scalar summaries, not factor matrices.

**Implication:** The V-module signal -- the most discriminative catastrophe predictor in the sidecar program -- cannot be computed from checkpoint summary representations. Any claim that "Gradience detects catastrophic risk across artifact classes" would be false at the structural-metric level. It detects different things in different classes.

**Stable rank and energy concentration.** Both LoRA and checkpoint delta pipelines compute metrics with these names, but they measure different quantities. LoRA stable rank measures factor utilization within a low-rank decomposition. Checkpoint delta stable rank measures the spectral shape of the full delta matrix. The numbers are not comparable even when the names match.

### Type B -- Summary-profile-specific (1 signal)

**Checkpoint summary profile shape.** Representation C's pairwise comparison operates on cosine similarity of per-layer summary vectors. This comparison geometry has no analog in factor-based merge-audit. High cosine similarity in summary space (e.g., 0.998 for same-task seed pairs) does not imply low subspace conflict in factor space.

### Type C -- Extraction-artifact-specific (3 signals)

**Shim extraction artifacts.** The LoHa materialized shim applies truncated SVD to the Hadamard product, introducing approximation error. Any signal unique to shimmed LoHa may be partially caused by the shim rather than by underlying compatibility. This confound does not affect native LoRA or checkpoint deltas.

**Compatibility score numeric scale.** All three classes produce compatibility scores, but the scales and derivation differ. LoRA/LoHa use factor-based spectral compatibility. Checkpoint deltas use summary-based cosine + energy-delta scoring. Same-task LoRA: 0.475. Same-task checkpoint delta: 0.892. These numbers are not on the same scale.

**Pair risk categorical label.** All three classes produce low/medium/high risk labels, but the thresholds are calibrated to different scoring systems. LoHa same-task pairs are "low" risk; checkpoint delta same-task pair is "medium" risk. The labels look comparable but are not.

### Type D -- Execution-context-local (1 signal)

**Merge strategy recommendation.** Strategy strings (linear, ties, norm_equalized, audit_aware) are meaningful only for factor-based merge execution. LoHa receives recommendations through the shimmed pipeline but merge execution is unvalidated. Checkpoint deltas do not receive strategy recommendations at all.

---

## Portability classification

| Portability | Signals | Count |
|-------------|---------|-------|
| Local only | Factor subspace overlap, merge strategy, shim artifacts, summary profile shape | 4 |
| Partially portable | Compatibility score, stable rank, pair risk label | 3 |
| Fully portable | (none at the structural-metric level) | 0 |

**Key finding:** No structural metric is fully portable across all three artifact classes. The portable signals identified in Stage B (QA gating, narrowing, task-relation separation) are all workflow-level, not metric-level. The individual structural numbers that drive compatibility judgments are always representation-local.

---

## What this means for generalization claims

1. **Do not claim cross-artifact metric equivalence.** The same metric name (stable rank, compatibility score) can mean different things in different classes.

2. **Do not extend V-module findings to checkpoint deltas.** The sidecar's strongest discriminative signal requires factor-level geometry.

3. **Do treat workflow-level signals as genuinely portable.** QA gating, conservative narrowing, and task-relation ordering operate above the representation layer and survive artifact broadening.

4. **Do acknowledge shim confounds.** LoHa results are filtered through a materialized SVD that introduces its own approximation signature.

---

## Output artifacts

- `sidecar/results/cross_artifact_portability/local_signal_table.json`
- `sidecar/results/cross_artifact_portability/local_signal_table.md`
- `sidecar/notes/n78_representation_local_signal_audit.md` (this note)
