# n79 -- Cross-Artifact Compatibility Framework

**Type:** synthesis note
**Date:** 2026-03-31
**Program:** Cross-Artifact Compatibility Research (Route 2)
**Stage:** D
**Depends on:** n76 (panel), n77 (invariants), n78 (local signals)
**Status:** complete

---

## Question

What is the simplest scientifically honest framework for talking about compatibility across artifact classes?

---

## Framework

The framework has three layers. They are not alternatives; they stack.

### Layer 1 -- Artifact-Invariant Compatibility Signals

These signals recur across LoRA, LoHa, and checkpoint delta artifact classes regardless of representation form.

**Strong invariants:**
- **Evidence regime gating.** QA status and behavioral evidence availability dominate triage decisions in all three classes. This is the single most portable compatibility signal because it operates on metadata, not on structural measurements.
- **Conservative candidate narrowing.** The workflow reduces broad input to a smaller actionable subset in all three classes. The narrowing ratio varies (70-100%) but the logic -- QA gating first, then task boundaries, then structural risk -- is consistent.

**Moderate invariants:**
- **Task-relation ordering.** The ordering `same_task > same_family > cross_task` appears in all classes where it can be tested (LoRA and checkpoint delta). This reflects task similarity, which is upstream of representation form.
- **Same-family intermediate status.** Same-family pairs occupy a distinct middle ground in both LoRA and checkpoint delta. This is a direct consequence of the task-relation ordering.

These four signals are the cross-artifact substrate. They are workflow-level and task-relational, not structural-metric-level.

### Layer 2 -- Representation-Family Features

These signals are meaningful only within one representation family and should not be generalized across classes.

**Factor-based family (LoRA + shimmed LoHa):**
- Subspace overlap, principal angles, V-module dimensionality ratio
- Merge strategy recommendations (linear, ties, norm_equalized, audit_aware)
- Spectral audit metrics from SVD of factor matrices

**Summary-based family (checkpoint delta):**
- Summary profile cosine similarity
- Delta spectral statistics from full-matrix SVD

**Shimmed artifacts (LoHa):**
- Shim approximation error as potential confound

**Critical observation:** The sidecar's strongest finding -- V-module dimensionality ratio (d=3.36) -- is Layer 2, not Layer 1. It requires factor-level subspace geometry. It does not transfer to checkpoint deltas. The project's most discriminative signal and its most portable signals are at different layers.

**Partially portable (with caveats):**
- Compatibility scores exist in all classes but use different scales. Ordinal ranking may transfer; absolute values do not.
- Pair risk labels (low/medium/high) use shared vocabulary but different calibration.
- Stable rank metrics have the same name but different semantic content across representation families.

### Layer 3 -- Decision-Dependent Interpretation

How a compatibility signal is used depends on the decision scenario.

| Scenario | Aggregation style | Which Layer 2 is relevant | Cross-artifact coverage |
|----------|-------------------|---------------------------|------------------------|
| Merge | Worst-case | Factor geometry (strategy, conflict, imbalance) | LoRA only (native) |
| Routing | Distributional | Factor geometry + confusability | LoRA only (pilot) |
| Triage | QA-gate-first | Evidence status + any structural risk | All three classes |

**Triage is the only decision scenario with complete cross-artifact coverage.** Merge and routing are operationally restricted to factorized artifacts. This is not a temporary limitation -- it follows from the structural requirements of the decision.

---

## How the layers interact

1. **Layer 1 provides the shared scaffold.** Evidence gating and task-relation ordering work everywhere. A reader can trust that "same-task pairs are safer than cross-task pairs" across artifact classes.

2. **Layer 2 provides the local detail.** The specific structural metrics that drive compatibility judgments differ by representation family. A reader must qualify any structural claim by artifact class.

3. **Layer 3 provides the decision context.** The same Layer 2 signal has different implications for merge (worst-case), routing (distributional), and triage (QA-first). A reader must know which decision is being made.

**The key interaction:** Layer 1 invariants support Layer 3 triage across all artifact classes. Layer 2 features support Layer 3 merge/routing only for factorized artifacts. The cross-artifact compatibility framework is therefore strongest for triage and weakest for merge.

---

## What a reader should be able to answer

After reading this framework:

**Q: What is shared across artifacts?**
A: QA gating, conservative narrowing, task-relation ordering, same-family intermediate status.

**Q: What is local?**
A: Factor geometry (including V-module signal), merge strategies, summary-profile comparison geometry, numeric compatibility scales, shim-specific artifacts.

**Q: How does scenario dependence sit on top of that?**
A: Triage uses Layer 1 (shared) plus any available Layer 2. Merge and routing use Layer 2 (local to factorized artifacts). Only triage is cross-artifact.

---

## Output artifacts

- `sidecar/results/cross_artifact_portability/framework_table.json`
- `sidecar/results/cross_artifact_portability/framework_table.md`
- `sidecar/notes/n79_cross_artifact_compatibility_framework.md` (this note)
