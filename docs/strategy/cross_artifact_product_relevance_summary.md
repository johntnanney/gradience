# Cross-Artifact Product Relevance Summary

**Date:** 2026-03-31
**Source:** Cross-Artifact Compatibility Research Program (Route 2), Stages A-E
**Sidecar notes:** n76-n80

---

## What was tested

A 9-case panel spanning three artifact classes:
- **LoRA** (3 cases: same-task, same-family, cross-task)
- **LoHa** (3 cases: same-task only, shimmed via materialized SVD)
- **Checkpoint delta** (3 cases: same-task, same-family, cross-task; summary representation)

All cases used `distilbert-base-uncased`. Classification tasks only. CPU-only analysis.

---

## What transfers across artifact classes

### Confidently portable

1. **Evidence gating.** QA status and behavioral evidence availability dominate triage in all three classes. Missing or weak evidence triggers the same narrowing regardless of representation form.

2. **Conservative narrowing.** The workflow reduces broad candidate space to an actionable subset in all three classes. The narrowing logic is consistent: QA first, task boundaries second, structural risk third.

### Portable with stated scope

3. **Task-relation ordering.** Same-task pairs are more compatible than cross-task pairs in LoRA and checkpoint delta. Not yet testable in LoHa.

4. **Same-family intermediate status.** Same-family pairs sit between same-task and cross-task where the task-family registry applies.

5. **Triage as a cross-artifact scenario.** Triage works across all three artifact classes. Merge and routing are operationally restricted to LoRA.

---

## What does not transfer

1. **Structural metrics.** V-module dimensionality ratio, subspace overlap, and factor-level geometry require low-rank factorization. They do not apply to checkpoint delta summaries.

2. **Merge strategies.** Strategy recommendations (linear, ties, norm_equalized) are meaningful only for native LoRA.

3. **Numeric compatibility scores.** Scores use different scales by representation family. Ordinal ranking may transfer; absolute values do not.

4. **Risk label calibration.** Low/medium/high risk labels use different thresholds per representation.

---

## Product language guidance

**Safe to say:**
- "Gradience gates candidates by evidence quality regardless of artifact type."
- "Gradience reduces candidate space to an actionable subset across supported artifact classes."
- "The same-task / same-family / cross-task distinction is observed across tested artifact types."

**Not safe to say:**
- "Compatibility scores are comparable across artifact types."
- "Merge and routing work for checkpoint deltas."
- "Structural risk signals mean the same thing across representations."

---

## Scope limits

- Single backbone (`distilbert-base-uncased`).
- LoHa coverage is same-task only.
- Checkpoint delta coverage uses summary representation (Representation C), not full-matrix analysis.
- No behavioral evaluation data outside LoRA field trials.
- No decoder models, generative tasks, or prompt-only tuning.
