# Task-Pair Severity Generalization — Implications

## Anchor finding

Exact task-pair severity does **not** replicate strongly enough across backbones to justify treating task-pair identity as a Gradience product input.

The strongest single example: QNLI × MRPC is catastrophic on DistilBERT (20.3pp mean degradation, 3/4 catastrophic) but mild on RoBERTa (4.5pp mean, 2/4 near-safe). Most task-pair families shift by at least one severity zone across backbones. Zero families showed high cross-backbone stability.

## What generalizes

| Signal | Cross-backbone stability |
|--------|------------------------|
| **Task identity boundary** | Strong — all cross-task pairs degrade on both backbones |
| **Advisory selectivity** | Strong — 0 false positives on both backbones |
| **Same-task safety** | Strong — controls near-safe on both backbones |
| Severity ordering (relative) | Partial — most families shift ~1 zone |
| Exact severity (absolute) | **Weak** — QNLI×MRPC catastrophic→mild across backbone |
| Which pairs are catastrophic | **Not stable** — backbone-specific |

## What this means for Gradience

### 1. Do not featureize task-pair identity

The evidence does not support a task-pair severity lookup table or task-pair advisory. Severity levels are backbone-dependent, and a feature built on DistilBERT data would be misleading on RoBERTa (and vice versa). This is a clear negative result that prevents premature feature construction.

### 2. Advisory remains correctly scoped as boundary-only

The advisory fires on all cross-task pairs on both backbones with zero false positives. It is correctly positioned as a binary same/different signal, not a severity grader. This study confirms that boundary detection generalizes even though severity grading does not.

### 3. Core-space becomes the stronger next candidate for severity triage

On DistilBERT, core-space shared-basis separated catastrophic pairs (~0.85) from non-catastrophic (~0.90+). This is a structural measurement, not a task-pair lookup — so it has a better chance of generalizing across backbones than exact task-pair identity does. The next useful study would test whether core-space severity triage replicates on RoBERTa.

### 4. Do not claim any specific task pair is "always catastrophic"

QNLI × MRPC was the poster case for catastrophic interference on DistilBERT. It is mild on RoBERTa. Any public-facing language about specific task pairs being inherently dangerous should be qualified as backbone-specific or removed entirely.

## Strategic update

The project's severity research line should now pivot from:
> "Which task pairs are catastrophic?"

to:
> "What structural properties predict severity, and do they generalize across backbones?"

Core-space is the best current candidate for that structural signal. Task-pair identity is not.

## What to do next

1. **Test core-space severity triage on RoBERTa.** If shared-basis ~0.85 still separates the most degraded pairs on RoBERTa, that would be a generalizable structural severity signal — much stronger than a task-pair lookup.

2. **Update public claims.** Remove or qualify any language that implies specific task pairs are always catastrophic.

3. **Close the task-pair featureization question.** Record this as a confirmed negative result. Task-pair identity is interesting for interpretation but not reliable enough for product use.
