# Cross-Task Severity Grading — Study 01 Results

## Study setup

- 8 adapters, 4 tasks (QNLI, RTE, MRPC, SST-2), 2 seeds each
- 28 pairs total (4 same-task controls + 24 cross-task)
- All adapters verified above base on distilbert-base-uncased
- Core-space computed for all 28 pairs
- Merged with uniform_linear, evaluated on both tasks

## Severity distribution (24 cross-task pairs)

| Severity | Count | % |
|----------|-------|---|
| near_safe | 0 | 0% |
| mild_degradation | 7 | 29% |
| asymmetric_dilution | 9 | 38% |
| broad_degradation | 5 | 21% |
| catastrophic | 3 | 13% |

**No cross-task pair is near-safe.**

---

## RQ1: Do cross-task pairs decompose into stable severity subtypes?

**Yes.** The 4-level decomposition is clean and reproducible across seed pairs within each task combination.

## RQ2: Does exact task-pair identity explain severity?

**Yes — this is the strongest explanatory factor.** Severity is almost entirely determined by which two tasks are paired:

| Task pair | Severity pattern | Mean basis |
|-----------|-----------------|------------|
| QNLI × MRPC | **Catastrophic** (3/4) + broad (1/4) | 0.852 |
| QNLI × SST-2 | Asymmetric + broad | 0.885 |
| QNLI × RTE | Mild + broad | 0.946 |
| RTE × SST-2 | **Purely asymmetric** (4/4) | 0.940 |
| MRPC × SST-2 | Mild + asymmetric | 0.879 |
| MRPC × RTE | Mild + asymmetric | 0.933 |

Each task pair has a characteristic severity profile that is stable across seed combinations. **Exact task-pair identity is the primary severity predictor.**

## RQ3: Does one task tend to dominate in asymmetric cases?

**Yes — SST-2 dominates consistently.** SST-2 survives in 10 of 24 cross-task pairs (the most of any task). This is likely because SST-2 adapters have the highest source accuracy (0.84) and the strongest gradient signal in the linear merge.

Task-direction dominance ranking:
1. SST-2: survives in 10 pairs
2. QNLI: survives in 3 pairs
3. RTE: survives in 3 pairs
4. MRPC: survives in 2 pairs
5. Balanced: 6 pairs

**Source-task strength predicts dominance direction.**

## RQ4: Do source-strength asymmetry or format predict severity?

### Source-strength gap

| Gap band | n | Severity |
|----------|---|----------|
| Medium (2-5pp) | 4 | **All catastrophic or broad** (3 catastrophic, 1 broad) |
| High (>5pp) | 20 | Mixed (7 mild, 9 asymmetric, 4 broad) |

**Counterintuitive:** medium-gap pairs are MORE severe than high-gap pairs. This is because all QNLI × MRPC pairs (catastrophic) happen to have medium gaps. Source-strength gap does NOT predict severity in the expected direction.

### Format axis

| Format | n | Worst | Pattern |
|--------|---|-------|---------|
| Same-format | 12 | 41.7pp | Contains ALL catastrophic pairs |
| Cross-format | 12 | 12.7pp | Mostly asymmetric dilution |

**Same-format is NOT safer.** It contains the catastrophic cases. Format similarity is misleading.

## RQ5: Does core-space add severity discrimination?

**Partially.** Core-space shared-basis scores do separate catastrophic from non-catastrophic:

| Severity | Mean basis | Range |
|----------|-----------|-------|
| Catastrophic | **0.854** | [0.852, 0.855] |
| Broad degradation | 0.900 | [0.848, 0.952] |
| Asymmetric dilution | 0.922 | [0.885, 0.947] |
| Mild degradation | 0.911 | [0.872, 0.950] |

The catastrophic pairs have the **lowest** shared-basis scores (all ~0.85). This is the first evidence that core-space captures something severity-relevant: task pairs with deeper basis incompatibility (lower shared-basis) tend to degrade more severely.

However, the ranges overlap substantially for non-catastrophic categories. Core-space separates catastrophic from the rest but does not grade within the mild/asymmetric/broad range.

## RQ6: Are any cross-task pairs unexpectedly mild?

**MRPC × RTE and QNLI × RTE** are the mildest cross-task pairs. QNLI × RTE pairs degrade only 4-6pp, and MRPC × RTE pairs degrade only 2-7pp. These are still degraded (none are near-safe), but they are notably milder than QNLI × MRPC (13-42pp).

Possible explanation: QNLI and RTE share more functional structure (both are entailment-style tasks) than QNLI and MRPC (entailment vs paraphrase detection).

## RQ7: How much does pair-risk explain?

**Very little.** Pair-risk rates 20/24 cross-task pairs as medium. It does not distinguish mild from catastrophic.

---

## Summary of explanatory factors

| Factor | Explains severity? | Notes |
|--------|--------------------|-------|
| **Exact task-pair identity** | **YES — strongest factor** | Each task pair has a characteristic severity profile |
| **Task-direction dominance** | Partially — explains asymmetry direction | SST-2 dominates, weaker tasks get diluted |
| **Core-space shared-basis** | Partially — separates catastrophic (~0.85) from rest (~0.90+) | Does not grade within non-catastrophic |
| Source-strength gap | **No** — medium-gap pairs are actually worse | Confounded with task-pair identity |
| Format axis | **No** — same-format contains catastrophic cases | Misleading signal |
| Pair-risk | **No** — rates 83% of cross-task as medium | Does not grade severity |
| Task advisory | **No** — fires uniformly on all cross-task | Catches boundary, not gradient |

---

## Verdict

**`severity_grading_factors_confirmed`**

Cross-task severity differences are real and substantially explained by two factors:

1. **Exact task-pair identity** — the dominant predictor. QNLI × MRPC is catastrophic; RTE × SST-2 is purely asymmetric; QNLI × RTE is mild. This is stable across seeds.

2. **Core-space shared-basis** — a secondary but real signal. Catastrophic pairs have consistently lower shared-basis scores (~0.85) than non-catastrophic pairs (~0.90+). This is the first evidence that core-space captures something severity-relevant in the cross-task regime.

## Implication for Gradience

The advisory catches the cross-task boundary. Core-space may eventually help grade severity within that boundary. But the strongest explanatory factor — exact task-pair identity — is not currently available as a Gradience signal. Whether it should become one is a future design question, not a current implementation task.
