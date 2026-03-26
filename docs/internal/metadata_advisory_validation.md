# Metadata Advisory Validation

## Purpose

Validate whether task-identity metadata from QA artifacts cleanly separates safe from degraded merges, across both verified adjudication studies.

## Data

23 pairs total from two studies:

- **Study 1 (S1):** SST-2 x QNLI regime — 6 adapters (3 SST-2, 3 QNLI), 15 pairs
- **Study 2 (S2):** NLI-family regime — 6 adapters (2 QNLI, 2 RTE, 2 MNLI), 8 pairs

All adapters independently verified above base. Merges via uniform_linear. Evaluated on both source tasks.

## Consolidated table

| # | Study | Pair | Relationship | Risk | CS | Outcome | Asymmetric |
|---|-------|------|-------------|------|----|---------|-----------|
| 1 | S1 | sst2_r16_s42 x sst2_r16_s123 | same-task | redundant | marginal | safe | n/a |
| 2 | S1 | sst2_r16_s42 x sst2_r8_s42 | same-task | redundant | compatible | safe | n/a |
| 3 | S1 | sst2_r16_s123 x sst2_r8_s42 | same-task | redundant | marginal | safe | n/a |
| 4 | S1 | qnli_r16_s42 x qnli_r16_s123 | same-task | redundant | incompatible | safe | n/a |
| 5 | S1 | qnli_r16_s42 x qnli_r8_s42 | same-task | redundant | marginal | safe | n/a |
| 6 | S1 | qnli_r16_s123 x qnli_r8_s42 | same-task | redundant | incompatible | safe | n/a |
| 7 | S2 | qnli_s42 x qnli_s7 | same-task | redundant | marginal | safe | n/a |
| 8 | S2 | rte_s42 x rte_s7 | same-task | redundant | incompatible | safe | n/a |
| 9 | S2 | mnli_s42 x mnli_s7 | same-task | redundant | marginal | safe | n/a |
| 10 | S1 | sst2_r16_s42 x qnli_r16_s42 | cross-task | imbalanced | marginal | degraded | both |
| 11 | S1 | sst2_r16_s42 x qnli_r16_s123 | cross-task | imbalanced | incompatible | degraded | both |
| 12 | S1 | sst2_r16_s42 x qnli_r8_s42 | cross-task | redundant | incompatible | degraded | both |
| 13 | S1 | sst2_r16_s123 x qnli_r16_s42 | cross-task | imbalanced | marginal | degraded | both |
| 14 | S1 | sst2_r16_s123 x qnli_r16_s123 | cross-task | imbalanced | incompatible | degraded | both |
| 15 | S1 | sst2_r16_s123 x qnli_r8_s42 | cross-task | redundant | incompatible | degraded | both |
| 16 | S1 | sst2_r8_s42 x qnli_r16_s42 | cross-task | imbalanced | marginal | degraded | both |
| 17 | S1 | sst2_r8_s42 x qnli_r16_s123 | cross-task | imbalanced | incompatible | degraded | both |
| 18 | S1 | sst2_r8_s42 x qnli_r8_s42 | cross-task | redundant | incompatible | degraded | both |
| 19 | S2 | qnli_s42 x rte_s42 | adjacent-task | redundant | marginal | degraded | weaker diluted |
| 20 | S2 | qnli_s42 x mnli_s42 | adjacent-task | redundant | incompatible | degraded | weaker diluted |
| 21 | S2 | rte_s42 x mnli_s42 | adjacent-task | redundant | marginal | degraded | weaker diluted |
| 22 | S2 | qnli_s7 x rte_s7 | adjacent-task | redundant | marginal | degraded | weaker diluted |
| 23 | S2 | rte_s7 x mnli_s7 | adjacent-task | redundant | marginal | degraded | weaker diluted |

## Discrimination results

### Task identity as discriminator

| Relationship | Count | Safe | Degraded | Accuracy |
|-------------|-------|------|----------|----------|
| Same-task | 9 | **9** | 0 | 100% safe |
| Different-task | 14 | 0 | **14** | 100% degraded |
| **Total** | **23** | **9** | **14** | **100%** |

**Task identity perfectly discriminates safe from degraded across 23 pairs.**

### Pair-risk as discriminator

| Pair-risk | Same-task (safe) | Different-task (degraded) |
|-----------|-----------------|--------------------------|
| redundant | 9 | 8 |
| imbalanced | 0 | 6 |

Pair-risk "redundant" includes 8 degraded pairs — **it does not separate safe from unsafe when spectral overlap is high.**

Pair-risk "imbalanced" does correctly flag 6 cross-task pairs, but only because SST-2 x QNLI happen to show norm differences. The NLI-family adjacent-task pairs all escaped (rated "redundant" despite degrading).

### Core-space as discriminator

| CS status | Same-task (safe) | Different-task (degraded) |
|-----------|-----------------|--------------------------|
| compatible | 1 | 0 |
| marginal | 5 | 7 |
| incompatible | 3 | 7 |

Core-space "incompatible" includes 3 safe pairs and 7 degraded pairs — **it does not cleanly separate either.**

## Degradation patterns by task distance

| Regime | Pairs | Pattern |
|--------|-------|---------|
| Same-task | 9 | Safe (<=1.2pp, all cases) |
| Adjacent-task (NLI family) | 5 | Weaker task diluted 5-7pp; stronger task preserved |
| Cross-task (SST-2 x QNLI) | 9 | Both tasks degraded 8-18pp |

The degradation severity scales with task distance, and the pattern changes qualitatively:
- **Adjacent tasks:** asymmetric dilution (one task survives)
- **Distant tasks:** symmetric degradation (both tasks suffer)

## Conclusion

The `eval_dataset` field in the QA artifact is a **perfect discriminator** in this verified data. It achieves 23/23 accuracy across two studies, two backbone regimes, three task families, and both symmetric and asymmetric degradation patterns.

Neither pair-risk (misclassifies 8/14 different-task pairs as "redundant") nor core-space (3 safe pairs rated "incompatible," 7 degraded pairs rated "marginal") comes close.

## Recommendation

The evidence supports adding a task-relationship advisory to merge reporting. The implementation is simple:

1. If both adapters have QA artifacts with `eval_dataset` fields
2. And the fields differ
3. Emit a note in the merge report: cross-task linear merges may degrade the weaker task

This is an additive advisory — it does not change pair-risk, core-space, or any existing recommendation logic. It uses metadata already collected by the workflow.
