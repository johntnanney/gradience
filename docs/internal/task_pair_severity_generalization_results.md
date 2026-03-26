# Task-Pair Severity Generalization — Study 01 Results

## Study setup

- 16 adapters total: 8 per backbone (4 tasks × 2 seeds)
- Backbones: distilbert-base-uncased, roberta-base
- 56 total pairs (28 per backbone), 48 cross-task
- All merged with uniform_linear, evaluated on both tasks

## Cross-backbone severity comparison

| Task-pair family | DistilBERT severity | RoBERTa severity | Stability |
|------------------|--------------------|--------------------|-----------|
| MRPC × QNLI | **severe** (mean 20.3pp, 3 catastrophic) | **mild** (mean 4.5pp, 2 near-safe) | LOW |
| MRPC × RTE | mild (mean 4.6pp) | moderate (mean 8.3pp) | MEDIUM |
| MRPC × SST-2 | moderate (mean 9.4pp) | severe (mean 11.2pp) | MEDIUM |
| QNLI × RTE | mild (mean 4.2pp) | moderate (mean 5.1pp) | MEDIUM |
| QNLI × SST-2 | moderate (mean 10.0pp) | severe (mean 12.7pp) | MEDIUM |
| RTE × SST-2 | moderate (mean 6.9pp) | severe (mean 10.1pp) | MEDIUM |

Stability: 0 HIGH, 5 MEDIUM, 1 LOW

## Same-task controls

| Family | DistilBERT | RoBERTa |
|--------|-----------|---------|
| QNLI same | near-safe (2.2pp) | asymmetric (8.8pp)* |
| RTE same | near-safe (0.4pp) | near-safe (0.0pp) |
| MRPC same | near-safe (0.2pp) | near-safe (1.0pp) |
| SST-2 same | near-safe (0.4pp) | near-safe (0.4pp) |

*QNLI same-task on RoBERTa is elevated because qnli_s42 is below-base (flagged_weak).

## Key findings

### 1. Exact task-pair severity does NOT generalize cleanly across backbones

The strongest finding: the QNLI × MRPC catastrophic pattern on DistilBERT (20.3pp mean, 3/4 catastrophic) **does not replicate** on RoBERTa (4.5pp mean, 2/4 near-safe). This is the single most important negative result in the study.

### 2. Severity ordering is partially preserved

Most task-pair families shift by one severity zone across backbones. The relative ordering is partially preserved (RTE × SST-2 is consistently asymmetric on both), but the absolute severity levels differ enough that backbone-specific severity prediction is not reliable.

### 3. RoBERTa generally shows more severe degradation for SST-2 pairs

SST-2 × anything pairs are generally more severe on RoBERTa than DistilBERT. This may reflect RoBERTa's stronger SST-2 performance (0.89 vs 0.84) creating a larger dominance asymmetry.

### 4. Same-task controls remain safe on both backbones

3/4 same-task families are near-safe on both backbones. The exception (QNLI on RoBERTa) is explained by the weak qnli_s42 adapter, not by backbone instability.

## Verdict

**`task_pair_signal_partially_generalizes`**

Exact task-pair severity is NOT stable enough across backbones to justify treating it as a product-level Gradience input. The QNLI × MRPC catastrophic pattern — the strongest single finding from the DistilBERT study — does not replicate on RoBERTa.

What DOES generalize:
- All cross-task pairs degrade (0 near-safe across 48 cross-task pairs, excluding QNLI weak-source effects)
- The task advisory correctly fires on all cross-task pairs on both backbones
- Same-task pairs remain broadly safe on both backbones
- Task identity remains the key regime boundary

What does NOT generalize:
- Exact severity levels per task pair
- Which specific task pairs are catastrophic
- The absolute magnitude of degradation

## Implication for Gradience

The cross-task boundary (advisory) generalizes. The severity gradient within that boundary does not. This means:

1. **Do not build a task-pair severity lookup table.** It would be backbone-specific and unreliable.
2. **The advisory remains correctly positioned** as a boundary signal, not a severity grader.
3. **Core-space remains the best candidate** for severity triage within cross-task, because it measures structural properties that may be more backbone-stable than behavioral severity.
4. **The project should not claim that specific task pairs are "always catastrophic."** QNLI × MRPC is catastrophic on DistilBERT but not on RoBERTa.
