# Ambiguous Relationship Adjudication Protocol

## Core question

When two adapters are neither obvious same-task redundancies nor obvious cross-task mismatches, does core-space add behaviorally useful discrimination beyond ordinary pair-risk?

## Backbone

distilbert-base-uncased

## Task family

NLI-family cluster (all inference/entailment tasks):
- QNLI (question-answering NLI, 2 classes)
- RTE (recognizing textual entailment, 2 classes)
- MNLI (multi-genre NLI, 3 classes)

## Base model accuracy

| Task | Base accuracy | Labels | Eval samples |
|------|-------------|--------|-------------|
| QNLI | 0.542 | 2 | 500 |
| RTE | 0.458 | 2 | 277 (full val) |
| MNLI | 0.348 | 3 | 500 |

## Adapters

6 total: 2 per task, trained with different seeds.

All must independently beat base by >= 3pp.

Training config:
- r=16, alpha=16
- target_modules: q_lin, k_lin, v_lin, out_lin
- 2000 training samples, 1000 steps
- lr=5e-5, batch_size=8
- seeds: 42 and 7

## Pair panel (8 pairs)

### Group 1: Same-task controls (2 pairs)
- QNLI seed42 x QNLI seed7
- RTE seed42 x RTE seed7

Expected: safe merges. Tests whether core-space still overwarns.

### Group 2: Adjacent-task ambiguous (4 pairs)
- QNLI seed42 x RTE seed42
- QNLI seed42 x MNLI seed42
- RTE seed42 x MNLI seed42
- QNLI seed7 x RTE seed7

These are the center of the study. Tasks are related (all NLI-family) but not identical.

### Group 3: Contrast pairs (2 pairs)
- MNLI seed42 x MNLI seed7 (same-task control for 3-class)
- RTE seed42 x MNLI seed42 (adjacent but different label count)

## Merge strategy

uniform_linear for all pairs (the naive default).

## Evaluation

Each merged adapter evaluated on:
- The task of source A (500 samples or full val)
- The task of source B (500 samples or full val)

## Degradation thresholds

| Category | Drop from best individual |
|----------|--------------------------|
| Safe | <= 2pp |
| Caution | 2-5pp |
| Degraded | > 5pp |

## Classifier head strategy

For same-task merges: average both classifier heads.
For cross-task merges (different num_labels): use source A's classifier head when evaluating on task A, source B's head when evaluating on task B.

## Success criteria

The study answers its question if it produces at least one of:
- Result A: core-space disagreement predicts degradation in ambiguous pairs (justifies keeping it)
- Result B: pair-risk remains sufficient even in ambiguous pairs (justifies narrowing further)
- Result C: core-space helps only in a specific sub-regime (tells us where it belongs)
