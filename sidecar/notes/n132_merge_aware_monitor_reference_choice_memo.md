# N132 — Merge-Aware Monitor Reference-Choice Memo

## What Was Tested

One fixed tiny training trajectory was monitored under three reference choices:

- same-task
- same-family
- cross-task

All non-reference settings were held constant; merge-aware callback monitoring
was enabled in all runs.

## What the Traces Showed

- same-task: cleanest read (`toward_compatibility`, increasing overlap/score)
- same-family: usable but mixed (`mixed`)
- cross-task: mostly inconclusive (`inconclusive`)

Observed interpretability scores (0/1/2 rubric):

- same-task: 2
- same-family: 1
- cross-task: 0

## What Is Now Safe to Say

For bounded internal use of this prototype:

1. same-task references are the best default.
2. same-family references are acceptable fallback probes.
3. cross-task references are exploratory and often less interpretable.

## What Remains Bounded

- tiny synthetic demo only
- no predictive merge-outcome claim
- no training-control or optimization claim
- no product-surface implication

## Decision

`same_task_preferred` guidance is justified for internal docs under current
bounded scope, with same-family fallback and explicit caution on cross-task use.
