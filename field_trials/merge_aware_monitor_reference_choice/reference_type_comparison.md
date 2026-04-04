# Reference-Type Comparison

## Central Readout

In this bounded demo, reference-type interpretability followed:

`same_task` > `same_family` > `cross_task`

based on run-level trend coherence and simple interpretability scoring.

## Dimension Checks

### 1) Summary-label coherence

- same-task: `toward_compatibility`
- same-family: `mixed`
- cross-task: `inconclusive`

### 2) Metric consistency

- same-task: overlap and compatibility score both increased.
- same-family: both overlap and score were mixed/unstable.
- cross-task: score decreased but overall trace still lacked clean coherence.

### 3) Qualitative interpretability

Using rubric (`0=inconclusive`, `1=mixed`, `2=clearly interpretable`):

- same-task: `2`
- same-family: `1`
- cross-task: `0`

### 4) Relative ordering

- observed scores: `[2, 1, 0]`
- monotone match with expected ordering: `true`

## Bounded Interpretation

This is a tiny synthetic demo, so it supports only a usage heuristic:

- use same-task references first
- same-family can be a fallback
- cross-task should remain exploratory

It does not support predictive merge-success or training-control claims.
