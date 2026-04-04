# Merge-Aware Monitor Reference-Choice (Bounded Summary)

## Status

Bounded internal guidance update from tiny demo study:

- preferred default: `same_task` reference
- fallback: `same_family`
- exploratory only: `cross_task`

## Evidence Basis

From `field_trials/merge_aware_monitor_reference_choice`:

- same-task trace: `toward_compatibility` (most coherent)
- same-family trace: `mixed`
- cross-task trace: `inconclusive`

Interpretability rubric (0/1/2) produced:

- same-task = 2
- same-family = 1
- cross-task = 0

## How to Use This Internally

When using merge-aware training monitor in research mode:

1. choose a same-task reference first when available.
2. use same-family if same-task is unavailable.
3. treat cross-task traces as exploratory context, not primary signal.

## Boundaries

- this is still diagnostic-only telemetry
- no optimizer or training-control interpretation
- no predictive claims from this demo
- no product-facing escalation from this result
