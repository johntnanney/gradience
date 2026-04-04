# Merge-Aware Monitor Trend Rules (v1)

## Purpose

Define transparent, conservative rules for converting per-step compatibility
snapshots into per-metric and run-level trend summaries.

## Inputs

Ordered snapshot series with these numeric fields when available:

- `mean_overlap`
- `conflict_fraction`
- `imbalance_fraction`
- `compatibility_score`

## Per-Metric Labels

For each metric, return one label:

- `increasing`
- `decreasing`
- `mixed_unstable`
- `flat`
- `inconclusive`

Rules:

1. Fewer than 3 valid points -> `inconclusive`
2. Near-zero span -> `flat`
3. Direction changes across meaningful deltas -> `mixed_unstable`
4. Otherwise sign of start/end delta determines `increasing`/`decreasing`

## Desired Compatibility Directions

Compatibility-improving direction assumptions:

- `mean_overlap`: up
- `compatibility_score`: up
- `conflict_fraction`: down
- `imbalance_fraction`: down

## Run-Level Label

Count metric labels as:

- `toward`: metric moved in desired direction
- `away`: metric moved opposite desired direction
- `mixed`: metric label is `mixed_unstable`

Decision:

1. `toward_compatibility` if `toward >= 2`, `away == 0`, and limited mixed
2. `away_from_compatibility` if `away >= 2`, `toward == 0`, and limited mixed
3. `mixed` if both toward and away are present, or mixed dominates
4. `inconclusive` otherwise

## Interpretation Guardrail

These labels summarize trajectory shape only. They do not imply:

- causal improvement
- training intervention recommendation
- guaranteed merge behavior outcomes
