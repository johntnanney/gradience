# Public Ecosystem Spectral Census -- Study Memo

**Date**: 2026-04-03
**Outcome**: Partial Success

## Cohort

- Adapters attempted: 150
- Successfully audited: 26
- Excluded: 97
- Failed: 27
- Fingerprints extracted: 26

### By architecture family

| Family | Count |
|--------|-------|
| llama | 18 |
| mistral | 8 |

### By task category

| Category | Count |
|----------|-------|
| chat_instruct | 13 |
| classification | 10 |
| general_unknown | 3 |

## Confound Assessment

- Max confound R-squared: 0.6598
- Residualization recommended: True
- Confound R^2 > 0.3 detected; residualize metrics before architecture/task analysis

## Architecture-vs-Task Decomposition

- Mean architecture eta-squared: 0.1155
- Mean task eta-squared: 0.2599
- Architecture metrics with eta-squared > 0.10: 2
- Task metrics with eta-squared > 0.10: 4
- Dominant factor: task

### Effect sizes by metric

| Metric | Arch eta-sq | Task eta-sq |
|--------|-------------|-------------|
| stable_rank_mean | 0.0785 | 0.267 |
| utilization_mean | 0.2518 | 0.3967 |
| energy_rank_90_p50 | 0.0029 | 0.0505 |
| entropy_erank_mean | 0.2378 | 0.5054 |
| edge_gap_mean | 0.0363 | 0.0296 |

## Clustering Validation (k=5 NN purity)

- Architecture purity: 0.9 (random baseline: 0.574)
- Task purity: 0.7 (random baseline: 0.4112)

## Module-Type Replication

- Attention < MLP in 2/8 adapters (25%)
- Replicates encoder-era pattern: False

## Outcome Assessment

**Partial Success**

Partial signal: architecture signal detectable for 2 metric(s); kNN purity above random baseline.

### Guardrails

- These are observational findings from found artifacts, not causal claims.
- Confound assessment must be considered before interpreting architecture/task effects.
- Census findings do not replace the controlled GPU-return study.
- Hub download count and popularity are not quality evidence.
