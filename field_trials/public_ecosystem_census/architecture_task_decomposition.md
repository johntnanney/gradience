# Architecture-vs-Task Decomposition

One-way ANOVA effect sizes (eta-squared) for spectral fingerprint metrics.

## Architecture Effect

| Metric | eta-sq | F | n_groups | n_total |
|--------|--------|---|----------|---------|
| edge_gap_mean | 0.0363 | 0.9049 | 2 | 26 |
| energy_rank_90_p50 | 0.0029 | 0.0704 | 2 | 26 |
| entropy_erank_mean | 0.2378 | 7.4894 | 2 | 26 |
| stable_rank_mean | 0.0785 | 2.0444 | 2 | 26 |
| stable_rank_std | 0.0859 | 2.2552 | 2 | 26 |
| utilization_mean | 0.2518 | 8.077 | 2 | 26 |

## Task Effect

| Metric | eta-sq | F | n_groups | n_total |
|--------|--------|---|----------|---------|
| edge_gap_mean | 0.0296 | 0.3513 | 3 | 26 |
| energy_rank_90_p50 | 0.0505 | 0.6111 | 3 | 26 |
| entropy_erank_mean | 0.5054 | 11.7522 | 3 | 26 |
| stable_rank_mean | 0.267 | 4.1892 | 3 | 26 |
| stable_rank_std | 0.3102 | 5.1713 | 3 | 26 |
| utilization_mean | 0.3967 | 7.5606 | 3 | 26 |

## Summary

- Dominant factor: **task**
- Mean architecture eta-sq: 0.1155
- Mean task eta-sq: 0.2599

Note: These are observational effect sizes from found artifacts. They do not establish causal architecture or task effects.
