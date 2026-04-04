# Rank-Reduction Retain-Ratio Rerun (Bounded)

## Protocol (unchanged)
- informative families: `sst2`, `imdb`
- adapters: 4
- panels: 3 fixed + 3 random
- ablation samples: 24, 48, 72
- budgets: 0.35, 0.50, 0.65
- mode: `rank_reduction` only
- retain ratios tested: 0.50 (reference), 0.75, 0.85

## Comparison
| retain_ratio | mean_low_info | mean_flat | mean_valid_pair_frac | mean_nonzero_frac | mean_tie_pair_frac | mean_spearman | mean_kendall | mean_gamma | alloc_spearman_vs_oht |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.50 | 1.000 | 0.819 | 0.125 | 0.030 | 0.940 | 1.000 | 1.000 | 1.000 | 0.506 |
| 0.75 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | n/a | n/a | n/a | 0.506 |
| 0.85 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | n/a | n/a | n/a | 0.506 |

## Result Against Success Criteria
- `0.75` and `0.85` did not reduce degeneracy; both were fully low-information in aggregate (`mean_low_info = 1.000`).
- Valid pair coverage collapsed to `0.000` for both `0.75` and `0.85` (worse than `0.50` at `0.125`).
- Stability coefficients become non-evaluable (`n/a`) at `0.75/0.85` due flat vectors, so interpretability decreases.
- Apparent policy agreement vs OHT remains high but is not interpretable under full degeneracy.

## Bounded Decision
- Do not expand rank-reduction further at this point in this regime.
- Keep `gradient` as operational default proxy.
- Keep `attenuate` as explanatory ablation companion.
- Keep `hard_zero` as a simple sanity probe.

## Source Artifacts
- `rr050`: `/Users/john/code/gradience/field_trials/rank_proxy_validation/rank_reduction_soft_ablation_pilot_sweep.json`
- `rr075`: `/Users/john/code/gradience/field_trials/rank_proxy_validation/rank_reduction_soft_ablation_rr075_sweep.json`
- `rr085`: `/Users/john/code/gradience/field_trials/rank_proxy_validation/rank_reduction_soft_ablation_rr085_sweep.json`
