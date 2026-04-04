# Rank Proxy Validation v2 Compression Evaluation

## Primary Informative Summary by Method
| method | n | mean_delta_vs_full | mean_delta_vs_uniform | mean_delta_vs_random | mean_realized_budget |
| --- | --- | --- | --- | --- | --- |
| energy_90 | 18 | -0.0035 | 0.0067 | -0.0023 | 0.494 |
| erank | 18 | -0.0081 | 0.0020 | -0.0069 | 0.494 |
| knee | 18 | -0.0017 | 0.0084 | -0.0006 | 0.494 |
| oht | 18 | -0.0003 | 0.0098 | 0.0009 | 0.494 |
| proxy_ablation_attenuate | 18 | -0.0078 | 0.0023 | -0.0067 | 0.494 |
| proxy_gradient | 18 | 0.0017 | 0.0119 | 0.0029 | 0.494 |
| random_matched_budget | 18 | -0.0012 | 0.0090 | 0.0000 | 0.494 |
| stable_rank_ceil | 18 | -0.0101 | 0.0000 | -0.0090 | 0.494 |
| uniform | 18 | -0.0101 | 0.0000 | -0.0090 | 0.494 |

- Lead spectral policy in primary subset: `oht`
