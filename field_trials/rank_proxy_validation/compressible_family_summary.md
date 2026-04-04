# Compressible-Family-Only Summary

## Scope
- informative families: sst2, imdb
- non-informative families: tweet_eval, ag_news
- compression rows (informative): 162
- allocation rows (informative): 180

## Method Performance
| method | n | mean_delta_vs_uniform | var_delta_vs_uniform | mean_delta_vs_full | var_delta_vs_full | mean_realized_budget |
| --- | --- | --- | --- | --- | --- | --- |
| proxy_gradient | 18 | 0.0119 | 0.000452 | 0.0017 | 0.000090 | 0.494 |
| oht | 18 | 0.0098 | 0.000455 | -0.0003 | 0.000152 | 0.494 |
| baseline_random | 18 | 0.0090 | 0.000557 | -0.0012 | 0.000258 | 0.494 |
| knee | 18 | 0.0084 | 0.000305 | -0.0017 | 0.000172 | 0.494 |
| energy_90 | 18 | 0.0067 | 0.000385 | -0.0035 | 0.000440 | 0.494 |
| proxy_ablation | 18 | 0.0023 | 0.000022 | -0.0078 | 0.000628 | 0.494 |
| erank | 18 | 0.0020 | 0.000190 | -0.0081 | 0.000486 | 0.494 |
| stable_rank_ceil | 18 | 0.0000 | 0.000003 | -0.0101 | 0.000616 | 0.494 |
| baseline_uniform | 18 | 0.0000 | 0.000000 | -0.0101 | 0.000610 | 0.494 |

## Policy-Proxy Agreement
| policy | proxy | n | mean_spearman | mean_topk_overlap | mean_policy_minus_proxy_acc |
| --- | --- | --- | --- | --- | --- |
| energy_90 | proxy_gradient | 18 | -0.392 | 0.074 | -0.005 |
| energy_90 | proxy_ablation | 18 | 0.154 | 0.519 | 0.004 |
| knee | proxy_gradient | 18 | -0.386 | 0.056 | -0.003 |
| knee | proxy_ablation | 18 | 0.271 | 0.611 | 0.006 |
| erank | proxy_gradient | 18 | -0.376 | 0.074 | -0.010 |
| erank | proxy_ablation | 18 | 0.150 | 0.537 | -0.000 |
| oht | proxy_gradient | 18 | -0.195 | 0.167 | -0.002 |
| oht | proxy_ablation | 18 | 0.244 | 0.556 | 0.008 |
| stable_rank_ceil | proxy_gradient | 18 | -0.405 | 0.074 | -0.012 |
| stable_rank_ceil | proxy_ablation | 18 | 0.259 | 0.574 | -0.002 |

## Proxy Agreement Split
| proxy | n | mean_spearman | mean_topk_overlap | mean_policy_minus_proxy_acc |
| --- | --- | --- | --- | --- |
| proxy_ablation | 90 | 0.215 | 0.559 | 0.003 |
| proxy_gradient | 90 | -0.351 | 0.089 | -0.006 |
