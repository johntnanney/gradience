# Rank Proxy Validation v2 Compressible-Family Summary

- Informative families: imdb, sst2
- Non-informative context families: ag_news, tweet_eval

## Method Performance (Primary Informative)
| method | n | mean_delta_vs_uniform | var_delta_vs_uniform | mean_delta_vs_full |
| --- | --- | --- | --- | --- |
| energy_90 | 18 | 0.0067 | 0.000385 | -0.0035 |
| erank | 18 | 0.0020 | 0.000190 | -0.0081 |
| knee | 18 | 0.0084 | 0.000305 | -0.0017 |
| oht | 18 | 0.0098 | 0.000455 | -0.0003 |
| proxy_ablation_attenuate | 18 | 0.0023 | 0.000022 | -0.0078 |
| proxy_gradient | 18 | 0.0119 | 0.000452 | 0.0017 |
| random_matched_budget | 18 | 0.0090 | 0.000557 | -0.0012 |
| stable_rank_ceil | 18 | 0.0000 | 0.000003 | -0.0101 |
| uniform | 18 | 0.0000 | 0.000000 | -0.0101 |

## Proxy Agreement Split (Primary Informative)
| proxy_method | n | mean_spearman | mean_topk_overlap | mean_policy_minus_proxy_acc |
| --- | --- | --- | --- | --- |
| proxy_ablation_attenuate | 90 | 0.215 | 0.559 | 0.0031 |
| proxy_gradient | 90 | -0.351 | 0.089 | -0.0065 |
