# Source-Quality Gap Informative-Subset Analysis

## Scope
- informative families: sst2, imdb
- bands present: near_top, mid_gap, large_gap

## Method Behavior by Source-Quality Band
| gap_band | method | n | mean_delta_vs_uniform | mean_delta_vs_full |
| --- | --- | --- | --- | --- |
| near_top | baseline_random | 9 | -0.0017 | -0.0104 |
| near_top | baseline_uniform | 9 | 0.0000 | -0.0087 |
| near_top | energy_90 | 9 | 0.0006 | -0.0081 |
| near_top | erank | 9 | 0.0006 | -0.0081 |
| near_top | knee | 9 | 0.0012 | -0.0075 |
| near_top | oht | 9 | 0.0006 | -0.0081 |
| near_top | proxy_ablation | 9 | 0.0046 | -0.0041 |
| near_top | proxy_gradient | 9 | 0.0052 | -0.0035 |
| near_top | stable_rank_ceil | 9 | 0.0000 | -0.0087 |
| mid_gap | baseline_random | 3 | 0.0590 | 0.0017 |
| mid_gap | baseline_uniform | 3 | 0.0000 | -0.0573 |
| mid_gap | energy_90 | 3 | 0.0382 | -0.0191 |
| mid_gap | erank | 3 | 0.0174 | -0.0399 |
| mid_gap | knee | 3 | 0.0469 | -0.0104 |
| mid_gap | oht | 3 | 0.0573 | 0.0000 |
| mid_gap | proxy_ablation | 3 | 0.0000 | -0.0573 |
| mid_gap | proxy_gradient | 3 | 0.0573 | 0.0000 |
| mid_gap | stable_rank_ceil | 3 | 0.0000 | -0.0573 |
| large_gap | baseline_random | 6 | 0.0000 | 0.0113 |
| large_gap | baseline_uniform | 6 | 0.0000 | 0.0113 |
| large_gap | energy_90 | 6 | 0.0000 | 0.0113 |
| large_gap | erank | 6 | -0.0035 | 0.0078 |
| large_gap | knee | 6 | 0.0000 | 0.0113 |
| large_gap | oht | 6 | 0.0000 | 0.0113 |
| large_gap | proxy_ablation | 6 | 0.0000 | 0.0113 |
| large_gap | proxy_gradient | 6 | -0.0009 | 0.0104 |
| large_gap | stable_rank_ceil | 6 | 0.0000 | 0.0113 |

## Proxy-Gradient Concentration Check
| gap_band | proxy_gradient_dvu | oht_dvu | proxy_gradient_minus_oht |
| --- | --- | --- | --- |
| near_top | 0.0052 | 0.0006 | 0.0046 |
| mid_gap | 0.0573 | 0.0573 | 0.0000 |
| large_gap | -0.0009 | 0.0000 | -0.0009 |
- proxy_gradient near_top minus large_gap: 0.0061

## Spectral-vs-Ablation Alignment by Source-Quality Band
| gap_band | n | mean_spearman | mean_topk_overlap | mean_policy_minus_ablation_acc |
| --- | --- | --- | --- | --- |
| near_top | 5 | -0.082 | 0.304 | -0.004 |
| mid_gap | 5 | 0.007 | 0.578 | 0.032 |
| large_gap | 5 | 0.765 | 0.933 | -0.001 |

## Spectral-vs-Gradient Alignment by Source-Quality Band
| gap_band | n | mean_spearman | mean_topk_overlap | mean_policy_minus_gradient_acc |
| --- | --- | --- | --- | --- |
| near_top | 5 | -0.345 | 0.096 | -0.005 |
| mid_gap | 5 | -0.203 | 0.111 | -0.025 |
| large_gap | 5 | -0.435 | 0.067 | 0.000 |

## Caution
- Small sample sizes in `mid_gap` and `large_gap` should be treated as directional only.
