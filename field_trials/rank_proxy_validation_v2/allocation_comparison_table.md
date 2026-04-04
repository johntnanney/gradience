# Rank Proxy Validation v2 Allocation Comparison

## Primary Informative Summary
| policy_method | proxy_method | n | mean_spearman | mean_topk_overlap | mean_policy_minus_proxy_acc |
| --- | --- | --- | --- | --- | --- |
| energy_90 | proxy_ablation_attenuate | 18 | 0.154 | 0.519 | 0.0043 |
| energy_90 | proxy_gradient | 18 | -0.392 | 0.074 | -0.0052 |
| erank | proxy_ablation_attenuate | 18 | 0.150 | 0.537 | -0.0003 |
| erank | proxy_gradient | 18 | -0.376 | 0.074 | -0.0098 |
| knee | proxy_ablation_attenuate | 18 | 0.271 | 0.611 | 0.0061 |
| knee | proxy_gradient | 18 | -0.386 | 0.056 | -0.0035 |
| oht | proxy_ablation_attenuate | 18 | 0.244 | 0.556 | 0.0075 |
| oht | proxy_gradient | 18 | -0.195 | 0.167 | -0.0020 |
| stable_rank_ceil | proxy_ablation_attenuate | 18 | 0.259 | 0.574 | -0.0023 |
| stable_rank_ceil | proxy_gradient | 18 | -0.405 | 0.074 | -0.0119 |
