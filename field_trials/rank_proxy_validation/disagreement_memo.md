# Rank Allocation Disagreement Memo

## Scope
- adapters: 12
- budgets: [0.35, 0.5, 0.65]
- primary informative families: sst2, imdb
- non-informative families: tweet_eval, ag_news

## Policy vs Proxy Summary (Informative Families Only)
| policy | proxy | n | mean_spearman | mean_topk_overlap | mean_abs_rank_dev | mean_policy_minus_proxy_acc |
| --- | --- | --- | --- | --- | --- | --- |
| energy_90 | proxy_ablation | 18 | 0.154 | 0.519 | 3.843 | 0.004 |
| energy_90 | proxy_gradient | 18 | -0.392 | 0.074 | 4.685 | -0.005 |
| erank | proxy_ablation | 18 | 0.150 | 0.537 | 3.481 | -0.000 |
| erank | proxy_gradient | 18 | -0.376 | 0.074 | 4.852 | -0.010 |
| knee | proxy_ablation | 18 | 0.271 | 0.611 | 3.194 | 0.006 |
| knee | proxy_gradient | 18 | -0.386 | 0.056 | 4.917 | -0.003 |
| oht | proxy_ablation | 18 | 0.244 | 0.556 | 3.269 | 0.008 |
| oht | proxy_gradient | 18 | -0.195 | 0.167 | 3.944 | -0.002 |
| stable_rank_ceil | proxy_ablation | 18 | 0.259 | 0.574 | 3.111 | -0.002 |
| stable_rank_ceil | proxy_gradient | 18 | -0.405 | 0.074 | 4.935 | -0.012 |

## Compression Method Summary (Informative Families Only)
| method | n | mean_delta_vs_full | mean_delta_vs_uniform | mean_realized_budget |
| --- | --- | --- | --- | --- |
| baseline_random | 18 | -0.0012 | 0.0090 | 0.494 |
| baseline_uniform | 18 | -0.0101 | 0.0000 | 0.494 |
| energy_90 | 18 | -0.0035 | 0.0067 | 0.494 |
| erank | 18 | -0.0081 | 0.0020 | 0.494 |
| knee | 18 | -0.0017 | 0.0084 | 0.494 |
| oht | 18 | -0.0003 | 0.0098 | 0.494 |
| proxy_ablation | 18 | -0.0078 | 0.0023 | 0.494 |
| proxy_gradient | 18 | 0.0017 | 0.0119 | 0.494 |
| stable_rank_ceil | 18 | -0.0101 | 0.0000 | 0.494 |

## Policy vs Proxy Summary
| policy | proxy | n | mean_spearman | mean_topk_overlap | mean_abs_rank_dev | mean_policy_minus_proxy_acc |
| --- | --- | --- | --- | --- | --- | --- |
| energy_90 | proxy_ablation | 36 | 0.077 | 0.759 | 1.921 | 0.002 |
| energy_90 | proxy_gradient | 36 | -0.196 | 0.537 | 2.343 | -0.003 |
| erank | proxy_ablation | 36 | 0.075 | 0.769 | 1.741 | -0.000 |
| erank | proxy_gradient | 36 | -0.188 | 0.537 | 2.426 | -0.005 |
| knee | proxy_ablation | 36 | 0.135 | 0.806 | 1.597 | 0.003 |
| knee | proxy_gradient | 36 | -0.193 | 0.528 | 2.458 | -0.002 |
| oht | proxy_ablation | 36 | 0.122 | 0.778 | 1.634 | 0.004 |
| oht | proxy_gradient | 36 | -0.098 | 0.583 | 1.972 | -0.001 |
| stable_rank_ceil | proxy_ablation | 36 | 0.129 | 0.787 | 1.556 | -0.001 |
| stable_rank_ceil | proxy_gradient | 36 | -0.203 | 0.537 | 2.468 | -0.006 |

## Compression Method Summary
| method | n | mean_delta_vs_full | mean_delta_vs_uniform | mean_realized_budget |
| --- | --- | --- | --- | --- |
| baseline_random | 36 | -0.0006 | 0.0045 | 0.747 |
| baseline_uniform | 36 | -0.0051 | 0.0000 | 0.747 |
| energy_90 | 36 | -0.0017 | 0.0033 | 0.747 |
| erank | 36 | -0.0041 | 0.0010 | 0.747 |
| knee | 36 | -0.0009 | 0.0042 | 0.747 |
| oht | 36 | -0.0001 | 0.0049 | 0.747 |
| proxy_ablation | 36 | -0.0039 | 0.0012 | 0.747 |
| proxy_gradient | 36 | 0.0009 | 0.0059 | 0.747 |
| stable_rank_ceil | 36 | -0.0051 | 0.0000 | 0.747 |

## Interpretation
- This pass is a bounded CPU proxy-comparison study. It tests whether cheap spectral suggestions recover useful layerwise structure and compression behavior under fixed budgets.
- Stronger control-corrected claims should wait for a larger matched cohort with broader top-tail budget representation.
