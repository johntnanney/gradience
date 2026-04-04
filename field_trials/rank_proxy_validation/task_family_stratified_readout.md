# Task-Family Stratified Readout

## Scope
- adapters: 12
- budgets: [0.35, 0.5, 0.65]
- primary informative families: sst2, imdb
- non-informative (saturated) families: tweet_eval, ag_news

## Family Coverage
| task_family | adapters | dataset_count | rows | effective_compression | mean_realized_budget | datasets |
| --- | --- | --- | --- | --- | --- | --- |
| sst2 | 3 | 1 | 81 | yes | 0.498 | sst2 |
| imdb | 3 | 1 | 81 | yes | 0.490 | imdb |
| tweet_eval | 5 | 2 | 135 | no | 1.000 | tweet_eval/emotion, tweet_eval/irony |
| ag_news | 1 | 1 | 27 | no | 1.000 | ag_news |

## Best Method by Family (mean delta vs uniform)
| task_family | best_method | mean_delta_vs_uniform | n |
| --- | --- | --- | --- |
| sst2 | proxy_gradient | 0.0237 | 9 |
| imdb | proxy_ablation | 0.0023 | 9 |
| tweet_eval | baseline_random | 0.0000 | 15 |
| ag_news | baseline_random | 0.0000 | 3 |

## Compression by Family and Method
| task_family | method | n | mean_delta_vs_full | mean_delta_vs_uniform | mean_compressed_acc | mean_realized_budget |
| --- | --- | --- | --- | --- | --- | --- |
| sst2 | baseline_random | 9 | -0.0081 | 0.0174 | 0.8200 | 0.498 |
| sst2 | baseline_uniform | 9 | -0.0255 | 0.0000 | 0.8027 | 0.498 |
| sst2 | energy_90 | 9 | -0.0139 | 0.0116 | 0.8142 | 0.498 |
| sst2 | erank | 9 | -0.0208 | 0.0046 | 0.8073 | 0.498 |
| sst2 | knee | 9 | -0.0087 | 0.0168 | 0.8194 | 0.498 |
| sst2 | oht | 9 | -0.0058 | 0.0197 | 0.8223 | 0.498 |
| sst2 | proxy_ablation | 9 | -0.0231 | 0.0023 | 0.8050 | 0.498 |
| sst2 | proxy_gradient | 9 | -0.0017 | 0.0237 | 0.8264 | 0.498 |
| sst2 | stable_rank_ceil | 9 | -0.0255 | 0.0000 | 0.8027 | 0.498 |
| imdb | baseline_random | 9 | 0.0058 | 0.0006 | 0.8513 | 0.490 |
| imdb | baseline_uniform | 9 | 0.0052 | 0.0000 | 0.8507 | 0.490 |
| imdb | energy_90 | 9 | 0.0069 | 0.0017 | 0.8524 | 0.490 |
| imdb | erank | 9 | 0.0046 | -0.0006 | 0.8501 | 0.490 |
| imdb | knee | 9 | 0.0052 | 0.0000 | 0.8507 | 0.490 |
| imdb | oht | 9 | 0.0052 | 0.0000 | 0.8507 | 0.490 |
| imdb | proxy_ablation | 9 | 0.0075 | 0.0023 | 0.8530 | 0.490 |
| imdb | proxy_gradient | 9 | 0.0052 | -0.0000 | 0.8507 | 0.490 |
| imdb | stable_rank_ceil | 9 | 0.0052 | 0.0000 | 0.8507 | 0.490 |
| tweet_eval | baseline_random | 15 | 0.0000 | 0.0000 | 0.6646 | 1.000 |
| tweet_eval | baseline_uniform | 15 | 0.0000 | 0.0000 | 0.6646 | 1.000 |
| tweet_eval | energy_90 | 15 | 0.0000 | 0.0000 | 0.6646 | 1.000 |
| tweet_eval | erank | 15 | 0.0000 | 0.0000 | 0.6646 | 1.000 |
| tweet_eval | knee | 15 | 0.0000 | 0.0000 | 0.6646 | 1.000 |
| tweet_eval | oht | 15 | 0.0000 | 0.0000 | 0.6646 | 1.000 |
| tweet_eval | proxy_ablation | 15 | 0.0000 | 0.0000 | 0.6646 | 1.000 |
| tweet_eval | proxy_gradient | 15 | 0.0000 | 0.0000 | 0.6646 | 1.000 |
| tweet_eval | stable_rank_ceil | 15 | 0.0000 | 0.0000 | 0.6646 | 1.000 |
| ag_news | baseline_random | 3 | 0.0000 | 0.0000 | 0.8906 | 1.000 |
| ag_news | baseline_uniform | 3 | 0.0000 | 0.0000 | 0.8906 | 1.000 |
| ag_news | energy_90 | 3 | 0.0000 | 0.0000 | 0.8906 | 1.000 |
| ag_news | erank | 3 | 0.0000 | 0.0000 | 0.8906 | 1.000 |
| ag_news | knee | 3 | 0.0000 | 0.0000 | 0.8906 | 1.000 |
| ag_news | oht | 3 | 0.0000 | 0.0000 | 0.8906 | 1.000 |
| ag_news | proxy_ablation | 3 | 0.0000 | 0.0000 | 0.8906 | 1.000 |
| ag_news | proxy_gradient | 3 | 0.0000 | 0.0000 | 0.8906 | 1.000 |
| ag_news | stable_rank_ceil | 3 | 0.0000 | 0.0000 | 0.8906 | 1.000 |

## Policy-Proxy Agreement by Family
| task_family | policy | proxy | n | mean_spearman | mean_topk_overlap | mean_abs_rank_dev | mean_policy_minus_proxy_acc |
| --- | --- | --- | --- | --- | --- | --- | --- |
| sst2 | energy_90 | proxy_ablation | 9 | -0.032 | 0.481 | 6.519 | 0.009 |
| sst2 | energy_90 | proxy_gradient | 9 | -0.228 | 0.148 | 6.556 | -0.012 |
| sst2 | erank | proxy_ablation | 9 | 0.138 | 0.519 | 5.463 | 0.002 |
| sst2 | erank | proxy_gradient | 9 | -0.280 | 0.148 | 7.000 | -0.019 |
| sst2 | knee | proxy_ablation | 9 | 0.185 | 0.630 | 5.222 | 0.014 |
| sst2 | knee | proxy_gradient | 9 | -0.252 | 0.111 | 7.000 | -0.007 |
| sst2 | oht | proxy_ablation | 9 | 0.181 | 0.556 | 5.241 | 0.017 |
| sst2 | oht | proxy_gradient | 9 | 0.023 | 0.333 | 5.426 | -0.004 |
| sst2 | stable_rank_ceil | proxy_ablation | 9 | 0.211 | 0.593 | 4.926 | -0.002 |
| sst2 | stable_rank_ceil | proxy_gradient | 9 | -0.290 | 0.148 | 7.037 | -0.024 |
| imdb | energy_90 | proxy_ablation | 9 | 0.339 | 0.556 | 1.167 | -0.001 |
| imdb | energy_90 | proxy_gradient | 9 | -0.556 | 0.000 | 2.815 | 0.002 |
| imdb | erank | proxy_ablation | 9 | 0.163 | 0.556 | 1.500 | -0.003 |
| imdb | erank | proxy_gradient | 9 | -0.472 | 0.000 | 2.704 | -0.001 |
| imdb | knee | proxy_ablation | 9 | 0.356 | 0.593 | 1.167 | -0.002 |
| imdb | knee | proxy_gradient | 9 | -0.520 | 0.000 | 2.833 | 0.000 |
| imdb | oht | proxy_ablation | 9 | 0.307 | 0.556 | 1.296 | -0.002 |
| imdb | oht | proxy_gradient | 9 | -0.413 | 0.000 | 2.463 | 0.000 |
| imdb | stable_rank_ceil | proxy_ablation | 9 | 0.307 | 0.556 | 1.296 | -0.002 |
| imdb | stable_rank_ceil | proxy_gradient | 9 | -0.520 | 0.000 | 2.833 | 0.000 |
| tweet_eval | energy_90 | proxy_ablation | 15 | 0.000 | 1.000 | 0.000 | 0.000 |
| tweet_eval | energy_90 | proxy_gradient | 15 | 0.000 | 1.000 | 0.000 | 0.000 |
| tweet_eval | erank | proxy_ablation | 15 | 0.000 | 1.000 | 0.000 | 0.000 |
| tweet_eval | erank | proxy_gradient | 15 | 0.000 | 1.000 | 0.000 | 0.000 |
| tweet_eval | knee | proxy_ablation | 15 | 0.000 | 1.000 | 0.000 | 0.000 |
| tweet_eval | knee | proxy_gradient | 15 | 0.000 | 1.000 | 0.000 | 0.000 |
| tweet_eval | oht | proxy_ablation | 15 | 0.000 | 1.000 | 0.000 | 0.000 |
| tweet_eval | oht | proxy_gradient | 15 | 0.000 | 1.000 | 0.000 | 0.000 |
| tweet_eval | stable_rank_ceil | proxy_ablation | 15 | 0.000 | 1.000 | 0.000 | 0.000 |
| tweet_eval | stable_rank_ceil | proxy_gradient | 15 | 0.000 | 1.000 | 0.000 | 0.000 |
| ag_news | energy_90 | proxy_ablation | 3 | 0.000 | 1.000 | 0.000 | 0.000 |
| ag_news | energy_90 | proxy_gradient | 3 | 0.000 | 1.000 | 0.000 | 0.000 |
| ag_news | erank | proxy_ablation | 3 | 0.000 | 1.000 | 0.000 | 0.000 |
| ag_news | erank | proxy_gradient | 3 | 0.000 | 1.000 | 0.000 | 0.000 |
| ag_news | knee | proxy_ablation | 3 | 0.000 | 1.000 | 0.000 | 0.000 |
| ag_news | knee | proxy_gradient | 3 | 0.000 | 1.000 | 0.000 | 0.000 |
| ag_news | oht | proxy_ablation | 3 | 0.000 | 1.000 | 0.000 | 0.000 |
| ag_news | oht | proxy_gradient | 3 | 0.000 | 1.000 | 0.000 | 0.000 |
| ag_news | stable_rank_ceil | proxy_ablation | 3 | 0.000 | 1.000 | 0.000 | 0.000 |
| ag_news | stable_rank_ceil | proxy_gradient | 3 | 0.000 | 1.000 | 0.000 | 0.000 |
