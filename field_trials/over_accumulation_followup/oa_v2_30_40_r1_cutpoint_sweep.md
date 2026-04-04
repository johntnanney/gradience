# Over-Accumulation Cutpoint Sweep

## Threshold Grid
| threshold | activated_pairs | pair_fraction | activated_layers | layer_fraction | rerun_activated_n | rerun_mean_delta_vs_best | rerun_var_delta_vs_best | rerun_task_family_mix |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.10 | 55 | 0.2910 | 258 | 0.1118 | 21 | -0.0973 | 0.0242 | {'sentiment_binary': 13, 'topic_classification': 2, 'tweet_eval': 6} |
| 0.15 | 27 | 0.1429 | 140 | 0.0607 | 13 | -0.1078 | 0.0332 | {'sentiment_binary': 8, 'topic_classification': 1, 'tweet_eval': 4} |
| 0.20 | 17 | 0.0899 | 89 | 0.0386 | 8 | -0.1015 | 0.0222 | {'sentiment_binary': 7, 'tweet_eval': 1} |
| 0.25 | 8 | 0.0423 | 57 | 0.0247 | 2 | -0.0140 | 0.0001 | {'sentiment_binary': 2} |
| 0.30 | 7 | 0.0370 | 53 | 0.0230 | 2 | -0.0140 | 0.0001 | {'sentiment_binary': 2} |
| 0.35 | 4 | 0.0212 | 43 | 0.0186 | 1 | -0.0220 | 0.0000 | {'sentiment_binary': 1} |
| 0.40 | 1 | 0.0053 | 40 | 0.0173 | 0 | n/a | n/a | {} |
| 0.50 | 1 | 0.0053 | 24 | 0.0104 | 0 | n/a | n/a | {} |
| 0.60 | 1 | 0.0053 | 7 | 0.0030 | 0 | n/a | n/a | {} |

## Top-Tail Outcome Buckets
| bucket | n | mean_delta_vs_best | var_delta_vs_best | task_family_mix |
| --- | --- | --- | --- | --- |
| lt_0_15 | 17 | -0.1059 | 0.0172 | {'sentiment_binary': 14, 'tweet_eval': 2, 'topic_classification': 1} |
| 0_15_to_0_25 | 11 | -0.1249 | 0.0373 | {'topic_classification': 1, 'tweet_eval': 4, 'sentiment_binary': 6} |
| 0_25_to_0_35 | 1 | -0.0060 | 0.0000 | {'sentiment_binary': 1} |
| ge_0_35 | 1 | -0.0220 | 0.0000 | {'sentiment_binary': 1} |

## Notes
- Activated pair/layer rates are computed over full activation-audit inventory.
- Outcome stats are computed over strict-naive rerun subset only.
