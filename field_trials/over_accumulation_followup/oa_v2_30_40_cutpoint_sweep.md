# Over-Accumulation Cutpoint Sweep

## Threshold Grid
| threshold | activated_pairs | pair_fraction | activated_layers | layer_fraction | rerun_activated_n | rerun_mean_delta_vs_best | rerun_var_delta_vs_best | rerun_task_family_mix |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.10 | 55 | 0.2910 | 258 | 0.1118 | 13 | -0.0666 | 0.0224 | {'sentiment_binary': 6, 'tweet_eval': 5, 'topic_classification': 2} |
| 0.15 | 27 | 0.1429 | 140 | 0.0607 | 10 | -0.0690 | 0.0279 | {'sentiment_binary': 5, 'tweet_eval': 4, 'topic_classification': 1} |
| 0.20 | 17 | 0.0899 | 89 | 0.0386 | 5 | -0.0200 | 0.0003 | {'sentiment_binary': 4, 'tweet_eval': 1} |
| 0.25 | 8 | 0.0423 | 57 | 0.0247 | 2 | -0.0150 | 0.0000 | {'sentiment_binary': 2} |
| 0.30 | 7 | 0.0370 | 53 | 0.0230 | 2 | -0.0150 | 0.0000 | {'sentiment_binary': 2} |
| 0.35 | 4 | 0.0212 | 43 | 0.0186 | 1 | -0.0220 | 0.0000 | {'sentiment_binary': 1} |
| 0.40 | 1 | 0.0053 | 40 | 0.0173 | 0 | n/a | n/a | {} |
| 0.50 | 1 | 0.0053 | 24 | 0.0104 | 0 | n/a | n/a | {} |
| 0.60 | 1 | 0.0053 | 7 | 0.0030 | 0 | n/a | n/a | {} |

## Top-Tail Outcome Buckets
| bucket | n | mean_delta_vs_best | var_delta_vs_best | task_family_mix |
| --- | --- | --- | --- | --- |
| lt_0_15 | 3 | -0.0587 | 0.0042 | {'sentiment_binary': 1, 'topic_classification': 1, 'tweet_eval': 1} |
| 0_15_to_0_25 | 8 | -0.0825 | 0.0339 | {'tweet_eval': 4, 'sentiment_binary': 3, 'topic_classification': 1} |
| 0_25_to_0_35 | 1 | -0.0080 | 0.0000 | {'sentiment_binary': 1} |
| ge_0_35 | 1 | -0.0220 | 0.0000 | {'sentiment_binary': 1} |

## Notes
- Activated pair/layer rates are computed over full activation-audit inventory.
- Outcome stats are computed over strict-naive rerun subset only.
