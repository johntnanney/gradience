# Rank Proxy Validation v2 Task-Family Stratified Readout

- Primary informative families: imdb, sst2
- Secondary context families: ag_news, tweet_eval

## Best Method by Family (Primary Informative)
| task_family | best_method | mean_delta_vs_uniform | n |
| --- | --- | --- | --- |
| sst2 | proxy_gradient | 0.0237 | 9 |
| imdb | proxy_ablation_attenuate | 0.0023 | 9 |
