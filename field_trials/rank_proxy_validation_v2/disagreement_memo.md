# Rank Proxy Validation v2 Disagreement Memo

## Scope
- Primary informative families: imdb, sst2
- Secondary context families: ag_news, tweet_eval
- Compression rows (primary): 162
- Allocation comparison rows (primary): 180

## Gradient vs OHT by Source-Quality Band
| gap_band | oht_mean_delta_vs_uniform | gradient_mean_delta_vs_uniform | gradient_minus_oht |
| --- | --- | --- | --- |
| large_gap | 0.0000 | -0.0009 | -0.0009 |
| mid_gap | 0.0573 | 0.0573 | 0.0000 |
| near_top | 0.0006 | 0.0052 | 0.0046 |

## OHT Structural Alignment by Family
| task_family | oht_vs_ablation_spearman | oht_vs_gradient_spearman | oht_vs_ablation_topk | oht_vs_gradient_topk |
| --- | --- | --- | --- | --- |
| imdb | 0.307 | -0.413 | 0.556 | 0.000 |
| sst2 | 0.181 | 0.023 | 0.556 | 0.333 |

## Interpretation
- Structural similarity and operational superiority are distinct: OHT can align more with ablation-style structure while gradient remains stronger on mean compression outcome in this CPU-bounded setup.
- Source-quality control remains necessary: near-top and mid-gap bands can show different method ordering from large-gap bands.
- Saturated families remain non-informative for primary policy interpretation.
