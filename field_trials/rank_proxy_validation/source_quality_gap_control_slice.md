# Source-Quality-Gap Control Slice

## Analysis Scope
- primary informative families: sst2, imdb
- non-informative (saturated) families: tweet_eval, ag_news

## Definition
- dataset-matched source baseline: gap is measured against best full-adapter accuracy within the same dataset.
- gap metric: `best_full_adapter_accuracy(dataset) - full_adapter_accuracy(adapter)`.
- bands: `near_top` (<=0.01), `mid_gap` (0.01-0.05), `large_gap` (>0.05), `single_source_dataset` (<2 adapters in dataset).

## Band Coverage (Informative Subset)
| gap_band | adapters | dataset_count | datasets |
| --- | --- | --- | --- |
| near_top | 3 | 2 | imdb, sst2 |
| mid_gap | 1 | 1 | sst2 |
| large_gap | 2 | 2 | imdb, sst2 |

## Adapter Quality Table (Informative Subset)
| adapter_id | dataset | task_family | full_acc | dataset_best_acc | gap_vs_dataset_best | dataset_adapter_count | gap_band |
| --- | --- | --- | --- | --- | --- | --- | --- |
| NightPrince/peft-distilbert-sst2 | sst2 | sst2 | 0.7083 | 0.9062 | 0.1979 | 3 | large_gap |
| myselfmankar/distilbert-base-sst2-lora | sst2 | sst2 | 0.8698 | 0.9062 | 0.0365 | 3 | mid_gap |
| rambodazimi/distilbert-base-uncased-finetuned-LoRA-SST2 | sst2 | sst2 | 0.9062 | 0.9062 | 0.0000 | 3 | near_top |
| RAJESHCHAUHAN101/distilbert-base-uncased-lora-text-classification | imdb | imdb | 0.8594 | 0.8646 | 0.0052 | 3 | near_top |
| dipanjanS/distilbert-lora-finetuned-unmerged-imdb-sentiment | imdb | imdb | 0.8646 | 0.8646 | 0.0000 | 3 | near_top |
| wt-golf/distilbert-base-uncased-lora-text-classification-imdb-1k | imdb | imdb | 0.8125 | 0.8646 | 0.0521 | 3 | large_gap |

## Compression by Gap Band and Method (Informative Subset)
| gap_band | method | n | mean_delta_vs_full | mean_delta_vs_uniform | mean_compressed_acc |
| --- | --- | --- | --- | --- | --- |
| near_top | baseline_random | 9 | -0.0104 | -0.0017 | 0.8663 |
| near_top | baseline_uniform | 9 | -0.0087 | 0.0000 | 0.8681 |
| near_top | energy_90 | 9 | -0.0081 | 0.0006 | 0.8686 |
| near_top | erank | 9 | -0.0081 | 0.0006 | 0.8686 |
| near_top | knee | 9 | -0.0075 | 0.0012 | 0.8692 |
| near_top | oht | 9 | -0.0081 | 0.0006 | 0.8686 |
| near_top | proxy_ablation | 9 | -0.0041 | 0.0046 | 0.8727 |
| near_top | proxy_gradient | 9 | -0.0035 | 0.0052 | 0.8733 |
| near_top | stable_rank_ceil | 9 | -0.0087 | 0.0000 | 0.8681 |
| mid_gap | baseline_random | 3 | 0.0017 | 0.0590 | 0.8715 |
| mid_gap | baseline_uniform | 3 | -0.0573 | 0.0000 | 0.8125 |
| mid_gap | energy_90 | 3 | -0.0191 | 0.0382 | 0.8507 |
| mid_gap | erank | 3 | -0.0399 | 0.0174 | 0.8299 |
| mid_gap | knee | 3 | -0.0104 | 0.0469 | 0.8594 |
| mid_gap | oht | 3 | 0.0000 | 0.0573 | 0.8698 |
| mid_gap | proxy_ablation | 3 | -0.0573 | 0.0000 | 0.8125 |
| mid_gap | proxy_gradient | 3 | 0.0000 | 0.0573 | 0.8698 |
| mid_gap | stable_rank_ceil | 3 | -0.0573 | 0.0000 | 0.8125 |
| large_gap | baseline_random | 6 | 0.0113 | 0.0000 | 0.7717 |
| large_gap | baseline_uniform | 6 | 0.0113 | 0.0000 | 0.7717 |
| large_gap | energy_90 | 6 | 0.0113 | 0.0000 | 0.7717 |
| large_gap | erank | 6 | 0.0078 | -0.0035 | 0.7682 |
| large_gap | knee | 6 | 0.0113 | 0.0000 | 0.7717 |
| large_gap | oht | 6 | 0.0113 | 0.0000 | 0.7717 |
| large_gap | proxy_ablation | 6 | 0.0113 | 0.0000 | 0.7717 |
| large_gap | proxy_gradient | 6 | 0.0104 | -0.0009 | 0.7708 |
| large_gap | stable_rank_ceil | 6 | 0.0113 | 0.0000 | 0.7717 |

## Policy-Proxy Agreement by Gap Band (Informative Subset)
| gap_band | policy | proxy | n | mean_spearman | mean_topk_overlap | mean_abs_rank_dev | mean_policy_minus_proxy_acc |
| --- | --- | --- | --- | --- | --- | --- | --- |
| near_top | energy_90 | proxy_ablation | 9 | -0.097 | 0.259 | 4.167 | -0.004 |
| near_top | energy_90 | proxy_gradient | 9 | -0.330 | 0.111 | 4.296 | -0.005 |
| near_top | erank | proxy_ablation | 9 | -0.116 | 0.259 | 4.278 | -0.004 |
| near_top | erank | proxy_gradient | 9 | -0.320 | 0.111 | 4.296 | -0.005 |
| near_top | knee | proxy_ablation | 9 | -0.058 | 0.370 | 4.000 | -0.003 |
| near_top | knee | proxy_gradient | 9 | -0.386 | 0.074 | 4.889 | -0.004 |
| near_top | oht | proxy_ablation | 9 | -0.049 | 0.296 | 3.759 | -0.004 |
| near_top | oht | proxy_gradient | 9 | -0.341 | 0.074 | 4.796 | -0.005 |
| near_top | stable_rank_ceil | proxy_ablation | 9 | -0.089 | 0.333 | 4.019 | -0.005 |
| near_top | stable_rank_ceil | proxy_gradient | 9 | -0.345 | 0.111 | 4.611 | -0.005 |
| mid_gap | energy_90 | proxy_ablation | 3 | -0.385 | 0.444 | 9.333 | 0.038 |
| mid_gap | energy_90 | proxy_gradient | 3 | -0.214 | 0.000 | 8.333 | -0.019 |
| mid_gap | erank | proxy_ablation | 3 | 0.108 | 0.556 | 6.167 | 0.017 |
| mid_gap | erank | proxy_gradient | 3 | -0.434 | 0.000 | 9.722 | -0.040 |
| mid_gap | knee | proxy_ablation | 3 | 0.135 | 0.667 | 6.000 | 0.047 |
| mid_gap | knee | proxy_gradient | 3 | -0.408 | 0.000 | 9.389 | -0.010 |
| mid_gap | oht | proxy_ablation | 3 | -0.103 | 0.556 | 7.222 | 0.057 |
| mid_gap | oht | proxy_gradient | 3 | 0.456 | 0.556 | 4.222 | 0.000 |
| mid_gap | stable_rank_ceil | proxy_ablation | 3 | 0.280 | 0.667 | 5.056 | 0.000 |
| mid_gap | stable_rank_ceil | proxy_gradient | 3 | -0.415 | 0.000 | 9.611 | -0.057 |
| large_gap | energy_90 | proxy_ablation | 6 | 0.798 | 0.944 | 0.611 | 0.000 |
| large_gap | energy_90 | proxy_gradient | 6 | -0.576 | 0.056 | 3.444 | 0.001 |
| large_gap | erank | proxy_ablation | 6 | 0.571 | 0.944 | 0.944 | -0.003 |
| large_gap | erank | proxy_gradient | 6 | -0.432 | 0.056 | 3.250 | -0.003 |
| large_gap | knee | proxy_ablation | 6 | 0.831 | 0.944 | 0.583 | 0.000 |
| large_gap | knee | proxy_gradient | 6 | -0.375 | 0.056 | 2.722 | 0.001 |
| large_gap | oht | proxy_ablation | 6 | 0.856 | 0.944 | 0.556 | 0.000 |
| large_gap | oht | proxy_gradient | 6 | -0.301 | 0.111 | 2.528 | 0.001 |
| large_gap | stable_rank_ceil | proxy_ablation | 6 | 0.770 | 0.889 | 0.778 | 0.000 |
| large_gap | stable_rank_ceil | proxy_gradient | 6 | -0.490 | 0.056 | 3.083 | 0.001 |

## Non-Informative Context (All Families)
- Families with realized budget near 1.0 are reported for completeness but are excluded from primary interpretation.
| gap_band | adapters | dataset_count | datasets |
| --- | --- | --- | --- |
| near_top | 5 | 4 | imdb, sst2, tweet_eval/emotion, tweet_eval/irony |
| mid_gap | 3 | 2 | sst2, tweet_eval/irony |
| large_gap | 3 | 3 | imdb, sst2, tweet_eval/emotion |
| single_source_dataset | 1 | 1 | ag_news |
