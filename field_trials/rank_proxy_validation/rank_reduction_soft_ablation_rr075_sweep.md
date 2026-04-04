# Ablation Reliability Sweep (Informative Subset)

## Scope
- informative families: sst2, imdb
- adapters analyzed: 4
- ablation modes: rank_reduction
- ablation sample grid: [24, 48, 72]
- fixed panels per setting: 3
- random repeats per setting: 3
- budgets: [0.35, 0.5, 0.65]
- low-info flags: max_unique<=2, min_nonzero_fraction<=0.2, high_tie_pair_fraction>=0.8

## Stability Summary
| mode | panel_type | ablation_samples | n_adapters | mean_spearman | mean_kendall_tau_b | mean_gamma | mean_topk_q25 | mean_topk_q50 | spearman_valid_pair_fraction | flat_vector_fraction | low_info_vector_fraction | high_tie_vector_fraction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rank_reduction | fixed | 24 | 4 | n/a | n/a | n/a | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | 1.000 |
| rank_reduction | fixed | 48 | 4 | n/a | n/a | n/a | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | 1.000 |
| rank_reduction | fixed | 72 | 4 | n/a | n/a | n/a | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | 1.000 |
| rank_reduction | random | 24 | 4 | n/a | n/a | n/a | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | 1.000 |
| rank_reduction | random | 48 | 4 | n/a | n/a | n/a | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | 1.000 |
| rank_reduction | random | 72 | 4 | n/a | n/a | n/a | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | 1.000 |

## Stability Improvement (Max vs Min Sample)
| mode | panel_type | sample_range | delta_spearman | delta_kendall_tau_b | delta_gamma | delta_topk_q25 | delta_topk_q50 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| rank_reduction | fixed | 24->72 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| rank_reduction | random | 24->72 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

## Policy Agreement vs OHT Summary
| mode | panel_type | ablation_samples | budget | n_adapters | mean_alloc_spearman_vs_oht | mean_alloc_kendall_vs_oht | mean_alloc_gamma_vs_oht | mean_alloc_topk_overlap_vs_oht | spearman_valid_panel_fraction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rank_reduction | fixed | 24 | 0.35 | 4 | 0.434 | 0.432 | 0.432 | 0.583 | 1.000 |
| rank_reduction | fixed | 24 | 0.50 | 4 | 0.519 | 0.500 | 0.577 | 0.833 | 1.000 |
| rank_reduction | fixed | 24 | 0.65 | 4 | 0.566 | 0.545 | 0.646 | 0.917 | 1.000 |
| rank_reduction | fixed | 48 | 0.35 | 4 | 0.434 | 0.432 | 0.432 | 0.583 | 1.000 |
| rank_reduction | fixed | 48 | 0.50 | 4 | 0.519 | 0.500 | 0.577 | 0.833 | 1.000 |
| rank_reduction | fixed | 48 | 0.65 | 4 | 0.566 | 0.545 | 0.646 | 0.917 | 1.000 |
| rank_reduction | fixed | 72 | 0.35 | 4 | 0.434 | 0.432 | 0.432 | 0.583 | 1.000 |
| rank_reduction | fixed | 72 | 0.50 | 4 | 0.519 | 0.500 | 0.577 | 0.833 | 1.000 |
| rank_reduction | fixed | 72 | 0.65 | 4 | 0.566 | 0.545 | 0.646 | 0.917 | 1.000 |
| rank_reduction | random | 24 | 0.35 | 4 | 0.434 | 0.432 | 0.432 | 0.583 | 1.000 |
| rank_reduction | random | 24 | 0.50 | 4 | 0.519 | 0.500 | 0.577 | 0.833 | 1.000 |
| rank_reduction | random | 24 | 0.65 | 4 | 0.566 | 0.545 | 0.646 | 0.917 | 1.000 |
| rank_reduction | random | 48 | 0.35 | 4 | 0.434 | 0.432 | 0.432 | 0.583 | 1.000 |
| rank_reduction | random | 48 | 0.50 | 4 | 0.519 | 0.500 | 0.577 | 0.833 | 1.000 |
| rank_reduction | random | 48 | 0.65 | 4 | 0.566 | 0.545 | 0.646 | 0.917 | 1.000 |
| rank_reduction | random | 72 | 0.35 | 4 | 0.434 | 0.432 | 0.432 | 0.583 | 1.000 |
| rank_reduction | random | 72 | 0.50 | 4 | 0.519 | 0.500 | 0.577 | 0.833 | 1.000 |
| rank_reduction | random | 72 | 0.65 | 4 | 0.566 | 0.545 | 0.646 | 0.917 | 1.000 |

## Policy Agreement Change (Max vs Min Sample)
| mode | panel_type | budget | sample_range | delta_alloc_spearman_vs_oht | delta_alloc_kendall_vs_oht | delta_alloc_gamma_vs_oht | delta_alloc_topk_vs_oht |
| --- | --- | --- | --- | --- | --- | --- | --- |
| rank_reduction | fixed | 0.35 | 24->72 | 0.000 | 0.000 | 0.000 | 0.000 |
| rank_reduction | fixed | 0.50 | 24->72 | 0.000 | 0.000 | 0.000 | 0.000 |
| rank_reduction | fixed | 0.65 | 24->72 | 0.000 | 0.000 | 0.000 | 0.000 |
| rank_reduction | random | 0.35 | 24->72 | 0.000 | 0.000 | 0.000 | 0.000 |
| rank_reduction | random | 0.50 | 24->72 | 0.000 | 0.000 | 0.000 | 0.000 |
| rank_reduction | random | 0.65 | 24->72 | 0.000 | 0.000 | 0.000 | 0.000 |

## Caution
- This sweep is informative-subset-only and CPU-bounded; treat as directional evidence.
