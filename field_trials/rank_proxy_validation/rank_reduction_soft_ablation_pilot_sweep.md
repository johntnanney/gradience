# Ablation Reliability Sweep (Informative Subset)

## Scope
- informative families: sst2, imdb
- adapters analyzed: 4
- ablation modes: hard_zero, attenuate, rank_reduction
- ablation sample grid: [24, 48, 72]
- fixed panels per setting: 3
- random repeats per setting: 3
- budgets: [0.35, 0.5, 0.65]
- low-info flags: max_unique<=2, min_nonzero_fraction<=0.2, high_tie_pair_fraction>=0.8

## Stability Summary
| mode | panel_type | ablation_samples | n_adapters | mean_spearman | mean_kendall_tau_b | mean_gamma | mean_topk_q25 | mean_topk_q50 | spearman_valid_pair_fraction | flat_vector_fraction | low_info_vector_fraction | high_tie_vector_fraction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| attenuate | fixed | 24 | 4 | 0.250 | 0.250 | 0.500 | 0.722 | 0.861 | 0.083 | 0.750 | 1.000 | 0.750 |
| attenuate | fixed | 48 | 4 | 0.345 | 0.342 | 0.417 | 0.528 | 0.806 | 0.500 | 0.417 | 0.917 | 0.417 |
| attenuate | fixed | 72 | 4 | 0.878 | 0.858 | 0.963 | 0.667 | 0.917 | 0.583 | 0.333 | 0.583 | 0.333 |
| attenuate | random | 24 | 4 | 0.620 | 0.590 | 0.826 | 0.611 | 0.847 | 0.333 | 0.583 | 0.750 | 0.583 |
| attenuate | random | 48 | 4 | 0.287 | 0.277 | 0.502 | 0.667 | 0.778 | 0.333 | 0.583 | 0.833 | 0.583 |
| attenuate | random | 72 | 4 | 0.888 | 0.855 | 0.947 | 0.667 | 0.944 | 0.583 | 0.333 | 0.500 | 0.333 |
| hard_zero | fixed | 24 | 4 | 0.299 | 0.306 | 0.444 | 0.639 | 0.722 | 0.583 | 0.333 | 0.917 | 0.417 |
| hard_zero | fixed | 48 | 4 | 0.618 | 0.582 | 0.888 | 0.556 | 0.708 | 0.417 | 0.417 | 0.667 | 0.417 |
| hard_zero | fixed | 72 | 4 | 0.571 | 0.548 | 0.687 | 0.778 | 0.750 | 0.750 | 0.250 | 0.583 | 0.250 |
| hard_zero | random | 24 | 4 | 0.231 | 0.221 | 0.343 | 0.500 | 0.736 | 0.333 | 0.500 | 0.667 | 0.583 |
| hard_zero | random | 48 | 4 | 0.426 | 0.417 | 0.754 | 0.500 | 0.722 | 0.583 | 0.333 | 0.833 | 0.417 |
| hard_zero | random | 72 | 4 | 0.677 | 0.658 | 0.926 | 0.722 | 0.806 | 0.583 | 0.333 | 0.583 | 0.333 |
| rank_reduction | fixed | 24 | 4 | n/a | n/a | n/a | 1.000 | 1.000 | 0.000 | 0.917 | 1.000 | 0.917 |
| rank_reduction | fixed | 48 | 4 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.083 | 0.833 | 1.000 | 0.833 |
| rank_reduction | fixed | 72 | 4 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.083 | 0.833 | 1.000 | 0.833 |
| rank_reduction | random | 24 | 4 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.250 | 0.750 | 1.000 | 0.750 |
| rank_reduction | random | 48 | 4 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.083 | 0.833 | 1.000 | 0.833 |
| rank_reduction | random | 72 | 4 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.250 | 0.750 | 1.000 | 0.750 |

## Stability Improvement (Max vs Min Sample)
| mode | panel_type | sample_range | delta_spearman | delta_kendall_tau_b | delta_gamma | delta_topk_q25 | delta_topk_q50 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| attenuate | fixed | 24->72 | 0.628 | 0.608 | 0.463 | -0.056 | 0.056 |
| attenuate | random | 24->72 | 0.267 | 0.265 | 0.121 | 0.056 | 0.097 |
| hard_zero | fixed | 24->72 | 0.272 | 0.242 | 0.243 | 0.139 | 0.028 |
| hard_zero | random | 24->72 | 0.447 | 0.438 | 0.582 | 0.222 | 0.069 |
| rank_reduction | fixed | 24->72 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 |
| rank_reduction | random | 24->72 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

## Policy Agreement vs OHT Summary
| mode | panel_type | ablation_samples | budget | n_adapters | mean_alloc_spearman_vs_oht | mean_alloc_kendall_vs_oht | mean_alloc_gamma_vs_oht | mean_alloc_topk_overlap_vs_oht | spearman_valid_panel_fraction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| attenuate | fixed | 24 | 0.35 | 4 | 0.369 | 0.365 | 0.338 | 0.528 | 1.000 |
| attenuate | fixed | 24 | 0.50 | 4 | 0.338 | 0.319 | 0.380 | 0.750 | 1.000 |
| attenuate | fixed | 24 | 0.65 | 4 | 0.399 | 0.378 | 0.479 | 0.917 | 1.000 |
| attenuate | fixed | 48 | 0.35 | 4 | 0.067 | 0.075 | -0.106 | 0.306 | 1.000 |
| attenuate | fixed | 48 | 0.50 | 4 | 0.285 | 0.268 | 0.382 | 0.694 | 1.000 |
| attenuate | fixed | 48 | 0.65 | 4 | 0.383 | 0.357 | 0.481 | 0.889 | 1.000 |
| attenuate | fixed | 72 | 0.35 | 4 | -0.055 | -0.047 | -0.323 | 0.167 | 1.000 |
| attenuate | fixed | 72 | 0.50 | 4 | 0.043 | 0.037 | -0.021 | 0.556 | 1.000 |
| attenuate | fixed | 72 | 0.65 | 4 | 0.228 | 0.203 | 0.222 | 0.917 | 1.000 |
| attenuate | random | 24 | 0.35 | 4 | 0.116 | 0.120 | -0.058 | 0.306 | 1.000 |
| attenuate | random | 24 | 0.50 | 4 | 0.320 | 0.302 | 0.430 | 0.778 | 1.000 |
| attenuate | random | 24 | 0.65 | 4 | 0.288 | 0.267 | 0.346 | 0.889 | 1.000 |
| attenuate | random | 48 | 0.35 | 4 | 0.173 | 0.179 | 0.058 | 0.361 | 1.000 |
| attenuate | random | 48 | 0.50 | 4 | 0.398 | 0.385 | 0.468 | 0.778 | 1.000 |
| attenuate | random | 48 | 0.65 | 4 | 0.430 | 0.408 | 0.471 | 0.889 | 1.000 |
| attenuate | random | 72 | 0.35 | 4 | -0.104 | -0.092 | -0.418 | 0.167 | 1.000 |
| attenuate | random | 72 | 0.50 | 4 | 0.122 | 0.114 | 0.129 | 0.639 | 1.000 |
| attenuate | random | 72 | 0.65 | 4 | 0.156 | 0.139 | 0.126 | 0.917 | 1.000 |
| hard_zero | fixed | 24 | 0.35 | 4 | 0.315 | 0.313 | 0.321 | 0.472 | 1.000 |
| hard_zero | fixed | 24 | 0.50 | 4 | 0.364 | 0.351 | 0.471 | 0.750 | 1.000 |
| hard_zero | fixed | 24 | 0.65 | 4 | 0.331 | 0.310 | 0.345 | 0.833 | 1.000 |
| hard_zero | fixed | 48 | 0.35 | 4 | 0.102 | 0.108 | -0.033 | 0.333 | 1.000 |
| hard_zero | fixed | 48 | 0.50 | 4 | 0.257 | 0.245 | 0.279 | 0.583 | 1.000 |
| hard_zero | fixed | 48 | 0.65 | 4 | 0.367 | 0.348 | 0.430 | 0.806 | 1.000 |
| hard_zero | fixed | 72 | 0.35 | 4 | 0.027 | 0.029 | -0.160 | 0.250 | 1.000 |
| hard_zero | fixed | 72 | 0.50 | 4 | -0.001 | -0.007 | -0.117 | 0.500 | 1.000 |
| hard_zero | fixed | 72 | 0.65 | 4 | 0.153 | 0.139 | 0.123 | 0.667 | 1.000 |
| hard_zero | random | 24 | 0.35 | 4 | -0.033 | -0.025 | -0.320 | 0.222 | 1.000 |
| hard_zero | random | 24 | 0.50 | 4 | 0.197 | 0.183 | 0.224 | 0.639 | 1.000 |
| hard_zero | random | 24 | 0.65 | 4 | 0.264 | 0.244 | 0.295 | 0.778 | 1.000 |
| hard_zero | random | 48 | 0.35 | 4 | -0.041 | -0.033 | -0.298 | 0.222 | 1.000 |
| hard_zero | random | 48 | 0.50 | 4 | 0.087 | 0.078 | 0.046 | 0.556 | 1.000 |
| hard_zero | random | 48 | 0.65 | 4 | 0.311 | 0.295 | 0.340 | 0.778 | 1.000 |
| hard_zero | random | 72 | 0.35 | 4 | -0.076 | -0.068 | -0.359 | 0.194 | 1.000 |
| hard_zero | random | 72 | 0.50 | 4 | -0.105 | -0.105 | -0.253 | 0.389 | 1.000 |
| hard_zero | random | 72 | 0.65 | 4 | 0.156 | 0.145 | 0.137 | 0.778 | 1.000 |
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
| attenuate | fixed | 0.35 | 24->72 | -0.424 | -0.412 | -0.661 | -0.361 |
| attenuate | fixed | 0.50 | 24->72 | -0.296 | -0.282 | -0.401 | -0.194 |
| attenuate | fixed | 0.65 | 24->72 | -0.172 | -0.175 | -0.257 | 0.000 |
| attenuate | random | 0.35 | 24->72 | -0.220 | -0.212 | -0.360 | -0.139 |
| attenuate | random | 0.50 | 24->72 | -0.199 | -0.189 | -0.301 | -0.139 |
| attenuate | random | 0.65 | 24->72 | -0.132 | -0.128 | -0.219 | 0.028 |
| hard_zero | fixed | 0.35 | 24->72 | -0.288 | -0.283 | -0.481 | -0.222 |
| hard_zero | fixed | 0.50 | 24->72 | -0.365 | -0.358 | -0.587 | -0.250 |
| hard_zero | fixed | 0.65 | 24->72 | -0.179 | -0.171 | -0.222 | -0.167 |
| hard_zero | random | 0.35 | 24->72 | -0.043 | -0.043 | -0.039 | -0.028 |
| hard_zero | random | 0.50 | 24->72 | -0.301 | -0.288 | -0.477 | -0.250 |
| hard_zero | random | 0.65 | 24->72 | -0.107 | -0.098 | -0.157 | 0.000 |
| rank_reduction | fixed | 0.35 | 24->72 | 0.000 | 0.000 | 0.000 | 0.000 |
| rank_reduction | fixed | 0.50 | 24->72 | 0.000 | 0.000 | 0.000 | 0.000 |
| rank_reduction | fixed | 0.65 | 24->72 | 0.000 | 0.000 | 0.000 | 0.000 |
| rank_reduction | random | 0.35 | 24->72 | 0.000 | 0.000 | 0.000 | 0.000 |
| rank_reduction | random | 0.50 | 24->72 | 0.000 | 0.000 | 0.000 | 0.000 |
| rank_reduction | random | 0.65 | 24->72 | 0.000 | 0.000 | 0.000 | 0.000 |

## Caution
- This sweep is informative-subset-only and CPU-bounded; treat as directional evidence.
