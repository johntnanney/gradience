# OA Refinement Analysis

- Outcome metric: `merge_delta_vs_best_source`
- Pair count: 12
- Features tested: total=48, oa=44, control=4

## Baseline Comparison
| feature | spearman | median_split_gap | loo_sign_consistency |
| --- | --- | --- | --- |
| current_max_over_accumulation_score | -0.0176 | 0.1060 | 0.583 |
| current_watch_layer_count | -0.2195 | n/a | 1.000 |
- Best baseline |Spearman|: 0.2195

## Top OA Candidates
| feature | family | spearman | median_split_gap | loo_median_abs | loo_sign_consistency | rank_score |
| --- | --- | --- | --- | --- | --- | --- |
| concentration_oa_mass_top3_fraction | concentration_of_risk | 0.5835 | 0.1120 | 0.5897 | 1.000 | 0.4960 |
| concentration_oa_mass_top5_fraction | concentration_of_risk | 0.5694 | 0.1120 | 0.5737 | 1.000 | 0.4840 |
| raw_max_coefficient_exposure | raw_subfactor | 0.3951 | 0.0858 | 0.4018 | 1.000 | 0.3951 |
| raw_max_alignment | raw_subfactor | -0.3902 | -0.0807 | 0.3593 | 1.000 | 0.3902 |
| interaction_max_align_x_exposure | interaction | -0.3902 | -0.0807 | 0.3593 | 1.000 | 0.3511 |
| concentration_oa_mass_top1_fraction | concentration_of_risk | 0.3796 | 0.0800 | 0.4132 | 1.000 | 0.3227 |
| concentration_ace_mass_top1_fraction | concentration_of_risk | 0.3733 | 0.0787 | 0.3812 | 1.000 | 0.3173 |
| concentration_ace_mass_top3_fraction | concentration_of_risk | 0.3733 | 0.0787 | 0.3812 | 1.000 | 0.3173 |
| concentration_ace_mass_top5_fraction | concentration_of_risk | 0.3733 | 0.0787 | 0.3812 | 1.000 | 0.3173 |
| raw_mean_coefficient_exposure | raw_subfactor | 0.3161 | 0.0104 | 0.3085 | 1.000 | 0.3161 |
| burden_frac_layers_align_ge_0_45_and_exposure_ge_0_90 | burden | -0.3201 | -0.0647 | 0.3300 | 1.000 | 0.2881 |
| interaction_top3_align_x_exposure_mean | interaction | -0.3058 | -0.0807 | 0.2994 | 1.000 | 0.2752 |

## Features Exceeding Current Baseline
| feature | family | spearman | |spearman| |
| --- | --- | --- | --- |
| concentration_oa_mass_top3_fraction | concentration_of_risk | 0.5835 | 0.5835 |
| concentration_oa_mass_top5_fraction | concentration_of_risk | 0.5694 | 0.5694 |
| raw_max_coefficient_exposure | raw_subfactor | 0.3951 | 0.3951 |
| raw_max_alignment | raw_subfactor | -0.3902 | 0.3902 |
| interaction_max_align_x_exposure | interaction | -0.3902 | 0.3902 |
| concentration_oa_mass_top1_fraction | concentration_of_risk | 0.3796 | 0.3796 |
| concentration_ace_mass_top1_fraction | concentration_of_risk | 0.3733 | 0.3733 |
| concentration_ace_mass_top3_fraction | concentration_of_risk | 0.3733 | 0.3733 |
| concentration_ace_mass_top5_fraction | concentration_of_risk | 0.3733 | 0.3733 |
| raw_mean_coefficient_exposure | raw_subfactor | 0.3161 | 0.3161 |
| burden_frac_layers_align_ge_0_45_and_exposure_ge_0_90 | burden | -0.3201 | 0.3201 |
| interaction_top3_align_x_exposure_mean | interaction | -0.3058 | 0.3058 |

## Restricted High-Overlap/Low-Conflict Check
- Criteria: mean_overlap >= 0.20 and conflict_fraction <= 0.10
- n=12
| feature | n | spearman |
| --- | --- | --- |
| concentration_oa_mass_top3_fraction | 12 | 0.5835 |
| concentration_oa_mass_top5_fraction | 12 | 0.5694 |
| raw_max_coefficient_exposure | 12 | 0.3951 |
| raw_max_alignment | 12 | -0.3902 |
| interaction_max_align_x_exposure | 12 | -0.3902 |
| concentration_oa_mass_top1_fraction | 12 | 0.3796 |
| concentration_ace_mass_top1_fraction | 12 | 0.3733 |
| concentration_ace_mass_top3_fraction | 12 | 0.3733 |
| concentration_ace_mass_top5_fraction | 12 | 0.3733 |
| raw_mean_coefficient_exposure | 12 | 0.3161 |

## Control Feature Context
| feature | spearman | |spearman| |
| --- | --- | --- |
| control_source_score_gap | -0.8468 | 0.8468 |
| control_source_score_var | -0.8468 | 0.8468 |
| control_layer_count | -0.6559 | 0.6559 |
| control_source_score_mean | 0.3656 | 0.3656 |
