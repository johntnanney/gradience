# CPU Pipeline Report

_Generated: 2026-04-28T12:17:16.067088+00:00_

Per SPEC_CPU_v0_2 §8.13. Top-level navigation document for the benchmark-reliability-study CPU pipeline outputs.

## 1. Pipeline execution summary

**Study identity**

- Study ID: `n/a`
- Prereg version: `n/a`
- Config hash (short): `n/a`

**Manifests**

- not yet available (no manifests directory provided or empty)

**Inference outputs**

- raw-run directory: not yet available
- normalized files: not yet available

## 2. Headline numbers

**H1 (single-occasion tolerance dominates the precision frontier)**

- Decision: **CONFIRMED**
- Benchmarks exceeding threshold: 5 (required: 3)
- Exceeding: `arc_challenge`, `hellaswag`, `mmlu_panel`, `truthfulqa_mc`, `winogrande`
- Threshold: 0.005

**Per-benchmark cross-model median tolerance**

| benchmark_id | n_cells | tolerance_single_median | tolerance_single_median_ci_lower | tolerance_single_median_ci_upper | tolerance_full_design_median | exceeds_h1_threshold_point | exceeds_h1_threshold_ci_lower | licensed_precision_single | licensed_precision_full_design |
|---|---|---|---|---|---|---|---|---|---|
| arc_challenge | 6 | 0.21713665325931777 | 0.13396939588169265 | 0.19471301717729772 | 0.04432283374508054 | True | True | interval_required | interval_required |
| hellaswag | 6 | 0.07843237024470438 | 0.18370081140717076 | 0.21727932518274853 | 0.016009940534714658 | True | True | interval_required | interval_required |
| mmlu_panel | 6 | 0.21242743557255386 | 0.1605163940805585 | 0.1869184462893038 | 0.01939188304934385 | True | True | interval_required | interval_required |
| truthfulqa_mc | 6 | 0.5957881115152748 | 0.4213273934139635 | 0.9895399288073372 | 0.2106429069013889 | True | True | interval_required | interval_required |
| winogrande | 6 | 0.12582518531951045 | 0.38355879925880854 | 0.45451311389780846 | 0.02568395840199445 | True | True | interval_required | interval_required |

**Ranking stability**

- `kendall_tau_by_benchmark.csv`
- `pairwise_win_probabilities.csv`
- `ranking_reversals.csv`

**H4 (MMLU model x subject interaction)**

- Decision: **NOT CONFIRMED**
- Threshold: 0.1

## 3. Artifact links

**Analysis**

- [`analysis/gsm8k_case/gsm8k_extraction_sensitivity.csv`](analysis/gsm8k_case/gsm8k_extraction_sensitivity.csv)
- [`analysis/gsm8k_case/gsm8k_parseability.csv`](analysis/gsm8k_case/gsm8k_parseability.csv)
- [`analysis/gsm8k_case/gsm8k_tolerance_schedule.csv`](analysis/gsm8k_case/gsm8k_tolerance_schedule.csv)
- [`analysis/mmlu_subjects/h4_test.json`](analysis/mmlu_subjects/h4_test.json)
- [`analysis/mmlu_subjects/mmlu_subject_accuracy_matrix.csv`](analysis/mmlu_subjects/mmlu_subject_accuracy_matrix.csv)
- [`analysis/mmlu_subjects/mmlu_subject_variance_components.csv`](analysis/mmlu_subjects/mmlu_subject_variance_components.csv)
- [`analysis/ranking_stability/kendall_tau_by_benchmark.csv`](analysis/ranking_stability/kendall_tau_by_benchmark.csv)
- [`analysis/ranking_stability/pairwise_win_probabilities.csv`](analysis/ranking_stability/pairwise_win_probabilities.csv)
- [`analysis/ranking_stability/ranking_reversals.csv`](analysis/ranking_stability/ranking_reversals.csv)
- [`analysis/tolerance_schedules/h1_test.json`](analysis/tolerance_schedules/h1_test.json)
- [`analysis/tolerance_schedules/tolerance_by_benchmark_summary.csv`](analysis/tolerance_schedules/tolerance_by_benchmark_summary.csv)
- [`analysis/tolerance_schedules/tolerance_by_cell.csv`](analysis/tolerance_schedules/tolerance_by_cell.csv)
- [`analysis/variance_components/aggregate_vc.csv`](analysis/variance_components/aggregate_vc.csv)
- [`analysis/variance_components/item_level_vc.csv`](analysis/variance_components/item_level_vc.csv)
- [`analysis/variance_components/model_convergence_report.csv`](analysis/variance_components/model_convergence_report.csv)

**Tables**

- not yet available

**Figures**

- [`figures/mmlu_model_subject_heatmap.png`](figures/mmlu_model_subject_heatmap.png)
- [`figures/ranking_stability_by_benchmark.png`](figures/ranking_stability_by_benchmark.png)

**Reports**

- not yet available

## 4. Deviations

No deviations recorded.

## 5. Reproducibility trace status

Reproducibility trace not yet generated.
