# CPU Pipeline Report

_Generated: 2026-04-26T23:46:02.854649+00:00_

Per SPEC_CPU_v0_2 §8.13. Top-level navigation document for the benchmark-reliability-study CPU pipeline outputs.

## 1. Pipeline execution summary

**Study identity**

- Study ID: `benchmark_reliability_v1`
- Prereg version: `v1_1_LOCKED`
- Config hash (short): `89ce3f1f`
- Config hash (full): `89ce3f1fc05058b8df7a8a30455dca31544b5e1ac0ef7bf457de0eaf8b89f0e5`

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
| arc_challenge | 6 | 0.2 | 0.1340006280693672 | 0.1947401256466223 | 0.040824829046386304 | True | True | interval_required | interval_required |
| hellaswag | 4 | 0.2 | 0.19532289109528483 | 0.23271758428476774 | 0.043982640562744736 | True | True | interval_required | interval_required |
| mmlu_panel | 2 | 0.2 | 0.17345265860447873 | 0.22400219375005997 | 0.040824829046386304 | True | True | interval_required | interval_required |
| truthfulqa_mc | 2 | 0.2 | 0.2351934550900784 | 1.0393503489948035 | 0.07071067811865475 | True | True | interval_required | interval_required |
| winogrande | 2 | 0.2 | 0.43043607078434043 | 0.48664534276498933 | 0.040824829046386304 | True | True | interval_required | interval_required |

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

**Figures**

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

**Reports**

- not yet available

## 4. Deviations

No deviations recorded.

## 5. Reproducibility trace status

Reproducibility trace not yet generated.
