# Reproducibility Trace

_Generated: 2026-04-26T23:45:38.411429+00:00_

Per SPEC_CPU_v0_2 §13. Verifies that the pipeline's outputs
can be re-derived from the committed inputs.

## 1. Config state

- `config_hash_full`: `89ce3f1fc05058b8df7a8a30455dca31544b5e1ac0ef7bf457de0eaf8b89f0e5`
- `config_hash` (short): `89ce3f1f`

**Per-file SHA-256:**

| File | SHA-256 |
|---|---|
| `study_config.yaml` | `c3dfd7409f8ba057f7c205876e856b145d30c79c7dbec53c415869337f21ee4c` |
| `models.yaml` | `523f0715c6baf7f5a997e3d1c34ba6e4d5c8eac0e263e93f278a1123ba58c899` |
| `benchmarks.yaml` | `d9e21203c509b2d11b69451091ec29f8bb4f3439122339e812287e096b4482b2` |
| `prompts.yaml` | `0fbc105bde3590c28a37a5989e77e3496c266a0f9af6cd3c9af49698c0ad4251` |
| `scoring_rules.yaml` | `38d82ecbc4c8da811ae95b5ee641b31c565e1ebcb9b459daf1482082b2224a88` |
| `analysis_config.yaml` | `3d71714c753fa4c04f52c6725b8d48ddbeb3ddbc7798494c427f56d26222ba99` |

## 2. Manifest state

| Manifest | SHA-256 | Row count |
|---|---|---|
| `conditions_primary.csv` | `b553d9c8fba3fefeeffc8f83c79e26be55035b81e5eee28654d9550657c9a131` | 248 |
| `conditions_gsm8k.csv` | `80b1a1a1e442006527bd1b7d9b3ac24ea3cc17bcd5ef9e972180f55d2b9cf123` | 24 |
| `prompt_manifest.csv` | missing | — |
| `fewshot_manifest.csv` | missing | — |

## 3. Raw-run coverage

- Conditions marked `complete` in manifest: 248
- Raw-run subdirectories on disk: 275
- All complete conditions have raw directories.
- **Raw directories without a matching manifest row:**
    - `.tmp`
    - `arc_challenge____qwen2_5_1_5b_instruct__P2_lm_eval__s123__generate_parse`
    - `arc_challenge____qwen2_5_1_5b_instruct__P2_lm_eval__s123__ll_norm`
    - `gsm8k____pythia_1_4b__P1_original__s123__generate_parse_permissive`
    - `gsm8k____pythia_1_4b__P1_original__s123__generate_parse_strict`
    - `gsm8k____pythia_1_4b__P1_original__s2024__generate_parse_permissive`
    - `gsm8k____pythia_1_4b__P1_original__s2024__generate_parse_strict`
    - `gsm8k____pythia_1_4b__P1_original__s42__generate_parse_permissive`
    - `gsm8k____pythia_1_4b__P1_original__s42__generate_parse_strict`
    - `gsm8k____pythia_1_4b__P2_lm_eval__s123__generate_parse_permissive`
    - `gsm8k____pythia_1_4b__P2_lm_eval__s123__generate_parse_strict`
    - `gsm8k____pythia_1_4b__P2_lm_eval__s2024__generate_parse_permissive`
    - `gsm8k____pythia_1_4b__P2_lm_eval__s2024__generate_parse_strict`
    - `gsm8k____pythia_1_4b__P2_lm_eval__s42__generate_parse_permissive`
    - `gsm8k____pythia_1_4b__P2_lm_eval__s42__generate_parse_strict`
    - `gsm8k____pythia_1_4b__P3_helm_or_published__s123__generate_parse_permissive`
    - `gsm8k____pythia_1_4b__P3_helm_or_published__s123__generate_parse_strict`
    - `gsm8k____pythia_1_4b__P3_helm_or_published__s2024__generate_parse_permissive`
    - `gsm8k____pythia_1_4b__P3_helm_or_published__s2024__generate_parse_strict`
    - `gsm8k____pythia_1_4b__P3_helm_or_published__s42__generate_parse_permissive`
    - `gsm8k____pythia_1_4b__P3_helm_or_published__s42__generate_parse_strict`
    - `gsm8k____pythia_1_4b__P4_minimal_sourced__s123__generate_parse_permissive`
    - `gsm8k____pythia_1_4b__P4_minimal_sourced__s123__generate_parse_strict`
    - `gsm8k____pythia_1_4b__P4_minimal_sourced__s2024__generate_parse_permissive`
    - `gsm8k____pythia_1_4b__P4_minimal_sourced__s2024__generate_parse_strict`
    - `gsm8k____pythia_1_4b__P4_minimal_sourced__s42__generate_parse_permissive`
    - `gsm8k____pythia_1_4b__P4_minimal_sourced__s42__generate_parse_strict`

## 4. Per-condition recompute sample

- Sample size requested: N = 5; seed = 20260424

| condition_id | scoring_rule | n_items | recomputed | stored | delta | status |
|---|---|---|---|---|---|---|
| `hellaswag____pythia_1_4b__P4_minimal_sourced__s123__ll_norm` | `ll_norm` | 10042 | 0.247660 | 0.247660 | 0.00e+00 | pass |
| `mmlu_panel__professional_accounting__pythia_1_4b__P3_helm_or_published__s123__generate_parse` | `generate_parse` | 282 | 0.124113 | — | — | skipped |
| `mmlu_panel__professional_accounting__pythia_1_4b__P3_helm_or_published__s2024__ll_norm` | `ll_norm` | 282 | 0.223404 | — | — | skipped |
| `winogrande____pythia_1_4b__P1_original__s123__generate_parse` | `generate_parse` | 1267 | 0.056038 | 0.056038 | 0.00e+00 | pass |
| `winogrande____pythia_1_4b__P2_lm_eval__s123__ll_norm` | `ll_norm` | 1267 | 0.515391 | 0.515391 | 0.00e+00 | pass |

**Skipped reasons:**
- `mmlu_panel__professional_accounting__pythia_1_4b__P3_helm_or_published__s123__generate_parse`: no stored accuracy in normalized condition_level CSV
- `mmlu_panel__professional_accounting__pythia_1_4b__P3_helm_or_published__s2024__ll_norm`: no stored accuracy in normalized condition_level CSV

## 5. Analysis re-derivation

- **Variance components** (`analysis/variance_components/aggregate_vc.csv`)
    - status: fail
    - 06_variance_components.py invocation error: Command '['/usr/bin/python3', '/workspace/study/scripts/06_variance_components.py', '--item-level', 'normalized/item_level_primary.parquet', '--condition-level', 'normalized/condition_level_primary.csv', '--config', 'configs/study_config.yaml', '--out-dir', '/tmp/tmp27g8vtx8/variance_components']' timed out after 120 seconds
- **Tolerance schedule** (`analysis/tolerance_schedules/tolerance_by_cell.csv`)
    - status: fail
    - tolerance_by_cell.csv differs from stored output

## 6. Cross-environment check

- not performed (no `--compare-env` provided).

## 7. Summary

- Total reproducibility-critical artifacts checked: 16
- Total reproducibility failures: 3
- **Trace status: `fail`**
