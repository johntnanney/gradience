# Reproducibility Trace

_Generated: 2026-04-28T12:17:16.009647+00:00_

Per SPEC_CPU_v0_2 §13. Verifies that the pipeline's outputs
can be re-derived from the committed inputs.

## 1. Config state

- `config_hash_full`: `fbc4a5ddbf037c5fce38950b84e8735de5a058fc0ace8a3899df90bbb6c240c8`
- `config_hash` (short): `fbc4a5dd`

**Per-file SHA-256:**

| File | SHA-256 |
|---|---|
| `study_config.yaml` | `c3dfd7409f8ba057f7c205876e856b145d30c79c7dbec53c415869337f21ee4c` |
| `models.yaml` | `523f0715c6baf7f5a997e3d1c34ba6e4d5c8eac0e263e93f278a1123ba58c899` |
| `benchmarks.yaml` | `d9e21203c509b2d11b69451091ec29f8bb4f3439122339e812287e096b4482b2` |
| `prompts.yaml` | `0fbc105bde3590c28a37a5989e77e3496c266a0f9af6cd3c9af49698c0ad4251` |
| `scoring_rules.yaml` | `38d82ecbc4c8da811ae95b5ee641b31c565e1ebcb9b459daf1482082b2224a88` |
| `analysis_config.yaml` | `10c92d58366713316e9955429935ce284c040a18255e014f3ed65fb9e58fde63` |

## 2. Manifest state

| Manifest | SHA-256 | Row count |
|---|---|---|
| `conditions_primary.csv` | `4d171cb1cfe9e3a207842a81486cccb85c994975817df5840cbeb6a738ee1d49` | 600 |
| `conditions_gsm8k.csv` | `216421c4b622de1984d358e52c448840f48990fde98ff6e471aca9a83e06187b` | 72 |
| `prompt_manifest.csv` | `57dcb2512b3e001696d60b3109b16350c0503dfb934ab87a0f5c7670e4421315` | 24 |
| `fewshot_manifest.csv` | `e7b7bac3e33a3fe01a1f39cfef0bbc07ab4e6a991ed9cd0f10b3979d8eaf8eb2` | 136 |

## 3. Raw-run coverage

- Conditions marked `complete` in manifest: 600
- Raw-run subdirectories on disk: 624
- All complete conditions have raw directories.
- **Raw directories without a matching manifest row:**
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
| `mmlu_panel__elementary_mathematics__pythia_410m__P2_lm_eval__s42__generate_parse` | `generate_parse` | 378 | 0.044974 | 0.044974 | 0.00e+00 | pass |
| `mmlu_panel__world_religions__pythia_1_4b__P2_lm_eval__s2024__generate_parse` | `generate_parse` | 171 | 0.093567 | 0.093567 | 0.00e+00 | pass |
| `mmlu_panel__world_religions__pythia_1_4b__P3_helm_or_published__s123__ll_norm` | `ll_norm` | 171 | 0.321637 | 0.321637 | 0.00e+00 | pass |
| `winogrande____pythia_1_4b__P3_helm_or_published__s42__ll_norm` | `ll_norm` | 1267 | 0.501184 | 0.501184 | 0.00e+00 | pass |
| `winogrande____pythia_410m__P2_lm_eval__s42__ll_norm` | `ll_norm` | 1267 | 0.495659 | 0.495659 | 0.00e+00 | pass |

## 5. Analysis re-derivation

- **Variance components** (`analysis/variance_components/aggregate_vc.csv`)
    - status: pass
    - aggregate_vc.csv re-derived identically
- **Tolerance schedule** (`analysis/tolerance_schedules/tolerance_by_cell.csv`)
    - status: fail
    - tolerance_by_cell.csv differs from stored output

## 6. Cross-environment check

- not performed (no `--compare-env` provided).

## 7. Summary

- Total reproducibility-critical artifacts checked: 18
- Total reproducibility failures: 2
- **Trace status: `fail`**
