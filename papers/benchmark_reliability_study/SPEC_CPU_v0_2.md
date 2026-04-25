# CPU-Side Work Spec v0.2 — Coding-Agent Handoff

**Project:** Benchmark Accuracy as Measurement: Reliability and Tolerance Schedules for LLM Evaluation
**Repository path:** `papers/benchmark_reliability_study/`
**Spec version:** v0.2 (coding-agent-facing)
**Supersedes:** `SPEC_CPU.md` v0.1
**Date:** 2026-04-24
**Status:** ready for implementation after v1.1 pre-registration lock and after open-question resolution (§3.10 of `preregistration_v1.md`)

## Changes from v0.1

- **§9.1** Decimal-place licensing categorical restructured: single notation (decimal-accuracy), three-level categorical, unambiguous thresholds. Retires the `integer_percent_only` level, which duplicated `two_decimal_accuracy`.
- **§10.1** Mixed-effects convergence cascade specified as pre-registered four-level fallback with explicit triggers; no post-hoc simplification decisions at implementation time.
- **§4.4** Config-hash formalism defined: canonical YAML serialization, SHA-256, reference implementation pseudocode.
- **§9.3** Bootstrap confidence intervals added to SEM and tolerance estimates; H1 decision rule revised to use bootstrap lower bound.
- **§3.7** Few-shot leakage validation promoted to hard acceptance criterion with explicit set-intersection check.
- **§13** Reproducibility trace structure specified.
- **§5.1** GPU-side input data contract added as symmetric counterpart to the output contract.
- **§12** Error-code taxonomy added.
- **§16** Task inventory for coding agents added, ordered by dependency.
- **§17** Pinned library versions added.

All other sections retain v0.1 substance with minor tightenings for precision.

---

## 1. Scope

This spec defines the CPU-side infrastructure for the benchmark reliability study. The CPU pipeline is responsible for:

1. Defining the measurement universe (configs).
2. Enumerating all benchmark × model × prompt × seed × scoring-rule conditions (manifests).
3. Preparing prompt and exemplar manifests (pre-inference artifacts).
4. Validating prompt admissibility metadata (pre-lock gate).
5. Ingesting model outputs or log-likelihood scores (post-inference artifacts).
6. Normalizing item-level correctness data (data-hygiene layer).
7. Computing aggregate scores (condition-level summary layer).
8. Estimating variance components where possible (statistical layer).
9. Generating tolerance schedules (prescriptive layer).
10. Generating ranking-stability analyses (secondary analysis layer).
11. Producing reproducibility artifacts (audit layer).

**Out of scope for this spec:** large-scale model inference. Inference is executed through a separate GPU-side (or small-model CPU inference) backend, conformant to the input contract (§5.1) and producing outputs conformant to the output contract (§5.2).

**Agent-facing implementation principle:** every script in this spec has a deterministic input → output contract. No implementation is permitted to rely on filesystem state not produced by a prior script in the pipeline. No script may write to a location outside its declared output directory.

---

## 2. Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                         CPU-SIDE PIPELINE                        │
│                                                                  │
│   configs/*.yaml                                                 │
│        │                                                         │
│        ▼                                                         │
│   00_validate_config.py ── config_validation.json                │
│        │                                                         │
│        ▼                                                         │
│   01_build_manifests.py ── manifests/conditions_*.csv            │
│                          ── manifests/prompt_manifest.csv        │
│        │                                                         │
│        ▼                                                         │
│   02_draw_fewshot_examples.py ── manifests/fewshot_manifest.csv  │
│        │                         preregistration/appendices/     │
│        │                           fewshot_draws_LOCKED.json     │
│        ▼                                                         │
│   03_validate_prompts.py ── manifests/prompt_manifest.csv        │
│                                                                  │
│   ═══════════════════ GPU-SIDE INFERENCE ═══════════════════     │
│       consumes: conditions_*.csv, prompts/*, fewshot_manifest    │
│       produces: runs/raw/{run_id}/*                              │
│   ═══════════════════════════════════════════════════════════    │
│                                                                  │
│   04_normalize_outputs.py ── runs/normalized/item_level_*.pq     │
│        │                                                         │
│        ▼                                                         │
│   05_make_condition_scores.py ── runs/normalized/                │
│        │                           condition_level_*.csv         │
│        ▼                                                         │
│   06_variance_components.py ── analysis/variance_components/*    │
│        │                                                         │
│        ▼                                                         │
│   07_tolerance_schedule.py ── analysis/tolerance_schedules/*     │
│        │                                                         │
│        ▼                                                         │
│   08_ranking_stability.py ── analysis/ranking_stability/*        │
│        │                                                         │
│        ▼                                                         │
│   09_mmlu_subject_decomp.py ── analysis/mmlu_subjects/*          │
│        │                                                         │
│        ▼                                                         │
│   10_gsm8k_case.py ── analysis/gsm8k_case/*                      │
│        │                                                         │
│        ▼                                                         │
│   98_reproducibility_trace.py ── reports/                        │
│        │                           reproducibility_trace.md      │
│        ▼                                                         │
│   99_make_report.py ── reports/cpu_pipeline_report.md            │
│                         tables/* , figures/*                     │
└──────────────────────────────────────────────────────────────────┘
```

---

## 3. Directory Layout

Target structure (relative to repository root):

```
papers/benchmark_reliability_study/
  README.md
  SPEC_CPU_v0_2.md                         # this file
  preregistration/
    prereg_v1.md
    prereg_v1_1_LOCKED.md
    appendices/
      prompts_LOCKED.md
      admissibility_sources_LOCKED.md
      fewshot_draws_LOCKED.json
  configs/
    study_config.yaml
    models.yaml
    benchmarks.yaml
    prompts.yaml
    scoring_rules.yaml
    analysis_config.yaml
  prompts/
    arc_challenge/
      P1_original.txt
      P2_lm_eval.txt
      P3_helm_or_published.txt
      P4_minimal_sourced.txt
    hellaswag/
    truthfulqa_mc/
    mmlu_panel/
    winogrande/
    gsm8k/
  schemas/                                  # NEW in v0.2
    run_metadata.schema.json
    item_outputs.schema.json
    item_scores.schema.json
    item_level_normalized.schema.json
    condition_level.schema.json
    tolerance_schedule.schema.json
  manifests/
    conditions_primary.csv
    conditions_gsm8k.csv
    items_manifest.csv
    fewshot_manifest.csv
    prompt_manifest.csv
    scoring_manifest.csv
  runs/
    raw/
      {run_id}/
        run_metadata.json
        item_outputs.jsonl
        item_scores.jsonl                   # LL scoring rule only
    normalized/
      item_level_primary.parquet
      item_level_gsm8k.parquet
      condition_level_primary.csv
      condition_level_gsm8k.csv
  analysis/
    variance_components/
      item_level_vc.csv
      aggregate_vc.csv
      model_convergence_report.csv
      bootstrap_samples.parquet
    tolerance_schedules/
      tolerance_by_cell.csv
      tolerance_by_benchmark_summary.csv
      h1_test.json
    ranking_stability/
      ranking_reversals.csv
      pairwise_win_probabilities.csv
    mmlu_subjects/
      mmlu_subject_accuracy_matrix.csv
      mmlu_subject_variance_components.csv
    gsm8k_case/
      gsm8k_tolerance_schedule.csv
      gsm8k_extraction_sensitivity.csv
      gsm8k_parseability.csv
  figures/
  tables/
  reports/
    config_validation.json
    cpu_pipeline_report.md
    deviations.md
    reproducibility_trace.md
  scripts/
    00_validate_config.py
    01_build_manifests.py
    02_draw_fewshot_examples.py
    03_validate_prompts.py
    04_normalize_outputs.py
    05_make_condition_scores.py
    06_variance_components.py
    07_tolerance_schedule.py
    08_ranking_stability.py
    09_mmlu_subject_decomp.py
    10_gsm8k_case.py
    98_reproducibility_trace.py
    99_make_report.py
  tests/
    test_config_validation.py
    test_condition_manifest.py
    test_prompt_manifest.py
    test_fewshot_leakage.py                 # NEW in v0.2
    test_output_schema.py
    test_normalization.py
    test_tolerance_math.py
    test_bootstrap.py                       # NEW in v0.2
    test_mixed_effects_cascade.py           # NEW in v0.2
    test_config_hash.py                     # NEW in v0.2
    fixtures/
      make_fixtures.py                      # NEW in v0.2
      tiny_item_outputs.jsonl
      tiny_item_level.parquet
      tiny_condition_scores.csv
      pathological_all_parse_fail.jsonl     # NEW in v0.2
      pathological_near_tied_ll.jsonl       # NEW in v0.2
```

---

## 4. Configuration System

### 4.1 Config files

Six YAML files constitute the study configuration:

| File | Purpose | Required |
|---|---|---|
| `configs/study_config.yaml` | Top-level orchestration: study ID, tier enablement, output paths | Yes |
| `configs/models.yaml` | Model registry: `model_id`, HF name, family, type, parameter count | Yes |
| `configs/benchmarks.yaml` | Benchmark registry: task type, few-shot k, scoring rules, item split | Yes |
| `configs/prompts.yaml` | Prompt registry: prompt ID, source, provenance metadata, admissibility | Yes |
| `configs/scoring_rules.yaml` | Scoring rule definitions: identifier, implementation, normalization | Yes |
| `configs/analysis_config.yaml` | Analysis parameters: bootstrap counts, mixed-effects cascade, thresholds | Yes |

### 4.2 Top-level `study_config.yaml`

Required fields:

```yaml
study_id: benchmark_reliability_v1
prereg_version: v1_1_LOCKED
tier_primary_enabled: true
tier_secondary_enabled: true        # GSM8K; set false to skip
extension_7b_enabled: false          # optional Mistral-7B pass
paths:
  configs_dir: configs/
  manifests_dir: manifests/
  prompts_dir: prompts/
  runs_raw_dir: runs/raw/
  runs_normalized_dir: runs/normalized/
  analysis_dir: analysis/
  reports_dir: reports/
  schemas_dir: schemas/
includes:
  - configs/models.yaml
  - configs/benchmarks.yaml
  - configs/prompts.yaml
  - configs/scoring_rules.yaml
  - configs/analysis_config.yaml
```

### 4.3 `analysis_config.yaml` (new in v0.2)

```yaml
bootstrap:
  n_resamples: 10000
  random_seed: 20260424
  ci_lower_percentile: 2.5
  ci_upper_percentile: 97.5

mixed_effects_cascade:
  level_1:
    random_effects: [prompt, seed, scoring_rule, item,
                     model_prompt_interaction, model_scoring_rule_interaction]
  level_2:
    random_effects: [prompt, seed, scoring_rule, item]
  level_3:
    random_effects: [prompt, scoring_rule, item]
  level_4:
    method: aggregate_g_theory_only

convergence_triggers:
  max_singular_warning: trigger_fallback
  hessian_non_pd: trigger_fallback
  iteration_limit_exceeded: trigger_fallback
  gradient_norm_above: 1.0e-3

tolerance:
  decision_rule_threshold_h1: 0.005
  benchmarks_required_for_h1: 3

h2_generalizability_threshold: 0.80
h3_ranking_reversal_threshold: 0.20  # v1.1-draft: raised from 0.10 per prereg §14.5 resolution
h4_mmlu_interaction_threshold: 0.10
```

### 4.4 Config hash formalism

The `config_hash` field in condition manifests ties every row to a specific config state. Computation is deterministic and defined as follows:

**Canonical config object.** Load all six YAML files listed in §4.1 into a single dict with schema:

```python
merged = {
    "study_config":    <contents of study_config.yaml>,
    "models":          <contents of models.yaml>,
    "benchmarks":      <contents of benchmarks.yaml>,
    "prompts":         <contents of prompts.yaml>,
    "scoring_rules":   <contents of scoring_rules.yaml>,
    "analysis_config": <contents of analysis_config.yaml>,
}
```

**Canonical serialization.** Serialize the merged dict using `yaml.safe_dump` with deterministic flags:

```python
import yaml

def canonicalize(merged: dict) -> str:
    return yaml.safe_dump(
        merged,
        sort_keys=True,
        default_flow_style=False,
        allow_unicode=True,
        width=10**9,       # prevent line-wrapping differences
    )
```

**Hash.** Compute SHA-256 of the UTF-8 canonical bytes. Store the full hex digest in `reports/config_validation.json` as `config_hash_full`; use the first 8 hex chars as `config_hash` in the condition manifest for readability.

```python
import hashlib

def compute_config_hash(merged: dict) -> tuple[str, str]:
    canonical = canonicalize(merged).encode("utf-8")
    full = hashlib.sha256(canonical).hexdigest()
    short = full[:8]
    return full, short
```

**Acceptance criterion.** Rerunning `scripts/00_validate_config.py` with unchanged configs must produce an identical `config_hash_full`. Any bit-level change to any of the six YAML files, including comment changes or reordering, produces a different hash. (If comment-invariance is desired, the `canonicalize` function can strip comments during a YAML load; v0.2 specifies strict bit-level hashing because strict is simpler and safer.)

**Library pin.** Use `PyYAML >= 6.0`, pinned in §17.

### 4.5 Config validation rules

`scripts/00_validate_config.py` must verify:

- All six config files exist at declared paths.
- All required fields are present in each file.
- `primary` models count equals 3 (unless an explicit deviation is declared in `study_config.yaml`).
- Primary benchmarks count equals 5.
- For each benchmark, a `prompts.yaml` entry exists for each of P1, P2, P3, P4.
- Every prompt file referenced by `prompts.yaml` exists on disk.
- Every prompt file's computed SHA-256 matches the `content_hash` declared in `prompts.yaml` (if `content_hash` is present; if absent, validation computes and records it).
- `scoring_rules.yaml` declares exactly the scoring rule IDs referenced in `benchmarks.yaml`.

Validation failure exits with a non-zero status code per §12.

---

## 5. Data Contracts

### 5.1 GPU-side input contract (new in v0.2)

The GPU-side inference backend must accept:

| Input | Path | Purpose |
|---|---|---|
| Conditions manifest | `manifests/conditions_primary.csv` (and/or `conditions_gsm8k.csv`) | List of runs to execute |
| Prompt files | `prompts/{benchmark_id}/{prompt_id}.txt` | Template text with placeholders |
| Few-shot manifest | `manifests/fewshot_manifest.csv` | Per-(benchmark, seed) exemplar item IDs |
| Benchmark item data | Canonical Hugging Face dataset path, pinned to `dataset_version_hash` from `benchmarks.yaml` | Source items |
| Model checkpoints | Canonical Hugging Face model paths from `models.yaml` | Model weights |

The GPU-side backend must produce, for every completed condition in `manifests/conditions_*.csv`:

- Directory `runs/raw/{run_id}/` where `run_id = condition_id`.
- File `runs/raw/{run_id}/run_metadata.json` conforming to §5.2.
- File `runs/raw/{run_id}/item_outputs.jsonl` (for G&P scoring rules) conforming to §5.3.
- File `runs/raw/{run_id}/item_scores.jsonl` (for LL scoring rules) conforming to §5.4.

The contract is intentionally inference-tool-agnostic. Acceptable implementations include:

- Local `transformers` + `torch` with hand-rolled loops.
- `lm-evaluation-harness` with an output adapter that emits the contract schemas.
- Remote inference (e.g., RunPod, Modal) with an output-collection shim.
- API-backed evaluation (for models accessible via API) if the scoring rules can be realized through the API.

The GPU-side backend is responsible for honoring the `scoring_rule_id` in the conditions manifest: LL scoring produces `item_scores.jsonl`, G&P produces `item_outputs.jsonl`. A single run producing both is acceptable only if both scoring rules are listed in the condition (v0.2 does not require the GPU side to run both — each condition row is one scoring rule).

### 5.2 `run_metadata.json` schema

JSON Schema (store at `schemas/run_metadata.schema.json`):

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "required": [
    "run_id", "condition_id", "model_id", "hf_name",
    "benchmark_id", "prompt_id", "seed_id", "fewshot_k",
    "scoring_rule_id", "task_type", "num_items_expected",
    "num_items_completed", "inference_backend",
    "python_version", "transformers_version", "torch_version",
    "device", "dtype", "started_at", "finished_at", "status"
  ],
  "properties": {
    "run_id":                {"type": "string"},
    "condition_id":          {"type": "string"},
    "model_id":              {"type": "string"},
    "hf_name":               {"type": "string"},
    "benchmark_id":          {"type": "string"},
    "subject_id":            {"type": ["string", "null"]},
    "prompt_id":             {"type": "string"},
    "seed_id":               {"type": ["string", "null"]},
    "fewshot_k":             {"type": "integer", "minimum": 0},
    "scoring_rule_id":       {"type": "string"},
    "task_type":             {"type": "string",
                              "enum": ["constrained_choice", "open_generation"]},
    "num_items_expected":    {"type": "integer", "minimum": 1},
    "num_items_completed":   {"type": "integer", "minimum": 0},
    "inference_backend":     {"type": "string"},
    "python_version":        {"type": "string"},
    "transformers_version":  {"type": "string"},
    "torch_version":         {"type": "string"},
    "lm_eval_version":       {"type": ["string", "null"]},
    "device":                {"type": "string"},
    "dtype":                 {"type": "string"},
    "started_at":            {"type": "string", "format": "date-time"},
    "finished_at":           {"type": "string", "format": "date-time"},
    "status":                {"type": "string",
                              "enum": ["complete", "failed", "partial"]},
    "notes":                 {"type": "string"}
  }
}
```

`run_id` MUST equal `condition_id`. If they differ, the run is rejected by `04_normalize_outputs.py`.

`seed_id` is `null` when `fewshot_k == 0` (e.g., TruthfulQA-MC 0-shot).

### 5.3 `item_outputs.jsonl` schema (G&P scoring rule)

One JSON object per item per line. Required fields:

```json
{
  "condition_id":         "string",
  "item_id":              "string",
  "subject_id":           "string | null",
  "prompt_text_hash":     "sha256 hex string",
  "rendered_prompt_hash": "sha256 hex string",
  "gold_answer":          "string",
  "raw_generation":       "string",
  "parsed_answer":        "string | null",
  "parse_status":         "parsed | unparseable | empty | multiple_answers | invalid_choice | runtime_error",
  "is_correct":           "boolean",
  "scoring_rule_id":      "string",
  "generation_params":    {
    "decoding":     "greedy",
    "temperature":  0,
    "max_new_tokens": 32
  }
}
```

- `prompt_text_hash` is SHA-256 of the raw prompt template file content (no few-shot examples filled in).
- `rendered_prompt_hash` is SHA-256 of the final prompt sent to the model, including filled-in few-shot examples and the item's question.
- `is_correct` is computed by the GPU side per the scoring rule; normalized correctness logic is re-verified CPU-side in `04_normalize_outputs.py`.

### 5.4 `item_scores.jsonl` schema (LL scoring rule)

```json
{
  "condition_id":            "string",
  "item_id":                 "string",
  "subject_id":              "string | null",
  "prompt_text_hash":        "sha256 hex string",
  "rendered_prompt_hash":    "sha256 hex string",
  "gold_answer":             "string",
  "choices":                 ["string"],
  "choice_scores":           {"A": -2.44, "B": -1.92, "C": -2.31, "D": -3.10},
  "choice_token_counts":     {"A": 4, "B": 5, "C": 4, "D": 6},
  "normalization":           "length_normalized | sum",
  "selected_answer":         "string",
  "is_correct":              "boolean",
  "scoring_rule_id":         "string"
}
```

### 5.5 Normalized item-level parquet schema

Path: `runs/normalized/item_level_primary.parquet` (and `item_level_gsm8k.parquet`).

Required columns (Arrow types in parentheses):

```
condition_id              (string)
tier                      (string)  -- primary | secondary
benchmark_id              (string)
subject_id                (string, nullable)
model_id                  (string)
model_family              (string)
model_type                (string)  -- base | instruct
parameter_count_b         (float32)
prompt_id                 (string)
prompt_source_type        (string)
seed_id                   (string, nullable)
fewshot_k                 (int32)
scoring_rule_id           (string)
task_type                 (string)
item_id                   (string)
gold_answer               (string)
predicted_answer          (string, nullable)
is_correct                (bool)
parse_status              (string, nullable)   -- null for LL rows
parseable                 (bool, nullable)     -- null for LL rows
choice_count              (int32, nullable)    -- null for non-MC
prompt_text_hash          (string)
rendered_prompt_hash      (string)
run_id                    (string)
created_at                (timestamp[us, UTC])

-- LL-only columns, nullable for G&P rows
selected_answer           (string, nullable)
ll_margin_top2            (float32, nullable)
gold_choice_score         (float32, nullable)
selected_choice_score     (float32, nullable)

-- G&P-only columns, nullable for LL rows
raw_generation            (string, nullable)
parsed_answer             (string, nullable)
generation_length_tokens  (int32, nullable)
```

`ll_margin_top2` = (top-1 choice score) − (top-2 choice score); a near-tie diagnostic for LL scoring.

### 5.6 Condition-level CSV schema

Path: `runs/normalized/condition_level_primary.csv`.

Required columns:

```
condition_id             (string)
tier                     (string)
benchmark_id             (string)
subject_id               (string, nullable)
model_id                 (string)
prompt_id                (string)
seed_id                  (string, nullable)
fewshot_k                (int)
scoring_rule_id          (string)
num_items                (int)
num_correct              (int)
accuracy                 (float)
parseability_rate        (float, nullable)   -- null for LL
mean_generation_length_tokens (float, nullable)
condition_status         (string)
```

---

## 6. Study Objects

### 6.1 Model registry (`configs/models.yaml`)

```yaml
models:
  - model_id: pythia_410m
    hf_name: EleutherAI/pythia-410m
    hf_revision: <commit-hash-locked-at-v1.1>
    model_family: pythia
    model_type: base
    parameter_count_b: 0.41
    primary: true
    optional_extension: false

  - model_id: pythia_1_4b
    hf_name: EleutherAI/pythia-1.4b
    hf_revision: <commit-hash-locked-at-v1.1>
    model_family: pythia
    model_type: base
    parameter_count_b: 1.4
    primary: true
    optional_extension: false

  - model_id: qwen2_5_1_5b_instruct
    hf_name: Qwen/Qwen2.5-1.5B-Instruct
    hf_revision: <commit-hash-locked-at-v1.1>
    model_family: qwen2_5
    model_type: instruct
    parameter_count_b: 1.5
    primary: true
    optional_extension: false

  - model_id: mistral_7b_v0_3
    hf_name: mistralai/Mistral-7B-v0.3
    hf_revision: <commit-hash-locked-at-v1.1>
    model_family: mistral
    model_type: base
    parameter_count_b: 7.0
    primary: false
    optional_extension: true
```

`hf_revision` is a Hugging Face commit hash, locked at v1.1 pre-registration. Pipeline validation rejects model loads without an `hf_revision` field.

### 6.2 Benchmark registry (`configs/benchmarks.yaml`)

```yaml
benchmarks:
  - benchmark_id: arc_challenge
    display_name: ARC-Challenge
    tier: primary
    hf_dataset: allenai/ai2_arc
    hf_config: ARC-Challenge
    dataset_version_hash: <commit-hash-locked-at-v1.1>
    task_type: constrained_choice
    fewshot_k: 5
    seed_facet: true
    fewshot_source_split: train
    eval_split: test
    scoring_rules: [ll_norm, generate_parse]
    item_id_field: id
    answer_field: answerKey
    expected_num_items: 1172

  - benchmark_id: hellaswag
    display_name: HellaSwag
    tier: primary
    hf_dataset: Rowan/hellaswag
    hf_config: null
    dataset_version_hash: <commit-hash>
    task_type: constrained_choice
    fewshot_k: 5
    seed_facet: true
    fewshot_source_split: train
    eval_split: validation
    scoring_rules: [ll_norm, generate_parse]
    item_id_field: ind
    answer_field: label
    expected_num_items: 10042

  - benchmark_id: truthfulqa_mc
    display_name: TruthfulQA-MC
    tier: primary
    hf_dataset: truthful_qa
    hf_config: multiple_choice
    dataset_version_hash: <commit-hash>
    task_type: constrained_choice
    fewshot_k: 0
    seed_facet: false
    fewshot_source_split: null
    eval_split: validation
    scoring_rules: [ll_norm, generate_parse]
    item_id_field: question
    answer_field: mc1_targets.labels
    expected_num_items: 817

  - benchmark_id: mmlu_panel
    display_name: MMLU Subject Panel
    tier: primary
    hf_dataset: cais/mmlu
    hf_config: null
    dataset_version_hash: <commit-hash>
    task_type: constrained_choice
    fewshot_k: 5
    seed_facet: true
    fewshot_source_split: dev
    eval_split: test
    scoring_rules: [ll_norm, generate_parse]
    item_id_field: question
    answer_field: answer
    subjects:
      - world_religions
      - high_school_mathematics
      - psychology
      - professional_medicine
      - global_facts
    expected_num_items_per_subject: <locked-at-v1.1>

  - benchmark_id: winogrande
    display_name: Winogrande
    tier: primary
    hf_dataset: winogrande
    hf_config: winogrande_xl
    dataset_version_hash: <commit-hash>
    task_type: constrained_choice
    fewshot_k: 5
    seed_facet: true
    fewshot_source_split: train
    eval_split: validation
    scoring_rules: [ll_norm, generate_parse]
    item_id_field: qID
    answer_field: answer
    expected_num_items: 1267

  - benchmark_id: gsm8k
    display_name: GSM8K
    tier: secondary
    hf_dataset: gsm8k
    hf_config: main
    dataset_version_hash: <commit-hash>
    task_type: open_generation
    fewshot_k: 5
    seed_facet: true
    fewshot_source_split: train
    eval_split: test
    scoring_rules: [generate_parse_strict, generate_parse_permissive]
    item_id_field: question
    answer_field: answer
    expected_num_items: 1319
```

### 6.3 Prompt registry (`configs/prompts.yaml`)

Four prompts per benchmark, each with provenance:

```yaml
prompts:
  - benchmark_id: arc_challenge
    prompt_id: P1_original
    prompt_path: prompts/arc_challenge/P1_original.txt
    content_hash: <sha256-locked-at-v1.1>
    source_type: benchmark_authors
    source_url_or_ref: "<locked source URL or citation>"
    source_commit: "<commit hash if from GitHub>"
    admissibility_status: locked
    placeholders_used:
      - fewshot_examples
      - question
      - choices
      - answer_instruction
    notes: "Original benchmark prompt."
```

`source_type` must be one of:

```
benchmark_authors | lm_eval_harness | helm_reference |
published_variant | author_minimal_declared
```

`author_minimal_declared` is the only source type permitted for prompts not traceable to an external community source; it requires an explicit declaration in `preregistration/appendices/admissibility_sources_LOCKED.md` that the prompt is an author-constructed minimal variant justified under the §4.1 admissibility rule.

### 6.4 Scoring rule registry (`configs/scoring_rules.yaml`)

```yaml
scoring_rules:
  - scoring_rule_id: ll_norm
    display_name: Length-Normalized Log-Likelihood
    applies_to_task_types: [constrained_choice]
    normalization: length_normalized
    selection_rule: argmax

  - scoring_rule_id: generate_parse
    display_name: Generate and Regex-Parse
    applies_to_task_types: [constrained_choice]
    generation_params:
      decoding: greedy
      temperature: 0
      max_new_tokens: 32
    parse_strategy: benchmark_specific_regex

  - scoring_rule_id: generate_parse_strict
    display_name: GSM8K Strict Exact-Match
    applies_to_task_types: [open_generation]
    generation_params:
      decoding: greedy
      temperature: 0
      max_new_tokens: 256
    parse_strategy: exact_match_final_number

  - scoring_rule_id: generate_parse_permissive
    display_name: GSM8K Permissive Regex
    applies_to_task_types: [open_generation]
    generation_params:
      decoding: greedy
      temperature: 0
      max_new_tokens: 256
    parse_strategy: permissive_number_regex
```

---

## 7. Condition Manifest

### 7.1 Schema

`manifests/conditions_primary.csv` and `manifests/conditions_gsm8k.csv`. Required columns:

```
condition_id              (string, unique)
tier                      (string)  -- primary | secondary
benchmark_id              (string)
subject_id                (string, nullable)  -- populated for mmlu_panel rows
model_id                  (string)
prompt_id                 (string)
seed_id                   (string, nullable)
fewshot_k                 (int)
scoring_rule_id           (string)
task_type                 (string)
expected_num_items        (int)
condition_status          (string)
created_at                (ISO8601)
config_hash               (string)  -- 8-char short hash from §4.4
```

### 7.2 Condition ID scheme

```
condition_id = {benchmark_id}__{subject_id_or_empty}__{model_id}__{prompt_id}__{seed_id_or_0shot}__{scoring_rule_id}
```

Examples:

```
arc_challenge__pythia_410m__P1_original__s42__ll_norm
mmlu_panel__world_religions__qwen2_5_1_5b_instruct__P3_helm_or_published__s123__generate_parse
truthfulqa_mc__pythia_1_4b__P2_lm_eval__0shot__ll_norm
```

Subject field is empty (`""`) for benchmarks without subjects; `0shot` is the literal seed marker for `fewshot_k == 0` conditions.

### 7.3 Status values

```
pending              -- enumerated, not yet started
running              -- inference in progress
complete             -- inference finished, outputs present, schema-validated
failed               -- inference failed; reason logged; output absent
excluded_pre_run     -- removed from universe before inference; deviation required
excluded_post_run    -- removed from analysis after inference; deviation required
```

Removing a condition requires a `deviations.md` entry, per §14.

---

## 8. Scripts

### 8.1 `00_validate_config.py`

**Purpose.** Validate all six config files; compute and record config hash.

**Signature:**

```bash
python scripts/00_validate_config.py \
  --config configs/study_config.yaml \
  --out reports/config_validation.json
```

**Inputs.** All files referenced by `study_config.yaml.includes`.

**Outputs.** `reports/config_validation.json`:

```json
{
  "config_hash_full": "sha256-hex-digest...",
  "config_hash": "abc12345",
  "validation_status": "pass | fail",
  "validation_errors": [],
  "validation_warnings": [],
  "primary_model_count": 3,
  "primary_benchmark_count": 5,
  "total_prompts": 24,
  "total_nominal_conditions": 360,
  "timestamp": "2026-04-24T..."
}
```

**Exit codes:** 0 on pass; 2 on validation failure (per §12).

### 8.2 `01_build_manifests.py`

**Purpose.** Generate condition and prompt manifests from configs.

**Signature:**

```bash
python scripts/01_build_manifests.py \
  --config configs/study_config.yaml \
  --out-dir manifests/
```

**Outputs.**

- `manifests/conditions_primary.csv`
- `manifests/conditions_gsm8k.csv`
- `manifests/prompt_manifest.csv`
- `manifests/scoring_manifest.csv`
- `manifests/items_manifest.csv` (optional: list of all eval items per benchmark with subject tags)

**Acceptance criteria.**

- Rerunning with unchanged configs produces byte-identical CSV files (deterministic order, deterministic timestamps replaced with `config_hash`-derived seed).
- Nominal primary condition count equals `3 × 4 × 3 × 2 × (number of primary benchmarks with seed_facet=true) + 3 × 4 × 1 × 2 × (number of 0-shot primary benchmarks)` = 3 × 24 × 4 + 3 × 8 × 1 = 288 + 24 = 312 (if four primary benchmarks have seed_facet and one is 0-shot). The exact count is a function of the benchmark config; compute it in the script and log it.
- MMLU panel conditions are expanded per subject.

### 8.3 `02_draw_fewshot_examples.py`

**Purpose.** Deterministically draw few-shot exemplars per (benchmark, subject, seed); verify no leakage.

**Signature:**

```bash
python scripts/02_draw_fewshot_examples.py \
  --config configs/study_config.yaml \
  --seeds 42 123 2024 \
  --out manifests/fewshot_manifest.csv \
  --lock-out preregistration/appendices/fewshot_draws_LOCKED.json
```

**Draw algorithm:**

```python
def draw_fewshot(benchmark_cfg, subject_id, seed, k) -> list[item_id]:
    rng = numpy.random.default_rng(seed)
    source = load_hf_split(
        benchmark_cfg.hf_dataset,
        benchmark_cfg.hf_config,
        split=benchmark_cfg.fewshot_source_split,
        revision=benchmark_cfg.dataset_version_hash,
    )
    if subject_id is not None:
        source = source.filter(lambda x: x["subject"] == subject_id)
    indices = rng.choice(len(source), size=k, replace=False)
    return [source[i][benchmark_cfg.item_id_field] for i in indices]
```

**Leakage check (new acceptance criterion in v0.2):**

```python
def validate_no_leakage(fewshot_manifest, benchmarks_cfg) -> None:
    for benchmark in benchmarks_cfg:
        fewshot_ids = set(fewshot_manifest
                          [fewshot_manifest.benchmark_id == benchmark.benchmark_id]
                          ["example_item_id"])
        eval_ids = set(load_hf_split(
            benchmark.hf_dataset, benchmark.hf_config,
            split=benchmark.eval_split,
            revision=benchmark.dataset_version_hash,
        )[benchmark.item_id_field])
        intersection = fewshot_ids & eval_ids
        if intersection:
            raise FewshotLeakageError(
                f"Benchmark {benchmark.benchmark_id}: "
                f"{len(intersection)} item(s) in both fewshot and eval "
                f"sets. Leaked IDs: {sorted(intersection)[:10]}..."
            )
```

Pipeline aborts hard (exit code 3) on any leakage detected. No warnings, no overrides.

**Acceptance criteria.**

- Rerunning with identical seeds and identical `dataset_version_hash` produces identical draws.
- Leakage check passes for every benchmark.
- The lock file `fewshot_draws_LOCKED.json` has a declared format (JSON; schema in `schemas/` to be added) that includes the draw algorithm's commit hash.

### 8.4 `03_validate_prompts.py`

**Purpose.** Validate prompt provenance; compute prompt hashes; check placeholder presence.

**Signature:**

```bash
python scripts/03_validate_prompts.py \
  --prompts-config configs/prompts.yaml \
  --out manifests/prompt_manifest.csv
```

**Acceptance criteria.**

- Every prompt file exists on disk.
- SHA-256 of file content matches `content_hash` in `prompts.yaml`.
- Every prompt file contains all placeholders declared in its `placeholders_used` list (checked via literal string search for `{{placeholder_name}}`).
- `admissibility_status == "locked"` for all primary-panel prompts (pre-lock prompts produce a warning; unlocked prompts fail validation if `study_config.yaml` sets `require_locked_prompts: true`).
- P4 prompts with `source_type == "author_minimal_declared"` have a corresponding entry in `preregistration/appendices/admissibility_sources_LOCKED.md`.

### 8.5 `04_normalize_outputs.py`

**Purpose.** Ingest raw GPU-side outputs; validate schemas; normalize to item-level parquet.

**Signature:**

```bash
python scripts/04_normalize_outputs.py \
  --conditions manifests/conditions_primary.csv \
  --raw-dir runs/raw/ \
  --schemas-dir schemas/ \
  --out runs/normalized/item_level_primary.parquet
```

**Normalization rules.**

1. For each condition in the manifest with `condition_status == "complete"`, open `runs/raw/{run_id}/run_metadata.json`.
2. Validate against `schemas/run_metadata.schema.json`. Validation failure → condition marked `failed` in output manifest with reason logged.
3. Verify `run_metadata.condition_id == condition.condition_id`. Mismatch → condition failed.
4. Verify `run_metadata.num_items_completed == run_metadata.num_items_expected`. Partial completion → condition failed (no silent truncation).
5. Load `item_outputs.jsonl` (for G&P) or `item_scores.jsonl` (for LL) per `scoring_rule_id`.
6. Validate each row against corresponding schema.
7. Emit one row per item into the normalized parquet, populating the LL-specific or G&P-specific columns and leaving the other nullable.
8. Drop no rows silently. If a row fails schema validation, the entire condition is marked `failed`.

**Acceptance criteria.**

- For every `condition_status == "complete"` input row, the normalized parquet contains exactly `expected_num_items` rows.
- Zero silent row drops.
- Schema validation passes for every emitted row.

### 8.6 `05_make_condition_scores.py`

**Purpose.** Aggregate item-level correctness to condition-level accuracy; compute parseability.

**Signature:**

```bash
python scripts/05_make_condition_scores.py \
  --item-level runs/normalized/item_level_primary.parquet \
  --out runs/normalized/condition_level_primary.csv
```

**Aggregation rules.**

```python
for condition_id, group in item_level.groupby("condition_id"):
    num_items = len(group)
    num_correct = group["is_correct"].sum()
    accuracy = num_correct / num_items
    parseability = (
        group["parseable"].sum() / num_items
        if group["parseable"].notna().any()
        else None
    )
    mean_gen_length = (
        group["generation_length_tokens"].mean()
        if group["generation_length_tokens"].notna().any()
        else None
    )
    emit(condition_id, num_items, num_correct, accuracy,
         parseability, mean_gen_length)
```

**Acceptance criteria.**

- Number of output rows equals number of `condition_status == "complete"` input rows in the condition manifest.
- `accuracy == num_correct / num_items` for every row (deterministic float computation; bit-level reproducibility not required but recommended).

### 8.7 `06_variance_components.py`

**Purpose.** Fit mixed-effects logistic regression at item level with pre-registered cascade; run aggregate-score G-theory decomposition.

**Signature:**

```bash
python scripts/06_variance_components.py \
  --item-level runs/normalized/item_level_primary.parquet \
  --condition-level runs/normalized/condition_level_primary.csv \
  --config configs/analysis_config.yaml \
  --out-dir analysis/variance_components/
```

**Mixed-effects cascade (pre-registered):**

```
For each benchmark_id:
  attempt = 1
  For level in [level_1, level_2, level_3]:
    try:
      fit mixed-effects logistic model with random effects
      from analysis_config.mixed_effects_cascade[level]
      check convergence criteria from analysis_config.convergence_triggers
      if converged: break with success
    except ConvergenceFailure:
      log to model_convergence_report.csv with
        level, error_type, gradient_norm, attempt
      attempt += 1
      continue
  If no level in [1,2,3] converged:
    fall through to level_4: aggregate_g_theory_only
    log level_4 fallback to reports/deviations.md
```

**Convergence criteria (all must hold for "converged"):**

- No singular-fit warnings in the optimizer's output.
- Hessian is positive definite.
- Gradient norm < `analysis_config.convergence_triggers.gradient_norm_above` (default 1e-3).
- Iteration limit not exceeded.

**Libraries.** Use `statsmodels.MixedLM` or `rpy2` + `lme4::glmer` (binomial family, logit link). The choice is implementation-level; `statsmodels` is simpler and preferred unless convergence rates on the actual data require `lme4`.

**Outputs.**

- `analysis/variance_components/item_level_vc.csv`: per-benchmark variance components with columns `benchmark_id`, `cascade_level_used`, `random_effect`, `variance`, `proportion`.
- `analysis/variance_components/aggregate_vc.csv`: per-(benchmark, model) aggregate-score variance components with columns `benchmark_id`, `model_id`, `source`, `variance`, `proportion`, `df`.
- `analysis/variance_components/model_convergence_report.csv`: per-benchmark log of every attempted cascade level, convergence status, diagnostic.

### 8.8 `07_tolerance_schedule.py`

**Purpose.** Compute SEM, tolerance, decimal-place licensing; bootstrap CIs.

**Signature:**

```bash
python scripts/07_tolerance_schedule.py \
  --condition-level runs/normalized/condition_level_primary.csv \
  --variance-components analysis/variance_components/aggregate_vc.csv \
  --analysis-config configs/analysis_config.yaml \
  --out-dir analysis/tolerance_schedules/
```

**SEM definitions (§9.2 below for full discussion):**

```
SEM_single = sqrt(var_prompt + var_seed + var_scoring_rule + var_residual)
SEM_within_rule = sqrt(var_prompt + var_seed + var_residual)
SEM_prompt_avg_4 = SEM_single / sqrt(4)
SEM_full_design = SEM_single / sqrt(24)
tolerance = 2 * SEM
```

**Bootstrap CIs (new in v0.2):**

```python
def bootstrap_tolerance(condition_scores, analysis_config) -> dict:
    n_boot = analysis_config.bootstrap.n_resamples   # default 10000
    rng = np.random.default_rng(analysis_config.bootstrap.random_seed)
    boot_tolerances = []
    for _ in range(n_boot):
        resampled = rng.choice(condition_scores, size=len(condition_scores),
                               replace=True)
        sem = resampled.std(ddof=1)
        boot_tolerances.append(2 * sem)
    return {
        "tolerance_point":     2 * condition_scores.std(ddof=1),
        "tolerance_ci_lower":  np.percentile(boot_tolerances, 2.5),
        "tolerance_ci_upper":  np.percentile(boot_tolerances, 97.5),
    }
```

**Bootstrap is computed for all four tolerance levels** (single-occasion, 4-prompt-averaged, full-design, within-rule) and for both scoring rules separately.

**Decimal-place licensing (revised in v0.2):**

Three-level categorical (not four, as v0.1 had):

```python
def licensed_precision(tolerance: float) -> str:
    if tolerance < 0.0005:
        return "three_decimal_accuracy"   # 0.742 licensed; equiv 74.2%
    elif tolerance < 0.005:
        return "two_decimal_accuracy"     # 0.74 licensed;  equiv 74%
    else:
        return "interval_required"        # no rounded point estimate
```

The presentation table in the manuscript uses a single notation family (decimal-accuracy) as primary, with percentage-point equivalent as a parenthetical.

**Outputs.**

- `analysis/tolerance_schedules/tolerance_by_cell.csv`:
  ```
  benchmark_id, model_id, scoring_rule_id,
  sem_single, tolerance_single, tolerance_single_ci_lower, tolerance_single_ci_upper,
  sem_within_rule, tolerance_within_rule, tolerance_within_rule_ci_lower, tolerance_within_rule_ci_upper,
  sem_prompt_avg, tolerance_prompt_avg, tolerance_prompt_avg_ci_lower, tolerance_prompt_avg_ci_upper,
  sem_full_design, tolerance_full_design, tolerance_full_design_ci_lower, tolerance_full_design_ci_upper,
  licensed_precision_single, licensed_precision_full_design
  ```
- `analysis/tolerance_schedules/tolerance_by_benchmark_summary.csv`: cross-model-median tolerance per benchmark.
- `analysis/tolerance_schedules/h1_test.json`: H1 decision result.

**H1 decision rule (revised in v0.2 to use bootstrap lower bound):**

```python
def test_h1(benchmark_summaries, analysis_config) -> dict:
    threshold = analysis_config.tolerance.decision_rule_threshold_h1  # 0.005
    n_req    = analysis_config.tolerance.benchmarks_required_for_h1  # 3
    benchmarks_exceeding = [
        b for b in benchmark_summaries
        if b["tolerance_single_median_ci_lower"] > threshold
    ]
    return {
        "h1_hypothesis": "bootstrap lower bound of cross-model median "
                         "single-occasion tolerance > 0.005 for at least "
                         "3 of 5 benchmarks",
        "threshold": threshold,
        "n_required": n_req,
        "n_exceeding": len(benchmarks_exceeding),
        "benchmarks_exceeding": [b["benchmark_id"] for b in benchmarks_exceeding],
        "h1_confirmed": len(benchmarks_exceeding) >= n_req,
    }
```

### 8.9 `08_ranking_stability.py`

**Purpose.** Compute ranking reversals and pairwise win probabilities.

**Signature:**

```bash
python scripts/08_ranking_stability.py \
  --condition-level runs/normalized/condition_level_primary.csv \
  --analysis-config configs/analysis_config.yaml \
  --out-dir analysis/ranking_stability/
```

**Analyses:**

```python
for benchmark_id in primary_benchmarks:
    model_scores = pivot(condition_level, benchmark_id, index="condition_id", columns="model_id")
    # Kendall tau across condition pairs
    tau = mean([kendall_tau(model_scores.iloc[i], model_scores.iloc[j])
                for i, j in pairs])
    # Pairwise reversal fraction
    for (m_a, m_b) in model_pairs:
        overall_mean_diff = model_scores[m_a].mean() - model_scores[m_b].mean()
        reversal_count = sum(
            sign(model_scores[m_a][c] - model_scores[m_b][c]) != sign(overall_mean_diff)
            for c in conditions
        )
        reversal_fraction = reversal_count / len(conditions)
    # Bootstrap p(A > B)
    boot_probs = []
    for _ in range(n_boot):
        sample = rng.choice(conditions, size=len(conditions), replace=True)
        p_a_gt_b = mean(model_scores[m_a][sample] > model_scores[m_b][sample])
        boot_probs.append(p_a_gt_b)
    p_a_gt_b_mean = mean(boot_probs)
    p_a_gt_b_ci = (percentile(boot_probs, 2.5), percentile(boot_probs, 97.5))
```

**Outputs.**

- `analysis/ranking_stability/ranking_reversals.csv`
- `analysis/ranking_stability/pairwise_win_probabilities.csv`
- `figures/ranking_stability_by_benchmark.png`

### 8.10 `09_mmlu_subject_decomp.py`

**Purpose.** Decompose MMLU panel variance by subject.

**Signature:**

```bash
python scripts/09_mmlu_subject_decomp.py \
  --item-level runs/normalized/item_level_primary.parquet \
  --analysis-config configs/analysis_config.yaml \
  --out-dir analysis/mmlu_subjects/
```

**Outputs.**

- `analysis/mmlu_subjects/mmlu_subject_accuracy_matrix.csv`: rows=models, cols=subjects, values=mean accuracy averaged across conditions.
- `analysis/mmlu_subjects/mmlu_subject_variance_components.csv`: variance decomposition with `model_main`, `subject_main`, `prompt_main`, `model_x_subject`, `prompt_x_subject`, `residual`.
- `figures/mmlu_model_subject_heatmap.png`.

### 8.11 `10_gsm8k_case.py`

**Purpose.** Secondary-tier GSM8K analysis with extraction-variant facet.

**Signature:**

```bash
python scripts/10_gsm8k_case.py \
  --item-level runs/normalized/item_level_gsm8k.parquet \
  --condition-level runs/normalized/condition_level_gsm8k.csv \
  --analysis-config configs/analysis_config.yaml \
  --out-dir analysis/gsm8k_case/
```

**Outputs.**

- `analysis/gsm8k_case/gsm8k_tolerance_schedule.csv`
- `analysis/gsm8k_case/gsm8k_extraction_sensitivity.csv`: accuracy delta between strict and permissive extraction per (model, prompt, seed).
- `analysis/gsm8k_case/gsm8k_parseability.csv`: parse-failure rate per condition.

### 8.12 `98_reproducibility_trace.py` (new in v0.2)

**Purpose.** Generate the reproducibility-trace report.

**Signature:**

```bash
python scripts/98_reproducibility_trace.py \
  --config configs/study_config.yaml \
  --manifests-dir manifests/ \
  --raw-dir runs/raw/ \
  --normalized-dir runs/normalized/ \
  --analysis-dir analysis/ \
  --sample-n 5 \
  --seed 20260424 \
  --out reports/reproducibility_trace.md
```

**Trace content (structured sections):**

1. **Config state.**
   - `config_hash_full`, `config_hash` (from `reports/config_validation.json`).
   - SHA-256 of each of the six YAML files.
2. **Manifest state.**
   - SHA-256 of `conditions_primary.csv`, `conditions_gsm8k.csv`, `prompt_manifest.csv`, `fewshot_manifest.csv`.
   - Row counts of each manifest.
3. **Raw-run coverage.**
   - Count of `condition_status == "complete"` rows vs count of `runs/raw/{run_id}/` directories.
   - Any mismatches logged.
4. **Per-condition recompute sample** (new in v0.2).
   - Randomly sample N=5 conditions (seed = `--seed`).
   - For each, load the raw `item_outputs.jsonl` or `item_scores.jsonl`, recompute `is_correct` per the scoring rule, aggregate to condition-level accuracy.
   - Diff against the normalized parquet's value for the same condition.
   - Report deltas. Deltas > 1e-6 are flagged as reproducibility failures.
5. **Variance-components and tolerance re-derivation.**
   - Re-run `06_variance_components.py` and `07_tolerance_schedule.py` on the existing normalized data.
   - Diff against the stored outputs at byte level (CSV) or at float tolerance (JSON).
6. **Cross-environment check** (optional, but recommended for final paper version).
   - If a second environment's outputs are provided via `--compare-env <dir>`, compute the same diffs against that environment's analysis outputs.
7. **Summary.**
   - Total reproducibility-critical artifacts: N.
   - Reproducibility failures: M.
   - Trace status: `pass | fail`.

**Acceptance criteria.** Trace status must be `pass` before the paper is submitted.

### 8.13 `99_make_report.py`

**Purpose.** Assemble pipeline report with all artifacts referenced.

**Signature:**

```bash
python scripts/99_make_report.py \
  --analysis-dir analysis/ \
  --tables-dir tables/ \
  --figures-dir figures/ \
  --out reports/cpu_pipeline_report.md
```

**Content.** Structured markdown with:
- Pipeline execution summary (config hash, manifest hashes, success/failure counts).
- Link to every analysis output.
- Link to reproducibility trace.
- Link to deviations log.
- Headline numbers: H1 decision, per-benchmark tolerance medians, ranking-stability summary, MMLU subject interaction proportion.

---

## 9. Analysis Definitions

### 9.1 Decimal-place licensing (revised in v0.2)

**Notation choice.** The paper reports tolerances and licensed precision in **decimal-accuracy notation** as primary. Percentage-point equivalents are shown as parentheticals where they aid reader interpretation but are never presented as a separate licensing tier.

**Equivalences (informational, not licensing tiers):**

```
decimal-accuracy 0.001 ≡ percentage-point 0.1 pp
decimal-accuracy 0.010 ≡ percentage-point 1.0 pp
```

**Three-level licensing categorical:**

| Level | Tolerance threshold | Decimal-accuracy example | Percentage equivalent |
|---|---|---|---|
| `three_decimal_accuracy` | tolerance < 0.0005 | 0.742 | 74.2% |
| `two_decimal_accuracy` | 0.0005 ≤ tolerance < 0.005 | 0.74 | 74% |
| `interval_required` | tolerance ≥ 0.005 | report as 0.74 ± 0.02 or wider | — |

**Mapping function:**

```python
def licensed_precision(tolerance: float) -> str:
    if tolerance < 0.0005:
        return "three_decimal_accuracy"
    elif tolerance < 0.005:
        return "two_decimal_accuracy"
    else:
        return "interval_required"
```

No alternative categorical values are permitted. The v0.1 `integer_percent_only` level is retired because it duplicates `two_decimal_accuracy` at different notation.

### 9.2 SEM definitions

From the aggregate-score variance-components decomposition:

```
var_total   = var_prompt + var_seed + var_scoring_rule + var_residual

SEM_single           = sqrt(var_total)
SEM_within_rule      = sqrt(var_prompt + var_seed + var_residual)
SEM_prompt_averaged  = sqrt(var_total / 4)
SEM_full_design      = sqrt(var_total / n_conditions)     # n_conditions = 24 or reduced
```

For benchmarks with `seed_facet == false` (e.g., TruthfulQA-MC), `var_seed` is omitted from `var_total` and the full-design denominator uses `n_conditions = 8`.

### 9.3 Bootstrap confidence intervals (new in v0.2)

**Resampling unit.** Conditions (not items). A bootstrap resample draws N conditions with replacement from the N admissible conditions in a (benchmark, model) cell.

**Procedure.** For each (benchmark, model) cell:

```python
rng = numpy.random.default_rng(analysis_config.bootstrap.random_seed)
conditions = condition_level[
    (condition_level.benchmark_id == benchmark_id) &
    (condition_level.model_id == model_id)
]
n = len(conditions)
boot_sem = []
for _ in range(analysis_config.bootstrap.n_resamples):
    sample = conditions.sample(n=n, replace=True, random_state=rng.integers(2**32))
    boot_sem.append(sample["accuracy"].std(ddof=1))
boot_tolerance = 2 * numpy.array(boot_sem)
ci_lower = numpy.percentile(boot_tolerance, 2.5)
ci_upper = numpy.percentile(boot_tolerance, 97.5)
```

**H1 test.** Uses bootstrap lower bound of cross-model median single-occasion tolerance, not the point estimate. This is a stricter test than the v0.1 version and defends against the primary-hypothesis being driven by noisy SEM estimates.

### 9.4 Ranking reversal fraction

For each (benchmark, model_a, model_b) triple:

```python
overall_sign = sign(mean(scores[model_a]) - mean(scores[model_b]))
reversal_count = sum(
    sign(scores_at_condition[model_a] - scores_at_condition[model_b]) != overall_sign
    for condition in admissible_conditions
)
reversal_fraction = reversal_count / len(admissible_conditions)
```

H3 threshold check: `reversal_fraction > analysis_config.h3_ranking_reversal_threshold` (default 0.10).

---

## 10. Statistical Procedures

### 10.1 Mixed-effects convergence cascade (new in v0.2)

**Pre-registered cascade** (from `analysis_config.mixed_effects_cascade`):

- **Level 1 (preferred).** Random effects: `prompt + seed + scoring_rule + item + model:prompt + model:scoring_rule`. Fixed effects: `model`. Family: binomial, logit link.
- **Level 2 (fallback).** Drop the two crossed interaction terms. Random effects: `prompt + seed + scoring_rule + item`.
- **Level 3 (fallback).** Drop the smallest-power random effect (`seed`, which has only 3 levels). Random effects: `prompt + scoring_rule + item`.
- **Level 4 (last-resort).** Mixed-effects fit abandoned. Use aggregate-score G-theory decomposition from condition-level data as the primary variance-components estimate.

**Cascade trigger conditions (any triggers descent):**

- Optimizer reports singular fit warning.
- Estimated covariance matrix Hessian is not positive definite.
- Optimizer hits iteration limit without declaring convergence.
- Gradient norm at reported optimum exceeds `analysis_config.convergence_triggers.gradient_norm_above` (default 1e-3).

**Per-benchmark cascade logging.** Every attempted level is recorded in `analysis/variance_components/model_convergence_report.csv`:

```
benchmark_id, cascade_level, success, error_type, gradient_norm,
optimizer_iterations, n_items, fit_seconds
```

A benchmark falling to Level 4 produces a `deviations.md` entry and is flagged in the report.

**No post-hoc alternative fits.** The cascade is fully specified here; any deviation during implementation (e.g., trying a different optimizer, adding a random-slope term) is a pre-registration deviation and requires a `deviations.md` entry before the fit is used in any reported analysis.

### 10.2 Aggregate-score G-theory decomposition

Used as companion to Level 1–3 mixed-effects fits and as primary output for Level 4 fallback.

Decomposition: for each (benchmark, model) cell, treat the 24 condition-level accuracies as outcomes of a crossed random-effects design over prompt × seed × scoring_rule. Fit a random-effects ANOVA model:

```
accuracy = μ + α_prompt + α_seed + α_scoring_rule + ε
```

Variance estimates from the MS decomposition (`EMS` formulas for fully crossed design) yield:

```
var_prompt, var_seed, var_scoring_rule, var_residual
```

These feed directly into the SEM definitions in §9.2.

### 10.3 Leave-one-prompt-out sensitivity

For each benchmark, re-run the tolerance-schedule derivation four times, each time excluding one of the four prompt templates. Report the four resulting tolerance values as a sensitivity spread in `analysis/tolerance_schedules/tolerance_sensitivity.csv`.

**Acceptance criterion.** Sensitivity analysis is always reported; no prompt is retroactively excluded from the primary analysis on the basis of this check.

---

## 11. Error Semantics

### 11.1 Exit codes

All scripts use the following exit code convention:

| Code | Meaning |
|---|---|
| 0 | Success. |
| 1 | Generic failure (implementation error; bug). |
| 2 | Configuration validation failure. |
| 3 | Data contract violation (few-shot leakage, schema mismatch, hash mismatch). |
| 4 | Convergence failure at final cascade level (for `06_variance_components.py`). |
| 5 | Reproducibility trace failure (`98_reproducibility_trace.py`). |

### 11.2 Error types

Named exception classes for programmatic error handling:

```python
class ConfigValidationError(Exception): pass
class FewshotLeakageError(Exception): pass
class SchemaValidationError(Exception): pass
class ConfigHashMismatchError(Exception): pass
class ConvergenceFailureAllLevels(Exception): pass
class ReproducibilityTraceFailure(Exception): pass
class PromptProvenanceMissing(Exception): pass
class ConditionCountMismatch(Exception): pass
class PartialRunCompletion(Exception): pass
```

### 11.3 Logging

All scripts must log to stderr with structured format:

```
[YYYY-MM-DDTHH:MM:SSZ] [LEVEL] [script_name] message
```

Levels: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`.

Every error exit must be preceded by at least one `ERROR` or `CRITICAL` log line.

---

## 12. Acceptance Criteria

### 12.1 Pre-run acceptance

- [ ] `00_validate_config.py` exits 0.
- [ ] `config_validation.json` exists with `validation_status == "pass"`.
- [ ] `01_build_manifests.py` exits 0. Condition manifests exist and match pre-registered counts.
- [ ] `02_draw_fewshot_examples.py` exits 0. Few-shot leakage check passes.
- [ ] `03_validate_prompts.py` exits 0. All prompt content hashes match.
- [ ] `preregistration/appendices/fewshot_draws_LOCKED.json` exists.
- [ ] `preregistration/appendices/prompts_LOCKED.md` exists.
- [ ] `preregistration/prereg_v1_1_LOCKED.md` exists.

### 12.2 Post-run acceptance

- [ ] For every `condition_status == "complete"` row in the manifest, `runs/raw/{run_id}/run_metadata.json` exists.
- [ ] `04_normalize_outputs.py` exits 0.
- [ ] Item counts in normalized parquet match `expected_num_items` for every completed condition.
- [ ] Zero silent row drops (audit: sum of normalized rows per condition equals `num_items_completed` in raw metadata).
- [ ] `05_make_condition_scores.py` exits 0.

### 12.3 Analysis acceptance

- [ ] `06_variance_components.py` exits 0 or 4 (4 → Level 4 fallback used; deviations logged).
- [ ] `07_tolerance_schedule.py` exits 0. Bootstrap CIs present for every cell.
- [ ] `08_ranking_stability.py` exits 0.
- [ ] `09_mmlu_subject_decomp.py` exits 0.
- [ ] `10_gsm8k_case.py` exits 0 (if Tier 2 enabled).
- [ ] `98_reproducibility_trace.py` exits 0. Trace status `pass`.
- [ ] `99_make_report.py` exits 0.

### 12.4 Manuscript-readiness acceptance

- [ ] `tables/table_4_tolerance_schedule.md` contains per-benchmark cross-model median tolerance with bootstrap CIs.
- [ ] `h1_test.json` records H1 outcome.
- [ ] `reports/deviations.md` exists (possibly empty; empty is logged as "no deviations" line).
- [ ] `reports/reproducibility_trace.md` status `pass`.

---

## 13. Reproducibility Trace (new in v0.2)

### 13.1 Structure

`reports/reproducibility_trace.md` emitted by `scripts/98_reproducibility_trace.py`. Sections:

1. **Config state.** Full hash, short hash, per-file hashes, timestamp.
2. **Manifest state.** Per-manifest SHA-256, row counts.
3. **Raw-run coverage.** Completed-condition count vs raw-run directory count; mismatches listed.
4. **Per-condition recompute sample.** N=5 random conditions (seed-fixed); raw → recomputed-accuracy → normalized-accuracy diff; deltas > 1e-6 flagged.
5. **Analysis re-derivation.** Variance-components and tolerance-schedule recomputation from normalized data; diff against stored outputs.
6. **Cross-environment check.** Optional; activated via `--compare-env <dir>`; used for final paper version to confirm environment-independence.
7. **Summary.** Total artifacts checked, failures, status.

### 13.2 Acceptance

Reproducibility trace status must be `pass` before the paper is submitted. A `fail` status is a hard block; the failure must be remediated or documented as a deviation with a rationale.

---

## 14. Non-Negotiables

1. No prompt changes after pre-registration lock.
2. No benchmark substitution after lock without a `deviations.md` entry.
3. No silent dropping of failed or unparseable generations. Parseability is reported as an auxiliary measurement property.
4. No mixing decimal-accuracy and percentage-point notation without explicit conversion.
5. No post-hoc redefinition of tolerance thresholds.
6. No claiming universal benchmark tolerances from this study.
7. No treating LL and generate-and-parse as "the same score" without reporting within-rule and across-rule tolerances separately.
8. No reporting three-decimal accuracy without checking the tolerance schedule.
9. No post-hoc mixed-effects model modifications outside the pre-registered cascade (new in v0.2).
10. No removal of prompts from primary analysis on the basis of leave-one-prompt-out sensitivity (new in v0.2).
11. No silent completion of reproducibility-trace failures (new in v0.2).

---

## 15. Deliverables for Manuscript

### 15.1 Tables

```
tables/table_1_measurement_universe.md
tables/table_2_variance_components.md
tables/table_3_generalizability_coefficients.md
tables/table_4_tolerance_schedule.md            -- headline artifact
tables/table_5_ranking_stability.md
tables/table_6_mmlu_subject_decomp.md
```

### 15.2 Figures

**Pipeline-generated (produced by scripts):**

```
figures/fig_2_tolerance_by_benchmark.png
figures/fig_3_single_vs_averaged_tolerance.png
figures/fig_4_ranking_reversal_heatmap.png
figures/fig_5_mmlu_subject_interaction.png
figures/fig_6_gsm8k_parseability.png           -- only if Tier 2 run
```

**Author-authored (not pipeline-generated):**

```
figures/fig_1_construct_hierarchy.png           -- conceptual diagram
figures/fig_7_decision_rule_schematic.png       -- optional; decision-rule visualization
```

Author-authored figures are out of scope for the CPU pipeline.

### 15.3 Reports

```
reports/config_validation.json
reports/cpu_pipeline_report.md
reports/reproducibility_trace.md
reports/deviations.md
```

---

## 16. Task Inventory for Coding Agents (new in v0.2)

The pipeline can be implemented as a sequence of focused tasks with a defined dependency graph. Tasks are ordered by dependency; no task should be attempted before its predecessors are complete.

### 16.1 Pre-inference tasks

**Task P-01: Config schema and loader.**
- Implement `gradience_study.config` module with dataclasses for every config section.
- Implement `load_config(study_config_path)` → merged config object.
- Implement canonical YAML serialization per §4.4.
- Implement `compute_config_hash(merged_config)` → (full, short) tuple.
- Unit tests: `tests/test_config_hash.py` with fixture configs producing known hashes.

**Task P-02: Config validation script (`00_validate_config.py`).**
- Depends on P-01.
- Implement validation rules per §4.5.
- Emit `config_validation.json` per §8.1.
- Exit codes per §11.1.

**Task P-03: Manifest builder (`01_build_manifests.py`).**
- Depends on P-01.
- Enumerate conditions deterministically from configs.
- Emit all manifest CSVs.
- Unit tests: `tests/test_condition_manifest.py` verifying deterministic ordering and correct nominal counts.

**Task P-04: Few-shot drawer (`02_draw_fewshot_examples.py`).**
- Depends on P-01, P-03.
- Implement draw algorithm per §8.3.
- Implement leakage check per §8.3 (hard-fail on any intersection).
- Emit lock file and manifest.
- Unit tests: `tests/test_fewshot_leakage.py` with fixtures where leakage is synthetically introduced; check hard-fail behavior.

**Task P-05: Prompt validator (`03_validate_prompts.py`).**
- Depends on P-01.
- Implement placeholder-presence check, content-hash verification, admissibility-status gate.
- Emit prompt manifest.

### 16.2 GPU-side contract (out of CPU-spec scope but listed for completeness)

**Task G-01: Inference harness.**
- Read conditions manifest; for each pending condition, construct prompt (template + fewshot + item), execute model, emit outputs per §5.2–5.4.
- Must be contract-conformant but can be implemented by any backend.

### 16.3 Post-inference tasks

**Task N-01: Output normalizer (`04_normalize_outputs.py`).**
- Depends on Task G-01 outputs.
- Implement per §8.5.
- JSON Schema validation via `jsonschema` library.
- Unit tests: `tests/test_output_schema.py`, `tests/test_normalization.py` with fixtures containing valid and invalid raw runs.

**Task N-02: Condition-score aggregator (`05_make_condition_scores.py`).**
- Depends on N-01.
- Implement per §8.6.

### 16.4 Analysis tasks

**Task A-01: Variance-components script (`06_variance_components.py`).**
- Depends on N-02.
- Implement mixed-effects cascade per §10.1 with all four levels.
- Implement aggregate-score G-theory per §10.2.
- Emit outputs per §8.7.
- Unit tests: `tests/test_mixed_effects_cascade.py` with synthetic data triggering each cascade level.

**Task A-02: Tolerance-schedule script (`07_tolerance_schedule.py`).**
- Depends on A-01.
- Implement bootstrap CIs per §9.3.
- Implement decimal-place licensing per §9.1.
- Implement H1 test per §8.8.
- Emit outputs per §8.8.
- Unit tests: `tests/test_tolerance_math.py`, `tests/test_bootstrap.py` with known-answer fixtures.

**Task A-03: Ranking-stability script (`08_ranking_stability.py`).**
- Depends on N-02.
- Implement per §8.9 and §9.4.

**Task A-04: MMLU subject decomposition (`09_mmlu_subject_decomp.py`).**
- Depends on N-01 (item-level), N-02 (condition-level).
- Implement per §8.10.

**Task A-05: GSM8K case (`10_gsm8k_case.py`).**
- Depends on N-01, N-02.
- Implement per §8.11.

### 16.5 Reporting tasks

**Task R-01: Reproducibility trace (`98_reproducibility_trace.py`).**
- Depends on all A-* tasks.
- Implement per §13.
- Hard-fails on any reproducibility mismatch.

**Task R-02: Pipeline report (`99_make_report.py`).**
- Depends on all A-* and R-01.
- Implement per §8.13.

### 16.6 Testing tasks

**Task T-01: Fixture generation (`tests/fixtures/make_fixtures.py`).**
- Generate all fixture files programmatically and deterministically.
- Must be idempotent: rerunning produces byte-identical fixtures.

**Task T-02: Full test suite.**
- Every test file in `tests/` passes.
- Coverage target: ≥ 80% for all scripts; 100% for hash computation, leakage check, cascade trigger, decimal-place licensing.

### 16.7 Dependency graph summary

```
P-01 → P-02
P-01 → P-03 → P-04
P-01 → P-05
T-01 (independent; used by tests)

[GPU-side: G-01] 
    ↓
N-01 → N-02 → A-01 → A-02
                 ↓
                A-03
                 ↓
                A-04
(N-02)           ↓
                A-05 (if Tier 2)
                 ↓
                R-01 → R-02
```

Critical path: P-01 → P-02 → P-03 → P-04 → P-05 → [G-01] → N-01 → N-02 → A-01 → A-02 → R-01 → R-02.

---

## 17. Library Versions (pinned)

```
python         == 3.11.x        (3.11.6 or later)
pyyaml         >= 6.0, < 7
pandas         >= 2.1, < 3
pyarrow        >= 14, < 17
numpy          >= 1.26, < 2
scipy          >= 1.11, < 2
statsmodels    >= 0.14, < 0.16
jsonschema     >= 4.20, < 5
pytest         >= 7.4, < 9
pytest-cov     >= 4.1, < 6
datasets       >= 2.14, < 3   # Hugging Face datasets
matplotlib     >= 3.8, < 4
seaborn        >= 0.13, < 0.14
```

Pin file: `requirements.txt` at repository root. Exact minor versions are locked at v1.1 pre-registration time.

---

## 18. Minimal Command Sequence

Full primary pipeline:

```bash
cd papers/benchmark_reliability_study

# Pre-inference (CPU-side preparation)
python scripts/00_validate_config.py \
  --config configs/study_config.yaml \
  --out reports/config_validation.json

python scripts/01_build_manifests.py \
  --config configs/study_config.yaml \
  --out-dir manifests/

python scripts/02_draw_fewshot_examples.py \
  --config configs/study_config.yaml \
  --seeds 42 123 2024 \
  --out manifests/fewshot_manifest.csv \
  --lock-out preregistration/appendices/fewshot_draws_LOCKED.json

python scripts/03_validate_prompts.py \
  --prompts-config configs/prompts.yaml \
  --out manifests/prompt_manifest.csv

# [GPU-side inference executes here; produces runs/raw/{run_id}/*]

# Post-inference (CPU-side analysis)
python scripts/04_normalize_outputs.py \
  --conditions manifests/conditions_primary.csv \
  --raw-dir runs/raw/ \
  --schemas-dir schemas/ \
  --out runs/normalized/item_level_primary.parquet

python scripts/05_make_condition_scores.py \
  --item-level runs/normalized/item_level_primary.parquet \
  --out runs/normalized/condition_level_primary.csv

python scripts/06_variance_components.py \
  --item-level runs/normalized/item_level_primary.parquet \
  --condition-level runs/normalized/condition_level_primary.csv \
  --config configs/analysis_config.yaml \
  --out-dir analysis/variance_components/

python scripts/07_tolerance_schedule.py \
  --condition-level runs/normalized/condition_level_primary.csv \
  --variance-components analysis/variance_components/aggregate_vc.csv \
  --analysis-config configs/analysis_config.yaml \
  --out-dir analysis/tolerance_schedules/

python scripts/08_ranking_stability.py \
  --condition-level runs/normalized/condition_level_primary.csv \
  --analysis-config configs/analysis_config.yaml \
  --out-dir analysis/ranking_stability/

python scripts/09_mmlu_subject_decomp.py \
  --item-level runs/normalized/item_level_primary.parquet \
  --analysis-config configs/analysis_config.yaml \
  --out-dir analysis/mmlu_subjects/

python scripts/98_reproducibility_trace.py \
  --config configs/study_config.yaml \
  --manifests-dir manifests/ \
  --raw-dir runs/raw/ \
  --normalized-dir runs/normalized/ \
  --analysis-dir analysis/ \
  --sample-n 5 \
  --seed 20260424 \
  --out reports/reproducibility_trace.md

python scripts/99_make_report.py \
  --analysis-dir analysis/ \
  --tables-dir tables/ \
  --figures-dir figures/ \
  --out reports/cpu_pipeline_report.md
```

GSM8K secondary (after N-01/N-02 for GSM8K outputs):

```bash
python scripts/04_normalize_outputs.py \
  --conditions manifests/conditions_gsm8k.csv \
  --raw-dir runs/raw/ \
  --schemas-dir schemas/ \
  --out runs/normalized/item_level_gsm8k.parquet

python scripts/05_make_condition_scores.py \
  --item-level runs/normalized/item_level_gsm8k.parquet \
  --out runs/normalized/condition_level_gsm8k.csv

python scripts/10_gsm8k_case.py \
  --item-level runs/normalized/item_level_gsm8k.parquet \
  --condition-level runs/normalized/condition_level_gsm8k.csv \
  --analysis-config configs/analysis_config.yaml \
  --out-dir analysis/gsm8k_case/
```

---

## 19. Success Condition

The CPU-side pipeline succeeds if, after GPU-side inference outputs are present, one command sequence regenerates:

- Normalized item-level data.
- Condition-level accuracies.
- Variance-component tables (with cascade-level reporting).
- Tolerance schedules with bootstrap CIs.
- Decimal-place licensing decisions.
- Ranking-stability results.
- MMLU subject decomposition.
- GSM8K secondary-case outputs.
- Reproducibility trace with `pass` status.
- Manuscript-ready tables and figures.

The pipeline makes the study's central claim auditable:

> Benchmark accuracy is not a single number emitted by a model. It is an observed score produced by a declared measurement design, and the precision of that score must be estimated rather than assumed.

A reviewer who runs this pipeline against the committed raw-run directory and committed configs should reproduce every headline number in the manuscript bit-identically (or within documented float-tolerance for stochastic bootstrap operations, with the bootstrap seed pinned).

---

*End of spec v0.2. No implementation work should begin until v1.1 pre-registration is locked and the open-question resolution (§3.10 of `preregistration_v1.md`) is complete.*
