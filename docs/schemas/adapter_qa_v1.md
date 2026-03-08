# Adapter QA Artifact — Schema v1

Schema identifier: `gradience.adapter_qa/v1`

Produced by `gradience audit-adapter`.  Consumed by `gradience merge-audit` via `--source-a-qa` / `--source-b-qa`.

## Top-level structure

| Field | Type | Description |
|-------|------|-------------|
| `schema` | string | Always `"gradience.adapter_qa/v1"` |
| `adapter` | object | Adapter identity and configuration |
| `structural_summary` | object | Spectral metrics from the audit |
| `behavioral_summary` | object | Evaluation metrics (optional data) |
| `eligibility` | object | Judgment: status, confidence, reasons |

## `adapter`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | yes | Directory name of the adapter |
| `path` | string | yes | Path to the adapter directory |
| `base_model` | string | yes | Base model identifier (may be empty) |
| `rank_nominal` | int | yes | LoRA rank from adapter config |
| `n_layers` | int | yes | Number of LoRA layer pairs |

## `structural_summary`

Only includes metrics that are used for judgment or flags.

| Field | Type | Description |
|-------|------|-------------|
| `utilization_mean` | float | Mean `stable_rank / r` across layers |
| `utilization_median` | float | Median utilization |
| `stable_rank_mean` | float | Mean stable rank |
| `energy_rank_90_p50` | float or null | Median rank needed to capture 90% energy |
| `rank_waste_ratio` | float | `1 - utilization_mean` |
| `flags` | list[string] | Structural warning flags (see below) |

### Structural flags

| Flag | Condition |
|------|-----------|
| `low_utilization` | `utilization_mean < 0.25` |
| `high_rank_waste` | `rank_waste_ratio > 0.75` |
| `concentrated_spectrum` | `energy_rank_90_p50 <= 2.0` and `rank_nominal >= 8` |
| `underutilized_capacity` | `stable_rank_mean < rank_nominal * 0.2` |

These are warning signals, not proof of low quality.

## `behavioral_summary`

All score fields are `null` when no evaluation is available.

| Field | Type | Description |
|-------|------|-------------|
| `eval_available` | bool | Whether behavioral evaluation was provided |
| `eval_dataset` | string or null | Dataset used for evaluation |
| `metric_name` | string | Metric name (e.g. `"perplexity"`) |
| `adapter_score` | float or null | Adapter's score |
| `base_score` | float or null | Base model's score |
| `lower_is_better` | bool | True for perplexity-like metrics |
| `beats_base` | bool or null | Whether adapter outperforms base |

## `eligibility`

| Field | Type | Description |
|-------|------|-------------|
| `status` | string | One of the status values below |
| `confidence` | string | `"low"`, `"medium"`, or `"high"` |
| `reasons` | list[string] | Human-readable reasons for the judgment |

### Status values

| Value | Meaning |
|-------|---------|
| `eligible` | Adapter beats base on behavioral eval |
| `uncertain` | Evidence is inconclusive (within margin) |
| `flagged_weak` | Adapter is worse than base model |
| `unknown` | No behavioral evaluation available |

### Confidence values

| Value | When assigned |
|-------|---------------|
| `high` | Behavioral eval with clear result. Never assigned without behavioral evidence. |
| `medium` | Behavioral eval with marginal result, or clear result near margin |
| `low` | No behavioral eval available |
