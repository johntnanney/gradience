# Adapter QA Artifact

## 1. What It Is

A Gradience adapter QA artifact is the canonical record of an adapter's structural health, behavioral status, and eligibility judgment. It is a decision-bearing object: downstream commands consume it and change behavior based on its contents.

Schema identifier: `gradience.adapter_qa/v1`

## 2. How to Produce It

### CLI

```bash
gradience audit-adapter \
  --peft-dir ./adapters/my-adapter \
  --base-model meta-llama/Llama-2-7b-hf \
  --adapter-score 0.78 \
  --base-score 0.72 \
  --metric-name accuracy \
  --eval-dataset hellaswag \
  --output qa.json
```

Behavioral arguments (`--adapter-score`, `--base-score`, etc.) are optional. Without them the artifact will have `"status": "unknown_no_behavioral_eval"`.

### Python API

```python
from gradience.api import audit_adapter

artifact = audit_adapter(
    peft_dir="./adapters/my-adapter",
    base_model="meta-llama/Llama-2-7b-hf",
    adapter_score=0.78,
    base_score=0.72,
    metric_name="accuracy",
    eval_dataset="hellaswag",
)

# Serialize
import json
with open("qa.json", "w") as f:
    json.dump(artifact.to_dict(), f, indent=2)
```

Both paths use the same builder (`build_qa_artifact`) internally.

## 3. How to Read It

A v1 artifact has five top-level sections:

- **`adapter`** -- identity: name, path, base model, nominal rank, layer count.
- **`structural_summary`** -- what the spectral audit observed: utilization, stable rank, effective rank, rank waste, and warning flags.
- **`behavioral_summary`** -- evaluation evidence (if provided): scores, metric, dataset, whether the adapter beats the base model. All fields are `null` when no eval is available.
- **`eligibility`** -- the policy judgment derived from the evidence: status, confidence level, and human-readable reasons.
- **`notes`** -- optional list of caveats or annotations that are neither flags nor reasons.

The `behavioral_summary` is evidence. The `eligibility` is the decision based on that evidence.

## 4. How to Consume It

The canonical consumer is `merge-audit`:

```bash
gradience merge-audit \
  --source-a ./adapters/adapter-a \
  --source-b ./adapters/adapter-b \
  --source-a-qa qa_a.json \
  --source-b-qa qa_b.json
```

When a QA artifact is provided:

- Eligibility status is surfaced in the merge report.
- `--strict-qa` mode blocks the merge recommendation when either adapter is `flagged_weak` or `unknown_no_behavioral_eval`.
- Warnings are generated for adapters with non-eligible status.

### `--strict-qa` behavior

| Status | Behavior |
|--------|----------|
| `eligible` | Allow |
| `uncertain` | Allow with warning |
| `flagged_weak` | Block |
| `unknown_no_behavioral_eval` | Block |

## 5. Schema Contract

### Required fields

| Path | Type | Notes |
|------|------|-------|
| `schema` | `str` | Must be `"gradience.adapter_qa/v1"` |
| `adapter.name` | `str` | Adapter directory name |
| `adapter.path` | `str` | Path to adapter directory |
| `adapter.rank_nominal` | `int` | Nominal LoRA rank |
| `structural_summary.utilization_mean` | `float` | Mean utilization across layers |
| `structural_summary.rank_waste_ratio` | `float` | `1 - utilization_mean` |
| `behavioral_summary.eval_available` | `bool` | Whether behavioral scores are present |
| `eligibility.status` | `str` | One of the four status values |

### Optional fields

| Path | Type | Default | Notes |
|------|------|---------|-------|
| `adapter.base_model` | `str` | `""` | Base model identifier |
| `adapter.n_layers` | `int` | `0` | Number of adapter layers |
| `structural_summary.utilization_median` | `float` | `0.0` | |
| `structural_summary.stable_rank_mean` | `float` | `0.0` | |
| `structural_summary.effective_rank_90_median` | `float\|null` | `null` | Median effective rank at 90% energy |
| `structural_summary.flags` | `list[str]` | `[]` | Structural warning flags |
| `behavioral_summary.eval_dataset` | `str\|null` | `null` | |
| `behavioral_summary.metric_name` | `str\|null` | `null` | |
| `behavioral_summary.adapter_score` | `float\|null` | `null` | |
| `behavioral_summary.base_score` | `float\|null` | `null` | |
| `behavioral_summary.lower_is_better` | `bool\|null` | `null` | |
| `behavioral_summary.beats_base` | `bool\|null` | `null` | |
| `eligibility.confidence` | `str` | `"low"` | One of: `high`, `medium`, `low` |
| `eligibility.reasons` | `list[str]` | `[]` | Human-readable justifications |
| `notes` | `list[str]` | `[]` | Caveats or annotations |

Extra keys at any level are silently ignored (forward compatible).

### Validation rules

- Missing or wrong `schema` raises `QASchemaError`.
- Unknown `eligibility.status` values raise `QASchemaError`.
- `list[str]` fields must contain only strings if present.
- Numeric fields accept `int` or `float`, normalized to `float` internally.

## 6. Decision Semantics

### Eligibility status

| Value | Meaning |
|-------|---------|
| `eligible` | Adapter outperforms base on provided eval |
| `uncertain` | Performance within margin of base |
| `flagged_weak` | Adapter underperforms base on provided eval |
| `unknown_no_behavioral_eval` | No behavioral evaluation provided |

Status is a **policy judgment**, not a raw measurement. The margin (`--margin` CLI flag, default `0.0`) controls the uncertainty band:

- `delta > margin` -- eligible
- `-margin <= delta <= margin` -- uncertain
- `delta < -margin` -- flagged_weak

Where `delta` is always oriented so positive means "adapter is better."

### Confidence

| Level | When |
|-------|------|
| `high` | Behavioral evidence present AND delta > 2x margin |
| `medium` | Behavioral evidence present, status is `uncertain` or delta is small |
| `low` | No behavioral evidence (hard rule: never `high` without behavioral data) |

## 7. Versioning Policy

The schema identifier `gradience.adapter_qa/v1` is frozen.

- New fields may be added without a version bump.
- No existing field will be renamed, removed, or have its type changed.
- A future version that changes the contract must use a new schema identifier (e.g., `gradience.adapter_qa/v2`).
