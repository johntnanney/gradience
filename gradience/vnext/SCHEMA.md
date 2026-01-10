# Gradience vNext Telemetry Contract (v1)

This document freezes the **minimal, stable spine** of the Gradience vNext
telemetry schema.

Gradience is intended to function as a **flight recorder + mechanic**: the
telemetry must be stable enough that `check`, `monitor`, and `audit` can
interoperate and runs can be compared over time.

If you need to make a **breaking change** (change the meaning of existing
fields, change required fields, rename stable event types, etc.), you must bump
the schema version to `gradience.vnext.telemetry/v2`.

## Schema identifier

Current schema id:

```
gradience.vnext.telemetry/v1
```

## Record format

Telemetry is stored as **JSON Lines (JSONL)**: one JSON object per line.

Every record **must** include the following envelope fields:

| Field | Type | Required | Notes |
|---|---|---:|---|
| `schema` | string | ✅ | Must equal `gradience.vnext.telemetry/v1` |
| `ts` | number | ✅ | Unix timestamp in seconds (float allowed) |
| `run_id` | string | ✅ | Identifier for the run (UUID recommended) |
| `event` | string | ✅ | Event type (see below) |
| `step` | integer | null | ✅ | Training step if applicable; may be `null` |

All other keys are considered the **event payload**.

### Forward compatibility rules

Allowed:
- Extra keys on any event
- New `metrics(kind=...)` blocks
- New optional fields

Not allowed within `/v1`:
- Renaming or changing meaning of existing fields
- Changing required fields
- Reusing an event name with different semantics

## Stable event types

The following event names are **stable within `/v1`**:

- `run_start`
- `train_step`
- `eval`
- `metrics`
- `alert`
- `recommendation`
- `run_end`

### `run_start`

Required payload fields:
- `config`: object (a `ConfigSnapshot` dictionary)

Optional payload fields:
- `meta`: object (environment info, git hash, hostname, etc.)

### `train_step`

No additional payload fields are strictly required.

Recommended payload fields:
- `loss`: number
- `lr`: number or list of numbers (per param-group)

### `eval`

Required payload fields:
- `split`: string
- `metrics`: object

Split name conventions:
- `train`, `test` are canonical
- `val` is canonical but optional
- other values are allowed, but monitors may ignore them unless configured

Metric key conventions (minimal frozen set):

| Metric key | Type | Meaning |
|---|---|---|
| `ppl` | number | perplexity (lower is better) |
| `accuracy` | number | accuracy as a fraction (0..1) |
| `loss` | number | evaluation loss |
| `n` | integer | number of examples |

Notes:
- Not all metrics must be present for every eval.
- `gradience monitor` uses `ppl` on `train` and `test` to compute the generalization gap.

### `metrics`

Required payload fields:
- `kind`: string (e.g., `lora_audit`, `spectral`, `structural`, `dominance_act`)
- `metrics`: object

Notes:
- `kind` is used to namespace metric payloads so new metric blocks can be added without breaking consumers.

### `alert`

Required payload fields:
- `severity`: string (`info` | `warning` | `error` | `critical`)
- `code`: string (stable identifier, e.g. `memorization_gap`)
- `message`: string (human readable)

Optional payload fields:
- `context`: object

### `recommendation`

Required payload fields:
- `recommendations`: array of recommendation objects

### `run_end`

Required payload fields:
- `status`: string (`ok` | `aborted` | `error`)

Optional payload fields:
- `reason`: string
