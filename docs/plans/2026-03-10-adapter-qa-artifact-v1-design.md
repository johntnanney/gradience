# Adapter QA Artifact v1 — Design Document

**Date:** 2026-03-10
**Status:** Approved
**Goal:** Freeze the single-adapter QA artifact as a stable, decision-bearing product object.

## What It Is

A Gradience adapter QA artifact is the canonical record of an adapter's structural health, behavioral status, and current eligibility judgment.

It is not a bag of metrics. It is a decision-bearing object: downstream commands trust it, consume it, and change behavior based on it.

## Design Approach

Freeze the existing `AdapterQAArtifact` in place. The schema structure is already sound. The work is adding the contract layer: validation, public API promotion, examples, documentation, and explicit downstream consequences.

No field renaming. No breaking changes. Additive only.

## Section 1: Frozen v1 Schema

### Shape

```json
{
  "schema": "gradience.adapter_qa/v1",
  "adapter": {
    "name": "catsubcat-r16",
    "path": "./adapters/catsubcat-r16",
    "base_model": "meta-llama/Llama-2-7b-hf",
    "rank_nominal": 16,
    "n_layers": 32
  },
  "structural_summary": {
    "utilization_mean": 0.08,
    "utilization_median": 0.05,
    "stable_rank_mean": 1.9,
    "effective_rank_90_median": 2.3,
    "rank_waste_ratio": 0.92,
    "flags": ["low_utilization", "high_rank_waste"]
  },
  "behavioral_summary": {
    "eval_available": false,
    "eval_dataset": null,
    "metric_name": null,
    "adapter_score": null,
    "base_score": null,
    "lower_is_better": null,
    "beats_base": null
  },
  "eligibility": {
    "status": "unknown_no_behavioral_eval",
    "confidence": "low",
    "reasons": ["no behavioral evaluation available"]
  },
  "notes": []
}
```

### Changes From Current Code

| Change | Rationale |
|--------|-----------|
| `energy_rank_90_p50` renamed to `effective_rank_90_median` | Reads naturally: "median effective rank under 90% energy criterion" |
| `"unknown"` renamed to `"unknown_no_behavioral_eval"` | Precise about why eligibility is unknown |
| `metric_name` default `""` changed to `null` | Absent values use `null`, not empty string |
| `lower_is_better` becomes nullable | `null` when no eval exists; only asserted when a metric is present |
| `notes` field added | Top-level `list[str]` for caveats that aren't reasons or flags |

### Eligibility Status Values (Frozen)

| Value | Meaning |
|-------|---------|
| `"eligible"` | Adapter outperforms base model on the provided eval |
| `"uncertain"` | Adapter is not clearly better or worse than base under the current decision threshold |
| `"flagged_weak"` | Adapter underperforms base on the provided eval |
| `"unknown_no_behavioral_eval"` | No behavioral evaluation was provided, so eligibility cannot be established from performance evidence |

`eligibility.status` is a **policy judgment**, not a raw measurement. `behavioral_summary` is evidence; `eligibility` is the decision layer based on that evidence.

### Margin Semantics

The margin is a caller-provided absolute tolerance on the delta between adapter and base scores. Exposed as `--margin` CLI flag (default `0.0`). Symmetric:

- `delta > margin` -> `eligible`
- `-margin <= delta <= margin` -> `uncertain`
- `delta < -margin` -> `flagged_weak`

Where `delta = base_score - adapter_score` when `lower_is_better=True`, and `delta = adapter_score - base_score` otherwise. Positive delta always means "adapter is better."

The margin is not stored in the artifact -- it is a decision parameter, not an observation. The reasons list should describe the margin when non-zero.

### Confidence Semantics (Frozen)

- **high** -- only when behavioral evidence is present AND the delta is large (> 2x margin)
- **medium** -- behavioral evidence exists but status is `uncertain`, or delta is small relative to margin
- **low** -- no behavioral evidence exists (hard rule: never `high` without behavioral data)

### Versioning Policy

Additive only. New fields may be added to the v1 schema without a version bump. No existing field will be renamed, removed, or have its semantics changed. A future version that changes the contract must use a new schema identifier.

Room is left for an optional `provenance` block in future (`generated_by`, `timestamp`, `tool_version`).

## Section 2: Validation, Error Handling, and Public API

### Validation in `from_dict()`

`from_dict()` is the single canonical gatekeeper. One path, no drift.

**Rules:**

1. `schema` key required, must equal `"gradience.adapter_qa/v1"`. Missing or wrong raises `QASchemaError`.

2. Required sections (`adapter`, `structural_summary`, `behavioral_summary`, `eligibility`) must be present and must be dicts.

3. Required fields with type enforcement:
   - `adapter.name` (str), `adapter.path` (str), `adapter.rank_nominal` (int or numeric, normalized to int)
   - `structural_summary.utilization_mean` (numeric, normalized to float), `structural_summary.rank_waste_ratio` (numeric, normalized to float)
   - `behavioral_summary.eval_available` (bool)
   - `eligibility.status` (str, must be one of the four known values)
   - `eligibility.confidence` if present: must be one of `"high"`, `"medium"`, `"low"`
   - `structural_summary.flags` if present: must be `list[str]`
   - `eligibility.reasons` if present: must be `list[str]`
   - `notes` if present: must be `list[str]`

4. Unknown status values raise `QASchemaError`. A `/v1` producer that adds new statuses should not still call itself `/v1`.

5. Extra keys are silently ignored (forward compatible on additive fields).

6. `notes` on load: if missing, backfill to `[]`. Required in produced output, tolerant on input.

7. `eligibility.reasons`: required in produced output (the object must always explain itself), tolerant on input (backfill to `[]`).

8. Numeric fields (`utilization_mean`, `rank_waste_ratio`, `adapter_score`, `base_score`, etc.): accept int or float, normalize to float internally.

### Loader Boundary (`_load_source_qa`)

Three-way routing, no silent shape-shifting:

- `schema` key present and correct: strict `AdapterQAArtifact.from_dict()`
- `schema` key absent: legacy flat-format parser (`AdapterQAResult.from_dict()`)
- `schema` key present but wrong: `QASchemaError`, do not fall back to legacy

### New Exception

`QASchemaError` in `gradience/exceptions.py`, subclass of `GradienceError`.

### Public API

**`gradience/__init__.py` exports:**
- `AdapterQAArtifact`
- `EligibilityStatus`

**`gradience/api.py` adds:**
- `audit_adapter(peft_dir, *, base_model=None, adapter_score=None, base_score=None, metric_name=None, lower_is_better=True, eval_dataset=None, margin=0.0) -> AdapterQAArtifact`

`audit_adapter()` is the preferred stable Python entry point for producing adapter QA artifacts. It runs the structural audit internally and builds the artifact through the same policy path as the CLI. Python API and CLI are thin wrappers over the same builder -- not cousins who disagree.

**Not exported from `__init__`:**
- `build_qa_artifact` -- stays one layer down for power users who import from `gradience.vnext.audit.qa_artifact` directly
- `derive_structural_flags`, `derive_confidence`, `build_reasons` -- builder internals
- `AdapterQAResult` -- merge-facing type, stays internal
- Confidence constants -- string values are self-documenting in JSON

## Section 3: Examples, Documentation, and Downstream Behavior

### Canonical Example Files

Four files in `examples/qa/`, one per eligibility status:

- `eligible_adapter_qa.json` -- adapter clearly beats base (high confidence, no structural flags)
- `uncertain_adapter_qa.json` -- adapter within margin of base (medium confidence, notes explain delta)
- `flagged_weak_adapter_qa.json` -- adapter underperforms base (high confidence, multiple structural flags)
- `structural_only_qa.json` -- no behavioral evidence (low confidence, status `unknown_no_behavioral_eval`)

These serve as both human reference and test fixtures.

### Documentation

One page: `docs/adapter-qa-artifact.md`. Definition document, not tutorial.

Headings:
1. What it is
2. How to produce it (CLI and Python API)
3. How to read it (section-by-section walkthrough)
4. How to consume it (passing to merge-audit, the canonical workflow)
5. Schema contract (field table with types, required/optional, frozen semantics)
6. Decision semantics (status table, confidence rules, margin definition)
7. Versioning policy

Under "How to consume it," explicitly state: when a QA artifact is provided to merge-audit, eligibility status can change warnings, recommendations, and strict-QA behavior.

### Downstream Behavioral Consequences

**What already works:**
- `merge-audit --source-a-qa` / `--source-b-qa` consumes QA artifacts
- `screen_adapters()` generates warnings based on eligibility status
- `--strict-qa` blocks merge-audit when adapter is `flagged_weak`

**What this design adds:**

1. **`--strict-qa` blocks `unknown_no_behavioral_eval`** -- strict mode means "behavioral evidence is required before merge recommendation proceeds."

   | Status | `--strict-qa` behavior |
   |--------|----------------------|
   | `eligible` | allow |
   | `uncertain` | allow with warning |
   | `flagged_weak` | block |
   | `unknown_no_behavioral_eval` | block |

2. **Merge report surfaces QA status prominently** -- a "Source QA Summary" section near the top showing each adapter's status, confidence, and short reason. Not buried in warnings.

3. **`audit-adapter` always produces a valid v1 artifact** -- test invariant: any artifact emitted by `audit-adapter` must be loadable by `AdapterQAArtifact.from_dict()` without error.

4. **Future: aggregation** -- because eligibility and supporting evidence are stored explicitly, QA artifacts can later be aggregated across adapter inventories without re-running structural audit. Not in scope for this phase; the schema supports it.

### Canonical Workflow

Once adapter QA artifacts are present, the canonical workflow becomes:

1. `audit-adapter` produces the adapter QA artifact
2. `merge-audit` consumes one artifact per source adapter
3. Policy behavior changes based on eligibility status
