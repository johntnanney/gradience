# Merge Risk Report v1 — Design Document

**Date:** 2026-03-11
**Status:** Approved
**Goal:** Freeze the existing `MergeQAReport` as a stable, decision-bearing pair-level product object — the counterpart to `AdapterQAArtifact` (Phase A).

## What It Is

A Gradience merge risk report is the canonical record of a pairwise adapter comparison: structural compatibility, eligibility status of both sources, risk level, and recommended merge action.

It is not a metric dump. It is a decision-bearing object: downstream scripts and workflows consume it and change behavior based on its contents.

## Design Approach

Freeze the existing `MergeQAReport` in place. The shape is already sound. The work is: rename a few fields for clarity, split one stringly-typed field, add strict validation, promote to public API, add examples and documentation.

## Section 1: Frozen v1 Schema

### Shape

```json
{
  "schema": "gradience.merge_qa_report/v1",
  "adapter_a": {
    "path": "./adapters/catsubcat-r16",
    "rank": 16,
    "alpha": 16.0,
    "n_layers": 32,
    "base_model": "meta-llama/Llama-2-7b-hf",
    "eligibility_status": "flagged_weak"
  },
  "adapter_b": {
    "path": "./adapters/btgenbot-r8",
    "rank": 8,
    "alpha": 8.0,
    "n_layers": 32,
    "base_model": "meta-llama/Llama-2-7b-hf",
    "eligibility_status": null
  },
  "pair_risk": "high",
  "dominant_issue": "norm_imbalance",
  "dominant_issue_detail": "11.3x mean magnitude ratio across 15 layer(s)",
  "recommended_action": "Merge with caution using audit-aware strategy. Validate on downstream task.",
  "recommended_strategy": "audit_aware",
  "confidence": "medium",
  "confidence_note": "Low spectral compatibility (score=0.340).",
  "caveats": [
    "Source A underperforms base.",
    "No QA artifact provided for source B."
  ],
  "verdict_distribution": {
    "safe": 10,
    "redundant": 2,
    "conflicting": 5,
    "imbalanced": 15
  },
  "compatibility_score": 0.340
}
```

### Changes From Current Code

| Change | Rationale |
|--------|-----------|
| `eligibility` renamed to `eligibility_status` | Makes clear this is a categorical label, not a whole object |
| `"not provided"` replaced with `null` | Missing QA = absence, not pseudo-status. Canonical `EligibilityStatus` values or `null`. |
| `dominant_issue` split into label + detail | Machine-readable label for scripts, human-readable detail for explanation |
| `confidence` added as categorical field | `"high"` / `"medium"` / `"low"` — parallel to Phase A. `confidence_note` stays as prose companion. |
| `recommended_strategy` becomes operational | Derived from pair_risk + compression_needed, not hardcoded `"audit_aware"` |

### Eligibility Status Values in Adapter Summaries

- One of the four frozen `EligibilityStatus` values: `"eligible"`, `"uncertain"`, `"flagged_weak"`, `"unknown_no_behavioral_eval"`
- Or `null` when no QA artifact was provided for that adapter

`null` means "we were not given a QA artifact." `"unknown_no_behavioral_eval"` means "we have a QA artifact but it contains no behavioral evidence." These are different states.

### Dominant Issue Labels (Frozen)

| Label | When |
|-------|------|
| `"norm_imbalance"` | Imbalanced layers present with high magnitude ratio |
| `"subspace_conflict"` | Conflicting layers dominate |
| `"high_redundancy"` | Redundant layers outnumber safe layers |
| `"partial_redundancy"` | Some redundant layers, but safe layers dominate |
| `"none"` | Adapters are spectrally compatible |
| `"unknown"` | No layer data available (analysis incomplete or unsupported state) |

### Recommended Strategy (Frozen Vocabulary)

`recommended_strategy` is the primary machine-readable recommendation for downstream tooling. `recommended_action` provides explanatory prose but does not override the strategy label.

| `pair_risk` | Compression needed | `recommended_strategy` |
|-------------|-------------------|----------------------|
| `low` | no | `"linear"` |
| `low` | yes | `"audit_aware"` |
| `medium` | no | `"norm_equalized"` |
| `medium` | yes | `"audit_aware"` |
| `high` | any | `"audit_aware"` |

Compression needed: `True` when any layer in the pair diagnosis has `compress_first=True` (over-provisioned layers that should be truncated before merging).

### Pair Risk Values (Frozen)

| Value | Meaning |
|-------|---------|
| `"low"` | Adapters are structurally compatible for merging |
| `"medium"` | Some structural concerns; strategy matters |
| `"high"` | Significant structural risk; validate after merge |

`pair_risk` is always derived from structural analysis (layer verdicts, magnitude ratios). Eligibility status never affects `pair_risk`. Eligibility affects `caveats`, `recommended_action`, and `--strict-qa` behavior.

### Confidence Semantics

| Level | When |
|-------|------|
| `"high"` | Both adapters eligible, low structural risk, compatibility score >= 0.8 |
| `"medium"` | Behavioral evidence exists but incomplete, or moderate structural risk |
| `"low"` | No behavioral evidence for either adapter, or high structural risk |

### Versioning Policy

Additive only. New fields may be added without a version bump. No existing field will be renamed, removed, or have its semantics changed. A future version that changes the contract must use a new schema identifier.

## Section 2: Validation, Error Handling, and Public API

### Validation in `from_dict()`

`from_dict()` is the single canonical gatekeeper. One path, no drift.

**Rules:**

1. `schema` key required, must equal `"gradience.merge_qa_report/v1"`. Missing or wrong raises `QASchemaError`.

2. Required sections (`adapter_a`, `adapter_b`) must be present and must be dicts.

3. Required fields with type enforcement:
   - `adapter_a.path` (str), `adapter_a.rank` (numeric, normalized to int)
   - `adapter_b.path` (str), `adapter_b.rank` (numeric, normalized to int)
   - `pair_risk` (str, must be one of `"low"`, `"medium"`, `"high"`)
   - `dominant_issue` (str, must be one of the frozen labels)
   - `recommended_strategy` (str — required but not restricted to frozen vocabulary, for forward compat)
   - `confidence` (str, must be one of `"high"`, `"medium"`, `"low"`)
   - `compatibility_score` (numeric, normalized to float)

4. `eligibility_status` in adapter summaries: if present and non-null, must be one of the four `EligibilityStatus` values. Unknown values raise `QASchemaError`. If absent, backfill to `null`.

5. `alpha` in adapter summaries: accepts int or float, normalized to float.

6. `caveats`: if present, must be `list[str]`. Backfill to `[]` if absent. Required in produced output.

7. `verdict_distribution`: if present, must be a dict with string keys and integer values. Backfill to `{}` if absent.

8. `confidence_note`: required in produced output. Backfill to `""` on load if absent.

9. `dominant_issue_detail`: optional. Backfill to `""` on load if absent.

10. `recommended_action`: required in produced output. Backfill to `""` on load if absent.

11. Extra keys silently ignored (forward compatible).

### Loader Boundary

If a dict contains `"schema": "gradience.merge_qa_report/v1"`, it must go through `MergeQAReport.from_dict()` and may not silently fall back to older report parsing.

### Public API

**`gradience/__init__.py` exports:**
- `MergeQAReport`

**`gradience/api.py` adds:**
```python
def merge_risk_report(
    *,
    adapter_a: str | Path,
    adapter_b: str | Path,
    source_a_qa: str | Path | None = None,
    source_b_qa: str | Path | None = None,
    thresholds: str = "default",
    python: str | None = None,
    env: Mapping[str, str] | None = None,
    log_path: str | Path | None = None,
    check: bool = True,
) -> Any:
```

`merge_risk_report()` is the stable Python wrapper for the `merge-audit --qa-report --emit-report` workflow. It delegates report generation to the CLI, then loads the resulting JSON as a `MergeQAReport`. It is a thin wrapper over the CLI — not an alternate implementation path.

**Not exported from `__init__`:**
- `build_qa_report`, `format_qa_report` — builder internals, stay one layer down
- `AdapterSummary` — internal to the report

## Section 3: CLI Behavior, Examples, and Documentation

### CLI Changes

**`--emit-report <path>`** — repurposed to write `MergeQAReport.to_dict()` as JSON (v1 schema). This is the canonical machine-readable output.

**`--qa-report`** — continues to print the 4-section terminal format. Updates to `format_qa_report()`:
- Print `eligibility_status` (renamed), show `null` as `"not provided"` in terminal only
- Print `dominant_issue` label + detail on separate lines
- Print categorical `confidence` alongside `confidence_note`

### `--strict-qa` Policy

| Adapter A | Adapter B | `--strict-qa` behavior |
|-----------|-----------|----------------------|
| `eligible` | `eligible` | allow |
| `eligible` | `uncertain` | allow with warning |
| `eligible` | `null` (no QA) | block |
| `eligible` | `flagged_weak` | block |
| `null` | `null` | block |
| any | `unknown_no_behavioral_eval` | block |

`null` (no QA artifact provided) is treated the same as `unknown_no_behavioral_eval` under `--strict-qa`. Strict mode means "behavioral evidence is required."

Without `--strict-qa`, `null` eligibility generates a caveat but does not block.

### Pairwise Decision Policy

Structural analysis determines `pair_risk` and the primary merge recommendation; eligibility status determines `caveats`, blocking behavior, and the tone/content of `recommended_action`.

| Condition | `pair_risk` | `recommended_strategy` | Caveats |
|-----------|-------------|----------------------|---------|
| Both eligible, low risk, no compression | `low` | `linear` | none |
| Both eligible, medium risk | `medium` | `norm_equalized` | structural note |
| Both eligible, high risk | `high` | `audit_aware` | validate after merge |
| One uncertain | varies | varies | warning about uncertain source |
| One flagged_weak | varies | varies | warning about weak source |
| Both flagged_weak | varies | varies | "both underperform base" |
| One/both null (no QA) | varies | varies | "structural only" caveat |
| Compression needed | varies | `audit_aware` | pre-compress caveat |

### Canonical Examples

Three files in `examples/reports/`:

1. **`safe_merge_report.json`** — both eligible, low risk, `dominant_issue: "none"`, confidence `"high"`, strategy `"linear"`
2. **`high_risk_warn_report.json`** — one flagged_weak, high risk, `dominant_issue: "norm_imbalance"`, confidence `"low"`, strategy `"audit_aware"`, multiple caveats
3. **`strict_blocked_report.json`** — one null eligibility, high risk, caveats include "No QA artifact provided for source B."

### Documentation

One page: `docs/merge-risk-report.md`. Definition document, not tutorial.

Headings:
1. What it is
2. How to produce it (CLI `--emit-report` and Python `merge_risk_report()`)
3. How to read it (section walkthrough)
4. How to consume it (scripting, `--strict-qa`, exit codes)
5. Schema contract (field table with types, required/optional, frozen semantics)
6. Decision semantics (policy table, strategy derivation, confidence rules, strict-qa)
7. Versioning policy

Under "Schema contract," explicitly define:
- `compatibility_score`: range 0-1, higher = more compatible, derived from layer verdict distribution
- `recommended_strategy` is the primary machine-readable recommendation; `recommended_action` is explanatory prose and does not override it

### Tests

- `from_dict` validation: missing schema, wrong schema, missing sections, bad types, unknown `dominant_issue`, unknown `eligibility_status`, valid round-trip
- Builder: `build_qa_report()` produces valid v1 output for each fixture pair
- `recommended_strategy` derivation: low-risk → `"linear"`, medium-risk → `"norm_equalized"`, high-risk → `"audit_aware"`, compression → `"audit_aware"`
- CLI: `--emit-report` writes valid JSON loadable by `from_dict()`
- Strict-QA: `null` eligibility blocked under `--strict-qa`
- Example files: all 3 load through `from_dict()` without error
