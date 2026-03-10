# Pipeline Integration Test Design

## Problem

The merge recommendation pipeline has four stages — audit, diagnose, recommend, format — each well unit-tested in isolation. But no test wires real on-disk adapter fixtures through the full chain:

```
merge_audit() → diagnose_pair() → recommend_merge() → format_recommendation()
```

`test_integration.py` stops at `MergeAuditReport`. `test_recommend.py` feeds hand-built `_FakeReport` objects into the downstream stages. If the data shapes drift between stages, nothing catches it.

## Design

### File

`tests/merge/test_pipeline_integration.py`

### Structure

One test class `TestMergeRecommendPipeline` with four methods, one per existing fixture type from `tests/merge/conftest.py`:

- `test_orthogonal_safe_pipeline` — expects safe verdict, LinearMerge strategy
- `test_redundant_pipeline` — expects redundant verdict, TIESMerge strategy
- `test_conflicting_pipeline` — expects conflicting verdict, DARETIESMerge strategy
- `test_imbalanced_pipeline` — expects imbalanced verdict, LinearMerge with rebalanced coefficients

Each test follows the same shape:

1. Call `merge_audit(adapter_a, adapter_b)` on real adapter directories
2. Call `diagnose_pair(report)` on the result
3. Call `recommend_merge(report)` on the same result
4. Call `format_recommendation(rec)` on the recommendation
5. Assert properties at each stage boundary

### Assertion Strategy: Property-Based with Targeted Exact Checks

Assertions verify structural properties (non-empty, correct types, valid ranges) rather than pinned values. A few deterministic outcomes are checked exactly where the fixture design guarantees them.

**After `merge_audit()` → `MergeAuditReport`:**
- `report.layer_verdicts` is non-empty, length matches expected layer count (2)
- Each layer verdict has required keys: `layer_name`, `verdict`, `confidence`, `metrics`
- `report.overall_verdict` is a non-empty string
- `report.overall_score` is a float in [0, 1]

**After `diagnose_pair()` → `PairDiagnosis`:**
- `diagnosis.layers` length matches `report.layer_verdicts` length
- Each `LayerDiagnosis` has valid `risk_level`
- `diagnosis.overall_risk` is a valid risk level
- `diagnosis.compression_needed` is a bool
- Verdicts propagate correctly from the fixture type

**After `recommend_merge()` → `MergeRecommendation`:**
- `rec.layer_recommendations` length matches layer count
- Each `LayerRecommendation` has a non-empty `strategy` string
- Strategy families match fixture expectations (orthogonal → LinearMerge, redundant → TIESMerge, conflicting → DARETIESMerge, imbalanced → LinearMerge)
- `rec.overall_strategy` is non-empty
- `rec.warnings` is a list

**After `format_recommendation()` → `str`:**
- Output is non-empty
- Contains the overall strategy name
- Contains at least one shortened layer reference (e.g. "L0.")

### Fixtures

Reuses existing fixtures from `tests/merge/conftest.py` — no new fixtures needed:
- `orthogonal_pair`, `redundant_pair`, `conflicting_pair`, `imbalanced_pair`
- Each returns `tuple[Path, Path]` of real on-disk PEFT adapter directories

### Team Approach

Two agents with sequential handoff:

1. **Researcher** (read-only) — traces exact field names, attribute names, strategy string values, and risk level enums across the dataclasses. Produces a concrete assertion spec.
2. **Implementer** (worktree) — writes the test file from the spec, runs pytest and ruff to verify.

The researcher runs first because getting field names wrong is the primary risk — dataclasses are spread across `recommend.py`, `report.py`, and `__init__.py`.
