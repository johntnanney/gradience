# Inventory Drift / History Support — Field Audit

**Date:** 2026-03-27
**Purpose:** Map existing stable fields to drift summary components.

---

## 1. Fields in preflight_summary.json

These are the fields written by `build_preflight_summary_json()` in
`run_bundle.py`. All are available for drift comparison.

| Key | Type | Drift use |
|-----|------|-----------|
| `adapter_count` | int | adapter_count_delta |
| `pair_count` | int | pair_count_delta |
| `retained_candidate_count` | int | retained_candidate_delta, same_task_safe_zone_change, status |
| `advisory_pair_count` | int | cross_task_caution_zone_change |
| `behavioral_evidence_count` | int | evidence_profile_change numerator |
| `total_source_count` | int | evidence_profile_change denominator |
| `reduction_ratio` | float | informational (not directly used for labels) |
| `excluded_sources` | list[str] | source_composition_changed detection |
| `inventory_policy_summary` | dict (optional) | policy_change sub-object |
| `same_task_priority_pairs` | list[str] | not used directly (retained_candidate_count covers it) |
| `cross_task_caution_regions` | list[str] | not used directly (advisory_pair_count covers it) |
| `reduced_candidate_subset` | list[str] | not used for drift (already in compare_to_previous) |
| `summary_line` | str | not used (dynamic prose, not diffable) |

---

## 2. What Already Exists in build_comparison_md()

The existing comparison function (run_bundle.py lines 285-366) already
computes:

| Comparison | Drift field it maps to |
|-----------|----------------------|
| `adapter_count` prev → curr | `adapter_count_delta` |
| `pair_count` prev → curr | `pair_count_delta` |
| `retained_candidate_count` prev → curr | `retained_candidate_delta`, `same_task_safe_zone_change` |
| `advisory_pair_count` prev → curr | `cross_task_caution_zone_change` |
| `behavioral_evidence_count` / `total_source_count` | `evidence_profile_change` |
| `excluded_sources` set diff | `source_composition_changed` detection |
| `reduced_candidate_subset` set diff | not used for drift labels |
| Interpretation block (narrower/broader/unchanged) | `status` and `implication` |

**Key insight:** `build_comparison_md()` already computes nearly all
the raw deltas. The drift summary adds structured labels on top.

---

## 3. What Already Exists in batch.py

| Feature | Drift use |
|---------|-----------|
| `build_batch_summary()` → `trend` field | Already does narrowing/broadening/stable based on first/last retained_candidate_count. The drift extension adds per-consecutive-pair drift and richer trend fields. |
| `_BATCH_COLUMNS` | Already tracks adapter_count, pair_count, retained_candidate_count, advisory_pair_count, behavioral_evidence_count, total_source_count, reduction_ratio. All needed for drift. |
| `format_batch_summary()` → table | Needs a Drift column. |
| `emit_batch_summary()` → JSON/MD | Needs drift objects in per-run rows. |

---

## 4. What Must Be Built New

| Component | Location | New logic? |
|-----------|----------|-----------|
| `derive_drift_summary()` | `run_bundle.py` | Yes — new function. Takes two preflight_summary.json dicts, returns drift dict. Pure derivation from existing fields. |
| HISTORY / DRIFT SUMMARY section | `build_comparison_md()` | Minimal — append new section using derive_drift_summary output. |
| `inventory_drift_summary` in preflight JSON | `build_preflight_summary_json()` | Minimal — optional key, passed from caller. |
| Per-run drift in batch | `build_batch_summary()` | Moderate — compute drift between consecutive runs in the sorted list. |
| Drift column in batch table | `format_batch_summary()` | Minimal — add column. |
| Drift in batch markdown | `emit_batch_summary()` | Minimal — add column. |
| Policy change tracking in batch | `build_batch_summary()` | Moderate — extract and compare policy_summary across runs. |

---

## 5. Integration Points

### compare_to_previous.md

Append new `## History / Drift Summary` section at the end of
`build_comparison_md()`. The drift summary is computed inside this
function since it already has both `current_summary` and
`previous_summary`.

### preflight_summary.json

The drift summary should be computed by the caller (`emit_run_bundle`
or CLI) and passed to `build_preflight_summary_json()` as an optional
parameter, same pattern as `policy_summary`.

### batch_summary.json / batch_summary.md

`build_batch_summary()` iterates over sorted runs. For each
consecutive pair (i, i+1), compute `derive_drift_summary()` and
attach the result to the later run's row. The first run gets no drift.

### Terminal batch output

`format_batch_summary()` adds a Drift column (8-char width)
showing the status label.

---

## 6. No Dataclass Changes Required

The drift summary is a derived dict computed at rendering time.
It is never stored in `InventorySummary`, `InventoryActionPlan`,
or any frozen dataclass.
