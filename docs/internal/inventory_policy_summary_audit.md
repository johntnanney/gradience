# Inventory Policy Summary — Field Audit

**Date:** 2026-03-27
**Purpose:** Map existing stable signals to the four policy summary components.

---

## 1. Available Ingredients

### From InventorySummary

| Field | What it provides | Maps to |
|-------|-----------------|---------|
| `sources["merge_report_count"]` | Total pair reports | inventory_type (`empty` when 0) |
| `sources["qa_artifact_count"]` | Total adapters | weak_evidence detection (all weak?) |
| `adapter_status_counts["flagged_weak"]` | Weak source count | inventory_type, dominant_driver |
| `adapter_status_counts["unknown_no_behavioral_eval"]` | Unknown source count | inventory_type, dominant_driver |
| `pair_risk_counts["high"]` | High-risk pair count | dominant_driver (`structural_risk`) |
| `evidence_tier_counts` | Behavioral evidence breakdown | Corroborates weak_evidence type |
| `strict_qa_block_candidates` | Block-candidate count | Not used directly (redundant with weak counts) |

### From InventoryActionPlan

| Field | What it provides | Maps to |
|-------|-----------------|---------|
| `total_pairs` | Total pair count | exploration_posture denominator |
| `retained_count` | Same-task retained pairs | exploration_posture numerator |
| `cross_task_count` | Cross-task pair count | inventory_type, dominant_driver |
| `exclude` | Excluded source list | Corroborates source_qa driver |
| `summary_line` | Dynamic reduction sentence | Not used (policy summary has its own constraint) |

---

## 2. What Already Exists (Overlap Analysis)

### _inventory_headline() in summary.py

Currently classifies into three buckets:
- "Mixed-quality inventory" — when weak + unknown > 0
- "high structural risk dominates" — when high > total // 2
- generic "{n_qa} adapters, {n_reports} pairs"

**Overlap:** The first two cases map directly to `mixed_quality` / `structural_risk`.
The policy summary generalizes this into a proper enum with more cases.

### action_plan.summary_line

Currently produces dynamic sentences like:
- "This same-task inventory is mostly confirmatory."
- "QA dominates this inventory; no credible same-task candidates remain."
- "All pairs are cross-task; no same-task safe region exists."
- "QA and task boundary dominate this inventory."
- "Inventory is mostly explained by task boundary."

**Overlap:** These map to inventory_type + dominant_driver combinations.
The policy summary replaces the ad-hoc classification with a controlled
vocabulary, but the summary_line remains as a complementary dynamic sentence.

### INTERPRETATION section in format_inventory_summary()

Currently branches on:
- weak_total > 0 → source QA guidance
- high_risk > total_pairs // 2 → structural risk guidance
- total_pairs > 0 → task-boundary guidance
- fallback → no pairs available

**Overlap:** Nearly identical branching logic to the dominant_driver
derivation. The INTERPRETATION section stays as-is (it gives actionable
prose). The policy summary adds the machine-readable classification above it.

---

## 3. What Must Be Built New

| Component | New logic needed? | Source |
|-----------|------------------|--------|
| inventory_type | Yes — new enum derivation | Combines weak counts + cross_task_count + retained_count |
| dominant_driver | Minimal — mirrors existing branching in _inventory_headline and INTERPRETATION | Existing conditionals, refactored into a clean function |
| exploration_posture | Yes — new ratio-based classification | retained_count / total_pairs ratio |
| constraint | Yes — new frozen wording lookup table | Keyed on (dominant_driver, inventory_type) |

---

## 4. No Dataclass Changes Required

The `InventorySummary` dataclass is frozen at schema v1. The policy
summary is a **derived** presentation layer:

- Computed by a new `derive_inventory_policy_summary()` function
- Passed to formatters as a separate argument
- Added to JSON output as an optional top-level key
- Never stored in the dataclass itself

This is the correct approach because:
1. The policy summary is fully deterministic from existing fields
2. Storing derived data in a frozen schema creates a consistency obligation
3. Old consumers should not be forced to handle the new field

---

## 5. Integration Points

### Terminal output (format_inventory_summary)

Insert new block between STRUCTURAL DETAIL and INTERPRETATION.
The derivation function must be called inside `format_inventory_summary`,
which means it needs access to the `InventoryActionPlan`. Currently
`format_inventory_summary` takes only an `InventorySummary`.

**Options:**
- (A) Add optional `action_plan` parameter to `format_inventory_summary`
- (B) Pass pre-computed policy summary dict as optional parameter
- (C) New standalone `format_policy_summary()` function; caller concatenates

**Recommended: (B).** Pass the pre-computed dict. This keeps the derivation
function separate from the formatter, avoids changing the existing
function signature in a breaking way (the param is optional with default
None), and lets the JSON and terminal paths share the same derivation output.

### Preflight summary JSON (build_preflight_summary_json)

Add `"inventory_policy_summary": {...}` at top level. The derivation
function must be called by the caller (run_bundle.py or CLI) and passed in.

### Preflight summary MD (build_preflight_summary_md)

Add a new ## Inventory Policy Summary section after ## Source QA.

### CLI (cmd_summarize_inventory)

The CLI already builds both `InventorySummary` and `InventoryActionPlan`.
It calls `derive_inventory_policy_summary(summary, action_plan)` once
and passes the result to all output paths.
