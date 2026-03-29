# Inventory Policy Summary — Design Document

**Status:** Draft
**Date:** 2026-03-27
**Schema:** `gradience.inventory_summary/v1` (additive extension)

---

## 1. Purpose

The inventory policy summary is a bounded, policy-style interpretation block
that tells the operator three things at a glance:

1. **What kind of inventory this is** (inventory type).
2. **What is driving interpretation** (dominant driver).
3. **How broadly or narrowly exploration should proceed** (exploration posture).

It sits between the existing STRUCTURAL DETAIL and INTERPRETATION sections
in the terminal formatter, and appears as a compact JSON object
(`inventory_policy_summary`) in the inventory summary JSON and preflight
summary JSON.

The block adds no new scoring or recommendation logic. It derives entirely
from existing stable signals already present in `InventorySummary` and
`InventoryActionPlan`.

---

## 2. Components

The policy summary has exactly four components. Each is a controlled
vocabulary (enum). Free-text is not permitted in any component.

### 2.1 inventory_type

Classifies the inventory by its dominant composition.

| Value | Derivation rule |
|-------|----------------|
| `same_task` | All pairs are same-task (action_plan.cross_task_count == 0 and no excluded sources) |
| `cross_task` | All pairs are cross-task (action_plan.retained_count == 0 and action_plan.cross_task_count > 0) |
| `mixed_task` | Both same-task and cross-task pairs present, no weak/unknown sources |
| `mixed_quality` | At least one source has status `flagged_weak` or `unknown_no_behavioral_eval` |
| `weak_evidence` | All sources have status `flagged_weak` or `unknown_no_behavioral_eval`, or all evidence tiers are `behavioral_missing` |
| `empty` | No pair reports available (sources.merge_report_count == 0) |

**Priority order:** `empty` > `weak_evidence` > `mixed_quality` > `same_task` > `cross_task` > `mixed_task`.
Higher-priority types override lower when multiple conditions hold.

### 2.2 dominant_driver

Identifies the single signal that most narrows the candidate space.

| Value | Derivation rule |
|-------|----------------|
| `source_qa` | At least one source excluded by weak/unknown status |
| `task_boundary` | Cross-task pairs present and source QA is clean |
| `structural_risk` | More than half of retained pairs are high risk |
| `none` | Same-task, all low/medium risk, clean sources |

**Priority order:** `source_qa` > `task_boundary` > `structural_risk` > `none`.

### 2.3 exploration_posture

Tells the operator how broadly to explore.

| Value | Derivation rule |
|-------|----------------|
| `narrow` | retained_count / total_pairs < 0.25, or total_pairs == 0 |
| `moderate` | 0.25 ≤ retained_count / total_pairs < 0.75 |
| `broad` | retained_count / total_pairs ≥ 0.75 |
| `confirmatory` | inventory_type == `same_task` and dominant_driver == `none` |

**Priority order:** `confirmatory` > `narrow` > `moderate` > `broad`.

### 2.4 constraint

A single sentence (from a frozen set) stating the binding constraint.

| Condition | Wording |
|-----------|---------|
| dominant_driver == `source_qa` | `"Source QA is the binding constraint; resolve weak evidence before exploring merges."` |
| dominant_driver == `task_boundary` | `"Task boundary is the binding constraint; same-task pairs are the credible exploration space."` |
| dominant_driver == `structural_risk` | `"Structural risk dominates retained pairs; audit individual reports before merging."` |
| dominant_driver == `none` and inventory_type == `same_task` | `"This same-task inventory is mostly confirmatory; low-risk merges can proceed."` |
| dominant_driver == `none` and inventory_type != `same_task` | `"No single driver dominates; review individual pair reports for case-by-case guidance."` |
| inventory_type == `empty` | `"No pair reports available; run pairwise merge audits to populate this inventory."` |

These are the **only** permitted constraint strings. No other wording
may be emitted. This prevents drift.

---

## 3. Derivation Rules — Detailed

The derivation function takes two inputs:

- `summary: InventorySummary`
- `action_plan: InventoryActionPlan`

Both are already computed by the time the policy summary is needed.

```python
def derive_inventory_policy_summary(
    summary: InventorySummary,
    action_plan: InventoryActionPlan,
) -> dict[str, str]:
    """Derive the four policy summary components.

    Pure derivation from existing stable signals. No new scoring.
    """
```

### Step 1: Count ingredients

```
n_reports = summary.sources.get("merge_report_count", 0)
n_qa = summary.sources.get("qa_artifact_count", 0)
status_counts = summary.adapter_status_counts or {}
weak_count = status_counts.get("flagged_weak", 0)
unknown_count = status_counts.get("unknown_no_behavioral_eval", 0)
weak_total = weak_count + unknown_count
risk_counts = summary.pair_risk_counts or {}
high_risk = risk_counts.get("high", 0)
total_pairs = action_plan.total_pairs
retained = action_plan.retained_count
cross_task = action_plan.cross_task_count
```

### Step 2: Derive inventory_type

```
if n_reports == 0:
    inventory_type = "empty"
elif n_qa > 0 and weak_total == n_qa:
    inventory_type = "weak_evidence"
elif weak_total > 0:
    inventory_type = "mixed_quality"
elif cross_task == 0:
    inventory_type = "same_task"
elif retained == 0:
    inventory_type = "cross_task"
else:
    inventory_type = "mixed_task"
```

### Step 3: Derive dominant_driver

```
if weak_total > 0:
    dominant_driver = "source_qa"
elif cross_task > 0:
    dominant_driver = "task_boundary"
elif retained > 0 and high_risk > retained // 2:
    dominant_driver = "structural_risk"
else:
    dominant_driver = "none"
```

### Step 4: Derive exploration_posture

```
if inventory_type == "same_task" and dominant_driver == "none":
    exploration_posture = "confirmatory"
elif total_pairs == 0 or (total_pairs > 0 and retained / total_pairs < 0.25):
    exploration_posture = "narrow"
elif retained / total_pairs < 0.75:
    exploration_posture = "moderate"
else:
    exploration_posture = "broad"
```

### Step 5: Derive constraint

Constraint is looked up from the frozen wording table (§2.4) based
on `dominant_driver` and `inventory_type`.

---

## 4. JSON Schema

The `inventory_policy_summary` field is an optional additive extension
to the existing `inventory_summary/v1` schema. Old consumers that do
not recognize it will ignore it (forward compatibility).

```json
{
  "inventory_policy_summary": {
    "inventory_type": "mixed_task",
    "dominant_driver": "task_boundary",
    "exploration_posture": "moderate",
    "constraint": "Task boundary is the binding constraint; same-task pairs are the credible exploration space."
  }
}
```

**Placement in existing outputs:**

1. `InventorySummary.to_dict()` — new optional key at top level.
2. `preflight_summary.json` — new optional key at top level.
3. Terminal output — new INVENTORY POLICY SUMMARY block between
   STRUCTURAL DETAIL and INTERPRETATION.
4. `preflight_summary.md` — new section after Source QA.

---

## 5. Human-Readable Block

Terminal format (from `format_inventory_summary`):

```
  INVENTORY POLICY SUMMARY
  ----------------------------------------
  Type:        mixed_task
  Driver:      task_boundary
  Posture:     moderate
  Constraint:  Task boundary is the binding constraint; same-task pairs
               are the credible exploration space.
```

The block is always emitted when the policy summary is available.
It appears after STRUCTURAL DETAIL and before INTERPRETATION.

---

## 6. Regime-Specific Examples

### 6.1 Same-task control

```
inventory_type:       same_task
dominant_driver:      none
exploration_posture:  confirmatory
constraint:           This same-task inventory is mostly confirmatory; low-risk merges can proceed.
```

Scenario: 4 adapters, all QNLI seeds, all eligible, all same-task pairs,
all low risk. The operator should feel safe to merge.

### 6.2 Standard mixed-task

```
inventory_type:       mixed_task
dominant_driver:      task_boundary
exploration_posture:  moderate
constraint:           Task boundary is the binding constraint; same-task pairs are the credible exploration space.
```

Scenario: 5 adapters across QNLI and SST-2, all eligible. 6 same-task
pairs retained out of 10 total. Cross-task pairs excluded.

### 6.3 Messy mixed-quality

```
inventory_type:       mixed_quality
dominant_driver:      source_qa
exploration_posture:  narrow
constraint:           Source QA is the binding constraint; resolve weak evidence before exploring merges.
```

Scenario: 6 adapters, 2 flagged_weak. After excluding weak sources and
cross-task pairs, only 1 of 15 pairs remains.

### 6.4 Weak-evidence

```
inventory_type:       weak_evidence
dominant_driver:      source_qa
exploration_posture:  narrow
constraint:           Source QA is the binding constraint; resolve weak evidence before exploring merges.
```

Scenario: 3 adapters, all unknown_no_behavioral_eval. No credible pairs.

### 6.5 Empty

```
inventory_type:       empty
dominant_driver:      none
exploration_posture:  narrow
constraint:           No pair reports available; run pairwise merge audits to populate this inventory.
```

Scenario: QA artifacts exist but no pair reports have been generated.

---

## 7. Guardrails

1. **No free-text.** Every component value comes from a frozen enum.
   The constraint string comes from a frozen lookup table. No
   interpolation, no f-strings, no dynamic wording.

2. **No new scoring.** The derivation function reads only from
   `InventorySummary` and `InventoryActionPlan`. It does not access
   raw QA artifacts, merge reports, or any other data source.

3. **No behavioral claims.** The policy summary never asserts that
   adapters are good or bad. It describes the *shape* of the inventory
   and the *binding constraint*, not the quality of individual adapters.

4. **Deterministic.** Given the same `InventorySummary` and
   `InventoryActionPlan`, the output is always identical. No randomness,
   no timestamp dependence, no ordering sensitivity.

5. **Additive only.** The `inventory_policy_summary` field is optional
   in JSON. Old consumers ignore it. The terminal block is appended,
   not inserted into existing sections. No existing output is modified
   or removed.

6. **Wording table is the single source of truth.** If the constraint
   wording needs to change, it changes in one place (the lookup table
   in the derivation function). Tests assert exact string equality
   against the table.

---

## 8. Implementation Surface

| File | Change |
|------|--------|
| `gradience/vnext/inventory/summary.py` | Add `derive_inventory_policy_summary()`, update `format_inventory_summary()` |
| `gradience/vnext/inventory/run_bundle.py` | Add policy summary to `build_preflight_summary_json()` and `build_preflight_summary_md()` |
| `tests/test_inventory_summary.py` | Add test class for each regime + JSON mirror test |

No changes to `InventorySummary` dataclass (frozen schema). The policy
summary is a derived presentation layer, not a stored field.

---

## 9. Relationship to Existing Output

The policy summary does **not** replace the INTERPRETATION section.
INTERPRETATION provides context-sensitive prose (e.g., "Check
task-boundary advisories on individual pair reports"). The policy
summary provides machine-readable classification (type, driver,
posture) plus a single frozen constraint sentence. They are
complementary: the policy summary tells you *what kind* of inventory
this is; INTERPRETATION tells you *what to do next*.

The policy summary also does **not** replace the action plan's
`summary_line`. The `summary_line` is a dynamic sentence about
candidate-space reduction. The policy summary's `constraint` is a
frozen sentence about the binding constraint. They may say similar
things in simple cases but diverge in complex ones.
