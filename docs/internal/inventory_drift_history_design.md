# Inventory Drift / History Support — Design Document

**Status:** Draft
**Date:** 2026-03-27

---

## 1. Purpose

The inventory drift/history layer answers a single question for
operators who run preflight repeatedly:

> **What changed since the last run, and what does that mean for how
> I should treat this inventory now?**

It derives entirely from existing stable run-bundle outputs
(`preflight_summary.json`). It adds no new scoring, no hidden
heuristics, and no behavioral claims.

---

## 2. Tracked Concepts

The drift summary tracks six stable concepts. Each produces a
directional label from a controlled vocabulary (§3).

### 2.1 Inventory size

| Field | Source key | Delta |
|-------|-----------|-------|
| Adapter count | `adapter_count` | integer delta |
| Pair count | `pair_count` | integer delta |

### 2.2 Evidence profile

| Field | Source key | Delta |
|-------|-----------|-------|
| Behavioral evidence count | `behavioral_evidence_count` | integer delta |
| Total source count | `total_source_count` | integer delta |

**Evidence profile change** is derived from the ratio shift:

| Condition | Label |
|-----------|-------|
| evidence ratio increased (more sources now have behavioral evidence) | `improved` |
| evidence ratio decreased | `degraded` |
| evidence ratio unchanged | `unchanged` |
| either run has zero total sources | `unchanged` |

Evidence ratio = `behavioral_evidence_count / total_source_count`.

### 2.3 Same-task safe zone

| Field | Source key | Delta |
|-------|-----------|-------|
| Same-task pair count | `retained_candidate_count` | integer delta |

**Same-task safe zone change:**

| Condition | Label |
|-----------|-------|
| retained_candidate_count increased | `expanded` |
| retained_candidate_count decreased | `shrunk` |
| retained_candidate_count unchanged | `unchanged` |

### 2.4 Cross-task caution zone

| Field | Source key | Delta |
|-------|-----------|-------|
| Advisory pair count | `advisory_pair_count` | integer delta |

**Cross-task caution zone change:**

| Condition | Label |
|-----------|-------|
| advisory_pair_count increased | `expanded` |
| advisory_pair_count decreased | `shrunk` |
| advisory_pair_count unchanged | `unchanged` |

### 2.5 Reduced candidate set

| Field | Source key | Delta |
|-------|-----------|-------|
| Retained candidate count | `retained_candidate_count` | integer delta |
| Reduction ratio | `reduction_ratio` | float delta |

This overlaps with §2.3 intentionally — the same underlying field
drives both the "safe zone" directional label and the candidate-set
delta display.

### 2.6 Policy summary change

Compares `inventory_policy_summary` objects (when present in both runs).

| Subfield | Tracked? |
|----------|----------|
| `inventory_type` | yes — boolean changed |
| `dominant_driver` | yes — boolean changed |
| `exploration_posture` | yes — boolean changed |
| `constraint` | yes — boolean changed |

If either run lacks `inventory_policy_summary`, the policy change
section is omitted (not marked as changed).

---

## 3. Directional Vocabulary

### Zone/profile labels (controlled set)

| Label | Meaning |
|-------|---------|
| `expanded` | zone or count grew |
| `shrunk` | zone or count decreased |
| `unchanged` | no change |
| `improved` | evidence quality ratio increased |
| `degraded` | evidence quality ratio decreased |

### Top-line drift status (controlled set)

| Label | Derivation |
|-------|-----------|
| `narrowing` | retained_candidate_count decreased |
| `broadening` | retained_candidate_count increased |
| `stable` | retained_candidate_count unchanged |

### Implication labels (controlled set)

| Label key | Wording |
|-----------|---------|
| `current_run_materially_narrower` | `"Current run is materially narrower than the previous run."` |
| `current_run_materially_broader` | `"Current run has a broader candidate set than the previous run."` |
| `source_composition_changed` | `"Source composition changed; candidate subset is effectively similar."` |
| `no_substantial_change` | `"No substantial preflight change."` |
| `evidence_improved_posture_unchanged` | `"Evidence quality improved, but exploration posture is unchanged."` |
| `evidence_degraded` | `"Evidence quality degraded; review source QA before proceeding."` |

The implication is selected by priority:

1. If retained delta < 0 → `current_run_materially_narrower`
2. If retained delta > 0 → `current_run_materially_broader`
3. If excluded sources changed → `source_composition_changed`
4. If evidence profile changed to `improved` → `evidence_improved_posture_unchanged`
5. If evidence profile changed to `degraded` → `evidence_degraded`
6. Otherwise → `no_substantial_change`

---

## 4. JSON Schema

```json
{
  "inventory_drift_summary": {
    "status": "narrowing",
    "adapter_count_delta": 1,
    "pair_count_delta": 6,
    "retained_candidate_delta": -2,
    "same_task_safe_zone_change": "shrunk",
    "cross_task_caution_zone_change": "expanded",
    "evidence_profile_change": "improved",
    "policy_change": {
      "inventory_type_changed": false,
      "dominant_driver_changed": true,
      "exploration_posture_changed": false,
      "constraint_changed": true
    },
    "implication": "current_run_materially_narrower"
  }
}
```

**Placement:**

1. `preflight_summary.json` — new optional key at top level (only
   when a previous run exists).
2. `compare_to_previous.md` — new HISTORY / DRIFT SUMMARY section.
3. `batch_summary.json` — per-run drift objects in the `runs` array
   (for consecutive pairs).
4. `batch_summary.md` — drift status column in the comparison table.

---

## 5. Human-Readable Block

### In compare_to_previous.md

```
## History / Drift Summary

**Status:** narrowing

**Key changes:**
- Adapters: 4 → 5 (+1)
- Pairs: 6 → 12 (+6)
- Reduced candidate set: 4 → 2 (-2)
- Same-task safe zone: shrunk
- Cross-task caution zone: expanded
- Evidence profile: improved

**Policy change:**
- Dominant driver changed: source_qa → task_boundary
- Exploration posture: unchanged

**Implication:** Current run is materially narrower than the previous run.
```

### In batch summary

The existing table gains a `Drift` column showing the per-run
drift status (`narrowing`, `broadening`, `stable`, or `—` for
the first run).

---

## 6. Examples by Regime

### 6.1 Narrowing

Previous: 10 pairs, 6 retained. Current: 12 pairs, 2 retained.

```
status: narrowing
retained_candidate_delta: -4
same_task_safe_zone_change: shrunk
implication: current_run_materially_narrower
```

### 6.2 Broadening

Previous: 6 pairs, 2 retained. Current: 10 pairs, 6 retained.

```
status: broadening
retained_candidate_delta: +4
same_task_safe_zone_change: expanded
implication: current_run_materially_broader
```

### 6.3 Stable

Previous: 6 pairs, 3 retained. Current: 6 pairs, 3 retained.

```
status: stable
retained_candidate_delta: 0
same_task_safe_zone_change: unchanged
implication: no_substantial_change
```

### 6.4 Evidence improvement

Previous: 2/4 behavioral. Current: 4/4 behavioral. Retained unchanged.

```
status: stable
evidence_profile_change: improved
implication: evidence_improved_posture_unchanged
```

### 6.5 Evidence degradation

Previous: 4/4 behavioral. Current: 2/4 behavioral. Retained unchanged.

```
status: stable
evidence_profile_change: degraded
implication: evidence_degraded
```

### 6.6 Policy change

Previous: same_task / none / confirmatory. Current: mixed_task / task_boundary / moderate.

```
policy_change:
  inventory_type_changed: true
  dominant_driver_changed: true
  exploration_posture_changed: true
  constraint_changed: true
```

---

## 7. Guardrails

1. **No drift score.** The drift summary is a structured label set,
   not a numeric score.

2. **No new analysis.** All inputs come from `preflight_summary.json`
   fields that already exist.

3. **No behavioral claims.** The summary describes structural and
   compositional change, not adapter quality change.

4. **No forecasting.** The summary describes what changed between
   two runs. It does not predict future runs.

5. **Deterministic.** Given the same two `preflight_summary.json`
   inputs, the output is always identical.

6. **Additive only.** The `inventory_drift_summary` key is optional
   in JSON. Old consumers ignore it. The HISTORY / DRIFT SUMMARY
   section is appended to `compare_to_previous.md`, not inserted
   into existing sections.

7. **Controlled vocabulary only.** All directional labels come from
   the frozen sets in §3. No free-text, no f-string interpolation
   in labels.

---

## 8. Implementation Surface

| File | Change |
|------|--------|
| `gradience/vnext/inventory/run_bundle.py` | Add `derive_drift_summary()`, update `build_comparison_md()`, update `build_preflight_summary_json()` |
| `gradience/vnext/inventory/batch.py` | Add drift column to `build_batch_summary()` and formatters, add policy tracking |
| `tests/test_inventory_summary.py` | Add drift test classes |

No changes to `InventorySummary` or `InventoryActionPlan` dataclasses.

---

## 9. Relationship to Existing Outputs

The drift summary does **not** replace `compare_to_previous.md`.
The existing comparison shows raw deltas per field. The drift summary
adds a structured interpretation layer on top: a status label,
directional labels, and an implication sentence.

The drift summary does **not** replace the batch summary's existing
`trend` field. The batch trend is a first/last comparison. The drift
summary adds per-consecutive-pair drift objects, which are richer.

The drift summary does **not** replace the inventory policy summary.
The policy summary describes the current run. The drift summary
describes the change between runs.
