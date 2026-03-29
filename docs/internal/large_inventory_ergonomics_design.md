# Large-Inventory Ergonomics — Design Document

## Purpose

Add a presentation layer that renders large mixed-task inventories as a small number
of meaningful **regions** instead of flat pair lists. No new analysis logic — grouping
is derived from existing stable signals only.

---

## 1. Activation Threshold

Large-inventory mode activates when:

    adapter_count >= 8  OR  pair_count >= 20

This is a deterministic, documented threshold. Below this threshold, current flat
rendering is sufficient.

Constants:

```python
LARGE_INVENTORY_ADAPTER_THRESHOLD = 8
LARGE_INVENTORY_PAIR_THRESHOLD = 20
```

Helper:

```python
def is_large_inventory(adapter_count: int, pair_count: int) -> bool
```

---

## 2. Region Types

Four stable region types, used as presentation labels (not analytical categories):

| Label                         | Machine key                    | Definition                                                      |
|-------------------------------|--------------------------------|-----------------------------------------------------------------|
| Same-task safe region         | `same_task_safe_region`        | Same-task pairs grouped by shared eval-dataset task label        |
| Cross-task caution region     | `cross_task_caution_region`    | Cross-task pairs grouped by task-pair label (e.g. QNLI × SST-2) |
| Weak-evidence region          | `weak_evidence_region`         | Pairs touching at least one weak/missing-evidence adapter        |
| Priority evaluation region    | `priority_evaluation_region`   | Subset of same-task safe that overlaps with evaluate_first       |

---

## 3. Region Derivation Rules

### Inputs (all from existing stable outputs)

- `InventoryActionPlan.same_task_priority` — same-task pair labels
- `InventoryActionPlan.cross_task_caution` — cross-task region strings
- `InventoryActionPlan.evaluate_first` — priority subset
- `InventoryActionPlan.exclude` — weak/excluded adapters
- `InventoryActionPlan.retained_pair_detail` — (label, risk, strategy) per retained pair
- `InventoryActionPlan.total_pairs`, `.retained_count`, `.cross_task_count`
- QA artifacts — adapter eligibility, eval_dataset
- Merge reports — task_relationship_advisory, pair_risk, recommended_strategy

### Derivation steps

1. **Extract task labels** from eval_dataset on each QA artifact.
   Normalize: `"qnli_dev"` → `"QNLI"`, `"sst2_dev"` → `"SST-2"`.

2. **Map each pair to its task-pair key**: for same-task pairs, the key is the
   shared task label (e.g. `"QNLI"`). For cross-task pairs, the key is the
   sorted task-pair string already in `cross_task_caution` (e.g. `"QNLI × SST-2 region"`).

3. **Group same-task pairs by task label** → each group is a same-task safe region.

4. **Cross-task caution regions** are already grouped — one per entry in
   `cross_task_caution`.

5. **Weak-evidence region**: count pairs excluded due to weak sources.
   If `action_plan.exclude` is non-empty, one weak-evidence region exists.
   (Multiple weak regions are not distinguished — all weak-evidence exclusions
   are grouped into a single region.)

6. **Priority evaluation regions**: same-task safe regions that contain at least
   one pair from `evaluate_first`.

### Not allowed

- New clustering or graph algorithms
- Sidecar signals or neighborhood data
- Subjective ranking heuristics
- New scoring systems

---

## 4. Output Block: LARGE INVENTORY REGION SUMMARY

Appears in the terminal/markdown summary when the threshold is met.

### Format

```
  LARGE INVENTORY REGION SUMMARY
  ============================================================

  Region counts:
    Same-task safe regions:         3
    Cross-task caution regions:     5
    Weak-evidence regions:          1
    Priority evaluation regions:    2

  This inventory is best understood as a small number of grouped
  regions rather than a flat list of pairs.
```

The interpretation line is frozen wording (not free text).

---

## 5. Output Block: CANDIDATE-SPACE MAP

A compact markdown table showing every region, its type, pair count, and status.

### Format

```
  CANDIDATE-SPACE MAP
  ============================================================

  | Region             | Type                    | Pairs | Status                    |
  |--------------------|-------------------------|------:|---------------------------|
  | QNLI               | same-task safe          |     3 | evaluate first            |
  | RTE                 | same-task safe          |     1 | evaluate first            |
  | QNLI × SST-2       | cross-task caution      |     4 | do not explore casually   |
  | MRPC × SST-2        | cross-task caution      |     4 | do not explore casually   |
  | (weak evidence)     | weak-evidence           |     3 | excluded                  |
```

### Region label rules

- Same-task safe: use the normalized task label (e.g. `QNLI`, `SST-2`)
- Cross-task caution: use the task-pair label without the word "region"
  (e.g. `QNLI × SST-2`)
- Weak-evidence: literal `(weak evidence)`

### Status labels (controlled vocabulary)

| Status                    | Machine key                  | When used                          |
|---------------------------|------------------------------|------------------------------------|
| evaluate first            | `evaluate_first`             | Region overlaps evaluate_first set |
| same-task safe            | `same_task_safe`             | Same-task, no priority overlap     |
| do not explore casually   | `do_not_explore_casually`    | Cross-task caution                 |
| excluded                  | `excluded`                   | Weak-evidence                      |

---

## 6. Region-Aware Reduced-Candidate Table

When large-inventory mode is active, the evaluate-first list is grouped by region:

```
  | Region | Retained candidates          |
  |--------|------------------------------|
  | QNLI   | qnli_a × qnli_b             |
  | RTE    | rte_a × rte_b                |
```

This replaces the flat list in large-inventory mode only.

---

## 7. JSON Mirror

```json
{
  "large_inventory_region_summary": {
    "enabled": true,
    "threshold_trigger": "8_adapters",
    "region_counts": {
      "same_task_safe_regions": 3,
      "cross_task_caution_regions": 5,
      "weak_evidence_regions": 1,
      "priority_evaluation_regions": 2
    },
    "candidate_space_map": [
      {
        "region": "QNLI",
        "type": "same_task_safe_region",
        "pair_count": 3,
        "status": "evaluate_first",
        "pairs": ["qnli_a × qnli_b", "qnli_a × qnli_c", "qnli_b × qnli_c"]
      },
      {
        "region": "QNLI × SST-2",
        "type": "cross_task_caution_region",
        "pair_count": 4,
        "status": "do_not_explore_casually",
        "pairs": []
      },
      {
        "region": "(weak evidence)",
        "type": "weak_evidence_region",
        "pair_count": 3,
        "status": "excluded",
        "pairs": []
      }
    ]
  }
}
```

### Field contracts

- `enabled`: `true` when threshold met, `false` otherwise
- `threshold_trigger`: `"8_adapters"` or `"20_pairs"` — whichever fired first
- `region_counts`: four integer counts
- `candidate_space_map`: list of region objects, sorted: priority evaluation first,
  then same-task safe, then cross-task caution, then weak-evidence
- `pairs`: populated for same-task regions; empty list for cross-task and weak-evidence
  (individual cross-task pairs are not tracked in the action plan)

---

## 8. Integration Points

### 8.1 `format_inventory_summary()` / `format_action_plan()`

When threshold met, append the LARGE INVENTORY REGION SUMMARY and CANDIDATE-SPACE MAP
blocks after the existing content.

### 8.2 `build_preflight_summary_md()` / `build_preflight_summary_json()`

Add `large_inventory_region_summary` to the JSON output.
Add region summary section to the markdown output.

### 8.3 `build_review_packet_md()` / `build_review_packet_json()`

Add region summary as a new section in the review packet when threshold met.

---

## 9. Guardrails

1. **No new scoring.** Region types are presentation labels, not risk scores.
2. **No free-text in region labels or status labels.** All from controlled vocabulary.
3. **Deterministic.** Same inputs always produce identical region assignments.
4. **No behavioral claims.** Regions do not assert merge safety — they organize
   existing judgments.
5. **Additive only.** This is a new optional block; existing outputs unchanged.
6. **Threshold is documented and tested.** Below threshold, no region block appears.
7. **No clustering algorithms.** Regions are grouped by task label, not by graph
   analysis or geometric similarity.

---

## 10. Regime Examples

### Example 1: 8 adapters, all same-task (QNLI)
- 1 same-task safe region (QNLI), 28 pairs
- 0 cross-task caution regions
- 0 weak-evidence regions
- 1 priority evaluation region (QNLI)
- Interpretation: "This inventory is best understood as a small number of grouped regions rather than a flat list of pairs."

### Example 2: 10 adapters, 3 tasks (QNLI, SST-2, MRPC)
- 3 same-task safe regions
- 3 cross-task caution regions (MRPC × QNLI, MRPC × SST-2, QNLI × SST-2)
- 0 weak-evidence regions
- 2 priority evaluation regions
- Candidate-space map: 6 rows

### Example 3: 12 adapters, 4 tasks, 2 weak sources
- 3 same-task safe regions (after weak exclusion)
- 4 cross-task caution regions (after weak exclusion)
- 1 weak-evidence region
- 2 priority evaluation regions

### Example 4: 20 pairs, 6 adapters (threshold via pair count)
- threshold_trigger: "20_pairs"
- Large-inventory mode active despite < 8 adapters

### Example 5: 7 adapters, 18 pairs
- Below both thresholds
- Large-inventory mode NOT active
- `enabled: false`
