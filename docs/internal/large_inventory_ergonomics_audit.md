# Large-Inventory Ergonomics — Audit Document

## 1. Where Current Large-Inventory Outputs Become Flat

### `format_action_plan()` in `summary.py`
- **Same-task safe zone**: renders `same_task_priority` as a flat list of
  `pair_label (risk, strategy)` lines. At 8+ adapters / 28+ same-task pairs,
  this becomes a wall of text with no grouping by task.
- **Cross-task caution zone**: renders `cross_task_caution` as flat region
  strings. These are already grouped ("QNLI × SST-2 region") but lack pair
  counts or status labels.
- **Evaluate first**: capped at 4 pairs for readability, but the flat list
  gives no sense of which regions they belong to.

### `build_preflight_summary_md()` in `run_bundle.py`
- **Reduced candidate set** section: flat list of evaluate_first pairs.
- **Task-boundary partition**: shows counts only (same-task N, cross-task M),
  no per-region breakdown.

### `build_review_packet_md()` in `run_bundle.py`
- **Action Plan** section: flat evaluate-first list, flat same-task safe zone,
  flat cross-task caution zone — inherits all flat-list limitations from
  action plan rendering.

### `build_action_plan_md()` in `run_bundle.py`
- Markdown version of `format_action_plan()` — same flat-list issues.

---

## 2. What Current Outputs Already Imply Grouped Structure

### Task-label grouping (in `build_action_plan()`)
- Already extracts normalized task labels: `"qnli_dev"` → `"QNLI"`.
- Already constructs cross-task region strings: `"QNLI × SST-2 region"`.
- Same-task pairs are partitioned but **not subgrouped** by task label.

### `InventoryActionPlan` fields
- `same_task_priority`: all same-task pairs — could be grouped by task label.
- `cross_task_caution`: already region strings — natural region boundaries.
- `evaluate_first`: subset of same-task — could be tagged per region.
- `exclude`: weak adapters — could form a single weak-evidence region.
- `retained_pair_detail`: per-pair (label, risk, strategy) — provides the
  data needed for per-region pair tables.

### Existing neighborhoods module (`neighborhoods.py`)
- Graph-based clustering exists but is a separate pipeline (different
  abstraction level). The large-inventory pass should NOT depend on
  neighborhoods — it should use task-label grouping only, which is
  simpler, deterministic, and already present in `build_action_plan()`.

---

## 3. Renderer Files That Need Updating

| File | Function | Change |
|------|----------|--------|
| `summary.py` | (new functions) | Add `derive_region_summary()`, `format_region_summary()`, `format_candidate_space_map()` |
| `run_bundle.py` | `build_preflight_summary_json()` | Add `large_inventory_region_summary` key |
| `run_bundle.py` | `build_preflight_summary_md()` | Add region summary section |
| `run_bundle.py` | `build_review_packet_md()` | Add region summary section |
| `run_bundle.py` | `build_review_packet_json()` | Add `large_inventory_region_summary` key |
| `run_bundle.py` | `emit_run_bundle()` | Pass region summary through |

---

## 4. Review Packet Integration

The review packet should include the region summary as a new section
(between Policy Summary and Source QA / Trust Snapshot) when the threshold
is met. The JSON companion should include the full
`large_inventory_region_summary` object.

When the threshold is not met, neither the markdown section nor the JSON
key should appear (graceful absence, not empty block).

---

## 5. Data Flow

```
QA artifacts + merge reports
  → build_action_plan()        [existing]
  → derive_region_summary()    [NEW — from action_plan + qa_artifacts]
  → format_region_summary()    [NEW — terminal/md block]
  → format_candidate_space_map()  [NEW — md table]
  → build_region_summary_json()   [NEW — JSON mirror]
```

The new functions are pure derivation from `InventoryActionPlan` and
QA artifact data. No merge report re-analysis needed.
