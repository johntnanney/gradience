# Inventory Summary v1 Design

> Phase C of the product object trilogy (A: Adapter QA Artifact, B: Merge Risk Report, C: Inventory Summary).

## Goal

Take the adapter QA artifacts and pairwise merge-risk reports produced by Phases A and B and make them useful at inventory scale. Reliable aggregation, useful summaries, a stable batch-level object.

Not graphing. Not dashboards. Just counting.

## Approach

Direct Python aggregation over already-produced JSON artifacts. No subprocess delegation, no GPU, no model loading. Pure parsing + counting.

## Schema

Schema identifier: `gradience.inventory_summary/v1`

```json
{
  "schema": "gradience.inventory_summary/v1",
  "sources": {
    "qa_artifact_count": 5,
    "merge_report_count": 3
  },
  "adapter_status_counts": {
    "eligible": 2,
    "uncertain": 1,
    "flagged_weak": 1,
    "unknown_no_behavioral_eval": 1
  },
  "adapter_flag_counts": {
    "low_utilization": 3,
    "high_rank_waste": 2,
    "concentrated_spectrum": 1,
    "underutilized_capacity": 1
  },
  "pair_risk_counts": {
    "low": 1,
    "medium": 1,
    "high": 1
  },
  "recommended_strategy_counts": {
    "linear": 1,
    "norm_equalized": 1,
    "audit_aware": 1
  },
  "dominant_issue_counts": {
    "none": 1,
    "norm_imbalance": 1,
    "subspace_conflict": 1
  },
  "strict_qa_block_candidates": 2,
  "notes": []
}
```

### Field Semantics

- **`sources`** -- input artifact counts. Both `qa_artifact_count` and `merge_report_count` required, both int.
- **`adapter_status_counts`** -- counts by `EligibilityStatus` value. Only keys with count > 0 need be present. Values are int.
- **`adapter_flag_counts`** -- aggregated structural flag counts across all QA artifacts. One adapter can contribute multiple flags. Only non-zero keys present.
- **`pair_risk_counts`** -- counts by pair risk level (`low`/`medium`/`high`). Only non-zero keys present.
- **`recommended_strategy_counts`** -- counts by machine-readable strategy. Only non-zero keys present.
- **`dominant_issue_counts`** -- counts by dominant issue label. Only non-zero keys present.
- **`strict_qa_block_candidates`** -- int. Number of pair reports where either adapter has `eligibility_status` of `flagged_weak`, `null`, or `unknown_no_behavioral_eval`.
- **`notes`** -- optional list of strings.

### What Is NOT in v1

No per-file paths, no per-adapter details, no compatibility score aggregation, no graphing data. This is a summary, not an index.

### Validation Rules

- `schema` required, must equal `"gradience.inventory_summary/v1"`.
- `sources`, `adapter_status_counts`, `adapter_flag_counts`, `pair_risk_counts`, `recommended_strategy_counts`, `dominant_issue_counts` required sections (all `dict[str, int]`).
- `strict_qa_block_candidates` required, int.
- `notes` optional, backfill to `[]`, must be `list[str]` if present.
- Count map values must be int; keys are strings (not validated against frozen vocabularies -- sparse summaries with unknown keys tolerated).
- Extra keys at any level silently ignored.
- Raises `QASchemaError` on contract violations (reuses existing exception).

### Versioning

Additive only. Same policy as Phases A and B.

## Aggregation Logic

### Core Function

`build_inventory_summary(qa_artifacts, merge_reports) -> InventorySummary`

Takes two lists of already-parsed objects:
- `qa_artifacts: list[AdapterQAArtifact]`
- `merge_reports: list[MergeQAReport]`

Pure counting -- no recomputation of eligibility or risk. Iterates once through each list, tallies counts, returns frozen dataclass. This object is descriptive, not decision-bearing.

### `strict_qa_block_candidates` Definition

A pair report counts as a block candidate if either adapter has:
- `eligibility_status == "flagged_weak"`
- `eligibility_status is None` (no QA provided)
- `eligibility_status == "unknown_no_behavioral_eval"`

Matches `--strict-qa` blocking behavior from Phase B exactly.

### Malformed Input Handling

The aggregation function takes already-parsed objects, so validation has already happened. Malformed-file-skipping logic lives in the loader layer (CLI and `summarize_inventory()` API).

- **Default:** skip malformed files with a warning to stderr.
- **`--strict-input`:** fail hard on first malformed file.
- **Malformed** = fails `from_dict()`, not valid JSON, or missing `schema` field.

## Public API

### Python API (`gradience/api.py`)

```python
def summarize_inventory(
    *,
    qa_dir: str | Path | None = None,
    report_dir: str | Path | None = None,
    qa_paths: list[str | Path] | None = None,
    report_paths: list[str | Path] | None = None,
    strict_input: bool = False,
) -> InventorySummary:
```

Direct Python -- not subprocess. Scans directories for `*.json`, loads via `from_dict()`, calls `build_inventory_summary()`. Two ways to provide inputs: directory scan or explicit file lists. At least one source must be provided.

### Exports

- `InventorySummary` added to `gradience.__init__.__all__`.
- `summarize_inventory` in `gradience.api`.

### File Location

`gradience/vnext/inventory/summary.py` -- new `inventory/` subpackage under vnext.

## CLI

### Command

```bash
gradience summarize-inventory \
  --qa-dir examples/qa \
  --report-dir examples/reports \
  --emit-report inventory_summary.json \
  --strict-input
```

### Flags

- `--qa-dir` -- directory to scan for QA artifact JSON files
- `--report-dir` -- directory to scan for merge report JSON files
- `--emit-report` -- write v1 JSON to file
- `--strict-input` -- fail on first malformed file instead of skipping

Without `--emit-report`, prints terminal summary only.

### Terminal Format

```
  INVENTORY SUMMARY
  ============================================================

  SOURCES
  ----------------------------------------
  QA artifacts:    5
  Merge reports:   3

  ADAPTER STATUS
  ----------------------------------------
  eligible:                     2
  uncertain:                    1
  flagged_weak:                 1
  unknown_no_behavioral_eval:   1

  STRUCTURAL FLAGS
  ----------------------------------------
  low_utilization:       3
  high_rank_waste:       2

  PAIR RISK
  ----------------------------------------
  low:      1
  medium:   1
  high:     1

  RECOMMENDED STRATEGIES
  ----------------------------------------
  linear:           1
  norm_equalized:   1
  audit_aware:      1

  DOMINANT ISSUES
  ----------------------------------------
  none:                1
  norm_imbalance:      1
  subspace_conflict:   1

  STRICT-QA BLOCK CANDIDATES: 2
```

Sections with all-zero counts are omitted.

## Examples and Documentation

### Canonical Example

One file: `examples/inventory/inventory_summary.json` -- hand-crafted, loadable via `from_dict()`.

Test fixtures reuse existing `examples/qa/*.json` and `examples/reports/*.json` as aggregation inputs.

### Definition Doc

`docs/inventory-summary.md` with sections:
1. What it is
2. How to produce it (CLI + Python API)
3. How to read it
4. How to consume it
5. Schema contract
6. Malformed input behavior
7. Versioning policy

## Testing

### Validation Tests
- Missing/wrong schema, wrong count types, missing required sections, notes backfill.

### Aggregation Tests
- Adapter status counts correct, flag counts correct, pair risk/strategy/issue counts correct, strict block candidates counted correctly.
- Empty inputs (no QA, no reports) produce zero counts.
- Single adapter with multiple flags counted correctly.

### CLI Tests
- Emits valid JSON via `--emit-report`.
- Terminal output contains expected sections.
- Malformed file skipped with warning (default).
- `--strict-input` fails on malformed file.

### Example File Smoke Test
- Load canonical example via `from_dict()`.

## Key Design Decisions

1. **Summary is descriptive, not decision-bearing.** No new judgments invented. Only existing ones aggregated.
2. **Skip malformed files by default, optionally fail hard.** Most practical batch behavior.
3. **Direct Python aggregation, not subprocess wrapper.** Parsing/counting layer, not heavy computation.
4. **No per-file index in v1.** Summary only.
5. **No compatibility graphing yet.** This summary is the staging ground for that later.
6. **Reuse `QASchemaError`.** Already the schema validation exception for both Phase A and B objects.
