# Inventory Summary

## 1. What It Is

A Gradience inventory summary is a descriptive record of an adapter fleet: how many adapters exist, what their eligibility statuses are, how many structural flags were raised, and what the pairwise merge risk distribution looks like. It aggregates counts from adapter QA artifacts and merge risk reports. It is not a decision-bearing object -- it summarizes existing judgments but does not invent new ones.

Schema identifier: `gradience.inventory_summary/v1`

## 2. How to Produce It

### CLI

```bash
gradience summarize-inventory \
  --qa-dir ./qa_artifacts/ \
  --report-dir ./merge_reports/ \
  --emit-report inventory.json
```

`--qa-dir` globs for `*.json` adapter QA artifacts. `--report-dir` globs for `*.json` merge risk reports. Both are optional (you can summarize only adapters, only merge reports, or both). `--emit-report <path>` writes the v1 JSON to a file. Without it, only the terminal summary is printed.

### Python API

```python
from gradience.api import summarize_inventory

summary = summarize_inventory(
    qa_dir="./qa_artifacts/",
    report_dir="./merge_reports/",
)

# Serialize
import json
with open("inventory.json", "w") as f:
    json.dump(summary.to_dict(), f, indent=2)
```

`summarize_inventory()` is direct Python aggregation -- it loads files, validates them through `from_dict()`, and counts. It does not shell out to a subprocess.

Advanced extension:
- For rule-based grouping suggestions across the same inventory inputs, use `gradience suggest-neighborhoods` (see `docs/merge-neighborhoods.md`).
- This is a conservative diagnostic aid, not part of the default preflight path.

## 3. How to Read It

A v1 summary has these top-level sections:

- **`sources`** -- input counts: `qa_artifact_count` (number of adapter QA artifacts loaded) and `merge_report_count` (number of merge risk reports loaded).
- **`adapter_status_counts`** -- distribution of eligibility statuses across all loaded QA artifacts. Keys are status values (`eligible`, `uncertain`, `flagged_weak`, `unknown_no_behavioral_eval`). Only non-zero keys are present.
- **`adapter_flag_counts`** -- distribution of structural warning flags across all loaded QA artifacts. Keys are flag strings. Only non-zero keys are present.
- **`pair_risk_counts`** -- distribution of `pair_risk` values across all loaded merge reports. Keys are `low`, `medium`, `high`. Only non-zero keys are present.
- **`recommended_strategy_counts`** -- distribution of `recommended_strategy` values across all loaded merge reports. Only non-zero keys are present.
- **`dominant_issue_counts`** -- distribution of `dominant_issue` labels across all loaded merge reports. Only non-zero keys are present.
- **`strict_qa_block_candidates`** -- number of merge reports that would be blocked under `--strict-qa`. A report is a block candidate if either adapter has an eligibility status of `flagged_weak`, `unknown_no_behavioral_eval`, or `null` (no QA artifact provided).
- **`evidence_tier_counts`** -- distribution of evidence tiers across all loaded QA artifacts. Keys are `behavioral_reported` (eval available + eligible/uncertain), `behavioral_weak` (eval available + flagged_weak), `behavioral_missing` (no eval data). Only non-zero keys are present. Additive field: old v1 dicts without this field are backfilled to `{}`.
- **`notes`** -- optional list of caveats or annotations.

**Trust language note:** Behavioral scores are user-reported; Gradience does not independently verify claimed evaluation results. The word "verified" is reserved for a future capability where Gradience could cross-check evaluation claims. Until then, all behavioral evidence is described as "reported."

Count maps only include non-zero keys. An empty map (`{}`) is valid and means no items of that category were observed.

## 4. How to Consume It

### Loading in Python

```python
import json
from gradience import InventorySummary

with open("inventory.json") as f:
    summary = InventorySummary.from_dict(json.load(f))

print(f"Adapters scanned: {summary.sources['qa_artifact_count']}")
print(f"Merge reports:    {summary.sources['merge_report_count']}")

if summary.strict_qa_block_candidates > 0:
    print(f"WARNING: {summary.strict_qa_block_candidates} pair(s) would be blocked under --strict-qa")
```

### Scripting

`strict_qa_block_candidates` counts the number of merge risk reports that would be blocked if `--strict-qa` were enabled. This lets operators estimate the impact of enabling strict mode without re-running audits.

Count-map keys are not validated against frozen vocabularies. Unknown keys are tolerated, making the summary forward-compatible with future status values, flag names, or strategy labels.

## 5. Schema Contract

### Required fields

| Path | Type | Notes |
|------|------|-------|
| `schema` | `str` | Must be `"gradience.inventory_summary/v1"` |
| `sources` | `dict[str, int]` | Must contain `qa_artifact_count` and `merge_report_count` |
| `adapter_status_counts` | `dict[str, int]` | Eligibility status distribution |
| `adapter_flag_counts` | `dict[str, int]` | Structural flag distribution |
| `pair_risk_counts` | `dict[str, int]` | Pair risk distribution |
| `recommended_strategy_counts` | `dict[str, int]` | Strategy distribution |
| `dominant_issue_counts` | `dict[str, int]` | Dominant issue distribution |
| `strict_qa_block_candidates` | `int` | Number of reports blocked under `--strict-qa` |

### Optional fields

| Path | Type | Default | Notes |
|------|------|---------|-------|
| `notes` | `list[str]` | `[]` | Caveats or annotations |

Extra keys at any level are silently ignored (forward compatible).

### Validation rules

- Missing or wrong `schema` raises `QASchemaError`.
- Missing any required section raises `QASchemaError`.
- Count-map sections must be `dict`. All values must be `int`.
- `strict_qa_block_candidates` must be `int`.
- `notes` must be `list[str]` if present.
- Count-map keys are not validated against frozen vocabularies (sparse summaries with unknown keys are tolerated).

## 6. Malformed Input Behavior

By default, files that fail to load are skipped with a warning to stderr. This includes files that are not valid JSON, have the wrong `schema` identifier, or fail `from_dict()` validation.

`--strict-input` (CLI) or `strict_input=True` (Python API) changes this behavior: the first malformed file raises an exception and halts the summary.

## 7. Inventory Action Plan

The CLI also prints an **Inventory Action Plan** after the summary. This is a structured presentation of the same underlying data, organized for action:

- **Reduced candidate set**: Starting pairs, retained candidates, reduction percentage
- **Evaluate first**: Top same-task candidates with per-pair risk level and recommended strategy
- **Exclude / deprioritize**: Sources with weak or missing behavioral evidence
- **Same-task safe zone**: Retained same-task pairs (with risk and strategy)
- **Cross-task caution zone**: Task regions that cross the task boundary
- **Provenance**: How many sources have behavioral evidence
- **Summary**: One-line interpretation

The action plan uses only existing stable signals (source QA status, pair-risk, task-relationship advisory). It introduces no new scoring or recommendation logic.

`InventoryActionPlan` is available as a public API export:

```python
from gradience import InventoryActionPlan
from gradience.vnext.inventory.summary import build_action_plan
```

## 8. Preflight Run Bundle

For repeated or hand-off workflows, `--emit-bundle <DIR>` writes a standardized run bundle:

```bash
gradience summarize-inventory \
  --qa-dir ./qa/ --report-dir ./reports/ \
  --emit-bundle ./runs/run_001 \
  --previous-run ./runs/run_000
```

The bundle directory contains:

| File | Description |
|------|-------------|
| `preflight_summary.md` | Human-readable top-level summary (start here) |
| `preflight_summary.json` | Machine-readable summary for scripting |
| `run_manifest.json` | Run metadata and provenance |
| `inventory_action_plan.md` | Standalone action plan in markdown |
| `compare_to_previous.md` | Change summary vs. previous run (if `--previous-run` provided) |

Additional flags: `--inventory-id` (default: directory basename), `--run-id` (default: timestamp).

A `latest/` symlink is maintained at the bundle's parent directory, pointing to the most recent run.

## 9. Versioning Policy

The schema identifier `gradience.inventory_summary/v1` is frozen.

- New fields may be added without a version bump.
- No existing field will be renamed, removed, or have its semantics changed.
- A future version that changes the contract must use a new schema identifier (e.g., `gradience.inventory_summary/v2`).
