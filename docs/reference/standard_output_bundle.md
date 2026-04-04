# Standard Output Bundle

**Audience:** practitioner, operator, maintainer  
**Status:** stable  
**Purpose:** define the single recommended deliverable for a normal inventory run  
**Canonical for:** default output bundle expectations in product workflows  
**Supersedes:** ad-hoc artifact collection choices  
**See also:** [`../workflows/canonical_merge_triage_workflow.md`](../workflows/canonical_merge_triage_workflow.md), [`../inventory-summary.md`](../inventory-summary.md), [`../merge-risk-report.md`](../merge-risk-report.md), [`../adapter-qa-artifact.md`](../adapter-qa-artifact.md)

For a normal inventory run, the canonical bundle is:

1. Adapter QA results
2. Merge QA reports
3. Inventory summary
4. Human-readable report (Markdown and/or HTML)
5. Structured JSON run bundle

This is the default product deliverable. Additional diagnostics can exist, but they are non-default.

## Canonical Bundle Layout

```text
<run_root>/
  qa/
    <adapter_id>_qa.json
  reports/
    <adapter_a>_vs_<adapter_b>.json
  inventory/
    summary.json
  runs/
    run_001/
      preflight_summary.md
      preflight_summary.json
      run_manifest.json
      inventory_action_plan.md
      compare_to_previous.md        (optional)
      preflight_report.html         (when generated)
```

## How to Produce the Bundle

### 1) Adapter QA artifacts

```bash
gradience audit-adapter \
  --peft-dir ./adapters/<adapter_id> \
  --eval-dataset <dataset> \
  --metric-name <metric> \
  --adapter-score <adapter_score> \
  --base-score <base_score> \
  --out qa/<adapter_id>_qa.json
```

Primary output:

- `qa/<adapter_id>_qa.json` (`gradience.adapter_qa/v1`)

### 2) Pairwise merge QA reports

```bash
gradience merge-audit \
  --adapter-a ./adapters/<adapter_a> \
  --adapter-b ./adapters/<adapter_b> \
  --source-a-qa qa/<adapter_a>_qa.json \
  --source-b-qa qa/<adapter_b>_qa.json \
  --qa-report \
  --emit-report reports/<adapter_a>_vs_<adapter_b>.json
```

Primary output:

- `reports/<adapter_a>_vs_<adapter_b>.json` (`gradience.merge_qa_report/v1`)

### 3) Inventory summary + structured run bundle

```bash
gradience summarize-inventory \
  --qa-dir qa/ \
  --report-dir reports/ \
  --emit-report inventory/summary.json \
  --emit-bundle runs/run_001
```

Primary outputs:

- `inventory/summary.json` (`gradience.inventory_summary/v1`)
- `runs/run_001/preflight_summary.json` (structured bundle summary)
- `runs/run_001/run_manifest.json` (bundle metadata/provenance)
- `runs/run_001/preflight_summary.md` (human-readable summary)
- `runs/run_001/inventory_action_plan.md` (human-readable action plan)

### 4) Human-readable HTML report (recommended)

```bash
gradience preflight-report --bundle-dir runs/run_001
```

Primary output:

- `runs/run_001/preflight_report.html`

## What Counts as “Standard”

A run is considered complete for normal product use when all of the following exist:

1. QA artifact set (`qa/*.json`)
2. Pair report set (`reports/*.json`)
3. Inventory summary (`inventory/summary.json`)
4. Bundle JSON (`preflight_summary.json`, `run_manifest.json`)
5. At least one human-readable summary (`preflight_summary.md` or `preflight_report.html`)

## Non-Default Outputs

These may be produced for advanced/research use, but are not part of the standard bundle:

1. Neighborhood suggestion outputs
2. Core-space diagnostics
3. Over-accumulation companion diagnostics
4. Telemetry and monitor outputs
5. Experimental probe artifacts

Treat these as add-ons, not required bundle elements.
