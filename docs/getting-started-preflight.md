# Getting Started: Preflight Check

This guide walks through the complete Gradience artifact pipeline using the bundled example adapter. By the end, you'll have produced all three artifact types and verified they're consistent.

## Prerequisites

```bash
pip install gradience[hf]
```

## Step 1: Audit a Single Adapter

Produce an `AdapterQAArtifact` from the bundled example adapter:

```bash
gradience audit-adapter \
  --peft-dir examples/adapters/tiny_lora \
  --out /tmp/gradience_preflight/qa_artifact.json
```

Expected: exit code 0, JSON file written with `"schema": "gradience.adapter_qa/v1"`.

Since this adapter has no behavioral evaluation, the artifact will have `eligibility.status = "unknown_no_behavioral_eval"`.

## Step 2: Run a Merge Audit

Produce a `MergeQAReport` comparing two adapters. For this demo, we compare the adapter against itself (a real workflow would use two different adapters). We feed in the QA artifact from Step 1 so the report includes source eligibility context:

```bash
gradience merge-audit \
  --adapter-a examples/adapters/tiny_lora \
  --adapter-b examples/adapters/tiny_lora \
  --source-a-qa /tmp/gradience_preflight/qa_artifact.json \
  --source-b-qa /tmp/gradience_preflight/qa_artifact.json \
  --emit-report /tmp/gradience_preflight/merge_report.json
```

Expected: exit code 0, JSON file written with `"schema": "gradience.merge_qa_report/v1"`.

## Step 3: Summarize the Inventory

Produce an `InventorySummary` from the artifacts:

```bash
gradience summarize-inventory \
  --qa-dir /tmp/gradience_preflight \
  --report-dir /tmp/gradience_preflight \
  --emit-report /tmp/gradience_preflight/inventory_summary.json
```

Expected: exit code 0, terminal summary printed, JSON file written with `"schema": "gradience.inventory_summary/v1"`.

## Step 4: Verify

Check that all three files were created and contain valid JSON:

```bash
for f in /tmp/gradience_preflight/*.json; do
  echo "=== $(basename "$f") ==="
  python3 -c "import json; d=json.load(open('$f')); print(d.get('schema', 'NO SCHEMA'))"
done
```

Expected output:
```
=== inventory_summary.json ===
gradience.inventory_summary/v1
=== merge_report.json ===
gradience.merge_qa_report/v1
=== qa_artifact.json ===
gradience.adapter_qa/v1
```

## Strict-QA Blocking Example

The `--strict-qa` flag on `merge-audit` blocks merges when either adapter lacks behavioral evaluation. Since our example adapter has no eval:

```bash
gradience merge-audit \
  --adapter-a examples/adapters/tiny_lora \
  --adapter-b examples/adapters/tiny_lora \
  --source-a-qa /tmp/gradience_preflight/qa_artifact.json \
  --source-b-qa /tmp/gradience_preflight/qa_artifact.json \
  --strict-qa
```

Expected: non-zero exit code, error message about adapter eligibility (because neither adapter has behavioral evaluation).

## Cleanup

```bash
rm -rf /tmp/gradience_preflight
```

## Python API

The same workflow is available programmatically:

```python
from gradience.api import audit_adapter, merge_risk_report, summarize_inventory

# Step 1: Audit
qa = audit_adapter(peft_dir="examples/adapters/tiny_lora")

# Step 2: Merge audit (delegates to CLI subprocess)
report = merge_risk_report(
    adapter_a="examples/adapters/tiny_lora",
    adapter_b="examples/adapters/tiny_lora",
)

# Step 3: Summarize (direct aggregation from JSON files)
summary = summarize_inventory(qa_dir="/tmp/gradience_preflight", report_dir="/tmp/gradience_preflight")
```

## Demo Bundle

For a pre-built set of artifacts covering all key cases (eligible, weak, and missing-QA adapters; safe and risky merge pairs; inventory summary), see `examples/demo/`.

## Next Steps

- See `docs/adapter-qa-artifact.md` for the adapter QA schema contract
- See `docs/merge-risk-report.md` for the merge report schema contract
- See `docs/inventory-summary.md` for the inventory summary schema contract
- See `docs/preflight-policy.md` for cross-artifact consistency rules
