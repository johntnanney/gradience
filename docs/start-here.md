# Start Here: Gradience Preflight Workflow

Gradience helps you answer one question before spending evaluation budget: **which adapter pairs are worth testing, and which should be excluded or deprioritized?**

The core workflow is a four-command pipeline that screens adapters individually, analyzes pairwise compatibility, and partitions your inventory by task boundary.

---

## 1. Run preflight

Preflight has three stages. Run each command against your adapter collection.

**Stage 1: Screen each adapter**

```bash
for d in ./adapters/*/; do
  name=$(basename "$d")
  gradience audit-adapter \
    --peft-dir "$d" \
    --eval-dataset gsm8k \
    --metric-name accuracy \
    --adapter-score 0.71 --base-score 0.63 \
    --out qa/${name}.json
done
```

Each `qa/<name>.json` is an `AdapterQAArtifact` (schema `gradience.adapter_qa/v1`). The key field is `eligibility.status`:

```json
{
  "schema": "gradience.adapter_qa/v1",
  "adapter": { "name": "math-lora-r64", "rank_nominal": 64, "n_layers": 32 },
  "structural_summary": { "utilization_mean": 0.81, "rank_waste_ratio": 0.19 },
  "eligibility": { "status": "eligible", "confidence": "high" }
}
```

Statuses: `eligible` | `uncertain` | `flagged_weak` | `unknown_no_behavioral_eval`

**Stage 2: Run all pairwise merge audits**

```bash
gradience merge-audit \
  --adapter-a ./adapters/adapter_1 --adapter-b ./adapters/adapter_2 \
  --source-a-qa qa/adapter_1.json --source-b-qa qa/adapter_2.json \
  --qa-report --emit-report reports/adapter_1_vs_adapter_2.json
```

The `--qa-report` flag prints a four-section terminal summary. The `--emit-report` writes the structured JSON (schema `gradience.merge_qa_report/v1`). Repeat for each pair.

**Stage 3: Summarize and group**

```bash
gradience summarize-inventory \
  --qa-dir qa/ --report-dir reports/ \
  --emit-report inventory/summary.json

gradience suggest-neighborhoods \
  --qa-dir qa/ --report-dir reports/ \
  --emit-report inventory/neighborhoods.json
```

---

## 2. Read the review packet

After Stage 3, you have a set of files that function as a structured review packet:

| File | What it contains |
|------|-----------------|
| `inventory/summary.json` | Aggregate counts: adapter status distribution, pair risk distribution, strategy recommendations |
| `inventory/neighborhoods.json` | Merge groups, exclusions, and cross-group boundary warnings |
| Terminal output from `--qa-report` | Per-pair: risk level, dominant issue, recommended strategy |

**What to look at first:**

The inventory summary immediately tells you how many adapters to exclude and how many pairs are high risk:

```json
{
  "adapter_status_counts": { "eligible": 1, "flagged_weak": 1, "unknown_no_behavioral_eval": 1 },
  "pair_risk_counts": { "high": 1, "low": 1 },
  "recommended_strategy_counts": { "audit_aware": 1, "linear": 1 },
  "strict_qa_block_candidates": 2
}
```

Source: `examples/demo/inventory_summary.json`

For a clean same-task inventory, you will see `"strict_qa_block_candidates": 0` and all pairs at `medium` or `low` risk — that confirmatory result is also useful.

---

## 3. Interpret a pair report

Each merge report (schema `gradience.merge_qa_report/v1`) has five decision fields:

```json
{
  "pair_risk": "high",
  "dominant_issue": "subspace_conflict",
  "dominant_issue_detail": "14 layers show directional disagreement (mean overlap 0.71)",
  "recommended_strategy": "audit_aware",
  "compatibility_score": 0.298,
  "task_relationship_advisory": "adapters evaluated on different tasks (gsm8k vs oasst2)"
}
```

Source: `examples/demo/reports/code_chat_risky.json`

For contrast, a safe same-task pair:

```json
{
  "pair_risk": "low",
  "dominant_issue": "none",
  "recommended_strategy": "linear",
  "compatibility_score": 0.874
}
```

Source: `examples/demo/reports/code_math_safe.json`

**The most important signal** is `task_relationship_advisory`. When it is present, the two adapters were evaluated on different tasks. Empirically, cross-task merges degrade at least one task on small encoder models — treat all advisory-present pairs as low-priority unless you have a specific justification.

---

## 4. Check drift across runs

When you re-run preflight on an updated adapter pool, compare against the previous run's artifacts manually:

```bash
# Track what changed in eligibility distribution
diff <(python3 -c "
import json, sys
d = json.load(open('inventory/summary.json'))
print(json.dumps(d['adapter_status_counts'], indent=2))
") <(python3 -c "
import json, sys
d = json.load(open('inventory_prev/summary.json'))
print(json.dumps(d['adapter_status_counts'], indent=2))
")
```

Key signals to watch across runs:
- **`strict_qa_block_candidates`** increasing — more adapters failing QA gate
- **`pair_risk_counts.high`** increasing — structural risk growing in the pool
- **advisory pair count** — cross-task pairs accumulating as the inventory grows

---

## 5. Portfolio view across multiple inventories

When you manage multiple adapter pools (e.g., different tasks, model families, or experiments), run `summarize-inventory` against each separately, then compare summaries side by side.

```bash
# Per-inventory summary
gradience summarize-inventory --qa-dir exp_a/qa/ --report-dir exp_a/reports/ \
  --emit-report results/exp_a_summary.json

gradience summarize-inventory --qa-dir exp_b/qa/ --report-dir exp_b/reports/ \
  --emit-report results/exp_b_summary.json
```

`suggest-neighborhoods` with `--exclude-unknown` gives a clean merge graph for each pool:

```bash
gradience suggest-neighborhoods \
  --qa-dir exp_a/qa/ --report-dir exp_a/reports/ \
  --exclude-unknown --emit-report results/exp_a_neighborhoods.json
```

The neighborhoods output groups adapters into merge clusters, flags excluded adapters, and surfaces cross-group boundary warnings:

```json
{
  "groups": [
    { "group_id": "cluster_01", "members": ["code_lora", "sql_lora"],
      "characterization": "likely-safe neighborhood", "common_strategy": "linear" },
    { "group_id": "cluster_02", "members": ["chat_lora", "instruct_lora"],
      "characterization": "audit-aware neighborhood", "common_strategy": "audit_aware" }
  ],
  "excluded": [{ "adapter": "btgenbot-r8", "reason": "flagged_weak" }],
  "boundary_warnings": [{ "between": ["cluster_01", "cluster_02"],
    "reason": "high cross-group merge risk" }]
}
```

Source: `examples/neighborhoods/sample_merge_neighborhoods.json`

Across portfolios, the `recommended_strategy_counts` distribution is the fastest way to compare fleet health: a portfolio dominated by `"audit_aware"` has more structural problems than one dominated by `"linear"`.

---

## Reference

- `examples/demo/` — pre-built artifacts covering all key cases
- `examples/inventory_preflight_same_task_control/` — clean same-task confirmation example
- `examples/inventory_preflight_mixed_task/` — mixed-task inventory with advisory partitioning
- `docs/adapter-qa-artifact.md` — adapter QA schema contract
- `docs/merge-risk-report.md` — merge report schema contract
- `docs/inventory-preflight.md` — deeper workflow guidance and evidence table
