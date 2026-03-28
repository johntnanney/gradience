# Repeated-Use Workflow

Operational playbook for reviewers and collaborators running Gradience as a standing process. Assumes you have already read [inventory-preflight.md](inventory-preflight.md) and understand the artifact schema.

---

## Directory convention

This doc uses a consistent layout throughout:

```
project/
├── adapters/          # PEFT adapter directories
├── qa/                # adapter_qa/v1 artifacts (one per adapter)
├── reports/           # merge_qa_report/v1 artifacts (one per pair)
└── inventory/         # summary + neighborhoods (regenerated each run)
```

Adjust paths to match your setup. The `qa/` and `reports/` directories are the durable state; `inventory/` is always regenerated from them.

---

## 1. First run

**Goal:** Establish baseline artifacts for a new adapter pool.

### Step 1 — Screen each adapter

```bash
for d in ./adapters/*/; do
  name=$(basename "$d")
  gradience audit-adapter \
    --peft-dir "$d" \
    --eval-dataset <your-dataset> \
    --metric-name <accuracy|perplexity|...> \
    --adapter-score <score> \
    --base-score <base-score> \
    --higher-is-better \
    --out qa/${name}_qa.json
done
```

If you do not have behavioral evaluation scores yet, omit `--eval-dataset` through `--higher-is-better`. The artifact will record `status: "unknown_no_behavioral_eval"` — structurally valid, but weak evidence. Fill in scores later with a rerun.

Each artifact written to `qa/` is a `gradience.adapter_qa/v1` document:

```json
{
  "schema": "gradience.adapter_qa/v1",
  "adapter": { "name": "math-lora-r16", "rank_nominal": 16, "n_layers": 32 },
  "structural_summary": {
    "utilization_mean": 0.71,
    "effective_rank_90_median": 11.0,
    "rank_waste_ratio": 0.31,
    "flags": []
  },
  "eligibility": {
    "status": "eligible",
    "confidence": "high",
    "reasons": []
  }
}
```

**Triage immediately:** adapters with `flagged_weak` or structural flags (`low_utilization`, `high_rank_waste`) are candidates for exclusion before any pairwise work.

### Step 2 — Run pairwise reports

For N adapters you have N×(N-1)/2 pairs. Run them all:

```bash
# Example loop for a small pool
adapters=(adapter_a adapter_b adapter_c)
for ((i=0; i<${#adapters[@]}; i++)); do
  for ((j=i+1; j<${#adapters[@]}; j++)); do
    a=${adapters[$i]}
    b=${adapters[$j]}
    gradience merge-audit \
      --adapter-a "./adapters/$a" \
      --adapter-b "./adapters/$b" \
      --source-a-qa "qa/${a}_qa.json" \
      --source-b-qa "qa/${b}_qa.json" \
      --emit-report "reports/${a}_vs_${b}.json"
  done
done
```

Key flags:
- `--emit-report` writes the `gradience.merge_qa_report/v1` artifact. Required for downstream aggregation.
- `--qa-report` (optional) prints a concise 4-section terminal summary of each pair during the run.
- `--strict-qa` blocks the run if either source adapter is `flagged_weak` or lacks QA data. Use in CI; skip interactively if you want to audit weak pairs anyway.

### Step 3 — Build the inventory

```bash
gradience summarize-inventory \
  --qa-dir qa/ \
  --report-dir reports/ \
  --emit-report inventory/summary.json

gradience suggest-neighborhoods \
  --qa-dir qa/ \
  --report-dir reports/ \
  --emit-report inventory/neighborhoods.json
```

The summary counts every status, flag, and risk level across the pool. The neighborhoods compress the pair matrix into clusters and surface cross-task boundary warnings.

### Step 4 — Read the inventory summary

```bash
python -m json.tool inventory/summary.json
```

What to check first:

| Field | What a healthy pool looks like |
|-------|-------------------------------|
| `adapter_status_counts.flagged_weak` | 0 or low — weak adapters should be excluded |
| `pair_risk_counts.high` | Low — most pairs should be `low` or `medium` |
| `strict_qa_block_candidates` | 0 if all adapters have good behavioral evidence |
| `dominant_issue_counts.subspace_conflict` | Low — conflicts require `audit_aware` strategy |

Check `neighborhoods.json` for `boundary_warnings`: pairs listed there cross a task boundary. In mixed-task inventories, these consistently degrade at least one task and should not be your first merge candidates.

---

## 2. Follow-up run

**Goal:** Re-evaluate the inventory after adding, removing, or retraining adapters.

### What changes, what stays the same

- **Re-run `audit-adapter`** only for adapters that changed (retrained, updated weights, or newly added).
- **Re-run `merge-audit`** for any pair involving a changed adapter. Pairs between two unchanged adapters do not need reanalysis.
- **Always regenerate** `inventory/summary.json` and `inventory/neighborhoods.json` — these are derived from the current artifact state and are cheap to rebuild.

### Adding a new adapter

```bash
# Audit the new adapter
gradience audit-adapter \
  --peft-dir ./adapters/new_adapter \
  --eval-dataset <dataset> \
  --metric-name <metric> \
  --adapter-score <score> \
  --base-score <base-score> \
  --higher-is-better \
  --out qa/new_adapter_qa.json

# Run reports against all existing adapters
for name in adapter_a adapter_b adapter_c; do
  gradience merge-audit \
    --adapter-a ./adapters/new_adapter \
    --adapter-b "./adapters/$name" \
    --source-a-qa qa/new_adapter_qa.json \
    --source-b-qa "qa/${name}_qa.json" \
    --emit-report "reports/new_adapter_vs_${name}.json"
done

# Regenerate inventory
gradience summarize-inventory \
  --qa-dir qa/ --report-dir reports/ \
  --emit-report inventory/summary.json

gradience suggest-neighborhoods \
  --qa-dir qa/ --report-dir reports/ \
  --emit-report inventory/neighborhoods.json
```

### Retraining an existing adapter

Same as adding a new adapter: overwrite the QA artifact and re-run all pairs involving it. Delete the stale pair reports first so they do not contaminate the new inventory:

```bash
rm reports/retrained_adapter_vs_*.json
rm reports/*_vs_retrained_adapter.json
# then re-run merge-audit for each affected pair
```

### Removing an adapter

Delete its QA artifact and all pair reports that reference it, then regenerate the inventory.

```bash
rm qa/removed_adapter_qa.json
rm reports/removed_adapter_vs_*.json
rm reports/*_vs_removed_adapter.json

gradience summarize-inventory \
  --qa-dir qa/ --report-dir reports/ \
  --emit-report inventory/summary.json

gradience suggest-neighborhoods \
  --qa-dir qa/ --report-dir reports/ \
  --emit-report inventory/neighborhoods.json
```

### Strict-input mode

If you want the aggregation to fail immediately on any malformed artifact (rather than skip-with-warning), add `--strict-input`:

```bash
gradience summarize-inventory \
  --qa-dir qa/ --report-dir reports/ \
  --strict-input \
  --emit-report inventory/summary.json
```

Use `--strict-input` in CI pipelines where silent skips would mask a bad artifact.

---

## 3. Compare to previous

**Goal:** Diff two inventory runs to detect regressions, improvements, or risk escalations over time.

### Archive the baseline

Before any run that modifies the pool, copy the current inventory:

```bash
cp inventory/summary.json inventory/summary_$(date +%Y%m%d).json
cp inventory/neighborhoods.json inventory/neighborhoods_$(date +%Y%m%d).json
```

Or version with a label:

```bash
cp inventory/summary.json inventory/summary_before_adapter_d.json
```

### Diff summary counts

The `inventory_summary/v1` schema is all flat count maps — straightforward to diff:

```bash
# Quick field-by-field comparison
python3 - <<'EOF'
import json, sys

a = json.load(open("inventory/summary_before_adapter_d.json"))
b = json.load(open("inventory/summary.json"))

keys = [
    "adapter_status_counts", "adapter_flag_counts",
    "pair_risk_counts", "dominant_issue_counts",
    "recommended_strategy_counts",
]

for section in keys:
    print(f"\n{section}")
    all_keys = set(a.get(section, {}).keys()) | set(b.get(section, {}).keys())
    for k in sorted(all_keys):
        before = a.get(section, {}).get(k, 0)
        after  = b.get(section, {}).get(k, 0)
        delta  = after - before
        marker = " +" if delta > 0 else (" -" if delta < 0 else "  ")
        print(f"  {marker} {k}: {before} -> {after}")

print(f"\nstrict_qa_block_candidates: "
      f"{a.get('strict_qa_block_candidates', 0)} -> "
      f"{b.get('strict_qa_block_candidates', 0)}")
EOF
```

Sample output when one high-risk pair resolved and one new low-risk pair was added:

```
pair_risk_counts
    high: 1 -> 0
    low:  2 -> 3

dominant_issue_counts
    subspace_conflict: 1 -> 0
    none:              2 -> 3
```

### Diff individual pair reports

For a specific pair that changed risk level, compare the raw reports:

```bash
diff <(python3 -m json.tool reports/before/adapter_a_vs_adapter_b.json) \
     <(python3 -m json.tool reports/adapter_a_vs_adapter_b.json)
```

Fields to watch across runs:

| Field | Escalation to flag |
|-------|-------------------|
| `pair_risk` | `low` → `high` after retraining |
| `compatibility_score` | Drop of >0.1 |
| `dominant_issue` | New `subspace_conflict` or `norm_imbalance` |
| `eligibility_status` (adapter A or B) | `eligible` → `flagged_weak` |

### Track risk trends over a series of runs

If you run preflight on every training checkpoint:

```bash
# Write each run's summary to a timestamped file
gradience summarize-inventory \
  --qa-dir qa/ --report-dir reports/ \
  --emit-report "inventory/summary_$(date +%Y%m%dT%H%M).json"

# Tabulate high-risk counts over time
python3 - <<'EOF'
import json, glob, os

files = sorted(glob.glob("inventory/summary_*.json"))
print(f"{'run':<30} {'high_risk':>9} {'flagged_weak':>12} {'strict_blocks':>13}")
for f in files:
    data = json.load(open(f))
    name = os.path.basename(f)
    high = data.get("pair_risk_counts", {}).get("high", 0)
    weak = data.get("adapter_status_counts", {}).get("flagged_weak", 0)
    blocks = data.get("strict_qa_block_candidates", 0)
    print(f"{name:<30} {high:>9} {weak:>12} {blocks:>13}")
EOF
```

---

## 4. Portfolio triage across inventories

**Goal:** Prioritize work when managing multiple separate adapter pools simultaneously.

### When this applies

You have multiple inventories — different projects, model families, task groups, or teams — and need to decide where to direct attention. This is a cross-inventory view, not within one pool.

### Step 1 — Emit a summary per inventory

```bash
# Run from a workspace root that contains multiple inventory directories
for inv in project_a project_b project_c; do
  gradience summarize-inventory \
    --qa-dir "${inv}/qa/" \
    --report-dir "${inv}/reports/" \
    --emit-report "${inv}/inventory/summary.json"
done
```

### Step 2 — Aggregate across inventories

The `inventory_summary/v1` schema is identical for every inventory, so aggregation is a direct count merge:

```bash
python3 - <<'EOF'
import json, glob, os

inventories = sorted(glob.glob("*/inventory/summary.json"))

header = f"{'inventory':<24} {'adapters':>8} {'pairs':>6} {'high_risk':>9} {'flagged':>8} {'strict_blk':>11}"
print(header)
print("-" * len(header))

totals = {"adapters": 0, "pairs": 0, "high": 0, "weak": 0, "blocks": 0}

for path in inventories:
    data = json.load(open(path))
    name = path.split("/")[0]
    n_adapters = data["sources"]["qa_artifact_count"]
    n_pairs    = data["sources"]["merge_report_count"]
    high       = data.get("pair_risk_counts", {}).get("high", 0)
    weak       = data.get("adapter_status_counts", {}).get("flagged_weak", 0)
    blocks     = data.get("strict_qa_block_candidates", 0)
    print(f"{name:<24} {n_adapters:>8} {n_pairs:>6} {high:>9} {weak:>8} {blocks:>11}")
    totals["adapters"] += n_adapters; totals["pairs"] += n_pairs
    totals["high"] += high; totals["weak"] += weak; totals["blocks"] += blocks

print("-" * len(header))
print(f"{'TOTAL':<24} {totals['adapters']:>8} {totals['pairs']:>6} "
      f"{totals['high']:>9} {totals['weak']:>8} {totals['blocks']:>11}")
EOF
```

Sample output:

```
inventory                adapters  pairs  high_risk  flagged  strict_blk
--------------------------------------------------------------------------
project_a                      10      8          1        1           3
project_b                       4      6          3        2           4
project_c                       6      6          0        0           0
--------------------------------------------------------------------------
TOTAL                          20     20          4        3           7
```

### Step 3 — Prioritize by risk signal

Read the table from highest attention to lowest:

1. **High `high_risk` count** — pairs with `subspace_conflict` or `norm_imbalance` that need per-layer strategy selection. Do not run linear merges on these without reviewing the individual reports.
2. **High `strict_qa_block_candidates`** — adapters or pairs that would be blocked under `--strict-qa`. These have weak or missing behavioral evidence. Either run evaluations to produce proper QA artifacts, or exclude the adapters from merge candidates.
3. **High `flagged_weak`** — source adapters that underperform the base model. Merging with these is structurally possible but behaviorally risky. Typically retrain or exclude.
4. **High `adapter_flag_counts`** — structural flags (`low_utilization`, `high_rank_waste`) that don't necessarily indicate behavioral failure but suggest inefficient adapter design.

`project_c` in the example above needs no immediate action. `project_b` has 3 high-risk pairs and 2 flagged adapters — start there.

### Step 4 — Drill into problem inventories

For each inventory that shows elevated risk, inspect the individual pair reports to find which specific pairs are driving the counts:

```bash
# Find all high-risk pair reports in project_b
python3 - <<'EOF'
import json, glob

for path in sorted(glob.glob("project_b/reports/*.json")):
    data = json.load(open(path))
    if data.get("pair_risk") == "high":
        a = data["adapter_a"]["path"].split("/")[-1]
        b = data["adapter_b"]["path"].split("/")[-1]
        issue = data.get("dominant_issue", "unknown")
        score = data.get("compatibility_score", 0.0)
        print(f"{a} vs {b}: {issue} (score={score:.3f})")
EOF
```

Then use `suggest-neighborhoods` on the flagged inventory to see if any clusters can still be formed within the safe region:

```bash
gradience suggest-neighborhoods \
  --qa-dir project_b/qa/ \
  --report-dir project_b/reports/ \
  --emit-report project_b/inventory/neighborhoods.json

python -m json.tool project_b/inventory/neighborhoods.json | grep -E '"characterization"|"members"'
```

Groups with `"characterization": "safe neighborhood"` are viable merge candidates. Groups with `"characterization": "caution neighborhood"` and `boundary_warnings` between them should not be merged across group lines without further justification.

### Using `--strict-qa` across inventories

To count how many pairs across all inventories would pass strict QA gating, check the `strict_qa_block_candidates` field from each summary. Any pair where either adapter has `flagged_weak` or `unknown_no_behavioral_eval` status counts as a block candidate.

To enforce gating during actual merge-audit runs:

```bash
gradience merge-audit \
  --adapter-a ./adapters/a \
  --adapter-b ./adapters/b \
  --source-a-qa qa/a_qa.json \
  --source-b-qa qa/b_qa.json \
  --strict-qa \
  --emit-report reports/a_vs_b.json
```

`--strict-qa` requires both QA files to be present and both adapters to be non-weak. If either fails, the command exits with an error and withholds the merge recommendation while still printing structural diagnostics.

---

## Quick reference

| Task | Command |
|------|---------|
| Audit one adapter (with eval) | `gradience audit-adapter --peft-dir DIR --eval-dataset D --metric-name M --adapter-score S --base-score B --higher-is-better --out qa/NAME.json` |
| Audit one adapter (structural only) | `gradience audit-adapter --peft-dir DIR --out qa/NAME.json` |
| Pairwise report | `gradience merge-audit --adapter-a A --adapter-b B --source-a-qa qa/A.json --source-b-qa qa/B.json --emit-report reports/A_vs_B.json` |
| Pairwise report (terminal summary) | add `--qa-report` |
| Regenerate inventory summary | `gradience summarize-inventory --qa-dir qa/ --report-dir reports/ --emit-report inventory/summary.json` |
| Regenerate neighborhoods | `gradience suggest-neighborhoods --qa-dir qa/ --report-dir reports/ --emit-report inventory/neighborhoods.json` |
| Strict-mode pairwise | add `--strict-qa` to `merge-audit` |
| Strict-mode aggregation | add `--strict-input` to `summarize-inventory` |
| Print formatted artifact | `python -m json.tool <file.json>` |
| Check eligibility status | `python -m json.tool qa/NAME.json \| grep -A5 eligibility` |
