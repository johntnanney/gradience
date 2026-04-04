# Canonical Merge Triage Workflow

**Audience:** practitioner, operator  
**Status:** stable (canonical happy path)  
**Purpose:** exact run order for inventory preflight and merge candidate narrowing  
**Canonical for:** the primary product workflow  
**Supersedes:** fragmented workflow references across multiple docs  
**See also:** [`../product_surface.md`](../product_surface.md), [`../playbook.md`](../playbook.md), [`report-interpretation.md`](report-interpretation.md)

This is the validated product flow to answer:

1. What do I run?
2. In what order?
3. Which outputs should I trust?

Use this order:

1. Single-adapter QA
2. Inventory ingest
3. Task-boundary and family classification
4. Pairwise merge audit
5. Inventory summary and action plan
6. Behavioral evaluation only on retained candidates

---

## Stage 1 — Single-Adapter QA

**Command**

```bash
mkdir -p qa reports inventory runs

gradience audit-adapter \
  --peft-dir ./adapters/<adapter_id> \
  --eval-dataset <dataset_name> \
  --metric-name <metric_name> \
  --adapter-score <adapter_score> \
  --base-score <base_score> \
  --out qa/<adapter_id>_qa.json
```

**Input artifact**

- Adapter directory (`adapter_config.json`, adapter weights)
- Evaluation context (dataset, metric, adapter/base score)

**Output artifact**

- `qa/<adapter_id>_qa.json` (`gradience.adapter_qa/v1`)

**Decision rule**

- `eligible` → include in candidate pool
- `uncertain` → include as near-miss/caution
- `flagged_weak` or `unknown_no_behavioral_eval` → exclude from default merge shortlist

**What to do next**

- Repeat for every adapter in inventory, then proceed to Stage 2.

---

## Stage 2 — Inventory Ingest

**Command**

```bash
find ./adapters -mindepth 1 -maxdepth 1 -type d | sort > inventory/adapter_dirs.txt
find ./qa -name '*_qa.json' | sort > inventory/qa_artifacts.txt
```

**Input artifact**

- Adapter directories
- QA artifacts from Stage 1

**Output artifact**

- `inventory/adapter_dirs.txt`
- `inventory/qa_artifacts.txt`

**Decision rule**

- Every adapter intended for pairwise audit must have a QA artifact.
- Missing QA artifacts block that adapter from the canonical path.

**What to do next**

- Confirm ingest lists are complete, then proceed to Stage 3.

---

## Stage 3 — Task-Boundary and Family Classification

**Command**

```bash
python - <<'PY' > inventory/task_labels.tsv
import glob
import json
import os

for path in sorted(glob.glob("qa/*_qa.json")):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    name = data.get("adapter", {}).get("name", os.path.basename(path))
    dataset = data.get("behavioral_summary", {}).get("eval_dataset") or "unknown"
    print(f"{name}\t{dataset}")
PY
```

**Input artifact**

- QA artifacts from Stage 1

**Output artifact**

- `inventory/task_labels.tsv` (adapter-to-task labels used for boundary interpretation)

**Decision rule**

- `unknown` task labels are low-confidence boundary inputs and should be resolved before trusting cross-task interpretations.
- Mixed-task inventories should be expected to produce task-boundary advisories in Stage 4.

**What to do next**

- Proceed to Stage 4 and run pairwise audits with QA context attached.

---

## Stage 4 — Pairwise Merge Audit

**Command**

```bash
adapters=(<adapter_a> <adapter_b> <adapter_c>)  # replace with your adapter IDs

for i in "${!adapters[@]}"; do
  for j in $(seq $((i+1)) $((${#adapters[@]}-1))); do
    a="${adapters[$i]}"
    b="${adapters[$j]}"
    gradience merge-audit \
      --adapter-a "./adapters/${a}" \
      --adapter-b "./adapters/${b}" \
      --source-a-qa "qa/${a}_qa.json" \
      --source-b-qa "qa/${b}_qa.json" \
      --qa-report \
      --emit-report "reports/${a}_vs_${b}.json"
  done
done
```

**Input artifact**

- Adapter directories
- QA artifacts for both sides of each pair

**Output artifact**

- `reports/*_vs_*.json` (`gradience.merge_qa_report/v1`)

**Decision rule**

- Advisory absent + lower structural risk → primary retained candidates
- `task_relationship_advisory` present → cross-task caution zone (not first-line)
- High-risk/conflicting/imbalanced pairs → deprioritize unless explicitly justified

**What to do next**

- Aggregate all QA + pair reports in Stage 5.

---

## Stage 5 — Inventory Summary and Action Plan

**Command**

```bash
gradience summarize-inventory \
  --qa-dir qa/ \
  --report-dir reports/ \
  --emit-report inventory/summary.json \
  --emit-bundle runs/run_001

gradience preflight-report runs/run_001/
```

**Input artifact**

- All QA artifacts
- All pairwise merge reports

**Output artifact**

- `inventory/summary.json` (`gradience.inventory_summary/v1`)
- `runs/run_001/preflight_summary.md`
- `runs/run_001/inventory_action_plan.md`
- `runs/run_001/preflight_report.html`

**Decision rule**

- Trust `inventory_action_plan.md` as the canonical shortlist surface:
  - evaluate-first
  - near-miss
  - exclude/deprioritize

**What to do next**

- Run downstream behavioral evaluation only for retained candidates (Stage 6).

---

## Stage 6 — Behavioral Evaluation on Retained Candidates Only

**Command**

```bash
# 1) Build a retained pair list from the action plan (tab-separated: adapter_a<TAB>adapter_b)
cat runs/run_001/inventory_action_plan.md

# 2) Evaluate only retained pairs with your merge + task evaluator
while IFS=$'\t' read -r a b; do
  echo "Evaluate retained pair: $a vs $b"
  # merge command here (example)
  # your_merge_command --adapter-a "./adapters/$a" --adapter-b "./adapters/$b"
  # behavioral evaluation command here (example)
  # your_eval_command --model ./merged/"${a}_vs_${b}" --dataset <dataset>
done < inventory/retained_pairs.tsv
```

**Input artifact**

- `runs/run_001/inventory_action_plan.md`
- `runs/run_001/preflight_summary.md`

**Output artifact**

- Behavioral merge evaluation results for retained candidates only

**Decision rule**

- Final merge decision is behavioral.
- Preflight narrows candidates; it does not replace outcome evaluation.

**What to do next**

- Promote winning pairs into deployment workflows; keep excluded pairs out of default merge plans.

---

## Output Trust Order

When signals disagree, use this trust order in the canonical path:

1. QA eligibility (`gradience.adapter_qa/v1`)
2. Task-boundary advisory + pair risk (`gradience.merge_qa_report/v1`)
3. Inventory action plan (`gradience.inventory_summary/v1` bundle outputs)
4. Final behavioral evaluation on retained pairs

This keeps structural triage and behavioral adjudication in their validated roles.
