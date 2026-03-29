# Gradience Playbook

A practical, step-by-step guide for the five workflows practitioners use most. Each section is self-contained: find the one that matches your situation and follow it through.

For the conceptual framework behind these workflows, see [Inventory Preflight Workflow](inventory-preflight.md). For the full CLI reference, see [CLI Reference](cli.md).

---

## 1. Running Your First Inventory

**Situation:** You have 4–10 LoRA adapters and want to know which pairs are worth evaluating.

### What you need

- Adapter directories (each containing `adapter_config.json` and `adapter_model.safetensors`)
- For each adapter: its base model name, the evaluation dataset, the metric, and the adapter's score on that metric
- The base model's score on the same metric (so Gradience can determine whether each adapter actually beats the base)

### The sequence

**Step 1 — Audit each adapter.** This creates a QA artifact per adapter, recording its structural profile and behavioral eligibility.

```bash
mkdir -p qa/ reports/ inventory/

# For each adapter, supply its evaluation context
gradience audit-adapter \
  --peft-dir ./adapters/sentiment_s42 \
  --eval-dataset sst2 \
  --metric-name accuracy \
  --adapter-score 0.91 \
  --base-score 0.85 \
  --out qa/sentiment_s42_qa.json

# Repeat for each adapter in your pool
gradience audit-adapter \
  --peft-dir ./adapters/sentiment_s99 \
  --eval-dataset sst2 \
  --metric-name accuracy \
  --adapter-score 0.89 \
  --base-score 0.85 \
  --out qa/sentiment_s99_qa.json

gradience audit-adapter \
  --peft-dir ./adapters/nli_s42 \
  --eval-dataset qnli \
  --metric-name accuracy \
  --adapter-score 0.88 \
  --base-score 0.51 \
  --out qa/nli_s42_qa.json
```

Each command prints a terminal summary and writes a `gradience.adapter_qa/v1` JSON artifact. Check the `eligibility.status` field — adapters marked `eligible` are good to proceed; those marked `flagged_weak` or `unknown_no_behavioral_eval` will be flagged in downstream reports.

**Step 2 — Run pairwise merge reports.** For every pair you want to assess, pass both adapter directories and their QA artifacts:

```bash
gradience merge-audit \
  --adapter-a ./adapters/sentiment_s42 \
  --adapter-b ./adapters/sentiment_s99 \
  --source-a-qa qa/sentiment_s42_qa.json \
  --source-b-qa qa/sentiment_s99_qa.json \
  --qa-report \
  --emit-report reports/sentiment_s42_vs_s99.json
```

The `--qa-report` flag prints a 4-section terminal summary. The `--emit-report` flag writes the machine-readable `gradience.merge_qa_report/v1` JSON. Each report contains a `pair_risk` level (low/medium/high), a `dominant_issue`, a `recommended_strategy`, and — when the two adapters target different tasks — a `task_relationship_advisory`.

For larger pools, script the pairwise loop:

```bash
adapters=(sentiment_s42 sentiment_s99 nli_s42)
for i in "${!adapters[@]}"; do
  for j in $(seq $((i+1)) $((${#adapters[@]}-1))); do
    a="${adapters[$i]}"
    b="${adapters[$j]}"
    gradience merge-audit \
      --adapter-a ./adapters/$a --adapter-b ./adapters/$b \
      --source-a-qa qa/${a}_qa.json --source-b-qa qa/${b}_qa.json \
      --emit-report reports/${a}_vs_${b}.json
  done
done
```

**Step 3 — Summarize the inventory and generate the run bundle.**

```bash
gradience summarize-inventory \
  --qa-dir qa/ \
  --report-dir reports/ \
  --emit-report inventory/summary.json \
  --emit-bundle inventory/run_001
```

This produces:

- A terminal summary with counts and the action plan
- `inventory/summary.json` — the machine-readable inventory summary
- `inventory/run_001/` — a run bundle directory containing:
  - `preflight_summary.md` — start here for the human-readable overview
  - `preflight_summary.json` — for scripting and downstream tools
  - `run_manifest.json` — provenance and metadata
  - `inventory_action_plan.md` — the structured action plan
  - A `latest/` symlink for run-to-run tracking

**Step 4 — Generate the HTML report.**

```bash
gradience preflight-report inventory/run_001/
```

This reads the run bundle and renders a single-page HTML report at `inventory/run_001/preflight_report.html`. Open it in any browser — it has no external dependencies.

**Step 5 — Act on the results.** Read the action plan. It tells you:

- Which pairs to evaluate first (same-task, structurally clean)
- Which adapters to exclude or deprioritize (weak evidence, failed QA)
- Which pairs cross a task boundary (proceed with caution)
- How many candidates were eliminated (typically 65–90% in mixed-task pools)

---

## 2. Using the Evidence Bootstrap

**Situation:** You have downloaded adapters from a hub (HuggingFace, etc.) and do not have evaluation scores for them. You need behavioral evidence to feed Gradience's QA pipeline, but you do not want to invest in a full evaluation suite yet.

### What the evidence bootstrap is

A lightweight CPU evaluation pass that runs each adapter on a small sample (300–500 examples) of a relevant dataset. The goal is not precise measurement — it is producing enough behavioral signal to distinguish "plausibly working" from "clearly broken" adapters. This distinction is what Gradience's evidence gate needs.

### When to use it

- You pulled adapters from a public hub and have no eval scores
- You want to screen a batch of adapters before investing in full evaluation
- You need to fill in the `--adapter-score` and `--base-score` fields for `audit-adapter`

### The procedure

**Step 1 — Choose a dataset and metric.** Pick the dataset your adapters were (probably) trained on, or the closest proxy. For classification tasks, accuracy on the validation split is the simplest choice.

**Step 2 — Evaluate the base model.** Run your base model on a sample to get the baseline:

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset
import torch

model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
model.eval()

ds = load_dataset("glue", "sst2", split="validation[:500]")

correct = 0
for ex in ds:
    inputs = tokenizer(ex["sentence"], return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        logits = model(**inputs).logits
    pred = logits.argmax(dim=-1).item()
    if pred == ex["label"]:
        correct += 1

base_score = correct / len(ds)
print(f"Base score: {base_score:.3f}")
```

**Step 3 — Evaluate each adapter.** Load the adapter on top of the base model and repeat:

```python
from peft import PeftModel

adapter_model = PeftModel.from_pretrained(model, "./adapters/sentiment_s42")
adapter_model.eval()

correct = 0
for ex in ds:
    inputs = tokenizer(ex["sentence"], return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        logits = adapter_model(**inputs).logits
    pred = logits.argmax(dim=-1).item()
    if pred == ex["label"]:
        correct += 1

adapter_score = correct / len(ds)
print(f"Adapter score: {adapter_score:.3f}")
```

**Step 4 — Feed the scores into Gradience.**

```bash
gradience audit-adapter \
  --peft-dir ./adapters/sentiment_s42 \
  --eval-dataset sst2 \
  --metric-name accuracy \
  --adapter-score 0.91 \
  --base-score 0.85 \
  --out qa/sentiment_s42_qa.json
```

### What to watch for

- **Delta matters more than absolute score.** An adapter with accuracy 0.62 on a base of 0.60 is barely beating the base — it will be classified as `eligible` but marginal. An adapter at 0.55 on a base of 0.60 will be `flagged_weak`.
- **Sample size affects precision, not the decision.** 500 examples are enough to distinguish "working" from "broken." If you need precise deltas (e.g., to compare two adapters that both score ~0.88), use a larger sample or the full validation split.
- **Consistency across adapters matters.** Use the same sample, same metric, and same base score for all adapters in the same inventory. The QA pipeline compares adapters against each other — inconsistent evaluation conditions will produce misleading relative judgments.

### What the evidence bootstrap taught us (from field trials)

In 5 field trials across 3 backbones, the evidence bootstrap was the single most impactful step. Without it (Pilot 1), the pipeline produced nothing useful — all adapters were `unknown_no_behavioral_eval` and the action plan was empty. With it, even a quick 500-sample CPU pass was sufficient to correctly classify adapters across the full range of outcomes: genuine failures, misleading evals, marginal passes, ambiguous ties, and strong performers.

The one caveat: adapters that barely beat the base (delta +0.01 to +0.06) pass as `eligible` but contribute little to merges. The evidence gate is well-calibrated except at this margin.

---

## 3. Reading the HTML Report

**Situation:** You have generated a preflight HTML report and want to understand what it is telling you.

### Generating the report

```bash
# From an existing run bundle
gradience preflight-report ./runs/run_001/

# Custom output path
gradience preflight-report ./runs/run_001/ -o ./reports/my_report.html

# Custom title
gradience preflight-report ./runs/run_001/ --title "Sentiment Inventory Preflight"
```

### What the report contains

The HTML report renders the preflight run bundle as a single self-contained page. It has no external CSS or JavaScript dependencies — you can email it, archive it, or open it offline.

**Header section.** Run metadata: inventory ID, run ID, timestamp, adapter count, pair count.

**Inventory overview.** The adapter-level summary: how many adapters by eligibility status, how many have behavioral evidence, any structural flags.

**Pair matrix.** The pairwise results table, showing for each pair: risk level, dominant issue, recommended strategy, and whether a task-relationship advisory is present. Pairs are grouped by risk level (high → medium → low) so the most concerning pairs appear first.

**Action plan.** The same structured action plan that appears in the terminal output, rendered with formatting:

- **Evaluate first**: your best merge candidates (same-task, structurally clean)
- **Near-miss candidates**: pairs that are structurally plausible but excluded because one source has weak or missing evidence. These are your second-tier candidates — worth revisiting if you can strengthen the weak source.
- **Exclude/deprioritize**: adapters or pairs that QA flagged
- **Cross-task caution zone**: pairs that cross a task boundary

**Reduction summary.** The final line: how many pairs the preflight eliminated from the candidate set.

### How to read the signals

The three most important fields in the pair table:

1. **`task_relationship_advisory`** — If present, the pair crosses a task boundary. On small encoder models, this is the single most important signal: same-task pairs are broadly safe; cross-task pairs are where failures live.

2. **`pair_risk`** — The structural risk level. Low-risk same-task pairs are your safest candidates. Medium-risk same-task pairs typically involve partial redundancy (harmless but less valuable). High-risk pairs have structural issues (norm imbalance, subspace conflict) that may need specialized merge strategies.

3. **`recommended_strategy`** — What Gradience suggests for the merge: `linear` (low risk, standard merge), `norm_equalized` (medium risk, rebalance norms), or `audit_aware` (high risk, use per-layer strategy decisions).

### What the report does NOT tell you

- Whether a merge will actually improve your metric (you still need to evaluate)
- How much a cross-task merge will degrade (severity is not portable across backbones)
- Anything about decoder-only or large-scale models

---

## 4. Interpreting Retained, Near-Miss, and Excluded Pairs

**Situation:** The action plan has categorized your pairs into groups. You want to understand what each category means and what to do with it.

### Retained pairs (evaluate first)

These are pairs where both adapters are eligible (behavioral evidence present, beats base model) and the pair has low or medium structural risk with no task-boundary advisory. These are your best merge candidates.

**What to do:** Evaluate these first. They have the highest probability of producing a useful merge. In field trials, retained same-task pairs either improved over both sources (+0.028, +0.006) or degraded modestly (-0.006 to -0.088). Average delta: -0.024.

### Near-miss candidates

These are pairs that look structurally plausible — same task, low-to-medium risk, no task-boundary advisory — but one of the two sources has weak or missing behavioral evidence. The pair was excluded from the "evaluate first" list because of the evidence gap, not because of a structural problem.

**What to do:** These are your second-tier candidates. In field trials across 3 backbones and 3 task families, near-miss pairs degraded comparably to retained pairs and 5× less than cross-task controls. Average delta: -0.006 (near-miss) vs -0.024 (retained) vs -0.047 (cross-task control).

Two sub-cases:

- **Weak source is barely below the threshold** (delta -0.002 to -0.004 vs base): The merge will likely be indistinguishable from a retained pair. Consider strengthening the weak source's evidence (more training, better hyperparameters) and re-running preflight.
- **Weak source is deeply weak** (delta -0.10 or worse vs base): More variance. The merge might still be acceptable, but it is less predictable.

### Excluded / deprioritized

These fall into two sub-categories:

- **Weak-evidence adapters**: Sources marked `flagged_weak` or `unknown_no_behavioral_eval`. They appear in the exclusion list because including them in merge recommendations without evidence would compromise the pipeline's reliability.
- **Cross-task pairs**: Pairs where the task-relationship advisory fired. On small encoder models, same-task pairs are broadly safe; cross-task pairs are where the meaningful failure modes live.

**What to do with excluded adapters:** If you believe an adapter is actually good despite its weak evidence, run a more thorough evaluation and re-supply the scores. The evidence bootstrap (Section 2 above) is the fastest path.

**What to do with cross-task pairs:** Treat these as a caution zone, not a prohibition. Some cross-task merges are mild (2pp degradation); others are catastrophic (40pp+). The advisory catches the boundary but does not grade severity within it. Proceed only if you have a specific reason and can afford the evaluation cost.

---

## 5. Using the Portfolio View Across Inventories

**Situation:** You have run preflight on multiple inventories (different adapter pools, different runs, before-and-after comparisons) and want to compare across them.

### Run-to-run comparison (same inventory, evolving)

When you re-run preflight after changing the adapter pool (adding, removing, or re-evaluating adapters), use `--previous-run` to get a structured comparison:

```bash
# First run
gradience summarize-inventory \
  --qa-dir qa/ --report-dir reports/ \
  --emit-bundle runs/run_001

# ... make changes to adapters or evaluations ...

# Second run, with comparison
gradience summarize-inventory \
  --qa-dir qa_v2/ --report-dir reports_v2/ \
  --emit-bundle runs/run_002 \
  --previous-run runs/run_001
```

The bundle for `run_002` will include a `compare_to_previous.md` that shows what changed: new adapters, removed adapters, risk-level shifts, strategy changes.

If you omit `--previous-run`, Gradience auto-discovers the most recent sibling run via the `latest/` symlink in the parent directory.

### Cross-inventory comparison (batch summary)

When you have multiple inventories (e.g., one per backbone, one per task family, one per project), use `batch-summary` to see them side by side:

```bash
# Assuming runs/ contains run_001/, run_002/, run_003/
gradience batch-summary --run-dir runs/
```

This reads all `preflight_summary.json` files from the subdirectories and produces a comparison table showing key metrics across runs: adapter count, pair count, reduction percentage, risk distribution, evidence coverage.

### Practical patterns

**Before-and-after evidence bootstrap.** Run preflight without evidence (all adapters `unknown_no_behavioral_eval`), then again with evidence. The comparison shows the impact of behavioral grounding on the candidate set.

**Backbone comparison.** Run the same adapter set on multiple backbones (DistilBERT, RoBERTa, BERT). Batch summary reveals which backbone produces the cleanest merge candidates.

**Inventory growth tracking.** As you add adapters over time, each preflight run captures the state. The run-to-run comparison tracks how the candidate set evolves as the inventory grows.

---

## Quick reference: the full command sequence

```bash
# 1. Audit adapters
for d in ./adapters/*/; do
  name=$(basename "$d")
  gradience audit-adapter --peft-dir "$d" \
    --eval-dataset $DATASET --metric-name accuracy \
    --adapter-score $SCORE --base-score $BASE \
    --out qa/${name}_qa.json
done

# 2. Pairwise merge reports
# (use the scripted loop from Section 1, or run manually)

# 3. Summarize + bundle
gradience summarize-inventory \
  --qa-dir qa/ --report-dir reports/ \
  --emit-report inventory/summary.json \
  --emit-bundle runs/run_001

# 4. HTML report
gradience preflight-report runs/run_001/

# 5. (Optional) Batch comparison
gradience batch-summary --run-dir runs/
```
