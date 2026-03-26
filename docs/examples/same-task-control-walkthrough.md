# Same-Task Control Walkthrough

This walkthrough shows what Gradience looks like when the inventory is already clean. The workflow is mostly confirmatory — and that is the expected behavior.

## Starting inventory

| Adapter | Task | Rank |
|---------|------|------|
| qnli_r16_s42 | QNLI | 16 |
| qnli_r16_s123 | QNLI | 16 |
| qnli_r8_s42 | QNLI | 8 |

**3 adapters. 1 task. 3 possible pairs.**

---

## Step 1: Source QA

```bash
for adapter in qnli_r16_s42 qnli_r16_s123 qnli_r8_s42; do
  gradience audit --peft-dir ./adapters/$adapter --json > qa/${adapter}.json
done
```

**Result:** All 3 adapters are eligible. No exclusions.

---

## Step 2: Pairwise merge reports

| Pair | Risk | Advisory | Issue |
|------|------|----------|-------|
| qnli_r16_s42 × qnli_r16_s123 | medium | — | high redundancy |
| qnli_r16_s42 × qnli_r8_s42 | medium | — | partial redundancy |
| qnli_r16_s123 × qnli_r8_s42 | medium | — | partial redundancy |

### Advisory behavior

**The advisory is silent on all 3 pairs.** This is expected and correct — all adapters share the same evaluation task. Same-task pairs on small encoder models are broadly safe.

---

## Step 3: Interpretation

This is a same-task safe pool. All 3 pairs are reasonable merge candidates.

| Zone | Pairs | Action |
|------|-------|--------|
| **Same-task safe** | All 3 | Any of these can be evaluated |

---

## Before and after

| Stage | Candidate pairs |
|-------|----------------|
| Starting pool | **3** |
| After source QA | 3 |
| After advisory | 3 (all same-task, advisory silent) |

**No reduction needed.** The inventory was already clean.

---

## When this happens

This is the **lower-value regime** for Gradience. When you already know:
- all adapters are on the same task
- all pass basic quality checks
- the pool is small

...the full preflight workflow confirms what context already suggests. The main value is the explicit QA record and the confidence that no hidden task-boundary risk exists.

**Gradience is most useful on mixed-task inventories.** See the [mixed-task walkthrough](mixed-task-inventory-walkthrough.md) for the flagship use case.

---

## Data source

This walkthrough uses real data. Artifacts are available in two locations:
- `examples/inventory_preflight_same_task_control/` — self-contained example bundle
- `results/task_advisory_round/same_task_qnli_control/` — original execution output
