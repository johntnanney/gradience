# Example Gallery

Five curated scenarios covering the situations practitioners encounter most often. Each example describes what the inventory looks like, what happens when you run preflight, and what the action plan tells you to do. They are ordered from simplest to most subtle.

For command-level detail, see the [Playbook](playbook.md). For full walkthroughs with pair matrices and terminal output, see the [Mixed-Task Walkthrough](examples/mixed-task-inventory-walkthrough.md) and [Same-Task Control Walkthrough](examples/same-task-control-walkthrough.md).

---

## 1. Same-Task Control

**What it is:** 3–4 adapters, all trained on the same task, all with behavioral evidence showing they beat the base model. This is the simplest and most common scenario.

**Inventory shape:**

| Adapter | Task | Eligibility |
|---------|------|-------------|
| qnli_r16_s42 | QNLI | eligible |
| qnli_r16_s123 | QNLI | eligible |
| qnli_r8_s42 | QNLI | eligible |

**What preflight does:**

- All adapters pass QA. No exclusions.
- All pairs are same-task. The task-relationship advisory is silent on every pair.
- Pair-risk is medium (typical for same-task pairs showing partial or high redundancy).
- The action plan lists all 3 pairs in the "evaluate first" section.

**What this tells you:** The inventory is clean. Preflight confirms what you already suspected — these are all reasonable merge candidates. The workflow is confirmatory, and that confirmation is useful: it means no hidden task-boundary risk exists.

**Expected outcome:** Candidate set is not reduced (there is nothing to remove). All pairs are retained.

**Full walkthrough:** [Same-Task Control Walkthrough](examples/same-task-control-walkthrough.md)

---

## 2. Mixed-Task Inventory

**What it is:** 5–8 adapters drawn from 2–4 different tasks. This is the scenario where Gradience provides the most value, because the pair matrix contains a mix of safe same-task pairs and risky cross-task pairs that look similar on structural metrics alone.

**Inventory shape:**

| Adapter | Task | Eligibility |
|---------|------|-------------|
| sst2_r16_s42 | SST-2 | eligible |
| sst2_r16_s123 | SST-2 | eligible |
| qnli_r16_s42 | QNLI | eligible |
| qnli_r16_s123 | QNLI | eligible |
| mnli_s42 | MNLI | eligible |
| rte_s42 | RTE | eligible |

**What preflight does:**

- All 6 adapters pass QA.
- 15 possible pairs. Of these, 2 are same-task (sst2×sst2, qnli×qnli). The remaining 13 are cross-task.
- The task-relationship advisory fires on all 13 cross-task pairs.
- The action plan retains the 2 same-task pairs in "evaluate first" and moves the 13 cross-task pairs to the caution zone.
- **Candidate reduction: 87% (15 → 2).**

**What this tells you:** Task identity is the dominant signal. Without preflight, you might have evaluated all 15 pairs. With it, you know that 13 of them cross a task boundary and should be deprioritized. The 2 same-task pairs are your best candidates.

**Expected outcome:** 65–90% candidate reduction, depending on the task composition. The higher the task diversity, the more pairs are cross-task, and the larger the reduction.

**Full walkthrough:** [Mixed-Task Inventory Walkthrough](examples/mixed-task-inventory-walkthrough.md)

---

## 3. Large Mixed-Task Inventory

**What it is:** 8–12 adapters from 3+ task families, producing 28–66 candidate pairs. This is where neighborhoods become useful for visual organization in addition to the pair-level analysis.

**Inventory shape (example with 8 adapters, 4 tasks):**

| Adapter | Task | Eligibility |
|---------|------|-------------|
| sst2_a | SST-2 | eligible |
| sst2_b | SST-2 | eligible |
| sst2_c | SST-2 | eligible |
| qnli_a | QNLI | eligible |
| qnli_b | QNLI | eligible |
| mnli_a | MNLI | eligible |
| rte_a | RTE | eligible |
| rte_b | RTE | eligible |

**What preflight does:**

- 28 possible pairs. 5 are same-task (3 SST-2 pairs, 1 QNLI pair, 1 RTE pair). 23 are cross-task.
- The advisory fires on all 23 cross-task pairs.
- Neighborhoods partition the pair matrix into 4 same-task groups and a cross-task boundary zone.
- **Candidate reduction: ~82% (28 → 5).**

**What the action plan looks like:**

- "Evaluate first" lists 5 same-task pairs, each with its risk level and recommended strategy.
- "Cross-task caution zone" lists the 4 task-boundary crossings (SST-2↔QNLI, SST-2↔MNLI, SST-2↔RTE, QNLI↔MNLI, etc.).
- The reduction summary shows the narrowing from 28 to 5.

**When to add neighborhoods:**

```bash
gradience suggest-neighborhoods \
  --qa-dir qa/ --report-dir reports/ \
  --emit-report inventory/neighborhoods.json
```

Neighborhoods are most useful at 6+ adapters, where the pair matrix becomes dense enough that visual grouping aids interpretation. Below 6 adapters, the pair table itself is readable enough.

**Fixture inventories:** The `examples/inventories/` directory contains fixture inventories for testing neighborhood behavior at various scales, including `inventory_large_realistic` (12 adapters, 3 tasks).

---

## 4. Weak-Evidence Inventory

**What it is:** An inventory where some adapters lack behavioral evidence or underperform the base model. This is the common real-world case when you have pulled adapters from a public hub and have not yet evaluated all of them.

**Inventory shape:**

| Adapter | Task | Score | Base | Delta | Eligibility |
|---------|------|-------|------|-------|-------------|
| hate_tg_base | hate | 0.514 | 0.502 | +0.012 | eligible (marginal) |
| hate_aviator | hate | 0.498 | 0.502 | -0.004 | flagged_weak |
| hate_hatexplain | hate | 0.588 | 0.502 | +0.086 | eligible |
| emotion_tg_base | emotion | 0.752 | 0.286 | +0.466 | eligible |
| emotion_fabriceyhc | emotion | 0.204 | 0.286 | -0.082 | eligible (very low) |
| emotion_hatexplain | emotion | 0.136 | 0.286 | -0.150 | flagged_weak |

**What preflight does:**

- 2 adapters are `flagged_weak`. They are listed in the "exclude/deprioritize" section.
- The hate_tg_base adapter is technically `eligible` but barely beats the base (+0.012). It passes the gate but is a marginal contributor.
- Pairs involving `flagged_weak` adapters are candidate near-miss pairs (not excluded outright — see Section 5 below).
- The action plan separates fully eligible pairs from evidence-constrained pairs.

**What this tells you:** The evidence gate is doing its job. Without QA screening, you might merge a `flagged_weak` adapter (one that actually performs worse than the base model) with a strong adapter and wonder why the merge degraded. The QA layer catches this before pairwise analysis begins.

**What to do:** For `flagged_weak` adapters you believe might be salvageable: run the evidence bootstrap with a larger sample, try different hyperparameters, or verify the adapter was loaded correctly (wrong label mapping is a common cause of below-base performance on hub adapters). Then re-run preflight with updated scores.

**Fixture inventory:** `examples/inventories/inventory_with_weak_sources/` contains a fixture with weak-source patterns for testing.

---

## 5. Near-Miss Inventory

**What it is:** An inventory where some pairs are structurally plausible (same task, low-to-medium risk, no task-boundary advisory) but excluded from the "evaluate first" list because one source has weak or missing evidence. These are the near-miss candidates — the second tier after retained pairs.

**Inventory shape:**

| Adapter | Task | Eligibility | Notes |
|---------|------|-------------|-------|
| irony_JB173 | irony | eligible | delta +0.202 |
| irony_vaariis | irony | eligible | delta +0.060 |
| irony_neibla | irony | eligible | delta +0.068 |
| irony_phailyoor | irony | flagged_weak | delta -0.004 |

**What preflight does:**

- 3 eligible adapters, 1 `flagged_weak`.
- 6 possible pairs. 3 involve only eligible adapters → retained. 3 involve the `flagged_weak` adapter → near-miss candidates.
- The action plan lists the 3 retained pairs in "evaluate first" and the 3 near-miss pairs in a separate "near-miss candidates" section.

**What the action plan tells you about near-miss pairs:**

Each near-miss entry identifies which source is evidence-constrained and what the pair's structural risk level is. The section heading makes clear that these are structurally plausible pairs excluded only because of the evidence gap — not because of a structural problem.

**Field trial evidence:** Near-miss pairs were validated across 3 backbones (DistilBERT, BERT, RoBERTa) and 3 task families (irony, hate, ag_news). Key findings:

- Near-miss average delta: -0.006 (comparable to retained pairs at -0.024)
- Cross-task control average delta: -0.047 (5× worse)
- Weak-source severity modulates the outcome: adapters that barely miss the evidence threshold (delta -0.002 to -0.004 vs base) produce merges indistinguishable from retained pairs; deeply weak adapters (delta -0.150) introduce more variance.

**What to do with near-miss pairs:** Treat them as a structured second tier. If your retained pairs are few, near-miss pairs expand the candidate set with a known risk profile. If the weak source is barely below threshold, consider strengthening its evidence (re-evaluate with more data) and promoting the pair to retained status on the next preflight run.

---

## Choosing the right example

| Your situation | Start here |
|---------------|-----------|
| All adapters are same-task and well-evidenced | [Same-Task Control](#1-same-task-control) |
| Mixed tasks, all adapters well-evidenced | [Mixed-Task Inventory](#2-mixed-task-inventory) |
| Large pool, multiple task families | [Large Mixed-Task Inventory](#3-large-mixed-task-inventory) |
| Some adapters have weak or missing evidence | [Weak-Evidence Inventory](#4-weak-evidence-inventory) |
| Structurally good pairs excluded by the evidence gate | [Near-Miss Inventory](#5-near-miss-inventory) |
| You want the detailed pair-by-pair walkthrough | [Mixed-Task Walkthrough](examples/mixed-task-inventory-walkthrough.md) |
