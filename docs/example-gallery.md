# Example Gallery

Six canonical scenarios covering the situations practitioners encounter most often. Each example has a one-sentence purpose, the key outputs to inspect, and what it demonstrates about the preflight workflow. Ordered from simplest to most subtle.

For command-level detail, see the [Playbook](playbook.md). For full walkthroughs with pair matrices and terminal output, see the [Mixed-Task Walkthrough](examples/mixed-task-inventory-walkthrough.md) and [Same-Task Control Walkthrough](examples/same-task-control-walkthrough.md).

---

## 1. Same-Task Control

**Purpose:** Confirm that a pool of same-task adapters is clean and all pairs are reasonable merge candidates.

**Setup:** 3 QNLI adapters (rank 16 × 2 seeds + rank 8), all eligible.

| Adapter | Task | Eligibility |
|---------|------|-------------|
| qnli_r16_s42 | QNLI | eligible |
| qnli_r16_s123 | QNLI | eligible |
| qnli_r8_s42 | QNLI | eligible |

**Key outputs to open:**

- `examples/inventory_preflight_same_task_control/inventory/` — preflight summary and action plan
- `examples/inventory_preflight_same_task_control/reports/` — 3 pair reports, all medium risk
- `examples/inventories/same_task_control_example.md` — annotated walkthrough

**What it demonstrates:** The confirmatory workflow. Advisory is silent on all 3 pairs. Candidate set is not reduced because there is nothing to remove. The value is the explicit QA record — confirmation that no hidden task-boundary risk exists.

**Reduction:** 3 → 3 (none needed).

---

## 2. Mixed-Task Inventory

**Purpose:** Show how task-boundary detection reduces a mixed-task candidate set by 87%.

**Setup:** 6 adapters across 4 tasks (SST-2, QNLI, MNLI, RTE), all eligible. 15 possible pairs.

| Adapter | Task | Eligibility |
|---------|------|-------------|
| sst2_r16_s42 | SST-2 | eligible |
| sst2_r16_s123 | SST-2 | eligible |
| qnli_r16_s42 | QNLI | eligible |
| qnli_r16_s123 | QNLI | eligible |
| mnli_s42 | MNLI | eligible |
| rte_s42 | RTE | eligible |

**Key outputs to open:**

- `examples/inventory_preflight_mixed_task/inventory/` — summary showing 15 → 2 reduction
- `examples/inventory_preflight_mixed_task/reports/` — 15 pair reports (compare same-task vs cross-task advisories)
- `examples/inventories/mixed_task_preflight_example.md` — annotated walkthrough with commands

**What it demonstrates:** Task identity as the dominant signal. Without preflight, 11 of 15 pairs look structurally plausible (medium risk). The advisory separates the 2 same-task safe pairs from 13 cross-task caution pairs that structural metrics alone cannot distinguish.

**Reduction:** 15 → 2 (87%).

---

## 3. Large Mixed-Task Inventory

**Purpose:** Show how neighborhoods organize a dense pair matrix at scale.

**Setup:** 8–12 adapters from 3+ task families, producing 28–66 candidate pairs. This is where the pair table alone becomes hard to read and neighborhoods add value.

**Key outputs to open:**

- `examples/inventories/inventory_large_realistic/` — fixture with 12 adapters, 3 tasks
- `examples/inventories/inventory_large_realistic/expected_notes.md` — expected grouping behavior

**What it demonstrates:** Neighborhoods partition the pair matrix into same-task clusters and a cross-task boundary zone. The action plan's "evaluate first" section lists only the within-cluster pairs. At 6+ adapters, visual grouping aids interpretation beyond what the flat pair table provides.

**Reduction:** ~82% (28 → 5 in the 8-adapter, 4-task case).

**When to add neighborhoods:**

```bash
gradience suggest-neighborhoods \
  --qa-dir qa/ --report-dir reports/ \
  --emit-report inventory/neighborhoods.json
```

---

## 4. Weak-Evidence Inventory

**Purpose:** Show how the evidence gate catches adapters that underperform the base model before they contaminate pairwise analysis.

**Setup:** 6 adapters (hate + emotion), 2 flagged_weak (perform worse than base), 1 marginal (barely beats base).

| Adapter | Task | Delta vs base | Eligibility |
|---------|------|---------------|-------------|
| hate_tg_base | hate | +0.012 | eligible (marginal) |
| hate_aviator | hate | -0.004 | flagged_weak |
| hate_hatexplain | hate | +0.086 | eligible |
| emotion_tg_base | emotion | +0.466 | eligible |
| emotion_fabriceyhc | emotion | -0.082 | eligible (very low) |
| emotion_hatexplain | emotion | -0.150 | flagged_weak |

**Key outputs to open:**

- `examples/inventories/inventory_with_weak_sources/` — fixture with weak-source patterns
- `examples/inventories/inventory_with_weak_sources/expected_notes.md` — expected exclusion behavior
- `examples/qa/` — canonical QA artifacts showing eligible, uncertain, and flagged_weak statuses

**What it demonstrates:** The three-way eligibility classification (eligible / uncertain / flagged_weak) is the most impactful single feature. Without it, you might merge a below-base adapter with a strong one and wonder why the result degraded. The gate catches this before pairwise analysis begins.

**What to do:** For flagged_weak adapters you believe are salvageable — re-evaluate with larger sample, check label mapping, try different hyperparameters, then re-run preflight.

---

## 5. Near-Miss Case

**Purpose:** Show what happens when structurally plausible pairs are excluded only because one source has weak evidence — and why that exclusion is well-calibrated.

**Setup:** 4 irony adapters, 3 eligible + 1 flagged_weak (delta -0.004 vs base). 6 pairs total: 3 retained, 3 near-miss.

| Adapter | Task | Eligibility | Delta |
|---------|------|-------------|-------|
| irony_JB173 | irony | eligible | +0.202 |
| irony_vaariis | irony | eligible | +0.060 |
| irony_neibla | irony | eligible | +0.068 |
| irony_phailyoor | irony | flagged_weak | -0.004 |

**Key outputs to open:**

- `examples/inventories/inventory_fragmented_small/` — fixture with near-miss grouping patterns
- `examples/inventories/inventory_fragmented_small/expected_notes.md` — expected near-miss separation

**What it demonstrates:** Near-miss pairs are a structured second tier, not rejects. Field trial validation across 3 backbones and 3 task families shows near-miss avg Δ = -0.006 (comparable to retained at -0.024), while cross-task controls average -0.047 (5× worse). The evidence gate is well-calibrated: adapters that barely miss (delta -0.002 to -0.004) produce merges indistinguishable from retained. Deeply weak sources introduce more variance.

**What to do:** If retained pairs are few, near-miss pairs expand the candidate set with known risk profile. Consider strengthening the weak source's evidence to promote the pair on the next run.

---

## 6. Retained vs Control — Evaluation Outcome

**Purpose:** Show what actually happens when you evaluate retained pairs versus cross-task controls, confirming the preflight narrowing was correct.

**Setup:** This uses real field trial data from Phase 2 evaluation. 16 merges evaluated across 3 categories: retained same-task, near-miss, and cross-task control.

| Category | Pairs | Avg Δ vs best source | Improvers |
|----------|-------|----------------------|-----------|
| Retained same-task | 7 | -0.024 | 2/7 (29%) |
| Near-miss | 7 | -0.006 | 1/7 (14%) |
| Cross-task control | 4 | -0.047 | 0/4 (0%) |

**Key outputs to open:**

- `field_trials/phase2_eval_130608/phase2_results.json` — raw evaluation results for all 16 merges
- `field_trials/product_validation_memo.md` — analysis of what the pipeline got right and where limits are
- `field_trials/near_miss_validation.md` — detailed near-miss evaluation across backbones
- `field_trials/phase2_evaluation_report.md` — full Phase 2 evaluation narrative

**Representative cases from the results:**

| Pair | Role | Merged acc | Notes |
|------|------|-----------|-------|
| p2_retained_agnews | retained | 0.944 | Same-task, norm_equalized. Gradience recommended correctly. |
| p3_retained_sst2 | retained | 0.820 | Same-task with rank mismatch (r=16 vs r=8). Modest degradation. |
| p3_nearmiss_hate | near-miss | 0.598 | One source flagged_weak. Merge outcome comparable to retained. |
| p2_control_agnews×mnli | control | 0.938 | Cross-task, high risk. Advisory fired correctly. |
| p3_control_sst2×agnews | control | 0.838 | Cross-task, r=16 vs r=1 mismatch. Degraded as predicted. |

**What it demonstrates:** The narrowing logic works. Retained pairs are the right first choices — they either improve over both sources or degrade modestly. Cross-task controls degrade substantially more, confirming the advisory signal. Near-miss pairs perform comparably to retained, validating the evidence gate's calibration. Zero false positives across 5 inventories and 53+ pairs.

---

## 7. Checkpoint Triage Alpha (Route 2)

**Purpose:** Show the first polished broadened workflow beyond adapter merge preflight.

**Setup:** Canonical checkpoint inventory trial `field_trials/checkpoint_inventory_t02/` (same-task + same-family + cross-task, shared base model, CPU-only).

**Key outputs to open:**

- `field_trials/checkpoint_inventory_t02/preflight/alpha_bundle/report.html` — clean alpha HTML report
- `field_trials/checkpoint_inventory_t02/preflight/alpha_bundle/alpha_summary.json` — compact machine-readable summary
- `docs/examples/checkpoint-triage-alpha-workflow.md` — short usage walkthrough
- `docs/strategy/checkpoint_triage_alpha_scope.md` — explicit alpha scope contract

**What it demonstrates:** Evidence bootstrap remains the first-class gate, pairwise structure remains useful, and the workflow produces conservative narrowing with clear same-task/same-family/cross-task distinctions in a real checkpoint-inventory triage setting.

---

## Choosing the right example

| Your situation | Start here |
|---------------|-----------|
| All adapters are same-task and well-evidenced | [Same-Task Control](#1-same-task-control) |
| Mixed tasks, all adapters well-evidenced | [Mixed-Task Inventory](#2-mixed-task-inventory) |
| Large pool, multiple task families | [Large Mixed-Task Inventory](#3-large-mixed-task-inventory) |
| Some adapters have weak or missing evidence | [Weak-Evidence Inventory](#4-weak-evidence-inventory) |
| Structurally good pairs excluded by the evidence gate | [Near-Miss Case](#5-near-miss-case) |
| Want to see whether preflight predictions hold up | [Retained vs Control](#6-retained-vs-control--evaluation-outcome) |
| Want the detailed pair-by-pair walkthrough | [Mixed-Task Walkthrough](examples/mixed-task-inventory-walkthrough.md) |
