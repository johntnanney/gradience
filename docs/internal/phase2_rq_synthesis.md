# Phase 2 RQ Synthesis

Answers to the five research questions, based on wave 2 evidence (5 inventories) plus wave 1 (3 inventories).

---

## RQ1: When all/most adapters have behavioral evidence, does source QA still dominate narrowing?

**Evidence:** Target 1 (behaviorally_complete_5)

**Finding:** Source QA still contributes meaningfully (removing 1 weak adapter, 40% pair reduction) but no longer dominates. In the surviving 4-eligible pool, core-space changed structural judgments on all 3 cross-task low-risk pairs (2 incompatible, 1 marginal).

**Strength:** Moderate after adjudication. Core-space changed structural judgments, but verified adjudication (2026-03) showed same-task merges were safe even when flagged as incompatible, and cross-task degradation was already captured by ordinary pair-risk. The structural judgment changes were real but not broadly behaviorally decisive.

**Implication:** The phrase "inventory mistakes before pair mistakes" still holds in the weak sense (QA still narrows first), but the practical weight shifts. In credible pools, pair-risk separates same-task safe merges from cross-task unsafe ones. Core-space adds structural detail but its decision value is regime-dependent.

---

## RQ2: At what inventory size do neighborhoods add more value than raw pair reports?

**Evidence:** Target 2 (7 adapters, 21 pairs), Target 3 (6 adapters, 15 pairs), Target 5 (3 adapters, 3 pairs)

**Finding:** Neighborhoods provide genuine compression at 7 adapters (21 pairs → 3 groups + 3 boundary warnings). At 6 adapters with mixed QA (T3), they compress to 3 groups + 1 excluded. At 3 adapters (T5), they produce 1 group — no compression.

**Caveat:** The grouping at 7 adapters was QA-driven (eligible together, unknown isolated), not structurally driven. The hypothesized QNLI-vs-Final two-cluster structure did not emerge because eligible adapters from different tasks were uniformly low-risk to each other.

**Strength:** Moderate. Neighborhoods scale with size but their grouping reflects whatever signal dominates the pair matrix — usually QA status, not structural similarity.

**Implication:** Neighborhoods are operationally useful at 6+ adapters for compression. But they cannot distinguish between structurally different groups when pair-risk is uniformly low. Core-space can detect structural differences in that regime, though verified adjudication showed those differences are not always behaviorally meaningful.

---

## RQ3: How often does core-space change a real decision when ordinary pair risk is low?

**Evidence:** Target 4 (full 6-pair census), Target 1 (3 cross-task pairs), Target 2 (2 cross-group pairs)

**Finding:** Core-space changed judgment on 10/10 pairs across 3 inventories where it was used. The full census (T4) found ALL 6 pairs incompatible or marginal — including same-task (qnli_probe × qnli_uniform: 0.807) and same-group (final × priority: 0.824).

**Strength:** Strong but with a calibration concern. The 100% disagreement rate in T4 suggests either: (a) core-space overfires on structurally diverse pools, or (b) different rank allocation policies create genuinely incompatible basis representations regardless of task.

**Implication:** Core-space is a structurally informative diagnostic but verified adjudication (2026-03) showed its behavioral decision value is narrower than this rate suggests. Same-task merges were safe even when flagged as incompatible (shared_basis 0.870-0.873). Cross-task merges degraded, but ordinary pair-risk already captured that boundary. Core-space overwarned on same-task seed variants.

**Adjudication answer:** The downstream merge evaluation was performed. "Incompatible" same-task merges did NOT fail. "Incompatible" cross-task merges did degrade, but pair-risk already separated them. The shared_basis_score band 0.80-0.87 appears to reflect real structural divergence that is not by itself sufficient to predict behavioral harm.

---

## RQ4: In what inventories does the workflow become merely confirmatory?

**Evidence:** Target 5 (all-eligible same-task control), Target 3 (messy pool)

**Finding:** Two distinct "merely confirmatory" regimes:
1. **Clean same-task pools (T5):** All eligible, all same-task. Workflow confirms everything is fine. No narrowing, no surprises. The full pipeline adds no value beyond reading 3 pair reports.
2. **Messy mixed-QA pools (T3):** Source QA does almost all the work. The pattern is identical to wave 1. No new insight.

**Strength:** Strong. Both regimes are clearly identified and empirically supported.

**Implication:** The workflow adds the most value in the middle — pools that are partially credible, cross-task, and structurally ambiguous. The extremes (all clean or all messy) are solved by simpler means.

---

## RQ5: How often does the inventory-level view change the next action more than pairwise scores alone?

**Evidence:** All 5 inventories

**Finding:**

| Inventory | Inventory view changed action? |
|-----------|-------------------------------|
| T5 (control) | No — same conclusion from pair reports |
| T1 (behavioral) | Yes — core-space + QA exclusion produced a stronger conclusion than pair matrix alone |
| T4 (core-space) | Yes — the census showed uniform incompatibility, reframing the entire pool |
| T2 (neighborhood) | Partially — neighborhoods compressed 21 pairs into 3 groups, operationally useful |
| T3 (messy) | No — QA status counts alone reach the same conclusion |

The inventory-level view changed action in 2 of 5 inventories, was operationally helpful in 1 more, and added nothing in 2. The cases where it helped most were the behaviorally complete inventories (T1, T4) — precisely the regime where source QA does not dominate and deeper analysis carries the decision.

**Strength:** Moderate. The inventory view is not always necessary, but when it matters, it matters a lot.

---

## Cross-cutting synthesis

The most important finding from waves 1-2 was a preliminary regime map. Verified adjudication (2026-03) and the task-relationship advisory validation round sharpened it into a mature version:

**Regime map (updated after adjudication + advisory validation):**

| Pool regime | Main narrowing driver | Neighborhoods | Task advisory | Core-space | Workflow value |
|------------|----------------------|---------------|---------------|------------|----------------|
| Messy (many unknown/weak) | Source QA | Confirm QA | Mostly silent (QA already narrowed) | Not needed | High (saves time on bad pools) |
| Same-task, all eligible | None | Uninformative | Silent (same eval_dataset) | Overfires; not behaviorally decisive | Low (merely confirmatory) |
| Adjacent-task, credible pool | Task advisory + pair-risk | Useful if pool is large enough | **Primary discriminator** — cleanly separates safe from caution zones | Noisy; does not reliably predict this failure mode | High (advisory catches what pair-risk misses) |
| Distant cross-task | Pair-risk + task advisory | Useful at scale | Clarifying — reinforces structural caution | Confirmatory only | High (advisory + pair-risk converge) |
| Large, mixed task + mixed QA | Source QA + neighborhoods + task advisory | Compress pair matrix | Partitions matrix into same-task safe / cross-task caution | Selective use only | High (compression + narrowing + task-aware partitioning) |

Key changes from the pre-adjudication map:
- **Task advisory added as a column.** It is the strongest discriminator in the adjacent-task regime.
- **Core-space downgraded.** Verified adjudication showed it is not broadly behaviorally decisive. It overwarns on same-task pairs and is noisy in the adjacent-task middle.
- **Adjacent-task regime identified.** This was previously lumped into "behaviorally complete, cross-task." It is now a distinct regime with its own narrowing hierarchy.

This is now a mature, empirically grounded map. Each cell is backed by verified adjudication data (29 pairs, 3 studies across 2 backbones: distilbert-base-uncased and roberta-base) and a 52-pair advisory validation set (5 inventories + 1 replication round, 0 false positives).

See:
- `docs/internal/verified_adjudication_implications.md`
- `docs/internal/adjacent_task_adjudication_implications.md`
- `docs/internal/task_relationship_advisory_round_01_synthesis.md`
- `docs/internal/roberta_replication_results.md`
