# Task-Relationship Advisory — Round 01 Candidates

Staging doc for candidate inventories by category. Tag each with the regime it covers.

## Category A — Same-task control

Purpose: advisory should stay silent. Confirm no false alarms.

| Candidate | Adapters | Task | Base model | Notes |
|-----------|----------|------|------------|-------|
| same_task_sst2_control | 3 SST-2 (r16s42, r16s123, r8s42) | sst2 | distilbert | Already run in round 01. 0/3 advisories. |
| same_task_qnli_control | 3 QNLI (s42, s7, r16s42v2) | qnli | distilbert | Available from existing adapters. |

## Category B — Adjacent-task / related-task

Purpose: main target regime. Advisory should fire and add value.

| Candidate | Adapters | Tasks | Base model | Notes |
|-----------|----------|-------|------------|-------|
| nli_family_adjacent | 4 (2 QNLI + 2 RTE) | qnli + rte | distilbert | Already run in round 01. 4/6 advisories. |
| nli_family_three_task | 4 (QNLI + RTE + MNLI + MNLI) | qnli + rte + mnli | distilbert | Extends to 3-task NLI family. |

## Category C — Distant cross-task

Purpose: advisory restating the obvious or adding genuine clarity?

| Candidate | Adapters | Tasks | Base model | Notes |
|-----------|----------|-------|------------|-------|
| cross_task_sst2_qnli | 4 (2 SST-2 + 2 QNLI) | sst2 + qnli | distilbert | Already run in round 01. 4/6 advisories. |

## Category D — Messy mixed-quality

Purpose: does advisory matter after QA already dominates?

| Candidate | Adapters | Tasks | Base model | Notes |
|-----------|----------|-------|------------|-------|
| messy_mixed_quality | 5 (QNLI + RTE + 2 MNLI + RTE) | qnli + rte + mnli | distilbert | Already run in round 01. 8/10 advisories. |

## Category E — Larger pool (6-8 adapters)

Purpose: advisory at scale. Does it help compress the pair matrix alongside neighborhoods?

| Candidate | Adapters | Tasks | Base model | Notes |
|-----------|----------|-------|------------|-------|
| large_diverse_pool | 7 (2 SST-2 + 3 QNLI + 2 RTE) | sst2 + qnli + rte | distilbert | Already run in round 01. 16/21 advisories. |

## Round 01 status

All 5 candidates above were executed in round 01. Results are in `results/task_advisory_round/` and synthesized in `docs/internal/task_relationship_advisory_round_01_synthesis.md`.

## Future candidates (round 02, if needed)

- Roberta-based inventory (test generalization across base models)
- Inventory with intentionally mismatched eval_dataset labels (test advisory robustness to naming)
- Inventory mixing verified and unverified adapters (test advisory interaction with QA gaps)
