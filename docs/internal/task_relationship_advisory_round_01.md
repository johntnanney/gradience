# Task-Relationship Advisory — Validation Round 01

## Purpose

Evaluate the new `task_relationship_advisory` field across a fresh set of real inventories to determine whether it improves real inventory decisions and in which regimes it matters most.

This is an observation and validation phase, not a calibration phase.

## Research questions

- **RQ1:** How often does `task_relationship_advisory` appear in real inventories?
- **RQ2:** When it appears, how often does it correspond to a pair that would otherwise have looked too safe under structural reporting alone?
- **RQ3:** How often does it actually change the recommended next action for the inventory?
- **RQ4:** Does it mostly confirm what QA / neighborhoods already implied, or does it add genuinely new decision value?
- **RQ5:** Are there cases where the advisory is noisy or unhelpful enough that the wording or placement should change?

## Target inventories

5 inventories, one per category:

| # | ID | Category | Adapters | Task mix | Purpose |
|---|---|---------|----------|----------|---------|
| 1 | same_task_sst2_control | A (same-task) | 3 SST-2 | all sst2 | Advisory should stay silent |
| 2 | nli_family_adjacent | B (adjacent-task) | 4 NLI-family | qnli + rte | Main target regime |
| 3 | cross_task_sst2_qnli | C (distant cross-task) | 4 | sst2 + qnli | Advisory restating the obvious? |
| 4 | messy_mixed_quality | D (messy) | 5 | qnli + rte + mnli, mixed QA | Advisory after QA dominates |
| 5 | large_diverse_pool | E (larger pool) | 7 | sst2 + qnli + rte | Advisory at scale |

## Freeze conditions

- No threshold changes during the round
- No pair-risk modifications
- No advisory wording changes mid-round
- Advisory is observed as shipped

## Required outputs per inventory

- QA artifacts
- Pair reports (with advisory field)
- Inventory summary
- Neighborhoods
- Advisory effect note (structured)

## Advisory effect classification

For each advisory-bearing pair, classify as:

1. **No effect** — advisory appeared but did not change pair treatment
2. **Clarifying effect** — did not change the plan but made reasoning more legible
3. **Caution-raising effect** — pushed a pair from "looks fine" to "treat cautiously"
4. **Action-changing effect** — materially changed what pair/neighborhood would be explored next
5. **Redundant effect** — added little because QA or pair-risk already made the same point

## Success criteria

The round is a success if:

- Advisory appears in expected different-task regimes
- Advisory stays mostly silent in same-task control inventories
- At least 2 inventories show clarifying, caution-raising, or action-changing value
- No strong evidence that the advisory is misleading or noisy
- The round produces a clear recommendation: keep as-is / reword / expand later / demote

## Advisory impact ratio

Per inventory: (# advisory-bearing pairs that changed or clarified action) / (# advisory-bearing pairs total)
