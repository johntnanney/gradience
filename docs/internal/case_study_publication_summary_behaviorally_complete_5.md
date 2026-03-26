# Publication Summary — behaviorally_complete_5

Series: wave 2, Target 1
RQ: RQ1 — When all adapters have behavioral evidence, does source QA still dominate?
Date: 2026-03-22

## One-line summary

In a behaviorally complete inventory, source QA still helps but core-space becomes the primary decision driver for the surviving credible pool.

## Setup

- 5 distilbert-base-uncased adapters, all with behavioral evaluation
- 4 eligible (3 QNLI, 1 generic fine-tuning) + 1 flagged_weak (QNLI)
- Ranks: r=32 (QNLI uniform/probe), r=8+alpha (QNLI per-layer), r=16 (generic)

## What happened

1. **Source QA** excluded the weak adapter (4 of 10 pairs removed). Meaningful but not dominant.
2. **Pair audit** found 6 low-risk, 4 medium-risk among the surviving 4 eligible adapters. 3 cross-task pairs (QNLI × generic) all low-risk at layer level.
3. **Core-space** on all 3 cross-task low-risk pairs: 2 incompatible (shared_basis 0.859-0.860), 1 marginal (0.931). All 3 structurally demoted. Verified adjudication later confirmed cross-task merges do degrade, but also showed ordinary pair-risk already separates cross-task from same-task pairs in this regime. Core-space added structural detail consistent with the degradation, but was not the only signal available.
4. **Neighborhoods** placed all 4 eligible in one group. Correct but uninformative.

## Where the workflow was strong

- Core-space flagged all 3 cross-task pairs. However, verified adjudication showed ordinary pair-risk already separates cross-task degradation from same-task safety in this regime. Core-space confirmed the structural divergence but was not the only path to the correct decision.
- Source QA still contributed meaningfully (40% pair reduction) even though only 1 adapter was weak.

## Where the workflow was merely helpful

- Neighborhoods added nothing beyond confirming the QA exclusion.
- Pair-risk distribution (6 low, 4 medium) was not by itself fully actionable — the low-risk cross-task pairs benefited from core-space for structural confirmation, though pair-risk verdicts (imbalanced for cross-task pairs) already provided the main separation.

## Inventory-level lesson

**RQ1 answer:** Source QA no longer dominates when the pool is behaviorally complete. The narrowing hierarchy shifts: QA does initial cleanup (removes weak), then core-space becomes the primary discriminator among credible adapters. This is a qualitatively different regime from wave 1, where QA dominance was the consistent pattern.

**Implication:** The phrase "inventory mistakes before pair mistakes" still holds (QA still narrows first), but the relative weight shifts. In credible pools, the deeper structural checks matter most.
