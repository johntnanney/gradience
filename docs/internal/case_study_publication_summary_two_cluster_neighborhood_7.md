# Publication Summary — two_cluster_neighborhood_7

Series: wave 2, Target 2
RQ: RQ2 — At what inventory size do neighborhoods add more value than raw pair reports?
Date: 2026-03-22

## One-line summary

At 7 adapters (21 pairs), neighborhoods provided genuine compression but grouped by QA status rather than structural similarity.

## Setup

- 7 distilbert-base-uncased adapters: 5 eligible (3 QNLI + 2 generic) + 2 unknown (generic cycle02)
- Expected: two structural clusters (QNLI vs Final). Actual: three QA-driven groups.

## What happened

1. Pair audit: 18 low-risk, 3 medium. Extremely flat — per-layer norm imbalance did not appear.
2. Neighborhoods: 3 groups. All 5 eligible adapters in one audit-aware group (regardless of task). 2 unknown adapters each in caution singletons. 3 boundary warnings.
3. Core-space on 2 cross-group pairs: incompatible (0.867) and marginal (0.863). Consistent with prior pattern.
4. Strict-QA blocks 11/21 pairs.

## Where neighborhoods were strong

21 pairs is hard to parse manually. Neighborhoods compressed it to one actionable group + two isolated singletons + 3 boundary warnings. Real operational value.

## Where neighborhoods surprised

The grouping was QA-driven, not structurally driven. QNLI and generic eligible adapters grouped together because their pairwise risk was uniformly low. The expected structural boundary between task groups did not appear in the neighborhood output.

## Inventory-level lesson

Neighborhoods scale with inventory size and provide genuine compression at 7 adapters. But they reflect whatever signal dominates the pair matrix — in this case, QA status. When all eligible adapters are low-risk to each other, neighborhoods cannot distinguish between structurally different but spectrally compatible groups. That requires core-space.
