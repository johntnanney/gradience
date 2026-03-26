# Publication Summary — messy_heterogeneous_6

Series: wave 2, Target 3
RQ: RQ2 + RQ4 + RQ5
Date: 2026-03-22

## One-line summary

A deliberately messy 6-adapter pool confirms that heterogeneous inventories with unknown adapters are solved by source QA, adding no new insight beyond wave 1.

## Setup

- 6 adapters: 3 eligible, 2 unknown, 1 weak. Ranks: 2, 8+alpha, 16, 16, 32, 32.
- Deliberately no clean grouping.

## What happened

1. Source QA: strict-QA blocks 12/15 pairs (3 eligible-only pairs survive).
2. Pair audit: 4 high-risk (all from r=2 per-layer adapter), 2 medium, 9 low.
3. Neighborhoods: 3 groups + 1 excluded. 3 eligible in likely-safe group, 2 unknown in caution singletons, weak excluded.
4. Core-space: skipped (no ambiguous low-risk cross-task pair).

## Where the workflow was confirmatory

Everywhere. This inventory produced the same pattern as wave 1's messy inventories: QA dominates, per-layer adapters create norm imbalance, neighborhoods reflect QA status. No new insight.

## Inventory-level lesson

Messy pools with mixed QA are the workflow's easiest regime — source QA does almost all the work. The harder question (answered by Targets 1 and 4) is what happens when QA doesn't help.
