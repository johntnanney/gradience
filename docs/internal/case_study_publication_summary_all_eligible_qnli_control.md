# Publication Summary — all_eligible_qnli_control

Series: wave 2, Target 5 (low-drama control)
RQ: RQ4 — In what inventories does the workflow become merely confirmatory?
Date: 2026-03-22

## One-line summary

A clean 3-adapter inventory where the workflow adds no narrowing — the expected and useful baseline case.

## Setup

- 3 distilbert-base-uncased adapters, all eligible, all QNLI
- Rank policies: uniform r=32, probe r=32, per-layer r=8+alpha

## What happened

1. Source QA: all eligible. Nothing excluded. Strict-QA blocks nothing.
2. Pair audit: 2 low-risk (uniform×probe, probe×per_layer), 1 medium (uniform×per_layer, partial redundancy at compat=0.524).
3. Neighborhoods: all 3 in one audit-aware group. No boundaries, no exclusions.
4. Core-space: skipped (no ambiguous cross-task pair).

## Where the workflow was merely confirmatory

Everywhere. The pool was already clean. The workflow correctly confirmed this and flagged one strategy recommendation (audit-aware for the partially redundant pair).

## Inventory-level lesson

When all adapters have behavioral evidence and all are on the same task, source QA does nothing and neighborhoods add no structure beyond restating the pair matrix. The workflow's value here is verification, not discovery. This is the regime where a lighter-weight pass (just pair reports) would suffice.
