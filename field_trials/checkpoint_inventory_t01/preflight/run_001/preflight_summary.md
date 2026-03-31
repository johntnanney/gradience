# Preflight Summary — checkpoint_inventory_t01

**Run:** run_001
**Generated:** 2026-03-31 00:26 UTC

## Source QA

- eligible: 1
- flagged_weak: 3

## Inventory Policy Summary

- **Type:** mixed_quality
- **Driver:** source_qa
- **Posture:** narrow
- **Constraint:** Source QA is the binding constraint; resolve weak evidence before exploring merges.

## Task-boundary partition

- Same-task pairs (advisory silent): 1
- Cross-task pairs (advisory active): 5
- Total pairs: 6

## Reduced candidate set

- no clear priority candidates identified

## Near-miss candidates

Structurally plausible, evidence-constrained. Optional if evaluation budget allows.

- sst2_s42 × sst2_s123  (medium risk, norm_equalized — sst2_s42, sst2_s123 are evidence-constrained)

**QA dominates this inventory; no credible same-task candidates remain.**

## Provenance

Sources with behavioral evidence: 4/4

*Behavioral scores are user-reported; Gradience does not independently
verify claimed evaluation results.*

## Inventory action plan

See `inventory_action_plan.md` for the full structured plan.

## Detailed artifacts

- `qa/` — source QA artifacts
- `pair_reports/` — pairwise merge reports
- `inventory/` — inventory summary
- `neighborhoods/` — neighborhood grouping
