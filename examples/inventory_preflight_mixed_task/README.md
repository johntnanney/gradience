# Mixed-Task Inventory Preflight Example

Real preflight artifacts from a 6-adapter, 4-task inventory (SST-2, QNLI, MNLI, RTE).

Demonstrates 15 → 2 pair reduction (87% search-space reduction).

See [walkthrough](../../docs/examples/mixed-task-inventory-walkthrough.md) for the full interpretation.

## Contents

- `qa/` — 6 adapter QA artifacts (all eligible)
- `reports/` — 15 pairwise merge reports (13 with task-boundary advisory, 2 same-task safe)
- `inventory/` — inventory summary and neighborhood groupings
