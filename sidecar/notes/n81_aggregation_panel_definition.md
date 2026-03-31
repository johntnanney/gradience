# n81 -- Aggregation Panel Definition

**Type:** panel definition
**Date:** 2026-03-31
**Program:** Aggregation-Sensitive Compatibility (Route 2)
**Stage:** A
**Depends on:** n70-n74 (decision-dependent compatibility), n76-n80 (cross-artifact portability), routing pilot, adapter T01, checkpoint T02
**Status:** complete

---

## Question

How do different aggregation rules transform the same underlying structural measurements into different operationally relevant compatibility judgments?

---

## Panel design

12 cases across 4 groups, drawn from existing validated artifacts. All cases share `distilbert-base-uncased` as backbone.

### Group 1 -- Merge-facing (3 cases, QA-clear)

Cases with behavioral merge evaluation data. QA does not constrain. Structural risk and merge outcome are the operative signals.

### Group 2 -- Routing-facing (3 cases, structural-only)

Cases from the routing pilot with explicit confusability scoring. Distributional aggregation is the natural fit. Worst-case flattens the confusability gradient.

### Group 3 -- Triage-facing / QA-blocked (3 cases)

Checkpoint delta pairs where QA status dominates. Even structurally compatible cases are blocked by missing or weak behavioral evidence.

### Group 4 -- Triage-facing / QA-clear (3 cases)

Adapter pairs with clear QA. Same task relations as Group 3 but without the QA constraint. Isolates the structural signal.

### Key design feature: matched pairs across QA regimes

Groups 3 and 4 form matched contrasts:
- same_task with QA blocked vs QA clear
- same_family with QA blocked vs QA clear
- cross_task with QA blocked/mixed vs QA clear

This lets the analysis isolate what QA-dominant aggregation does that structural-only aggregation does not.

---

## Success criteria assessment

| Criterion | Met? |
|-----------|------|
| Shared multi-scenario case panel exists | Yes (12 cases, 3 scenarios) |
| Each aggregation family applicable to same cases | Yes |
| Enough diversity to expose divergence | Yes (3 QA regimes, 3 task relations, 2 artifact classes) |

---

## Output artifacts

- `sidecar/results/aggregation_sensitive_compatibility/panel_table.json`
- `sidecar/results/aggregation_sensitive_compatibility/panel_table.md`
- `sidecar/notes/n81_aggregation_panel_definition.md` (this note)
