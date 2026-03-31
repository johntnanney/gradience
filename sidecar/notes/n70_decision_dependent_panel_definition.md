# n70 — Decision-Dependent Compatibility Panel Definition

**Type:** panel definition  
**Date:** 2026-03-31  
**Depends on:** routing pilot, targeted confirmation runs, checkpoint triage trials (T01/T02)  
**Status:** Stage A complete

---

## Objective

Define a shared, compact case panel for comparing how the same learned-variant relations are interpreted under three scenarios:

1. merge compatibility
2. routing/confusability
3. inventory triage

The panel is intentionally small and CPU-native. It is designed to compare *meanings* of compatibility, not to run a new benchmark.

---

## Panel outputs

- `sidecar/results/decision_dependent_compatibility/panel_table.json`
- `sidecar/results/decision_dependent_compatibility/panel_table.md`

Panel size: **9 cases**, split 3/3/3 across merge-sensitive, routing-sensitive, and triage-sensitive groups.

---

## Case groups

### Group 1 — Merge-sensitive pairs

- `mrg_safe_same_task_sst2_sst2_t01` (retained same-task anchor)
- `mrg_near_miss_substantial_hate_t02` (near-miss substantial)
- `mrg_cross_task_control_sst2_agnews_t01` (cross-task control)

### Group 2 — Routing-sensitive pairs

- `rte_seed_pair_confusable` (high confusability same-task)
- `mnli_qnli_moderate_confusable` (moderate confusability same-family)
- `qnli_rte_separable` (clearly separable pair)

### Group 3 — Triage-sensitive checkpoint cases

- `tri_same_task_near_miss_sst2_pair_t02`
- `tri_same_family_review_sst2_yelp_t02`
- `tri_cross_task_weak_region_yelp_qnli_t02`

---

## Required-field coverage

Each case includes:

- `case_id`
- `artifact_type`
- `backbone`
- `task_relation` (`same_task`, `same_family`, `cross_task`)
- scenario availability (`merge`, `routing`, `triage`)
- structural output availability
- behavioral output availability
- why the case is informative

---

## Overlap criterion check

The overlap requirement is satisfied:

- routing pilot cases are explicitly dual-scenario (`merge` + `routing`) via `routing_report.json` merge-vs-routing comparison rows.
- merge confirmation cases are dual-scenario (`merge` + `triage`) via merge outcomes plus inventory action-plan placement.

So the panel supports cross-scenario interpretation of shared relations, not isolated one-off outputs.

---

## Scope and caveat

This panel is intentionally bounded:

- small encoders only
- CPU-only artifacts
- existing sidecar/field-trial outputs only
- no new model training or benchmark expansion

That constraint is deliberate: Stage A is a comparability scaffold, not an external validity claim.
