# n118 -- Collapse vs Contamination Baseline Freeze

**Type:** replication setup note  
**Date:** 2026-03-31  
**Program:** Route2 Collapse vs Contamination Replication  
**Stage:** A  
**Depends on:** n88, n92, sidecar/results/behavioral_route2_bridge/profile_behavior_table.json  
**Status:** complete

---

## Objective

Freeze the original collapse-vs-contamination behavioral distinction before replication.

---

## Frozen baseline

The baseline distinction is anchored on:

1. `FR-01` (`worst_case_collapse`) -- uncertainty-weighted collapse channel.
2. `CT-01` (`cross_task_separable`) -- confident contamination channel.

Both cases exhibit similar novel-failure pressure by neither-source behavior (~14%), but different confidence channels:

- `FR-01`: high confidence-collapse, near-zero high-confidence wrong.
- `CT-01`: low confidence-collapse, high high-confidence wrong.

---

## Baseline metrics (frozen)

| case_id | profile_label | failure_rate | confidence_collapse_rate | high_confidence_wrong_rate | confusion_or_neither_source_rate | channel interpretation |
|---|---|---:|---:|---:|---:|---|
| FR-01 | worst_case_collapse | 0.336 | 0.060 | 0.000 | 0.146 | collapse-like (uncertainty-dominant) |
| CT-01 | cross_task_separable | 0.174 | 0.006 | 0.046 | 0.144 | contamination-like (confident misassignment) |

---

## Scope limits (frozen)

1. This distinction is currently merge-facing and panel-bounded.
2. It is not yet a universal cross-scenario behavioral law.
3. Stability claims require nearby case/slice replication rather than broader ontology expansion.

---

## Outputs

- `sidecar/notes/n118_collapse_vs_contamination_baseline.md`
- `sidecar/results/route2_stress_tests/collapse_vs_contamination/baseline_snapshot.json`
