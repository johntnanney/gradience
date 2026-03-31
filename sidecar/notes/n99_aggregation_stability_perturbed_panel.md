# n99 -- Aggregation Stability Check: Perturbed Panel

**Type:** substudy setup note
**Date:** 2026-03-31
**Program:** Route2 Substudy 2 -- Aggregation-Sensitive Compatibility Stability Check
**Stage:** B
**Depends on:** n98
**Status:** complete

---

## Objective

Construct a minimally perturbed panel that preserves original scenario-family coverage while swapping only nearby cases.

---

## Perturbation design

Rules applied:

1. Preserve merge/routing/triage family coverage.
2. Replace one case per scenario family (3 total).
3. Use nearby, already-available Route 2 cases.
4. Preserve case-role function (near-miss, moderate routing, mixed-QA review).

Substitutions:

- merge: mrg_near_miss_substantial -> mrg_near_miss_marginal
- routing: mnli_qnli_moderate -> mnli_rte_moderate_alt
- triage: tri_cross_task_qa_review -> tri_cross_task_qa_review_alt

Unchanged anchors were retained in each family for comparability.

---

## Comparability assessment

- Panel size preserved (12).
- Group counts preserved (3/3/3/3).
- Aggregation families unchanged.
- Task-relation and QA-regime diversity preserved.

The perturbation is narrow enough for local robustness testing, not a redesign.

---

## Outputs

- sidecar/results/route2_stability/aggregation/perturbed_panel_table.json
- sidecar/results/route2_stability/aggregation/perturbed_panel_table.md
- sidecar/results/route2_stability/aggregation/panel_diff_table.md
- sidecar/notes/n99_aggregation_stability_perturbed_panel.md (this note)
