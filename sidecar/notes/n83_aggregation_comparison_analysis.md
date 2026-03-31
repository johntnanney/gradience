# n83 -- Aggregation Comparison Analysis

**Type:** findings note
**Date:** 2026-03-31
**Program:** Aggregation-Sensitive Compatibility (Route 2)
**Stage:** C
**Depends on:** n81 (panel), n82 (family audit)
**Status:** complete

---

## Question

What truths are exposed, flattened, or erased when the same cases are passed through different aggregation rules?

---

## Method

Apply four aggregation families (worst-case, distributional, QA-dominant, QA-gated distributional) to the same 12-case panel. Label each case under each family. Record agreement patterns.

---

## Results

### Agreement distribution

| Pattern | Count | Cases |
|---------|-------|-------|
| Full agreement | 2 | mrg_cross_task_control, tri_cross_task_qa_clear |
| Partial agreement | 8 | All same-task and same-family cases with clear or structural QA |
| Strong divergence | 2 | qnli_rte_separable, tri_same_task_qa_blocked |

### What each family sees uniquely

**Worst-case sees** catastrophic local risk but cannot distinguish confusable from separable same-family pairs. All three routing cases receive the same merge_caution label. It flattens the confusability gradient.

**Distributional sees** the confusable/moderate/separable ordering that worst-case destroys. It also distinguishes near-miss (needs_disambiguation) from safe retained (confusable). But it is blind to QA status and insensitive to catastrophic local pathology.

**QA-dominant sees** the evidence boundary. It overrides structurally positive signals when behavioral evidence is missing. It cannot distinguish structural gradation within the blocked set — the best pair (0.892 compatibility) and the worst pair (0.489) both receive qa_blocked.

**QA-gated distributional sees** both the evidence boundary and the structural gradient. When QA clears, it produces the same output as distributional. When QA blocks, it produces the same output as QA-dominant. It is the only family that preserves both lenses.

---

## The two sharpest divergences

### 1. qnli_rte_separable (routing gradient destruction)

Worst-case: merge_caution (same label as confusable and moderate cases).
Distributional: routing_separable (clearly distinct from confusable and moderate).

This case demonstrates that worst-case aggregation destroys operationally relevant routing gradation. Under worst-case, a practitioner would treat this pair the same as the confusable pair. Under distributional, they would correctly identify it as easily routed.

### 2. tri_same_task_qa_blocked (structural-QA contradiction)

Worst-case: merge_caution (structurally sound).
Distributional: routing_confusable (high compatibility, high overlap).
QA-dominant: qa_blocked (both sources lack evidence).

This case demonstrates that QA-dominant aggregation is not a refinement of structural analysis — it is a genuinely different lens. The pair has the highest compatibility score in the panel (0.892) and is structurally the safest case. But QA blocks it because neither source has behavioral evidence. The structural truth and the operational truth flatly contradict.

---

## The matched-pair insight

The panel includes three matched pairs across QA regimes:
- same_task: QA-blocked (0.892 compat) vs QA-clear (0.475 compat)
- same_family: QA-blocked (0.652) vs QA-clear (0.314)
- cross_task: QA-mixed (0.584) vs QA-clear (0.111)

In every matched pair, the QA-blocked version has **higher** structural compatibility than the QA-clear version. This is not an accident — it reflects the panel construction (checkpoint deltas vs LoRA adapters). But it sharply illustrates that structural compatibility and operational readiness are independent dimensions. Aggregation is the mechanism that mediates between them.

---

## Confirmation of H1

H1 stated: "Aggregation is not presentation." The evidence confirms this. Aggregation changes operational meaning, not just wording. A pair that is qa_blocked under C is not the same thing presented differently as merge_caution under A. It is a genuinely different operational judgment about the same structural evidence. The proof is the matched pairs: same structure, opposite actions, because of aggregation choice.

---

## Output artifacts

- `sidecar/results/aggregation_sensitive_compatibility/aggregation_comparison.json`
- `sidecar/results/aggregation_sensitive_compatibility/aggregation_comparison.md`
- `sidecar/notes/n83_aggregation_comparison_analysis.md` (this note)
