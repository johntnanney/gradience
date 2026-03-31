# n72 — Aggregation-Sensitive Compatibility

**Type:** analysis note  
**Date:** 2026-03-31  
**Depends on:** n70 panel, n71 audit, checkpoint triage T02 artifacts  
**Status:** Stage C complete

---

## Question

What changes when the same structural inputs are aggregated as:

1. worst-case (merge-like),
2. distributional (routing-like),
3. QA-gate-first (triage-like)?

---

## Outputs

- `sidecar/results/decision_dependent_compatibility/aggregation_comparison.json`
- `sidecar/results/decision_dependent_compatibility/aggregation_comparison.md`
- `sidecar/figures/decision_dependent_aggregation_matrix.svg`
- `sidecar/results/decision_dependent_compatibility/aggregation_comparison_adapter_t01.json` (narrow adapter-triage stress test)
- `sidecar/figures/decision_dependent_aggregation_matrix_adapter_t01.svg`

Panel used: **10 checkpoint pairs** from `field_trials/checkpoint_inventory_t02/`.

---

## Aggregation families

### A — Worst-case (merge-like)

- emphasizes local catastrophic sensitivity
- produces conservative pair risk categories

### B — Distributional (routing-like)

- emphasizes spread across the pair-level structure
- separates confusable vs disambiguate vs separable cases

### C — QA-gate-first (triage-like)

- source quality can override pair structure
- prioritizes trust/evidence over geometric plausibility

---

## Key results

From `aggregation_comparison.json`:

- Worst-case labels: `merge_risky=8`, `merge_caution=2`
- Distributional labels: `routing_confusable=1`, `routing_needs_disambiguation=7`, `routing_separable=2`
- QA gate-first labels: `qa_blocked=9`, `qa_review=1`

Two strong contrasts appear:

1. **Worst-case collapse vs distributional separation**
   - both worst-case buckets split into multiple routing-like labels.
   - same structural panel, different operational structure.

2. **QA override**
   - QA gate overrode structurally non-separable cases on 8 pairs.
   - effective QA dominance ratio in this panel: **1.0** (all pairs blocked/review).

---

## What this means

Aggregation is not cosmetic. It changes what is operationally visible:

- worst-case exposes catastrophic sensitivity,
- distributional exposes separability gradient,
- gate-first exposes source-trust constraints.

That directly supports RQ3 and H2: aggregation strategy is a real mechanism of decision dependence.

---

## Limits

This stage uses one QA-heavy checkpoint inventory (T02). So:

- QA dominance should be interpreted as a panel property plus scenario design, not a universal constant.
- the result is still useful: it demonstrates a real regime where pairwise structure is subordinate to evidence quality.

### Addendum: adapter-triage stress pass

To stress-test profile stability on a second triage substrate, a narrow adapter pass was run on targeted confirmation T01 (`field_trials/targeted_confirmation_same_family/inventory_t01/preflight`).

Summary:

- worst-case labels: `merge_caution=2`, `merge_risky=1`
- distributional labels: `routing_confusable=1`, `routing_needs_disambiguation=1`, `routing_separable=1`
- QA gate labels: `qa_clear=3` (`qa_dominance_ratio=0.0`)

Interpretation:

- same-task / same-family / cross-task separation under distributional aggregation remains intact,
- QA regime contrast flips as expected (clear in adapter T01, dominant in checkpoint T02),
- this supports stability of the six-profile picture in bounded scope.
