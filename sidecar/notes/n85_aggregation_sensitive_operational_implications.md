# n85 -- Aggregation-Sensitive Compatibility: Operational Implications

**Type:** assessment note
**Date:** 2026-03-31
**Program:** Aggregation-Sensitive Compatibility (Route 2)
**Stage:** E
**Depends on:** n81 (panel), n82 (family audit), n83 (comparison), n84 (taxonomy)
**Status:** complete

---

## Question

What do the aggregation-sensitivity findings mean for Gradience's operational compatibility workflows? What is safe to build on, what requires caution, and what remains research-only?

---

## Summary of findings

1. **Aggregation is not presentation.** Different aggregation rules produce genuinely different operational judgments from the same structural evidence. A pair labeled merge_caution under worst-case and qa_blocked under QA-dominant is not the same thing described differently — it is a different operational truth.

2. **Five stable patterns.** Aggregation sensitivity clusters into five recognizable patterns, predictable from two observable features (QA regime and task relation).

3. **The hybrid is the richest.** QA-gated distributional produces the most nuanced output: it respects evidence constraints AND preserves distributional gradation. It is not a compromise — it is the natural workflow.

4. **Cross-task exclusion is aggregation-invariant.** When structural risk and task boundary both point to exclusion, no aggregation rule disagrees. This is the floor of the problem.

5. **The routing gradient is distributional-only.** The confusable/moderate/separable ordering that matters for routing is invisible under worst-case aggregation.

---

## Operational implications

### Safe to build on

**1. Aggregation family selection should be decision-context-dependent.**

Merge workflows should use worst-case (or QA-gated worst-case) because a single bad layer can cause catastrophic degradation. Routing workflows should use distributional because confusability depends on prevalence, not worst-case. Triage workflows should use QA-dominant because proceeding without evidence is the dominant risk.

This is not a preference — it is a consequence of what each decision context optimizes for.

**2. The hybrid (QA-gated distributional) is the correct default for general-purpose compatibility.**

When the decision context is unknown or mixed, the hybrid preserves the most information. It never produces less information than any single family. It is the only family that respects both evidence gating and structural gradation.

**3. Cross-task exclusion does not require aggregation sophistication.**

For cross-task pairs with clear QA, any aggregation family produces the correct judgment. Resource allocation for aggregation design should focus on same-task and same-family cases, where aggregation choice genuinely changes the operational outcome.

### Guarded — requires further validation

**4. Aggregation family auto-selection from case metadata.**

The taxonomy shows patterns are predictable from QA regime and task relation. This suggests a system could auto-select the aggregation family. But the panel is 12 cases on one backbone — the predictors need validation on a larger and more diverse panel before this becomes operational.

**5. The near-miss distinction as distributional signal.**

The near-miss (needs_disambiguation) label is visible only under distributional aggregation. It is operationally important for inventory triage. But the threshold between confusable and needs_disambiguation is not yet well-calibrated across artifact classes.

### Research-only — not ready for product

**6. Aggregation-sensitive pattern taxonomy as a classification system.**

The five patterns are descriptive of this panel. Whether they generalize to other panels, backbones, or artifact classes is unknown. They should not be hardcoded as a classification system.

**7. Numeric agreement rates as quality metrics.**

"2/12 full agreement" and "10/12 aggregation-sensitive" are properties of this panel's construction, not of the underlying phenomenon. Different panel designs would produce different ratios.

---

## What this means for Route 2

Route 2 asked: "How do different aggregation rules transform the same structural measurements into different operational compatibility judgments?"

The answer: aggregation is a genuine computational step that selects which structural truths become operative. The selection is not arbitrary — it is determined by the decision context (merge vs routing vs triage). The appropriate aggregation family is a function of what the practitioner is trying to decide, not a property of the adapter pair.

This finding is stable enough to inform workflow design but not yet stable enough to automate aggregation family selection.

---

## Relationship to cross-artifact findings (n76-n80)

The cross-artifact program found that compatibility exists primarily at the level of workflow invariants (Layer 1), not structural metrics (Layer 2). The aggregation program adds: even within Layer 2, the same structural metrics produce different operational judgments depending on aggregation choice. This reinforces the cross-artifact finding — structural compatibility is representation-local AND aggregation-dependent. The stable, portable layer is the workflow (evidence gating, decision-context-appropriate aggregation), not the numbers.

---

## Output artifacts

- `sidecar/notes/n85_aggregation_sensitive_operational_implications.md` (this note)
- `docs/strategy/aggregation_sensitive_route2_summary.md`
- `sidecar/results/aggregation_sensitive_compatibility/operational_implications.json`
