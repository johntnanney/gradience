# Aggregation-Sensitive Compatibility — Route 2 Summary

**Date:** 2026-03-31
**Program:** Aggregation-Sensitive Compatibility (Route 2)
**Status:** Complete (5 stages)
**Research notes:** n81–n85

---

## One-line summary

Different aggregation rules produce genuinely different operational judgments from the same structural evidence — aggregation is a computational step, not a presentation layer.

---

## What we learned

### The core finding

We applied four aggregation families (worst-case, distributional, QA-dominant, QA-gated distributional) to the same 12-case compatibility panel. Only 2/12 cases showed full agreement across all families. The remaining 10 cases were aggregation-sensitive — the same structural evidence produced different operational labels depending on the aggregation rule.

### The five patterns

| Pattern | Frequency | What it means |
|---------|-----------|--------------|
| Aggregation-invariant exclusion | 2/12 | Cross-task exclusion: every family agrees. No aggregation sophistication needed. |
| Distributional gradient | 4/12 | The confusable/moderate/separable ordering is visible only under distributional aggregation. Worst-case collapses it. |
| QA dominance override | 2/12 | QA blocks structurally positive cases. Structure is necessary but not sufficient. |
| QA-gated enrichment | 3/12 | When QA clears, the hybrid preserves both evidence constraint and structural gradient. |
| Mixed evidence nuance | 1/12 | QA-dominant is not binary. Mixed evidence produces a third state (review). |

### The sharpest divergences

1. **Routing gradient destruction:** Worst-case assigns the same label to confusable, moderate, and separable pairs. Distributional correctly separates them. For routing, worst-case is uninformative.

2. **Structural-QA contradiction:** The pair with the highest compatibility score in the panel (0.892) is blocked by QA. Structural truth and operational truth flatly contradict.

---

## Product guidance

### Safe to say

- **Aggregation family should match decision context.** Merge → worst-case. Routing → distributional. Triage → QA-dominant. General-purpose → QA-gated distributional (hybrid).
- **Cross-task exclusion is aggregation-invariant.** No special handling needed for cross-task cases with clear QA.
- **The hybrid is the correct default.** It never loses information relative to any single family.

### Not safe to say

- "This aggregation family is always best" — it depends on what you're deciding.
- "These five patterns are exhaustive" — they describe this panel; generalization is unvalidated.
- "Aggregation family can be auto-selected from metadata" — predictors identified but not validated at scale.

---

## Relationship to cross-artifact findings

Cross-artifact compatibility (n76-n80) found that portable signals are workflow-level, not metric-level. Aggregation sensitivity reinforces this: even within one artifact class, the same metrics produce different operational judgments depending on aggregation choice. The stable layer is the workflow design (evidence gating, context-appropriate aggregation), not the numeric outputs.

---

## Source artifacts

| Artifact | Location |
|----------|----------|
| Panel definition | `sidecar/notes/n81_aggregation_panel_definition.md` |
| Family audit | `sidecar/notes/n82_aggregation_family_audit.md` |
| Comparison analysis | `sidecar/notes/n83_aggregation_comparison_analysis.md` |
| Pattern taxonomy | `sidecar/notes/n84_aggregation_sensitive_pattern_taxonomy.md` |
| Operational implications | `sidecar/notes/n85_aggregation_sensitive_operational_implications.md` |
| Panel data | `sidecar/results/aggregation_sensitive_compatibility/panel_table.json` |
| Comparison data | `sidecar/results/aggregation_sensitive_compatibility/aggregation_comparison.json` |
| Taxonomy data | `sidecar/results/aggregation_sensitive_compatibility/pattern_taxonomy.json` |
