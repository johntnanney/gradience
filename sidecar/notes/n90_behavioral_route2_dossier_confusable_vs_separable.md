# n90 -- Dossier: Confusable vs Separable

**Type:** case dossier
**Date:** 2026-04-01
**Program:** Behavioral Route 2 Bridge
**Stage:** D
**Depends on:** n86 (panel), n87 (protocol), n88 (findings)
**Status:** complete

---

## Why this comparison matters

The routing-confusable and cross-task separable profiles are the two endpoints of the distributional aggregation gradient. Under distributional aggregation, confusable cases have high overlap (hard to route between), while separable cases have clear gaps (easy to route between). The structural distinction is well-defined.

But do they produce different *behavior*? If a confusable pair looks just like a separable pair at the example level, the distributional gradient might be structurally meaningful but behaviorally empty. If they look different, the distributional gradient has behavioral reality.

---

## Cases compared

| | NM-01 (confusable) | CT-01 (separable) |
|---|---|---|
| **Backbone** | DistilBERT | BERT |
| **Task** | irony (binary) | ag_news/hate (cross) |
| **Task relation** | same_task | cross_task |
| **Source A** | 0.632 | 0.922 |
| **Source B** | 0.618 | n/a (wrong task) |
| **Merged** | 0.620 | 0.826 |
| **Δ vs best** | -0.012 | -0.096 |

### Why NM-01 is the confusable case
Both sources are trained on the same task (irony), have similar performance (~0.62), and produce overlapping predictions. Under distributional aggregation, this pair would be labeled "routing_confusable" — the sources are functionally interchangeable on most examples, making routing between them uninformative.

### Why CT-01 is the separable case
The sources are trained on different tasks (ag_news vs hate). Under every aggregation family, this pair is excluded/separated. The sources are trivially distinguishable — routing is easy because they serve different functions entirely.

---

## Metric comparison

| Metric | NM-01 (confusable) | CT-01 (separable) | Interpretation |
|--------|-----|------|-----|
| Preservation rate | 0.673 | 0.874 | CT-01 preserves more because source A is very strong (0.922) |
| Joint breakage rate | 0.042 | 0.126 | CT-01 breaks more joint-correct examples |
| Neither-source rate | **0.018** | **0.144** | **8x difference** — the key discriminator |
| Confidence collapse | **0** | **3** | Both low, but CT-01 has a few |
| High-conf wrong | **1** | **23** | **The sharpest separation** |
| Better-source loss | 0.491 | n/a | Not comparable (CT-01 has no source B reference) |

### What the numbers show

The two cases are in different behavioral tiers entirely:
- NM-01 is in Tier 1 (no pathology): <2% neither-source, zero confidence collapse
- CT-01 is in Tier 2 (localized pathology): 14.4% neither-source, 23 high-confidence wrong

The behavioral difference is not subtle. It is a categorical separation, not a gradient.

---

## Category distribution

| Category | NM-01 | CT-01 |
|----------|-------|-------|
| A: Preserved stable | 310 (62%) | 413 (83%) |
| Better-source loss | 143 (29%) | — |
| Source A loss | — | 58 (12%) |
| F: Confident misassignment | 1 (0.2%) | 23 (4.6%) |
| Shared failure | 40 (8%) | 29 (6%) |
| D: Collapse | 7 (1.4%) | 0 |

CT-01's dominant failure mode is source A loss (58 examples where the strong source was correct but the merge is wrong) plus 23 confident misassignments. The failures are confident — the model predicts with high confidence and is wrong, presumably because the cross-task adapter's learned features interfere with the primary task's decision structure.

NM-01's dominant non-preserved category is better-source loss (143 examples) — the merge doesn't always preserve the slightly better source's answer. But there are essentially zero pathological failures. The "confusion" that distributional aggregation identifies as routing-relevant does not produce confusion-like example behavior.

---

## The behavioral meaning of structural confusability

The most important finding from this comparison is negative: **structural confusability does not produce behavioral confusion.**

NM-01 is the canonical routing-confusable case — same task, similar performance, high overlap. If routing-confusability had a behavioral signature, it would be diffuse low-confidence predictions, source-mixing, or ambiguous category assignments. Instead, NM-01 looks behaviorally safe-like. The merge handles the overlap constructively.

This has two possible interpretations:

1. **Routing-confusability is a decision-context property, not a behavioral property.** The confusability matters when a routing system must choose between the sources — it's about the difficulty of the routing decision, not about what happens after the decision is made. In a merge (where both sources are averaged), the confusability is harmless.

2. **Behavioral confusion might appear in actual routing scenarios.** If a router assigns examples to one source or the other based on structural similarity, the confusable pair would produce high error rates on the misrouted examples. But this is a routing-system failure, not a model-behavior failure.

Either way, the behavioral signature of confusability is likely to be routing-specific rather than merge-visible. The distributional gradient (confusable vs separable) has structural meaning but its behavioral meaning is decision-context-dependent.

---

## What this means for Route 2

**The confusable/separable distinction is structurally real but behaviorally asymmetric.** Separable cases have a clear behavioral signature (confident contamination). Confusable cases do not have a corresponding behavioral signature in the merge setting — they look safe-like.

**This supports decision-context-dependent aggregation.** The distributional gradient matters for routing (where confusability determines routing difficulty) but not for merge (where confusable sources average constructively). Using distributional aggregation for merge decisions would correctly identify the confusability but incorrectly suggest it matters for merge outcomes.

**The cross-task separable behavioral signature is qualitatively distinct.** CT-01's confident misassignment pattern (23 high-confidence wrong) is not a more extreme version of NM-01's mild information loss. It is a different failure channel entirely. This is the behavioral confirmation that cross-task separation and within-task confusability are not points on the same continuum.

---

## Output artifacts

- `sidecar/notes/n90_behavioral_route2_dossier_confusable_vs_separable.md` (this note)
