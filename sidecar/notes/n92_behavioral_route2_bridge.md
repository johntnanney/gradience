# n92 -- Behavioral Route 2 Bridge

**Type:** synthesis note
**Date:** 2026-04-01
**Program:** Behavioral Route 2 Bridge
**Stage:** E
**Depends on:** n86-n91 (panel, protocol, findings, dossiers), n70-n74 (decision-dependent), n76-n80 (cross-artifact), n81-n85 (aggregation-sensitive)
**Status:** complete

---

## Question

What does Route 2 compatibility science now mean behaviorally?

---

## The short answer

Broadened Route 2 compatibility profiles are not just structural categories — at least four of five have distinct behavioral signatures observable at the example level. The framework is grounded in model behavior, not only in structural measurement and architectural reasoning.

---

## Which profiles have the clearest behavioral signatures?

### Strong behavioral signatures

**Worst-case collapse** has the strongest behavioral footprint: concentrated breakage, confidence collapse (28-30 examples), elevated neither-source behavior (~15%), and progressive intensification as the weak source gets weaker. The pathology is localized (specific examples fail sharply) rather than diffuse (general degradation). This is the V-module pathology signature expressed at the example level.

**Cross-task separable** has an equally strong but qualitatively distinct signature: confident misassignment (23 high-confidence wrong predictions) rather than uncertainty-driven collapse (3 confidence collapses vs 30 in worst-case). The merge model doesn't know it's wrong. This is the readout contamination signature expressed at the example level.

**QA-dominant review** has a clear signature of a different kind: behavioral stasis. Shared failure dominates (65% of examples), the merge neither helps nor hurts, and all pathological metrics are zero. The behavioral content is absence — there is nothing to preserve or destroy.

### Moderate behavioral signature

**Aggregation-invariant safe** has a clear signature (zero pathology, stable preservation) but it is a negative signature — the absence of all failure modes. It is identifiable but not distinctive in the way that collapse or contamination are distinctive.

### No distinct behavioral signature (in the merge setting)

**Same-family optional (routing-confusable)** is behaviorally indistinguishable from aggregation-invariant safe on all discriminating metrics. The structural near-miss classification and the routing-confusability label do not produce a behavioral footprint in the merge setting. This is not a failure of the profile — it means the behavioral signature of confusability is likely to be routing-specific (appearing when a router must choose between sources) rather than merge-visible (appearing in the averaged model).

---

## Which profile distinctions remain mostly structural/architectural?

Two distinctions are real at the structural level but not visible at the behavioral level in the current panel:

1. **Aggregation-invariant safe vs same-family optional.** Both profiles are in Tier 1 (no pathology). The distinction matters for evidence gating and QA policy, not for behavioral outcome.

2. **Routing-confusable vs routing-needs-disambiguation vs routing-separable.** The distributional gradient (the three-tier ordering revealed by distributional aggregation in n83) does not produce three corresponding behavioral tiers in the merge setting. In the merge setting, confusable looks safe and separable looks like cross-task exclusion. The intermediate state (needs-disambiguation) would need to be tested in an actual routing scenario.

These structural/architectural distinctions are not "wrong" — they identify real properties of the adapter pair. But their operational meaning is decision-context-dependent, and their behavioral signature may only appear in the decision context they are designed for.

---

## Connecting to prior Route 2 programs

### Cross-artifact compatibility (n76-n80)

The cross-artifact program found that portable compatibility signals live at the workflow level (evidence gating, conservative narrowing), not at the structural metric level. The behavioral bridge adds: **the workflow-level signals have behavioral reality.** Evidence gating (QA-dominant aggregation) correctly identifies behavioral stasis (AN-01). Conservative narrowing (same-family optional treated as safe-but-gated) correctly reflects behavioral safety (NM-01/NM-02). The workflow is portable because it tracks behavioral truth, not because it is abstractly convenient.

### Decision-dependent compatibility (n70-n74)

The decision-dependent program established that the same structural relation means different things under merge, routing, and triage. The behavioral bridge adds: **the decision-dependent interpretation is behaviorally grounded.** The same pair (NM-01) is structurally confusable (relevant for routing) and behaviorally safe (relevant for merge). These are not contradictions — they are properties of different decision contexts acting on the same structural truth.

### Aggregation-sensitive compatibility (n81-n85)

The aggregation program found five stable aggregation-sensitive patterns and established that aggregation is a computational step, not a presentation layer. The behavioral bridge adds: **the aggregation families track different behavioral failure channels.** Worst-case aggregation tracks concentration of pathology (collapse). Distributional aggregation tracks the confusability gradient (routing difficulty). QA-dominant aggregation tracks evidence presence (stasis vs content). Each family is behaviorally appropriate for its decision context because it is sensitive to the failure channel that matters for that context.

---

## The three-tier behavioral model

The five profiles group into three behavioral tiers:

| Tier | Profiles | Behavioral signature | Discriminating metrics |
|------|----------|---------------------|----------------------|
| 1: No pathology | Aggregation-invariant safe, same-family optional | Stable preservation, no novel failure | Neither-source <2%, confidence collapse = 0 |
| 2: Localized pathology | Worst-case collapse, cross-task separable | Concentrated failure, two distinct channels | Neither-source ~14%, with mode split: collapse (confidence <0.4) vs contamination (confidence >0.7) |
| 3: Stasis | QA-dominant review | Shared failure, no merge-induced change | Neither-source 7%, shared failure 65%, all pathology metrics = 0 |

The tier boundaries are sharp, not gradual. The Tier 1/Tier 2 boundary is the <2% vs ~14% neither-source threshold identified in n63. The Tier 2/Tier 3 boundary is the presence vs absence of directional pathology (collapse or contamination vs stasis).

---

## What is safe to say now

1. **Route 2 compatibility profiles have behavioral reality.** Four of five profiles correspond to distinct, identifiable behavioral signatures. The fifth (routing-confusable) likely has a behavioral signature that is decision-context-specific.

2. **The collapse/contamination split is the most operationally important behavioral finding.** Same neither-source rate (~14%), different failure channels, different operational consequences. This justifies decision-context-dependent aggregation at the behavioral level.

3. **Evidence gating is behaviorally grounded.** QA-dominant aggregation identifies behavioral stasis (nothing to preserve or destroy), not structural weakness. It is correct for the right reason.

4. **Same-family optional is behaviorally safe-like.** The near-miss structural classification does not predict behavioral pathology. QA constraints on these cases are about evidence gaps, not behavioral risk.

5. **The three-tier model is a useful behavioral summary.** No pathology / localized pathology / stasis captures the behavioral landscape of the current panel.

---

## What remains thin

1. **Routing-confusability behavioral signature.** NM-01 does not show confusion-like behavior in the merge setting. The behavioral signature of routing-confusability may require an actual routing scenario to manifest.

2. **Single-panel evidence base.** All behavioral findings come from 8 cases on 2 backbones (DistilBERT, BERT) and 4 tasks. The three-tier model has not been tested on a broader panel.

3. **No non-LoRA behavioral data.** The cross-artifact program (n76-n80) covers LoHa and checkpoint deltas structurally, but per-example behavioral data exists only for LoRA. Whether the behavioral tiers transfer across artifact classes is unknown.

4. **The Tier 2 mode split is well-evidenced but single-instance for contamination.** CT-01 is the only cross-task case. The confident-misassignment channel needs replication.

---

## What should remain sidecar-only

1. **The three-tier model as a classification system.** It describes this panel. Whether it generalizes is untested.

2. **Specific neither-source thresholds (<2% vs ~14%).** These are empirical observations on this panel, not calibrated boundaries.

3. **The "stasis" interpretation of QA-dominant review.** AN-01 is a specific case (both sources near chance). Other QA-blocked cases with stronger sources might show different behavioral patterns.

---

## Output artifacts

- `sidecar/notes/n92_behavioral_route2_bridge.md` (this note)
- `sidecar/results/behavioral_route2_bridge/behavior_bridge_table.json`
- `docs/strategy/behavioral_route2_summary.md`
