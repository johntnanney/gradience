# Behavioral Route 2 Bridge — Summary

**Date:** 2026-04-01
**Program:** Behavioral Route 2 Bridge
**Status:** Complete (5 stages)
**Research notes:** n86-n92

---

## One-line summary

Broadened Route 2 compatibility profiles are not just structural categories — four of five correspond to distinct behavioral signatures observable at the example level, grounding the framework in model behavior.

---

## What we learned

### The core finding

An 8-case panel with 4,000 examples was analyzed through 5 Route 2 profiles. The behavioral signatures cluster into three tiers:

| Tier | Profiles | What it looks like |
|------|----------|-------------------|
| No pathology | Aggregation-invariant safe, same-family optional | Stable, <2% novel failure, no confidence collapse |
| Localized pathology | Worst-case collapse, cross-task separable | ~14% novel failure, with mode split: uncertainty-driven collapse vs confident contamination |
| Stasis | QA-dominant review | Shared failure dominates, merge neither helps nor hurts |

### The most important behavioral distinction

Worst-case collapse and cross-task contamination produce the same rate of novel failure (~14% neither-source) but through different channels:
- **Collapse** → the model becomes uncertain (28-30 confidence collapses, 0 high-confidence wrong)
- **Contamination** → the model becomes confidently wrong (23 high-confidence wrong, 3 confidence collapses)

This justifies decision-context-dependent aggregation: worst-case aggregation detects concentration of pathology (relevant for merge), distributional aggregation detects confusability gradients (relevant for routing).

### Replication update (bounded)

A follow-on replication pass (`n118`-`n122`) re-tested this distinction on nearby case/slice perturbations and found the same qualitative split with guardrails:

- collapse-like targets remained confidence-collapse dominant,
- contamination-like targets remained high-confidence-wrong dominant,
- neither-source pressure remained close across channels and did not collapse the distinction.

### The routing-confusability finding

Structural routing-confusability (same task, similar performance, high overlap) does not produce confusion-like behavior in the merge setting. The confusable case (NM-01) is behaviorally safe-like. The behavioral signature of confusability likely only appears in actual routing scenarios where a system must choose between sources.

---

## Product guidance

### Safe to say

- **Route 2 compatibility profiles have behavioral meaning.** They are not only structural categories — they correspond to observable differences in model behavior.
- **Evidence gating is behaviorally justified.** QA-dominant aggregation identifies cases where there is nothing to preserve or destroy (behavioral stasis), not cases that are structurally risky.
- **Same-family optional cases are behaviorally safe-like.** QA constraints on these cases are about evidence gaps, not behavioral risk.
- **The collapse/contamination distinction matters.** Same failure rate, different failure channels, different operational consequences for downstream users.
- **The collapse/contamination split is replication-supported in bounded scope.** Safe for merge-facing explanatory language with explicit non-universal guardrails.

### Not safe to say

- "Routing-confusability predicts behavioral confusion" — it doesn't, at least in the merge setting.
- "The three-tier model is a validated classification system" — it describes this panel; generalization is untested.
- "These thresholds (<2%, ~14%) are calibrated boundaries" — they are empirical observations in bounded panels.
- "All five profiles have equally strong behavioral signatures" — same-family optional is behaviorally indistinguishable from aggregation-invariant safe.
- "Collapse vs contamination is a universal cross-context law" — current support is strong but bounded to tested merge-facing settings.

---

## Relationship to prior Route 2 programs

| Program | What it established | What behavioral bridge adds |
|---------|--------------------|-----------------------------|
| Cross-artifact (n76-n80) | Portable signals are workflow-level | Workflow signals track behavioral truth (evidence gating = behavioral stasis) |
| Decision-dependent (n70-n74) | Same structure means different things under different decisions | The different meanings correspond to different behavioral failure channels |
| Aggregation-sensitive (n81-n85) | Aggregation is computational, not presentational | Each aggregation family tracks a different behavioral failure channel |

---

## Source artifacts

| Artifact | Location |
|----------|----------|
| Panel definition | `sidecar/notes/n86_behavioral_route2_panel_definition.md` |
| Protocol | `sidecar/notes/n87_behavioral_route2_protocol.md` |
| Findings | `sidecar/notes/n88_behavioral_route2_findings.md` |
| Dossier: optional vs fragile | `sidecar/notes/n89_behavioral_route2_dossier_optional_vs_fragile.md` |
| Dossier: confusable vs separable | `sidecar/notes/n90_behavioral_route2_dossier_confusable_vs_separable.md` |
| Dossier: QA-override stasis | `sidecar/notes/n91_behavioral_route2_dossier_qa_override.md` |
| Bridge synthesis | `sidecar/notes/n92_behavioral_route2_bridge.md` |
| Profile behavior table | `sidecar/results/behavioral_route2_bridge/profile_behavior_table.json` |
| Behavior summary | `sidecar/results/behavioral_route2_bridge/behavior_summary.json` |
| Bridge table | `sidecar/results/behavioral_route2_bridge/behavior_bridge_table.json` |
| Figure | `sidecar/figures/behavioral_route2_profile_matrix.svg` |
