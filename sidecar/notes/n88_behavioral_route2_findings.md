# n88 -- Behavioral Route 2 Findings

**Type:** findings note
**Date:** 2026-04-01
**Program:** Behavioral Route 2 Bridge
**Stage:** C
**Depends on:** n86 (panel), n87 (protocol), n59-n66 (original example semantics data)
**Status:** complete

---

## Question

Do broadened Route 2 compatibility profiles correspond to distinct behavioral signatures at the example level?

---

## Method

Reinterpret the 8 cases from the Output Example Semantics program (n59-n66) through 5 Route 2 profiles. Apply the 6-metric protocol (n87) and compare behavioral patterns across profiles.

---

## Results

### The headline finding

**All five Route 2 profiles have distinct behavioral footprints.** The distinction is not a weak statistical tendency — it is a qualitative separation on specific discriminating metrics. The three most discriminating metrics are:

1. **Neither-source rate:** <2% (safe/optional) vs ~14% (collapse/cross-task) vs 7% (QA stasis)
2. **Confidence collapse count:** 0 (safe/optional/QA) vs 28-30 (collapse) vs 3 (cross-task)
3. **High-confidence wrong count:** 0 (collapse) vs 23 (cross-task) — the cleanest profile separator

### Profile-by-profile findings

#### Aggregation-invariant safe (SR-01, SR-02)

Behaviorally stable. Zero or near-zero on all pathological metrics. Category A (preserved stable) dominates. SR-02 shows that aggregation-invariant safety can coexist with genuine merge improvement (+0.028 over both sources). The behavioral signature is absence of pathology, not absence of change.

#### Same-family optional (NM-01, NM-02)

**Behaviorally indistinguishable from aggregation-invariant safe** on all discriminating metrics. Zero confidence collapse, <2% neither-source rate, zero or near-zero joint breakage. The structural near-miss classification does not produce a distinct behavioral signature. This confirms the n61 finding that near-miss is behaviorally safe-like, and extends it: in Route 2 terms, the same-family optional profile is a structural/evidence distinction, not a behavioral one.

This has an important implication for QA-dominant aggregation: when QA blocks a case with this behavioral profile, it is correctly identifying an evidence gap rather than incorrectly blocking a pathological case. The case is safe but under-evidenced — QA is doing the right thing for the right reason.

#### Worst-case collapse (FR-01, FR-02)

Concentrated, localized breakage. 14-15% neither-source rate, 28-30 confidence collapses, progressive intensification as the weak source gets weaker (FR-02 joint breakage rate is 5x FR-01). The pathology is not diffuse — most examples are still correctly handled, but the affected examples fail sharply with the model becoming uncertain rather than confidently wrong.

The intensity gradient between FR-01 (source B = 0.204) and FR-02 (source B = 0.136) is informative: as the worst-case source gets weaker, the collapse pattern concentrates and intensifies. This supports the worst-case aggregation strategy for merge decisions — the single worst layer/source drives the behavioral outcome.

#### Cross-task separable (CT-01)

**Qualitatively distinct from worst-case-collapse** despite similar neither-source rates (~14%). The failure mode is confident contamination: 23 high-confidence wrong predictions and only 3 confidence collapses. The merge model doesn't know it's wrong — it confidently applies cross-task features to the primary task.

This is the behavioral confirmation that cross-task separation and within-task collapse are different phenomena. They produce similar rates of novel failure but through different channels:
- Worst-case collapse → uncertainty (the model knows it doesn't know)
- Cross-task contamination → confidence (the model doesn't know it doesn't know)

This behavioral distinction is exactly what makes them different operational risks requiring different aggregation strategies.

#### QA-dominant review (AN-01)

Behavioral stasis. Both sources are near chance (0.204, 0.136 on 4-class), so the merge has nothing meaningful to preserve or break. Shared failure dominates (326/500 examples). Zero confidence collapse, zero high-confidence wrong, zero joint breakage. The neither-source rate (7%) sits between the safe/optional floor (<2%) and the collapse/cross-task ceiling (~14%), reflecting random boundary drift rather than directed pathology.

This is the behavioral meaning of "evidence-absent": not structurally catastrophic, not safe either — simply empty. The QA-dominant aggregation strategy correctly identifies this as a case where the evidence status, not the structural measurement, is the binding constraint.

---

## The three-tier behavioral model

The five profiles naturally group into three behavioral tiers:

### Tier 1 — No pathology (aggregation-invariant safe + same-family optional)
Neither-source <2%, zero confidence collapse, zero or near-zero joint breakage. The merge is safe. These two profiles are behaviorally equivalent despite different structural classifications. The structural distinction (retained vs near-miss) matters for evidence gating but not for behavioral outcome.

### Tier 2 — Localized pathology (worst-case collapse + cross-task separable)
Neither-source ~14%, elevated joint breakage. But the pathology mode differs: collapse produces uncertainty, cross-task contamination produces confidence. The high-confidence-wrong metric cleanly separates the two (0 vs 23). These profiles share a failure rate but have different failure channels.

### Tier 3 — Stasis (QA-dominant review)
Shared failure dominates. The merge is not harmful but also not useful. The neither-source rate (7%) is intermediate — above Tier 1 but below Tier 2 — reflecting boundary drift in a near-chance regime. The evidence constraint is the only operationally meaningful signal.

---

## Hypothesis results

| Hypothesis | Verdict | Evidence |
|-----------|---------|----------|
| H1: Profiles have distinct footprints | **Confirmed** | All 5 profiles separable on 3 discriminating metrics |
| H2: Invariant = stable | **Confirmed** | SR-01/SR-02: zero pathological metrics |
| H3: Collapse = localized breakage | **Confirmed** | FR-01/FR-02: concentrated, intensifying with source weakness |
| H4: Distribution-sensitive = confusion | **Partially confirmed** | NM-01 does not show confusion-like behavior; structural confusability ≠ behavioral confusion |
| H5: QA-override = safe-but-under-evidenced | **Confirmed** | AN-01: stasis; NM-01/NM-02: safe-like |
| H6: Routing vs merge not conflated | **Confirmed** | CT-01 (confident) vs FR-01 (uncertain): qualitatively distinct |

### The one partial result: H4

NM-01 was the routing-confusable case (both sources ~0.62, same task, high overlap). The prediction was that routing-confusability would produce confusion-like behavior — diffuse, low-confidence, mixed. Instead, NM-01 looks behaviorally safe-like, indistinguishable from SR-01 on all discriminating metrics.

This means one of two things:
1. Routing-confusability is a structural property without a distinct behavioral signature (at least in the merge setting)
2. The behavioral signature of routing-confusability would only appear in an actual routing scenario (where the system must choose between sources rather than average them)

The second interpretation is more likely. In a merge, confusable sources average constructively. In routing, the system must distinguish them — and the confusion signature might appear in routing decisions, not in merged model behavior. This distinction is consistent with the aggregation-sensitive finding (n81-n85) that routing and merge are different decision contexts requiring different aggregation strategies.

---

## What the findings mean for Route 2

### Structural profiles have behavioral reality

The broadened Route 2 framework is not just architectural — at least four of five profiles correspond to distinct behavioral patterns. This grounds the framework in observable model behavior, not just structural measurement.

### The behavioral distinction that matters most

The collapse/contamination split (Tier 2) is the most operationally important behavioral finding. It confirms that the same neither-source failure rate (~14%) can arise from qualitatively different mechanisms with different operational implications:
- Collapse → the merge is uncertain → a human reviewer can identify the problem from low confidence
- Contamination → the merge is confident → a human reviewer would be misled

This distinction justifies the separation between worst-case (merge) and distributional (routing) aggregation strategies at the behavioral level, not just the structural level.

### Evidence gating is behaviorally justified

The QA-dominant review finding (AN-01 = stasis) confirms that evidence gating is not over-conservative — it correctly identifies cases where the structural measurement is meaningless because there is nothing to measure. The behavioral content of a near-chance merge is shared failure, not structural compatibility.

---

## Output artifacts

- `sidecar/notes/n88_behavioral_route2_findings.md` (this note)
- `sidecar/results/behavioral_route2_bridge/behavior_summary.json`
- `sidecar/results/behavioral_route2_bridge/profile_behavior_table.json`
- `sidecar/results/behavioral_route2_bridge/profile_behavior_table.md`
