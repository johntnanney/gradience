# n74 — Decision Semantics Bridge

**Type:** synthesis note  
**Date:** 2026-03-31  
**Depends on:** n73 taxonomy, output-example semantics notes (n61/n63/n66), routing pilot, checkpoint triage T02  
**Status:** Stage E complete (bounded)

---

## Question

Do decision-dependent profiles map to distinguishable behavioral manifestations?

---

## Output

- `sidecar/results/decision_dependent_compatibility/decision_semantics_table.json`

---

## Bridge summary

### Merge context (example-level evidence available)

From n61/n63/n66:

- fragile/unsafe regimes show elevated neither-source behavior and confidence collapse,
- cross-task controls can show high-confidence wrong behavior,
- near-miss cases are often behaviorally close to safe retained cases.

These signatures support profile differentiation for merge-facing decisions.

### Routing context (behavioral proxy still structural-operational)

From routing pilot:

- high confusability pairs trigger dedup/disambiguation recommendations,
- low confusability pairs are easily routed.

Routing evidence is currently operational (structural + recommendation), not full example-level misroute telemetry.

### Triage context (follow-through behavior available, small)

From checkpoint T02:

- QA gate dominated final stance,
- same-family follow-through was asymmetric by dataset (`sst2_s42` below base on SST-2; `yelp_s42` above base on Yelp),
- this supports "same-family optional/review" semantics without claiming interchangeability.

---

## Main conclusion

A bounded behavioral bridge exists:

- some signatures are scenario-invariant (quality gating pressure, weak-source sensitivity),
- others are scenario-specific (confidence-collapse vs misroute risk vs QA blocking),
- and these map cleanly to the Stage D profile set.

This is enough for a cautious decision-dependent compatibility framing, but not enough for causal claims across all scenarios.

---

## Open gaps

1. Routing lacks example-level confusion logs (currently recommendation-level only).
2. Triage bridge is based on a small number of follow-through probes.
3. No decoder/generative scenarios were tested.

So Stage E is a **bounded positive**: useful bridge structure with explicit remaining uncertainty.
