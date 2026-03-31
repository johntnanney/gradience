# Decision-Dependent Compatibility Implications

Date: 2026-03-31  
Status: Route 2 consolidation note (post n70-n74)

## Settled architecture implication

The stable architecture story is now:

- measurement is mostly reusable,
- diagnosis is reusable with scenario translation,
- aggregation is the first major scenario-specific seam,
- policy is the second scenario-specific seam.

This has been shown across merge, routing, and triage in bounded CPU evidence.

## What is conceptual only (for now)

- The six decision profiles are a research taxonomy, not a UI taxonomy.
- Decision-profile labels should not be treated as hard product categories without new external use pressure.

## What is architecture-relevant now

1. Keep a four-layer stack in docs and internal reasoning:
- measurement,
- diagnosis,
- aggregation,
- policy.

2. Preserve shared measurement artifacts and relation tags across scenarios.
3. Keep aggregation configurable per scenario objective (worst-case, distributional, gate-first).

## What is potentially product-relevant later

- Scenario-specific report modes that share measurement inputs but expose different action language.
- Explicit QA-regime signaling to distinguish QA-clear vs QA-dominant blocking inventories.
- Optional decision-profile summaries when a real workflow demands them.

## Boundaries

- No forced product UI changes from the sidecar taxonomy.
- No expansion beyond bounded scenarios without new practical confirmation.
- Keep this line frozen unless a new external use case, GPU-backed validation, or contradiction appears.

Primary evidence: `sidecar/notes/n70_decision_dependent_panel_definition.md` through `sidecar/notes/n74_decision_semantics_bridge.md` and `sidecar/results/decision_dependent_compatibility/`.
