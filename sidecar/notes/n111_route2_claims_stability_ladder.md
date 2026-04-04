# n111 -- Route 2 Claims Stability Ladder

**Type:** synthesis result note  
**Date:** 2026-04-01  
**Program:** Route 2 Claims Stability Ladder  
**Stage:** D  
**Depends on:** n108-n110, n123 (R1 edge refinement)  
**Status:** complete

---

## Objective

Assign final ladder status for each Route 2 claim using the fixed five-dimension scoring scheme.

---

## Final ladder distribution

- `stable`: 11 claims
- `moderately_stable`: 7 claims
- `thin`: 2 claims
- `local_only`: 0 claims
- `blocked_or_open`: 0 claims

This distribution matches the current Route 2 pattern: seam-level and workflow-level claims are strongest, while optionality portability and some behavioral transfer claims remain guarded. The collapse-vs-contamination claim was promoted from local-only to moderately-stable after bounded replication (n118-n122).

## R1 edge-refinement overlay

Secondary edge buckets (no ladder-status change):

- `core_stable_non_edge`: 8
- `stable_but_local`: 5
- `moderate_but_product_relevant`: 5
- `thin_suppress_public`: 2

The overlay clarifies communication posture rather than scientific status: stable local claims stay bounded, product-relevant moderate claims stay guardrailed, and thin claims are explicitly suppressed from public language.

---

## Main ladder readout

### Stable cluster

Most stable claims are workflow invariants and aggregation seam claims:

- evidence gating
- conservative narrowing
- aggregation as seam
- distinct aggregation families
- workflow-level portability over metric-level portability

### Moderately stable cluster

Most moderate claims are middle-state and threshold-sensitive claims:

- same-family intermediate behavior
- same-family optional safe-likeness
- aggregation taxonomy usability with bounded thresholds
- broader behavioral profile mapping in current scope

### Thin and local cluster

- Cross-artifact optional/near-miss portability remains thin.
- Routing-confusability behavioral non-transfer remains thin.
- Collapse-vs-contamination channel distinction is now replication-supported and treated as moderately-stable with guardrails in merge-facing scope.

---

## Deliverables

- `sidecar/results/route2_claims_ladder/stability_ladder.json`
- `sidecar/results/route2_claims_ladder/stability_ladder.md`
- `sidecar/results/route2_claims_ladder/edge_refinement_table.json`
- `sidecar/results/route2_claims_ladder/edge_refinement_table.md`
- `sidecar/figures/route2_claims_ladder.svg`

---

## Stage D assessment

The ladder is now explicit enough to drive communication choices (public/product/research) without flattening bounded or local findings into over-broad claims.
