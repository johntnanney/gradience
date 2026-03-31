# n111 -- Route 2 Claims Stability Ladder

**Type:** synthesis result note  
**Date:** 2026-03-31  
**Program:** Route 2 Claims Stability Ladder  
**Stage:** D  
**Depends on:** n108-n110  
**Status:** complete

---

## Objective

Assign final ladder status for each Route 2 claim using the fixed five-dimension scoring scheme.

---

## Final ladder distribution

- `stable`: 11 claims
- `moderately_stable`: 6 claims
- `thin`: 2 claims
- `local_only`: 1 claim
- `blocked_or_open`: 0 claims

This distribution matches the current Route 2 pattern: seam-level and workflow-level claims are strongest, while optionality portability and some behavioral transfer claims remain guarded.

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
- Collapse-vs-contamination channel distinction is strong in current panel but still local to current merge-facing behavioral coverage.

---

## Deliverables

- `sidecar/results/route2_claims_ladder/stability_ladder.json`
- `sidecar/results/route2_claims_ladder/stability_ladder.md`
- `sidecar/figures/route2_claims_ladder.svg`

---

## Stage D assessment

The ladder is now explicit enough to drive communication choices (public/product/research) without flattening bounded or local findings into over-broad claims.
