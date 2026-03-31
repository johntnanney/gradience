# n110 -- Route 2 Claim Dimension Scoring

**Type:** synthesis staging note  
**Date:** 2026-03-31  
**Program:** Route 2 Claims Stability Ladder  
**Stage:** C  
**Depends on:** n109 claim evidence map  
**Status:** complete

---

## Objective

Score each Route 2 claim on five fixed dimensions:

1. evidence base
2. perturbation survival
3. artifact coverage
4. behavioral grounding
5. product relevance

---

## Scoring approach

The scoring is disciplined judgment constrained by explicit source mapping.

- No dimension was assigned without source-backed justification.
- Perturbation survival uses existing stability work where available.
- Claims not directly stress-tested were marked as such rather than inferred upward.

---

## Key calibration choices

- Strong workflow-level claims with repeated perturbation support were scored highest.
- Behavioral claims were scored down on artifact coverage when evidence was panel-local.
- Representation-local claims were treated as strong if the locality conclusion itself was replicated.
- Product relevance distinguishes `safe_to_expose` from `safe_with_guardrails` to prevent overclaim drift.

---

## Deliverables

- `sidecar/results/route2_claims_ladder/claim_scoring.json`
- `sidecar/results/route2_claims_ladder/claim_scoring.md`

---

## Stage C assessment

All 20 claims now have complete five-dimension scoring with explicit short-form justifications and no hidden scoring logic.
