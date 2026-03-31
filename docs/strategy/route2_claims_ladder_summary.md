# Route 2 Claims Ladder Summary

Date: 2026-03-31  
Status: synthesis complete (confidence calibration pass)

## Purpose

This summary translates the Route 2 claims ladder into communication and workflow guidance.

It answers:

- what is stable enough to state plainly,
- what is usable with guardrails,
- what should stay research-only.

Primary sidecar artifacts:

- `sidecar/notes/n108_route2_claims_inventory.md`
- `sidecar/notes/n109_route2_claim_evidence_map.md`
- `sidecar/notes/n110_route2_claim_dimension_scoring.md`
- `sidecar/notes/n111_route2_claims_stability_ladder.md`
- `sidecar/notes/n112_route2_claims_ladder_implications.md`
- `sidecar/results/route2_claims_ladder/stability_ladder.json`

## Ladder snapshot

- stable: 11 claims
- moderately_stable: 6 claims
- thin: 2 claims
- local_only: 1 claim

## Safe stable Route 2 language

Use these statements confidently (with existing scope bounds):

1. Evidence gating and conservative narrowing remain the strongest cross-artifact workflow invariants.
2. Aggregation is a first-class decision seam.
3. Worst-case, distributional, and QA-dominant aggregation families are operationally distinct.
4. Cross-artifact broadening is stronger at workflow level than metric level.
5. The broadened substrate is real but narrow.

## Guarded-but-usable language

Use with explicit caveats:

1. Same-task vs cross-task directional separation across tested classes.
2. Same-family intermediate and optional states as review-relevant middle states.
3. Aggregation taxonomy usage at coarse level only.
4. Behavioral profile distinctions in current bounded panel.

## Research-only or local language

Keep these out of broad product/public claims for now:

1. Optional/near-miss portability outside LoRA.
2. Routing-confusability behavioral non-transfer as a general rule.
3. Collapse-vs-contamination as a cross-context law (currently local to merge-facing behavioral evidence).

## Route 2 communication framing

### Public writing

- emphasize stable workflow and aggregation seam claims,
- keep scope explicit (`CPU-only`, `small encoder`, `shared base`, `classification`).

### Product and alpha docs

- lead with evidence bootstrap and conservative narrowing,
- describe same-family optional as review-like with guardrails,
- avoid hard threshold language.

### Internal architecture and sidecar work

- keep representation-local metric semantics explicit,
- keep thin/local claims in sidecar framing until strengthened.

## Bottom line

The claims ladder supports stronger and cleaner Route 2 messaging: stable seam-level and workflow-level claims are now clear enough for synthesis, while middle-state and behavioral transfer claims remain intentionally bounded.
