# Route 2 Claims Ladder Summary

Date: 2026-04-03  
Status: synthesis complete + reinforcement communication policy active + rank-proxy bounded policy freeze

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
- `sidecar/notes/n113_mixed_evidence_triage_baseline.md` through `sidecar/notes/n117_mixed_evidence_triage_stress_test_memo.md` (post-ladder soft-middle stress reinforcement)
- `sidecar/notes/n118_collapse_vs_contamination_baseline.md` through `sidecar/notes/n122_collapse_vs_contamination_replication_memo.md` (bounded behavioral channel replication reinforcement)
- `sidecar/notes/n123a_route2_claims_ladder_reinforcement_baseline.md` (reinforcement freeze baseline)
- `sidecar/notes/n123b_route2_claim_reinforcement_map.md` (claim-level reinforcement impact map)
- `sidecar/notes/n123_route2_claims_edge_refinement_r1.md` (R1 edge-case communication refinement overlay)
- `sidecar/notes/n123d_route2_communication_policy.md` (communication policy synthesis)
- `sidecar/results/route2_claims_ladder/stability_ladder.json`
- `sidecar/results/route2_claims_ladder/edge_refinement_table.json`
- `sidecar/results/route2_claims_ladder/reinforcement_baseline_snapshot.json`
- `sidecar/results/route2_claims_ladder/reinforcement_impact_map.json`
- `sidecar/results/route2_claims_ladder/communication_policy.json`
- `docs/strategy/route2_communication_policy_summary.md`

## Ladder snapshot

- stable: 11 claims
- moderately_stable: 7 claims
- thin: 2 claims
- local_only: 0 claims

## Edge refinement snapshot (R1)

- core_stable_non_edge: 8 claims
- stable_but_local: 5 claims
- moderate_but_product_relevant: 5 claims
- thin_suppress_public: 2 claims

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
5. Collapse-vs-contamination as a bounded merge-facing channel distinction.
6. Rank-proxy bounded validation claim:
In the compressible encoder subset, Gradience spectral rank policies are competitive fixed-budget compression guides; allocation structure aligns with a structurally meaningful importance notion, while gradient remains the stronger operational target under the current CPU protocol due to substantially higher resampling stability. Operationally, use gradient as primary comparator and attenuate as companion ablation proxy.

## Research-only or local language

Keep these out of broad product/public claims for now:

1. Optional/near-miss portability outside LoRA.
2. Routing-confusability behavioral non-transfer as a general rule.
3. Collapse-vs-contamination as a broad cross-context law beyond tested merge-facing scope.
4. Rank-reduction ablation expansion as a currently viable comparator in this encoder/compressible regime.

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

## Post-ladder reinforcement

The mixed-evidence triage stress test (`n113`-`n117`) supports the ladder's core middle-state posture: stronger confidence at family/structure level, guarded confidence at threshold/fine-ranking level.

The collapse-vs-contamination replication pass (`n118`-`n122`) supports the ladder's behavioral posture: channel-level distinction strengthened in bounded merge-facing settings, while universality claims remain guarded.

The edge-refinement pass (`n123`) converts this calibration into explicit communication policy: thin claims are suppressed from public language, stable-but-local claims are bounded, and moderate-but-product-relevant claims are allowed with guardrails.

## Bottom line

The claims ladder supports stronger and cleaner Route 2 messaging: stable seam-level and workflow-level claims are now clear enough for synthesis, while middle-state and behavioral transfer claims remain intentionally bounded.

Addendum (2026-04-03):
The rank-proxy line is frozen as a bounded internal validation claim in this ladder's guarded-but-usable tier, with explicit scope limits to compressible encoder families and CPU protocol conditions. Proxy policy is frozen as: `gradient` operational default, `attenuate` companion ablation proxy, `rank_reduction` paused for expansion in this regime.
