# n115 -- Mixed-Evidence Triage Rerun and Aggregation Analysis

**Type:** stress-test findings note  
**Date:** 2026-03-31  
**Program:** Route2 Mixed-Evidence Triage Stress Test  
**Stage:** C  
**Depends on:** n113, n114  
**Status:** complete

---

## Objective

Apply existing triage logic to the mixed-evidence panel and test whether outputs remain interpretable under soft-middle weighting.

---

## Aggregation families used

- worst-case
- distributional
- QA-dominant
- optional reference: QA-gated distributional (existing family only)

No new aggregation family was introduced.

---

## Agreement summary

- `full_agreement`: 0
- `partial_agreement`: 6
- `strong_divergence`: 2

Strong divergence appears where QA-dominant blocks structurally plausible cases (`anchor_blocked_same_task_checkpoint`) or clearly low-value cases (`anchor_blocked_cross_task_checkpoint`).

---

## Key readouts

1. Mixed-evidence review cases remain primarily `qa_review`, not collapse-like.
2. Same-family optional cases stay in clear/review lanes, not blocked lanes.
3. Review-worthy vs low-value differentiation remains visible inside `qa_review` through secondary structural nuance.
4. The middle is blurrier than anchor regions but remains interpretable.

---

## Outputs

- `sidecar/results/route2_stress_tests/mixed_evidence_triage/aggregation_comparison.json`
- `sidecar/results/route2_stress_tests/mixed_evidence_triage/aggregation_comparison.md`
- `sidecar/results/route2_stress_tests/mixed_evidence_triage/triage_outputs.json`
- `sidecar/figures/mixed_evidence_triage_matrix.svg`
- `sidecar/notes/n115_mixed_evidence_triage_rerun.md`
