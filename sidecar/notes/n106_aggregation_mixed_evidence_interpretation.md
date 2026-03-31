# n106 -- Aggregation Mixed-Evidence Soft-Middle Interpretation

**Type:** substudy interpretation note
**Date:** 2026-03-31
**Program:** Route2 Aggregation Mixed-Evidence Triage Perturbation
**Stage:** D
**Depends on:** n105
**Status:** complete

---

## Objective

Interpret whether the triage soft middle remains structured under mixed-evidence weighting.

---

## Claim-by-claim read

1. QA-dominant family coherence: coherent.
2. Same-family optional safe-likeness: coherent with guardrails.
3. Taxonomy usability in the soft middle: coherent with guardrails.
4. Structural nuance in mixed evidence: coherent with guardrails.
5. Review-worthy vs low-value mixed distinction: coherent with guardrails.

---

## What remained structured

- QA-dominant keeps a stable three-state partition (`qa_clear`, `qa_review`, `qa_blocked`).
- Blocked anchors remain blocked even when structural cues look positive.
- Same-family optional cases stay in clear/review lanes rather than collapsing into blocked lanes.

---

## What remains blurrier

- Review-state boundaries are not sharp numeric thresholds.
- Secondary review prioritization is visible but still lightly evidenced.
- Pattern counts remain panel-local and should not be generalized as fixed rates.

---

## Interpretation summary

The soft middle is not an artifact of clean panels. It remains interpretable, but it should be described as structured-with-guardrails rather than sharply partitioned.

---

## Outputs

- `sidecar/results/route2_stability/aggregation_mixed_evidence/soft_middle_verdicts.json`
- `sidecar/results/route2_stability/aggregation_mixed_evidence/soft_middle_verdicts.md`
- `sidecar/notes/n106_aggregation_mixed_evidence_interpretation.md` (this note)
