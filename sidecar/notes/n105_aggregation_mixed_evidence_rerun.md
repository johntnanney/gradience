# n105 -- Aggregation Mixed-Evidence Rerun

**Type:** substudy findings note
**Date:** 2026-03-31
**Program:** Route2 Aggregation Mixed-Evidence Triage Perturbation
**Stage:** C
**Depends on:** n103, n104
**Status:** complete

---

## Objective

Re-run the existing aggregation families on a soft-middle-weighted triage panel.

---

## Method

Applied existing families unchanged:

- worst-case
- distributional
- QA-dominant
- QA-gated distributional

Panel size: 8 cases (clear/mixed/blocked intentionally imbalanced toward mixed and optional).

---

## Results

Agreement distribution:

- full_agreement: 0
- partial_agreement: 6
- strong_divergence: 2

Primary observations:

1. QA-dominant remains distinct (`qa_clear`/`qa_review`/`qa_blocked` split is coherent).
2. Same-family optional cases remain in clear/review lanes, not blocked lanes.
3. Mixed-evidence review cases keep a common primary state (`qa_review`) while retaining secondary structural gradation under distributional outputs.
4. Blocked anchors still show strong divergence against structural-only families.

---

## Interpretation

The soft-middle panel does not collapse into noise. The middle is blurrier than anchor regions, but still structured enough for guarded triage interpretation.

---

## Outputs

- `sidecar/results/route2_stability/aggregation_mixed_evidence/aggregation_comparison.json`
- `sidecar/results/route2_stability/aggregation_mixed_evidence/aggregation_comparison.md`
- `sidecar/figures/aggregation_mixed_evidence_matrix.svg`
- `sidecar/notes/n105_aggregation_mixed_evidence_rerun.md` (this note)
