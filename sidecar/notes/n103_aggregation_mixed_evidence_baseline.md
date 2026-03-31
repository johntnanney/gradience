# n103 -- Aggregation Mixed-Evidence Baseline Freeze

**Type:** substudy setup note
**Date:** 2026-03-31
**Program:** Route2 Aggregation Mixed-Evidence Triage Perturbation
**Stage:** A
**Depends on:** n81-n85, n98-n102
**Status:** complete

---

## Objective

Freeze the current aggregation-sensitive conclusions before running a triage-middle stress test weighted toward mixed-evidence and same-family optional cases.

---

## Baseline claims under stress

- Claim A: QA-dominant aggregation is a distinct operational family.
- Claim B: Same-family optional cases are closer to safe/review states than collapse states.
- Claim C: Taxonomy remains usable with guardrails.
- Claim D: Structural nuance may re-enter inside mixed-evidence review.

Baseline status summary:

- Stable: aggregation-as-seam, QA-dominant distinctness, worst-case collapse.
- Moderately stable: taxonomy usability.
- Guarded/open: fine-grained review-state boundaries and mixed-evidence internal gradation.

---

## Scope isolation

This pass does not retest the whole aggregation line. It probes one narrow question:

- whether the triage soft middle remains coherent when the panel is intentionally weighted toward review/optional cases.

---

## Outputs

- `sidecar/results/route2_stability/aggregation_mixed_evidence/baseline_claims_snapshot.json`
- `sidecar/notes/n103_aggregation_mixed_evidence_baseline.md` (this note)
