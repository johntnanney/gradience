# n113 -- Mixed-Evidence Triage Baseline Freeze

**Type:** stress-test setup note  
**Date:** 2026-03-31  
**Program:** Route2 Mixed-Evidence Triage Stress Test  
**Stage:** A  
**Depends on:** n81-n85, n98-n102, n103-n107, n92  
**Status:** complete

---

## Objective

Freeze the current Route 2 triage-middle claim state before running a new mixed-evidence stress pass.

---

## Baseline claim state

1. QA-dominant triage remains a distinct aggregation family (`stable`).
2. Same-family optional cases usually behave review-like/safe-like, not collapse-like (`moderately_stable`, guarded).
3. The triage middle is structured enough for review-first narrowing (`moderately_stable`, guarded).
4. Fine-grained internal review thresholds are still non-canonical (`guarded/open`).

---

## Targeted open question

Does this soft-middle structure remain coherent when the panel is intentionally skewed toward mixed-evidence review and same-family optional cases?

---

## Outputs

- `sidecar/results/route2_stress_tests/mixed_evidence_triage/baseline_claims_snapshot.json`
- `sidecar/notes/n113_mixed_evidence_triage_baseline.md`
