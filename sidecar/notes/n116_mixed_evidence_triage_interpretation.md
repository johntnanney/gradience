# n116 -- Mixed-Evidence Triage Soft-Middle Interpretation

**Type:** stress-test interpretation note  
**Date:** 2026-03-31  
**Program:** Route2 Mixed-Evidence Triage Stress Test  
**Stage:** D  
**Depends on:** n115  
**Status:** complete

---

## Objective

Interpret whether the triage middle remains coherent, guarded, or ambiguous under soft-middle-heavy panel stress.

---

## Verdict summary

- `coherent`: 2
- `coherent_with_guardrails`: 3
- `ambiguous`: 0
- `weakened`: 0

---

## Claim-target interpretation

1. QA-dominant family coherence: coherent.
2. Same-family optional safe-likeness: coherent_with_guardrails.
3. Review-like vs collapse-like distinction: coherent.
4. Triage-middle structure: coherent_with_guardrails.
5. Structural nuance inside mixed evidence: coherent_with_guardrails.

---

## What remained coherent

- QA-dominant keeps clear primary partitioning (`qa_clear`, `qa_review`, `qa_blocked`).
- Review-like and optional cases do not collapse into blocked/collapse-like states.

---

## What stayed guarded

- Exact internal boundaries inside review remain soft.
- Secondary review ordering is useful but lightly evidenced.
- Hard threshold statements remain non-canonical.

---

## Interpretation

The soft middle remains structured enough for review-first triage, but should still be described as structured-with-guardrails rather than sharply thresholded.

---

## Outputs

- `sidecar/results/route2_stress_tests/mixed_evidence_triage/soft_middle_verdicts.json`
- `sidecar/results/route2_stress_tests/mixed_evidence_triage/soft_middle_verdicts.md`
- `sidecar/notes/n116_mixed_evidence_triage_interpretation.md`
