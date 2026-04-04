# n121 -- Collapse vs Contamination Replication Verdict

**Type:** replication verdict note  
**Date:** 2026-03-31  
**Program:** Route2 Collapse vs Contamination Replication  
**Stage:** D  
**Depends on:** n120  
**Status:** complete

---

## Verdict

**Overall verdict:** `replicated_with_guardrails`

The distinction survives this local replication pass with clear channel separation and explicit scope bounds.

---

## Why this verdict

1. **Channel separation clarity:** strong.
   - collapse-like targets retain high confidence-collapse and near-zero high-confidence wrong.
   - contamination-like targets retain low confidence-collapse and elevated high-confidence wrong.
2. **Slice stability:** strong for contamination (even/odd both preserve signature), adequate for collapse (slice + nearby case).
3. **Case stability:** moderate.
   - collapse replicated on nearby case (`FR-02`).
   - contamination still anchored primarily to one underlying case family (`CT-01`) via slices.
4. **Scope breadth:** narrow by design (same backbone, merge-facing context, existing case pool).
5. **Communication relevance:** high for bounded Route 2 explanation.

---

## Interpretation

This pass materially strengthens confidence that collapse-vs-contamination is not just a one-table artifact.  
It remains a **stable bounded behavioral distinction** rather than a universal failure-channel law.

---

## Outputs

- `sidecar/results/route2_stress_tests/collapse_vs_contamination/verdicts.json`
- `sidecar/results/route2_stress_tests/collapse_vs_contamination/verdicts.md`
