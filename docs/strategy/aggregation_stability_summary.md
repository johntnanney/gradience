# Aggregation Stability Summary (Route2 Substudy 2)

**Date:** 2026-03-31
**Scope:** CPU-only, local panel perturbation of aggregation-sensitive compatibility panel

---

## What this check did

This substudy reran the completed aggregation-sensitive conclusions under a small, disciplined perturbation:

- one merge-facing substitution
- one routing-facing substitution
- one triage-facing substitution
- no aggregation-family definition changes

The goal was claim stability testing, not numeric replication.

---

## Stability outcomes

### Stable

- A1: Aggregation is the first major decision seam.
- B1: Worst-case collapse behavior remains visible.
- B3: QA-dominant remains a distinct operational family.
- C2: Both aggregation-invariant and aggregation-sensitive case types remain present.
- D1: Aggregation should be treated as a design variable in Route 2 workflows.

### Moderately stable

- B2: Distributional gradation remains visible, but exact tier boundaries are still panel-dependent.
- C1: Compact taxonomy viability remains interpretable, with scope guardrails.

### Panel-sensitive / still-thin in this pass

- None of the previously strong claims degraded to panel-sensitive in this local perturbation.

---

## Route 2 language guidance

Safe to state more confidently:

- Aggregation is a real operational seam, not presentation.
- Scenario-appropriate aggregation remains necessary (worst-case, distributional, QA-dominant).
- QA-dominant logic remains operationally distinct from structural-only analysis.

Keep guarded:

- exact distributional cut points
- exact pattern frequencies as if they were universal

---

## Bottom line

The aggregation-sensitive Route 2 story remains sturdy under local perturbation. Treat aggregation as a first-class workflow design choice, while keeping fine-grained taxonomy thresholds explicitly bounded.

Follow-on clarification pass: `docs/strategy/aggregation_mixed_evidence_summary.md` stress-tests the triage soft middle and finds the same pattern holds with guardrails (QA-dominant coherence remains, same-family optional stays review-like).
