# Cross-Artifact Portability Stability Summary (Route 2 Substudy 1)

**Date:** 2026-03-31  
**Scope:** CPU-only, local panel perturbation (no artifact expansion)

---

## What this check did

This substudy reran the completed cross-artifact portability conclusions under a small, disciplined panel perturbation (4 substitutions across LoRA and checkpoint delta; LoHa unchanged with documented fallback).

The goal was not numeric replication; it was claim stability testing.

---

## Stability outcomes

### Stable

- QA / evidence gating as cross-artifact invariant.
- Conservative narrowing as cross-artifact invariant.
- Structural-locality claim: strongest structural metrics remain representation-local.

### Moderately stable

- Same-task vs cross-task separation where testable.

### Panel-sensitive

- Same-family strict intermediate ordering.

### Still inconclusive

- Near-miss / optional middle-state portability across artifact classes.

---

## Route 2 language guidance

Safe to state more confidently:

- Gradience’s broadened substrate supports evidence-aware triage across tested artifact classes.
- Cross-artifact stability is strongest at workflow level (evidence gating, conservative narrowing).
- Structural metrics remain representation-family-specific.

Keep guarded:

- strict three-way ordering claims (`same_task > same_family > cross_task`) as universal cross-artifact behavior
- same-family intermediate claims without scenario and evidence caveats

Keep research-only / thin:

- cross-artifact optional/near-miss portability claims outside LoRA

---

## Bottom line

The cross-artifact portability story remains valid but narrow after perturbation: workflow invariants are sturdy, structural metric parity is not, and same-family/optionality claims still require caution.
