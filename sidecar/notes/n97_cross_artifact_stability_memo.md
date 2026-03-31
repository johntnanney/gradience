# n97 — Cross-Artifact Stability Memo (Route 2 Substudy 1)

**Type:** substudy memo  
**Date:** 2026-03-31  
**Program:** Route2 Substudy 1 — Cross-Artifact Portability Stability Check  
**Status:** Stage E complete

---

## Section 1 — What remained stable

1. **A1: QA / evidence gating** remained strong under local panel perturbation.
2. **A2: Conservative narrowing** remained strong under local panel perturbation.
3. **C1: Representation-local structural metrics** remained stable; no new cross-artifact structural invariant emerged.

These are the sturdiest Route 2 cross-artifact conclusions from this substudy.

---

## Section 2 — What remained moderate

1. **B1: Task-relation separation** remains moderately stable where testable (`same_task > cross_task` survived in LoRA and checkpoint delta slices).

Caveat: LoHa remains coverage-limited, and strict full ordering claims should stay guarded.

---

## Section 3 — What remained local

1. Factor-geometry diagnostics remain local to factorized artifacts.
2. Checkpoint summary-profile diagnostics remain local to summary-based checkpoint representation.
3. Merge strategy strings remain execution-context-local (not cross-artifact primitives).

---

## Section 4 — What remained thin

1. **D1: Near-miss / optional portability** remains inconclusive across artifact classes.

Near-miss remains clearly validated in LoRA, but this perturbation does not produce a broader non-LoRA confirmation.

---

## Section 5 — Implications for Route 2

Safe to treat as stable cross-artifact knowledge:

- evidence-first gating
- conservative narrowing behavior
- representation-locality of strongest structural metrics

Keep guarded:

- strict same-task/same-family/cross-task ordering claims
- same-family intermediate ordering claims

Keep sidecar-thin:

- cross-artifact optional/near-miss portability

Bottom line: the original portability story survives this local perturbation, but relation-ordering language should remain cautious while optionality portability remains open.
