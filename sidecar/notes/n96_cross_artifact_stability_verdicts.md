# n96 — Cross-Artifact Stability Check: Claim Verdicts

**Type:** substudy verdict note  
**Date:** 2026-03-31  
**Program:** Route2 Substudy 1 — Cross-Artifact Portability Stability Check  
**Status:** Stage D complete

---

## Objective

Assign claim-by-claim stability verdicts by comparing the frozen original claims (Stage A) to perturbed rerun outcomes (Stage C).

---

## Verdicts

- **Stable:** A1, A2, C1
- **Moderately stable:** B1
- **Panel-sensitive:** B2
- **Still inconclusive:** D1

Machine-readable outputs:

- `sidecar/results/route2_stability/cross_artifact/stability_verdicts.json`
- `sidecar/results/route2_stability/cross_artifact/stability_verdicts.md`

---

## Practical read

- Workflow-level invariants are robust.
- Relation-ordering claims need guarded language.
- Structural-locality claim remains one of the most stable outcomes.
- Optional/near-miss portability still needs broader non-LoRA coverage.
