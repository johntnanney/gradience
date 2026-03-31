# n96 — Cross-Artifact Stability Check: Stage C Findings

**Type:** findings note
**Date:** 2026-03-31
**Program:** Route 2 Stability and Replication Check, Substudy 1
**Depends on:** n94 (original panel), n95 (perturbed panel)
**Status:** Complete. Ready for Stage D (verdicts).

---

## What was done

Re-ran the invariant signal audit and local signal audit on the perturbed panel (4 of 9 cases substituted). Compared signal verdicts to original.

## Invariant signal results

**Unchanged (3 of 5):**
- QA / evidence gating: strong → strong. Operates on metadata, not structural measurements. Representation-agnostic by design.
- Conservative narrowing: strong → strong. Workflow shape (broad → narrow) is invariant to which cases fill the slots.
- Near-miss: inconclusive → still_inconclusive. Perturbation did not create the structural conditions for near-miss emergence.

**Weakened (2 of 5):**
- Task-relation ordering: moderate → weakened. The checkpoint delta MRPC substitution (compat=0.798) scores higher than same-family Yelp (0.641), violating the expected same_family > cross_task ordering.
- Same-family intermediate: moderate → weakened. Same cause — same-family is no longer intermediate by compatibility score when cross-task is structurally close.

## The B1/B2 finding in detail

This is the most important result from the perturbation.

**Original panel (checkpoint delta):**
```
same_task (0.892) > same_family (0.652) > cross_task (0.626)   ✓ ordering holds
                     gap: 0.240        gap: 0.026
```

**Perturbed panel (checkpoint delta):**
```
same_task (0.892) > cross_task (0.798) > same_family (0.641)   ✗ ordering violated
                     gap: 0.094        gap: 0.157
```

The violation occurs because MRPC (paraphrase detection) is structurally closer to SST-2 than Yelp (sentiment, same family) is. Task-family membership does not guarantee structural proximity in checkpoint delta representation.

**LoRA is unaffected.** SST-2 x IMDB (same-family, sentiment_binary) remains intermediate between same-task (0.475) and cross-task (0.111). The ordering survives for a different task family than the original (NLI), providing task-family generality.

**Interpretation:** The task-relation ordering has two regimes:
1. **Robust regime** (LoRA, where structural similarity tracks task similarity well): ordering holds across task families.
2. **Fragile regime** (checkpoint delta, where structural similarity may not track task similarity): ordering holds for same_task > cross_task but breaks for same_family > cross_task when the cross-task pair is structurally close.

This is an important narrowing, not a falsification. The ordering is still useful for triage (where task labels drive priority) but cannot be treated as a structural invariant (where scores drive decisions).

## Local signal results

All 5 local signals stable. Two strengthened:
- Compatibility score locality: MRPC ckpt cross-task (0.798) > LoRA same-task (0.475) makes cross-class comparison obviously meaningless.
- Pair risk locality: same checkpoint slot shifts high → medium with MRPC swap, confirming representation-specific calibration.

## Data produced

| File | Content |
|------|---------|
| `results/route2_stability/cross_artifact/perturbed_invariant_signal_matrix.json` | 5 invariant signals, 3 local signal audits |
| `results/route2_stability/cross_artifact/perturbed_local_signal_table.json` | 5 local signals with portability assessment |
| `results/route2_stability/cross_artifact/perturbed_signal_summary.md` | Human-readable summary with key finding |

## Implications for Stage D

Expected verdicts:
- A1 (QA gating): **stable**
- A2 (narrowing): **stable**
- B1 (ordering): **moderately_stable** — survives in LoRA, weakens in checkpoint delta
- B2 (intermediate): **moderately_stable** — same pattern
- C1 (locality): **stable** — strengthened by MRPC illustration
- D1 (near-miss): **still_inconclusive** — no new evidence
