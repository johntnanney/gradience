# n97 — Cross-Artifact Stability Check: Stage D Verdicts

**Type:** verdict note
**Date:** 2026-03-31
**Program:** Route 2 Stability and Replication Check, Substudy 1
**Depends on:** n94 (original panel), n95 (perturbed panel), n96 (rerun findings)
**Status:** Complete. Ready for Stage E (stability memo).

---

## Verdicts

Six claims assessed against four-level scale: stable, moderately_stable, panel_sensitive, still_inconclusive.

### Stable (3 claims)

**A1 — QA / evidence gating.** Strong → stable. QA gating operates on evidence metadata (behavioral_reported, structural_only, unknown_no_behavioral_eval), which is assigned before structural analysis. Changing which adapters fill the slots changes evidence status in some cases (LoRA same-family gained behavioral evidence) but the gating logic responds correctly. This is the most robust cross-artifact invariant.

**A2 — Conservative narrowing.** Strong → stable. The workflow shape (broad candidate set → narrow output via QA → task boundaries → structural risk) is identical on the perturbed panel. Narrowing ratios unchanged. This is the second most robust invariant.

**C1 — Structural locality.** Strong_local → stable. All 5 local signals retain their original portability. The MRPC substitution strengthened the locality argument: checkpoint cross-task (0.798) > LoRA same-task (0.475) is an even more dramatic illustration that cross-class score comparison is meaningless.

### Moderately stable (2 claims)

**B1 — Task-relation ordering.** Moderate → moderately_stable. The three-way ordering (same_task > same_family > cross_task) survives in LoRA for a different task family (sentiment_binary, not just NLI). In checkpoint delta, the same_task > cross_task ordering holds (gap: 0.094) but same_family > cross_task breaks when the cross-task pair is structurally close. Scope narrowing: the ordering is robust for categorical triage but fragile for score-based ranking in checkpoint delta.

**B2 — Same-family intermediate.** Moderate → moderately_stable. Same pattern as B1. In LoRA, same-family intermediate status is task-family-general. In checkpoint delta, same-family is not necessarily intermediate by compatibility score when the cross-task comparator is structurally close. Scope narrowing: same-family routing to same-task priority bucket remains justified (label-based); score-based intermediate position is representation-dependent.

### Still inconclusive (1 claim)

**D1 — Near-miss portability.** Inconclusive → still_inconclusive. The perturbation did not create evidence-borderline cases needed to test near-miss emergence. This is expected — the perturbation was designed to test B1/B2 robustness, not D1. Near-miss remains a LoRA-validated category awaiting cross-artifact evidence.

## No claims are panel-sensitive

The perturbation found a genuine fragility (B1/B2 in checkpoint delta) but this narrows the scope rather than changing the verdict. The original claims were "moderate" and remain moderate — the perturbation specifies where the moderation applies.

## Data

| File | Content |
|------|---------|
| `results/route2_stability/cross_artifact/stability_verdicts.json` | Machine-readable verdicts with evidence and product implications |
| `results/route2_stability/cross_artifact/stability_verdicts.md` | Human-readable verdict summary |
