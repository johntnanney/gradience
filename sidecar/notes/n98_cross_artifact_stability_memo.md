# n98 — Cross-Artifact Stability Memo

**Type:** stability memo (Stage E synthesis)
**Date:** 2026-03-31
**Program:** Route 2 Stability and Replication Check, Substudy 1
**Depends on:** n94–n97 (full stability check chain)
**Status:** Final.

---

## One-paragraph summary

The cross-artifact compatibility conclusions from the Route 2 research program (n76–n80) survive a disciplined 4-of-9 panel perturbation. The strongest claims — QA gating, conservative narrowing, and structural locality — are fully stable. The moderate claims — task-relation ordering and same-family intermediate status — survive in LoRA (and generalize to a second task family) but weaken in checkpoint delta when the cross-task pair is structurally close. This is a genuine scope narrowing, not a falsification. Near-miss portability remains inconclusive. No claim was panel-sensitive. The overall program structure (what is safe to expose, what needs guardrails, what is research-only) is unchanged.

---

## What remained stable

**QA gating (A1) and conservative narrowing (A2)** are the backbone of the cross-artifact program. They operate on evidence metadata and workflow structure, not on structural measurements. This makes them representation-agnostic by design. The perturbation confirmed this: changing adapters, task families, seeds, and cross-task partners does not affect the gating logic or the narrowing pathway.

**Structural locality (C1)** is also stable, and the perturbation strengthened the argument. The MRPC substitution produced a checkpoint cross-task pair (0.798) scoring higher than a LoRA same-task pair (0.475), providing an even more dramatic illustration that cross-class score comparison is meaningless. All 5 local signals retained their original portability level.

**Product implication:** A1, A2, and C1 can be exposed in product with their current scope and confidence. No guardrail changes needed.

## What is moderately stable

**Task-relation ordering (B1) and same-family intermediate status (B2)** have two regimes:

1. **Robust in LoRA:** The three-way ordering holds for a second task family (sentiment_binary, after NLI). Same-family intermediate status is task-family-general, not NLI-specific. This is the positive result from the perturbation.

2. **Fragile in checkpoint delta for same_family vs cross_task:** MRPC (compat=0.798, cross-task) scores higher than Yelp (compat=0.641, same-family). Task-family membership does not guarantee structural proximity in checkpoint delta representation. The same_task > cross_task ordering still holds (0.892 > 0.798), but the same_family > cross_task ordering breaks.

**Interpretation:** The ordering is a good heuristic for triage (where task labels determine priority buckets) but not a structural invariant (where compatibility scores determine ranking). This distinction matters for product: same-family routing to the same-task priority bucket in inventory action plans remains justified, but documentation should note that checkpoint delta compatibility scores may not follow the task-relation ordering.

**Product implication:** B1 and B2 need scope guardrails specifying where the ordering holds (LoRA, categorical triage) and where it may not (checkpoint delta score ordering when cross-task is structurally close).

## What is local (not portable)

The locality claim was always part of C1, not a weakness. Five structural signals remain representation-local:

- Factor subspace overlap: LoRA + LoHa only (requires pre-factored geometry)
- V-module dim ratio: LoRA only (requires module-level factor isolation)
- Compatibility score: exists in all classes but different scales
- Pair risk categorical: exists in all classes but different calibrations
- Routing confusability: LoRA only (routing scenario)

This is architectural, not panel-dependent. The perturbation confirmed that the locality boundary does not shift with different cases.

## What is still thin

**Near-miss portability (D1)** remains inconclusive. Near-miss is well-validated in LoRA (7 pairs, avg delta -0.006, behaviorally safe) but has zero cross-artifact evidence. The perturbation was not designed to test this — it would require evidence-borderline cases that did not arise in the perturbed panel. Resolution requires either:

- A dedicated perturbation designed to create near-miss conditions in LoHa or checkpoint delta, or
- GPU-based behavioral evaluation of borderline checkpoint pairs

Neither is feasible without new experimental work.

## Implications for Route 2

### No changes to product recommendations

The stability check did not change any product-level recommendation. What was safe to expose is still safe. What needed guardrails still needs guardrails (with slightly more specific guardrails for B1/B2). What was research-only is still research-only.

### Scope guardrail update for B1/B2

The existing scope guardrail for task-relation ordering should be updated to note:

> Task-relation ordering (same_task > same_family > cross_task) is a reliable heuristic for triage priority but should not be assumed to hold for compatibility score ranking in checkpoint delta representation. Cross-task pairs that are structurally close to same-task (e.g., MRPC relative to SST-2) may score higher than same-family pairs.

### Confidence calibration

The stability check provides a calibration signal:
- **High confidence:** A1, A2, C1 — survived perturbation cleanly
- **Moderate confidence with specified limits:** B1, B2 — survived in LoRA, fragile in one checkpoint delta comparison
- **Low confidence / no evidence:** D1 — unchanged from baseline

This maps well to the three-tier product status: safe_to_expose, safe_with_scope_guardrail, not_stable_enough.

### What would improve confidence further

1. **For B1/B2:** Additional checkpoint cross-task pairs of varying structural distance. If the ordering breaks consistently when structural distance is small, the scope guardrail is confirmed. If it only breaks for MRPC, the MRPC case may be an outlier.
2. **For D1:** Behavioral evaluation of borderline checkpoint pairs. This requires GPU.
3. **For all claims:** A second backbone (not distilbert-base-uncased) would test backbone generality. This is out of scope for this substudy.

---

## Program completion

This note completes the Cross-Artifact Portability Stability Check (Substudy 1). All five stages are done:

| Stage | Deliverable | Status |
|-------|-----------|--------|
| A — Panel freeze | n94, original_panel_snapshot.json, original_claims_snapshot.json | Complete |
| B — Perturbed panel | n95, perturbed_panel_table.json, panel_diff_table.md | Complete |
| C — Rerun audit | n96, perturbed_invariant_signal_matrix.json, perturbed_local_signal_table.json, perturbed_signal_summary.md | Complete |
| D — Verdicts | n97, stability_verdicts.json, stability_verdicts.md | Complete |
| E — Memo | n98 (this note), cross_artifact_stability_summary.md | Complete |
