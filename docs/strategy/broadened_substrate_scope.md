# Broadened Substrate Scope (Route 2)

Date: 2026-03-31  
Status: Route 2 consolidation snapshot (bounded)

---

## What Gradience is today

Gradience is an evidence-aware triage system for learned variants from a shared base model.

The anchor workflow remains merge preflight for adapter inventories.  
Broadened workflows are included only where representation, evidence, aggregation, and practical confirmation align.

---

## What is already generalized

1. Scenario generality: merge vs routing/confusability (shared spectral substrate, different aggregation/policy).
2. Artifact-class generality within low-rank PEFT: LoRA vs LoHa (thin extraction shim, no core refactor).
3. Representation-path generality to full checkpoints: summary-based checkpoint-delta triage (Ring 2).
4. Decision-dependent compatibility framing: measurement reuse with aggregation and policy as scenario-specific seams.

---

## Validated broadened workflow classes (bounded)

1. Adapter inventory preflight (stable).
2. Checkpoint inventory triage (bounded-supported in tested settings; experimental outside those bounds).
3. Decision-relative compatibility analysis for merge/routing/triage (conceptually stabilized, bounded).

---

## Hard boundaries still in effect

- CPU-only validation.
- Small encoder classification settings only.
- Shared-base inventories only for checkpoint workflows.
- Merge execution remains out of scope for checkpoint-delta Route 2 work.
- No claim of universal artifact/scenario coverage.

---

## Route 2 working rule

Broaden only where all four gates pass:

1. representation clarity
2. workflow usefulness
3. decision-relative interpretability
4. practical confirmation

Anything else remains experimental, sidecar-only, or deferred.

---

## Companion Route 2 docs

- `docs/strategy/artifact_class_matrix.md`
- `docs/strategy/scenario_aggregation_matrix.md`
- `docs/strategy/checkpoint_triage_summary.md`
- `docs/strategy/checkpoint_triage_alpha_scope.md`
- `docs/strategy/decision_dependent_compatibility_implications.md`
- `docs/strategy/external_use_case_scan.md`
- `docs/examples/checkpoint-triage-alpha-workflow.md`

---

## Milestone Checkpoint

Route 2 initial implementation checkpoint is complete in bounded form:

1. broadened substrate documented
2. checkpoint triage stabilized in tested settings
3. decision-dependent compatibility consolidated
4. cross-artifact compatibility portability pass completed in bounded scope
5. external use-case scan framed
6. expansion remains gated by representation clarity, workflow usefulness, interpretability, and practical confirmation
