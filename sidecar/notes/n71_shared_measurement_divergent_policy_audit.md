# n71 — Shared-Measurement / Divergent-Policy Audit

**Type:** audit note  
**Date:** 2026-03-31  
**Depends on:** n70 panel, routing pilot, checkpoint triage T02  
**Status:** Stage B complete

---

## Question

At what layer does scenario dependence first appear across:

- merge
- routing
- triage

---

## Outputs

- `sidecar/results/decision_dependent_compatibility/scenario_stack_matrix.json`
- `sidecar/results/decision_dependent_compatibility/shared_vs_specific_table.md`

---

## Decomposition used

Each scenario was decomposed into:

1. measurement
2. diagnosis
3. aggregation
4. policy

For each layer we marked:

- shared unchanged
- shared with translation
- scenario-specific

---

## Main result

**First divergence appears at aggregation in all three scenario-pair audits**:

- merge vs routing
- merge vs triage
- routing vs triage

Measurement and diagnosis are largely shared or translation-level variants. Policy obviously diverges, but the decisive seam appears one layer earlier: *aggregation*.

---

## Pairwise summary

### Merge vs Routing

- Shared: extraction + per-layer geometry (`SubspaceMetrics`)
- Divergence: worst-case aggregation (merge) vs distributional aggregation (routing)
- Policy split: merge candidacy vs dedup/disambiguation/easy-route decisions

### Merge vs Triage

- Shared: pairwise structural compatibility descriptors
- Divergence: merge worst-case risk vs QA-gate-first narrowing
- Policy split: merge strategy vs evaluation-budget prioritization

### Routing vs Triage

- Shared: relation-aware structural interpretation
- Divergence: distributional separability vs source-quality gating
- Policy split: route disambiguation vs review/exclusion stance

---

## Interpretation

This audit supports H1 and H2 from the program spec:

- **H1 (measurement more general than policy):** supported in bounded scope.
- **H2 (aggregation as first decision seam):** supported directly in the scenario-pair matrix.

The practical consequence is that scenario expansion pressure should target aggregation/policy parameterization first, not measurement refactor.

---

## Caveat

The audit is cross-artifact but not universal:

- routing evidence currently comes from LoRA pilot cases
- triage evidence includes checkpoint summary-based path
- decoder/generative settings are out of scope

So this is a structured bounded claim, not a universal compatibility law.
