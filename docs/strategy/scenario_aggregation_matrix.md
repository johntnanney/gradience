# Scenario Aggregation Matrix (Route 2)

Date: 2026-03-31  
Status: bounded consolidation snapshot

## Matrix

| Scenario | Measurement layer | Diagnosis layer | Aggregation rule | Policy layer | Validation status |
|---|---|---|---|---|---|
| Merge compatibility | Shared structural geometry (`SubspaceMetrics` for adapters; summary-compat primitives for checkpoints) | Shared with translation (risk/redundancy descriptors + relation tags) | Worst-case sensitivity (local catastrophic layers can dominate) | Retain / caution / exclude + merge strategy for retained candidates | stable for adapter workflows; checkpoint merge execution out of scope |
| Routing/confusability | Same structural measurement substrate as merge (routing pilot reuses existing API) | Shared with translation (confusability interpretations) | Distributional spread sensitivity (mean/fraction style summaries) | Dedup / disambiguate / separable routing guidance | pilot-validated in bounded adapter setting |
| Inventory triage | Pairwise structure + source QA/evidence artifacts | Shared with translation (compatibility + source quality states) | QA-gate-first (source quality can override pair structure) | Evaluate-first / near-miss review / optional / exclusion planning | validated for adapter inventories; bounded-supported for checkpoint inventories in tested settings |

## Practical interpretation

- First divergence across scenarios appears at aggregation, not at measurement.
- Policy differences are downstream consequences of aggregation and decision objective.
- A shared measurement substrate is real, but scenario-specific aggregation and policy remain required.

Primary source: `sidecar/results/decision_dependent_compatibility/scenario_stack_matrix.json`.
