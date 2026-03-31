# Cross-Artifact Compatibility Framework

Generated: 2026-03-31

## Three-Layer Structure

### Layer 1 -- Artifact-Invariant Signals

| Signal | Strength | Product Status |
|--------|----------|---------------|
| Evidence regime gating | Strong | Safe to expose |
| Conservative candidate narrowing | Strong | Safe to expose |
| Task-relation ordering (same > family > cross) | Moderate | Guarded |
| Same-family intermediate status | Moderate | Guarded |

### Layer 2 -- Representation-Family Features

**Factor-based family** (LoRA, LoHa shimmed):
- Subspace overlap / principal angles / V-module ratio
- Merge strategy recommendation
- Spectral audit metrics (stable rank, utilization, energy rank)

**Summary-based family** (checkpoint delta):
- Summary profile cosine similarity
- Delta spectral statistics (effective rank, stable rank, energy concentrations)

**Shimmed family** (LoHa only):
- Shim approximation error confound

### Layer 3 -- Decision-Dependent Interpretation

| Scenario | Aggregation | Artifact Coverage |
|----------|-------------|-------------------|
| Merge | Worst-case | LoRA, LoHa (unvalidated) |
| Routing | Distributional | LoRA only |
| Triage | QA-gate-first | **All three classes** |

## Interaction Rules

1. Layer 1 signals can be referenced across artifact classes without qualification.
2. Layer 2 signals must be qualified by artifact class.
3. Layer 3 determines which Layer 2 signals apply.
4. Only triage has full cross-artifact coverage.
5. Numeric scores cannot be compared across representation families.
