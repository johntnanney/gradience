# Cross-Artifact Representation-Local Signal Table

Generated: 2026-03-31

## Local Signals

| Signal | Type | Present In | Absent From | Portability | Product |
|--------|------|-----------|-------------|-------------|---------|
| Factor subspace overlap | Factor-geometry | LoRA, LoHa | Ckpt delta | Local only | Research only |
| Merge strategy recommendation | Execution-context | LoRA, LoHa | Ckpt delta | Local only | Research only |
| Compatibility score (numeric) | Extraction-artifact | All three | - | Partial | Guarded |
| Stable rank / energy metrics | Factor-geometry | LoRA, LoHa | Ckpt delta (analogous) | Partial | Research only |
| Shim extraction artifacts | Extraction-artifact | LoHa only | LoRA, Ckpt delta | Local only | Research only |
| Summary profile shape | Summary-specific | Ckpt delta only | LoRA, LoHa | Local only | Research only |
| Pair risk categorical | Extraction-artifact | All three | - | Partial | Guarded |

## Key Findings

1. **The strongest sidecar signal is representation-locked.** V-module dimensionality ratio (d=3.36, zero overlap) requires factor-level subspace geometry. It cannot be computed from checkpoint summary representations.

2. **Numeric scores are not comparable across classes.** LoRA same-task compatibility (0.475) and checkpoint delta same-task compatibility (0.892) use different scales and semantics. Ordinal ranking may transfer; absolute values do not.

3. **Merge execution is artifact-local.** Strategy recommendations (linear, ties, norm_equalized) are meaningful only for native LoRA. LoHa merge execution is unvalidated. Checkpoint delta merge is out of scope.

4. **Analogous metrics have different semantic content.** "Stable rank" means factor utilization in LoRA but delta spectrum shape in checkpoint deltas. Same name, different populations.

5. **Shim artifacts are a confound for LoHa.** The materialized SVD shim introduces approximation error. Unusually low LoHa compatibility scores may reflect shim loss, not intrinsic incompatibility.
