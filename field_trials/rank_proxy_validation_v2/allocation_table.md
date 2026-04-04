# Rank Proxy Validation v2 Allocation Table

## Notes
- This v2 bundle canonicalizes existing v1 outputs without a heavy re-run.
- Full layerwise vectors/top-k layer identities were not persisted in v1 and are left null here.
- This is not a full layer-vector comparison archive.
- Structure-level interpretations in this pass are based on persisted comparison artifacts rather than a complete vector-preserving bundle.

## Primary Informative Summary by Method
| method | n | mean_realized_budget | mean_attn_share | mean_mlp_share |
| --- | --- | --- | --- | --- |
| energy_90 | 18 | 0.494 | 1.000 | 0.000 |
| erank | 18 | 0.494 | 1.000 | 0.000 |
| knee | 18 | 0.494 | 1.000 | 0.000 |
| oht | 18 | 0.494 | 1.000 | 0.000 |
| proxy_ablation_attenuate | 18 | 0.494 | 1.000 | 0.000 |
| proxy_gradient | 18 | 0.494 | 1.000 | 0.000 |
| random_matched_budget | 18 | 0.494 | n/a | n/a |
| stable_rank_ceil | 18 | 0.494 | 1.000 | 0.000 |
| uniform | 18 | 0.494 | n/a | n/a |
