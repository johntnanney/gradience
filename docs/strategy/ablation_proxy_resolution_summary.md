# Ablation Proxy Resolution Summary (Bounded CPU Regime)

## Status

`resolved (bounded)` for current encoder/compressible CPU line.

## Final Proxy Roles

- operational comparator: `proxy_gradient`
- explanatory companion: `proxy_ablation_attenuate`
- simple sanity probe: `hard_zero`
- paused branch: `rank_reduction`

## Why This Is Resolved Enough

1. Tie-aware reliability cleanup and low-information diagnostics were completed.
2. Rank-reduction soft-ablation reruns (`retain_ratio=0.75`, optional `0.85`) remained degenerate in this regime.
3. Gradient remained substantially more stable operationally under resampling.
4. Attenuate remained useful as explanatory structure-level companion evidence.

## Practical Rule

For bounded CPU comparisons in this line:

- use gradient for operational matched-budget comparison
- use attenuate as structural companion evidence
- avoid further rank-reduction expansion unless a narrow new question justifies it

## Boundaries

- no claim that ablation-style proxies are universally weaker
- no claim beyond small-encoder compressible bounded regime
- no product-surface escalation from this line

## Canonical Evidence

- [`docs/00_start_here/bounded-validation-summary.md`](../00_start_here/bounded-validation-summary.md)
- [`docs/strategy/rank_proxy_bounded_validation_summary.md`](rank_proxy_bounded_validation_summary.md)
- `field_trials/rank_proxy_validation/rank_reduction_soft_ablation_retain_ratio_comparison.md`
