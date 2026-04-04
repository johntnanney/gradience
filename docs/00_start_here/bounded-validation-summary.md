# Bounded Validation Summary

**Audience:** practitioner, collaborator, maintainer  
**Status:** stable (bounded-policy freeze)  
**Purpose:** one-page bounded validation truth for current comparator policy  
**Canonical for:** what rank-proxy validation supports today (and what it does not)  
**Supersedes:** scattered bounded-claim wording across strategy and field-trial notes  
**See also:** [`project-map.md`](project-map.md), [`stable-vs-experimental.md`](stable-vs-experimental.md), [`../strategy/rank_proxy_bounded_validation_summary.md`](../strategy/rank_proxy_bounded_validation_summary.md)

## Frozen Proxy Policy (Current Regime)

- `proxy_gradient` is the **primary operational comparator**.
- `attenuate` is the **companion ablation proxy** for structural/explanatory evidence.
- `rank_reduction` is **paused for expansion** in the current encoder/compressible CPU regime.

## Regime Where This Applies

- shared-base small encoders
- classification tasks
- compressible-family interpretation centered on `sst2` and `imdb`
- CPU-only bounded validation protocol

## What Is Supported

1. Spectral policies (especially `oht`) are competitive fixed-budget guides in the bounded compressible subset.
2. Spectral allocations capture a structurally meaningful importance notion.
3. Gradient remains the stronger operational target under the current protocol due higher stability and outcome behavior.
4. Attenuated ablation is useful as companion evidence when interpreted with tie-aware reliability diagnostics.

## What Is Not Supported

1. Universal policy dominance claims outside this bounded regime.
2. Replacing gradient as operational default comparator.
3. Treating rank-reduction ablation as a reliable operational comparator in this regime.
4. Escalating to broader claims from saturated/non-informative family slices.

## Why Rank-Reduction Is Paused

- `retain_ratio=0.5` showed heavy low-information/flat-vector behavior.
- follow-up reruns at `0.75` and `0.85` did not recover interpretability:
  - low-information remained maximal,
  - valid pair coverage collapsed,
  - tie-aware stability metrics became non-evaluable.

## Canonical Evidence Sources

- bounded strategy summary: [`../strategy/rank_proxy_bounded_validation_summary.md`](../strategy/rank_proxy_bounded_validation_summary.md)
- bounded validation memo: [`../../field_trials/rank_proxy_validation/bounded_validation_memo.md`](../../field_trials/rank_proxy_validation/bounded_validation_memo.md)
- retain-ratio comparison: [`../../field_trials/rank_proxy_validation/rank_reduction_soft_ablation_retain_ratio_comparison.md`](../../field_trials/rank_proxy_validation/rank_reduction_soft_ablation_retain_ratio_comparison.md)

## Practical Rule

For new bounded CPU comparisons in this line:

- operational comparison target: `proxy_gradient`
- explanatory ablation companion: `attenuate`
- sanity reference: `hard_zero`
- no additional rank-reduction expansion unless a tightly scoped new question is defined first
