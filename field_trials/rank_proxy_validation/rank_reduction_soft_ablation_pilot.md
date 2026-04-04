# Rank-Reduction Soft-Ablation Pilot (CPU, Informative Subset)

## Scope
- Subset: informative/compressible families only (`sst2`, `imdb`).
- Adapters: 4.
- Modes compared: `hard_zero`, `attenuate`, `rank_reduction` (`rank_retain_ratio=0.5`).
- Panel protocol: `3` fixed + `3` random panels.
- Ablation sample grid: `24, 48, 72` (non-saturated).
- Budgets: `0.35, 0.50, 0.65`.
- Metrics: tie-aware stability (`spearman`, `kendall_tau_b`, `gamma`), top-k overlap (`q25`, `q50`), low-information flags, and policy agreement vs OHT.

## Why This Pilot
- Directly test whether softer, compression-like perturbation (`rank_reduction`) reduces tie/flat-vector pathology relative to `hard_zero`/`attenuate`.
- Keep protocol matched to the prior reliability pass.

## Results By Your Four Questions

1. Which mode produced the fewest low-information vectors?
- Best: `hard_zero`.
- Aggregate low-information burden:
  - `hard_zero`: `0.708`
  - `attenuate`: `0.764`
  - `rank_reduction`: `1.000`
- `rank_reduction` was worst on low-information burden in this pilot.

2. Which mode had the best resampling stability?
- Nominally highest correlations: `rank_reduction` (`1.0`), but not meaningful here due severe degeneracy.
- Validity context:
  - mean valid-pair fraction: `rank_reduction=0.125` vs `attenuate=0.403` vs `hard_zero=0.542`.
- Among non-degenerate modes, `attenuate` had higher tie-aware stability at high sample budget (`72`) than `hard_zero`.

3. Which mode agreed most cleanly with spectral policies?
- Raw policy agreement vs OHT is highest for `rank_reduction`.
- But this appears confounded by degenerate ablation profiles:
  - `rank_reduction` has extremely high tie burden (`mean_tie_pair_fraction ~0.940`) and low nonzero signal (`~0.030`).
- Operationally, treat `rank_reduction` agreement here as low-confidence until degeneracy is reduced.

4. Which mode is most behaviorally interpretable for compression relevance?
- Most interpretable in this run: `attenuate` (with `hard_zero` as secondary anchor).
- `rank_reduction` at `retain_ratio=0.5` is not yet interpretable as a reliability target because it collapses too often into low-information vectors.

## Key Diagnostics (Aggregate)
- `hard_zero`:
  - mean low-info fraction: `0.708`
  - mean flat fraction: `0.361`
  - mean valid-pair fraction: `0.542`
- `attenuate`:
  - mean low-info fraction: `0.764`
  - mean flat fraction: `0.500`
  - mean valid-pair fraction: `0.403`
- `rank_reduction`:
  - mean low-info fraction: `1.000`
  - mean flat fraction: `0.819`
  - mean valid-pair fraction: `0.125`

## Interpretation
- This pilot does not support replacing current ablation companions with `rank_reduction` at `retain_ratio=0.5`.
- The mode is conceptually relevant to compression, but currently too degenerate for stable reliability analysis.
- The line remains exploratory and bounded:
  - no proxy-role change,
  - no strategy/policy escalation,
  - no scope expansion.

## Next Bounded Move (If Continuing)
- Keep this small and protocol-matched.
- Re-run `rank_reduction` with less aggressive retain ratios (for example `0.75`, optionally `0.85`) to test whether low-information collapse can be reduced while preserving compression relevance.

## Artifacts
- Source sweep JSON: `/Users/john/code/gradience/field_trials/rank_proxy_validation/rank_reduction_soft_ablation_pilot_sweep.json`
- Source sweep MD: `/Users/john/code/gradience/field_trials/rank_proxy_validation/rank_reduction_soft_ablation_pilot_sweep.md`
