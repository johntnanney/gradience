# CPU Rank-Proxy Validation Summary

## Frozen Proxy Policy (Bounded)
- `proxy_gradient` is the main operational comparison target in this bounded CPU encoder setup.
- `attenuate` is the preferred ablation-style companion comparator.
- `rank_reduction` is paused in this regime due degenerate low-information behavior in bounded reruns.

## Outcome
- CPU-only proxy comparison run completed for spectral rank policies, offline proxy targets, and matched-budget baselines.
- adapters evaluated: 12
- budgets: [0.35, 0.5, 0.65]

## What This Supports
- Whether Gradience's cheap spectral suggestions recover non-trivial layerwise allocation structure compared to gradient/ablation proxies.
- Whether matched-budget compression behavior is competitive with simple baselines in this bounded encoder regime.

## Task-Family Readout
- Primary interpretation subset: sst2, imdb.
- Saturated/non-informative context: tweet_eval, ag_news.
- `sst2`: adapters=3, effective_compression=yes, mean_realized_budget=0.498.
- `imdb`: adapters=3, effective_compression=yes, mean_realized_budget=0.490.
- `tweet_eval`: adapters=5, effective_compression=no, mean_realized_budget=1.000.
- `ag_news`: adapters=1, effective_compression=no, mean_realized_budget=1.000.
- Best-by-family method should only be interpreted where effective compression is active.
- `sst2` best-by-mean-delta-vs-uniform: `proxy_gradient` (0.0237).
- `imdb` best-by-mean-delta-vs-uniform: `proxy_ablation` (0.0023).

## Source-Quality Gap Control (Informative Subset)
- Primary slice uses dataset-matched source-quality gap bands only inside informative families.
- `near_top` best-by-mean-delta-vs-uniform: `proxy_gradient` (0.0052).
- `mid_gap` best-by-mean-delta-vs-uniform: `baseline_random` (0.0590).
- `large_gap` best-by-mean-delta-vs-uniform: `baseline_random` (0.0000).

## Guardrail
- Treat this as bounded advisor-oriented evidence, not adaptive-training SOTA equivalence.
- Families with realized budget near 1.0 are weak evidence for allocation-method differences.
- Do not treat ablation-path findings as grounds to replace gradient as operational comparator in this regime.
