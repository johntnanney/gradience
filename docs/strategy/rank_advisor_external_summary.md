# Rank Advisor External Summary (Bounded)

## Intended Use
This is the short external-writing summary for the CPU-only adaptive-rank comparison line.

## Bounded Claim
In the compressible encoder subset (`sst2`, `imdb`), Gradience spectral rank policies are competitive fixed-budget allocation guides. `oht` is the lead spectral policy in this bounded regime, while `proxy_gradient` remains the stronger operational comparator and `proxy_ablation_attenuate` remains a structural/explanatory companion.

## What To Say
- Scope is bounded to CPU-only encoder classification adapters in compressible families.
- Spectral guidance is competitive under matched budgets, with `oht` as the current lead spectral policy.
- Operationally, `proxy_gradient` still leads mean compression outcome in the primary subset.
- Structurally, spectral allocations align more with ablation-style patterns than gradient-style patterns.
- This v2 package is a bounded canonicalization of existing CPU evidence rather than a full layer-vector archive.

## What Not To Say
- Do not claim equivalence to adaptive-rank training methods.
- Do not generalize to decoder models or broad cross-architecture settings.
- Do not use saturated families (`tweet_eval`, `ag_news`) as primary evidence.
- Do not frame this as universal policy dominance.
- Do not imply structure-level findings come from a complete vector-preserving bundle; they come from persisted comparison artifacts.

## Canonical Sources
- [bounded_validation_memo.md](/Users/john/code/gradience/field_trials/rank_proxy_validation_v2/bounded_validation_memo.md)
- [bounded_validation_summary.json](/Users/john/code/gradience/field_trials/rank_proxy_validation_v2/bounded_validation_summary.json)
- [disagreement_memo.md](/Users/john/code/gradience/field_trials/rank_proxy_validation_v2/disagreement_memo.md)
- [source_quality_gap_control_slice.md](/Users/john/code/gradience/field_trials/rank_proxy_validation_v2/source_quality_gap_control_slice.md)
