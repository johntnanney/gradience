# Rank Proxy Validation v2 Bounded Validation Memo

## 1. What Was Tested
- Primary informative subset: imdb, sst2.
- Secondary context subset: ag_news, tweet_eval.
- Methods: energy_90, erank, knee, oht, proxy_ablation_attenuate, proxy_gradient, random_matched_budget, stable_rank_ceil, uniform.
- Budgets: 0.35, 0.5, 0.65.
- Evaluation is dataset-matched per adapter and budget, with source-quality-gap slices retained in the primary interpretation.

## 2. Strongest Positive Result
- Lead spectral policy in the primary informative subset is `oht` (mean delta_vs_uniform=0.0098).
- Spectral policies remain competitive against simple matched-budget baselines in the compressible encoder subset.
- Structural agreement remains stronger against ablation-style proxy patterns than against gradient-style patterns in primary allocation-comparison summaries.

## 3. What Remains Bounded
- Evidence remains bounded to CPU-only encoder classification settings with primary claims restricted to compressible families.
- Saturated families are retained only as secondary context and do not drive main policy interpretation.
- This does not support equivalence to adaptive-training rank-allocation methods.
- v2 is a bounded canonicalization of existing CPU outputs, not a full layer-vector comparison archive.
- Some structure-level claims are therefore based on already-produced comparison artifacts rather than a complete vector-preserving bundle.

## 4. Current Policy Interpretation
- `proxy_gradient` remains the operational default comparator.
- `proxy_ablation_attenuate` remains the explanatory companion comparator.
- `oht` remains the lead spectral policy in this bounded regime.
- Keep the cheap-rank-advisor claim as competitive and bounded, not dominant or universal.

## 5. What Would Strengthen This Line Next
- Recover or regenerate full layerwise allocation vectors as first-class persisted artifacts for future reproducibility slices.
- Expand compressible-family cohort size with explicit source-quality controls before broader claim escalation.
- Add external recovered allocation targets from published adaptive methods before any stronger external validation claim.
