# Rank Advisor Role Table (OHT vs Gradient vs Attenuate)

## Bounded Role Snapshot
| Method | Bounded Role | Primary Subset Outcome (`mean delta_vs_uniform`) | Structural/diagnostic signal | External wording guidance |
| --- | --- | --- | --- | --- |
| `oht` | Lead spectral policy | `+0.0098` | Aligns more with ablation-style structure than gradient-style structure in family readouts | "Lead spectral allocator in this bounded CPU encoder regime." |
| `proxy_gradient` | Operational default comparator | `+0.0119` (best among compared methods in primary subset) | Higher resampling stability than ablation (`mean pairwise Spearman ~0.903`) | "Current strongest operational proxy target under this protocol." |
| `proxy_ablation_attenuate` | Explanatory companion comparator | `+0.0023` | Useful structural companion signal; lower resampling stability than gradient (`~0.400`) | "Companion structural signal, not current operational default." |

## Scope Notes
- Primary informative subset: `sst2`, `imdb`.
- Secondary context only: `tweet_eval`, `ag_news` (saturated in this pass).
- Values above come from the canonical v2 artifacts listed below.

## Sources
- [compression_evaluation_table.json](/Users/john/code/gradience/field_trials/rank_proxy_validation_v2/compression_evaluation_table.json)
- [bounded_validation_summary.json](/Users/john/code/gradience/field_trials/rank_proxy_validation_v2/bounded_validation_summary.json)
- [disagreement_memo.md](/Users/john/code/gradience/field_trials/rank_proxy_validation_v2/disagreement_memo.md)
