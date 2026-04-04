# N133 — Direction-Aware Compatibility Baseline

- Generated: `2026-04-04T17:55:28+00:00`
- Source cohort: `field_trials/over_accumulation_followup/oa_v2_30_40_r1_strict_naive_rerun_results.json`
- Strict-naive pairs: `30`
- Coarse audit threshold: `0.9`
- Direction-aware threshold: `0.999`

## Frozen Baseline
- Outcome bins: `{'catastrophic': 9, 'fragile': 3, 'near_miss': 2, 'retained': 16}`
- Dataset mix: `{'tweet_eval/emotion': 2, 'sst2': 16, 'imdb': 6, 'tweet_eval/hate': 1, 'tweet_eval/irony': 3, 'ag_news': 2}`
- Cohort groups: `{'high_tail': 15, 'lower_tail_matched': 15}`

## Study Focus
- Test whether direction-aware structure (top-k + band-partitioned) explains variance beyond coarse summaries.
- Keep this as bounded reanalysis only (no policy/verdict replacement).

## Candidate Metric Families
- Top-band directional alignment (`top1`, `top4`, head-band).
- Mid-spectrum disagreement and head-minus-middle gaps.
- Conflict concentration in small directional subsets.
