# Explanatory Analysis

## Group Comparison
- Group 1 count: 4 | mean delta vs best: -0.0190
- Group 2 count: 5 | mean delta vs best: -0.0156
- Group 3 anchors count: 4 | mean delta vs best: -0.0435

## Advisory-Band Check
- Official advisory distribution in high-overlap subset: {'none': 9}
- If the distribution is degenerate (all `none`), explanatory comparisons use the bounded proxy split only.

## Continuous Relationships
- Spearman(delta_vs_best, max_over_accumulation_score): 0.2661
- Spearman(delta_vs_best, high_risk_layer_count): n/a
- High-overlap pairs analyzed: 9

## Contrast vs Existing Factors
- Spearman(delta_vs_best, mean_overlap): 0.0258
- Spearman(delta_vs_best, conflict_fraction): n/a
- Spearman(delta_vs_best, imbalance_fraction): -0.8437

## Compact OLS (High-overlap subset)
- Model: `delta_vs_best ~ max_over_accumulation + mean_overlap + conflict_frac + imbalance_frac`
- n=9, rank=4
- Coefficients: intercept=-0.031767, max_oa=-0.007429, overlap=0.085580, conf=0.000000, imb=-0.054692

## Notes
- Official over_accumulation_advisory may be degenerate (all 'none') in this cohort.
- Proxy split is only for bounded first-pass analysis; it does not modify production advisory semantics.
- Delta vs source uses native-task evidence proxy when source eval scores are not dataset-matched.
