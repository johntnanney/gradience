# OA-v2 Failure Anatomy (30-pair Strict-Naive Run)

## Scope
- Cross-check source: `field_trials/analytical_spectral_geometry/empirical_crosscheck_oa_v2_30_40_r1.json`
- Strict-naive source: `field_trials/over_accumulation_followup/oa_v2_30_40_r1_strict_naive_rerun_results.json`
- Pairs analyzed: `30`
- Intended slice pairs (high-overlap/low-conflict): `9`

## Core Readout
- Overall Spearman(delta, OA-v1)=0.1791, OA-v2=0.2513.
- Intended slice Spearman(delta, OA-v1)=0.1604, OA-v2=-0.0928.
- Intended slice mean delta vs best: `-0.0304`; overall mean delta vs best: `-0.1067`.

## Findings
- OA-v2 improves overall rank correlation but does not hold directionally in the intended high-overlap/low-conflict slice.
- Overall Spearman(delta, OA-v1)=0.1791, OA-v2=0.2513; intended-slice Spearman OA-v1=0.1604, OA-v2=-0.0928.
- Intended slice coverage is 9/30 pairs, which remains a key stability limiter.
- Promotion gate remains failed: rule1=False, rule2=False, rule3=False, rule4=True.

## Task-Family Stratification

| group | n | mean_delta_vs_best | rho_v1 | rho_v2 | rho_gain_v2-v1 | poor_rate | mean_source_gap |
|---|---:|---:|---:|---:|---:|---:|---:|
| sentiment_binary | 22 | -0.1105 | 0.1640 | 0.2251 | 0.0611 | 0.4091 | 0.1895 |
| tweet_eval | 6 | -0.1253 | 0.0857 | 0.6571 | 0.5714 | 0.5000 | 0.1390 |
| topic_classification | 2 | -0.0090 | n/a | n/a | n/a | 0.0000 | 0.0280 |

## Backbone-Family Stratification

| group | n | mean_delta_vs_best | rho_v1 | rho_v2 | rho_gain_v2-v1 | poor_rate | mean_source_gap |
|---|---:|---:|---:|---:|---:|---:|---:|
| distilbert | 28 | -0.0937 | 0.1999 | 0.2931 | 0.0932 | 0.3929 | 0.1573 |
| bert | 1 | -0.5680 | n/a | n/a | n/a | 1.0000 | 0.6200 |
| roberta | 1 | -0.0100 | n/a | n/a | n/a | 0.0000 | 0.0340 |

## Source-Gap Band Stratification

| group | n | mean_delta_vs_best | rho_v1 | rho_v2 | rho_gain_v2-v1 | poor_rate | mean_source_gap |
|---|---:|---:|---:|---:|---:|---:|---:|
| large_gap | 13 | -0.2117 | -0.7015 | -0.8776 | -0.1761 | 0.6154 | 0.3308 |
| near_top | 11 | -0.0069 | -0.3042 | -0.5346 | -0.2304 | 0.0000 | 0.0193 |
| mid_gap | 6 | -0.0623 | 0.7143 | 0.4857 | -0.2286 | 0.6667 | 0.0910 |

## Rank-Mismatch Band Stratification

| group | n | mean_delta_vs_best | rho_v1 | rho_v2 | rho_gain_v2-v1 | poor_rate | mean_source_gap |
|---|---:|---:|---:|---:|---:|---:|---:|
| large_mismatch | 14 | -0.1507 | -0.3894 | -0.3630 | 0.0264 | 0.5000 | 0.2629 |
| matched | 10 | -0.0772 | 0.2866 | 0.0854 | -0.2012 | 0.3000 | 0.0912 |
| moderate_mismatch | 6 | -0.0533 | 0.5161 | 0.5161 | 0.0000 | 0.3333 | 0.0777 |

## Gate Snapshot
- Rule 1 abs Spearman gain >= 0.15: `False`
- Rule 2 recall gain >= 0.20: `False`
- Rule 3 sign consistency >= 0.70: `False`
- Rule 4 interpretability decomposition: `True`
- All-pass: `False`

## Status
- OA-v2 remains exploratory; no threshold/policy promotion from this run.
- Next-run candidate should increase intended-slice coverage before any further gate decision.

