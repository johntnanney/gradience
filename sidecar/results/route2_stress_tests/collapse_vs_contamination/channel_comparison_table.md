# Collapse vs Contamination Replication -- Channel Comparison

| target_id | expected_channel | failure_rate | confidence_collapse_rate | high_confidence_wrong_rate | confusion_or_neither_source_rate | dominant signature |
|---|---|---:|---:|---:|---:|---|
| R1_FR02_case | collapse_like | 0.336 | 0.056 | 0.008 | 0.154 | collapse_uncertainty_dominant |
| R2_FR01_even_slice | collapse_like | 0.344 | 0.068 | 0.000 | 0.160 | collapse_uncertainty_dominant |
| R3_CT01_even_slice | contamination_like | 0.176 | 0.008 | 0.056 | 0.140 | contamination_confident_wrong_dominant |
| R4_CT01_odd_slice | contamination_like | 0.172 | 0.004 | 0.036 | 0.148 | contamination_confident_wrong_dominant |

## Channel means

| channel | mean_failure_rate | mean_confidence_collapse_rate | mean_high_confidence_wrong_rate | mean_confusion_or_neither_source_rate |
|---|---:|---:|---:|---:|
| collapse_like | 0.340 | 0.062 | 0.004 | 0.157 |
| contamination_like | 0.174 | 0.006 | 0.046 | 0.144 |

## Readout

1. Confidence collapse separates channels strongly (collapse_like about 10.3x contamination_like).
2. High-confidence wrong separates channels strongly in the opposite direction (contamination_like about 11.5x collapse_like).
3. Confusion/neither-source rate remains close across channels, so rate-level novelty pressure alone does not identify the channel.
