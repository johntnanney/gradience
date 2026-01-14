# Gradience Bench v0.1

- **Model:** distilbert-base-uncased
- **Task:** glue/sst2

## Probe

- **Rank:** 16
- **LoRA params:** 1,181,954
- **Accuracy:** 0.795

## Compression results

| Variant | Params | Accuracy | Δ vs probe | Param reduction | Verdict |
|---|---:|---:|---:|---:|---|
| `uniform_median` | 739,586 | 0.685 | -0.110 | 37.4% | FAIL |
| `uniform_p90` | 1,181,954 | 0.765 | -0.030 | 0.0% | FAIL |
| `per_layer` | 1,046,786 | 0.745 | -0.050 | 11.4% | FAIL |


## Interpretation (Conservative)

- **PASS** means the compressed model didn't hurt accuracy beyond tolerance (±0.005)
- **FAIL** means accuracy dropped more than the tolerance threshold
- You should still validate these results on your real workload before deployment
- Parameter reduction shows the percentage decrease in trainable LoRA parameters

## Summary

- **Recommendations validated:** 0/3
- **Best compression:** None

*Generated on 2026-01-13 17:33:10*
