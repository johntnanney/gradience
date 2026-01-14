# Gradience Bench v0.1

- **Model:** distilbert-base-uncased
- **Task:** glue/sst2

## Probe

- **Rank:** 16
- **LoRA params:** 1,181,954
- **Accuracy:** 0.630

## Compression results

| Variant | Params | Accuracy | Δ vs probe | Param reduction | Verdict |
|---|---:|---:|---:|---:|---|
| `uniform_median` | 1,181,954 | 0.715 | +0.085 | 0.0% | PASS |
| `uniform_p90` | 1,181,954 | 0.715 | +0.085 | 0.0% | PASS |
| `per_layer` | 887,042 | 0.705 | +0.075 | 25.0% | PASS |


## Interpretation (Conservative)

- **PASS** means the compressed model didn't hurt accuracy beyond tolerance (±0.005)
- **FAIL** means accuracy dropped more than the tolerance threshold
- You should still validate these results on your real workload before deployment
- Parameter reduction shows the percentage decrease in trainable LoRA parameters

## Summary

- **Recommendations validated:** 3/3
- **Best compression:** per_layer

*Generated on 2026-01-13 15:41:10*
