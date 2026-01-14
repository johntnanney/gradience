# Gradience Bench v0.1

- **Model:** distilbert-base-uncased
- **Task:** glue/sst2

## Probe

- **Rank:** 32
- **LoRA params:** 1,771,778
- **Accuracy:** 0.800

## Compression results

| Variant | Params | Accuracy | Δ vs probe | Param reduction | Verdict |
|---|---:|---:|---:|---:|---|
| `uniform_median` | 887,042 | 0.695 | -0.105 | 49.9% | FAIL |
| `uniform_p90` | 1,771,778 | 0.805 | +0.005 | 0.0% | PASS |
| `per_layer` | 1,409,282 | 0.790 | -0.010 | 20.5% | PASS |


## Interpretation (Conservative)

- **PASS** means the compressed model didn't hurt accuracy beyond tolerance (±0.025)
- **FAIL** means accuracy dropped more than the tolerance threshold
- You should still validate these results on your real workload before deployment
- Parameter reduction shows the percentage decrease in trainable LoRA parameters

## Summary

- **Recommendations validated:** 2/3
- **Best compression:** per_layer

*Generated on 2026-01-13 18:05:24*
