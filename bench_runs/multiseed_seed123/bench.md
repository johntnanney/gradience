# Gradience Bench v0.1

- **Model:** distilbert-base-uncased
- **Task:** glue/sst2

## Probe

- **Rank:** 32
- **LoRA params:** 1,771,778
- **Accuracy:** 0.795

## Compression results

| Variant | Params | Accuracy | Δ vs probe | Param reduction | Verdict |
|---|---:|---:|---:|---:|---|
| `uniform_median` | 739,586 | 0.600 | -0.195 | 58.3% | FAIL |
| `uniform_p90` | 1,771,778 | 0.780 | -0.015 | 0.0% | PASS |
| `per_layer` | 1,366,274 | 0.760 | -0.035 | 22.9% | FAIL |


## Interpretation (Conservative)

- **PASS** means the compressed model didn't hurt accuracy beyond tolerance (±0.025)
- **FAIL** means accuracy dropped more than the tolerance threshold
- You should still validate these results on your real workload before deployment
- Parameter reduction shows the percentage decrease in trainable LoRA parameters

## Summary

- **Recommendations validated:** 1/3
- **Best compression:** uniform_p90

*Generated on 2026-01-13 18:05:47*
