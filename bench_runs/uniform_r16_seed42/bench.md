# Gradience Bench v0.1

- **Model:** distilbert-base-uncased
- **Task:** glue/sst2
- **Validation Level:** Screening
  - *Single seed, 200 steps (quick validation only)*

## Probe

- **Rank:** 32
- **LoRA params:** 1,771,778
- **Accuracy:** 0.790

## Compression results

| Variant | Params | Accuracy | Δ vs probe | Param reduction | Verdict |
|---|---:|---:|---:|---:|---|
| `uniform_median` | 1,181,954 | 0.765 | -0.025 | 33.3% | FAIL |
| `uniform_p90` | 1,181,954 | 0.765 | -0.025 | 33.3% | FAIL |


## Interpretation (Screening Only)

- **Screening only** - Single-seed validation, suitable for rapid development iteration
- **PASS** means the compressed model didn't hurt accuracy beyond tolerance (±0.025)
- **FAIL** means accuracy dropped more than the tolerance threshold
- You should still validate these results on your real workload before deployment
- Parameter reduction shows the percentage decrease in trainable LoRA parameters

## Summary

- **Recommendations validated:** 0/2
- **Best compression:** None

*Generated on 2026-01-13 18:31:18*
