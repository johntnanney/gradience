# Gradience Bench v0.1

- **Model:** distilbert-base-uncased
- **Task:** glue/sst2
- **Validation Level:** Screening
  - *Single seed, 500 steps (no variance estimation)*

## Probe

- **Rank:** 32
- **LoRA params:** 1,771,778
- **Accuracy:** 0.832

## Compression results

| Variant | Params | Accuracy | Δ vs probe | Param reduction | Verdict |
|---|---:|---:|---:|---:|---|
| `uniform_median` | 665,858 | 0.826 | -0.006 | 62.4% | PASS |


## Interpretation (Screening Only)

- **Screening only** - Single-seed validation, suitable for rapid development iteration
- **PASS** means the compressed model didn't hurt accuracy beyond tolerance (±0.025)
- **FAIL** means accuracy dropped more than the tolerance threshold
- You should still validate these results on your real workload before deployment
- Parameter reduction shows the percentage decrease in trainable LoRA parameters

## Summary

- **Recommendations validated:** 1/1
- **Best compression:** uniform_median

*Generated on 2026-01-14 08:24:07*
