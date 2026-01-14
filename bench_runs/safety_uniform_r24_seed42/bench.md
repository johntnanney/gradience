# Gradience Bench v0.1

- **Model:** distilbert-base-uncased
- **Task:** glue/sst2
- **Validation Level:** Screening
  - *Single seed, 500 steps (no variance estimation)*

## Probe

- **Rank:** 32
- **LoRA params:** 1,771,778
- **Accuracy:** 0.828

## Compression results

| Variant | Params | Accuracy | Δ vs probe | Param reduction | Verdict |
|---|---:|---:|---:|---:|---|
| `uniform_median` | 1,476,866 | 0.832 | +0.004 | 16.6% | PASS |
| `uniform_p90` | 1,476,866 | 0.832 | +0.004 | 16.6% | PASS |


## Interpretation (Screening Only)

- **Screening only** - Single-seed validation, suitable for rapid development iteration
- **PASS** means the compressed model didn't hurt accuracy beyond tolerance (±0.025)
- **FAIL** means accuracy dropped more than the tolerance threshold
- You should still validate these results on your real workload before deployment
- Parameter reduction shows the percentage decrease in trainable LoRA parameters

## Summary

- **Recommendations validated:** 2/2
- **Best compression:** uniform_median

*Generated on 2026-01-13 18:47:11*
