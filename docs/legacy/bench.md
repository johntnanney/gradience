# Gradience Bench v0.1

- **Model:** distilbert-base-uncased
- **Task:** glue/sst2
- **Validation Level:** Screening
  - *Single seed, 200 steps (quick validation only)*

## Probe

- **Rank:** 16
- **LoRA params:** 1,181,954
- **Accuracy:** 0.790

## Compression results

| Variant | Params | Accuracy | Δ vs probe | Param reduction | Verdict |
|---|---:|---:|---:|---:|---|
| `uniform_median` | 665,858 | 0.590 | -0.200 | 43.7% | FAIL |
| `uniform_p90` | 887,042 | 0.730 | -0.060 | 25.0% | FAIL |
| `per_layer` | 1,108,226 | 0.785 | -0.005 | 6.2% | PASS |
| `per_layer_shuffled` | 1,108,226 | 0.775 | -0.015 | 6.2% | PASS |


## Interpretation (Screening Only)

- **Screening only** - Single-seed validation, suitable for rapid development iteration
- **PASS** means the compressed model didn't hurt accuracy beyond tolerance (±0.025)
- **FAIL** means accuracy dropped more than the tolerance threshold
- You should still validate these results on your real workload before deployment
- Parameter reduction shows the percentage decrease in trainable LoRA parameters

## Magnitude diagnostics (LoRA ΔW)

### Update magnitude

- **Mean ||ΔW||_F:** 0.131405
- **Mean ||ΔW||_2:** 0.122396

### Top 5 layers by Δ energy

1. **Layer 5:** 26.6% (0.125952)
2. **Layer 4:** 19.6% (0.093121)
3. **Layer 3:** 19.5% (0.092707)
4. **Layer 2:** 13.2% (0.062481)
5. **Layer 1:** 12.1% (0.057331)

### Energy concentration

- **Top-1 layers (10%):** 26.6% of energy
- **Concentration index (HHI):** 0.187
- ✅ **Well distributed** adaptation


## Summary

- **Recommendations validated:** 2/4
- **Best compression:** per_layer

*Generated on 2026-01-26 15:53:59*
