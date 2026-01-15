# Gradience Bench Aggregate Report

**Model:** distilbert-base-uncased  
**Task:** glue/sst2  
**Policy:** Safe Uniform Baseline v0.1  
**Date:** 2026-01-14T23:30:00.000000  

## Executive Summary

✅ **POLICY COMPLIANT**: uniform_median achieves 61% compression with 100% pass rate

## Policy Compliance

- **Status:** COMPLIANT
- **Pass Rate:** 100% (3/3 seeds)
- **Worst Case:** -1.0% (threshold: -2.5%)
- **Details:** All seeds passed accuracy tolerance (Δ ≥ -2.5%)

## Best Compression Variant

**uniform_median** - 61% parameter reduction
- Mean accuracy delta: -0.9%
- Std accuracy delta: 0.08%
- All seeds preserved accuracy within tolerance

## Detailed Results by Seed

| Seed | Variant | Compression | Accuracy Delta | Status |
|------|---------|-------------|----------------|--------|
| 42   | uniform_median | 61% | -1.0% | ✅ PASS |
| 123  | uniform_median | 61% | -0.8% | ✅ PASS |
| 456  | uniform_median | 61% | -0.9% | ✅ PASS |

## Protocol Invariants

All required invariants verified:
- ✅ Compression < Probe (61% < 100%)
- ✅ Accuracy preserved (worst: -1.0% > -2.5%)
- ✅ Multi-seed consistency (std: 0.08%)

## Recommendations

Based on these results, **uniform_median with 61% compression** is the recommended safe baseline for DistilBERT/SST-2. This configuration:

1. Achieves substantial parameter reduction (61%)
2. Maintains accuracy within safe bounds (worst-case -1.0%)
3. Shows consistent behavior across seeds (low variance)
4. Satisfies all policy requirements

## Citation

To cite these results:
```
Gradience Bench v0.1: DistilBERT/SST-2
Compression: uniform_median (61% parameter reduction)
Policy compliance: COMPLIANT (100% pass rate, 3/3 seeds)
Repository: https://github.com/johntnanney/gradience
```

## Notes

- Results are specific to DistilBERT on SST-2 classification
- Hardware and environment details in env.txt
- Protocol version: Gradience Bench 0.1