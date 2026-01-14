# Gradience Bench v0.1 - Aggregate Report

- **Model:** distilbert-base-uncased
- **Task:** glue/sst2
- **Validation Level:** Certifiable
- **Seeds:** 3

## Probe Baseline

- **Accuracy:** 0.831 ± 0.002
- **Range:** [0.828, 0.832]

## Compression Results

| Variant | Pass Rate | Worst Δ | Mean Accuracy | Param Reduction | Policy Status |
|---------|-----------|---------|---------------|-----------------|---------------|
| uniform_median | 100% | -0.010 | 0.823 | 61.0% | ✅ COMPLIANT |

## Safety Policy

**Safe Uniform Baseline Policy:**
- Pass rate ≥ 67%
- Worst-case Δ ≥ -0.025

## Protocol Invariants

**Overall Status:** ⚠️ WARNING

| Invariant | Status | Seeds Passed |
|-----------|--------|--------------|
| Probe Quality | ✅ PASSED | 3/3 |
| Rank Heterogeneity | ✅ PASSED | 3/3 |
| Layer Consistency | ✅ PASSED | 3/3 |
| Param Counting | ⚠️ WARNING | 0/3 |

## Summary

- **Total variants tested:** 2
- **Policy-compliant variants:** 1
- **Best compression:** uniform_median (61.0% reduction)

### Recommendations
- Use uniform_median for 61.0% compression (policy-compliant)

*Generated on 2026-01-14T08:28:42.374963*