# Analytical Spectral Geometry of Merge Operations — Summary

## Status

Phase 2 rank-r parallel OA-v2 implemented (CPU-only).

## Objective

Express merge-strategy spectral behavior in terms of observables already
computed by Gradience (`SubspaceMetrics`), then validate analytically and
numerically on synthetic matrix pairs.

## Implemented Foundations

- Exact rank-1 linear-merge formula utilities
- Sigma-1 linear merge bounds (Weyl/triangle style)
- Norm-equalized scaling factors
- DARE expected Frobenius multiplier helper
- Synthetic rank-1 geometry generators for validation tests
- Over-accumulation Phase-2 surrogate in SubspaceMetrics coordinates
- Analytical-vs-heuristic comparison API
- Rank-r interaction geometry utility with full left/right principal-angle spectra
- OA-v2 interaction-first estimator (parallel to OA-v1, non-authoritative)
- Higher-rank synthetic sweep runner and comparison artifacts
- Strict-naive empirical cross-check runner and gate artifact

Code:

- `gradience/vnext/merge/spectral_theory.py`
- `gradience/vnext/merge/spectral_theory_test_utils.py`

## Phase-2 Result Snapshot

- In higher-rank controlled sweeps (`r=4/8/16/32`), OA-v2 correlation with true
  inflation margin is materially stronger than OA-v1.
- Strict-naive empirical cross-check remains gate-failing for threshold/policy
  promotion, so OA-v1 remains authoritative.
- OA-v2 is currently exploratory/parallel evidence, not policy.

## Next Phases

1. Lift the surrogate from rank-1 proxy to tighter general-rank conditions.
2. Add semi-analytical TIES/DARE bounds.
3. Build strategy-selection map against verdict regions.
4. Run field-trial cross-check on real adapter pairs before any policy change.

## Boundaries

- no behavioral-performance claims
- no immediate threshold/policy changes
- theory and synthetic validation first
