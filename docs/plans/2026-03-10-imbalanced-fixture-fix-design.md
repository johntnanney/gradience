# Imbalanced Fixture Fix

## Problem

The `imbalanced_pair` test fixture uses identical subspaces (same U/Vt) with 10x magnitude difference. Because `mean_overlap ≈ 1.0`, verdict Branch 2 (REDUNDANT) fires before Branch 0 (IMBALANCED), so no test actually exercises the imbalanced → linear-rebalanced pipeline path.

## Fix

Change `imbalanced_pair` to use separate random SVDs for each adapter, giving orthogonal subspaces (`mean_overlap ≈ 0`) with `frobenius_ratio ≈ 10`. This satisfies Branch 0: `frobenius_ratio > 5.0 AND mean_overlap < 0.5` → IMBALANCED.

## Files

1. `tests/merge/conftest.py` — two separate SVDs instead of one shared
2. `tests/merge/test_integration.py` — verify/update imbalanced assertions
3. `tests/merge/test_pipeline_integration.py` — restore original intent: verdict "imbalanced", strategy "linear", rebalanced coefficients
