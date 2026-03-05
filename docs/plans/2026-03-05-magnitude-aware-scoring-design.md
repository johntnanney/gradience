# Design: Magnitude-Aware Scoring in Merge Verdicts

## Problem

The 5-branch decision tree in `assess_layer()` (`verdicts.py`) checks magnitude
imbalance at Branch 4, after Branch 1 (orthogonal subspaces -> SAFE) has already
exited. Real-world LoRA adapters are nearly always orthogonal (rank 8-64 in
4096-dim space), so Branch 4 never fires.

Study 16 demonstrated the impact: adapters with 8-20x Frobenius norm ratios all
received "SAFE" verdicts with 50/50 merge coefficients, producing merges where
the smaller adapter's signal was completely drowned out (D > 0.77 in 3/5 pairs).

## Approach

**Reorder branches** so Frobenius imbalance is checked first, but only emit
IMBALANCED when subspace overlap is low. High-overlap layers fall through to the
existing overlap-based verdicts (REDUNDANT/CONFLICTING), which handle shared
subspaces correctly.

## Changes

### 1. `VerdictThresholds` -- new field

Add `imbalanced_frob: float = 5.0` for the Frobenius ratio threshold.

Profile defaults:
- `conservative()`: `imbalanced_frob=3.0`
- `permissive()`: `imbalanced_frob=10.0`

The existing `imbalanced` field (sigma-1 ratio) is retained for backward
compatibility but Branch 4 switches to use `frobenius_ratio`.

### 2. `assess_layer()` -- new branch ordering

```
Branch 0 (NEW):  frobenius_ratio > imbalanced_frob AND mean_overlap < high_overlap
                 -> IMBALANCED (rebalanced coefficients)

Branch 1:        mean_overlap < low_overlap -> SAFE  (unchanged)
Branch 2:        high overlap + aligned -> REDUNDANT  (unchanged)
Branch 3:        high overlap + opposing -> CONFLICTING  (unchanged)
Branch 4:        frobenius_ratio > imbalanced_frob (high-overlap remainder)
                 -> IMBALANCED
Branch 5:        fallback -> SAFE  (unchanged)
```

Branch 0 compound condition: Frobenius imbalance AND low-to-moderate overlap.
Catches orthogonal-but-imbalanced (the Study 16 pattern) while letting
high-overlap layers reach the overlap-based branches.

Rebalancing formula (existing, unchanged):
```python
ratio = metrics.frobenius_ratio
coeff_strong = 1.0 / (1.0 + ratio)
coeff_weak = ratio / (1.0 + ratio)
```

Coefficient ordering uses `frobenius_norm_a` vs `frobenius_norm_b` to map
strong/weak to (coeff_a, coeff_b).

### 3. Tests

New test cases for `assess_layer()`:

| Scenario | frobenius_ratio | mean_overlap | agreement | Expected |
|----------|-----------------|--------------|-----------|----------|
| Orthogonal + imbalanced | 10.0 | 0.05 | - | IMBALANCED |
| High overlap + aligned + imbalanced | 10.0 | 0.6 | 0.7 | REDUNDANT |
| High overlap + opposing + imbalanced | 10.0 | 0.6 | -0.5 | CONFLICTING |
| Orthogonal + balanced | 3.0 | 0.05 | - | SAFE |

### 4. No changes needed

- `SubspaceMetrics` -- already has `frobenius_ratio`, `frobenius_norm_a/b`
- `recommend.py` -- already maps IMBALANCED -> rebalanced linear
- `plan.py` -- already handles IMBALANCED recommendations
- `assess_overall()` -- already counts IMBALANCED in priority ordering
- `scale.py`, `outcomes.py`, `spectral_compat.py` -- unchanged

## Files Touched

| File | Change |
|------|--------|
| `gradience/vnext/merge/verdicts.py` | New threshold field, reorder branches, use `frobenius_ratio` |
| Tests for `verdicts.py` | New test cases for reordered logic |

## Expected Impact (Study 16 pairs)

| Pair | Frob ratio | Old verdict | New verdict |
|------|------------|-------------|-------------|
| metamath x openwebmath (3.1x) | 3.1 | safe | safe |
| metamath x magicoder (2.1x) | 2.1 | safe | safe |
| magicoder x btgenbot (8.3x) | 8.3 | safe | **imbalanced** |
| openwebmath64 x btgenbot (19.7x) | 19.7 | safe | **imbalanced** |
| catsubcat x btgenbot (11.3x) | 11.3 | safe | **imbalanced** |

Pairs 3, 4, 6 would receive rebalanced coefficients instead of 50/50, directly
addressing the dominance (D > 0.77) observed in Phase 1.
