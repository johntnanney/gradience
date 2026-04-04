# n127 - Spectral Edge-Gap Probe Implementation Note

## Implementation
Added `compute_spectral_edge_gap(...)` in `gradience/research/spectral_extended.py`.

Primary output:
- `edge_gap_12 = sigma1 / sigma2`

Companion outputs:
- `edge_gap_13 = sigma1 / sigma3` (if available)
- `sigma1_share_top3`
- `edge_valid`
- `warning`

## Why this shape
- Keeps probe simple and interpretable.
- Avoids proliferation of many edge variants in first pass.
- Matches study goal: additive observable comparison, not metric-ecosystem expansion.

## Validation
Unit coverage added in `tests/test_research_spectral_probes.py` for:
- nominal edge-gap computation,
- short-spectrum invalid behavior.

## Bounded interpretation
Edge-gap is a descriptive dominance probe. It is not, by itself, a causal transition detector.
