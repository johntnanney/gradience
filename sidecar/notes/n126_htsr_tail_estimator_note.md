# n126 - HTSR Tail Estimator Implementation Note

## Implementation
Added `fit_htsr_tail_exponent(...)` in `gradience/research/spectral_extended.py`.

Estimator behavior:
- accepts saved singular values,
- evaluates tail windows at fractions `{0.5, 0.6, 0.7}`,
- fits log-log slope per candidate,
- selects highest-`R^2` fit,
- emits validity and warning flags.

Output surface:
- `htsr_alpha`
- `alpha_fit_quality`
- `alpha_valid`
- `tail_start_index`
- `tail_points`
- `tail_fraction`
- `warning`

## Guarded failure paths
- `insufficient_spectrum_length`
- `tail_fit_failed`
- `non_positive_alpha`
- `low_fit_quality`

## Validation
Unit coverage added in `tests/test_research_spectral_probes.py` for:
- recoverable synthetic power-law signal,
- short-spectrum invalid behavior.

## Bounded interpretation
This estimator is a lightweight probe for comparative observables analysis. It is not a full HTSR program or a universal tail-law claim.
