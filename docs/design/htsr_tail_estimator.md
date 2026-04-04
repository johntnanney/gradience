# HTSR Tail Estimator (Bounded, CPU-Only)

## Purpose
Define a lightweight HTSR-style tail-shape probe for saved singular-value spectra that is consistent, transparent, and cheap to run on existing artifacts.

## Scope
- Re-analysis only (no new training)
- Saved singular-value vectors from existing checkpoints/adapters
- Observational diagnostic only

## Estimator
Given descending singular values `sigma_1 >= ... >= sigma_n > 0`, fit:

`log(sigma_i) = -alpha * log(i) + c`

on a tail segment `i >= i0`.

### Tail-selection rule
- Candidate tail fractions: `{0.5, 0.6, 0.7}`
- For each fraction, choose start index `i0 = floor(fraction * n)` with a minimum tail length
- Fit each candidate segment
- Keep the segment with highest `R^2`

This keeps the estimator deterministic while avoiding a single brittle tail cutoff.

## Outputs
- `htsr_alpha`: fitted exponent
- `alpha_fit_quality`: `R^2`
- `alpha_valid`: validity flag
- `tail_start_index`: selected start index
- `tail_points`: points used in tail fit
- `tail_fraction`: selected fraction
- `warning`: explicit failure/low-confidence reason

## Validity gates
`alpha_valid = True` only if all hold:
- spectrum length is sufficient
- fitted `alpha > 0`
- `R^2` meets threshold
- tail points meet minimum count

## Limitations
- This is a bounded empirical probe, not a full heavy-tail theory implementation
- Tail fit quality can degrade for short or very flat spectra
- Results are diagnostic; they are not causal proof of phase structure by themselves
