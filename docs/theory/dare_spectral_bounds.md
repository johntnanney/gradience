# DARE Spectral Bounds (Planned)

## Scope

DARE introduces random dropout + rescaling before merge. The analysis target is
expected spectral behavior and concentration bounds.

## Immediate helper

Implemented:

- `expected_dare_frobenius_multiplier(p) = sqrt(1 / (1 - p))`

This is the expected norm inflation factor under Bernoulli dropout + rescale.

## Next derivation targets

1. Expected perturbation of leading singular values under dropout.
2. Concentration bounds on run-to-run spectral deviation.
3. Over-accumulation reduction as function of dropout fraction and overlap.
