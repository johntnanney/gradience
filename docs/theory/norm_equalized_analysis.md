# Norm-Equalized Merge Analysis

## Definition

Norm-equalized merge rescales both adapters to geometric-mean Frobenius norm
before linear combination.

For norms `||A||_F`, `||B||_F` and `g = sqrt(||A||_F ||B||_F)`:

- scale for `A`: `g / ||A||_F`
- scale for `B`: `g / ||B||_F`

Implemented utility:

- `norm_equalized_scaling(...)`

## Immediate implication

When Frobenius ratio is large, norm-equalization compresses the dominant
adapter and amplifies the weaker one, reducing scale imbalance before merge.

## Next derivation targets

1. Derive over-accumulation suppression conditions under norm-equalization.
2. Characterize distortion when spectral shapes differ at equal Frobenius norm.
3. Compare analytical predictions to synthetic sweeps over `ρ_F`.
