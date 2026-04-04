# Linear Merge Spectral Analysis

## Core Setup

For two adapter deltas:

`ΔW_m = αΔW_a + βΔW_b`

with SVD decompositions:

`ΔW_a = U_a Σ_a V_a^T`, `ΔW_b = U_b Σ_b V_b^T`.

Principal-angle structure enters through `U_a^T U_b` and `V_a^T V_b`.

## Rank-1 exact expression

For rank-1 terms:

`ΔW_m = αΔW_a + βΔW_b`, with:

- `a = |α|σ_a`
- `b = |β|σ_b`
- `z = sign(δ)·cos(θ)·cos(φ)`

The non-zero eigenvalues of `(ΔW_m)(ΔW_m)^T` satisfy:

`λ² - Tλ + D = 0`

where:

- `T = a² + b² + 2abz`
- `D = a²b²(1-cos²(θ))(1-cos²(φ))`

and the leading singular value is:

`σ_1(ΔW_m) = sqrt((T + sqrt(T² - 4D))/2)`

where:

- `cos(θ)` is left-space overlap,
- `cos(φ)` is right-space overlap,
- `δ` is directional agreement sign proxy.

Implemented utility:

- `rank1_linear_merge_result(...)`

## General sigma_1 bounds

Using sigma-1 bounds:

- upper: `σ_1(αA+βB) <= |α|σ_1(A) + |β|σ_1(B)`
- lower: `σ_1(αA+βB) >= max(0, ||α|σ_1(A) - |β|σ_1(B)|)`

Implemented utility:

- `linear_merge_sigma1_bounds(...)`

## Equal-rank equal-spectrum exact solution

When `Σ_a = Σ_b = diag(σ_1,...,σ_r)` and principal angle decomposition
aligns with spectral decomposition, the merged Gram matrix decomposes
into `r` independent 2x2 blocks.

For direction `i`, with `a_i = ασ_i`, `b_i = βσ_i`:

- `T_i = a_i² + b_i² + 2a_ib_i cos(θ_i)cos(φ_i) sign_i`
- `D_i = a_i²b_i² sin²(θ_i) sin²(φ_i)`
- `λ_{i,±} = (T_i ± sqrt(T_i² - 4D_i)) / 2`

Special cases:
- **Orthogonal** (`cos=0`): `λ_+ = max(α,β)²σ²`, `λ_- = min(α,β)²σ²`
- **Identical** (`cos=1`): `λ_+ = (α+β)²σ²`, `λ_- = 0`

Implemented utility: `equal_rank_equal_spectrum_merge(...)`

## Frobenius norm of merge (exact)

`||αA + βB||²_F = α²||A||²_F + β²||B||²_F + 2αβ·tr(A^T B)`

where the cross trace under the diagonal-dominant approximation is:

`tr(A^T B) ≈ sign(δ) · Σ_i s_{a,i} s_{b,i} cos(θ_i) cos(φ_i)`

Implemented utility: `linear_merge_frobenius(...)`

## General rank-r Gram decomposition

For arbitrary rank and spectrum, the cross-term spectral norm is bounded by:

`σ_1(cross) ≤ 2αβ · Σ_i s_{a,i} s_{b,i} cos(θ_i) cos(φ_i)`

This is tight when principal directions align with spectral directions.
Combined with Weyl bounds on σ_1, this gives:

- Over-accumulation possible iff cross-term has positive sign AND non-zero overlap
- Cross-term Frobenius bounded by `2αβ · ||A||_F · ||B||_F · max(cos(θ))`

Implemented utility: `general_rank_gram_decomposition(...)`

## Stable rank bounds

Stable rank = `||ΔW_m||²_F / σ_1(ΔW_m)²`.

Using exact Frobenius (above) and Weyl σ_1 bounds:

- `sr_lower = ||ΔW_m||²_F / σ_1_upper²`
- `sr_upper = ||ΔW_m||²_F / σ_1_lower²`

Implemented utility: `linear_merge_stable_rank_bounds(...)`

## Validation hooks

- synthetic rank-1 generators in `spectral_theory_test_utils.py`
- synthetic rank-r generators with controlled geometry: `make_rankr_pair_with_geometry()`
- equal-spectrum generators: `make_equal_spectrum_pair()`
- rank-1 formula validation grid: `test_rank1_formula_validation_grid_small`
- Frobenius validation grid: `test_frobenius_validation_rankr_grid`
- Equal-spectrum validation grid: `test_equal_spectrum_validation_grid`
- Weyl bounds validation grid: `test_weyl_bounds_validation_grid`
