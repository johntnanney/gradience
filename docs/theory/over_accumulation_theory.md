# Over-Accumulation Theory (Analytical Line)

## Target condition

Over-accumulation is defined as:

`σ_1(ΔW_m) > max(|α|σ_1(ΔW_a), |β|σ_1(ΔW_b))`

for merge `ΔW_m = αΔW_a + βΔW_b`.

## Rank-1 exact condition

For rank-1 factors:

- `ΔW_a = σ_a u_a v_a^T`
- `ΔW_b = σ_b u_b v_b^T`
- `cos(θ) = <u_a, u_b>`
- `cos(φ) = <v_a, v_b>`

the merge cross term is:

`2|α||β|σ_aσ_b cos(θ)cos(φ)sign(δ)`

where `δ` is directional agreement.

Inflation requires this cross term to be positive and large enough to push
`σ_1(ΔW_m)` above the weighted baseline.

## Phase-2/3 implementation bridge (rank-r interaction)

The OA line now has a parallel OA-v2 implementation that uses full rank-r
geometry from `SubspaceMetrics`:

- `right_principal_angle_cosines` (row-space principal angles)
- `effective_singular_values_a`
- `effective_singular_values_b`

Core interaction construction:

- `w_i = s_{a,i} * s_{b,i}`
- `ŵ_i = w_i / (Σ_j w_j + eps)`
- `pair_i = clamp(c_left_i,0,1) * clamp(c_right_i,0,1)`
- `spectral_overlap_weighted = Σ_i ŵ_i * pair_i`

This is implemented in:

- `compute_rankr_interaction_geometry(...)` in
  [`gradience/vnext/merge/spectral_theory.py`](../../gradience/vnext/merge/spectral_theory.py)
- `estimate_over_accumulation_v2(...)` in
  [`gradience/vnext/merge/over_accumulation.py`](../../gradience/vnext/merge/over_accumulation.py)

OA-v2 scoring contract:

- `directional_gate = clamp(max(direction_agreement, 0), 0, 1)`
- `coeff_exposure = 4ab` (normalized coefficients)
- `interaction_primary = directional_gate * coeff_exposure * spectral_overlap_weighted`
- `score_v2 = clamp(interaction_primary * (0.85 + 0.15 * concentration_secondary), 0, 1)`

where concentration is secondary modulation only.

OA-v1 remains the authoritative production advisory path; OA-v2 is parallel and
experimental.

## Validation status

- Higher-rank synthetic sweep (`r in {4,8,16,32}`) has been added via:
  `field_trials/analytical_spectral_geometry/run_phase2_heuristic_comparison.py`
- Strict-naive field-trial cross-check has been added via:
  `field_trials/analytical_spectral_geometry/run_empirical_crosscheck.py`
- Expanded strict-naive 30-pair run (`oa_v2_30_40_r1`) completed and analyzed.
- Failure-anatomy decomposition added (task family / backbone / source-gap / rank-mismatch)
  via `field_trials/analytical_spectral_geometry/run_oa_v2_failure_anatomy.py`.
- Threshold/policy updates remain blocked behind gate criteria in the empirical
  cross-check artifact.

## Boundaries

- This is an analytical interaction surrogate, not a proved optimal general-rank theorem.
- It is intended for bounded explanatory comparison first.
- Policy changes should require synthetic + field-trial validation.

## Exact rank-r over-accumulation condition

The cross-term spectral bound for rank-r linear merge is:

`σ_1(cross) ≤ 2αβ · Σ_i s_{a,i} s_{b,i} cos(θ_i) cos(φ_i)`

Over-accumulation requires:
1. Positive directional agreement (sign > 0)
2. Non-zero cross-term spectral bound
3. Cross-term large enough to push σ_1 above baseline

The inflation ratio upper bound is:

`inflation_ratio = σ_1_upper / baseline_σ_1`

where `baseline_σ_1 = max(α·s_{a,1}, β·s_{b,1})`.

Analytical risk score: `clip01(inflation_ratio - 1.0)`.

The `inflation_driven_by` field classifies the dominant factor:
- `"alignment"`: high overlap (mean_overlap > 0.5) is the primary driver
- `"scale"`: Frobenius imbalance (ratio > 3.0) with low overlap
- `"both"`: moderate overlap + moderate imbalance
- `"none"`: negligible cross-term

Implemented utility: `over_accumulation_exact_condition(...)`

### Comparison to heuristic

The heuristic formula `alignment × (0.7·concentration + 0.3·coeff_exposure)`:

- Uses `mean_overlap` and `max_overlap` as proxies for the full principal
  angle spectrum — loses information about per-direction interaction strength
- Uses `stable_rank/effective_rank` as concentration proxy — this is a
  spectral shape measure but doesn't directly enter the cross-term bound
- Does not use right-space principal angles (`cos(φ_i)`) at all — the
  heuristic was designed before `right_principal_angle_cosines` was available

The analytical condition is more directly tied to the Gram matrix structure
but requires the v2 geometry fields (`effective_singular_values_*`,
`right_principal_angle_cosines`).

The `analytical_vs_heuristic_alignment` field reports whether the two
estimates agree, or which one is higher.

## Norm-equalized over-accumulation analysis

Norm equalization rescales adapters to geometric-mean Frobenius norm:
- `scale_a = sqrt(||B||_F / ||A||_F)`
- `scale_b = sqrt(||A||_F / ||B||_F)`

This changes the effective σ_1 contributions to the cross-term.
Norm equalization compresses the dominant adapter's spectrum but
**amplifies the subordinate adapter's spectrum**. If the subspaces
overlap (non-trivial principal angle cosines), the amplified subordinate
creates a larger cross-term, potentially *increasing* inflation beyond
the linear-merge baseline.

**Empirical finding:** For a 30x Frobenius-imbalanced pair with
moderate subspace overlap, norm equalization inflated the leading
singular value by 3.26x vs only 1.07x for linear merge. This is not
an edge case — it occurs precisely when the IMBALANCED verdict fires
(large `frobenius_ratio` + non-trivial overlap).

**When norm equalization helps:** Low subspace overlap (orthogonal
adapters) or balanced norms (ρ_F ≈ 1). In these regimes the cross-term
amplification is negligible.

**Pipeline implication:** The `_derive_strategy` policy routes
medium-risk imbalanced pairs to `"linear"` with rebalanced coefficients
rather than `"norm_equalized"`.

Implemented utility: `norm_equalized_over_accumulation_analysis(...)`

The `reduction_factor` field is `normeq_ratio / linear_ratio`:
- `< 1.0`: norm equalization reduces inflation risk
- `= 1.0`: no effect (already balanced)
- `> 1.0`: norm equalization increases risk (common for imbalanced
  pairs with non-trivial subspace overlap)

## Next derivation targets

1. Add strategy-specific analytical extensions (TIES/DARE) on top of OA-v2 geometry.
2. Promote or pause OA-v2 only after empirical gate outcomes stabilize across larger cohorts.
3. Derive tighter bounds using Lidskii-Wielandt interlacing when full principal angle spectra are available.
