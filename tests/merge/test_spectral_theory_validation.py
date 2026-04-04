from __future__ import annotations

import math

from gradience.vnext.merge.spectral_theory import (
    Rank1MergeInputs,
    equal_rank_equal_spectrum_merge,
    general_rank_gram_decomposition,
    linear_merge_frobenius,
    rank1_linear_merge_result,
)
from gradience.vnext.merge.spectral_theory_test_utils import (
    frobenius,
    make_equal_spectrum_pair,
    make_rank1_pair_with_cosines,
    make_rankr_pair_with_geometry,
    sigma1,
)


def test_rank1_formula_validation_grid_small() -> None:
    alpha, beta = 0.5, 0.5
    sigma_a, sigma_b = 2.0, 1.5
    cosines = [0.0, 0.3, 0.6, 0.9]
    # Exact validation here uses generated matrices with positive alignment sign.
    agreements = [1.0]

    max_abs_err = 0.0
    for i, ctheta in enumerate(cosines):
        for j, cphi in enumerate(cosines):
            for k, da in enumerate(agreements):
                delta_a, delta_b = make_rank1_pair_with_cosines(
                    sigma_a=sigma_a,
                    sigma_b=sigma_b,
                    cos_theta=ctheta,
                    cos_phi=cphi,
                    seed=100 + i * 17 + j * 31 + k,
                )
                observed = sigma1(alpha * delta_a + beta * delta_b)
                pred = rank1_linear_merge_result(
                    Rank1MergeInputs(
                        alpha=alpha,
                        beta=beta,
                        sigma_a=sigma_a,
                        sigma_b=sigma_b,
                        cos_theta=ctheta,
                        cos_phi=cphi,
                        directional_agreement=da,
                    )
                ).sigma1
                max_abs_err = max(max_abs_err, abs(observed - pred))

    assert max_abs_err < 1e-5, f"max_abs_err={max_abs_err}"
    assert math.isfinite(max_abs_err)


def test_frobenius_validation_rankr_grid() -> None:
    """Validate Frobenius norm prediction across rank-r parameter grid."""
    ranks = [2, 4, 8]
    coeff_pairs = [(0.5, 0.5), (0.7, 0.3)]
    max_abs_err = 0.0
    n_cases = 0

    for rank in ranks:
        # Simple spectrum: power-law decay
        sa = [3.0 / (i + 1) for i in range(rank)]
        sb = [2.0 / (i + 1) for i in range(rank)]
        for overlap_level in [0.0, 0.3, 0.7, 1.0]:
            ct = [overlap_level] * rank
            cp = [overlap_level * 0.8] * rank
            for alpha, beta in coeff_pairs:
                da, db = make_rankr_pair_with_geometry(
                    rank=rank, sigma_a=sa, sigma_b=sb,
                    cos_theta=ct, cos_phi=cp, seed=200 + n_cases,
                )
                observed = frobenius(alpha * da + beta * db)
                predicted = linear_merge_frobenius(
                    sigma_a=tuple(sa), sigma_b=tuple(sb),
                    cos_theta=tuple(ct), cos_phi=tuple(cp),
                    alpha=alpha, beta=beta,
                    directional_agreement=1.0,
                )
                err = abs(observed - predicted.frobenius_norm)
                max_abs_err = max(max_abs_err, err)
                n_cases += 1

    assert n_cases >= 24
    assert max_abs_err < 0.05, f"max_abs_err={max_abs_err} across {n_cases} cases"


def test_equal_spectrum_validation_grid() -> None:
    """Validate equal-rank equal-spectrum formula against dense SVD."""
    ranks = [2, 4]
    max_sigma1_err = 0.0
    n_cases = 0

    for rank in ranks:
        sigma = [4.0 / (i + 1) for i in range(rank)]
        for overlap_level in [0.0, 0.5, 0.9]:
            ct = [overlap_level] * rank
            cp = [overlap_level] * rank
            for alpha, beta in [(0.5, 0.5), (0.7, 0.3)]:
                da, db = make_equal_spectrum_pair(
                    rank=rank, sigma=sigma, cos_theta=ct, cos_phi=cp, seed=300 + n_cases,
                )
                observed_s1 = sigma1(alpha * da + beta * db)
                result = equal_rank_equal_spectrum_merge(
                    sigma=tuple(sigma),
                    cos_theta=tuple(ct),
                    cos_phi=tuple(cp),
                    alpha=alpha,
                    beta=beta,
                )
                err = abs(observed_s1 - result.max_sigma1)
                max_sigma1_err = max(max_sigma1_err, err)
                n_cases += 1

    assert n_cases >= 12
    # The block decomposition is exact when principal angle directions align
    # with spectral directions (our construction ensures this). Tolerance
    # accounts for numerical conditioning at high overlap.
    assert max_sigma1_err < 0.15, f"max_sigma1_err={max_sigma1_err} across {n_cases} cases"


def test_weyl_bounds_validation_grid() -> None:
    """Validate that Weyl bounds contain observed σ_1 across parameter grid."""
    bounds_hold_count = 0
    n_cases = 0

    for rank in [2, 4, 8]:
        sa = [3.0 / (i + 1) for i in range(rank)]
        sb = [2.5 / (i + 1) for i in range(rank)]
        for overlap in [0.0, 0.3, 0.6, 0.9]:
            ct = [overlap] * rank
            cp = [overlap * 0.7] * rank
            for alpha, beta in [(0.5, 0.5), (0.8, 0.2)]:
                da, db = make_rankr_pair_with_geometry(
                    rank=rank, sigma_a=sa, sigma_b=sb,
                    cos_theta=ct, cos_phi=cp, seed=400 + n_cases,
                )
                observed = sigma1(alpha * da + beta * db)
                gram = general_rank_gram_decomposition(
                    sigma_a=tuple(sa), sigma_b=tuple(sb),
                    cos_theta=tuple(ct), cos_phi=tuple(cp),
                    alpha=alpha, beta=beta,
                )
                if gram.sigma1_weyl_lower <= observed + 1e-6 and observed <= gram.sigma1_weyl_upper + 1e-6:
                    bounds_hold_count += 1
                n_cases += 1

    assert n_cases >= 24
    assert bounds_hold_count == n_cases, f"bounds held for {bounds_hold_count}/{n_cases} cases"
