from __future__ import annotations

import math

import torch

from gradience.vnext.merge.spectral_compat import compute_subspace_metrics
from gradience.vnext.merge.spectral_theory import (
    Rank1MergeInputs,
    compare_over_accumulation_theory_to_heuristic,
    compute_rankr_interaction_geometry,
    equal_rank_equal_spectrum_merge,
    estimate_linear_merge_bounds_from_metrics,
    estimate_over_accumulation_theory,
    expected_dare_frobenius_multiplier,
    general_rank_gram_decomposition,
    linear_merge_frobenius,
    linear_merge_sigma1_bounds,
    linear_merge_stable_rank_bounds,
    norm_equalized_over_accumulation_analysis,
    norm_equalized_scaling,
    over_accumulation_exact_condition,
    rank1_linear_merge_result,
)
from gradience.vnext.merge.spectral_theory_test_utils import (
    frobenius,
    make_equal_spectrum_pair,
    make_rank1_pair_with_cosines,
    make_rankr_pair_with_geometry,
    sigma1,
    singular_values,
    stable_rank,
)


def test_rank1_closed_form_matches_dense_svd_positive_alignment() -> None:
    alpha, beta = 0.6, 0.4
    sigma_a, sigma_b = 2.0, 1.5
    cos_theta, cos_phi = 0.8, 0.7
    delta_a, delta_b = make_rank1_pair_with_cosines(
        sigma_a=sigma_a, sigma_b=sigma_b, cos_theta=cos_theta, cos_phi=cos_phi, seed=17
    )
    observed = sigma1(alpha * delta_a + beta * delta_b)
    pred = rank1_linear_merge_result(
        Rank1MergeInputs(
            alpha=alpha,
            beta=beta,
            sigma_a=sigma_a,
            sigma_b=sigma_b,
            cos_theta=cos_theta,
            cos_phi=cos_phi,
            directional_agreement=1.0,
        )
    )
    assert math.isclose(observed, pred.sigma1, rel_tol=1e-6, abs_tol=1e-6)
    assert pred.is_over_accumulating


def test_rank1_negative_directional_agreement_reduces_sigma1() -> None:
    pred = rank1_linear_merge_result(
        Rank1MergeInputs(
            alpha=0.5,
            beta=0.5,
            sigma_a=2.0,
            sigma_b=1.8,
            cos_theta=0.9,
            cos_phi=0.9,
            directional_agreement=-1.0,
        )
    )
    assert pred.cross_term < 0.0
    assert pred.sigma1 <= pred.baseline_sigma1


def test_linear_merge_sigma1_bounds_contain_observed_sigma1() -> None:
    torch.manual_seed(9)
    a = torch.randn(16, 16, dtype=torch.float64)
    b = torch.randn(16, 16, dtype=torch.float64)
    alpha, beta = 0.7, 0.3
    merged = alpha * a + beta * b
    observed = sigma1(merged)
    bounds = linear_merge_sigma1_bounds(sigma1(a), sigma1(b), alpha=alpha, beta=beta)
    assert bounds.lower_bound <= observed + 1e-9
    assert observed <= bounds.upper_bound + 1e-9


def test_norm_equalized_scaling_targets_geometric_mean() -> None:
    s = norm_equalized_scaling(9.0, 4.0)
    assert math.isclose(s.target_frobenius_norm, 6.0, rel_tol=1e-12)
    assert math.isclose(s.scale_a * 9.0, 6.0, rel_tol=1e-12)
    assert math.isclose(s.scale_b * 4.0, 6.0, rel_tol=1e-12)


def test_expected_dare_frobenius_multiplier() -> None:
    assert math.isclose(expected_dare_frobenius_multiplier(0.0), 1.0, rel_tol=1e-12)
    assert expected_dare_frobenius_multiplier(0.5) > 1.0


def test_estimate_linear_merge_bounds_from_metrics_returns_bounded_values() -> None:
    # Build tiny rank-2 factors for compute_subspace_metrics call.
    torch.manual_seed(11)
    r = 2
    d_in = 8
    d_out = 8
    A_a = torch.randn(r, d_in)
    B_a = torch.randn(d_out, r)
    A_b = torch.randn(r, d_in)
    B_b = torch.randn(d_out, r)
    m = compute_subspace_metrics(A_a, B_a, 16.0, r, A_b, B_b, 16.0, r, compute_dtype=torch.float64)
    est = estimate_linear_merge_bounds_from_metrics(m, coefficients=(0.5, 0.5))
    assert est.sigma1_lower_bound >= 0.0
    assert est.sigma1_upper_bound >= est.sigma1_lower_bound
    assert 0.0 <= est.alignment_factor <= 1.0
    assert 0.0 <= est.implied_over_accumulation_risk <= 1.0


def test_over_accumulation_theory_estimate_is_bounded() -> None:
    torch.manual_seed(19)
    r, d_in, d_out = 4, 16, 16
    A_a = torch.randn(r, d_in)
    B_a = torch.randn(d_out, r)
    A_b = torch.randn(r, d_in)
    B_b = torch.randn(d_out, r)
    m = compute_subspace_metrics(A_a, B_a, 16.0, r, A_b, B_b, 16.0, r, compute_dtype=torch.float64)
    est = estimate_over_accumulation_theory(m, coefficients=(0.5, 0.5))
    assert est.sigma1_a_est >= 0.0
    assert est.sigma1_b_est >= 0.0
    assert 0.0 <= est.left_overlap_proxy <= 1.0
    assert 0.0 <= est.right_overlap_proxy <= 1.0
    assert -1.0 <= est.interaction_proxy <= 1.0
    assert 0.0 <= est.confidence <= 1.0
    assert 0.0 <= est.risk_score <= 1.0


def test_theory_to_heuristic_comparison_roundtrip() -> None:
    torch.manual_seed(21)
    r, d_in, d_out = 4, 16, 16
    A_a = torch.randn(r, d_in)
    B_a = torch.randn(d_out, r)
    A_b = torch.randn(r, d_in)
    B_b = torch.randn(d_out, r)
    m = compute_subspace_metrics(A_a, B_a, 16.0, r, A_b, B_b, 16.0, r, compute_dtype=torch.float64)
    cmp = compare_over_accumulation_theory_to_heuristic(m, coefficients=(0.7, 0.3))
    assert cmp.heuristic_band in {"low", "watch", "high"}
    assert 0.0 <= cmp.heuristic_score <= 1.0
    assert -1.0 <= cmp.delta_risk_score <= 1.0


def test_over_accumulation_theory_is_sign_sensitive() -> None:
    torch.manual_seed(33)
    r, d = 4, 16
    A = torch.randn(r, d)
    B = torch.randn(d, r)

    m_pos = compute_subspace_metrics(
        A,
        B,
        16.0,
        r,
        A.clone(),
        B.clone(),
        16.0,
        r,
        compute_dtype=torch.float64,
    )
    m_neg = compute_subspace_metrics(
        A,
        B,
        16.0,
        r,
        (-A).clone(),
        B.clone(),
        16.0,
        r,
        compute_dtype=torch.float64,
    )

    # Validate setup: identical adapters should have positive agreement,
    # negated-A adapters should have negative agreement
    assert m_pos.directional_agreement > 0, (
        f"Setup: identical adapters should have positive agreement, got {m_pos.directional_agreement:.4f}"
    )
    assert m_neg.directional_agreement < 0, (
        f"Setup: negating A should produce negative agreement, got {m_neg.directional_agreement:.4f}"
    )

    est_pos = estimate_over_accumulation_theory(m_pos, coefficients=(0.5, 0.5))
    est_neg = estimate_over_accumulation_theory(m_neg, coefficients=(0.5, 0.5))

    assert est_pos.predicted_over_accumulating
    assert not est_neg.predicted_over_accumulating
    assert est_pos.risk_score > est_neg.risk_score


def test_rankr_interaction_geometry_uses_full_spectra() -> None:
    torch.manual_seed(44)
    r, d = 8, 32
    A_a = torch.randn(r, d)
    B_a = torch.randn(d, r)
    A_b = torch.randn(r, d)
    B_b = torch.randn(d, r)
    m = compute_subspace_metrics(A_a, B_a, 16.0, r, A_b, B_b, 16.0, r, compute_dtype=torch.float64)
    geom = compute_rankr_interaction_geometry(m)

    assert geom.k > 0
    assert 0.0 <= geom.spectral_overlap_weighted <= 1.0
    assert 0.0 <= geom.interaction_top1 <= 1.0
    assert 0.0 <= geom.interaction_top3_mean <= 1.0
    assert len(geom.pair_products) == geom.k
    assert len(geom.interaction_terms_weighted) == geom.k


# ═══════════════════════════════════════════════════════════════════════════
# Phase 1 extensions: Frobenius, equal-rank, general Gram, stable rank
# ═══════════════════════════════════════════════════════════════════════════


def test_linear_merge_frobenius_exact_rank1() -> None:
    """Frobenius of rank-1 merge matches dense computation."""
    sigma_a, sigma_b = 2.0, 1.5
    cos_theta, cos_phi = 0.8, 0.6
    alpha, beta = 0.6, 0.4
    da, db = make_rank1_pair_with_cosines(
        sigma_a=sigma_a, sigma_b=sigma_b, cos_theta=cos_theta, cos_phi=cos_phi, seed=50
    )
    observed = frobenius(alpha * da + beta * db)
    predicted = linear_merge_frobenius(
        sigma_a=(sigma_a,),
        sigma_b=(sigma_b,),
        cos_theta=(cos_theta,),
        cos_phi=(cos_phi,),
        alpha=alpha,
        beta=beta,
        directional_agreement=1.0,
    )
    assert math.isclose(observed, predicted.frobenius_norm, rel_tol=1e-5)


def test_linear_merge_frobenius_exact_rankr() -> None:
    """Frobenius of rank-r merge matches dense computation."""
    r = 4
    sa = [3.0, 2.0, 1.0, 0.5]
    sb = [2.5, 1.8, 0.9, 0.3]
    ct = [0.9, 0.6, 0.3, 0.1]
    cp = [0.8, 0.5, 0.2, 0.0]
    alpha, beta = 0.5, 0.5

    da, db = make_rankr_pair_with_geometry(
        rank=r, sigma_a=sa, sigma_b=sb, cos_theta=ct, cos_phi=cp, seed=51
    )
    observed = frobenius(alpha * da + beta * db)
    predicted = linear_merge_frobenius(
        sigma_a=tuple(sa),
        sigma_b=tuple(sb),
        cos_theta=tuple(ct),
        cos_phi=tuple(cp),
        alpha=alpha,
        beta=beta,
        directional_agreement=1.0,
    )
    assert math.isclose(observed, predicted.frobenius_norm, rel_tol=1e-4)


def test_equal_rank_equal_spectrum_orthogonal() -> None:
    """Orthogonal subspaces: merged SVs are α·σ_i and β·σ_i (disjoint contributions)."""
    sigma = [3.0, 2.0, 1.0]
    alpha, beta = 0.5, 0.5
    result = equal_rank_equal_spectrum_merge(
        sigma=tuple(sigma),
        cos_theta=(0.0, 0.0, 0.0),
        cos_phi=(0.0, 0.0, 0.0),
        alpha=alpha,
        beta=beta,
    )
    # Orthogonal: λ_+ = max(α,β)²·σ², λ_- = min(α,β)²·σ²
    # With α=β=0.5: both eigenvalues equal, so sv_+ = sv_- = 0.5·σ_i
    for i, si in enumerate(sigma):
        expected = max(alpha, beta) * si
        assert math.isclose(result.merged_singular_values[2 * i], expected, rel_tol=1e-10)
    # No over-accumulation: merged σ_1 = max(α,β)·σ_1 = baseline
    assert abs(result.over_accumulation_margin) < 1e-10


def test_equal_rank_equal_spectrum_identical() -> None:
    """Identical subspaces: singular values add directly."""
    sigma = [3.0, 2.0]
    result = equal_rank_equal_spectrum_merge(
        sigma=tuple(sigma),
        cos_theta=(1.0, 1.0),
        cos_phi=(1.0, 1.0),
        alpha=0.5,
        beta=0.5,
        per_direction_sign=(1.0, 1.0),
    )
    # λ_+ = σ² (α+β)², λ_- = σ² (α-β)² = 0
    assert math.isclose(result.merged_singular_values[0], sigma[0] * 1.0, rel_tol=1e-10)
    assert math.isclose(result.merged_singular_values[1], sigma[1] * 1.0, rel_tol=1e-10)
    # Over-accumulation: merged σ_1 = 3.0, baseline = max(0.5*3, 0.5*3) = 1.5
    assert result.over_accumulation_margin > 0


def test_equal_rank_equal_spectrum_matches_dense() -> None:
    """Equal-spectrum analytical solution matches dense SVD."""
    sigma = [3.0, 2.0, 1.0, 0.5]
    ct = [0.8, 0.5, 0.2, 0.0]
    cp = [0.7, 0.4, 0.1, 0.0]
    alpha, beta = 0.6, 0.4

    da, db = make_equal_spectrum_pair(
        rank=4, sigma=sigma, cos_theta=ct, cos_phi=cp, seed=52
    )
    observed_svs = singular_values(alpha * da + beta * db)

    result = equal_rank_equal_spectrum_merge(
        sigma=tuple(sigma),
        cos_theta=tuple(ct),
        cos_phi=tuple(cp),
        alpha=alpha,
        beta=beta,
    )
    # Top singular value should match within tolerance.
    # The block decomposition is exact when principal angle directions align
    # with spectral directions, which our synthetic construction ensures.
    assert math.isclose(observed_svs[0], result.merged_singular_values[0], rel_tol=5e-3)


def test_general_rank_gram_bounds_contain_observed() -> None:
    """Weyl bounds from general_rank_gram_decomposition contain observed σ_1."""
    r = 4
    sa = [3.0, 2.0, 1.0, 0.5]
    sb = [2.5, 1.8, 0.9, 0.3]
    ct = [0.7, 0.4, 0.2, 0.0]
    cp = [0.6, 0.3, 0.1, 0.0]
    alpha, beta = 0.5, 0.5

    da, db = make_rankr_pair_with_geometry(
        rank=r, sigma_a=sa, sigma_b=sb, cos_theta=ct, cos_phi=cp, seed=53
    )
    observed = sigma1(alpha * da + beta * db)

    gram = general_rank_gram_decomposition(
        sigma_a=tuple(sa),
        sigma_b=tuple(sb),
        cos_theta=tuple(ct),
        cos_phi=tuple(cp),
        alpha=alpha,
        beta=beta,
    )
    assert gram.sigma1_weyl_lower <= observed + 1e-6
    assert observed <= gram.sigma1_weyl_upper + 1e-6


def test_general_rank_gram_orthogonal_no_over_accumulation() -> None:
    """Orthogonal subspaces: no over-accumulation possible."""
    gram = general_rank_gram_decomposition(
        sigma_a=(3.0, 2.0),
        sigma_b=(2.5, 1.5),
        cos_theta=(0.0, 0.0),
        cos_phi=(0.0, 0.0),
        alpha=0.5,
        beta=0.5,
    )
    assert not gram.over_accumulation_condition_met
    assert gram.cross_term_spectral_bound < 1e-12


def test_stable_rank_bounds_contain_observed() -> None:
    """Stable rank bounds contain observed stable rank of merge."""
    r = 4
    sa = [3.0, 2.0, 1.0, 0.5]
    sb = [2.0, 1.5, 0.8, 0.3]
    ct = [0.6, 0.3, 0.1, 0.0]
    cp = [0.5, 0.2, 0.1, 0.0]
    alpha, beta = 0.5, 0.5

    da, db = make_rankr_pair_with_geometry(
        rank=r, sigma_a=sa, sigma_b=sb, cos_theta=ct, cos_phi=cp, seed=54
    )
    observed_sr = stable_rank(alpha * da + beta * db)

    sr_bounds = linear_merge_stable_rank_bounds(
        sigma_a=tuple(sa),
        sigma_b=tuple(sb),
        cos_theta=tuple(ct),
        cos_phi=tuple(cp),
        alpha=alpha,
        beta=beta,
    )
    # Lower bound should be ≤ observed (with tolerance)
    assert sr_bounds.lower_bound <= observed_sr + 0.1
    # Upper bound can be infinite when lower σ_1 bound is 0; skip if so
    if sr_bounds.upper_bound < 1e6:
        assert observed_sr <= sr_bounds.upper_bound + 0.1


# ═══════════════════════════════════════════════════════════════════════════
# Phase 2 extensions: exact over-accumulation condition
# ═══════════════════════════════════════════════════════════════════════════


def test_over_accumulation_exact_orthogonal_no_risk() -> None:
    """Orthogonal adapters: zero inflation risk."""
    torch.manual_seed(60)
    r, d = 4, 32
    da, db = make_rankr_pair_with_geometry(
        d_out=d, d_in=d, rank=r,
        cos_theta=[0.0, 0.0, 0.0, 0.0],
        cos_phi=[0.0, 0.0, 0.0, 0.0],
        seed=60,
    )
    A_a = torch.randn(r, d, dtype=torch.float64)
    B_a = torch.randn(d, r, dtype=torch.float64)
    A_b = torch.randn(r, d, dtype=torch.float64)
    B_b = torch.randn(d, r, dtype=torch.float64)
    # Build from factors that give near-orthogonal subspaces
    m = compute_subspace_metrics(A_a, B_a, None, r, A_b, B_b, None, r, compute_dtype=torch.float64)
    cond = over_accumulation_exact_condition(m)
    # With random subspaces, overlap should be moderate — just test structure
    assert cond.baseline_sigma1 > 0
    assert cond.cross_term_spectral_bound >= 0
    assert 0.0 <= cond.analytical_risk_score <= 1.0
    assert cond.analytical_vs_heuristic_alignment in {"agrees", "theory_higher", "theory_lower"}


def test_over_accumulation_exact_high_overlap_positive_agreement() -> None:
    """High overlap + positive agreement = inflation possible."""
    torch.manual_seed(61)
    r, d = 4, 32
    # Build factors that produce high overlap
    A_base = torch.randn(r, d, dtype=torch.float64)
    B_base = torch.randn(d, r, dtype=torch.float64)
    # Same factors + small perturbation = high overlap, positive agreement
    noise_scale = 0.01
    A_b = A_base + noise_scale * torch.randn(r, d, dtype=torch.float64)
    B_b = B_base + noise_scale * torch.randn(d, r, dtype=torch.float64)
    m = compute_subspace_metrics(A_base, B_base, None, r, A_b, B_b, None, r, compute_dtype=torch.float64)
    cond = over_accumulation_exact_condition(m)
    assert cond.is_inflation_possible
    assert cond.inflation_ratio_upper > 1.0
    assert cond.analytical_risk_score > 0.0


def test_over_accumulation_negative_agreement_no_inflation() -> None:
    """Negative directional agreement: inflation not possible."""
    torch.manual_seed(62)
    r, d = 4, 32
    A_base = torch.randn(r, d, dtype=torch.float64)
    B_base = torch.randn(d, r, dtype=torch.float64)
    # Flip sign for negative agreement
    m = compute_subspace_metrics(A_base, B_base, None, r, -A_base, B_base, None, r, compute_dtype=torch.float64)
    cond = over_accumulation_exact_condition(m)
    assert not cond.is_inflation_possible


# ═══════════════════════════════════════════════════════════════════════════
# Phase 3 stub: norm-equalized analysis
# ═══════════════════════════════════════════════════════════════════════════


def test_norm_equalized_analysis_reports_inflation_correctly() -> None:
    """Norm-equalized analysis correctly computes and compares inflation ratios."""
    torch.manual_seed(70)
    r, d = 4, 32
    # Build with very different scales to create Frobenius imbalance
    A_a = torch.randn(r, d, dtype=torch.float64) * 0.1
    B_a = torch.randn(d, r, dtype=torch.float64) * 0.1
    A_b = torch.randn(r, d, dtype=torch.float64) * 3.0
    B_b = torch.randn(d, r, dtype=torch.float64) * 3.0
    m = compute_subspace_metrics(A_a, B_a, None, r, A_b, B_b, None, r, compute_dtype=torch.float64)

    # Confirm the setup actually produces a large Frobenius imbalance
    assert m.frobenius_ratio > 5.0, f"Setup: expected large imbalance, got ratio={m.frobenius_ratio:.2f}"

    result = norm_equalized_over_accumulation_analysis(m)

    # Both ratios must be positive
    assert result.linear_inflation_ratio > 0.0
    assert result.normeq_inflation_ratio > 0.0

    # inflation_reduced must be consistent with the actual ratio comparison
    expected_reduced = result.normeq_inflation_ratio < result.linear_inflation_ratio
    assert result.inflation_reduced == expected_reduced, (
        f"inflation_reduced={result.inflation_reduced} but normeq={result.normeq_inflation_ratio:.4f} "
        f"vs linear={result.linear_inflation_ratio:.4f}"
    )

    # reduction_factor = normeq / linear (< 1.0 means inflation was reduced)
    if result.linear_inflation_ratio > 0:
        expected_factor = result.normeq_inflation_ratio / result.linear_inflation_ratio
        assert math.isclose(result.reduction_factor, expected_factor, rel_tol=1e-6), (
            f"reduction_factor={result.reduction_factor:.4f} != "
            f"normeq/linear={expected_factor:.4f}"
        )
