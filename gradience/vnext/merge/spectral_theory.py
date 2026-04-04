"""
Analytical spectral geometry helpers for merge operations.

This module is intentionally lightweight and CPU-only. It provides closed-form
or provable-bounds utilities that are expressed in terms of quantities already
computed by merge audit (`SubspaceMetrics`) so the analytical line can be
validated numerically without training compute.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from gradience.vnext.merge.spectral_compat import SubspaceMetrics


@dataclass(frozen=True)
class Rank1MergeInputs:
    """Inputs for the exact rank-1 linear-merge spectrum formula."""

    alpha: float
    beta: float
    sigma_a: float
    sigma_b: float
    cos_theta: float
    cos_phi: float
    directional_agreement: float


@dataclass(frozen=True)
class Rank1MergeResult:
    """Closed-form rank-1 linear-merge readout."""

    sigma1_squared: float
    sigma1: float
    baseline_sigma1: float
    over_accumulation_margin: float
    is_over_accumulating: bool
    cross_term: float


@dataclass(frozen=True)
class LinearMergeSigma1Bounds:
    """Provable sigma_1 bounds for A+B from Weyl/triangle inequalities."""

    lower_bound: float
    upper_bound: float


@dataclass(frozen=True)
class NormEqualizedScaling:
    """Scaling factors used by norm-equalized merge."""

    scale_a: float
    scale_b: float
    target_frobenius_norm: float


@dataclass(frozen=True)
class SubspaceBoundEstimate:
    """
    Coarse bound estimate from `SubspaceMetrics`.

    Note: these are conservative and intentionally expressed only via fields
    already available in `SubspaceMetrics` (no hidden extra geometry).
    """

    sigma1_lower_bound: float
    sigma1_upper_bound: float
    alignment_factor: float
    implied_over_accumulation_risk: float


@dataclass(frozen=True)
class OverAccumulationTheoryEstimate:
    """
    Phase-2 analytical surrogate for rank-r over-accumulation.

    This estimate maps rank-r geometry to a rank-1 equivalent surrogate using
    quantities already present in `SubspaceMetrics`.
    """

    sigma1_a_est: float
    sigma1_b_est: float
    left_overlap_proxy: float
    right_overlap_proxy: float
    directional_agreement: float
    interaction_proxy: float
    predicted_sigma1: float
    baseline_sigma1: float
    predicted_margin: float
    predicted_over_accumulating: bool
    confidence: float
    risk_score: float


@dataclass(frozen=True)
class OverAccumulationTheoryHeuristicComparison:
    """Side-by-side comparison between analytical surrogate and heuristic."""

    theory: OverAccumulationTheoryEstimate
    heuristic_score: float
    heuristic_band: str
    heuristic_alignment: float
    heuristic_concentration: float
    heuristic_coefficient_exposure: float
    delta_risk_score: float


@dataclass(frozen=True)
class RankRInteractionGeometry:
    """
    Rank-r interaction geometry derived from full principal-angle spectra.

    This is an additive analytical helper for OA-v2 style diagnostics.
    """

    k: int
    degraded_input: bool
    spectral_overlap_weighted: float
    interaction_primary_raw: float
    interaction_top1: float
    interaction_top3_mean: float
    pair_products: tuple[float, ...]
    interaction_terms_weighted: tuple[float, ...]


def _clip01(x: float) -> float:
    return min(1.0, max(0.0, float(x)))


def _normalize_coefficients(coefficients: tuple[float, float]) -> tuple[float, float]:
    a = max(0.0, float(coefficients[0]))
    b = max(0.0, float(coefficients[1]))
    z = a + b
    if z <= 1e-12:
        return (0.5, 0.5)
    return (a / z, b / z)


def compute_rankr_interaction_geometry(
    metrics: SubspaceMetrics,
    *,
    eps: float = 1e-12,
) -> RankRInteractionGeometry:
    """
    Compute full-spectrum rank-r interaction geometry.

    Core definition:
      w_i = s_a_i * s_b_i
      w_hat_i = w_i / (sum_j w_j + eps)
      pair_i = clamp(c_left_i,0,1) * clamp(c_right_i,0,1)
      spectral_overlap_weighted = sum_i w_hat_i * pair_i
    """
    c_left = [max(0.0, min(1.0, float(c))) for c in (metrics.principal_angle_cosines or ())]
    c_right = [max(0.0, min(1.0, float(c))) for c in (metrics.right_principal_angle_cosines or ())]
    s_a = [max(0.0, float(s)) for s in (metrics.effective_singular_values_a or ())]
    s_b = [max(0.0, float(s)) for s in (metrics.effective_singular_values_b or ())]

    k = min(len(c_left), len(c_right), len(s_a), len(s_b))
    degraded = k == 0
    if k <= 0:
        # Backward-compatible degraded mode for old artifacts that do not carry
        # right-angle/singular spectra payloads.
        fallback_pair = _clip01(float(metrics.mean_overlap)) * _clip01(float(metrics.max_overlap))
        return RankRInteractionGeometry(
            k=0,
            degraded_input=True,
            spectral_overlap_weighted=fallback_pair,
            interaction_primary_raw=fallback_pair,
            interaction_top1=fallback_pair,
            interaction_top3_mean=fallback_pair,
            pair_products=(fallback_pair,),
            interaction_terms_weighted=(fallback_pair,),
        )

    pair_products = [c_left[i] * c_right[i] for i in range(k)]
    weights = [s_a[i] * s_b[i] for i in range(k)]
    weight_sum = sum(weights)
    if weight_sum <= eps:
        w_hat = [1.0 / k] * k
        degraded = True
    else:
        w_hat = [w / weight_sum for w in weights]

    interaction_terms = [w_hat[i] * pair_products[i] for i in range(k)]
    spectral_overlap_weighted = sum(interaction_terms)

    top_terms = sorted(interaction_terms, reverse=True)
    top3 = top_terms[: min(3, len(top_terms))]
    interaction_top3 = sum(top3) / len(top3) if top3 else 0.0

    return RankRInteractionGeometry(
        k=k,
        degraded_input=degraded,
        spectral_overlap_weighted=_clip01(spectral_overlap_weighted),
        interaction_primary_raw=_clip01(spectral_overlap_weighted),
        interaction_top1=_clip01(max(top_terms) if top_terms else 0.0),
        interaction_top3_mean=_clip01(interaction_top3),
        pair_products=tuple(pair_products),
        interaction_terms_weighted=tuple(interaction_terms),
    )


def rank1_linear_merge_result(inputs: Rank1MergeInputs) -> Rank1MergeResult:
    """
    Exact rank-1 linear-merge leading singular value.

    For rank-1 factors:
      ΔW_a = σ_a u_a v_a^T
      ΔW_b = σ_b u_b v_b^T
      ΔW_m = αΔW_a + βΔW_b

    with principal-angle cosines:
      cos(theta) = <u_a, u_b>, cos(phi) = <v_a, v_b>

    and directional sign proxy from directional agreement:
      sign = +1 if directional_agreement >= 0 else -1

    Let:
      a = |α|σ_a, b = |β|σ_b, z = sign * cos(theta) * cos(phi)

    The non-zero eigenvalues of `(ΔW_m)(ΔW_m)^T` are roots of:
      λ² - Tλ + D = 0

    with:
      T = a² + b² + 2abz
      D = a²b²(1-cos²(theta))(1-cos²(phi))

    and:
      σ_1(ΔW_m) = sqrt((T + sqrt(T² - 4D))/2)
    """
    alpha = abs(float(inputs.alpha))
    beta = abs(float(inputs.beta))
    sigma_a = max(0.0, float(inputs.sigma_a))
    sigma_b = max(0.0, float(inputs.sigma_b))
    cos_theta = _clip01(inputs.cos_theta)
    cos_phi = _clip01(inputs.cos_phi)
    sign = 1.0 if float(inputs.directional_agreement) >= 0.0 else -1.0

    a = alpha * sigma_a
    b = beta * sigma_b
    z = sign * cos_theta * cos_phi
    trace_term = a * a + b * b + 2.0 * a * b * z
    det_term = (a * a) * (b * b) * (1.0 - cos_theta * cos_theta) * (1.0 - cos_phi * cos_phi)
    disc = max(0.0, trace_term * trace_term - 4.0 * det_term)
    lambda_max = 0.5 * (trace_term + math.sqrt(disc))
    sigma1_sq = max(0.0, lambda_max)
    sigma1 = math.sqrt(sigma1_sq)
    baseline = max(a, b)
    margin = sigma1 - baseline
    return Rank1MergeResult(
        sigma1_squared=sigma1_sq,
        sigma1=sigma1,
        baseline_sigma1=baseline,
        over_accumulation_margin=margin,
        is_over_accumulating=(margin > 0.0),
        cross_term=2.0 * a * b * z,
    )


def linear_merge_sigma1_bounds(
    sigma1_a: float,
    sigma1_b: float,
    *,
    alpha: float = 0.5,
    beta: float = 0.5,
) -> LinearMergeSigma1Bounds:
    """
    Bounds for sigma_1(αA + βB) using sigma_1(A), sigma_1(B).

    Upper bound (triangle inequality):
      σ_1(αA + βB) <= |α|σ_1(A) + |β|σ_1(B)

    Lower bound (reverse triangle):
      σ_1(αA + βB) >= max(0, ||α|σ_1(A) - |β|σ_1(B)|)
    """
    s1a = max(0.0, float(sigma1_a))
    s1b = max(0.0, float(sigma1_b))
    a = abs(float(alpha))
    b = abs(float(beta))
    upper = a * s1a + b * s1b
    lower = max(0.0, abs(a * s1a - b * s1b))
    return LinearMergeSigma1Bounds(lower_bound=lower, upper_bound=upper)


def norm_equalized_scaling(frob_a: float, frob_b: float, eps: float = 1e-12) -> NormEqualizedScaling:
    """
    Compute norm-equalization factors to geometric-mean Frobenius norm.
    """
    fa = max(float(frob_a), eps)
    fb = max(float(frob_b), eps)
    g = math.sqrt(fa * fb)
    return NormEqualizedScaling(
        scale_a=g / fa,
        scale_b=g / fb,
        target_frobenius_norm=g,
    )


def expected_dare_frobenius_multiplier(drop_fraction: float, eps: float = 1e-12) -> float:
    """
    Expected multiplicative factor for Frobenius norm under DARE dropout.

    With Bernoulli dropout `p` and rescale `1/(1-p)`, expectation of squared
    norm is inflated by `1/(1-p)`. So expected norm multiplier is:
      sqrt(1/(1-p))
    """
    p = float(drop_fraction)
    if p < 0.0 or p >= 1.0:
        raise ValueError(f"drop_fraction must be in [0,1), got {p}")
    return math.sqrt(1.0 / max(1.0 - p, eps))


def estimate_linear_merge_bounds_from_metrics(
    metrics: SubspaceMetrics,
    *,
    coefficients: tuple[float, float] = (0.5, 0.5),
) -> SubspaceBoundEstimate:
    """
    Conservative sigma_1 bounds and alignment readout using only SubspaceMetrics.

    Since SubspaceMetrics does not carry sigma_1 directly, this uses
    sigma_1 <= ||ΔW||_F and obtains coarse bounds in Frobenius terms.
    """
    alpha, beta = coefficients
    fa = max(0.0, float(metrics.frobenius_norm_a))
    fb = max(0.0, float(metrics.frobenius_norm_b))
    sigma1_bounds = linear_merge_sigma1_bounds(fa, fb, alpha=alpha, beta=beta)

    # Alignment proxy mirrors the analytical rank-1 cross-term structure:
    # positive only when directional agreement is positive.
    align = max(0.0, float(metrics.directional_agreement)) * max(0.0, float(metrics.mean_overlap))
    # simple bounded risk score from available observables.
    # high alignment + low stable-rank (more concentration) => higher risk.
    concentration_a = 1.0 / max(float(metrics.stable_rank_a), 1.0)
    concentration_b = 1.0 / max(float(metrics.stable_rank_b), 1.0)
    concentration = 0.5 * (concentration_a + concentration_b)
    risk = _clip01(align * concentration * (1.0 + min(float(metrics.frobenius_ratio), 10.0) / 10.0))

    return SubspaceBoundEstimate(
        sigma1_lower_bound=sigma1_bounds.lower_bound,
        sigma1_upper_bound=sigma1_bounds.upper_bound,
        alignment_factor=align,
        implied_over_accumulation_risk=risk,
    )


def estimate_over_accumulation_theory(
    metrics: SubspaceMetrics,
    *,
    coefficients: tuple[float, float] = (0.5, 0.5),
) -> OverAccumulationTheoryEstimate:
    """
    Phase-2 rank-r analytical surrogate in SubspaceMetrics coordinates.

    Construction:
    1. Estimate sigma_1 for each adapter from stable-rank identity:
         sigma1 ≈ ||ΔW||_F / sqrt(stable_rank)
    2. Build a rank-1 equivalent geometry surrogate with:
         left overlap proxy  = mean_overlap
         right overlap proxy = mean_overlap * |directional_agreement|
    3. Use exact rank-1 closed form on these proxies to get predicted
       merged sigma_1 and inflation margin.
    4. Produce a bounded risk score combining margin, alignment, and confidence.
    """
    coeff_a, coeff_b = _normalize_coefficients(coefficients)
    # sigma1 estimates from stable-rank definition
    sigma1_a = max(0.0, float(metrics.frobenius_norm_a)) / math.sqrt(max(float(metrics.stable_rank_a), 1.0))
    sigma1_b = max(0.0, float(metrics.frobenius_norm_b)) / math.sqrt(max(float(metrics.stable_rank_b), 1.0))

    left = _clip01(metrics.mean_overlap)
    agreement = max(-1.0, min(1.0, float(metrics.directional_agreement)))
    right = _clip01(left * abs(agreement))
    interaction = left * right * (1.0 if agreement >= 0.0 else -1.0)

    rank1 = rank1_linear_merge_result(
        Rank1MergeInputs(
            alpha=coeff_a,
            beta=coeff_b,
            sigma_a=sigma1_a,
            sigma_b=sigma1_b,
            cos_theta=left,
            cos_phi=right,
            directional_agreement=agreement,
        )
    )

    # Confidence increases with overlap support and metric richness.
    k = max(1, min(metrics.effective_rank_a, metrics.effective_rank_b))
    richness = min(1.0, k / 8.0)
    confidence = _clip01(0.6 * left + 0.4 * richness)

    scale = max(rank1.baseline_sigma1, 1e-12)
    normalized_margin = rank1.over_accumulation_margin / scale
    # Positive margin + positive interaction drive risk.
    risk = _clip01(max(0.0, normalized_margin) * (0.5 + 0.5 * max(0.0, interaction)) * confidence)

    return OverAccumulationTheoryEstimate(
        sigma1_a_est=sigma1_a,
        sigma1_b_est=sigma1_b,
        left_overlap_proxy=left,
        right_overlap_proxy=right,
        directional_agreement=agreement,
        interaction_proxy=interaction,
        predicted_sigma1=rank1.sigma1,
        baseline_sigma1=rank1.baseline_sigma1,
        predicted_margin=rank1.over_accumulation_margin,
        predicted_over_accumulating=rank1.is_over_accumulating and interaction > 0.0,
        confidence=confidence,
        risk_score=risk,
    )


def compare_over_accumulation_theory_to_heuristic(
    metrics: SubspaceMetrics,
    *,
    coefficients: tuple[float, float] = (0.5, 0.5),
) -> OverAccumulationTheoryHeuristicComparison:
    """
    Compare analytical phase-2 surrogate against current heuristic diagnostic.
    """
    from gradience.vnext.merge.over_accumulation import estimate_over_accumulation

    theory = estimate_over_accumulation_theory(metrics, coefficients=coefficients)
    heuristic = estimate_over_accumulation(metrics, assumed_coefficients=coefficients)
    return OverAccumulationTheoryHeuristicComparison(
        theory=theory,
        heuristic_score=float(heuristic.score),
        heuristic_band=str(heuristic.band),
        heuristic_alignment=float(heuristic.factors.alignment),
        heuristic_concentration=float(heuristic.factors.concentration),
        heuristic_coefficient_exposure=float(heuristic.factors.coefficient_exposure),
        delta_risk_score=float(theory.risk_score - heuristic.score),
    )


# ═══════════════════════════════════════════════════════════════════════════
# Phase 1 extensions: equal-rank spectrum, Frobenius of merge, stable rank
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class LinearMergeFrobeniusResult:
    """Frobenius norm of linear merge from SVD decompositions and principal angles.

    Exact result (not a bound):
      ||αΔW_a + βΔW_b||²_F = α²||ΔW_a||²_F + β²||ΔW_b||²_F + 2αβ·cross_frob

    where cross_frob = tr(ΔW_a^T ΔW_b) = Σ_i Σ_j s_{a,i} s_{b,j} (U_a^T U_b)_{ij} (V_a^T V_b)_{ij}
    """

    frobenius_squared: float
    frobenius_norm: float
    cross_frobenius_term: float
    self_a_term: float  # α²||ΔW_a||²_F
    self_b_term: float  # β²||ΔW_b||²_F


@dataclass(frozen=True)
class LinearMergeStableRankBounds:
    """Bounds on stable rank of linearly merged adapter.

    Lower bound: ||ΔW_m||²_F / σ_1²(ΔW_m)_upper²
    Upper bound: ||ΔW_m||²_F / σ_1²(ΔW_m)_lower²
    """

    lower_bound: float
    upper_bound: float
    frobenius_squared: float
    sigma1_lower: float
    sigma1_upper: float


@dataclass(frozen=True)
class EqualRankEqualSpectrumResult:
    """Eigenvalues of the Gram matrix for equal-rank, equal-spectrum merge.

    When r_a = r_b = r and Σ_a = Σ_b = Σ, the Gram matrix of
    ΔW_m = αΔW_a + βΔW_b decomposes into 2×2 blocks per spectral direction.

    For each spectral direction i, the 2×2 block eigenvalues are:
      λ_{i,±} = σ_i² · (α² + β² ± 2αβ · cos(θ_i) · cos(φ_i) · sign_i)

    where sign_i captures per-direction sign agreement.
    """

    block_eigenvalue_pairs: tuple[tuple[float, float], ...]  # (λ_plus, λ_minus) per direction
    merged_singular_values: tuple[float, ...]  # sorted descending
    rank: int
    merged_stable_rank: float
    merged_frobenius_squared: float
    max_sigma1: float
    over_accumulation_margin: float


@dataclass(frozen=True)
class GeneralRankGramDecomposition:
    """Gram matrix decomposition for general rank linear merge.

    ΔW_m^T ΔW_m = α² V_a Σ_a² V_a^T + β² V_b Σ_b² V_b^T
                 + αβ(V_a Σ_a C_left^T Σ_b V_b^T + V_b Σ_b C_left Σ_a V_a^T)

    where C_left = U_a^T U_b has singular values cos(θ_i).

    The cross-term magnitude is bounded by:
      |cross_term_ij| ≤ s_{a,i} · s_{b,j} · cos(θ_min(i,j))
    """

    alpha: float
    beta: float
    frobenius_squared_exact: float
    sigma1_weyl_upper: float
    sigma1_weyl_lower: float
    cross_term_frobenius_bound: float  # ||cross-term||_F upper bound
    cross_term_spectral_bound: float  # σ_1(cross-term) upper bound
    over_accumulation_condition_met: bool


@dataclass(frozen=True)
class OverAccumulationExactCondition:
    """Exact over-accumulation analysis for rank-r linear merge.

    Over-accumulation occurs when:
      σ_1(αΔW_a + βΔW_b) > max(α·σ_1(ΔW_a), β·σ_1(ΔW_b))

    From Weyl's inequality, a sufficient condition for this NOT to happen is:
      σ_1 of the cross-term ≤ baseline - max(α·s_{a,1}, β·s_{b,1})

    The cross-term spectral norm is bounded by:
      σ_1(cross) ≤ αβ · s_{a,1} · s_{b,1} · (cos(θ_1) · cos(φ_1) + ...)

    For the rank-1 case this is exact.
    """

    baseline_sigma1: float  # max(α·s_{a,1}, β·s_{b,1})
    cross_term_spectral_bound: float  # upper bound on σ_1 of cross-term
    sigma1_upper_bound: float  # baseline + cross_term_spectral_bound
    sigma1_lower_bound: float  # max(0, baseline - cross_term_spectral_bound)
    inflation_ratio_upper: float  # sigma1_upper_bound / baseline_sigma1
    is_inflation_possible: bool  # cross_term_spectral_bound > 0
    inflation_driven_by: str  # which factor dominates: "alignment", "scale", or "both"
    # Comparison to heuristic
    analytical_risk_score: float  # bounded [0,1] analytical risk
    analytical_vs_heuristic_alignment: str  # "agrees", "theory_higher", "theory_lower"


def linear_merge_frobenius(
    *,
    sigma_a: tuple[float, ...],
    sigma_b: tuple[float, ...],
    cos_theta: tuple[float, ...],
    cos_phi: tuple[float, ...],
    alpha: float = 0.5,
    beta: float = 0.5,
    directional_agreement: float = 1.0,
) -> LinearMergeFrobeniusResult:
    """Compute the exact Frobenius norm of a linear merge.

    Uses: ||αA + βB||²_F = α²||A||²_F + β²||B||²_F + 2αβ·tr(A^T B)

    The cross trace tr(A^T B) = Σ_i Σ_j s_{a,i} s_{b,j} (U_a^T U_b)_{ij} (V_b^T V_a)_{ij}.

    For the diagonal-dominant approximation (principal angles aligned with
    spectral directions), this simplifies to:
      tr(A^T B) ≈ sign(δ) · Σ_i s_{a,i} s_{b,i} cos(θ_i) cos(φ_i)
    """
    sign = 1.0 if directional_agreement >= 0.0 else -1.0
    frob_sq_a = sum(s * s for s in sigma_a)
    frob_sq_b = sum(s * s for s in sigma_b)

    k = min(len(sigma_a), len(sigma_b), len(cos_theta), len(cos_phi))
    cross = 0.0
    for i in range(k):
        cross += sigma_a[i] * sigma_b[i] * _clip01(cos_theta[i]) * _clip01(cos_phi[i])
    cross *= sign

    self_a = alpha * alpha * frob_sq_a
    self_b = beta * beta * frob_sq_b
    cross_total = 2.0 * alpha * beta * cross
    frob_sq = self_a + self_b + cross_total

    return LinearMergeFrobeniusResult(
        frobenius_squared=max(0.0, frob_sq),
        frobenius_norm=math.sqrt(max(0.0, frob_sq)),
        cross_frobenius_term=cross_total,
        self_a_term=self_a,
        self_b_term=self_b,
    )


def equal_rank_equal_spectrum_merge(
    *,
    sigma: tuple[float, ...],
    cos_theta: tuple[float, ...],
    cos_phi: tuple[float, ...],
    alpha: float = 0.5,
    beta: float = 0.5,
    per_direction_sign: tuple[float, ...] | None = None,
) -> EqualRankEqualSpectrumResult:
    """Exact merged singular values for equal-rank, equal-spectrum case.

    When Σ_a = Σ_b = diag(σ_1,...,σ_r) and the principal angles are
    (θ_1,...,θ_r) (left) and (φ_1,...,φ_r) (right), the merged Gram matrix
    decomposes into r independent 2×2 blocks (when principal angle decomposition
    aligns with spectral decomposition).

    For direction i, with a_i = α·σ_i, b_i = β·σ_i:
      T_i = a_i² + b_i² + 2·a_i·b_i·cos(θ_i)·cos(φ_i)·sign_i
      D_i = a_i²·b_i²·sin²(θ_i)·sin²(φ_i)
      λ_{i,±} = (T_i ± √(T_i² - 4D_i)) / 2

    The merged singular values are √(λ_{i,±}), sorted descending.

    Special cases:
      - Orthogonal (cos=0): λ+ = max(α,β)²σ², λ- = min(α,β)²σ²
      - Identical (cos=1): λ+ = (α+β)²σ², λ- = 0
    """
    r = len(sigma)
    if per_direction_sign is None:
        per_direction_sign = tuple(1.0 for _ in range(r))

    block_pairs: list[tuple[float, float]] = []
    all_sv: list[float] = []

    for i in range(r):
        si = max(0.0, float(sigma[i]))
        ct = _clip01(cos_theta[i]) if i < len(cos_theta) else 0.0
        cp = _clip01(cos_phi[i]) if i < len(cos_phi) else 0.0
        sign_i = 1.0 if per_direction_sign[i] >= 0 else -1.0

        a = alpha * si
        b = beta * si

        # Gram block trace and determinant
        trace = a * a + b * b + 2.0 * a * b * ct * cp * sign_i
        st = math.sqrt(max(0.0, 1.0 - ct * ct))  # sin(θ)
        sp = math.sqrt(max(0.0, 1.0 - cp * cp))  # sin(φ)
        det = a * a * b * b * st * st * sp * sp

        disc = max(0.0, trace * trace - 4.0 * det)
        sqrt_disc = math.sqrt(disc)

        lam_plus = max(0.0, 0.5 * (trace + sqrt_disc))
        lam_minus = max(0.0, 0.5 * (trace - sqrt_disc))

        block_pairs.append((lam_plus, lam_minus))
        all_sv.append(math.sqrt(lam_plus))
        all_sv.append(math.sqrt(lam_minus))

    all_sv.sort(reverse=True)
    frob_sq = sum(s * s for s in all_sv)
    max_sv = all_sv[0] if all_sv else 0.0

    baseline = max(alpha, beta) * (max(0.0, float(sigma[0])) if sigma else 0.0)
    margin = max_sv - baseline

    sr = frob_sq / (max_sv * max_sv) if max_sv > 1e-12 else 1.0

    return EqualRankEqualSpectrumResult(
        block_eigenvalue_pairs=tuple(block_pairs),
        merged_singular_values=tuple(all_sv),
        rank=r,
        merged_stable_rank=sr,
        merged_frobenius_squared=frob_sq,
        max_sigma1=max_sv,
        over_accumulation_margin=margin,
    )


def general_rank_gram_decomposition(
    *,
    sigma_a: tuple[float, ...],
    sigma_b: tuple[float, ...],
    cos_theta: tuple[float, ...],
    cos_phi: tuple[float, ...],
    alpha: float = 0.5,
    beta: float = 0.5,
    directional_agreement: float = 1.0,
) -> GeneralRankGramDecomposition:
    """Analyze the Gram matrix of a general-rank linear merge.

    The Gram matrix ΔW_m^T ΔW_m has cross-terms involving U_a^T U_b whose
    singular values are the principal angle cosines cos(θ_i).

    Cross-term spectral norm bound:
      σ_1(cross) ≤ 2αβ · Σ_i s_{a,i} · s_{b,i} · cos(θ_i) · cos(φ_i)

    This is tight when the principal directions align with spectral directions
    (the "aligned cross-term" assumption).

    Frobenius of cross-term:
      ||cross||_F ≤ 2αβ · ||S_a||_F · ||S_b||_F · max(cos(θ))
    """
    sign = 1.0 if directional_agreement >= 0.0 else -1.0

    frob_sq_a = sum(s * s for s in sigma_a)
    frob_sq_b = sum(s * s for s in sigma_b)

    k = min(len(sigma_a), len(sigma_b), len(cos_theta), len(cos_phi))

    # Cross-term spectral bound: sum of spectral-weighted overlaps
    cross_spectral = 0.0
    for i in range(k):
        cross_spectral += sigma_a[i] * sigma_b[i] * _clip01(cos_theta[i]) * _clip01(cos_phi[i])
    cross_spectral_bound = 2.0 * abs(alpha * beta) * cross_spectral

    # Cross-term Frobenius bound
    max_cos = max((_clip01(c) for c in cos_theta), default=0.0)
    cross_frob_bound = 2.0 * abs(alpha * beta) * math.sqrt(frob_sq_a) * math.sqrt(frob_sq_b) * max_cos

    # Exact Frobenius of merge
    cross_trace = sign * sum(
        sigma_a[i] * sigma_b[i] * _clip01(cos_theta[i]) * _clip01(cos_phi[i]) for i in range(k)
    )
    frob_sq_merged = alpha**2 * frob_sq_a + beta**2 * frob_sq_b + 2.0 * alpha * beta * cross_trace

    # σ_1 bounds via Weyl
    s1a = max(0.0, float(sigma_a[0])) if sigma_a else 0.0
    s1b = max(0.0, float(sigma_b[0])) if sigma_b else 0.0
    sigma1_upper = abs(alpha) * s1a + abs(beta) * s1b
    sigma1_lower = max(0.0, abs(abs(alpha) * s1a - abs(beta) * s1b))

    # Over-accumulation: possible iff cross-term can push σ_1 above baseline
    oa_possible = sign > 0 and cross_spectral_bound > 0.0

    return GeneralRankGramDecomposition(
        alpha=alpha,
        beta=beta,
        frobenius_squared_exact=max(0.0, frob_sq_merged),
        sigma1_weyl_upper=sigma1_upper,
        sigma1_weyl_lower=sigma1_lower,
        cross_term_frobenius_bound=cross_frob_bound,
        cross_term_spectral_bound=cross_spectral_bound,
        over_accumulation_condition_met=oa_possible,
    )


def linear_merge_stable_rank_bounds(
    *,
    sigma_a: tuple[float, ...],
    sigma_b: tuple[float, ...],
    cos_theta: tuple[float, ...],
    cos_phi: tuple[float, ...],
    alpha: float = 0.5,
    beta: float = 0.5,
    directional_agreement: float = 1.0,
) -> LinearMergeStableRankBounds:
    """Bound the stable rank of a linearly merged adapter.

    Stable rank = ||ΔW_m||²_F / σ_1(ΔW_m)².

    We use the exact Frobenius norm (from linear_merge_frobenius) and
    the Weyl bounds on σ_1 (from linear_merge_sigma1_bounds) to get:
      sr_lower = ||ΔW_m||²_F / σ_1_upper²
      sr_upper = ||ΔW_m||²_F / σ_1_lower²
    """
    frob = linear_merge_frobenius(
        sigma_a=sigma_a,
        sigma_b=sigma_b,
        cos_theta=cos_theta,
        cos_phi=cos_phi,
        alpha=alpha,
        beta=beta,
        directional_agreement=directional_agreement,
    )

    s1a = max(0.0, float(sigma_a[0])) if sigma_a else 0.0
    s1b = max(0.0, float(sigma_b[0])) if sigma_b else 0.0
    bounds = linear_merge_sigma1_bounds(s1a, s1b, alpha=alpha, beta=beta)

    eps = 1e-12
    sr_lower = frob.frobenius_squared / (bounds.upper_bound**2 + eps) if bounds.upper_bound > eps else 1.0
    sr_upper = frob.frobenius_squared / (bounds.lower_bound**2 + eps) if bounds.lower_bound > eps else float("inf")

    return LinearMergeStableRankBounds(
        lower_bound=max(1.0, sr_lower),
        upper_bound=sr_upper,
        frobenius_squared=frob.frobenius_squared,
        sigma1_lower=bounds.lower_bound,
        sigma1_upper=bounds.upper_bound,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Phase 2 extension: exact rank-r over-accumulation condition
# ═══════════════════════════════════════════════════════════════════════════


def over_accumulation_exact_condition(
    metrics: SubspaceMetrics,
    *,
    coefficients: tuple[float, float] = (0.5, 0.5),
) -> OverAccumulationExactCondition:
    """Derive the exact over-accumulation condition from SubspaceMetrics.

    For linear merge ΔW_m = αΔW_a + βΔW_b, over-accumulation occurs when
    σ_1(ΔW_m) > max(α·σ_1(ΔW_a), β·σ_1(ΔW_b)).

    The cross-term in the Gram matrix is:
      C = αβ(V_a Σ_a U_a^T U_b Σ_b V_b^T + h.c.)

    Its spectral norm is bounded by:
      σ_1(C) ≤ 2αβ · Σ_i s_{a,i} s_{b,i} cos(θ_i) cos(φ_i)

    When directional agreement < 0, the cross-term is sign-flipped,
    reducing rather than inflating σ_1.

    The analytical risk score is:
      risk = clip01(inflation_ratio_upper - 1.0)
    normalized so 0.0 = no inflation possible, 1.0 = ≥100% inflation.
    """
    alpha, beta = _normalize_coefficients(coefficients)

    # Extract singular values from SubspaceMetrics
    s_a = tuple(max(0.0, float(s)) for s in (metrics.effective_singular_values_a or ()))
    s_b = tuple(max(0.0, float(s)) for s in (metrics.effective_singular_values_b or ()))
    c_left = tuple(_clip01(float(c)) for c in (metrics.principal_angle_cosines or ()))
    c_right = tuple(_clip01(float(c)) for c in (metrics.right_principal_angle_cosines or ()))

    # σ_1 estimates
    if s_a:
        sigma1_a = s_a[0]
    else:
        sigma1_a = metrics.frobenius_norm_a / math.sqrt(max(metrics.stable_rank_a, 1.0))

    if s_b:
        sigma1_b = s_b[0]
    else:
        sigma1_b = metrics.frobenius_norm_b / math.sqrt(max(metrics.stable_rank_b, 1.0))

    baseline = max(alpha * sigma1_a, beta * sigma1_b)

    # Cross-term spectral bound
    k = min(len(s_a), len(s_b), len(c_left), len(c_right))
    sign = 1.0 if metrics.directional_agreement >= 0.0 else -1.0

    cross_bound = 0.0
    for i in range(k):
        cross_bound += s_a[i] * s_b[i] * c_left[i] * c_right[i]
    cross_bound *= 2.0 * alpha * beta

    # With negative agreement, cross-term reduces σ_1
    effective_cross = cross_bound * sign

    sigma1_upper = baseline + max(0.0, effective_cross)
    sigma1_lower = max(0.0, baseline - cross_bound)  # worst-case reduction

    inflation_ratio = sigma1_upper / baseline if baseline > 1e-12 else 1.0

    # Risk score: how much inflation is possible relative to baseline
    analytical_risk = _clip01(max(0.0, inflation_ratio - 1.0))

    # Determine what drives the inflation
    if k == 0 or cross_bound < 1e-12:
        driver = "none"
    elif metrics.frobenius_ratio > 3.0 and metrics.mean_overlap < 0.3:
        driver = "scale"
    elif metrics.mean_overlap > 0.5:
        driver = "alignment"
    else:
        driver = "both"

    # Compare to heuristic
    from gradience.vnext.merge.over_accumulation import estimate_over_accumulation

    heuristic = estimate_over_accumulation(metrics, assumed_coefficients=coefficients)
    heuristic_risk = heuristic.score

    if abs(analytical_risk - heuristic_risk) < 0.05:
        comparison = "agrees"
    elif analytical_risk > heuristic_risk:
        comparison = "theory_higher"
    else:
        comparison = "theory_lower"

    return OverAccumulationExactCondition(
        baseline_sigma1=baseline,
        cross_term_spectral_bound=cross_bound,
        sigma1_upper_bound=sigma1_upper,
        sigma1_lower_bound=sigma1_lower,
        inflation_ratio_upper=inflation_ratio,
        is_inflation_possible=(sign > 0 and cross_bound > 1e-12),
        inflation_driven_by=driver,
        analytical_risk_score=analytical_risk,
        analytical_vs_heuristic_alignment=comparison,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Phase 3 stub: norm-equalized over-accumulation analysis
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class NormEqualizedOverAccumulationResult:
    """Analysis of whether norm equalization reduces over-accumulation.

    Norm equalization rescales adapters to geometric-mean Frobenius norm:
      scale_a = sqrt(||B||_F / ||A||_F), scale_b = sqrt(||A||_F / ||B||_F)

    This changes the effective σ_1 contributions and thus the cross-term.
    """

    linear_inflation_ratio: float
    normeq_inflation_ratio: float
    inflation_reduced: bool
    reduction_factor: float  # normeq / linear (< 1 means reduced)


def norm_equalized_over_accumulation_analysis(
    metrics: SubspaceMetrics,
    *,
    coefficients: tuple[float, float] = (0.5, 0.5),
) -> NormEqualizedOverAccumulationResult:
    """Compare over-accumulation risk: linear vs norm-equalized merge.

    Norm equalization multiplies adapter a's spectrum by sqrt(||B||/||A||)
    and adapter b's by sqrt(||A||/||B||), equalizing Frobenius norms
    before linear combination.

    When ρ_F ≫ 1 (one adapter much larger), this compresses the dominant
    adapter's σ_1 but *amplifies* the subordinate adapter's σ_1.  If the
    subspaces overlap (non-trivial principal angle cosines), the amplified
    subordinate spectrum creates a larger cross-term, potentially increasing
    inflation beyond the linear-merge baseline.  Norm equalization reliably
    reduces inflation only when subspace overlap is low.
    """
    alpha, beta = _normalize_coefficients(coefficients)

    # Linear merge inflation
    linear_cond = over_accumulation_exact_condition(metrics, coefficients=coefficients)
    linear_ratio = linear_cond.inflation_ratio_upper

    # Norm-equalized: rescale singular values
    neq = norm_equalized_scaling(metrics.frobenius_norm_a, metrics.frobenius_norm_b)

    s_a = tuple(max(0.0, float(s)) for s in (metrics.effective_singular_values_a or ()))
    s_b = tuple(max(0.0, float(s)) for s in (metrics.effective_singular_values_b or ()))
    c_left = tuple(_clip01(float(c)) for c in (metrics.principal_angle_cosines or ()))
    c_right = tuple(_clip01(float(c)) for c in (metrics.right_principal_angle_cosines or ()))

    # Rescaled σ_1 values
    sigma1_a_neq = (s_a[0] * neq.scale_a) if s_a else 0.0
    sigma1_b_neq = (s_b[0] * neq.scale_b) if s_b else 0.0
    baseline_neq = max(alpha * sigma1_a_neq, beta * sigma1_b_neq)

    # Cross-term with rescaled spectra
    k = min(len(s_a), len(s_b), len(c_left), len(c_right))
    sign = 1.0 if metrics.directional_agreement >= 0.0 else -1.0

    cross_neq = 0.0
    for i in range(k):
        cross_neq += (s_a[i] * neq.scale_a) * (s_b[i] * neq.scale_b) * c_left[i] * c_right[i]
    cross_neq *= 2.0 * alpha * beta

    effective_cross_neq = cross_neq * sign
    sigma1_upper_neq = baseline_neq + max(0.0, effective_cross_neq)
    neq_ratio = sigma1_upper_neq / baseline_neq if baseline_neq > 1e-12 else 1.0

    reduction = neq_ratio / linear_ratio if linear_ratio > 1e-12 else 1.0

    return NormEqualizedOverAccumulationResult(
        linear_inflation_ratio=linear_ratio,
        normeq_inflation_ratio=neq_ratio,
        inflation_reduced=(neq_ratio < linear_ratio),
        reduction_factor=reduction,
    )
