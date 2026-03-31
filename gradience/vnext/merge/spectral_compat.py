"""
Spectral compatibility metrics between LoRA adapter pairs.

Mathematical foundation:
- Subspace overlap via principal angles (Bjorck & Golub, 1973)
- Directional agreement via projection coefficient analysis
- Magnitude balance via singular value ratio

All operations use float64 by default for numerical stability.
The QR-based SVD avoids materializing the full (d_out x d_in) matrix,
following the pattern in gradience.vnext.svd_truncate._compute_svd_truncation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch

from gradience.exceptions import MergeError
from gradience.vnext.audit.lora_audit import _energy_rank
from gradience.vnext.merge.scale import symmetric_frobenius_metrics, symmetric_scale_metrics

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SubspaceMetrics:
    """Immutable per-layer compatibility metrics.

    Frozen because these are computed once and referenced in multiple
    places (verdict logic, report generation, coefficient suggestion).
    """

    # Subspace geometry
    principal_angle_cosines: tuple[float, ...]  # cos(theta_i), descending
    mean_overlap: float  # mean of principal angle cosines [0, 1]
    max_overlap: float  # maximum single-direction overlap [0, 1]

    # Directional analysis
    directional_agreement: float  # projection cosine similarity [-1, 1]

    # Scale analysis (asymmetric, kept for backward compatibility)
    magnitude_ratio: float  # sigma_1(larger) / sigma_1(smaller), >= 1
    frobenius_ratio: float  # ||larger||_F / ||smaller||_F, >= 1
    frobenius_norm_a: float  # ||ΔW_a||_F (scaled)
    frobenius_norm_b: float  # ||ΔW_b||_F (scaled)

    # Symmetric scale analysis (Section 3)
    scale_bounded_ratio: float  # min(sigma1_A, sigma1_B) / max(sigma1_A, sigma1_B), [0,1]
    scale_log_ratio: float  # log(sigma1_A) - log(sigma1_B)
    frob_bounded_ratio: float  # min(||A||_F, ||B||_F) / max(||A||_F, ||B||_F), [0,1]
    frob_log_ratio: float  # log(||A||_F) - log(||B||_F)

    # Rank structure
    effective_rank_a: int
    effective_rank_b: int
    nominal_rank_a: int
    nominal_rank_b: int
    stable_rank_a: float
    stable_rank_b: float

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # Convert tuple to list for JSON serialization
        d["principal_angle_cosines"] = list(d["principal_angle_cosines"])
        return d


# ---------------------------------------------------------------------------
# SVD helpers
# ---------------------------------------------------------------------------


def compute_layer_svd(
    A: torch.Tensor,
    B: torch.Tensor,
    scaling: float = 1.0,
    *,
    compute_dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """QR-based full SVD of scaled ΔW = scaling * B @ A.

    Returns (U, S, Vt) without materializing the full ΔW matrix.

    Parameters
    ----------
    A : (r, d_in)
    B : (d_out, r)
    scaling : LoRA scaling factor (alpha / r)
    compute_dtype : precision for SVD computation

    Returns
    -------
    U  : (d_out, r) left singular vectors
    S  : (r,) singular values, descending
    Vt : (r, d_in) right singular vectors transposed
    """
    A_work = A.detach().to(dtype=compute_dtype, device="cpu")
    B_work = B.detach().to(dtype=compute_dtype, device="cpu")

    if scaling != 1.0:
        B_work = B_work * scaling

    # Step 1: QR(B) = Qb @ Rb
    Qb, Rb = torch.linalg.qr(B_work)  # Qb: (d_out, r), Rb: (r, r)

    # Step 2: QR(A^T) = Qa @ Ra
    At = A_work.T  # (d_in, r)
    Qa, Ra = torch.linalg.qr(At)  # Qa: (d_in, r), Ra: (r, r)

    # Step 3: SVD of small r x r matrix M = Rb @ Ra^T
    M = Rb @ Ra.T  # (r, r)
    U_small, S, Vt_small = torch.linalg.svd(M, full_matrices=False)

    # Step 4: Recover full singular vectors
    U = Qb @ U_small  # (d_out, r)
    Vt = Vt_small @ Qa.T  # (r, d_in)

    # Sort descending (should already be, but ensure)
    order = torch.argsort(S, descending=True)
    S = S[order]
    U = U[:, order]
    Vt = Vt[order, :]

    return U, S, Vt


def _stable_rank(s: torch.Tensor, eps: float = 1e-12) -> float:
    """Stable rank: ||s||_F^2 / s_max^2."""
    if s.numel() == 0 or s[0].item() < eps:
        return 1.0
    return (s.pow(2).sum() / s[0].pow(2)).item()


# ---------------------------------------------------------------------------
# Core comparison
# ---------------------------------------------------------------------------


def compute_subspace_metrics(
    A_a: torch.Tensor,
    B_a: torch.Tensor,
    alpha_a: float | None,
    r_a: int,
    A_b: torch.Tensor,
    B_b: torch.Tensor,
    alpha_b: float | None,
    r_b: int,
    *,
    energy_threshold: float = 0.90,
    compute_dtype: torch.dtype = torch.float32,
    eps: float = 1e-12,
) -> SubspaceMetrics:
    """Full per-layer spectral comparison.

    1. QR-based SVD for each adapter
    2. Truncate to effective rank (energy threshold)
    3. Principal angles via SVD of U_a_eff^T @ U_b_eff
    4. Directional agreement via projection coefficient analysis
    5. Magnitude balance via singular value ratios
    """
    if r_a <= 0:
        raise MergeError(f"rank_a must be positive, got {r_a}")
    if r_b <= 0:
        raise MergeError(f"rank_b must be positive, got {r_b}")

    scaling_a = (alpha_a / r_a) if alpha_a is not None else 1.0
    scaling_b = (alpha_b / r_b) if alpha_b is not None else 1.0

    # === SVD ===
    U_a, S_a, Vt_a = compute_layer_svd(A_a, B_a, scaling_a, compute_dtype=compute_dtype)
    U_b, S_b, Vt_b = compute_layer_svd(A_b, B_b, scaling_b, compute_dtype=compute_dtype)

    # === Effective rank ===
    eff_rank_a = max(_energy_rank(S_a, energy_threshold, eps), 1)
    eff_rank_b = max(_energy_rank(S_b, energy_threshold, eps), 1)

    # === Principal angles ===
    U_a_eff = U_a[:, :eff_rank_a]
    U_b_eff = U_b[:, :eff_rank_b]

    cross = U_a_eff.T @ U_b_eff  # (eff_rank_a, eff_rank_b)
    cos_angles = torch.linalg.svdvals(cross)
    # Clamp to [0, 1] for numerical stability
    cos_angles = cos_angles.clamp(0.0, 1.0)

    mean_overlap = cos_angles.mean().item() if cos_angles.numel() > 0 else 0.0
    max_overlap = cos_angles.max().item() if cos_angles.numel() > 0 else 0.0

    # === Directional agreement ===
    # Project ΔW_b = scaling_b * B_b @ A_b into ΔW_a's coordinate system.
    # Instead of materializing the full (d_out × d_in) matrix, chain
    # through the rank bottleneck:
    #   projection = U_a_eff^T @ (scaling_b * B_b @ A_b) @ V_a_eff
    #             = scaling_b * (U_a_eff^T @ B_b) @ (A_b @ V_a_eff)
    # Max intermediate shape: (eff_rank_a, r_b), never (d_out, d_in).
    Vt_a_eff = Vt_a[:eff_rank_a, :]  # (eff_rank_a, d_in)
    Ut_B = U_a_eff.T @ B_b.to(compute_dtype)  # (eff_rank_a, r_b)
    A_V = A_b.to(compute_dtype) @ Vt_a_eff.T  # (r_b, eff_rank_a)
    projection = scaling_b * (Ut_B @ A_V)  # (eff_rank_a, eff_rank_a)
    diag_proj = torch.diag(projection)
    s_a_trunc = S_a[:eff_rank_a]

    if s_a_trunc.norm() > eps and diag_proj.norm() > eps:
        agreement = torch.nn.functional.cosine_similarity(
            diag_proj.unsqueeze(0),
            s_a_trunc.unsqueeze(0),
        ).item()
    else:
        agreement = 0.0

    # === Magnitude balance ===
    s1_max = S_a[0].item() if S_a.numel() > 0 else 0.0
    s2_max = S_b[0].item() if S_b.numel() > 0 else 0.0

    if s1_max > eps and s2_max > eps:
        mag_ratio = max(s1_max, s2_max) / min(s1_max, s2_max)
    else:
        mag_ratio = float("inf")

    frob_a = S_a.pow(2).sum().sqrt().item()
    frob_b = S_b.pow(2).sum().sqrt().item()

    if frob_a > eps and frob_b > eps:
        frob_ratio = max(frob_a, frob_b) / min(frob_a, frob_b)
    else:
        frob_ratio = float("inf")

    # === Symmetric scale analysis (Section 3) ===
    sym_scale = symmetric_scale_metrics(s1_max, s2_max, eps=eps)
    sym_frob = symmetric_frobenius_metrics(frob_a, frob_b, eps=eps)

    return SubspaceMetrics(
        principal_angle_cosines=tuple(cos_angles.tolist()),
        mean_overlap=mean_overlap,
        max_overlap=max_overlap,
        directional_agreement=agreement,
        magnitude_ratio=mag_ratio,
        frobenius_ratio=frob_ratio,
        frobenius_norm_a=frob_a,
        frobenius_norm_b=frob_b,
        scale_bounded_ratio=sym_scale["scale_bounded_ratio"],
        scale_log_ratio=sym_scale["scale_log_ratio"],
        frob_bounded_ratio=sym_frob["frob_bounded_ratio"],
        frob_log_ratio=sym_frob["frob_log_ratio"],
        effective_rank_a=eff_rank_a,
        effective_rank_b=eff_rank_b,
        nominal_rank_a=r_a,
        nominal_rank_b=r_b,
        stable_rank_a=_stable_rank(S_a, eps),
        stable_rank_b=_stable_rank(S_b, eps),
    )


# ---------------------------------------------------------------------------
# Aggregate overlap utilities
# ---------------------------------------------------------------------------


def overlap_score(metrics: SubspaceMetrics) -> float:
    """Squared-Frobenius overlap from existing principal angle cosines.

    Computes ``sum(cos²(θ_i)) / min(eff_rank_a, eff_rank_b)`` using the
    principal angle cosines already stored on the metrics object.  This is
    a normalised measure of shared subspace volume: 0.0 for orthogonal
    subspaces, 1.0 for identical subspaces, independent of rank.
    """
    cosines = metrics.principal_angle_cosines
    if not cosines:
        return 0.0
    k = min(metrics.effective_rank_a, metrics.effective_rank_b)
    if k <= 0:
        return 0.0
    return sum(c * c for c in cosines) / k


def aggregate_overlap(
    layer_metrics: list[tuple[str, SubspaceMetrics]],
) -> float:
    """Parameter-weighted aggregate overlap across layers.

    Each layer's contribution is weighted by the larger of its two
    Frobenius norms (same weighting used by ``assess_overall`` for
    the aggregate compatibility score).

    Parameters
    ----------
    layer_metrics
        List of ``(layer_name, SubspaceMetrics)`` tuples.

    Returns
    -------
    Weighted mean overlap score in ``[0, 1]``.  Returns 0.0 for an
    empty list.
    """
    if not layer_metrics:
        return 0.0

    total_weight = 0.0
    weighted_sum = 0.0
    for _name, m in layer_metrics:
        weight = max(m.frobenius_norm_a, m.frobenius_norm_b)
        total_weight += weight
        weighted_sum += weight * overlap_score(m)

    if total_weight <= 0.0:
        return 0.0
    return weighted_sum / total_weight
