"""
Synthetic geometry helpers for analytical spectral-theory validation.
"""

from __future__ import annotations

import math

import torch


def _random_unit_vector(n: int, *, generator: torch.Generator) -> torch.Tensor:
    v = torch.randn(n, generator=generator, dtype=torch.float64)
    return v / (v.norm() + 1e-12)


def _orthonormal_pair(n: int, *, generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    v1 = _random_unit_vector(n, generator=generator)
    v2 = _random_unit_vector(n, generator=generator)
    v2 = v2 - (v2 @ v1) * v1
    v2 = v2 / (v2.norm() + 1e-12)
    return v1, v2


def make_rank1_pair_with_cosines(
    *,
    d_out: int = 128,
    d_in: int = 128,
    sigma_a: float = 2.0,
    sigma_b: float = 1.5,
    cos_theta: float = 0.7,
    cos_phi: float = 0.6,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Construct rank-1 updates with controlled left/right cosine overlaps.
    """
    ctheta = min(1.0, max(0.0, float(cos_theta)))
    cphi = min(1.0, max(0.0, float(cos_phi)))
    stheta = math.sqrt(max(0.0, 1.0 - ctheta * ctheta))
    sphi = math.sqrt(max(0.0, 1.0 - cphi * cphi))

    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))

    ua, ua_orth = _orthonormal_pair(d_out, generator=g)
    va, va_orth = _orthonormal_pair(d_in, generator=g)

    ub = ctheta * ua + stheta * ua_orth
    vb = cphi * va + sphi * va_orth

    delta_a = float(sigma_a) * torch.outer(ua, va)
    delta_b = float(sigma_b) * torch.outer(ub, vb)
    return delta_a, delta_b


def sigma1(x: torch.Tensor) -> float:
    return float(torch.linalg.svdvals(x.to(torch.float64))[0].item())


def frobenius(x: torch.Tensor) -> float:
    return float(torch.linalg.norm(x.to(torch.float64), "fro").item())


def stable_rank(x: torch.Tensor) -> float:
    s = torch.linalg.svdvals(x.to(torch.float64))
    if s[0].item() < 1e-12:
        return 1.0
    return float(s.pow(2).sum() / s[0].pow(2))


def singular_values(x: torch.Tensor) -> list[float]:
    return torch.linalg.svdvals(x.to(torch.float64)).tolist()


# ═══════════════════════════════════════════════════════════════════════════
# Rank-r synthetic matrix generation with controlled geometry
# ═══════════════════════════════════════════════════════════════════════════


def _random_orthonormal(n: int, k: int, *, generator: torch.Generator) -> torch.Tensor:
    """Generate a random n×k matrix with orthonormal columns."""
    M = torch.randn(n, k, generator=generator, dtype=torch.float64)
    Q, _ = torch.linalg.qr(M)
    return Q[:, :k]


def _rotate_basis(
    U: torch.Tensor,
    cos_angles: list[float],
    *,
    generator: torch.Generator,
) -> torch.Tensor:
    """Construct V such that U^T V has prescribed singular values (cos_angles).

    For each column i, constructs:
      v_i = cos(θ_i) * u_i + sin(θ_i) * w_i

    where w_i is a unit vector orthogonal to all of U's columns.
    This gives exact control over principal angle cosines.
    """
    n, k = U.shape
    assert len(cos_angles) == k

    # Get a random orthonormal complement to U
    # We need k vectors orthogonal to the k columns of U
    W_raw = torch.randn(n, k, generator=generator, dtype=torch.float64)
    # Project out U: W = W_raw - U(U^T W_raw)
    W = W_raw - U @ (U.T @ W_raw)
    # Orthonormalize W
    W, _ = torch.linalg.qr(W)
    W = W[:, :k]

    V = torch.zeros_like(U)
    for i in range(k):
        c = min(1.0, max(0.0, cos_angles[i]))
        s = math.sqrt(max(0.0, 1.0 - c * c))
        V[:, i] = c * U[:, i] + s * W[:, i]

    return V


def make_rankr_pair_with_geometry(
    *,
    d_out: int = 128,
    d_in: int = 128,
    rank: int = 4,
    sigma_a: list[float] | None = None,
    sigma_b: list[float] | None = None,
    cos_theta: list[float] | None = None,
    cos_phi: list[float] | None = None,
    directional_sign: float = 1.0,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Construct rank-r ΔW pairs with fully controlled spectral geometry.

    Parameters
    ----------
    d_out, d_in : matrix dimensions
    rank : rank of each adapter
    sigma_a : singular values for adapter A (length rank, descending)
    sigma_b : singular values for adapter B (length rank, descending)
    cos_theta : left principal angle cosines (length rank)
    cos_phi : right principal angle cosines (length rank)
    directional_sign : +1 for aligned, -1 for opposing
    seed : random seed

    Returns
    -------
    (delta_a, delta_b) : pair of (d_out, d_in) rank-r matrices
    """
    r = rank
    if sigma_a is None:
        sigma_a = [2.0 / (i + 1) for i in range(r)]
    if sigma_b is None:
        sigma_b = [1.5 / (i + 1) for i in range(r)]
    if cos_theta is None:
        cos_theta = [0.5] * r
    if cos_phi is None:
        cos_phi = [0.5] * r

    assert len(sigma_a) == r
    assert len(sigma_b) == r
    assert len(cos_theta) == r
    assert len(cos_phi) == r

    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))

    # Build adapter A: ΔW_a = U_a @ diag(σ_a) @ V_a^T
    U_a = _random_orthonormal(d_out, r, generator=g)
    V_a = _random_orthonormal(d_in, r, generator=g)

    Sigma_a = torch.diag(torch.tensor(sigma_a, dtype=torch.float64))
    delta_a = U_a @ Sigma_a @ V_a.T

    # Build adapter B with prescribed principal angles relative to A
    U_b = _rotate_basis(U_a, cos_theta, generator=g)
    V_b = _rotate_basis(V_a, cos_phi, generator=g)

    if directional_sign < 0:
        # Flip sign to get negative directional agreement
        U_b = -U_b

    Sigma_b = torch.diag(torch.tensor(sigma_b, dtype=torch.float64))
    delta_b = U_b @ Sigma_b @ V_b.T

    return delta_a, delta_b


def make_equal_spectrum_pair(
    *,
    d_out: int = 128,
    d_in: int = 128,
    rank: int = 4,
    sigma: list[float] | None = None,
    cos_theta: list[float] | None = None,
    cos_phi: list[float] | None = None,
    per_direction_sign: list[float] | None = None,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Construct rank-r pair with identical spectra (Σ_a = Σ_b).

    This is the special case where the equal-rank equal-spectrum analytical
    solution applies.
    """
    r = rank
    if sigma is None:
        sigma = [2.0 / (i + 1) for i in range(r)]

    return make_rankr_pair_with_geometry(
        d_out=d_out,
        d_in=d_in,
        rank=r,
        sigma_a=sigma,
        sigma_b=sigma,
        cos_theta=cos_theta,
        cos_phi=cos_phi,
        directional_sign=1.0 if per_direction_sign is None else per_direction_sign[0],
        seed=seed,
    )
