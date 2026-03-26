"""Optional core-space / shared-basis diagnostics for merge pairs.

Diagnostic intent:
- Estimate whether two adapter updates can be represented cleanly in one
  shared low-rank basis.
- Keep outputs bounded, explicit, and easy to reason about.
- Never alter default merge recommendations unless a caller explicitly
  opts into using these diagnostics.
"""

from __future__ import annotations

from typing import Final

import torch

from gradience.vnext.audit.lora_audit import _energy_rank
from gradience.vnext.merge.core_space_types import (
    CoreSpaceLayerDiagnostic,
    CoreSpacePairDiagnostic,
)
from gradience.vnext.merge.spectral_compat import compute_layer_svd

DEFAULT_CORE_SPACE_ENERGY_THRESHOLD: Final[float] = 0.90
_EPS: Final[float] = 1e-12


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _captured_energy_in_basis(
    basis_u: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    scaling: float,
    *,
    compute_dtype: torch.dtype,
) -> float:
    """Return ||U^T (scaling * B @ A)||_F^2 without materializing full update."""
    if basis_u.numel() == 0:
        return 0.0

    U = basis_u.detach().to(device="cpu", dtype=compute_dtype)
    A_work = A.detach().to(device="cpu", dtype=compute_dtype)
    B_work = B.detach().to(device="cpu", dtype=compute_dtype) * scaling

    projected = (U.T @ B_work) @ A_work
    return float(projected.pow(2).sum().item())


def _classify_layer_status(shared_basis_score: float, basis_distortion: float, rank_inflation: float) -> str:
    """Classify one layer using distortion + rank inflation + shared score.

    ``rank_inflation`` is:
      max(0, k_shared - max(k_a, k_b)) / max(k_a, k_b)

    Intuition:
    - Low distortion alone is not enough for compatibility.
    - A shared basis that needs substantially higher rank than either source
      should be downgraded because it indicates weak basis sharing.
    """
    if basis_distortion <= 0.08 and rank_inflation <= 0.20 and shared_basis_score >= 0.78:
        return "compatible"
    if basis_distortion <= 0.20 and rank_inflation <= 0.60 and shared_basis_score >= 0.55:
        return "marginal"
    return "incompatible"


def compute_core_space_layer_diagnostic(
    layer_name: str,
    A_a: torch.Tensor,
    B_a: torch.Tensor,
    alpha_a: float | None,
    r_a: int,
    A_b: torch.Tensor,
    B_b: torch.Tensor,
    alpha_b: float | None,
    r_b: int,
    *,
    energy_threshold: float = DEFAULT_CORE_SPACE_ENERGY_THRESHOLD,
    compute_dtype: torch.dtype = torch.float64,
    eps: float = _EPS,
) -> CoreSpaceLayerDiagnostic:
    """Compute shared-basis diagnostics for one matched layer.

    Formulas (all energy values are squared Frobenius norms):
    - ``separate_error`` = residual after each adapter uses its own basis at
      its own effective rank.
    - ``shared_error`` = residual after both adapters are projected to one
      shared basis learned from joint left-space energy.
    - ``basis_distortion`` = max(0, (shared_error - separate_error) / total_energy)
    - ``rank_inflation`` = max(0, k_shared - max(k_a, k_b)) / max(k_a, k_b)
    - ``shared_basis_score`` = weighted blend of:
      - ``(1 - basis_distortion)``
      - shared rank efficiency ``max(k_a, k_b) / k_shared``
      - retained shared energy ratio
    """
    if r_a <= 0 or r_b <= 0:
        return CoreSpaceLayerDiagnostic(
            layer_name=layer_name,
            shared_basis_score=0.0,
            basis_distortion=0.0,
            effective_shared_rank=0,
            status="not_applicable",
        )

    scaling_a = (alpha_a / r_a) if alpha_a is not None else 1.0
    scaling_b = (alpha_b / r_b) if alpha_b is not None else 1.0

    # Individual update decompositions
    U_a, S_a, _ = compute_layer_svd(A_a, B_a, scaling_a, compute_dtype=compute_dtype)
    U_b, S_b, _ = compute_layer_svd(A_b, B_b, scaling_b, compute_dtype=compute_dtype)

    energy_a = float(S_a.pow(2).sum().item()) if S_a.numel() else 0.0
    energy_b = float(S_b.pow(2).sum().item()) if S_b.numel() else 0.0
    total_energy = energy_a + energy_b
    if total_energy <= eps:
        return CoreSpaceLayerDiagnostic(
            layer_name=layer_name,
            shared_basis_score=0.0,
            basis_distortion=0.0,
            effective_shared_rank=0,
            status="not_applicable",
        )

    k_a = max(_energy_rank(S_a, energy_threshold, eps), 1) if energy_a > eps else 0
    k_b = max(_energy_rank(S_b, energy_threshold, eps), 1) if energy_b > eps else 0

    # Build joint left-space representation:
    #   J J^T = (ΔW_a)(ΔW_a)^T + (ΔW_b)(ΔW_b)^T
    #         = [U_a S_a, U_b S_b] [U_a S_a, U_b S_b]^T
    weighted_a = U_a * S_a.unsqueeze(0) if S_a.numel() else U_a
    weighted_b = U_b * S_b.unsqueeze(0) if S_b.numel() else U_b
    joint_left = torch.cat([weighted_a, weighted_b], dim=1).to(device="cpu", dtype=compute_dtype)

    if joint_left.numel() == 0:
        return CoreSpaceLayerDiagnostic(
            layer_name=layer_name,
            shared_basis_score=0.0,
            basis_distortion=0.0,
            effective_shared_rank=0,
            status="not_applicable",
        )

    U_joint, S_joint, _ = torch.linalg.svd(joint_left, full_matrices=False)
    if S_joint.numel() == 0:
        return CoreSpaceLayerDiagnostic(
            layer_name=layer_name,
            shared_basis_score=0.0,
            basis_distortion=0.0,
            effective_shared_rank=0,
            status="not_applicable",
        )

    k_shared = max(_energy_rank(S_joint, energy_threshold, eps), 1)
    U_shared = U_joint[:, :k_shared]

    separate_error_a = float(S_a[k_a:].pow(2).sum().item()) if k_a < S_a.numel() else 0.0
    separate_error_b = float(S_b[k_b:].pow(2).sum().item()) if k_b < S_b.numel() else 0.0
    separate_error = separate_error_a + separate_error_b

    captured_shared_a = _captured_energy_in_basis(
        U_shared,
        A_a,
        B_a,
        scaling_a,
        compute_dtype=compute_dtype,
    )
    captured_shared_b = _captured_energy_in_basis(
        U_shared,
        A_b,
        B_b,
        scaling_b,
        compute_dtype=compute_dtype,
    )
    shared_error = max(energy_a - captured_shared_a, 0.0) + max(energy_b - captured_shared_b, 0.0)

    distortion_raw = (shared_error - separate_error) / (total_energy + eps)
    basis_distortion = _clamp01(max(0.0, distortion_raw))
    shared_energy_ratio = _clamp01(1.0 - (shared_error / (total_energy + eps)))
    rank_baseline = max(k_a, k_b, 1)
    rank_inflation = max(float(k_shared - rank_baseline), 0.0) / float(rank_baseline)
    rank_efficiency = _clamp01(float(rank_baseline) / float(max(k_shared, 1)))
    shared_basis_score = _clamp01(
        0.45 * (1.0 - basis_distortion) + 0.40 * rank_efficiency + 0.15 * shared_energy_ratio
    )

    return CoreSpaceLayerDiagnostic(
        layer_name=layer_name,
        shared_basis_score=shared_basis_score,
        basis_distortion=basis_distortion,
        effective_shared_rank=int(k_shared),
        status=_classify_layer_status(shared_basis_score, basis_distortion, rank_inflation),
    )


def _classify_pair_status(
    mean_score: float,
    mean_distortion: float,
    n_layers_scored: int,
    n_layers_marginal: int,
    n_layers_incompatible: int,
) -> str:
    if n_layers_scored == 0:
        return "not_applicable"

    incompatible_ratio = n_layers_incompatible / n_layers_scored
    marginal_ratio = n_layers_marginal / n_layers_scored

    if incompatible_ratio > 0.25:
        return "incompatible"
    if n_layers_incompatible > 0:
        return "marginal"
    if marginal_ratio <= 0.20 and mean_distortion <= 0.10 and mean_score >= 0.75:
        return "compatible"
    if marginal_ratio <= 0.60 and mean_distortion <= 0.25 and mean_score >= 0.55:
        return "marginal"
    return "incompatible"


def aggregate_core_space_diagnostics(
    layer_diagnostics: list[CoreSpaceLayerDiagnostic],
) -> CoreSpacePairDiagnostic:
    """Aggregate per-layer core-space diagnostics to one pair-level object."""
    scored = [ld for ld in layer_diagnostics if ld.status != "not_applicable"]
    if not scored:
        return CoreSpacePairDiagnostic.from_layers(layer_diagnostics, status="not_applicable")

    mean_score = sum(ld.shared_basis_score for ld in scored) / len(scored)
    mean_distortion = sum(ld.basis_distortion for ld in scored) / len(scored)
    n_marginal = sum(1 for ld in scored if ld.status == "marginal")
    n_incompatible = sum(1 for ld in scored if ld.status == "incompatible")

    pair_status = _classify_pair_status(
        mean_score=mean_score,
        mean_distortion=mean_distortion,
        n_layers_scored=len(scored),
        n_layers_marginal=n_marginal,
        n_layers_incompatible=n_incompatible,
    )

    return CoreSpacePairDiagnostic.from_layers(layer_diagnostics, status=pair_status)
