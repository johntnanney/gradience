"""
Refactor a dense ΔW matrix back into LoRA (B, A) format via truncated SVD.

After merging two adapters, the result is a dense (d_out × d_in) matrix.
To produce a PEFT-compatible adapter, we need to decompose it back into
LoRA factors B (d_out × r) and A (r × d_in) such that:

    scaling * B @ A ≈ ΔW    where scaling = alpha / rank

The truncated SVD provides the best rank-r approximation in Frobenius norm.
The reconstruction error measures how much information is lost.

Usage::

    from gradience.vnext.merge.refactor import refactor_to_lora

    B, A, error = refactor_to_lora(merged_dW, target_rank=8, alpha=16.0)
    # scaling * B @ A ≈ merged_dW
    # error ∈ [0, 1] — fraction of Frobenius energy lost
"""

from __future__ import annotations

import torch
from torch import Tensor

from gradience.exceptions import MergeError

# dtype mapping (shared with merge/__init__.py)
_DTYPE_MAP = {
    "float64": torch.float64,
    "float32": torch.float32,
    "fp64": torch.float64,
    "fp32": torch.float32,
}


def refactor_to_lora(
    dW: Tensor,
    target_rank: int,
    alpha: float,
    compute_dtype: str = "float32",
) -> tuple[Tensor, Tensor, float]:
    """Decompose a dense ΔW into LoRA factors (B, A) via truncated SVD.

    The decomposition satisfies: ``scaling * B @ A ≈ dW`` where
    ``scaling = alpha / target_rank``.

    We distribute singular values evenly between B and A using sqrt(S):
    - B = U_r @ diag(sqrt(S_r / scaling))
    - A = diag(sqrt(S_r / scaling)) @ Vt_r

    Parameters
    ----------
    dW : (d_out, d_in) dense delta-weight matrix to decompose
    target_rank : desired rank for the LoRA adapter
    alpha : LoRA alpha scaling parameter
    compute_dtype : ``"float64"`` (default) or ``"float32"`` for SVD precision

    Returns
    -------
    B : (d_out, target_rank) — LoRA B matrix
    A : (target_rank, d_in) — LoRA A matrix
    reconstruction_error : ||dW - scaling * B @ A||_F / ||dW||_F
        Fraction of Frobenius energy lost by rank truncation.
        0.0 means perfect reconstruction; closer to 1.0 means more loss.

    Raises
    ------
    ValueError
        If target_rank is invalid or dW is not 2-D.
    """
    if dW.ndim != 2:
        raise MergeError(f"dW must be 2-D, got shape {dW.shape}")

    d_out, d_in = dW.shape
    max_rank = min(d_out, d_in)

    if target_rank < 1:
        raise MergeError(f"target_rank must be >= 1, got {target_rank}")
    if target_rank > max_rank:
        raise MergeError(f"target_rank ({target_rank}) exceeds max possible rank ({max_rank}) for shape {dW.shape}")

    dtype = _DTYPE_MAP.get(compute_dtype)
    if dtype is None:
        raise MergeError(f"Unsupported compute_dtype '{compute_dtype}'. Choose from: {sorted(_DTYPE_MAP.keys())}")

    scaling = alpha / max(target_rank, 1)
    if abs(scaling) < 1e-12:
        raise MergeError(
            f"LoRA scaling factor is effectively zero (alpha={alpha}, target_rank={target_rank}). "
            "Cannot factor ΔW into B @ A when scaling ≈ 0."
        )

    # Compute SVD in requested precision
    orig_dtype = dW.dtype
    dW_compute = dW.to(dtype)

    # Use randomized SVD when we only need a small number of components
    # relative to the matrix size. For merged LoRA adapters, dW is at most
    # rank ~2r (sum of two rank-r matrices), so we never need a full SVD
    # on the (d_out x d_in) matrix.
    d_out, d_in = dW_compute.shape
    if target_rank < min(d_out, d_in) // 2:
        # torch.svd_lowrank is O(d*r^2) vs O(d*min(d_out,d_in)) for full SVD
        # Use niter=4 for good accuracy on well-conditioned low-rank matrices
        U_r, S_r, V_r = torch.svd_lowrank(dW_compute, q=target_rank, niter=4)
        Vt_r = V_r.T
    else:
        U, S, Vt = torch.linalg.svd(dW_compute, full_matrices=False)
        U_r = U[:, :target_rank]
        S_r = S[:target_rank]
        Vt_r = Vt[:target_rank, :]

    # Factor: we need scaling * B @ A ≈ dW
    # dW ≈ U_r @ diag(S_r) @ Vt_r
    # So: B @ A = (1/scaling) * U_r @ diag(S_r) @ Vt_r
    #
    # Distribute S evenly via sqrt:
    # B = U_r @ diag(sqrt(S_r / scaling))
    # A = diag(sqrt(S_r / scaling)) @ Vt_r
    sqrt_S_scaled = torch.sqrt(S_r / scaling).unsqueeze(1)  # (target_rank, 1)

    B = U_r * sqrt_S_scaled.T  # (d_out, target_rank)
    A = sqrt_S_scaled * Vt_r  # (target_rank, d_in)

    # Compute reconstruction error
    dW_approx = scaling * B @ A
    residual_norm = torch.linalg.norm(dW_compute - dW_approx).item()
    original_norm = torch.linalg.norm(dW_compute).item()

    if original_norm > 0:
        reconstruction_error = residual_norm / original_norm
    else:
        reconstruction_error = 0.0

    # Return in original dtype
    return B.to(orig_dtype), A.to(orig_dtype), reconstruction_error
