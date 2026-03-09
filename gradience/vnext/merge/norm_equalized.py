"""
Norm-equalized linear merge (M2 baseline stub).

Scales both adapters to equal Frobenius norm before linear averaging.
This removes scale imbalance as a confounding factor in merge quality.

This is a minimal M2 stub -- full implementation and testing deferred to M2.
"""

from __future__ import annotations

from typing import Any

import torch


def norm_equalized_merge(
    dW_A: torch.Tensor,
    dW_B: torch.Tensor,
    weight_A: float = 0.5,
    *,
    eps: float = 1e-12,
) -> dict[str, Any]:
    """Merge two delta-weight matrices after equalizing their Frobenius norms.

    Steps:
    1. Compute Frobenius norms of both inputs
    2. Scale both to the geometric mean of the two norms
    3. Weighted linear combination

    Parameters
    ----------
    dW_A : (d_out, d_in) delta-weight matrix from adapter A
    dW_B : (d_out, d_in) delta-weight matrix from adapter B
    weight_A : weight for adapter A (weight_B = 1 - weight_A)
    eps : numerical stability constant

    Returns
    -------
    dict with keys:
        merged : (d_out, d_in) merged delta-weight
        norm_A_original : original Frobenius norm of A
        norm_B_original : original Frobenius norm of B
        target_norm : the geometric mean norm both were scaled to
        scale_factor_A : multiplicative factor applied to A
        scale_factor_B : multiplicative factor applied to B
    """
    norm_A = torch.linalg.norm(dW_A, "fro")
    norm_B = torch.linalg.norm(dW_B, "fro")

    norm_A_val = norm_A.item()
    norm_B_val = norm_B.item()

    # Edge case: if either adapter has zero norm, skip scaling for that adapter.
    # A zero-norm adapter contributes nothing regardless of scaling.
    A_is_zero = norm_A_val < eps
    B_is_zero = norm_B_val < eps

    if A_is_zero and B_is_zero:
        # Both are zero -- nothing to equalize, target norm is 0.
        target_norm = 0.0
        scale_A = 1.0
        scale_B = 1.0
    elif A_is_zero:
        # A is zero, can't scale it. Target is B's norm (no change to B).
        target_norm = norm_B_val
        scale_A = 1.0  # scaling zero doesn't matter
        scale_B = 1.0
    elif B_is_zero:
        # B is zero, can't scale it. Target is A's norm (no change to A).
        target_norm = norm_A_val
        scale_A = 1.0
        scale_B = 1.0  # scaling zero doesn't matter
    else:
        # Normal case: geometric mean of the two norms.
        target_norm = (norm_A_val * norm_B_val) ** 0.5
        scale_A = target_norm / norm_A_val
        scale_B = target_norm / norm_B_val

    weight_B = 1.0 - weight_A
    merged = weight_A * (scale_A * dW_A) + weight_B * (scale_B * dW_B)

    return {
        "merged": merged,
        "norm_A_original": norm_A_val,
        "norm_B_original": norm_B_val,
        "target_norm": target_norm,
        "scale_factor_A": scale_A,
        "scale_factor_B": scale_B,
    }
