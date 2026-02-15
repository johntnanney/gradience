"""
Null controls for merge audit metrics.

Provides baseline distributions for spectral metrics to validate
that observed overlap/alignment is meaningful (above chance).

- randomized_subspace_control: Random orthonormal basis vs real basis
- layer_shuffle_control: Cross-layer alignment (unrelated layer pairs)
"""

from __future__ import annotations

import torch
from typing import Any, Dict, List


def _principal_angle_cosines(
    U_A: torch.Tensor, U_B: torch.Tensor
) -> torch.Tensor:
    """Compute principal angle cosines between two orthonormal bases.

    Parameters
    ----------
    U_A : (d, k_a) orthonormal columns
    U_B : (d, k_b) orthonormal columns

    Returns
    -------
    Cosines clamped to [0, 1], shape (min(k_a, k_b),).
    """
    cross = U_A.T @ U_B
    cosines = torch.linalg.svdvals(cross)
    return cosines.clamp(0.0, 1.0)


def _random_orthonormal(
    d: int, k: int, generator: torch.Generator
) -> torch.Tensor:
    """Generate a random (d, k) orthonormal matrix via QR decomposition.

    All computation in float64 for stability.
    """
    M = torch.randn(d, k, dtype=torch.float64, generator=generator)
    Q, _ = torch.linalg.qr(M)
    return Q


def randomized_subspace_control(
    U_A: torch.Tensor,
    U_B: torch.Tensor,
    n_random: int = 10,
    seed: int = 42,
) -> Dict[str, Any]:
    """Compute principal angles between U_B and random orthonormal bases.

    For each trial:
    1. Generate a random matrix of same shape as U_A
    2. Orthonormalize via QR decomposition
    3. Compute SVD of cross-product with U_B to get principal angle cosines
    4. Record mean and max of cosines

    Parameters
    ----------
    U_A : (d_out, k_a) left singular vectors of adapter A (used only for shape)
    U_B : (d_out, k_b) left singular vectors of adapter B
    n_random : number of random trials
    seed : random seed for reproducibility

    Returns
    -------
    dict with keys:
        null_mean_overlaps : list of mean overlap per trial
        null_max_overlaps : list of max overlap per trial
        null_mean_of_means : overall mean of null distribution
        null_std_of_means : std of null distribution
        n_trials : number of trials run
    """
    U_A = U_A.detach().to(dtype=torch.float64, device="cpu")
    U_B = U_B.detach().to(dtype=torch.float64, device="cpu")

    d_out, k_a = U_A.shape
    generator = torch.Generator().manual_seed(seed)

    null_mean_overlaps: List[float] = []
    null_max_overlaps: List[float] = []

    for _ in range(n_random):
        Q_rand = _random_orthonormal(d_out, k_a, generator)
        cosines = _principal_angle_cosines(Q_rand, U_B)

        null_mean_overlaps.append(cosines.mean().item())
        null_max_overlaps.append(cosines.max().item())

    means_t = torch.tensor(null_mean_overlaps, dtype=torch.float64)

    return {
        "null_mean_overlaps": null_mean_overlaps,
        "null_max_overlaps": null_max_overlaps,
        "null_mean_of_means": means_t.mean().item(),
        "null_std_of_means": means_t.std(correction=1).item() if len(null_mean_overlaps) > 1 else 0.0,
        "n_trials": n_random,
    }


def _attempt_derangement(
    n: int, generator: torch.Generator, max_attempts: int = 100
) -> List[int]:
    """Try to produce a derangement (permutation with no fixed points).

    Falls back to a random permutation if a derangement is not found
    within *max_attempts* tries.
    """
    for _ in range(max_attempts):
        perm = torch.randperm(n, generator=generator).tolist()
        if all(perm[i] != i for i in range(n)):
            return perm
    # Fallback: return last random permutation
    return torch.randperm(n, generator=generator).tolist()


def layer_shuffle_control(
    layer_bases_A: List[torch.Tensor],
    layer_bases_B: List[torch.Tensor],
    n_shuffles: int = 10,
    seed: int = 42,
) -> Dict[str, Any]:
    """Compute principal angles between shuffled (cross-layer) pairs.

    For each trial:
    1. Generate a random permutation of layer indices for B
    2. Ensure no layer is paired with its original match (derangement
       attempt, fall back to random perm if derangement fails)
    3. Compute principal angles for each mismatched pair
    4. Record statistics

    Only layers with matching ``d_out`` dimensions are paired. Layers
    whose shuffled partner has a different ``d_out`` are skipped for
    that trial.

    Parameters
    ----------
    layer_bases_A : list of (d_out_i, k_i) tensors, one per shared layer
    layer_bases_B : list of (d_out_i, k_i) tensors, same order as A
    n_shuffles : number of shuffle trials
    seed : random seed

    Returns
    -------
    dict with keys:
        shuffle_mean_overlaps : list of mean overlap per trial (averaged
            across layers)
        shuffle_mean_of_means : overall mean
        shuffle_std_of_means : std
        n_trials : number of trials
        n_layers : number of layers used
    """
    n_layers = len(layer_bases_A)
    if n_layers == 0:
        return {
            "shuffle_mean_overlaps": [],
            "shuffle_mean_of_means": 0.0,
            "shuffle_std_of_means": 0.0,
            "n_trials": 0,
            "n_layers": 0,
        }

    # Convert all to float64 up front
    bases_A = [t.detach().to(dtype=torch.float64, device="cpu") for t in layer_bases_A]
    bases_B = [t.detach().to(dtype=torch.float64, device="cpu") for t in layer_bases_B]

    generator = torch.Generator().manual_seed(seed)
    shuffle_mean_overlaps: List[float] = []

    for _ in range(n_shuffles):
        if n_layers == 1:
            # With a single layer, derangement is impossible; the only
            # permutation is the identity.  We still record a trial so
            # the caller gets a result, but the overlap is the true
            # (non-shuffled) value -- consumers should interpret
            # single-layer results with caution.
            perm = [0]
        else:
            perm = _attempt_derangement(n_layers, generator)

        trial_overlaps: List[float] = []
        for i in range(n_layers):
            j = perm[i]
            U_A_i = bases_A[i]
            U_B_j = bases_B[j]

            # Only pair layers whose d_out dimensions match
            if U_A_i.shape[0] != U_B_j.shape[0]:
                continue

            cosines = _principal_angle_cosines(U_A_i, U_B_j)
            trial_overlaps.append(cosines.mean().item())

        if trial_overlaps:
            shuffle_mean_overlaps.append(
                sum(trial_overlaps) / len(trial_overlaps)
            )

    means_t = torch.tensor(shuffle_mean_overlaps, dtype=torch.float64) if shuffle_mean_overlaps else torch.zeros(0, dtype=torch.float64)

    return {
        "shuffle_mean_overlaps": shuffle_mean_overlaps,
        "shuffle_mean_of_means": means_t.mean().item() if means_t.numel() > 0 else 0.0,
        "shuffle_std_of_means": means_t.std(correction=1).item() if means_t.numel() > 1 else 0.0,
        "n_trials": len(shuffle_mean_overlaps),
        "n_layers": n_layers,
    }
