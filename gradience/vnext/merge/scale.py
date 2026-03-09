"""
Symmetric scale metrics for comparing adapter magnitudes.

Replaces asymmetric sigma_max(A)/sigma_max(B) ratios with:
- bounded_ratio: min/max, range [0, 1], 1 = perfectly balanced
- log_ratio: log(sigma_A) - log(sigma_B), range (-inf, inf), 0 = balanced

These metrics are symmetric in the sense that swapping A and B produces
related (not identical) values: bounded_ratio is invariant, log_ratio flips sign.
"""

from __future__ import annotations

import math
from typing import Any, Dict


def _bounded_and_log(
    val_a: float,
    val_b: float,
    eps: float,
) -> tuple[float, float]:
    """Shared logic for computing bounded ratio and log ratio.

    Takes absolute values first, then handles zero edge cases.
    """
    a = abs(val_a)
    b = abs(val_b)

    # If both are effectively zero, the adapters are identical in this metric.
    if a < eps and b < eps:
        return 0.0, 0.0

    # If only one is effectively zero, ratio collapses.
    if a < eps or b < eps:
        return 0.0, math.copysign(math.inf, math.log(a + eps) - math.log(b + eps))

    lo, hi = sorted((a, b))
    bounded_ratio = lo / hi
    log_ratio = math.log(a) - math.log(b)
    return bounded_ratio, log_ratio


def symmetric_scale_metrics(
    sigma_max_A: float,
    sigma_max_B: float,
    *,
    eps: float = 1e-12,
) -> dict[str, float]:
    """Compute symmetric scale comparison metrics.

    Parameters
    ----------
    sigma_max_A : largest singular value of adapter A's delta-W
    sigma_max_B : largest singular value of adapter B's delta-W
    eps : small constant for numerical stability

    Returns
    -------
    dict with keys:
        scale_bounded_ratio : min(|sigma_A|, |sigma_B|) / max(|sigma_A|, |sigma_B|), range [0, 1]
        scale_log_ratio : log(|sigma_A|) - log(|sigma_B|), range (-inf, inf)
    """
    bounded, log_r = _bounded_and_log(sigma_max_A, sigma_max_B, eps)
    return {
        "scale_bounded_ratio": bounded,
        "scale_log_ratio": log_r,
    }


def symmetric_frobenius_metrics(
    frob_A: float,
    frob_B: float,
    *,
    eps: float = 1e-12,
) -> dict[str, float]:
    """Same as symmetric_scale_metrics but for Frobenius norms.

    Parameters
    ----------
    frob_A : Frobenius norm of adapter A's delta-W
    frob_B : Frobenius norm of adapter B's delta-W
    eps : small constant for numerical stability

    Returns
    -------
    dict with keys:
        frob_bounded_ratio : min/max
        frob_log_ratio : log difference
    """
    bounded, log_r = _bounded_and_log(frob_A, frob_B, eps)
    return {
        "frob_bounded_ratio": bounded,
        "frob_log_ratio": log_r,
    }
