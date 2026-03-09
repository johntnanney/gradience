"""
Merge outcome metrics: Q_min (retention-min) and D (dominance index).

Given solo and merged evaluation scores for two adapters, these metrics
quantify how much capability is preserved (Q_min) and how balanced the
merge is across tasks (D).
"""

from __future__ import annotations

from typing import Any


def compute_merge_outcomes(
    score_A_solo: float,
    score_B_solo: float,
    score_A_merged: float,
    score_B_merged: float,
    *,
    eps: float = 1e-12,
) -> dict[str, Any]:
    """Compute merge outcome metrics.

    Parameters
    ----------
    score_A_solo : evaluation score for adapter A alone on task A
    score_B_solo : evaluation score for adapter B alone on task B
    score_A_merged : evaluation score for merged adapter on task A
    score_B_merged : evaluation score for merged adapter on task B
    eps : small constant to avoid division by zero

    Returns
    -------
    dict with keys:
        retention_A : score_A_merged / score_A_solo (clamped to [0, inf))
        retention_B : score_B_merged / score_B_solo
        Q_min : min(retention_A, retention_B) -- worst-case retention
        D : abs(score_A_merged - score_B_merged) / (score_A_merged + score_B_merged + eps) -- dominance index
        is_bad_merge : bool -- True if Q_min < 0.95 or D > 0.2
    """
    retention_A = score_A_merged / max(score_A_solo, eps)
    retention_B = score_B_merged / max(score_B_solo, eps)
    Q_min = min(retention_A, retention_B)

    denom = score_A_merged + score_B_merged + eps
    D = abs(score_A_merged - score_B_merged) / denom

    return {
        "retention_A": retention_A,
        "retention_B": retention_B,
        "Q_min": Q_min,
        "D": D,
        "is_bad_merge": is_bad_merge(Q_min, D),
    }


def is_bad_merge(
    Q_min: float,
    D: float,
    *,
    Q_min_threshold: float = 0.95,
    D_threshold: float = 0.2,
) -> bool:
    """Classify a merge as bad based on outcome thresholds.

    Parameters
    ----------
    Q_min : retention-min value
    D : dominance index value
    Q_min_threshold : minimum acceptable retention (default 0.95)
    D_threshold : maximum acceptable dominance (default 0.2)

    Returns
    -------
    True if the merge is classified as bad (Q_min below threshold OR D above threshold).
    """
    return Q_min < Q_min_threshold or D_threshold < D
