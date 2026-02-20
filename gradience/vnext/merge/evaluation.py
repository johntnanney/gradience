"""
Evaluation metrics for merge outcome prediction.

Given predicted risk scores (from spectral metrics) and actual outcomes
(Q_min, D, is_bad_merge), compute classification and regression metrics.

Classification: AUC-ROC, Average Precision for detecting bad merges
Regression: R² on Q_min prediction
Calibration: binned reliability analysis
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from gradience.exceptions import MergeError

# Use sklearn when available for classification metrics, but fall back
# to pure-numpy implementations so the module works without it.

try:
    from sklearn.metrics import average_precision_score, roc_auc_score

    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False


# ---------------------------------------------------------------------------
# Pure-numpy fallbacks
# ---------------------------------------------------------------------------


def _auc_roc_numpy(
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> float:
    """Compute AUC-ROC using the trapezoidal rule (no sklearn needed).

    Sorts by descending score, walks thresholds to build ROC curve,
    then integrates via the trapezoidal rule.
    """
    # Sort by descending score, break ties by putting positives first
    desc_order = np.lexsort((y_true, -y_score))
    y_true_sorted = y_true[desc_order].astype(float)

    n_pos = y_true_sorted.sum()
    n_neg = len(y_true_sorted) - n_pos

    # Cumulative TP and FP as we lower the threshold
    tps = np.cumsum(y_true_sorted)
    fps = np.cumsum(1.0 - y_true_sorted)

    tpr = tps / n_pos
    fpr = fps / n_neg

    # Prepend the origin (0, 0)
    tpr = np.concatenate([[0.0], tpr])
    fpr = np.concatenate([[0.0], fpr])

    # Trapezoidal integration (np.trapezoid in numpy >=2, np.trapz before)
    _trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")
    return float(_trapz(tpr, fpr))


def _average_precision_numpy(
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> float:
    """Compute Average Precision using step-function integration (no sklearn needed).

    AP = sum_n (R_n - R_{n-1}) * P_n  where n indexes thresholds in
    descending score order.
    """
    desc_order = np.lexsort((y_true, -y_score))
    y_true_sorted = y_true[desc_order].astype(float)

    n_pos = y_true_sorted.sum()

    tps = np.cumsum(y_true_sorted)
    precision = tps / np.arange(1, len(y_true_sorted) + 1)
    recall = tps / n_pos

    # Prepend recall = 0 for the step-function integral
    recall_with_zero = np.concatenate([[0.0], recall])
    delta_recall = np.diff(recall_with_zero)

    return float(np.sum(precision * delta_recall))


# ---------------------------------------------------------------------------
# R² via numpy linear regression
# ---------------------------------------------------------------------------


def _compute_r_squared(
    predicted: List[float],
    actual: List[float],
) -> Optional[float]:
    """Compute R² of linear regression: actual ~ predicted.

    Uses the normal equation (closed-form OLS) with numpy.
    Returns None if the data is degenerate (e.g. constant actual values
    or fewer than 2 samples).
    """
    x = np.array(predicted, dtype=float)
    y = np.array(actual, dtype=float)

    # Need at least 2 samples for meaningful regression
    if len(y) < 2:
        return None

    # If actual values are all the same, R² is undefined
    if np.std(y) < 1e-12:
        return None

    # OLS: y = a*x + b
    x_mean = np.mean(x)
    y_mean = np.mean(y)

    ss_xy = np.sum((x - x_mean) * (y - y_mean))
    ss_xx = np.sum((x - x_mean) ** 2)

    if ss_xx == 0:
        return None

    slope = ss_xy / ss_xx
    intercept = y_mean - slope * x_mean

    y_pred = slope * x + intercept
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y_mean) ** 2)

    return float(1.0 - ss_res / ss_tot)


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


def _compute_calibration(
    predicted_risk: List[float],
    actual_bad_merge: List[bool],
    n_bins: int,
) -> Dict[str, Any]:
    """Compute binned calibration (reliability) analysis.

    Divides the predicted risk range [0, 1] into equal-width bins and
    computes mean predicted risk and mean observed rate per bin.

    Returns
    -------
    dict with keys:
        bin_edges : list of (low, high) tuples for each bin
        mean_predicted : list of mean predicted risk per bin (None if bin empty)
        mean_observed : list of fraction actually bad per bin (None if bin empty)
        bin_counts : list of sample counts per bin
    """
    pred = np.array(predicted_risk, dtype=float)
    actual = np.array(actual_bad_merge, dtype=float)

    bin_edges: List[tuple[float, float]] = []
    mean_predicted: List[Optional[float]] = []
    mean_observed: List[Optional[float]] = []
    bin_counts: List[int] = []

    lo = 0.0
    step = 1.0 / n_bins

    for i in range(n_bins):
        hi = lo + step
        # Last bin includes the right edge
        if i == n_bins - 1:
            mask = (pred >= lo) & (pred <= hi)
        else:
            mask = (pred >= lo) & (pred < hi)

        count = int(mask.sum())
        bin_edges.append((round(lo, 10), round(hi, 10)))
        bin_counts.append(count)

        if count == 0:
            mean_predicted.append(None)
            mean_observed.append(None)
        else:
            mean_predicted.append(float(np.mean(pred[mask])))
            mean_observed.append(float(np.mean(actual[mask])))

        lo = hi

    return {
        "bin_edges": bin_edges,
        "mean_predicted": mean_predicted,
        "mean_observed": mean_observed,
        "bin_counts": bin_counts,
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def merge_prediction_evaluation(
    predicted_risk: List[float],
    actual_bad_merge: List[bool],
    actual_Q_min: List[float],
    *,
    n_calibration_bins: int = 5,
) -> Dict[str, Any]:
    """Compute evaluation metrics for merge outcome prediction.

    Parameters
    ----------
    predicted_risk : predicted probability/score that merge is bad (higher = worse)
        e.g. 1 - compatibility_score or a composite of spectral metrics
    actual_bad_merge : ground-truth binary labels (True = bad merge)
    actual_Q_min : ground-truth Q_min values for regression evaluation
    n_calibration_bins : number of bins for calibration analysis

    Returns
    -------
    dict with keys:
        auc_roc : Area Under ROC Curve (float or None)
        average_precision : Average Precision score (float or None)
        r_squared_Q_min : R² of linear regression: Q_min ~ predicted_risk
        calibration : dict with bin edges, mean predicted, mean observed
        n_samples : total number of samples
        n_bad : number of bad merges
        n_good : number of good merges

    Classification metrics are None when only one class is present (AUC
    undefined).  Uses sklearn when available, otherwise falls back to
    pure-numpy implementations.
    """
    n_samples = len(predicted_risk)

    if n_samples == 0:
        return {
            "auc_roc": None,
            "average_precision": None,
            "r_squared_Q_min": None,
            "calibration": {
                "bin_edges": [],
                "mean_predicted": [],
                "mean_observed": [],
                "bin_counts": [],
            },
            "n_samples": 0,
            "n_bad": 0,
            "n_good": 0,
        }

    if len(actual_bad_merge) != n_samples or len(actual_Q_min) != n_samples:
        raise MergeError(
            f"Length mismatch: predicted_risk has {n_samples} elements, "
            f"actual_bad_merge has {len(actual_bad_merge)}, "
            f"actual_Q_min has {len(actual_Q_min)}"
        )

    n_bad = sum(1 for b in actual_bad_merge if b)
    n_good = n_samples - n_bad

    # --- Classification metrics ---
    auc_roc: Optional[float] = None
    average_precision: Optional[float] = None

    # AUC-ROC and AP require both classes to be present
    if n_bad > 0 and n_good > 0:
        y_true = np.array(actual_bad_merge, dtype=float)
        y_score = np.array(predicted_risk, dtype=float)

        if _HAS_SKLEARN:
            try:
                auc_roc = float(roc_auc_score(y_true, y_score))
            except ValueError:
                auc_roc = None
            try:
                average_precision = float(
                    average_precision_score(y_true, y_score)
                )
            except ValueError:
                average_precision = None
        else:
            auc_roc = _auc_roc_numpy(y_true, y_score)
            average_precision = _average_precision_numpy(y_true, y_score)

    # --- Regression ---
    r_squared = _compute_r_squared(predicted_risk, actual_Q_min)

    # --- Calibration ---
    effective_bins = min(n_calibration_bins, n_samples)
    calibration = _compute_calibration(
        predicted_risk, actual_bad_merge, effective_bins,
    )

    return {
        "auc_roc": auc_roc,
        "average_precision": average_precision,
        "r_squared_Q_min": r_squared,
        "calibration": calibration,
        "n_samples": n_samples,
        "n_bad": n_bad,
        "n_good": n_good,
    }
