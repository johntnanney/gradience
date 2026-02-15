"""
Statistical utilities for bench reporting.

Provides 95% confidence intervals (t-distribution) and Cohen's d effect size
for multi-seed aggregation. All functions handle small sample sizes gracefully.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Any

import numpy as np


# t-distribution critical values for 95% CI (two-tailed)
# Key: degrees of freedom (n-1), Value: t critical value
# For n >= 30, use z = 1.96
_T_CRIT_95 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
    6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
    11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
    16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
    25: 2.060, 29: 2.045,
}


def _t_critical(df: int) -> float:
    """Get t critical value for 95% two-tailed CI."""
    if df <= 0:
        return float("inf")
    if df in _T_CRIT_95:
        return _T_CRIT_95[df]
    # Linear interpolation for missing values, z-approx for large df
    if df >= 30:
        return 1.96
    # Find nearest keys
    keys = sorted(_T_CRIT_95.keys())
    for i, k in enumerate(keys):
        if k > df:
            lo, hi = keys[i - 1], k
            frac = (df - lo) / (hi - lo)
            return _T_CRIT_95[lo] + frac * (_T_CRIT_95[hi] - _T_CRIT_95[lo])
    return 1.96


def confidence_interval_95(values: List[float]) -> Optional[Dict[str, float]]:
    """Compute 95% confidence interval using t-distribution.

    Returns None if fewer than 2 values (CI undefined for n=1).
    """
    n = len(values)
    if n < 2:
        return None

    arr = np.array(values, dtype=np.float64)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1))
    se = std / math.sqrt(n)
    t_crit = _t_critical(n - 1)
    margin = t_crit * se

    return {
        "mean": mean,
        "ci_lower": mean - margin,
        "ci_upper": mean + margin,
        "margin": margin,
        "se": se,
        "n": n,
    }


def cohens_d(group1: List[float], group2: List[float]) -> Optional[float]:
    """Compute Cohen's d effect size between two groups.

    Uses pooled standard deviation. Returns None if either group has < 2 values.
    """
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return None

    a1 = np.array(group1, dtype=np.float64)
    a2 = np.array(group2, dtype=np.float64)

    var1 = float(a1.var(ddof=1))
    var2 = float(a2.var(ddof=1))

    pooled_std = math.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0

    return float((a1.mean() - a2.mean()) / pooled_std)


def cohens_d_one_sample(values: List[float], mu: float = 0.0) -> Optional[float]:
    """Compute one-sample Cohen's d (effect size vs a reference value).

    Useful for measuring how far delta_vs_probe deviates from zero.
    Returns None if fewer than 2 values.
    """
    if len(values) < 2:
        return None

    arr = np.array(values, dtype=np.float64)
    std = float(arr.std(ddof=1))
    if std == 0:
        return 0.0

    return float((arr.mean() - mu) / std)


def augment_stats_with_ci(
    values: List[float],
    existing_stats: Dict[str, Any],
    prefix: str = "",
) -> Dict[str, Any]:
    """Add CI fields to an existing stats dict.

    Mutates and returns existing_stats for convenience.
    If prefix is provided, keys are prefixed (e.g., "delta_ci_lower").
    """
    ci = confidence_interval_95(values)
    if ci is None:
        return existing_stats

    p = f"{prefix}_" if prefix else ""
    existing_stats[f"{p}ci_lower"] = ci["ci_lower"]
    existing_stats[f"{p}ci_upper"] = ci["ci_upper"]
    existing_stats[f"{p}ci_margin"] = ci["margin"]
    existing_stats[f"{p}se"] = ci["se"]

    return existing_stats
