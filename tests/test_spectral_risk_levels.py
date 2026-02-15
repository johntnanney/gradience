"""
Tests for SpectralAnalyzer risk level classification.

Covers all 5 RiskLevel values plus the insufficient-data edge case,
and verifies get_telemetry() structure.
"""

import time
import pytest

from gradience.spectral import SpectralAnalyzer, SpectralSnapshot, RiskLevel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_snapshot(step: int, kappa: float) -> SpectralSnapshot:
    """Convenience factory: only step and kappa_tilde matter for risk."""
    return SpectralSnapshot(
        step=step,
        timestamp=time.time(),
        sigma_max=1.0,
        sigma_min=0.1,
        kappa_tilde=kappa,
    )


def _add_linear_observations(
    analyzer: SpectralAnalyzer,
    n: int,
    base_kappa: float,
    slope_per_step: float,
) -> None:
    """Add *n* observations whose kappa_tilde rises linearly.

    kappa(i) = base_kappa + slope_per_step * i

    Because the OLS slope is computed over (step, kappa_tilde), feeding
    steps 0..n-1 with this linear relationship yields exactly
    slope_per_step as the regression slope.
    """
    for i in range(n):
        analyzer.add_observation(_make_snapshot(step=i, kappa=base_kappa + slope_per_step * i))


# ---------------------------------------------------------------------------
# Insufficient data
# ---------------------------------------------------------------------------

def test_insufficient_data_returns_stable():
    """With fewer than 3 observations, assess_risk() returns STABLE."""
    analyzer = SpectralAnalyzer()

    # 0 observations
    level, metrics = analyzer.assess_risk()
    assert level is RiskLevel.STABLE
    assert metrics["spectral_slope"] is None
    assert metrics["spectral_volatility"] is None

    # 1 observation
    analyzer.add_observation(_make_snapshot(step=0, kappa=1.0))
    level, metrics = analyzer.assess_risk()
    assert level is RiskLevel.STABLE

    # 2 observations
    analyzer.add_observation(_make_snapshot(step=1, kappa=1.0))
    level, metrics = analyzer.assess_risk()
    assert level is RiskLevel.STABLE


# ---------------------------------------------------------------------------
# STABLE  (abs(slope) <= 0.0005, volatility <= 0.003)
# ---------------------------------------------------------------------------

def test_stable_risk_level():
    """Flat kappa values produce STABLE risk."""
    analyzer = SpectralAnalyzer()

    # All kappa values identical => slope == 0, volatility == 0
    for i in range(10):
        analyzer.add_observation(_make_snapshot(step=i, kappa=1.0))

    level, metrics = analyzer.assess_risk()
    assert level is RiskLevel.STABLE
    assert "spectral_slope" in metrics
    assert "spectral_volatility" in metrics
    assert "spectral_window_size" in metrics
    assert abs(metrics["spectral_slope"]) < 1e-9
    assert metrics["spectral_volatility"] < 1e-9


# ---------------------------------------------------------------------------
# LEARNING  (0.0005 < abs(slope) <= 0.002, volatility <= 0.003)
# ---------------------------------------------------------------------------

def test_learning_risk_level():
    """Moderate positive slope (0.001) triggers LEARNING."""
    analyzer = SpectralAnalyzer()

    # slope_per_step = 0.001 => abs(slope) is between 0.0005 and 0.002
    _add_linear_observations(analyzer, n=10, base_kappa=1.0, slope_per_step=0.001)

    level, metrics = analyzer.assess_risk()
    assert level is RiskLevel.LEARNING
    assert "spectral_slope" in metrics
    assert "spectral_volatility" in metrics
    assert "spectral_window_size" in metrics
    # Confirm the slope is in the LEARNING band
    assert 0.0005 < abs(metrics["spectral_slope"]) <= 0.002


# ---------------------------------------------------------------------------
# WARNING  (0.002 < abs(slope) <= 0.004, volatility <= 0.003)
# ---------------------------------------------------------------------------

def test_warning_risk_level():
    """Slope of 0.003 triggers WARNING.

    With only 3 observations the population std of a linear series
    stays below the default volatility threshold (0.003), so the
    slope-based classification dominates.
    """
    analyzer = SpectralAnalyzer()

    # 3 points: kappa = 1.0, 1.003, 1.006 => slope = 0.003, std ~ 0.00245
    _add_linear_observations(analyzer, n=3, base_kappa=1.0, slope_per_step=0.003)

    level, metrics = analyzer.assess_risk()
    assert level is RiskLevel.WARNING
    assert "spectral_slope" in metrics
    assert "spectral_volatility" in metrics
    assert "spectral_window_size" in metrics
    assert 0.002 < abs(metrics["spectral_slope"]) <= 0.004


# ---------------------------------------------------------------------------
# DANGER  (abs(slope) > 0.004, volatility <= 0.003)
# ---------------------------------------------------------------------------

def test_danger_risk_level():
    """Steep slope (0.005) triggers DANGER.

    For any linear series with n≥3 points, ``std = slope * sqrt((n²-1)/12)``.
    With default ``volatility_threshold=0.003`` and ``slope > 0.004`` this
    always exceeds the volatility gate, pushing the result to INSTABILITY.

    We therefore raise ``volatility_threshold`` so the slope-based
    classification path is exercised in isolation.
    """
    analyzer = SpectralAnalyzer(volatility_threshold=0.1)

    # slope_per_step = 0.005 => abs(slope) > 0.004
    _add_linear_observations(analyzer, n=10, base_kappa=1.0, slope_per_step=0.005)

    level, metrics = analyzer.assess_risk()
    assert level is RiskLevel.DANGER, (
        f"Expected DANGER, got {level} "
        f"(slope={metrics['spectral_slope']:.6f}, "
        f"volatility={metrics['spectral_volatility']:.6f})"
    )
    assert abs(metrics["spectral_slope"]) > 0.004


# ---------------------------------------------------------------------------
# INSTABILITY  (volatility > 0.003, regardless of slope)
# ---------------------------------------------------------------------------

def test_instability_risk_level():
    """High volatility triggers INSTABILITY regardless of slope.

    We alternate kappa between two values far enough apart so that
    the population standard deviation exceeds the 0.003 threshold.

    For values alternating between ``low`` and ``high``:
        std = (high - low) / 2

    Choosing low=0.99, high=1.01 gives std = 0.01 >> 0.003.
    The slope stays near zero because the mean doesn't drift.
    """
    analyzer = SpectralAnalyzer()

    low, high = 0.99, 1.01
    for i in range(20):
        kappa = low if i % 2 == 0 else high
        analyzer.add_observation(_make_snapshot(step=i, kappa=kappa))

    level, metrics = analyzer.assess_risk()
    assert level is RiskLevel.INSTABILITY
    assert "spectral_slope" in metrics
    assert "spectral_volatility" in metrics
    assert "spectral_window_size" in metrics
    assert metrics["spectral_volatility"] > 0.003


# ---------------------------------------------------------------------------
# Telemetry structure
# ---------------------------------------------------------------------------

def test_get_telemetry_structure():
    """get_telemetry() returns dict with expected keys."""
    analyzer = SpectralAnalyzer()

    # With insufficient data
    telemetry = analyzer.get_telemetry()
    assert "spectral_kappa" in telemetry
    assert "risk_level" in telemetry
    assert "spectral_slope" in telemetry
    assert "spectral_volatility" in telemetry
    assert "spectral_window_size" in telemetry
    assert telemetry["risk_level"] == RiskLevel.STABLE.value

    # With sufficient data
    for i in range(5):
        analyzer.add_observation(_make_snapshot(step=i, kappa=1.0 + 0.001 * i))

    telemetry = analyzer.get_telemetry()
    assert telemetry["spectral_kappa"] is not None
    assert isinstance(telemetry["risk_level"], str)
    assert isinstance(telemetry["spectral_slope"], float)
    assert isinstance(telemetry["spectral_volatility"], float)
    assert isinstance(telemetry["spectral_window_size"], int)
