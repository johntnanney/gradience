from __future__ import annotations

import math

from gradience.research.spectral_extended import (
    compute_spectral_edge_gap,
    fit_htsr_tail_exponent,
)


def test_fit_htsr_tail_exponent_recovers_power_law() -> None:
    alpha_true = 1.5
    singular_values = [1.0 / ((i + 1) ** alpha_true) for i in range(64)]
    estimate = fit_htsr_tail_exponent(
        singular_values,
        tail_fractions=(0.5, 0.6, 0.7),
        min_tail_points=8,
        min_fit_r2=0.8,
    )

    assert estimate.valid
    assert estimate.alpha is not None
    assert estimate.fit_r2 is not None
    assert estimate.fit_r2 > 0.95
    assert math.isclose(estimate.alpha, alpha_true, rel_tol=0.2)


def test_fit_htsr_tail_exponent_flags_short_spectrum() -> None:
    estimate = fit_htsr_tail_exponent([4.0, 2.0, 1.0], min_tail_points=5)

    assert not estimate.valid
    assert estimate.alpha is None
    assert estimate.warning == "insufficient_spectrum_length"


def test_compute_spectral_edge_gap() -> None:
    probe = compute_spectral_edge_gap([10.0, 5.0, 2.0, 1.0])

    assert probe.valid
    assert probe.edge_gap_12 is not None
    assert probe.edge_gap_13 is not None
    assert probe.sigma1_share_top3 is not None
    assert math.isclose(probe.edge_gap_12, 2.0, rel_tol=1e-8)
    assert math.isclose(probe.edge_gap_13, 5.0, rel_tol=1e-8)
    assert math.isclose(probe.sigma1_share_top3, 10.0 / 17.0, rel_tol=1e-8)


def test_compute_spectral_edge_gap_invalid_short() -> None:
    probe = compute_spectral_edge_gap([3.0])
    assert not probe.valid
    assert probe.edge_gap_12 is None
    assert probe.warning == "insufficient_spectrum_length"
