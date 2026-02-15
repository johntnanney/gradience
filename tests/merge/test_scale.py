"""Tests for symmetric scale metrics."""

from __future__ import annotations

import math

import pytest

from gradience.vnext.merge.scale import (
    symmetric_frobenius_metrics,
    symmetric_scale_metrics,
)


# ---------------------------------------------------------------------------
# symmetric_scale_metrics
# ---------------------------------------------------------------------------


class TestSymmetricScaleMetrics:
    """Tests for symmetric_scale_metrics."""

    def test_equal_values(self):
        result = symmetric_scale_metrics(5.0, 5.0)
        assert result["scale_bounded_ratio"] == pytest.approx(1.0)
        assert result["scale_log_ratio"] == pytest.approx(0.0)

    def test_two_to_one_ratio(self):
        result = symmetric_scale_metrics(2.0, 1.0)
        assert result["scale_bounded_ratio"] == pytest.approx(0.5)
        assert result["scale_log_ratio"] == pytest.approx(math.log(2))

    def test_one_to_two_ratio(self):
        result = symmetric_scale_metrics(1.0, 2.0)
        assert result["scale_bounded_ratio"] == pytest.approx(0.5)
        assert result["scale_log_ratio"] == pytest.approx(-math.log(2))

    def test_both_zero(self):
        result = symmetric_scale_metrics(0.0, 0.0)
        assert result["scale_bounded_ratio"] == 0.0
        assert result["scale_log_ratio"] == 0.0

    def test_one_zero(self):
        result = symmetric_scale_metrics(3.0, 0.0)
        assert result["scale_bounded_ratio"] == 0.0
        assert math.isinf(result["scale_log_ratio"])
        assert result["scale_log_ratio"] > 0  # log(3) - log(eps) > 0

    def test_zero_first(self):
        result = symmetric_scale_metrics(0.0, 3.0)
        assert result["scale_bounded_ratio"] == 0.0
        assert math.isinf(result["scale_log_ratio"])
        assert result["scale_log_ratio"] < 0  # log(eps) - log(3) < 0

    def test_negative_values_take_abs(self):
        pos_result = symmetric_scale_metrics(4.0, 2.0)
        neg_result = symmetric_scale_metrics(-4.0, -2.0)
        assert neg_result["scale_bounded_ratio"] == pytest.approx(
            pos_result["scale_bounded_ratio"]
        )
        assert neg_result["scale_log_ratio"] == pytest.approx(
            pos_result["scale_log_ratio"]
        )

    def test_symmetry_bounded_ratio_invariant(self):
        a, b = 7.0, 3.0
        r_ab = symmetric_scale_metrics(a, b)
        r_ba = symmetric_scale_metrics(b, a)
        assert r_ab["scale_bounded_ratio"] == pytest.approx(
            r_ba["scale_bounded_ratio"]
        )

    def test_symmetry_log_ratio_flips_sign(self):
        a, b = 7.0, 3.0
        r_ab = symmetric_scale_metrics(a, b)
        r_ba = symmetric_scale_metrics(b, a)
        assert r_ab["scale_log_ratio"] == pytest.approx(-r_ba["scale_log_ratio"])


# ---------------------------------------------------------------------------
# symmetric_frobenius_metrics
# ---------------------------------------------------------------------------


class TestSymmetricFrobeniusMetrics:
    """Tests for symmetric_frobenius_metrics."""

    def test_equal_values(self):
        result = symmetric_frobenius_metrics(10.0, 10.0)
        assert result["frob_bounded_ratio"] == pytest.approx(1.0)
        assert result["frob_log_ratio"] == pytest.approx(0.0)

    def test_two_to_one_ratio(self):
        result = symmetric_frobenius_metrics(2.0, 1.0)
        assert result["frob_bounded_ratio"] == pytest.approx(0.5)
        assert result["frob_log_ratio"] == pytest.approx(math.log(2))

    def test_one_to_two_ratio(self):
        result = symmetric_frobenius_metrics(1.0, 2.0)
        assert result["frob_bounded_ratio"] == pytest.approx(0.5)
        assert result["frob_log_ratio"] == pytest.approx(-math.log(2))

    def test_both_zero(self):
        result = symmetric_frobenius_metrics(0.0, 0.0)
        assert result["frob_bounded_ratio"] == 0.0
        assert result["frob_log_ratio"] == 0.0

    def test_one_zero(self):
        result = symmetric_frobenius_metrics(5.0, 0.0)
        assert result["frob_bounded_ratio"] == 0.0
        assert math.isinf(result["frob_log_ratio"])
        assert result["frob_log_ratio"] > 0

    def test_negative_values_take_abs(self):
        pos = symmetric_frobenius_metrics(6.0, 3.0)
        neg = symmetric_frobenius_metrics(-6.0, -3.0)
        assert neg["frob_bounded_ratio"] == pytest.approx(pos["frob_bounded_ratio"])
        assert neg["frob_log_ratio"] == pytest.approx(pos["frob_log_ratio"])

    def test_symmetry_bounded_ratio_invariant(self):
        a, b = 8.0, 2.0
        r_ab = symmetric_frobenius_metrics(a, b)
        r_ba = symmetric_frobenius_metrics(b, a)
        assert r_ab["frob_bounded_ratio"] == pytest.approx(r_ba["frob_bounded_ratio"])

    def test_symmetry_log_ratio_flips_sign(self):
        a, b = 8.0, 2.0
        r_ab = symmetric_frobenius_metrics(a, b)
        r_ba = symmetric_frobenius_metrics(b, a)
        assert r_ab["frob_log_ratio"] == pytest.approx(-r_ba["frob_log_ratio"])
