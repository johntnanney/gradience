"""Tests for gradience.vnext.merge.evaluation — merge prediction metrics."""

from __future__ import annotations

import pytest

from gradience.vnext.merge.evaluation import merge_prediction_evaluation

# ---------------------------------------------------------------------------
# Perfect predictions
# ---------------------------------------------------------------------------


class TestPerfectPredictions:
    """When predictions perfectly separate classes, AUC = 1.0, AP = 1.0."""

    def test_auc_and_ap_are_perfect(self):
        # Bad merges get high risk, good merges get low risk
        predicted_risk = [0.1, 0.2, 0.15, 0.9, 0.85, 0.95]
        actual_bad = [False, False, False, True, True, True]
        actual_Q = [0.98, 0.97, 0.99, 0.7, 0.6, 0.5]

        result = merge_prediction_evaluation(predicted_risk, actual_bad, actual_Q)

        assert result["auc_roc"] == pytest.approx(1.0)
        assert result["average_precision"] == pytest.approx(1.0)

    def test_counts(self):
        predicted_risk = [0.1, 0.2, 0.9, 0.85]
        actual_bad = [False, False, True, True]
        actual_Q = [0.98, 0.97, 0.7, 0.6]

        result = merge_prediction_evaluation(predicted_risk, actual_bad, actual_Q)

        assert result["n_samples"] == 4
        assert result["n_bad"] == 2
        assert result["n_good"] == 2


# ---------------------------------------------------------------------------
# Random / uninformative predictions
# ---------------------------------------------------------------------------


class TestRandomPredictions:
    """With large random data, AUC should be near 0.5."""

    def test_auc_near_half(self):
        import random

        random.seed(42)
        n = 1000
        predicted_risk = [random.random() for _ in range(n)]
        actual_bad = [random.random() > 0.5 for _ in range(n)]
        actual_Q = [random.uniform(0.5, 1.0) for _ in range(n)]

        result = merge_prediction_evaluation(predicted_risk, actual_bad, actual_Q)

        # With random predictions, AUC should be near 0.5 (within +-0.1)
        assert result["auc_roc"] is not None
        assert abs(result["auc_roc"] - 0.5) < 0.1


# ---------------------------------------------------------------------------
# All same class (AUC undefined)
# ---------------------------------------------------------------------------


class TestAllSameClass:
    """When all labels are the same, AUC and AP should be None."""

    def test_all_good(self):
        predicted_risk = [0.1, 0.2, 0.3, 0.4]
        actual_bad = [False, False, False, False]
        actual_Q = [0.98, 0.97, 0.96, 0.95]

        result = merge_prediction_evaluation(predicted_risk, actual_bad, actual_Q)

        assert result["auc_roc"] is None
        assert result["average_precision"] is None
        assert result["n_bad"] == 0
        assert result["n_good"] == 4

    def test_all_bad(self):
        predicted_risk = [0.9, 0.8, 0.7, 0.6]
        actual_bad = [True, True, True, True]
        actual_Q = [0.7, 0.6, 0.5, 0.4]

        result = merge_prediction_evaluation(predicted_risk, actual_bad, actual_Q)

        assert result["auc_roc"] is None
        assert result["average_precision"] is None
        assert result["n_bad"] == 4
        assert result["n_good"] == 0


# ---------------------------------------------------------------------------
# R² computation
# ---------------------------------------------------------------------------


class TestRSquared:
    """Tests for linear regression R² on Q_min prediction."""

    def test_perfect_linear_relationship(self):
        # Q_min = 1.0 - predicted_risk (perfect negative linear)
        predicted_risk = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        actual_bad = [False, False, False, True, True, True]
        actual_Q = [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]

        result = merge_prediction_evaluation(predicted_risk, actual_bad, actual_Q)

        assert result["r_squared_Q_min"] is not None
        assert result["r_squared_Q_min"] == pytest.approx(1.0, abs=1e-10)

    def test_no_linear_relationship(self):
        # Q_min values are constant -> R² should be None (std=0)
        predicted_risk = [0.1, 0.5, 0.9]
        actual_bad = [False, True, True]
        actual_Q = [0.8, 0.8, 0.8]

        result = merge_prediction_evaluation(predicted_risk, actual_bad, actual_Q)

        assert result["r_squared_Q_min"] is None

    def test_moderate_r_squared(self):
        # Noisy linear relationship
        predicted_risk = [0.1, 0.3, 0.5, 0.7, 0.9]
        actual_bad = [False, False, True, True, True]
        # Roughly decreasing but noisy
        actual_Q = [0.95, 0.80, 0.70, 0.55, 0.30]

        result = merge_prediction_evaluation(predicted_risk, actual_bad, actual_Q)

        assert result["r_squared_Q_min"] is not None
        # Should have a decent fit
        assert result["r_squared_Q_min"] > 0.8


# ---------------------------------------------------------------------------
# Calibration bins
# ---------------------------------------------------------------------------


class TestCalibration:
    """Tests for binned calibration analysis."""

    def test_calibration_structure(self):
        predicted_risk = [0.1, 0.3, 0.5, 0.7, 0.9]
        actual_bad = [False, False, True, True, True]
        actual_Q = [0.95, 0.85, 0.75, 0.55, 0.35]

        result = merge_prediction_evaluation(
            predicted_risk,
            actual_bad,
            actual_Q,
            n_calibration_bins=5,
        )

        cal = result["calibration"]
        assert "bin_edges" in cal
        assert "mean_predicted" in cal
        assert "mean_observed" in cal
        assert "bin_counts" in cal
        assert len(cal["bin_edges"]) == 5
        assert len(cal["mean_predicted"]) == 5
        assert len(cal["mean_observed"]) == 5
        assert len(cal["bin_counts"]) == 5

    def test_bin_edges_cover_unit_interval(self):
        predicted_risk = [0.1, 0.5, 0.9]
        actual_bad = [False, True, True]
        actual_Q = [0.95, 0.75, 0.35]

        result = merge_prediction_evaluation(
            predicted_risk,
            actual_bad,
            actual_Q,
            n_calibration_bins=3,
        )

        cal = result["calibration"]
        # First bin starts at 0, last bin ends at 1
        assert cal["bin_edges"][0][0] == pytest.approx(0.0)
        assert cal["bin_edges"][-1][1] == pytest.approx(1.0)

    def test_bin_counts_sum_to_n(self):
        predicted_risk = [0.1, 0.3, 0.5, 0.7, 0.9]
        actual_bad = [False, False, True, True, True]
        actual_Q = [0.95, 0.85, 0.75, 0.55, 0.35]

        result = merge_prediction_evaluation(
            predicted_risk,
            actual_bad,
            actual_Q,
            n_calibration_bins=5,
        )

        cal = result["calibration"]
        assert sum(cal["bin_counts"]) == 5

    def test_fewer_samples_than_bins(self):
        """When n_samples < n_calibration_bins, bins are reduced to n_samples."""
        predicted_risk = [0.3, 0.7]
        actual_bad = [False, True]
        actual_Q = [0.9, 0.6]

        result = merge_prediction_evaluation(
            predicted_risk,
            actual_bad,
            actual_Q,
            n_calibration_bins=10,
        )

        cal = result["calibration"]
        # Should use min(10, 2) = 2 bins
        assert len(cal["bin_edges"]) == 2


# ---------------------------------------------------------------------------
# Empty inputs
# ---------------------------------------------------------------------------


class TestEmptyInputs:
    """Empty input lists should return safe defaults."""

    def test_empty_returns_none_metrics(self):
        result = merge_prediction_evaluation([], [], [])

        assert result["auc_roc"] is None
        assert result["average_precision"] is None
        assert result["r_squared_Q_min"] is None
        assert result["n_samples"] == 0
        assert result["n_bad"] == 0
        assert result["n_good"] == 0

    def test_empty_calibration_is_empty(self):
        result = merge_prediction_evaluation([], [], [])

        cal = result["calibration"]
        assert cal["bin_edges"] == []
        assert cal["mean_predicted"] == []
        assert cal["mean_observed"] == []
        assert cal["bin_counts"] == []


# ---------------------------------------------------------------------------
# Sample counts
# ---------------------------------------------------------------------------


class TestSampleCounts:
    """n_samples, n_bad, n_good must be accurate."""

    def test_mixed(self):
        predicted_risk = [0.1, 0.2, 0.8, 0.9, 0.95]
        actual_bad = [False, False, True, True, True]
        actual_Q = [0.98, 0.97, 0.6, 0.5, 0.4]

        result = merge_prediction_evaluation(predicted_risk, actual_bad, actual_Q)

        assert result["n_samples"] == 5
        assert result["n_bad"] == 3
        assert result["n_good"] == 2

    def test_single_sample(self):
        result = merge_prediction_evaluation([0.5], [True], [0.7])

        assert result["n_samples"] == 1
        assert result["n_bad"] == 1
        assert result["n_good"] == 0
        # Single sample: AUC undefined (only one class)
        assert result["auc_roc"] is None


# ---------------------------------------------------------------------------
# Length mismatch
# ---------------------------------------------------------------------------


class TestLengthMismatch:
    """Mismatched input lengths should raise ValueError."""

    def test_raises_on_mismatch(self):
        with pytest.raises(ValueError, match="Length mismatch"):
            merge_prediction_evaluation([0.1, 0.2], [True], [0.9, 0.8])
