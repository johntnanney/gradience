"""Tests for gradience.vnext.merge.outcomes -- merge outcome metrics."""

from __future__ import annotations

import pytest

from gradience.vnext.merge.outcomes import compute_merge_outcomes, is_bad_merge


# ---------------------------------------------------------------------------
# compute_merge_outcomes -- basic computation
# ---------------------------------------------------------------------------


class TestComputeMergeOutcomes:
    """Tests for the compute_merge_outcomes function."""

    def test_basic_known_values(self):
        """Verify computation with hand-calculated values."""
        result = compute_merge_outcomes(
            score_A_solo=0.80,
            score_B_solo=0.90,
            score_A_merged=0.76,
            score_B_merged=0.81,
        )

        assert result["retention_A"] == pytest.approx(0.76 / 0.80)
        assert result["retention_B"] == pytest.approx(0.81 / 0.90)
        assert result["Q_min"] == pytest.approx(0.81 / 0.90)  # B is worse
        expected_D = abs(0.76 - 0.81) / (0.76 + 0.81 + 1e-12)
        assert result["D"] == pytest.approx(expected_D)

    def test_retention_one_when_merged_equals_solo(self):
        """When merged scores match solo scores, retention = 1.0."""
        result = compute_merge_outcomes(
            score_A_solo=0.85,
            score_B_solo=0.90,
            score_A_merged=0.85,
            score_B_merged=0.90,
        )

        assert result["retention_A"] == pytest.approx(1.0)
        assert result["retention_B"] == pytest.approx(1.0)
        assert result["Q_min"] == pytest.approx(1.0)

    def test_Q_min_picks_worse_retention(self):
        """Q_min should equal the lower of the two retention ratios."""
        result = compute_merge_outcomes(
            score_A_solo=1.0,
            score_B_solo=1.0,
            score_A_merged=0.90,   # retention_A = 0.90
            score_B_merged=0.70,   # retention_B = 0.70  (worse)
        )

        assert result["retention_A"] == pytest.approx(0.90)
        assert result["retention_B"] == pytest.approx(0.70)
        assert result["Q_min"] == pytest.approx(0.70)

    def test_D_zero_when_merged_scores_equal(self):
        """Dominance index D = 0 when both merged scores are identical."""
        result = compute_merge_outcomes(
            score_A_solo=0.80,
            score_B_solo=0.90,
            score_A_merged=0.75,
            score_B_merged=0.75,
        )

        assert result["D"] == pytest.approx(0.0, abs=1e-10)

    def test_D_approaches_one_when_one_score_dominates(self):
        """D should be close to 1 when one merged score dwarfs the other."""
        result = compute_merge_outcomes(
            score_A_solo=0.90,
            score_B_solo=0.90,
            score_A_merged=0.90,
            score_B_merged=0.01,
        )

        # D = |0.90 - 0.01| / (0.90 + 0.01 + eps) ~ 0.89 / 0.91 ~ 0.978
        assert result["D"] > 0.9

    def test_result_keys(self):
        """Return dict contains all expected keys."""
        result = compute_merge_outcomes(0.8, 0.9, 0.7, 0.8)
        expected_keys = {"retention_A", "retention_B", "Q_min", "D", "is_bad_merge"}
        assert set(result.keys()) == expected_keys

    def test_is_bad_merge_flag_good_merge(self):
        """A merge with high retention and low dominance is not bad."""
        result = compute_merge_outcomes(
            score_A_solo=0.80,
            score_B_solo=0.80,
            score_A_merged=0.79,
            score_B_merged=0.78,
        )

        # retention_A = 0.9875, retention_B = 0.975 -> Q_min = 0.975 (>0.95)
        # D = |0.79-0.78| / (0.79+0.78+eps) ~ 0.00637 (<0.2)
        assert result["is_bad_merge"] is False

    def test_is_bad_merge_flag_low_retention(self):
        """A merge with low retention is flagged as bad."""
        result = compute_merge_outcomes(
            score_A_solo=1.0,
            score_B_solo=1.0,
            score_A_merged=0.50,
            score_B_merged=0.50,
        )

        # Q_min = 0.50 < 0.95
        assert result["is_bad_merge"] is True

    def test_is_bad_merge_flag_high_dominance(self):
        """A merge with high dominance is flagged as bad."""
        result = compute_merge_outcomes(
            score_A_solo=1.0,
            score_B_solo=1.0,
            score_A_merged=0.99,
            score_B_merged=0.50,
        )

        # D = |0.99-0.50| / (0.99+0.50+eps) ~ 0.329 > 0.2
        assert result["is_bad_merge"] is True


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge case handling for compute_merge_outcomes."""

    def test_zero_solo_scores_use_eps(self):
        """When solo scores are zero, eps prevents division by zero."""
        result = compute_merge_outcomes(
            score_A_solo=0.0,
            score_B_solo=0.0,
            score_A_merged=0.5,
            score_B_merged=0.5,
        )

        # retention = 0.5 / eps -> very large number
        assert result["retention_A"] > 1e10
        assert result["retention_B"] > 1e10
        # Should not be NaN or inf
        assert result["D"] == pytest.approx(0.0, abs=1e-6)

    def test_zero_merged_scores(self):
        """When merged scores are both zero, D should be 0 (no dominance)."""
        result = compute_merge_outcomes(
            score_A_solo=0.80,
            score_B_solo=0.90,
            score_A_merged=0.0,
            score_B_merged=0.0,
        )

        assert result["retention_A"] == pytest.approx(0.0)
        assert result["retention_B"] == pytest.approx(0.0)
        assert result["Q_min"] == pytest.approx(0.0)
        assert result["D"] == pytest.approx(0.0, abs=1e-6)

    def test_all_zeros(self):
        """All scores zero should not raise or produce NaN."""
        result = compute_merge_outcomes(0.0, 0.0, 0.0, 0.0)

        # retention = 0 / eps = 0 (since numerator is 0)
        assert result["retention_A"] == pytest.approx(0.0)
        assert result["retention_B"] == pytest.approx(0.0)
        assert result["Q_min"] == pytest.approx(0.0)
        assert result["D"] == pytest.approx(0.0, abs=1e-6)

    def test_custom_eps(self):
        """Custom eps value is respected."""
        result = compute_merge_outcomes(
            score_A_solo=0.0,
            score_B_solo=1.0,
            score_A_merged=0.5,
            score_B_merged=0.5,
            eps=1.0,
        )

        # retention_A = 0.5 / max(0.0, 1.0) = 0.5
        assert result["retention_A"] == pytest.approx(0.5)

    def test_merged_exceeds_solo(self):
        """Retention can be > 1 if merged outperforms solo."""
        result = compute_merge_outcomes(
            score_A_solo=0.50,
            score_B_solo=0.50,
            score_A_merged=0.60,
            score_B_merged=0.55,
        )

        assert result["retention_A"] == pytest.approx(1.2)
        assert result["retention_B"] == pytest.approx(1.1)
        assert result["Q_min"] == pytest.approx(1.1)


# ---------------------------------------------------------------------------
# is_bad_merge -- standalone function
# ---------------------------------------------------------------------------


class TestIsBadMerge:
    """Tests for the is_bad_merge standalone function."""

    def test_good_merge_defaults(self):
        """Q_min >= 0.95 and D <= 0.2 -> not bad."""
        assert is_bad_merge(Q_min=0.96, D=0.1) is False

    def test_exactly_at_thresholds_is_not_bad(self):
        """At exactly the threshold boundary (Q_min=0.95, D=0.2) -> not bad."""
        assert is_bad_merge(Q_min=0.95, D=0.2) is False

    def test_low_Q_min_triggers_bad(self):
        """Q_min below threshold -> bad."""
        assert is_bad_merge(Q_min=0.94, D=0.1) is True

    def test_high_D_triggers_bad(self):
        """D above threshold -> bad."""
        assert is_bad_merge(Q_min=0.99, D=0.21) is True

    def test_both_bad(self):
        """Both Q_min low and D high -> still bad."""
        assert is_bad_merge(Q_min=0.50, D=0.80) is True

    def test_custom_thresholds_stricter(self):
        """Custom thresholds can be stricter."""
        # With defaults this would be fine
        assert is_bad_merge(Q_min=0.97, D=0.15) is False
        # With stricter thresholds it fails
        assert is_bad_merge(Q_min=0.97, D=0.15, Q_min_threshold=0.98) is True

    def test_custom_thresholds_looser(self):
        """Custom thresholds can be more permissive."""
        # With defaults this would be bad
        assert is_bad_merge(Q_min=0.90, D=0.25) is True
        # With looser thresholds it passes
        assert is_bad_merge(Q_min=0.90, D=0.25, Q_min_threshold=0.85, D_threshold=0.3) is False

    def test_perfect_merge(self):
        """Perfect retention and no dominance -> not bad."""
        assert is_bad_merge(Q_min=1.0, D=0.0) is False

    def test_above_one_retention(self):
        """Retention > 1 (merged outperforms solo) -> not bad if D is low."""
        assert is_bad_merge(Q_min=1.1, D=0.05) is False
