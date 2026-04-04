"""Tests for gradience.vnext.merge.over_accumulation."""

from __future__ import annotations

from gradience.vnext.merge.over_accumulation import (
    estimate_over_accumulation,
    estimate_over_accumulation_v2,
    summarize_over_accumulation,
)
from gradience.vnext.merge.spectral_compat import SubspaceMetrics


def _metrics(**overrides) -> SubspaceMetrics:
    defaults = dict(
        principal_angle_cosines=(0.8, 0.7, 0.6, 0.5),
        mean_overlap=0.75,
        max_overlap=0.85,
        directional_agreement=0.8,
        magnitude_ratio=1.2,
        frobenius_ratio=1.1,
        frobenius_norm_a=5.0,
        frobenius_norm_b=4.5,
        scale_bounded_ratio=0.9,
        scale_log_ratio=0.0,
        frob_bounded_ratio=0.9,
        frob_log_ratio=0.0,
        effective_rank_a=8,
        effective_rank_b=8,
        nominal_rank_a=64,
        nominal_rank_b=64,
        stable_rank_a=1.3,
        stable_rank_b=1.4,
    )
    defaults.update(overrides)
    return SubspaceMetrics(**defaults)


class TestEstimateOverAccumulation:
    def test_high_alignment_and_concentration_scores_high(self):
        est = estimate_over_accumulation(_metrics())
        assert est.score > 0.60
        assert est.band == "high"
        assert est.factors.alignment > 0.60
        assert est.factors.concentration > 0.70

    def test_high_overlap_but_diffuse_spectrum_stays_low(self):
        high = estimate_over_accumulation(_metrics())
        diffuse = estimate_over_accumulation(
            _metrics(
                stable_rank_a=8.0,
                stable_rank_b=8.0,
                nominal_rank_a=8,
                nominal_rank_b=8,
            )
        )
        assert diffuse.score < high.score
        assert diffuse.band == "low"

    def test_opposing_directions_reduce_score(self):
        est = estimate_over_accumulation(_metrics(directional_agreement=-0.8))
        assert est.score < 0.35
        assert est.band == "low"

    def test_rebalanced_coefficients_reduce_exposure(self):
        naive = estimate_over_accumulation(_metrics(), assumed_coefficients=(0.5, 0.5))
        rebalanced = estimate_over_accumulation(_metrics(), assumed_coefficients=(0.9, 0.1))
        assert rebalanced.factors.coefficient_exposure < naive.factors.coefficient_exposure
        assert rebalanced.score < naive.score


class TestEstimateOverAccumulationV2:
    def test_positive_alignment_increases_v2_risk(self):
        m_pos = _metrics(
            principal_angle_cosines=(0.9, 0.8, 0.7, 0.6),
            right_principal_angle_cosines=(0.85, 0.75, 0.65, 0.55),
            effective_singular_values_a=(4.0, 3.0, 2.0, 1.0),
            effective_singular_values_b=(3.5, 2.5, 1.5, 1.0),
            directional_agreement=0.9,
        )
        m_neg = _metrics(
            principal_angle_cosines=(0.9, 0.8, 0.7, 0.6),
            right_principal_angle_cosines=(0.85, 0.75, 0.65, 0.55),
            effective_singular_values_a=(4.0, 3.0, 2.0, 1.0),
            effective_singular_values_b=(3.5, 2.5, 1.5, 1.0),
            directional_agreement=-0.9,
        )
        est_pos = estimate_over_accumulation_v2(m_pos, assumed_coefficients=(0.5, 0.5))
        est_neg = estimate_over_accumulation_v2(m_neg, assumed_coefficients=(0.5, 0.5))
        assert est_pos.score > est_neg.score
        assert est_neg.factors.interaction_primary == 0.0

    def test_same_mean_overlap_but_different_right_spectrum_changes_score(self):
        m_low_right = _metrics(
            principal_angle_cosines=(0.9, 0.8, 0.7, 0.6),
            mean_overlap=0.75,
            right_principal_angle_cosines=(0.2, 0.2, 0.2, 0.2),
            effective_singular_values_a=(4.0, 3.0, 2.0, 1.0),
            effective_singular_values_b=(4.0, 3.0, 2.0, 1.0),
            directional_agreement=0.8,
        )
        m_high_right = _metrics(
            principal_angle_cosines=(0.9, 0.8, 0.7, 0.6),
            mean_overlap=0.75,
            right_principal_angle_cosines=(0.9, 0.8, 0.7, 0.6),
            effective_singular_values_a=(4.0, 3.0, 2.0, 1.0),
            effective_singular_values_b=(4.0, 3.0, 2.0, 1.0),
            directional_agreement=0.8,
        )
        est_low = estimate_over_accumulation_v2(m_low_right, assumed_coefficients=(0.5, 0.5))
        est_high = estimate_over_accumulation_v2(m_high_right, assumed_coefficients=(0.5, 0.5))
        assert est_high.factors.spectral_overlap_weighted > est_low.factors.spectral_overlap_weighted
        assert est_high.score > est_low.score

    def test_secondary_concentration_modulation_cannot_dominate(self):
        m = _metrics(
            principal_angle_cosines=(0.05, 0.02, 0.01, 0.0),
            right_principal_angle_cosines=(0.05, 0.02, 0.01, 0.0),
            effective_singular_values_a=(1.0, 1.0, 1.0, 1.0),
            effective_singular_values_b=(1.0, 1.0, 1.0, 1.0),
            directional_agreement=0.9,
            stable_rank_a=1.01,
            stable_rank_b=1.01,
        )
        est = estimate_over_accumulation_v2(m, assumed_coefficients=(0.5, 0.5))
        assert est.factors.interaction_primary < 0.01
        assert est.score < 0.02


class TestSummarizeOverAccumulation:
    def test_none_advisory_for_all_low(self):
        summary = summarize_over_accumulation(
            layer_scores=[0.1, 0.2, 0.3],
            layer_bands=["low", "low", "low"],
        )
        assert summary.advisory == "none"
        assert summary.summary == ""

    def test_watch_advisory_for_watch_layers(self):
        summary = summarize_over_accumulation(
            layer_scores=[0.4, 0.1, 0.2, 0.3],
            layer_bands=["watch", "low", "low", "low"],
        )
        assert summary.advisory == "watch"
        assert "naive linear merge" in summary.summary

    def test_elevated_advisory_for_high_layers(self):
        summary = summarize_over_accumulation(
            layer_scores=[0.62, 0.2],
            layer_bands=["high", "low"],
        )
        assert summary.advisory == "elevated"
        assert summary.high_risk_layer_count == 1

    def test_summary_mentions_primary_issue_when_non_overlap_is_present(self):
        summary = summarize_over_accumulation(
            layer_scores=[0.45, 0.2],
            layer_bands=["watch", "low"],
            non_overlap_issue_count=1,
        )
        assert summary.advisory == "watch"
        assert "primary concern" in summary.summary
