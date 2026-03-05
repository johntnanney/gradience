"""Tests for gradience.vnext.merge.verdicts — verdict decision logic."""

from __future__ import annotations

import pytest

from gradience.vnext.merge.spectral_compat import SubspaceMetrics
from gradience.vnext.merge.verdicts import (
    CompatibilityVerdict,
    LayerVerdict,
    VerdictThresholds,
    assess_layer,
    assess_overall,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_metrics(**overrides) -> SubspaceMetrics:
    """Build SubspaceMetrics with sensible defaults, overriding as needed."""
    defaults = dict(
        principal_angle_cosines=(0.1, 0.05, 0.02),
        mean_overlap=0.05,
        max_overlap=0.1,
        directional_agreement=0.1,
        magnitude_ratio=1.2,
        frobenius_ratio=1.1,
        frobenius_norm_a=5.0,
        frobenius_norm_b=4.5,
        scale_bounded_ratio=0.85,
        scale_log_ratio=0.12,
        frob_bounded_ratio=0.90,
        frob_log_ratio=0.08,
        effective_rank_a=3,
        effective_rank_b=3,
        nominal_rank_a=8,
        nominal_rank_b=8,
        stable_rank_a=2.5,
        stable_rank_b=2.3,
    )
    defaults.update(overrides)
    return SubspaceMetrics(**defaults)


# ---------------------------------------------------------------------------
# VerdictThresholds
# ---------------------------------------------------------------------------


class TestVerdictThresholds:
    """Tests for threshold presets."""

    def test_defaults(self):
        t = VerdictThresholds()
        assert t.low_overlap == 0.2
        assert t.high_overlap == 0.5
        assert t.aligned == 0.5
        assert t.conflicting == -0.3
        assert t.imbalanced == 5.0

    def test_conservative_stricter(self):
        t = VerdictThresholds.conservative()
        defaults = VerdictThresholds()
        # Conservative should flag more issues
        assert t.low_overlap < defaults.low_overlap
        assert t.high_overlap < defaults.high_overlap
        assert t.imbalanced < defaults.imbalanced

    def test_permissive_looser(self):
        t = VerdictThresholds.permissive()
        defaults = VerdictThresholds()
        # Permissive should flag fewer issues
        assert t.low_overlap > defaults.low_overlap
        assert t.high_overlap > defaults.high_overlap
        assert t.imbalanced > defaults.imbalanced

    def test_imbalanced_frob_default(self):
        """New Frobenius imbalance threshold exists with sensible default."""
        t = VerdictThresholds()
        assert t.imbalanced_frob == 5.0

    def test_imbalanced_frob_in_profiles(self):
        """Conservative/permissive profiles set imbalanced_frob."""
        c = VerdictThresholds.conservative()
        p = VerdictThresholds.permissive()
        assert c.imbalanced_frob < p.imbalanced_frob
        assert c.imbalanced_frob == 3.0
        assert p.imbalanced_frob == 10.0

    def test_imbalanced_frob_in_to_dict(self):
        """imbalanced_frob appears in serialized dict."""
        d = VerdictThresholds().to_dict()
        assert d["imbalanced_frob"] == 5.0

    def test_to_dict(self):
        t = VerdictThresholds()
        d = t.to_dict()
        assert isinstance(d, dict)
        assert d["low_overlap"] == 0.2
        assert d["imbalanced"] == 5.0


# ---------------------------------------------------------------------------
# assess_layer
# ---------------------------------------------------------------------------


class TestAssessLayer:
    """Tests for per-layer verdict decision tree."""

    def test_safe_orthogonal(self):
        """Low overlap -> SAFE."""
        metrics = _make_metrics(mean_overlap=0.05, max_overlap=0.1)
        lv = assess_layer("test.q_proj", "attn", metrics)

        assert lv.verdict == CompatibilityVerdict.SAFE
        assert lv.suggested_strategy == "linear"
        assert lv.conflict_dimensions == 0

    def test_redundant(self):
        """High overlap + aligned -> REDUNDANT."""
        metrics = _make_metrics(
            mean_overlap=0.8,
            max_overlap=0.9,
            directional_agreement=0.8,
            principal_angle_cosines=(0.9, 0.8, 0.7),
        )
        lv = assess_layer("test.q_proj", "attn", metrics)

        assert lv.verdict == CompatibilityVerdict.REDUNDANT
        assert lv.suggested_strategy == "ties"

    def test_conflicting(self):
        """High overlap + opposing -> CONFLICTING."""
        metrics = _make_metrics(
            mean_overlap=0.8,
            max_overlap=0.9,
            directional_agreement=-0.7,
            principal_angle_cosines=(0.9, 0.8, 0.7),
        )
        lv = assess_layer("test.q_proj", "attn", metrics)

        assert lv.verdict == CompatibilityVerdict.CONFLICTING
        assert lv.suggested_strategy == "dare"
        assert lv.conflict_dimensions > 0

    def test_imbalanced(self):
        """Extreme magnitude ratio -> IMBALANCED."""
        metrics = _make_metrics(
            mean_overlap=0.3,  # moderate overlap (between low and high)
            max_overlap=0.4,
            directional_agreement=0.2,
            magnitude_ratio=8.0,
        )
        lv = assess_layer("test.q_proj", "attn", metrics)

        assert lv.verdict == CompatibilityVerdict.IMBALANCED
        assert lv.suggested_coefficients is not None
        assert sum(lv.suggested_coefficients) == pytest.approx(1.0, abs=0.01)

    def test_moderate_defaults_safe(self):
        """Moderate overlap, no strong signal -> default SAFE."""
        metrics = _make_metrics(
            mean_overlap=0.3,
            max_overlap=0.4,
            directional_agreement=0.2,
            magnitude_ratio=1.5,
        )
        lv = assess_layer("test.q_proj", "attn", metrics)

        assert lv.verdict == CompatibilityVerdict.SAFE
        assert lv.confidence == 0.5  # moderate confidence

    def test_to_dict_serializable(self):
        """LayerVerdict.to_dict() produces JSON-serializable output."""
        import json

        metrics = _make_metrics()
        lv = assess_layer("test.q_proj", "attn", metrics)
        d = lv.to_dict()

        # Should not raise
        json_str = json.dumps(d)
        assert "verdict" in json_str
        assert "metrics" in json_str

    def test_custom_thresholds(self):
        """Custom thresholds change verdict boundaries."""
        # With default thresholds, overlap=0.4 is moderate (between 0.2 and 0.5)
        metrics = _make_metrics(mean_overlap=0.4, directional_agreement=0.7)
        lv_default = assess_layer("test.q_proj", "attn", metrics)

        # With conservative thresholds (high_overlap=0.35), 0.4 is high
        lv_conservative = assess_layer(
            "test.q_proj", "attn", metrics,
            thresholds=VerdictThresholds.conservative(),
        )

        # Conservative should catch this as redundant
        assert lv_conservative.verdict == CompatibilityVerdict.REDUNDANT
        # Default might not
        assert lv_default.verdict in (CompatibilityVerdict.SAFE, CompatibilityVerdict.REDUNDANT)


# ---------------------------------------------------------------------------
# assess_overall
# ---------------------------------------------------------------------------


class TestAssessOverall:
    """Tests for aggregate verdict computation."""

    def test_empty_layers(self):
        """No layers -> SAFE, score=0."""
        verdict, score, recs = assess_overall([])
        assert verdict == CompatibilityVerdict.SAFE
        assert score == 0.0
        assert len(recs) > 0

    def test_all_safe(self):
        """All SAFE layers -> SAFE overall."""
        metrics = _make_metrics(mean_overlap=0.05)
        lvs = [assess_layer(f"layer.{i}", "attn", metrics) for i in range(3)]

        verdict, score, recs = assess_overall(lvs)
        assert verdict == CompatibilityVerdict.SAFE

    def test_worst_case_wins(self):
        """If any layer is CONFLICTING, overall is CONFLICTING."""
        safe_metrics = _make_metrics(mean_overlap=0.05)
        conflict_metrics = _make_metrics(
            mean_overlap=0.8,
            directional_agreement=-0.7,
            principal_angle_cosines=(0.9, 0.8, 0.7),
        )

        lvs = [
            assess_layer("layer.0", "attn", safe_metrics),
            assess_layer("layer.1", "attn", safe_metrics),
            assess_layer("layer.2", "attn", conflict_metrics),
        ]

        verdict, _, recs = assess_overall(lvs)
        assert verdict == CompatibilityVerdict.CONFLICTING
        # Should have a conflict recommendation
        assert any("conflict" in r.lower() for r in recs)

    def test_recommendations_for_each_type(self):
        """Each verdict type gets appropriate recommendations."""
        redundant_metrics = _make_metrics(
            mean_overlap=0.8, directional_agreement=0.8,
            principal_angle_cosines=(0.9, 0.8, 0.7),
        )
        imbalanced_metrics = _make_metrics(
            mean_overlap=0.3, directional_agreement=0.2, magnitude_ratio=8.0,
        )

        lvs = [
            assess_layer("layer.0", "attn", redundant_metrics),
            assess_layer("layer.1", "attn", imbalanced_metrics),
        ]

        _, _, recs = assess_overall(lvs)
        assert any("redundancy" in r.lower() for r in recs)
        assert any("imbalance" in r.lower() for r in recs)

    def test_score_bounded(self):
        """Compatibility score is in [0, 1]."""
        metrics = _make_metrics(mean_overlap=0.5)
        lvs = [assess_layer(f"layer.{i}", "attn", metrics) for i in range(5)]

        _, score, _ = assess_overall(lvs)
        assert 0.0 <= score <= 1.0
