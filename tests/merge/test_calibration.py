"""Tests for gradience.vnext.merge.calibration — empirical threshold profiles."""

from __future__ import annotations

import pytest

from gradience.vnext.merge.calibration import (
    DISENTANGLED_ATTENTION,
    PROFILES,
    REFERENCE_POINTS,
    STANDARD_ATTENTION,
    ThresholdProfile,
    get_profile,
)
from gradience.vnext.merge.verdicts import VerdictThresholds

# ---------------------------------------------------------------------------
# ThresholdProfile basics
# ---------------------------------------------------------------------------


class TestThresholdProfile:
    """Core ThresholdProfile tests."""

    def test_standard_attention_exists(self):
        assert STANDARD_ATTENTION.name == "standard_attention"

    def test_disentangled_attention_exists(self):
        assert DISENTANGLED_ATTENTION.name == "disentangled_attention"

    def test_frozen(self):
        with pytest.raises(AttributeError):
            STANDARD_ATTENTION.alignment_safe = 0.99  # type: ignore[misc]

    def test_to_dict_roundtrip(self):
        d = STANDARD_ATTENTION.to_dict()
        assert isinstance(d, dict)
        assert d["name"] == "standard_attention"
        assert d["alignment_safe"] == STANDARD_ATTENTION.alignment_safe

    def test_to_verdict_thresholds_keys(self):
        vt_kwargs = STANDARD_ATTENTION.to_verdict_thresholds()
        assert "low_overlap" in vt_kwargs
        assert "high_overlap" in vt_kwargs
        assert "imbalanced_frob" in vt_kwargs


# ---------------------------------------------------------------------------
# Profile calibration sanity
# ---------------------------------------------------------------------------


class TestCalibrationSanity:
    """Verify that calibrated values are internally consistent."""

    @pytest.mark.parametrize("profile", list(PROFILES.values()), ids=lambda p: p.name)
    def test_alignment_safe_below_risk(self, profile: ThresholdProfile):
        """Safe threshold must be strictly below risk threshold."""
        assert profile.alignment_safe < profile.alignment_risk

    @pytest.mark.parametrize("profile", list(PROFILES.values()), ids=lambda p: p.name)
    def test_erank_safe_below_risk(self, profile: ThresholdProfile):
        """Erank safe ratio must be below risk ratio."""
        assert profile.erank_ratio_safe < profile.erank_ratio_risk

    @pytest.mark.parametrize("profile", list(PROFILES.values()), ids=lambda p: p.name)
    def test_alignment_safe_positive(self, profile: ThresholdProfile):
        assert profile.alignment_safe > 0.0

    @pytest.mark.parametrize("profile", list(PROFILES.values()), ids=lambda p: p.name)
    def test_alignment_risk_below_one(self, profile: ThresholdProfile):
        assert profile.alignment_risk < 1.0

    @pytest.mark.parametrize("profile", list(PROFILES.values()), ids=lambda p: p.name)
    def test_imbalanced_frob_positive(self, profile: ThresholdProfile):
        assert profile.imbalanced_frob > 1.0

    def test_disentangled_tighter_erank(self):
        """DeBERTa erank thresholds should be tighter than standard."""
        assert DISENTANGLED_ATTENTION.erank_ratio_safe < STANDARD_ATTENTION.erank_ratio_safe
        assert DISENTANGLED_ATTENTION.erank_ratio_risk < STANDARD_ATTENTION.erank_ratio_risk

    def test_disentangled_lower_alignment(self):
        """DeBERTa alignment thresholds should be lower than standard."""
        assert DISENTANGLED_ATTENTION.alignment_safe < STANDARD_ATTENTION.alignment_safe
        assert DISENTANGLED_ATTENTION.alignment_risk < STANDARD_ATTENTION.alignment_risk

    def test_standard_ck_predictive(self):
        assert STANDARD_ATTENTION.ck_predictive is True

    def test_disentangled_ck_not_predictive(self):
        assert DISENTANGLED_ATTENTION.ck_predictive is False


# ---------------------------------------------------------------------------
# VerdictThresholds.from_profile
# ---------------------------------------------------------------------------


class TestFromProfile:
    """Verify that VerdictThresholds.from_profile produces valid thresholds."""

    def test_standard_profile(self):
        vt = VerdictThresholds.from_profile(STANDARD_ATTENTION)
        assert vt.low_overlap == STANDARD_ATTENTION.alignment_safe
        assert vt.high_overlap == STANDARD_ATTENTION.alignment_risk

    def test_disentangled_profile(self):
        vt = VerdictThresholds.from_profile(DISENTANGLED_ATTENTION)
        assert vt.low_overlap == DISENTANGLED_ATTENTION.alignment_safe
        assert vt.high_overlap == DISENTANGLED_ATTENTION.alignment_risk

    def test_preserves_uncalibrated_defaults(self):
        """Fields without empirical calibration keep default values."""
        vt = VerdictThresholds.from_profile(STANDARD_ATTENTION)
        defaults = VerdictThresholds()
        assert vt.aligned == defaults.aligned
        assert vt.conflicting == defaults.conflicting

    def test_conservative_is_stricter_than_standard_profile(self):
        """Conservative profile should flag more than standard calibrated."""
        vt_cal = VerdictThresholds.from_profile(STANDARD_ATTENTION)
        vt_con = VerdictThresholds.conservative()
        # Conservative has lower high_overlap (flags more as high-overlap)
        assert vt_con.high_overlap <= vt_cal.high_overlap

    def test_profile_thresholds_differ_from_defaults(self):
        """Calibrated thresholds should differ from hardcoded defaults."""
        vt_cal = VerdictThresholds.from_profile(STANDARD_ATTENTION)
        vt_def = VerdictThresholds()
        # At least one threshold should differ
        assert vt_cal.low_overlap != vt_def.low_overlap or vt_cal.high_overlap != vt_def.high_overlap


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class TestRegistry:
    """Tests for profile registry and lookup."""

    def test_get_standard(self):
        p = get_profile("standard_attention")
        assert p is STANDARD_ATTENTION

    def test_get_disentangled(self):
        p = get_profile("disentangled_attention")
        assert p is DISENTANGLED_ATTENTION

    def test_unknown_raises(self):
        with pytest.raises(KeyError, match="Unknown threshold profile"):
            get_profile("nonexistent_arch")

    def test_all_profiles_in_registry(self):
        assert len(PROFILES) >= 2


# ---------------------------------------------------------------------------
# Reference points
# ---------------------------------------------------------------------------


class TestReferencePoints:
    """Verify reference points are well-formed."""

    def test_all_have_value(self):
        for name, ref in REFERENCE_POINTS.items():
            assert "value" in ref, f"Reference point {name!r} missing 'value'"

    def test_all_have_source(self):
        for name, ref in REFERENCE_POINTS.items():
            assert "source" in ref, f"Reference point {name!r} missing 'source'"
            assert "FINDINGS.md" in ref["source"], f"Reference point {name!r} source should cite FINDINGS.md"

    def test_key_values_match_findings(self):
        """Spot-check key values against FINDINGS.md numbers."""
        assert REFERENCE_POINTS["mistral_same_task_overlap"]["value"] == 0.473
        assert REFERENCE_POINTS["mistral_cross_task_overlap"]["value"] == 0.200
        assert REFERENCE_POINTS["erank_distilbert_ratio"]["value"] == pytest.approx(2.43, abs=0.01)
        assert REFERENCE_POINTS["erank_deberta_ratio"]["value"] == pytest.approx(1.23, abs=0.01)
        assert REFERENCE_POINTS["ck_alignment_rho_qnli"]["value"] == pytest.approx(0.56, abs=0.01)
