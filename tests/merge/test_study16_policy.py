"""Policy-level tests encoding Study 16 conclusions.

These tests encode the core findings from Study 16 as executable assertions:

1. Norm-equalized coefficients correctly rescale to geometric-mean Frobenius norm
2. Weak-source adapters (Pair 06 scenario) produce explicit warnings
3. Balanced compatible pairs yield safe/no-op recommendations
4. Missing QA data produces structural-only disclaimer
5. Flagged-weak adapters are never silently preserved without warning
"""

from __future__ import annotations

import pytest

from gradience.vnext.merge.containers import (
    AdapterMetadata,
    AggregateResult,
    PairAuditResult,
    WarningCode,
)
from gradience.vnext.merge.eligibility import EligibilityStatus
from gradience.vnext.merge.qa_report import build_qa_report
from gradience.vnext.merge.recommend import (
    EligibilityContext,
    _eligibility_warnings,
    norm_equalized_coefficients,
    recommend_merge,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_lv_dict(
    verdict: str = "safe",
    mean_overlap: float = 0.05,
    directional_agreement: float = 0.1,
    magnitude_ratio: float = 1.2,
    conflict_dimensions: int = 0,
    confidence: float = 0.8,
    suggested_coefficients=None,
    layer_name: str = "model.layers.0.self_attn.q_proj",
) -> dict:
    return {
        "layer_name": layer_name,
        "module_type": "attn",
        "verdict": verdict,
        "confidence": confidence,
        "recommendation": f"Test recommendation for {verdict}",
        "conflict_dimensions": conflict_dimensions,
        "safe_merge_rank": 6,
        "suggested_strategy": verdict,
        "suggested_coefficients": suggested_coefficients,
        "metrics": {
            "principal_angle_cosines": [0.1, 0.05, 0.02],
            "mean_overlap": mean_overlap,
            "max_overlap": mean_overlap + 0.05,
            "directional_agreement": directional_agreement,
            "magnitude_ratio": magnitude_ratio,
            "frobenius_ratio": 1.1,
            "frobenius_norm_a": 5.0,
            "frobenius_norm_b": 4.5,
            "scale_bounded_ratio": 0.85,
            "scale_log_ratio": 0.12,
            "frob_bounded_ratio": 0.90,
            "frob_log_ratio": 0.08,
            "effective_rank_a": 3,
            "effective_rank_b": 3,
            "nominal_rank_a": 64,
            "nominal_rank_b": 64,
            "stable_rank_a": 2.5,
            "stable_rank_b": 2.3,
        },
    }


class _FakeReport:
    """Minimal stand-in for MergeAuditReport."""

    def __init__(self, layer_verdicts, source_qa=None, aggregate=None):
        self.layer_verdicts = layer_verdicts
        self.source_qa = source_qa
        n_safe = sum(1 for lv in layer_verdicts if lv["verdict"] == "safe")
        n_redundant = sum(1 for lv in layer_verdicts if lv["verdict"] == "redundant")
        n_conflicting = sum(1 for lv in layer_verdicts if lv["verdict"] == "conflicting")
        n_imbalanced = sum(1 for lv in layer_verdicts if lv["verdict"] == "imbalanced")
        overlaps = [lv["metrics"]["mean_overlap"] for lv in layer_verdicts]
        mag_ratios = [lv["metrics"]["magnitude_ratio"] for lv in layer_verdicts]
        if aggregate is not None:
            self.aggregate = AggregateResult.from_dict(aggregate) if isinstance(aggregate, dict) else aggregate
        else:
            self.aggregate = AggregateResult(
                overall_verdict="safe",
                compatibility_score=0.75,
                mean_overlap=sum(overlaps) / len(overlaps) if overlaps else 0.0,
                n_safe=n_safe,
                n_redundant=n_redundant,
                n_conflicting=n_conflicting,
                n_imbalanced=n_imbalanced,
                mean_magnitude_ratio=sum(mag_ratios) / len(mag_ratios) if mag_ratios else 1.0,
            )
        self.adapter_a = AdapterMetadata(
            path="/tmp/a", rank=8, alpha=16.0, base_model="llama-7b", n_layers=len(layer_verdicts)
        )
        self.adapter_b = AdapterMetadata(
            path="/tmp/b", rank=8, alpha=16.0, base_model="llama-7b", n_layers=len(layer_verdicts)
        )
        self.recommendations = ["Test recommendation"]
        self.warnings = []


# ===========================================================================
# Test 1: Norm-equalized coefficient correctness
# ===========================================================================


class TestNormEqualizedCoefficientCorrectness:
    """Verify that norm_equalized_coefficients rescales both adapters to
    the geometric-mean Frobenius norm, then applies user weights.

    Study 16 finding: norm-equalized merge is a strong baseline because
    it removes scale imbalance as a confound.
    """

    def test_equal_norms_produce_equal_coefficients(self):
        """When both norms are equal, coefficients equal user weights."""
        ca, cb = norm_equalized_coefficients(10.0, 10.0, weight_a=0.5)
        assert ca == pytest.approx(0.5, abs=1e-3)
        assert cb == pytest.approx(0.5, abs=1e-3)

    def test_geometric_mean_rescaling(self):
        """Coefficients embed geometric-mean norm rescaling.

        For norm_a=1, norm_b=100:
          target = sqrt(1 * 100) = 10
          scale_a = 10 / 1 = 10
          scale_b = 10 / 100 = 0.1
          coeff_a = 0.5 * 10 = 5.0
          coeff_b = 0.5 * 0.1 = 0.05
        """
        ca, cb = norm_equalized_coefficients(1.0, 100.0, weight_a=0.5)
        assert ca == pytest.approx(5.0, abs=0.01)
        assert cb == pytest.approx(0.05, abs=0.01)

    def test_weight_a_respected(self):
        """Non-default weight_a is reflected in final coefficients."""
        ca, cb = norm_equalized_coefficients(5.0, 5.0, weight_a=0.7)
        assert ca == pytest.approx(0.7, abs=1e-3)
        assert cb == pytest.approx(0.3, abs=1e-3)

    def test_extreme_imbalance_1000x(self):
        """Even 1000x imbalance produces valid coefficients."""
        ca, cb = norm_equalized_coefficients(0.001, 1.0, weight_a=0.5)
        # A's scale factor = sqrt(0.001) / 0.001 ≈ 31.6
        # B's scale factor = sqrt(0.001) / 1.0 ≈ 0.0316
        assert ca > cb  # weaker adapter (smaller norm) gets boosted
        assert ca > 0
        assert cb > 0

    def test_zero_norm_handled_safely(self):
        """Zero norm adapter doesn't cause division by zero."""
        ca, cb = norm_equalized_coefficients(0.0, 5.0, weight_a=0.5)
        # Should not raise
        assert isinstance(ca, float)
        assert isinstance(cb, float)


# ===========================================================================
# Test 2: Pair 06 weak-source warning (Study 16 scenario)
# ===========================================================================


class TestPair06WeakSourceWarning:
    """Study 16 Pair 06 scenario: one adapter is structurally compatible
    but behaviorally weak.  The system must warn that rebalancing may
    preserve a useless signal.

    Policy requirement: when source QA flags an adapter as FLAGGED_WEAK,
    the recommendation must include an explicit warning about it.
    """

    def test_one_weak_adapter_produces_warning(self):
        """Single weak adapter → explicit 'weak adapter' warning."""
        layers = [_make_lv_dict("safe", layer_name=f"layer.{i}") for i in range(3)]
        report = _FakeReport(
            layers,
            source_qa={
                "adapter_a": {"status": "eligible", "adapter_path": "/tmp/a"},
                "adapter_b": {"status": "flagged_weak", "adapter_path": "/tmp/b"},
            },
        )
        rec = recommend_merge(report)

        assert any("weak adapter" in w.lower() for w in rec.warnings), (
            f"Expected 'weak adapter' warning, got: {rec.warnings}"
        )

    def test_weak_adapter_warning_in_qa_report_caveats(self):
        """QA report caveats mention the weak adapter."""
        layers = [_make_lv_dict("safe", layer_name="layer.0")]
        report = _FakeReport(
            layers,
            source_qa={
                "adapter_a": {"status": "flagged_weak", "adapter_path": "/tmp/a"},
                "adapter_b": {"status": "eligible", "adapter_path": "/tmp/b"},
            },
        )
        qa = build_qa_report(report)

        assert any("underperform" in c.lower() or "weak" in c.lower() for c in qa.caveats), (
            f"Expected weak-adapter caveat, got: {qa.caveats}"
        )

    def test_both_weak_produces_deployment_warning(self):
        """Both weak → warns about uncertain deployment value."""
        layers = [_make_lv_dict("safe", layer_name="layer.0")]
        report = _FakeReport(
            layers,
            source_qa={
                "adapter_a": {"status": "flagged_weak", "adapter_path": "/tmp/a"},
                "adapter_b": {"status": "flagged_weak", "adapter_path": "/tmp/b"},
            },
        )
        rec = recommend_merge(report)

        assert any("both" in w.lower() and "underperform" in w.lower() for w in rec.warnings), (
            f"Expected 'both underperform' warning, got: {rec.warnings}"
        )

    def test_both_weak_qa_recommends_reconsider(self):
        """QA report recommends reconsidering when both are weak."""
        layers = [_make_lv_dict("safe", layer_name="layer.0")]
        report = _FakeReport(
            layers,
            source_qa={
                "adapter_a": {"status": "flagged_weak", "adapter_path": "/tmp/a"},
                "adapter_b": {"status": "flagged_weak", "adapter_path": "/tmp/b"},
            },
        )
        qa = build_qa_report(report)

        assert "reconsider" in qa.recommended_action.lower(), (
            f"Expected 'reconsider' in action, got: {qa.recommended_action}"
        )


# ===========================================================================
# Test 3: Balanced pair returns no-op / safe recommendation
# ===========================================================================


class TestBalancedPairNoOp:
    """When both adapters are eligible and structurally compatible (all
    layers safe), the recommendation should be low risk with a
    straightforward merge strategy.

    Study 16 finding: balanced, compatible pairs should merge cleanly
    with linear or norm-equalized strategy.
    """

    def test_all_safe_low_risk(self):
        """All-safe layers → low overall risk."""
        layers = [_make_lv_dict("safe", layer_name=f"layer.{i}", mean_overlap=0.05) for i in range(5)]
        report = _FakeReport(
            layers,
            source_qa={
                "adapter_a": {"status": "eligible"},
                "adapter_b": {"status": "eligible"},
            },
        )
        rec = recommend_merge(report)

        assert rec.overall_risk == "low"
        assert all(lr.strategy == "linear" for lr in rec.layer_recommendations)
        assert all(lr.coefficients == (0.5, 0.5) for lr in rec.layer_recommendations)

    def test_no_eligibility_warnings_when_both_eligible(self):
        """Both eligible → no hard warnings emitted."""
        layers = [_make_lv_dict("safe", layer_name="layer.0")]
        report = _FakeReport(
            layers,
            source_qa={
                "adapter_a": {"status": "eligible"},
                "adapter_b": {"status": "eligible"},
            },
        )
        rec = recommend_merge(report)

        assert len(rec.warnings) == 0

    def test_qa_report_safe_action(self):
        """QA report recommends safe merge."""
        layers = [_make_lv_dict("safe", layer_name="layer.0", mean_overlap=0.05)]
        report = _FakeReport(
            layers,
            source_qa={
                "adapter_a": {"status": "eligible"},
                "adapter_b": {"status": "eligible"},
            },
        )
        qa = build_qa_report(report)

        assert qa.pair_risk == "low"
        assert "safe" in qa.recommended_action.lower() or "merge" in qa.recommended_action.lower()

    def test_norm_equalized_in_fallbacks(self):
        """norm_equalized is always offered as fallback."""
        layers = [_make_lv_dict("safe", layer_name="layer.0")]
        report = _FakeReport(layers)
        rec = recommend_merge(report)

        assert "norm_equalized" in rec.fallback_strategies


# ===========================================================================
# Test 4: Missing QA produces structural-only disclaimer
# ===========================================================================


class TestMissingQAStructuralDisclaimer:
    """When no source QA data is provided (the default case), the system
    must clearly state that the recommendation is structural only and
    cannot predict downstream task performance.

    Study 16 finding: structural analysis alone is necessary but not
    sufficient.  Users must know when behavioral data is missing.
    """

    def test_no_source_qa_warns_structural_only(self):
        """No source_qa → 'structural balance only' warning."""
        layers = [_make_lv_dict("safe", layer_name="layer.0")]
        report = _FakeReport(layers, source_qa=None)
        rec = recommend_merge(report)

        assert len(rec.warnings) >= 1
        assert any("structural" in w.lower() for w in rec.warnings), (
            f"Expected 'structural' in warnings, got: {rec.warnings}"
        )

    def test_empty_source_qa_treated_as_missing(self):
        """Empty source_qa dict is same as None."""
        layers = [_make_lv_dict("safe", layer_name="layer.0")]
        report = _FakeReport(layers, source_qa={})
        rec = recommend_merge(report)

        assert any("structural" in w.lower() for w in rec.warnings)

    def test_qa_report_caveat_about_missing_data(self):
        """QA report caveats include structural-only disclaimer."""
        layers = [_make_lv_dict("safe", layer_name="layer.0")]
        report = _FakeReport(layers, source_qa=None)
        qa = build_qa_report(report)

        assert any("structural" in c.lower() and ("only" in c.lower() or "cannot" in c.lower()) for c in qa.caveats), (
            f"Expected structural-only caveat, got: {qa.caveats}"
        )

    def test_qa_report_confidence_mentions_no_behavioral(self):
        """Confidence note mentions lack of behavioral data."""
        layers = [_make_lv_dict("safe", layer_name="layer.0")]
        report = _FakeReport(layers, source_qa=None)
        qa = build_qa_report(report)

        assert "behavioral" in qa.confidence_note.lower() or "structural" in qa.confidence_note.lower(), (
            f"Expected behavioral/structural mention in confidence, got: {qa.confidence_note}"
        )


# ===========================================================================
# Test 5: Flagged-weak adapters never silently preserved
# ===========================================================================


class TestFlaggedWeakNeverSilent:
    """Study 16's core safety invariant: a flagged-weak adapter must
    NEVER be silently preserved in a merge recommendation.  Any code
    path that sees FLAGGED_WEAK must emit at least one user-visible
    warning.

    This tests the invariant across multiple code paths.
    """

    def test_recommend_merge_warns_on_weak_a(self):
        """recommend_merge warns when adapter A is flagged weak."""
        layers = [_make_lv_dict("safe", layer_name="layer.0")]
        report = _FakeReport(
            layers,
            source_qa={
                "adapter_a": {"status": "flagged_weak", "adapter_path": "/tmp/a"},
                "adapter_b": {"status": "eligible", "adapter_path": "/tmp/b"},
            },
        )
        rec = recommend_merge(report)
        assert len(rec.warnings) > 0, "Weak adapter A must produce at least one warning"

    def test_recommend_merge_warns_on_weak_b(self):
        """recommend_merge warns when adapter B is flagged weak."""
        layers = [_make_lv_dict("safe", layer_name="layer.0")]
        report = _FakeReport(
            layers,
            source_qa={
                "adapter_a": {"status": "eligible", "adapter_path": "/tmp/a"},
                "adapter_b": {"status": "flagged_weak", "adapter_path": "/tmp/b"},
            },
        )
        rec = recommend_merge(report)
        assert len(rec.warnings) > 0, "Weak adapter B must produce at least one warning"

    def test_recommend_merge_warns_on_both_weak(self):
        """recommend_merge warns when both adapters are flagged weak."""
        layers = [_make_lv_dict("safe", layer_name="layer.0")]
        report = _FakeReport(
            layers,
            source_qa={
                "adapter_a": {"status": "flagged_weak", "adapter_path": "/tmp/a"},
                "adapter_b": {"status": "flagged_weak", "adapter_path": "/tmp/b"},
            },
        )
        rec = recommend_merge(report)
        assert len(rec.warnings) > 0, "Both weak adapters must produce warnings"

    def test_eligibility_warnings_function_on_weak(self):
        """_eligibility_warnings always fires on FLAGGED_WEAK."""
        ctx = EligibilityContext(
            status_a=EligibilityStatus.FLAGGED_WEAK,
            status_b=EligibilityStatus.ELIGIBLE,
        )
        warnings = _eligibility_warnings(ctx)
        assert len(warnings) > 0, "_eligibility_warnings must warn on weak"
        assert any("weak" in w.lower() for w in warnings)

    def test_qa_report_never_silent_on_weak(self):
        """QA report must surface weak-adapter info in caveats."""
        layers = [_make_lv_dict("safe", layer_name="layer.0")]
        report = _FakeReport(
            layers,
            source_qa={
                "adapter_a": {"status": "eligible", "adapter_path": "/tmp/a"},
                "adapter_b": {"status": "flagged_weak", "adapter_path": "/tmp/b"},
            },
        )
        qa = build_qa_report(report)

        # Must mention weakness somewhere user-visible
        all_text = " ".join([qa.recommended_action, qa.confidence_note] + list(qa.caveats))
        assert "weak" in all_text.lower() or "underperform" in all_text.lower(), (
            f"QA report must mention weak adapter. Got action='{qa.recommended_action}', caveats={qa.caveats}"
        )

    def test_pair_audit_result_classifies_weak_warnings(self):
        """PairAuditResult.from_audit_report classifies weak-adapter warnings."""
        from gradience.vnext.merge.containers import MatchingSummary
        from gradience.vnext.merge.report import MergeAuditReport

        report = MergeAuditReport(
            adapter_a=AdapterMetadata(path="/tmp/a", rank=8, alpha=16.0),
            adapter_b=AdapterMetadata(path="/tmp/b", rank=8, alpha=16.0),
            matching=MatchingSummary(n_shared=1),
            layer_verdicts=[],
            aggregate=AggregateResult(overall_verdict="safe", compatibility_score=0.8),
            recommendations=[],
            warnings=["Adapter B is weak and underperforms the base model"],
        )
        pair = PairAuditResult.from_audit_report(report)

        assert any(w.code == WarningCode.ADAPTER_WEAK for w in pair.typed_warnings), (
            f"Expected ADAPTER_WEAK warning code, got: {[w.code for w in pair.typed_warnings]}"
        )
