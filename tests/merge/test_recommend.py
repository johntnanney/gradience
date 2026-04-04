"""Tests for gradience.vnext.merge.recommend — recommendation engine."""

from __future__ import annotations

import pytest

from gradience.vnext.merge.eligibility import EligibilityStatus
from gradience.vnext.merge.recommend import (
    EligibilityContext,
    LayerDiagnosis,
    LayerRecommendation,
    MergeRecommendation,
    PairDiagnosis,
    _apply_layer_policy,
    _apply_policy,
    _compression_target,
    _compute_dare_drop_rate,
    _compute_trim_fraction,
    _recommend_layer,
    _should_compress,
    diagnose_layer,
    diagnose_pair,
    format_recommendation,
    norm_equalized_coefficients,
    rebalance_coefficients,
    recommend_merge,
)

# ---------------------------------------------------------------------------
# Helpers — synthetic layer verdict dicts
# ---------------------------------------------------------------------------


def _make_lv_dict(
    verdict: str = "safe",
    mean_overlap: float = 0.05,
    directional_agreement: float = 0.1,
    magnitude_ratio: float = 1.2,
    effective_rank_a: int = 3,
    effective_rank_b: int = 3,
    nominal_rank_a: int = 64,
    nominal_rank_b: int = 64,
    conflict_dimensions: int = 0,
    confidence: float = 0.8,
    suggested_coefficients=None,
    layer_name: str = "model.layers.0.self_attn.q_proj",
    principal_angle_cosines=(0.1, 0.05, 0.02),
    over_accumulation_score: float = 0.0,
    over_accumulation_band: str = "low",
) -> dict:
    """Build a synthetic LayerVerdict dict (as stored in MergeAuditReport)."""
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
        "over_accumulation_score": over_accumulation_score,
        "over_accumulation_band": over_accumulation_band,
        "metrics": {
            "principal_angle_cosines": list(principal_angle_cosines),
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
            "effective_rank_a": effective_rank_a,
            "effective_rank_b": effective_rank_b,
            "nominal_rank_a": nominal_rank_a,
            "nominal_rank_b": nominal_rank_b,
            "stable_rank_a": 2.5,
            "stable_rank_b": 2.3,
        },
    }


class _FakeReport:
    """Minimal stand-in for MergeAuditReport."""

    def __init__(self, layer_verdicts: list[dict], source_qa: dict | None = None):
        self.layer_verdicts = layer_verdicts
        self.source_qa = source_qa


# ---------------------------------------------------------------------------
# Parameter computation unit tests
# ---------------------------------------------------------------------------


class TestComputeTrimFraction:
    """Tests for overlap → trim fraction mapping."""

    def test_below_threshold_returns_zero(self):
        assert _compute_trim_fraction(0.3, high_overlap_threshold=0.5) == 0.0

    def test_at_threshold_returns_minimum(self):
        trim = _compute_trim_fraction(0.5, high_overlap_threshold=0.5)
        assert trim == pytest.approx(0.0, abs=0.01)

    def test_just_above_threshold(self):
        trim = _compute_trim_fraction(0.6, high_overlap_threshold=0.5)
        assert 0.1 <= trim <= 0.2

    def test_high_overlap_returns_high_trim(self):
        trim = _compute_trim_fraction(0.95, high_overlap_threshold=0.5)
        assert trim >= 0.4

    def test_maximum_overlap(self):
        trim = _compute_trim_fraction(1.0, high_overlap_threshold=0.5)
        assert trim == pytest.approx(0.5, abs=0.01)

    def test_monotonic(self):
        """Higher overlap → higher trim."""
        trims = [_compute_trim_fraction(o) for o in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]]
        for i in range(len(trims) - 1):
            assert trims[i] <= trims[i + 1]


class TestComputeDareDropRate:
    """Tests for conflict count → DARE drop rate mapping."""

    def test_zero_conflicts(self):
        rate = _compute_dare_drop_rate(0, 8)
        assert rate == 0.15

    def test_one_conflict(self):
        rate = _compute_dare_drop_rate(1, 8)
        assert 0.15 <= rate <= 0.25

    def test_all_conflicting(self):
        rate = _compute_dare_drop_rate(8, 8)
        assert rate == pytest.approx(0.5, abs=0.01)

    def test_bounded(self):
        """Drop rate stays in [0.15, 0.5]."""
        for n in range(0, 20):
            rate = _compute_dare_drop_rate(n, 10)
            assert 0.15 <= rate <= 0.5

    def test_monotonic(self):
        """More conflicts → higher drop rate."""
        rates = [_compute_dare_drop_rate(n, 10) for n in range(11)]
        for i in range(len(rates) - 1):
            assert rates[i] <= rates[i + 1]


class TestShouldCompress:
    """Tests for compression recommendation logic."""

    def test_low_overlap_no_compress(self):
        assert not _should_compress(0.2, 5, 64)

    def test_high_overlap_low_utilization(self):
        # overlap=0.7 > 0.5, utilization=5/64=0.078 < 0.4
        assert _should_compress(0.7, 5, 64)

    def test_high_overlap_high_utilization(self):
        # overlap=0.7, but utilization=50/64=0.78 > 0.4
        assert not _should_compress(0.7, 50, 64)

    def test_zero_nominal_rank(self):
        assert not _should_compress(0.8, 5, 0)


class TestCompressionTarget:
    """Tests for compression target rank computation."""

    def test_basic(self):
        target = _compression_target(10)
        assert target == 11  # ceil(10 * 1.1)

    def test_minimum_rank(self):
        target = _compression_target(2)
        assert target == 4  # min_rank enforced

    def test_zero_effective_rank(self):
        target = _compression_target(0)
        assert target == 4


# ---------------------------------------------------------------------------
# Per-layer recommendation tests
# ---------------------------------------------------------------------------


class TestRecommendLayer:
    """Tests for _recommend_layer() verdict → recommendation mapping."""

    def test_safe_returns_linear(self):
        lv = _make_lv_dict(verdict="safe", mean_overlap=0.05)
        rec = _recommend_layer(lv)
        assert rec.strategy == "linear"
        assert rec.coefficients == (0.5, 0.5)
        assert rec.trim_fraction == 0.0
        assert rec.risk_level == "low"

    def test_redundant_returns_ties_with_trim(self):
        lv = _make_lv_dict(verdict="redundant", mean_overlap=0.8, directional_agreement=0.8)
        rec = _recommend_layer(lv)
        assert rec.strategy == "ties"
        assert rec.trim_fraction > 0.0
        assert rec.risk_level == "medium"

    def test_conflicting_returns_dare_ties(self):
        lv = _make_lv_dict(
            verdict="conflicting",
            mean_overlap=0.8,
            directional_agreement=-0.7,
            conflict_dimensions=3,
            principal_angle_cosines=(0.9, 0.8, 0.7, 0.6),
        )
        rec = _recommend_layer(lv)
        assert rec.strategy == "dare_ties"
        assert rec.trim_fraction >= 0.15
        assert rec.risk_level == "high"

    def test_imbalanced_returns_rebalanced_linear(self):
        lv = _make_lv_dict(
            verdict="imbalanced",
            mean_overlap=0.3,
            magnitude_ratio=8.0,
            suggested_coefficients=[0.11, 0.89],
        )
        rec = _recommend_layer(lv)
        assert rec.strategy == "linear"
        assert rec.coefficients == (0.11, 0.89)
        assert rec.trim_fraction == 0.0

    def test_compress_flag_when_over_provisioned(self):
        lv = _make_lv_dict(
            verdict="redundant",
            mean_overlap=0.8,
            effective_rank_a=5,
            nominal_rank_a=64,  # utilization = 5/64 = 0.078
        )
        rec = _recommend_layer(lv)
        assert rec.compress_first is True
        assert rec.compress_target_rank_a is not None
        assert rec.compress_target_rank_a >= 5

    def test_no_compress_when_well_utilized(self):
        lv = _make_lv_dict(
            verdict="redundant",
            mean_overlap=0.8,
            effective_rank_a=50,
            nominal_rank_a=64,
        )
        rec = _recommend_layer(lv)
        assert rec.compress_target_rank_a is None

    def test_to_dict_serializable(self):
        import json

        lv = _make_lv_dict(verdict="safe")
        rec = _recommend_layer(lv)
        d = rec.to_dict()
        json_str = json.dumps(d)
        assert "strategy" in json_str
        assert "risk_level" in json_str


# ---------------------------------------------------------------------------
# Full recommendation tests
# ---------------------------------------------------------------------------


class TestRecommendMerge:
    """Tests for recommend_merge() on synthetic reports."""

    def test_all_safe_layers(self):
        layers = [_make_lv_dict(verdict="safe", layer_name=f"layer.{i}", mean_overlap=0.05) for i in range(3)]
        report = _FakeReport(layers)
        rec = recommend_merge(report)

        assert rec.overall_risk == "low"
        assert len(rec.layer_recommendations) == 3
        assert all(lr.strategy == "linear" for lr in rec.layer_recommendations)
        assert not rec.compression_needed

    def test_mixed_verdicts(self):
        layers = [
            _make_lv_dict(verdict="safe", layer_name="layer.0", mean_overlap=0.05),
            _make_lv_dict(
                verdict="conflicting",
                layer_name="layer.1",
                mean_overlap=0.8,
                directional_agreement=-0.7,
                conflict_dimensions=2,
            ),
            _make_lv_dict(
                verdict="redundant",
                layer_name="layer.2",
                mean_overlap=0.7,
                directional_agreement=0.8,
            ),
        ]
        report = _FakeReport(layers)
        rec = recommend_merge(report)

        assert rec.overall_risk == "high"  # worst case from conflicting
        assert rec.layer_recommendations[0].strategy == "linear"
        assert rec.layer_recommendations[1].strategy == "dare_ties"
        assert rec.layer_recommendations[2].strategy == "ties"

    def test_compression_needed_flag(self):
        layers = [
            _make_lv_dict(
                verdict="redundant",
                layer_name="layer.0",
                mean_overlap=0.8,
                effective_rank_a=5,
                nominal_rank_a=64,
            ),
        ]
        report = _FakeReport(layers)
        rec = recommend_merge(report)

        assert rec.compression_needed is True
        assert rec.n_layers_needing_compression >= 1

    def test_fallback_strategies_present(self):
        layers = [_make_lv_dict(verdict="safe", layer_name="layer.0")]
        report = _FakeReport(layers)
        rec = recommend_merge(report)

        assert len(rec.fallback_strategies) > 0

    def test_to_dict_serializable(self):
        import json

        layers = [
            _make_lv_dict(verdict="safe", layer_name="layer.0"),
            _make_lv_dict(verdict="redundant", layer_name="layer.1", mean_overlap=0.7),
        ]
        report = _FakeReport(layers)
        rec = recommend_merge(report)
        d = rec.to_dict()
        json_str = json.dumps(d)
        assert "overall_strategy" in json_str
        assert "layer_recommendations" in json_str


# ---------------------------------------------------------------------------
# CLI formatting tests
# ---------------------------------------------------------------------------


class TestFormatRecommendation:
    """Tests for format_recommendation() output."""

    def test_contains_strategy_header(self):
        layers = [_make_lv_dict(verdict="safe", layer_name="layer.0")]
        report = _FakeReport(layers)
        rec = recommend_merge(report)
        output = format_recommendation(rec)

        assert "MERGE STRATEGY RECOMMENDATION" in output
        assert "audit_aware" in output

    def test_contains_per_layer_table(self):
        layers = [
            _make_lv_dict(verdict="safe", layer_name="model.layers.0.self_attn.q_proj"),
            _make_lv_dict(
                verdict="conflicting",
                layer_name="model.layers.1.self_attn.q_proj",
                mean_overlap=0.8,
                conflict_dimensions=2,
            ),
        ]
        report = _FakeReport(layers)
        rec = recommend_merge(report)
        output = format_recommendation(rec)

        assert "L0.self_attn.q_proj" in output
        assert "L1.self_attn.q_proj" in output
        assert "dare_ties" in output

    def test_compression_section_when_needed(self):
        layers = [
            _make_lv_dict(
                verdict="redundant",
                layer_name="layer.0",
                mean_overlap=0.8,
                effective_rank_a=5,
                nominal_rank_a=64,
            ),
        ]
        report = _FakeReport(layers)
        rec = recommend_merge(report)
        output = format_recommendation(rec, adapter_a_path="./chat-lora")

        assert "Pre-compression recommended" in output
        assert "gradience truncate" in output
        assert "./chat-lora" in output

    def test_no_compression_section_when_not_needed(self):
        layers = [_make_lv_dict(verdict="safe", layer_name="layer.0")]
        report = _FakeReport(layers)
        rec = recommend_merge(report)
        output = format_recommendation(rec)

        assert "Pre-compression" not in output

    def test_contains_merge_command(self):
        layers = [_make_lv_dict(verdict="safe", layer_name="layer.0")]
        report = _FakeReport(layers)
        rec = recommend_merge(report)
        output = format_recommendation(rec)

        assert "gradience merge-plan" in output
        assert "--strategy audit_aware" in output

    def test_custom_adapter_paths(self):
        layers = [_make_lv_dict(verdict="safe", layer_name="layer.0")]
        report = _FakeReport(layers)
        rec = recommend_merge(report)
        output = format_recommendation(
            rec,
            adapter_a_path="./my-chat-adapter",
            adapter_b_path="./my-code-adapter",
        )

        assert "./my-chat-adapter" in output
        assert "./my-code-adapter" in output


# ---------------------------------------------------------------------------
# Eligibility hard-warning tests
# ---------------------------------------------------------------------------


class TestEligibilityHardWarnings:
    """Tests for hard warnings surfaced from source adapter eligibility."""

    def test_no_source_qa_warns_structural_only(self):
        """When no source_qa is provided, warn that recommendation is structural only."""
        layers = [_make_lv_dict(verdict="safe", layer_name="layer.0")]
        report = _FakeReport(layers, source_qa=None)
        rec = recommend_merge(report)

        assert len(rec.warnings) == 1
        assert "No source-eligibility data provided" in rec.warnings[0]
        assert "structural balance only" in rec.warnings[0]

    def test_empty_source_qa_warns_structural_only(self):
        """Empty source_qa dict is treated the same as None."""
        layers = [_make_lv_dict(verdict="safe", layer_name="layer.0")]
        report = _FakeReport(layers, source_qa={})
        rec = recommend_merge(report)

        assert len(rec.warnings) == 1
        assert "No source-eligibility data provided" in rec.warnings[0]

    def test_one_adapter_flagged_weak(self):
        """When one adapter is flagged weak, warn about preserving weak adapter."""
        layers = [_make_lv_dict(verdict="safe", layer_name="layer.0")]
        report = _FakeReport(
            layers,
            source_qa={
                "adapter_a": {"status": "flagged_weak", "adapter_path": "./a"},
                "adapter_b": {"status": "eligible", "adapter_path": "./b"},
            },
        )
        rec = recommend_merge(report)

        assert len(rec.warnings) == 1
        assert "Structural rebalance may preserve a behaviorally weak adapter" in rec.warnings[0]

    def test_both_adapters_flagged_weak(self):
        """When both adapters are flagged weak, warn about uncertain deployment value."""
        layers = [_make_lv_dict(verdict="safe", layer_name="layer.0")]
        report = _FakeReport(
            layers,
            source_qa={
                "adapter_a": {"status": "flagged_weak", "adapter_path": "./a"},
                "adapter_b": {"status": "flagged_weak", "adapter_path": "./b"},
            },
        )
        rec = recommend_merge(report)

        assert len(rec.warnings) == 1
        assert "Both source adapters underperform base" in rec.warnings[0]
        assert "deployment value is uncertain" in rec.warnings[0]

    def test_both_eligible_no_warnings(self):
        """When both adapters are eligible, no hard warnings are emitted."""
        layers = [_make_lv_dict(verdict="safe", layer_name="layer.0")]
        report = _FakeReport(
            layers,
            source_qa={
                "adapter_a": {"status": "eligible", "adapter_path": "./a"},
                "adapter_b": {"status": "eligible", "adapter_path": "./b"},
            },
        )
        rec = recommend_merge(report)

        assert len(rec.warnings) == 0

    def test_warnings_in_to_dict(self):
        """Warnings are serialized in to_dict()."""
        layers = [_make_lv_dict(verdict="safe", layer_name="layer.0")]
        report = _FakeReport(layers, source_qa=None)
        rec = recommend_merge(report)
        d = rec.to_dict()

        assert "warnings" in d
        assert len(d["warnings"]) == 1

    def test_warnings_in_format_output(self):
        """Warnings appear in CLI-formatted output."""
        layers = [_make_lv_dict(verdict="safe", layer_name="layer.0")]
        report = _FakeReport(
            layers,
            source_qa={
                "adapter_a": {"status": "flagged_weak", "adapter_path": "./a"},
                "adapter_b": {"status": "eligible", "adapter_path": "./b"},
            },
        )
        rec = recommend_merge(report)
        output = format_recommendation(rec)

        assert "Warnings:" in output
        assert "Structural rebalance may preserve a behaviorally weak adapter" in output


# ---------------------------------------------------------------------------
# Coefficient utility tests
# ---------------------------------------------------------------------------


class TestRebalanceCoefficients:
    """Tests for the rebalance_coefficients utility."""

    def test_ratio_1_gives_equal(self):
        """No imbalance → equal coefficients."""
        a, b = rebalance_coefficients(1.0)
        assert a == b == 0.5

    def test_ratio_10(self):
        """10x imbalance → strong adapter gets ~0.09, weak gets ~0.91."""
        a, b = rebalance_coefficients(10.0)
        assert a < 0.15
        assert b > 0.85
        assert abs(a + b - 1.0) < 1e-3

    def test_sum_to_one(self):
        """Coefficients always sum to ~1.0."""
        for ratio in [1.0, 2.0, 5.0, 10.0, 100.0]:
            a, b = rebalance_coefficients(ratio)
            assert abs(a + b - 1.0) < 1e-3

    def test_monotonic(self):
        """As ratio increases, coeff_strong decreases."""
        prev_a = 1.0
        for ratio in [1.0, 2.0, 5.0, 10.0]:
            a, _ = rebalance_coefficients(ratio)
            assert a <= prev_a
            prev_a = a


class TestNormEqualizedCoefficients:
    """Tests for the norm_equalized_coefficients utility."""

    def test_equal_norms_gives_half(self):
        """Equal Frobenius norms → (0.5, 0.5) with default weight."""
        a, b = norm_equalized_coefficients(5.0, 5.0)
        assert a == pytest.approx(0.5, abs=1e-3)
        assert b == pytest.approx(0.5, abs=1e-3)

    def test_imbalanced_norms(self):
        """10x imbalance → A scaled up, B scaled down."""
        a, b = norm_equalized_coefficients(1.0, 10.0)
        # A has smaller norm, so its effective coefficient should be > 0.5
        # B has larger norm, so its effective coefficient should be < 0.5
        assert a > 0.5
        assert b < 0.5

    def test_custom_weight(self):
        """Custom weight_a is respected."""
        a1, b1 = norm_equalized_coefficients(5.0, 5.0, weight_a=0.7)
        assert a1 == pytest.approx(0.7, abs=1e-3)
        assert b1 == pytest.approx(0.3, abs=1e-3)


# ---------------------------------------------------------------------------
# Norm-equalized baseline in recommendations
# ---------------------------------------------------------------------------


class TestNormEqualizedBaseline:
    """Tests that norm_equalized appears as a recommended baseline."""

    def test_norm_equalized_in_fallbacks(self):
        """norm_equalized appears in fallback strategies."""
        layers = [_make_lv_dict(verdict="safe", layer_name="layer.0")]
        report = _FakeReport(layers)
        rec = recommend_merge(report)

        assert "norm_equalized" in rec.fallback_strategies

    def test_norm_equalized_baseline_in_format(self):
        """CLI output includes norm-equalized baseline command."""
        layers = [_make_lv_dict(verdict="safe", layer_name="layer.0")]
        report = _FakeReport(layers)
        rec = recommend_merge(report)
        output = format_recommendation(rec)

        assert "Norm-equalized baseline" in output
        assert "--strategy norm_equalized" in output


# ---------------------------------------------------------------------------
# Stage A — Diagnosis tests
# ---------------------------------------------------------------------------


class TestDiagnoseLayer:
    """Tests for diagnose_layer (Stage A)."""

    def test_returns_layer_diagnosis(self):
        """diagnose_layer produces a LayerDiagnosis, not a recommendation."""
        lv = _make_lv_dict(verdict="safe", mean_overlap=0.05)
        diag = diagnose_layer(lv)

        assert isinstance(diag, LayerDiagnosis)
        assert diag.verdict == "safe"
        assert diag.risk_level == "low"
        assert diag.mean_overlap == pytest.approx(0.05)

    def test_extracts_all_metrics(self):
        """All key spectral metrics are extracted."""
        lv = _make_lv_dict(
            verdict="conflicting",
            mean_overlap=0.8,
            directional_agreement=-0.7,
            magnitude_ratio=3.0,
            conflict_dimensions=2,
        )
        diag = diagnose_layer(lv)

        assert diag.mean_overlap == pytest.approx(0.8)
        assert diag.directional_agreement == pytest.approx(-0.7)
        assert diag.magnitude_ratio == pytest.approx(3.0)
        assert diag.n_conflict == 2
        assert diag.risk_level == "high"

    def test_compression_diagnosed(self):
        """Compression need is diagnosed without choosing a strategy."""
        lv = _make_lv_dict(
            verdict="redundant",
            mean_overlap=0.8,
            effective_rank_a=5,
            nominal_rank_a=64,
        )
        diag = diagnose_layer(lv)

        assert diag.compress_first is True
        assert diag.compress_target_rank_a is not None
        assert diag.compress_target_rank_a >= 5

    def test_suggested_coefficients_preserved(self):
        """Upstream suggested_coefficients are passed through."""
        lv = _make_lv_dict(
            verdict="imbalanced",
            suggested_coefficients=[0.2, 0.8],
        )
        diag = diagnose_layer(lv)

        assert diag.suggested_coefficients == (0.2, 0.8)

    def test_no_suggested_coefficients(self):
        """None when no upstream coefficients."""
        lv = _make_lv_dict(verdict="safe")
        diag = diagnose_layer(lv)

        assert diag.suggested_coefficients is None

    def test_to_dict_serializable(self):
        """LayerDiagnosis serializes to JSON-compatible dict."""
        import json

        lv = _make_lv_dict(verdict="safe")
        diag = diagnose_layer(lv)
        d = diag.to_dict()
        json_str = json.dumps(d)
        assert "verdict" in json_str
        assert "risk_level" in json_str
        assert "mean_overlap" in json_str
        assert "over_accumulation_score" in json_str

    def test_extracts_over_accumulation_fields(self):
        """Over-accumulation fields are propagated from layer verdicts."""
        lv = _make_lv_dict(
            verdict="redundant",
            over_accumulation_score=0.53,
            over_accumulation_band="watch",
        )
        diag = diagnose_layer(lv)
        assert diag.over_accumulation_score == pytest.approx(0.53)
        assert diag.over_accumulation_band == "watch"


class TestEligibilityContext:
    """Tests for EligibilityContext."""

    def test_no_data(self):
        ctx = EligibilityContext(status_a=None, status_b=None)
        assert not ctx.has_data
        assert not ctx.any_weak
        assert not ctx.both_weak
        assert not ctx.both_eligible

    def test_both_eligible(self):
        ctx = EligibilityContext(
            status_a=EligibilityStatus.ELIGIBLE,
            status_b=EligibilityStatus.ELIGIBLE,
        )
        assert ctx.has_data
        assert ctx.both_eligible
        assert not ctx.any_weak

    def test_one_weak(self):
        ctx = EligibilityContext(
            status_a=EligibilityStatus.FLAGGED_WEAK,
            status_b=EligibilityStatus.ELIGIBLE,
        )
        assert ctx.any_weak
        assert not ctx.both_weak

    def test_both_weak(self):
        ctx = EligibilityContext(
            status_a=EligibilityStatus.FLAGGED_WEAK,
            status_b=EligibilityStatus.FLAGGED_WEAK,
        )
        assert ctx.both_weak
        assert ctx.any_weak


class TestDiagnosePair:
    """Tests for diagnose_pair (full Stage A)."""

    def test_returns_pair_diagnosis(self):
        """diagnose_pair produces a PairDiagnosis."""
        layers = [
            _make_lv_dict(verdict="safe", layer_name="layer.0", mean_overlap=0.05),
            _make_lv_dict(verdict="conflicting", layer_name="layer.1", mean_overlap=0.8),
        ]
        report = _FakeReport(layers)
        diag = diagnose_pair(report)

        assert isinstance(diag, PairDiagnosis)
        assert len(diag.layer_diagnoses) == 2
        assert diag.layer_diagnoses[0].verdict == "safe"
        assert diag.layer_diagnoses[1].verdict == "conflicting"

    def test_overall_risk_worst_case(self):
        """Overall risk is the worst across layers."""
        layers = [
            _make_lv_dict(verdict="safe", layer_name="layer.0"),
            _make_lv_dict(verdict="conflicting", layer_name="layer.1", mean_overlap=0.8),
        ]
        report = _FakeReport(layers)
        diag = diagnose_pair(report)

        assert diag.overall_risk == "high"

    def test_compression_aggregated(self):
        """Compression needs are aggregated."""
        layers = [
            _make_lv_dict(
                verdict="redundant",
                layer_name="layer.0",
                mean_overlap=0.8,
                effective_rank_a=5,
                nominal_rank_a=64,
            ),
            _make_lv_dict(verdict="safe", layer_name="layer.1"),
        ]
        report = _FakeReport(layers)
        diag = diagnose_pair(report)

        assert diag.compression_needed is True
        assert diag.n_layers_needing_compression == 1

    def test_eligibility_parsed(self):
        """Source QA is parsed into EligibilityContext."""
        layers = [_make_lv_dict(verdict="safe", layer_name="layer.0")]
        report = _FakeReport(
            layers,
            source_qa={
                "adapter_a": {"status": "eligible", "adapter_path": "./a"},
                "adapter_b": {"status": "flagged_weak", "adapter_path": "./b"},
            },
        )
        diag = diagnose_pair(report)

        assert diag.eligibility.has_data
        assert diag.eligibility.status_a == EligibilityStatus.ELIGIBLE
        assert diag.eligibility.status_b == EligibilityStatus.FLAGGED_WEAK
        assert diag.eligibility.any_weak

    def test_no_source_qa_produces_empty_context(self):
        """Missing source_qa produces a no-data EligibilityContext."""
        layers = [_make_lv_dict(verdict="safe", layer_name="layer.0")]
        report = _FakeReport(layers, source_qa=None)
        diag = diagnose_pair(report)

        assert not diag.eligibility.has_data

    def test_to_dict_serializable(self):
        """PairDiagnosis serializes to JSON."""
        import json

        layers = [_make_lv_dict(verdict="safe", layer_name="layer.0")]
        report = _FakeReport(
            layers,
            source_qa={
                "adapter_a": {"status": "eligible", "adapter_path": "./a"},
            },
        )
        diag = diagnose_pair(report)
        d = diag.to_dict()
        json_str = json.dumps(d)
        assert "layer_diagnoses" in json_str
        assert "eligibility" in json_str
        assert "over_accumulation_advisory" in json_str

    def test_over_accumulation_advisory_aggregated(self):
        """Pair diagnosis aggregates over-accumulation watch/elevated state."""
        layers = [
            _make_lv_dict(
                verdict="redundant",
                layer_name="layer.0",
                over_accumulation_score=0.65,
                over_accumulation_band="high",
            ),
            _make_lv_dict(
                verdict="safe",
                layer_name="layer.1",
                over_accumulation_score=0.20,
                over_accumulation_band="low",
            ),
        ]
        report = _FakeReport(layers)
        diag = diagnose_pair(report)
        assert diag.over_accumulation_advisory == "elevated"
        assert diag.over_accumulation_high_risk_layers == 1


# ---------------------------------------------------------------------------
# Stage B — Policy tests
# ---------------------------------------------------------------------------


class TestApplyLayerPolicy:
    """Tests for _apply_layer_policy (Stage B per-layer)."""

    def test_safe_gets_linear(self):
        """SAFE diagnosis → linear strategy."""
        diag = diagnose_layer(_make_lv_dict(verdict="safe", mean_overlap=0.05))
        ctx = EligibilityContext(status_a=None, status_b=None)
        rec = _apply_layer_policy(diag, ctx)

        assert isinstance(rec, LayerRecommendation)
        assert rec.strategy == "linear"
        assert rec.coefficients == (0.5, 0.5)

    def test_conflicting_gets_dare_ties(self):
        """CONFLICTING diagnosis → dare_ties strategy."""
        diag = diagnose_layer(
            _make_lv_dict(
                verdict="conflicting",
                mean_overlap=0.8,
                conflict_dimensions=3,
            )
        )
        ctx = EligibilityContext(status_a=None, status_b=None)
        rec = _apply_layer_policy(diag, ctx)

        assert rec.strategy == "dare_ties"
        assert rec.trim_fraction >= 0.15

    def test_policy_receives_eligibility(self):
        """Policy function receives eligibility context (hook for future use)."""
        diag = diagnose_layer(_make_lv_dict(verdict="safe"))
        ctx = EligibilityContext(
            status_a=EligibilityStatus.FLAGGED_WEAK,
            status_b=EligibilityStatus.ELIGIBLE,
        )
        rec = _apply_layer_policy(diag, ctx)

        # Currently policy doesn't change per-layer strategy based on
        # eligibility, but the hook is there.  Just verify it runs.
        assert isinstance(rec, LayerRecommendation)


class TestApplyPolicy:
    """Tests for _apply_policy (Stage B aggregate)."""

    def test_composes_diagnosis_to_recommendation(self):
        """_apply_policy converts PairDiagnosis → MergeRecommendation."""
        layers = [_make_lv_dict(verdict="safe", layer_name="layer.0")]
        report = _FakeReport(layers)
        diag = diagnose_pair(report)
        rec = _apply_policy(diag)

        assert isinstance(rec, MergeRecommendation)
        assert len(rec.layer_recommendations) == 1
        assert rec.overall_risk == "low"

    def test_eligibility_warnings_propagated(self):
        """Warnings from eligibility context flow through to recommendation."""
        layers = [_make_lv_dict(verdict="safe", layer_name="layer.0")]
        report = _FakeReport(
            layers,
            source_qa={
                "adapter_a": {"status": "flagged_weak", "adapter_path": "./a"},
                "adapter_b": {"status": "eligible", "adapter_path": "./b"},
            },
        )
        diag = diagnose_pair(report)
        rec = _apply_policy(diag)

        assert len(rec.warnings) == 1
        assert "weak adapter" in rec.warnings[0]

    def test_equivalent_to_recommend_merge(self):
        """diagnose_pair + _apply_policy == recommend_merge."""
        layers = [
            _make_lv_dict(verdict="safe", layer_name="layer.0"),
            _make_lv_dict(verdict="redundant", layer_name="layer.1", mean_overlap=0.7),
        ]
        report = _FakeReport(layers)

        # Two-stage path
        diag = diagnose_pair(report)
        rec_staged = _apply_policy(diag)

        # One-shot path
        rec_direct = recommend_merge(report)

        # Should produce identical recommendations
        assert rec_staged.to_dict() == rec_direct.to_dict()
