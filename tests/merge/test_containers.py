"""Tests for gradience.vnext.merge.containers — typed data containers."""

from __future__ import annotations

import pytest

from gradience.vnext.merge.containers import (
    AdapterMetadata,
    AggregateResult,
    MatchingSummary,
    MergeWarning,
    PairAuditResult,
    RecommendationResult,
    WarningCode,
    _classify_warning_string,
)

# ---------------------------------------------------------------------------
# AdapterMetadata
# ---------------------------------------------------------------------------


class TestAdapterMetadata:
    def test_attribute_access(self):
        m = AdapterMetadata(path="/tmp/a", rank=8, alpha=16.0, base_model="llama", n_layers=32)
        assert m.path == "/tmp/a"
        assert m.rank == 8
        assert m.alpha == 16.0
        assert m.base_model == "llama"
        assert m.n_layers == 32

    def test_dict_compat(self):
        m = AdapterMetadata(path="/tmp/a", rank=8, alpha=16.0)
        assert m["path"] == "/tmp/a"
        assert m["rank"] == 8
        assert m.get("base_model") == "unknown"
        assert "path" in m

    def test_to_dict_roundtrip(self):
        m = AdapterMetadata(path="/tmp/a", rank=8, alpha=16.0, base_model="llama", n_layers=32)
        d = m.to_dict()
        restored = AdapterMetadata.from_dict(d)
        assert restored == m

    def test_from_dict_defaults(self):
        m = AdapterMetadata.from_dict({})
        assert m.path == "unknown"
        assert m.rank == 0
        assert m.base_model == "unknown"

    def test_frozen(self):
        m = AdapterMetadata(path="/tmp/a", rank=8, alpha=16.0)
        with pytest.raises(AttributeError):
            m.rank = 16  # type: ignore[misc]


# ---------------------------------------------------------------------------
# MatchingSummary
# ---------------------------------------------------------------------------


class TestMatchingSummary:
    def test_attribute_access(self):
        m = MatchingSummary(n_shared=10, n_only_a=2, n_only_b=1, shared_layers=("a", "b"))
        assert m.n_shared == 10
        assert m.shared_layers == ("a", "b")

    def test_dict_compat(self):
        m = MatchingSummary(n_shared=5)
        assert m["n_shared"] == 5
        assert m.get("n_only_a") == 0
        assert "n_shared" in m

    def test_roundtrip(self):
        m = MatchingSummary(n_shared=5, shared_layers=("x", "y"))
        restored = MatchingSummary.from_dict(m.to_dict())
        assert restored.n_shared == m.n_shared
        assert restored.shared_layers == m.shared_layers


# ---------------------------------------------------------------------------
# AggregateResult
# ---------------------------------------------------------------------------


class TestAggregateResult:
    def test_attribute_access(self):
        a = AggregateResult(overall_verdict="safe", compatibility_score=0.85, n_safe=10, n_conflicting=2)
        assert a.overall_verdict == "safe"
        assert a.compatibility_score == 0.85
        assert a.n_safe == 10
        assert a.n_conflicting == 2

    def test_dict_compat(self):
        a = AggregateResult(overall_verdict="safe", compatibility_score=0.5)
        assert a["overall_verdict"] == "safe"
        assert a.get("n_safe") == 0
        assert "compatibility_score" in a

    def test_roundtrip(self):
        a = AggregateResult(
            overall_verdict="conflicting",
            compatibility_score=0.3,
            mean_overlap=0.6,
            n_conflicting=5,
            mean_magnitude_ratio=3.2,
        )
        restored = AggregateResult.from_dict(a.to_dict())
        assert restored.overall_verdict == a.overall_verdict
        assert restored.n_conflicting == a.n_conflicting
        assert restored.mean_magnitude_ratio == pytest.approx(a.mean_magnitude_ratio)

    def test_contains_and_iteration(self):
        a = AggregateResult(overall_verdict="safe", compatibility_score=0.5)
        assert "overall_verdict" in a
        assert "nonexistent_key" not in a
        keys = list(a)
        assert "overall_verdict" in keys
        assert "compatibility_score" in keys


# ---------------------------------------------------------------------------
# WarningCode and MergeWarning
# ---------------------------------------------------------------------------


class TestWarningCode:
    def test_enum_values(self):
        assert WarningCode.NO_ELIGIBILITY_DATA == "no_eligibility_data"
        assert WarningCode.ADAPTER_WEAK == "adapter_weak"
        assert WarningCode.BOTH_ADAPTERS_WEAK == "both_adapters_weak"
        assert WarningCode.HIGH_STRUCTURAL_RISK == "high_structural_risk"

    def test_string_classification(self):
        assert _classify_warning_string("both adapters underperform") == WarningCode.BOTH_ADAPTERS_WEAK
        assert _classify_warning_string("adapter is weak") == WarningCode.ADAPTER_WEAK
        assert _classify_warning_string("No source-eligibility data") == WarningCode.NO_ELIGIBILITY_DATA
        assert _classify_warning_string("only in adapter a") == WarningCode.LAYERS_ONLY_IN_A
        assert _classify_warning_string("only in adapter b") == WarningCode.LAYERS_ONLY_IN_B
        assert _classify_warning_string("compression recommended") == WarningCode.COMPRESSION_NEEDED


class TestMergeWarning:
    def test_creation(self):
        w = MergeWarning(code=WarningCode.ADAPTER_WEAK, message="Adapter A is weak", adapter="a")
        assert w.code == WarningCode.ADAPTER_WEAK
        assert w.adapter == "a"
        assert w.severity == "warning"

    def test_roundtrip(self):
        w = MergeWarning(
            code=WarningCode.HIGH_STRUCTURAL_RISK,
            message="High risk merge",
            severity="error",
        )
        d = w.to_dict()
        restored = MergeWarning.from_dict(d)
        assert restored.code == w.code
        assert restored.message == w.message
        assert restored.severity == w.severity

    def test_no_adapter_key_when_none(self):
        w = MergeWarning(code=WarningCode.NO_ELIGIBILITY_DATA, message="No data")
        d = w.to_dict()
        assert "adapter" not in d


# ---------------------------------------------------------------------------
# PairAuditResult
# ---------------------------------------------------------------------------


class TestPairAuditResult:
    def test_from_audit_report_typed(self):
        """Converts a report with typed fields."""
        from gradience.vnext.merge.report import MergeAuditReport

        report = MergeAuditReport(
            adapter_a=AdapterMetadata(path="/tmp/a", rank=8, alpha=16.0),
            adapter_b=AdapterMetadata(path="/tmp/b", rank=8, alpha=16.0),
            matching=MatchingSummary(n_shared=2),
            layer_verdicts=[{"layer_name": "l0", "verdict": "safe", "metrics": {}}],
            aggregate=AggregateResult(overall_verdict="safe", compatibility_score=0.8),
            recommendations=["merge safely"],
            warnings=["No source-eligibility data provided"],
        )
        pair = PairAuditResult.from_audit_report(report)

        assert pair.adapter_a.path == "/tmp/a"
        assert pair.adapter_b.rank == 8
        assert pair.matching.n_shared == 2
        assert pair.aggregate.overall_verdict == "safe"
        assert len(pair.recommendations) == 1
        assert len(pair.typed_warnings) == 1
        assert pair.typed_warnings[0].code == WarningCode.NO_ELIGIBILITY_DATA

    def test_to_dict(self):
        pair = PairAuditResult(
            adapter_a=AdapterMetadata(path="/tmp/a", rank=8, alpha=16.0),
            adapter_b=AdapterMetadata(path="/tmp/b", rank=8, alpha=16.0),
            matching=MatchingSummary(n_shared=1),
            aggregate=AggregateResult(overall_verdict="safe", compatibility_score=0.8),
            layer_verdicts=(),
            recommendations=(),
        )
        d = pair.to_dict()
        assert d["adapter_a"]["path"] == "/tmp/a"
        assert d["aggregate"]["overall_verdict"] == "safe"


# ---------------------------------------------------------------------------
# RecommendationResult
# ---------------------------------------------------------------------------


class TestRecommendationResult:
    def test_basic(self):
        r = RecommendationResult(
            strategy="audit_aware",
            risk="low",
            action="Merge is safe.",
            confidence="High spectral compatibility",
            dominant_issue="none",
            caveats=("No source-eligibility data.",),
            typed_warnings=(
                MergeWarning(code=WarningCode.NO_ELIGIBILITY_DATA, message="No data"),
            ),
            fallback_strategies=("norm_equalized",),
        )
        assert r.strategy == "audit_aware"
        assert r.risk == "low"
        assert len(r.typed_warnings) == 1

    def test_to_dict(self):
        r = RecommendationResult(
            strategy="audit_aware",
            risk="medium",
            action="Merge with caution.",
            confidence="Moderate",
            dominant_issue="subspace conflict",
            caveats=(),
            typed_warnings=(),
            fallback_strategies=("norm_equalized", "uniform_linear"),
        )
        d = r.to_dict()
        assert d["strategy"] == "audit_aware"
        assert d["risk"] == "medium"
        assert d["fallback_strategies"] == ["norm_equalized", "uniform_linear"]
