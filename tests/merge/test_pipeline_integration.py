"""Integration test: audit → diagnose → recommend → format pipeline.

Verifies that real adapter fixtures flow through all four stages
with structurally valid outputs at each boundary.
"""

from __future__ import annotations

from pathlib import Path

from gradience.vnext.merge import (
    diagnose_pair,
    format_recommendation,
    merge_audit,
    recommend_merge,
)

VALID_VERDICTS = {"safe", "redundant", "conflicting", "imbalanced"}
VALID_RISKS = {"low", "medium", "high"}
VALID_STRATEGIES = {"linear", "ties", "dare_ties", "dare_linear"}
LAYER_VERDICT_REQUIRED_KEYS = {"layer_name", "verdict", "confidence", "metrics"}


def _assert_report(report, *, expected_n_layers: int = 2):
    """Structural assertions on MergeAuditReport."""
    assert len(report.layer_verdicts) == expected_n_layers
    for lv in report.layer_verdicts:
        assert LAYER_VERDICT_REQUIRED_KEYS <= set(lv.keys())
        assert lv["verdict"] in VALID_VERDICTS
        assert isinstance(lv["confidence"], float)
    assert report.aggregate.overall_verdict in VALID_VERDICTS
    score = report.aggregate.compatibility_score
    assert isinstance(score, float) and 0.0 <= score <= 1.0


def _assert_diagnosis(diag, report):
    """Structural assertions on PairDiagnosis."""
    assert len(diag.layer_diagnoses) == len(report.layer_verdicts)
    for ld in diag.layer_diagnoses:
        assert ld.verdict in VALID_VERDICTS
        assert ld.risk_level in VALID_RISKS
        assert isinstance(ld.compress_first, bool)
    assert diag.overall_risk in VALID_RISKS
    assert isinstance(diag.compression_needed, bool)


def _assert_recommendation(rec, report, *, expected_strategy: str):
    """Structural assertions on MergeRecommendation."""
    assert len(rec.layer_recommendations) == len(report.layer_verdicts)
    for lr in rec.layer_recommendations:
        assert lr.strategy in VALID_STRATEGIES
        assert lr.strategy == expected_strategy
        assert isinstance(lr.coefficients, tuple) and len(lr.coefficients) == 2
        assert lr.risk_level in VALID_RISKS
        assert isinstance(lr.reasoning, str) and len(lr.reasoning) > 0
    assert rec.overall_strategy == "audit_aware"
    assert rec.overall_risk in VALID_RISKS
    assert isinstance(rec.warnings, tuple)


def _assert_formatted(text, rec):
    """Structural assertions on formatted output string."""
    assert isinstance(text, str) and len(text) > 0
    assert "MERGE STRATEGY RECOMMENDATION" in text
    # At least one shortened layer name appears
    assert "L0." in text


class TestMergeRecommendPipeline:
    """End-to-end pipeline: merge_audit → diagnose → recommend → format."""

    def test_orthogonal_safe_pipeline(self, orthogonal_pair: tuple[Path, Path]):
        dir_a, dir_b = orthogonal_pair
        report = merge_audit(str(dir_a), str(dir_b))
        _assert_report(report)
        assert report.aggregate.overall_verdict == "safe"

        diag = diagnose_pair(report)
        _assert_diagnosis(diag, report)
        assert all(ld.risk_level == "low" for ld in diag.layer_diagnoses)

        rec = recommend_merge(report)
        _assert_recommendation(rec, report, expected_strategy="linear")

        text = format_recommendation(rec)
        _assert_formatted(text, rec)

    def test_redundant_pipeline(self, redundant_pair: tuple[Path, Path]):
        dir_a, dir_b = redundant_pair
        report = merge_audit(str(dir_a), str(dir_b))
        _assert_report(report)
        assert report.aggregate.overall_verdict == "redundant"

        diag = diagnose_pair(report)
        _assert_diagnosis(diag, report)
        assert all(ld.risk_level == "medium" for ld in diag.layer_diagnoses)

        rec = recommend_merge(report)
        _assert_recommendation(rec, report, expected_strategy="ties")

        text = format_recommendation(rec)
        _assert_formatted(text, rec)

    def test_conflicting_pipeline(self, conflicting_pair: tuple[Path, Path]):
        dir_a, dir_b = conflicting_pair
        report = merge_audit(str(dir_a), str(dir_b))
        _assert_report(report)
        assert report.aggregate.overall_verdict == "conflicting"

        diag = diagnose_pair(report)
        _assert_diagnosis(diag, report)
        assert all(ld.risk_level == "high" for ld in diag.layer_diagnoses)

        rec = recommend_merge(report)
        _assert_recommendation(rec, report, expected_strategy="dare_ties")

        text = format_recommendation(rec)
        _assert_formatted(text, rec)

    def test_imbalanced_pipeline(self, imbalanced_pair: tuple[Path, Path]):
        dir_a, dir_b = imbalanced_pair
        report = merge_audit(str(dir_a), str(dir_b))
        _assert_report(report)
        # The imbalanced fixture uses identical subspaces with 10x magnitude
        # difference.  Because subspace overlap is high and directions are
        # aligned, the verdict engine classifies this as "redundant" (Branch 2
        # fires before the high-overlap imbalanced Branch 4).
        assert report.aggregate.overall_verdict == "redundant"

        diag = diagnose_pair(report)
        _assert_diagnosis(diag, report)
        assert all(ld.risk_level == "medium" for ld in diag.layer_diagnoses)

        rec = recommend_merge(report)
        _assert_recommendation(rec, report, expected_strategy="ties")

        text = format_recommendation(rec)
        _assert_formatted(text, rec)
