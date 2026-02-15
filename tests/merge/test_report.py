"""Tests for gradience.vnext.merge.report — report generation."""

from __future__ import annotations

import json

import pytest

from gradience.vnext.merge.spectral_compat import SubspaceMetrics
from gradience.vnext.merge.verdicts import (
    CompatibilityVerdict,
    LayerVerdict,
    VerdictThresholds,
    assess_layer,
)
from gradience.vnext.merge.report import (
    MergeAuditReport,
    build_report,
    to_json,
    to_markdown,
    write_reports,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_metrics(**overrides) -> SubspaceMetrics:
    defaults = dict(
        principal_angle_cosines=(0.1, 0.05),
        mean_overlap=0.05,
        max_overlap=0.1,
        directional_agreement=0.1,
        magnitude_ratio=1.2,
        frobenius_ratio=1.1,
        frobenius_norm_a=5.0,
        frobenius_norm_b=4.5,
        # Symmetric scale metrics
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


class _FakeAdapterInfo:
    """Minimal stand-in for AdapterInfo in report tests."""

    def __init__(self, path, rank, alpha, n_layers, base_model="test-model"):
        self.path = path
        self.rank = rank
        self.alpha = alpha
        self.lora_pairs = [("layer",)] * n_layers

        class _FakeConfig:
            raw = {"base_model_name_or_path": base_model}

        self.config = _FakeConfig()


def _make_report(n_layers=3, include_conflict=False) -> MergeAuditReport:
    """Build a sample report for testing."""
    info_a = _FakeAdapterInfo("/tmp/a", 8, 16.0, n_layers)
    info_b = _FakeAdapterInfo("/tmp/b", 8, 16.0, n_layers)

    layer_verdicts = []
    for i in range(n_layers):
        if include_conflict and i == 0:
            metrics = _make_metrics(
                mean_overlap=0.8,
                directional_agreement=-0.7,
                principal_angle_cosines=(0.9, 0.8),
            )
        else:
            metrics = _make_metrics()

        lv = assess_layer(f"model.layers.{i}.self_attn.q_proj", "attn", metrics)
        layer_verdicts.append(lv)

    from gradience.vnext.merge.verdicts import assess_overall

    overall, score, recs = assess_overall(layer_verdicts)

    return build_report(
        adapter_a_info=info_a,
        adapter_b_info=info_b,
        shared=[f"layer.{i}" for i in range(n_layers)],
        only_a=[],
        only_b=[],
        layer_verdicts=layer_verdicts,
        overall_verdict=overall,
        score=score,
        recommendations=recs,
        thresholds=VerdictThresholds(),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBuildReport:
    """Tests for build_report()."""

    def test_report_has_all_fields(self):
        report = _make_report()
        assert report.adapter_a is not None
        assert report.adapter_b is not None
        assert report.matching is not None
        assert report.layer_verdicts is not None
        assert report.aggregate is not None
        assert report.recommendations is not None
        assert report.timestamp != ""
        assert report.gradience_version != ""

    def test_aggregate_counts(self):
        report = _make_report(n_layers=3)
        agg = report.aggregate
        total = agg["n_safe"] + agg["n_redundant"] + agg["n_conflicting"] + agg["n_imbalanced"]
        assert total == 3

    def test_aggregate_has_symmetric_scale_metrics(self):
        """Aggregate includes the new symmetric scale metric averages."""
        report = _make_report(n_layers=3)
        agg = report.aggregate

        for key in (
            "mean_scale_bounded_ratio",
            "mean_scale_log_ratio",
            "mean_frob_bounded_ratio",
            "mean_frob_log_ratio",
            "mean_magnitude_ratio",
        ):
            assert key in agg, f"Missing aggregate key: {key}"
            assert isinstance(agg[key], float), f"{key} should be float"

        # Values should match what _make_metrics defaults produce
        assert agg["mean_scale_bounded_ratio"] == pytest.approx(0.85, abs=1e-3)
        assert agg["mean_scale_log_ratio"] == pytest.approx(0.12, abs=1e-3)
        assert agg["mean_frob_bounded_ratio"] == pytest.approx(0.90, abs=1e-3)
        assert agg["mean_frob_log_ratio"] == pytest.approx(0.08, abs=1e-3)
        assert agg["mean_magnitude_ratio"] == pytest.approx(1.2, abs=1e-3)


class TestToJson:
    """Tests for JSON serialization."""

    def test_valid_json(self):
        """to_json produces valid JSON."""
        report = _make_report()
        data = to_json(report)
        json_str = json.dumps(data, indent=2)
        parsed = json.loads(json_str)
        assert isinstance(parsed, dict)

    def test_has_schema_version(self):
        report = _make_report()
        data = to_json(report)
        assert "schema_version" in data
        assert data["schema_version"] == "gradience.merge_audit/v2"

    def test_required_keys(self):
        report = _make_report()
        data = to_json(report)
        required = [
            "timestamp", "gradience_version", "adapter_a", "adapter_b",
            "matching", "aggregate", "per_layer", "recommendations",
            "issues", "warnings", "thresholds",
        ]
        for key in required:
            assert key in data, f"Missing key: {key}"

    def test_per_layer_matches_count(self):
        report = _make_report(n_layers=5)
        data = to_json(report)
        assert len(data["per_layer"]) == 5

    def test_json_includes_symmetric_scale_in_aggregate(self):
        """JSON output contains the new symmetric scale aggregate fields."""
        report = _make_report(n_layers=2)
        data = to_json(report)
        agg = data["aggregate"]

        for key in (
            "mean_scale_bounded_ratio",
            "mean_scale_log_ratio",
            "mean_frob_bounded_ratio",
            "mean_frob_log_ratio",
            "mean_magnitude_ratio",
        ):
            assert key in agg, f"Missing JSON aggregate key: {key}"

        # Verify round-trip through JSON serialization
        json_str = json.dumps(data, indent=2)
        parsed = json.loads(json_str)
        assert parsed["aggregate"]["mean_scale_bounded_ratio"] == pytest.approx(0.85, abs=1e-3)

    def test_schema_version_is_v2(self):
        report = _make_report()
        data = to_json(report)
        assert data["schema_version"] == "gradience.merge_audit/v2"


class TestToMarkdown:
    """Tests for markdown rendering."""

    def test_has_title(self):
        report = _make_report()
        md = to_markdown(report)
        assert "# Merge Compatibility Audit" in md

    def test_has_adapter_table(self):
        report = _make_report()
        md = to_markdown(report)
        assert "Adapter A" in md
        assert "Adapter B" in md
        assert "Rank" in md

    def test_has_verdict_banner(self):
        report = _make_report()
        md = to_markdown(report)
        assert "Overall verdict" in md

    def test_has_per_layer_table(self):
        report = _make_report(n_layers=3)
        md = to_markdown(report)
        assert "Per-Layer Analysis" in md
        assert "Overlap" in md
        assert "Agreement" in md
        assert "Scale" in md

    def test_conflict_details_only_when_present(self):
        """Conflict Details section only appears when conflicts exist."""
        safe_report = _make_report(n_layers=3, include_conflict=False)
        md_safe = to_markdown(safe_report)
        assert "Conflict Details" not in md_safe

        conflict_report = _make_report(n_layers=3, include_conflict=True)
        md_conflict = to_markdown(conflict_report)
        assert "Conflict Details" in md_conflict

    def test_recommendations_section(self):
        report = _make_report()
        md = to_markdown(report)
        assert "Recommendations" in md


class TestWriteReports:
    """Tests for file writing."""

    def test_writes_both_files(self, tmp_path):
        report = _make_report()
        write_reports(report, tmp_path / "output")

        assert (tmp_path / "output" / "merge_audit.json").exists()
        assert (tmp_path / "output" / "merge_audit.md").exists()

    def test_json_file_parseable(self, tmp_path):
        report = _make_report()
        write_reports(report, tmp_path / "output")

        with open(tmp_path / "output" / "merge_audit.json") as f:
            data = json.load(f)
        assert data["schema_version"] == "gradience.merge_audit/v2"

    def test_creates_parent_dirs(self, tmp_path):
        report = _make_report()
        deep_path = tmp_path / "a" / "b" / "c"
        write_reports(report, deep_path)
        assert (deep_path / "merge_audit.json").exists()
