"""Tests for gradience.vnext.merge.qa_report — QA report builder and formatter."""

from __future__ import annotations

import glob
import json
from pathlib import Path
from typing import Any

import pytest

from gradience.exceptions import QASchemaError
from gradience.vnext.merge.containers import AdapterMetadata, AggregateResult
from gradience.vnext.merge.qa_report import (
    CORE_SPACE_STATUS_VALUES,
    DOMINANT_ISSUE_LABELS,
    SCHEMA_ID,
    AdapterSummary,
    CoreSpaceSummary,
    MergeQAReport,
    build_qa_report,
    format_qa_report,
)

# ---------------------------------------------------------------------------
# Helpers — minimal report stand-in
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


class _FakeCoreSpace:
    """Simple stand-in for core-space pair diagnostic."""

    def __init__(
        self,
        shared_basis_score_mean: float = 0.72,
        basis_distortion_mean: float = 0.18,
        effective_shared_rank_median: int = 6,
        status: str = "compatible",
    ):
        self.shared_basis_score_mean = shared_basis_score_mean
        self.basis_distortion_mean = basis_distortion_mean
        self.effective_shared_rank_median = effective_shared_rank_median
        self.status = status


class _FakeReport:
    """Minimal stand-in for MergeAuditReport."""

    def __init__(self, layer_verdicts, source_qa=None, aggregate=None, core_space=None):
        self.layer_verdicts = layer_verdicts
        self.source_qa = source_qa
        self.core_space = core_space
        n_safe = sum(1 for lv in layer_verdicts if lv["verdict"] == "safe")
        n_redundant = sum(1 for lv in layer_verdicts if lv["verdict"] == "redundant")
        n_conflicting = sum(1 for lv in layer_verdicts if lv["verdict"] == "conflicting")
        n_imbalanced = sum(1 for lv in layer_verdicts if lv["verdict"] == "imbalanced")
        overlaps = [lv["metrics"]["mean_overlap"] for lv in layer_verdicts]
        mag_ratios = [lv["metrics"]["magnitude_ratio"] for lv in layer_verdicts]
        if aggregate is not None and isinstance(aggregate, dict):
            self.aggregate = AggregateResult.from_dict(aggregate)
        elif aggregate is not None:
            self.aggregate = aggregate
        else:
            self.aggregate = AggregateResult(
                overall_verdict="safe",
                compatibility_score=0.75,
                mean_overlap=sum(overlaps) / len(overlaps) if overlaps else 0.0,
                median_overlap=sorted(overlaps)[len(overlaps) // 2] if overlaps else 0.0,
                max_overlap=max(overlaps) if overlaps else 0.0,
                mean_agreement=0.5,
                n_safe=n_safe,
                n_redundant=n_redundant,
                n_conflicting=n_conflicting,
                n_imbalanced=n_imbalanced,
                mean_magnitude_ratio=sum(mag_ratios) / len(mag_ratios) if mag_ratios else 1.0,
            )
        self.adapter_a = AdapterMetadata(
            path="/tmp/adapter_a",
            rank=8,
            alpha=16.0,
            base_model="meta-llama/Llama-2-7b",
            n_layers=len(layer_verdicts),
        )
        self.adapter_b = AdapterMetadata(
            path="/tmp/adapter_b",
            rank=8,
            alpha=16.0,
            base_model="meta-llama/Llama-2-7b",
            n_layers=len(layer_verdicts),
        )


# ---------------------------------------------------------------------------
# Tests — AdapterSummary
# ---------------------------------------------------------------------------


class TestAdapterSummary:
    def test_to_dict_roundtrip(self):
        s = AdapterSummary(
            path="/tmp/a",
            rank=8,
            alpha=16.0,
            n_layers=32,
            base_model="llama",
            eligibility_status="eligible",
        )
        d = s.to_dict()
        assert d["path"] == "/tmp/a"
        assert d["rank"] == 8
        assert d["eligibility_status"] == "eligible"


# ---------------------------------------------------------------------------
# Tests — build_qa_report
# ---------------------------------------------------------------------------


class TestBuildQAReport:
    def test_all_safe_layers(self):
        """All-safe pair produces low risk, no dominant issue."""
        report = _FakeReport(
            [
                _make_lv_dict("safe", layer_name="model.layers.0.self_attn.q_proj"),
                _make_lv_dict("safe", layer_name="model.layers.0.self_attn.v_proj"),
            ]
        )
        qa = build_qa_report(report)

        assert qa.pair_risk == "low"
        assert qa.dominant_issue == "none"
        assert qa.dominant_issue_detail  # non-empty
        assert qa.adapter_a.rank == 8
        assert qa.adapter_b.rank == 8
        assert qa.verdict_distribution["safe"] == 2
        assert qa.verdict_distribution["conflicting"] == 0

    def test_conflicting_layers_high_risk(self):
        """Conflicting layers produce high risk."""
        report = _FakeReport(
            [
                _make_lv_dict(
                    "conflicting",
                    mean_overlap=0.6,
                    directional_agreement=-0.5,
                    conflict_dimensions=3,
                    layer_name="model.layers.0.self_attn.q_proj",
                ),
            ]
        )
        qa = build_qa_report(report)

        assert qa.pair_risk == "high"
        assert qa.dominant_issue == "subspace_conflict"
        assert "caution" in qa.recommended_action.lower() or "dare" in qa.recommended_action.lower()

    def test_imbalanced_layers_norm_issue(self):
        """Imbalanced layers flag norm imbalance as dominant issue."""
        report = _FakeReport(
            [
                _make_lv_dict(
                    "imbalanced",
                    magnitude_ratio=8.0,
                    layer_name="model.layers.0.self_attn.q_proj",
                ),
            ],
            aggregate={
                "overall_verdict": "imbalanced",
                "compatibility_score": 0.4,
                "mean_overlap": 0.05,
                "median_overlap": 0.05,
                "max_overlap": 0.1,
                "mean_agreement": 0.5,
                "n_safe": 0,
                "n_redundant": 0,
                "n_conflicting": 0,
                "n_imbalanced": 1,
                "mean_magnitude_ratio": 8.0,
            },
        )
        qa = build_qa_report(report)

        assert qa.dominant_issue == "norm_imbalance"
        assert qa.verdict_distribution["imbalanced"] == 1

    def test_redundant_layers(self):
        report = _FakeReport(
            [
                _make_lv_dict(
                    "redundant",
                    mean_overlap=0.8,
                    directional_agreement=0.7,
                    layer_name="model.layers.0.self_attn.q_proj",
                ),
                _make_lv_dict(
                    "redundant",
                    mean_overlap=0.9,
                    directional_agreement=0.8,
                    layer_name="model.layers.0.self_attn.v_proj",
                ),
            ]
        )
        qa = build_qa_report(report)

        assert qa.dominant_issue in ("high_redundancy", "partial_redundancy")

    def test_eligibility_not_provided(self):
        """No source QA → eligibility_status is None."""
        report = _FakeReport([_make_lv_dict("safe")])
        qa = build_qa_report(report)

        assert qa.adapter_a.eligibility_status is None
        assert qa.adapter_b.eligibility_status is None
        assert any("no source-eligibility" in c.lower() for c in qa.caveats)

    def test_eligibility_both_eligible(self):
        """Both eligible adapters → clean eligibility labels."""
        report = _FakeReport(
            [_make_lv_dict("safe")],
            source_qa={
                "adapter_a": {"status": "eligible"},
                "adapter_b": {"status": "eligible"},
            },
        )
        qa = build_qa_report(report)

        assert qa.adapter_a.eligibility_status == "eligible"
        assert qa.adapter_b.eligibility_status == "eligible"
        assert "verified" in qa.confidence_note.lower() or "both" in qa.confidence_note.lower()

    def test_eligibility_one_weak(self):
        """One weak adapter → caveat mentions it."""
        report = _FakeReport(
            [_make_lv_dict("safe")],
            source_qa={
                "adapter_a": {"status": "eligible"},
                "adapter_b": {"status": "flagged_weak"},
            },
        )
        qa = build_qa_report(report)

        assert qa.adapter_b.eligibility_status == "flagged_weak"
        assert any("underperform" in c.lower() or "weak" in c.lower() for c in qa.caveats)

    def test_eligibility_both_weak(self):
        """Both weak → recommended action says to reconsider."""
        report = _FakeReport(
            [_make_lv_dict("safe")],
            source_qa={
                "adapter_a": {"status": "flagged_weak"},
                "adapter_b": {"status": "flagged_weak"},
            },
        )
        qa = build_qa_report(report)

        assert "reconsider" in qa.recommended_action.lower()
        assert any("both" in c.lower() and "weak" in c.lower() for c in qa.caveats)

    def test_task_relationship_advisory_same_task(self):
        """Same eval_dataset → no advisory."""
        report = _FakeReport(
            [_make_lv_dict("safe")],
            source_qa={
                "adapter_a": {"status": "eligible", "eval_dataset": "qnli_dev"},
                "adapter_b": {"status": "eligible", "eval_dataset": "qnli_dev"},
            },
        )
        qa = build_qa_report(report)
        assert qa.task_relationship_advisory is None

    def test_task_relationship_advisory_different_task(self):
        """Different eval_dataset → advisory present."""
        report = _FakeReport(
            [_make_lv_dict("safe")],
            source_qa={
                "adapter_a": {"status": "eligible", "eval_dataset": "qnli_dev"},
                "adapter_b": {"status": "eligible", "eval_dataset": "rte_dev"},
            },
        )
        qa = build_qa_report(report)
        assert qa.task_relationship_advisory is not None
        assert "cross-task" in qa.task_relationship_advisory.lower()
        assert "qnli_dev" in qa.task_relationship_advisory
        assert "rte_dev" in qa.task_relationship_advisory

    def test_task_relationship_advisory_no_qa(self):
        """No source QA → no advisory."""
        report = _FakeReport([_make_lv_dict("safe")])
        qa = build_qa_report(report)
        assert qa.task_relationship_advisory is None

    def test_task_relationship_advisory_missing_eval_dataset(self):
        """QA present but no eval_dataset → no advisory."""
        report = _FakeReport(
            [_make_lv_dict("safe")],
            source_qa={
                "adapter_a": {"status": "eligible"},
                "adapter_b": {"status": "eligible"},
            },
        )
        qa = build_qa_report(report)
        assert qa.task_relationship_advisory is None

    def test_core_space_summary_is_included_when_present(self):
        report = _FakeReport(
            [_make_lv_dict("safe")],
            core_space=_FakeCoreSpace(
                shared_basis_score_mean=0.71,
                basis_distortion_mean=0.18,
                effective_shared_rank_median=6,
                status="compatible",
            ),
        )
        qa = build_qa_report(report)
        assert qa.core_space is not None
        assert qa.core_space.shared_basis_score == pytest.approx(0.71)
        assert qa.core_space.basis_distortion == pytest.approx(0.18)
        assert qa.core_space.effective_shared_rank == 6
        assert qa.core_space.status == "compatible"


# ---------------------------------------------------------------------------
# Tests — format_qa_report
# ---------------------------------------------------------------------------


class TestFormatQAReport:
    def test_contains_all_sections(self):
        """Formatted output contains all key sections."""
        report = _FakeReport(
            [
                _make_lv_dict("safe", layer_name="model.layers.0.self_attn.q_proj"),
                _make_lv_dict(
                    "conflicting",
                    mean_overlap=0.6,
                    directional_agreement=-0.5,
                    conflict_dimensions=2,
                    layer_name="model.layers.1.self_attn.q_proj",
                ),
            ]
        )
        qa = build_qa_report(report)
        text = format_qa_report(qa)

        assert "MERGE QA REPORT" in text
        assert "1. STRUCTURAL RESULT" in text
        assert "2. BEHAVIORAL STATUS" in text
        assert "3. ELIGIBILITY WARNING" in text
        assert "4. RECOMMENDED ACTION" in text
        assert "Adapter A" in text
        assert "Adapter B" in text
        assert "Pair risk:" in text
        assert "Dominant issue:" in text
        assert "Confidence" in text

    def test_eligibility_section_present_when_needed(self):
        """Eligibility warning section shows caveats when present."""
        report = _FakeReport([_make_lv_dict("safe")])
        qa = build_qa_report(report)
        text = format_qa_report(qa)

        # No source QA → should have caveat in eligibility section
        assert "ELIGIBILITY WARNING" in text
        assert "source-eligibility" in text.lower() or "structural" in text.lower()

    def test_task_relationship_advisory_in_output(self):
        """Advisory appears in formatted output when present."""
        report = _FakeReport(
            [_make_lv_dict("safe")],
            source_qa={
                "adapter_a": {"status": "eligible", "eval_dataset": "qnli_dev"},
                "adapter_b": {"status": "eligible", "eval_dataset": "sst2_dev"},
            },
        )
        qa = build_qa_report(report)
        text = format_qa_report(qa)
        assert "TASK-BOUNDARY WARNING" in text
        assert "Cross-task merge" in text

    def test_task_relationship_advisory_absent_from_output(self):
        """No advisory → no advisory section in output."""
        report = _FakeReport(
            [_make_lv_dict("safe")],
            source_qa={
                "adapter_a": {"status": "eligible", "eval_dataset": "qnli_dev"},
                "adapter_b": {"status": "eligible", "eval_dataset": "qnli_dev"},
            },
        )
        qa = build_qa_report(report)
        text = format_qa_report(qa)
        assert "TASK-BOUNDARY WARNING" not in text

    def test_report_headline_cross_task(self):
        """Cross-task pair report includes cross-task headline."""
        report = _FakeReport(
            [_make_lv_dict("safe")],
            source_qa={
                "adapter_a": {"status": "eligible", "eval_dataset": "qnli_dev"},
                "adapter_b": {"status": "eligible", "eval_dataset": "sst2_dev"},
            },
        )
        qa = build_qa_report(report)
        text = format_qa_report(qa)
        assert "Cross-task pair" in text
        assert "caution region" in text

    def test_report_headline_same_task(self):
        """Same-task pair report includes same-task headline."""
        report = _FakeReport(
            [_make_lv_dict("safe")],
            source_qa={
                "adapter_a": {"status": "eligible", "eval_dataset": "qnli_dev"},
                "adapter_b": {"status": "eligible", "eval_dataset": "qnli_dev"},
            },
        )
        qa = build_qa_report(report)
        text = format_qa_report(qa)
        assert "Same-task pair" in text

    def test_report_interpretation_line_present(self):
        """Every formatted report includes an INTERPRETATION section."""
        report = _FakeReport(
            [_make_lv_dict("safe")],
            source_qa={
                "adapter_a": {"status": "eligible", "eval_dataset": "qnli_dev"},
                "adapter_b": {"status": "eligible", "eval_dataset": "qnli_dev"},
            },
        )
        qa = build_qa_report(report)
        text = format_qa_report(qa)
        assert "INTERPRETATION" in text

    def test_core_space_in_advanced_section(self):
        """Core-space appears under ADVANCED DIAGNOSTIC DETAIL, not inline."""
        report = _FakeReport(
            [_make_lv_dict("safe")],
            source_qa={
                "adapter_a": {"status": "eligible", "eval_dataset": "qnli_dev"},
                "adapter_b": {"status": "eligible", "eval_dataset": "qnli_dev"},
            },
            core_space={"shared_basis_score": 0.85, "basis_distortion": 0.05,
                        "effective_shared_rank": 7, "status": "marginal"},
        )
        qa = build_qa_report(report)
        text = format_qa_report(qa)
        assert "ADVANCED DIAGNOSTIC DETAIL" in text
        # Core-space should appear after INTERPRETATION
        interp_pos = text.index("INTERPRETATION")
        advanced_pos = text.index("ADVANCED DIAGNOSTIC DETAIL")
        assert advanced_pos > interp_pos

    def test_no_caveats_when_clean(self):
        """Low-risk pair with both eligible → minimal caveats."""
        report = _FakeReport(
            [_make_lv_dict("safe")],
            source_qa={
                "adapter_a": {"status": "eligible"},
                "adapter_b": {"status": "eligible"},
            },
        )
        qa = build_qa_report(report)
        text = format_qa_report(qa)

        # Should not have eligibility caveats
        assert "underperform" not in text.lower()


# ---------------------------------------------------------------------------
# Tests — serialization
# ---------------------------------------------------------------------------


class TestQAReportSerialization:
    def test_to_dict_has_schema(self):
        report = _FakeReport([_make_lv_dict("safe")])
        qa = build_qa_report(report)
        d = qa.to_dict()

        assert d["schema"] == "gradience.merge_qa_report/v1"
        assert "adapter_a" in d
        assert "adapter_b" in d
        assert "pair_risk" in d
        assert "dominant_issue" in d
        assert "dominant_issue_detail" in d
        assert "recommended_action" in d
        assert "confidence" in d
        assert "confidence_note" in d
        assert "caveats" in d

    def test_to_dict_emits_core_space_when_present(self):
        qa = MergeQAReport.from_dict(
            _minimal_report_dict(
                core_space={
                    "shared_basis_score": 0.71,
                    "basis_distortion": 0.18,
                    "effective_shared_rank": 6,
                    "status": "compatible",
                }
            )
        )
        d = qa.to_dict()
        assert "core_space" in d
        assert d["core_space"]["status"] == "compatible"

    def test_json_roundtrip(self, tmp_path):
        report = _FakeReport(
            [
                _make_lv_dict("safe", layer_name="model.layers.0.self_attn.q_proj"),
                _make_lv_dict("imbalanced", magnitude_ratio=5.0, layer_name="model.layers.1.self_attn.q_proj"),
            ]
        )
        qa = build_qa_report(report)

        json_path = tmp_path / "qa_report.json"
        qa.to_json(json_path)

        with open(json_path) as f:
            loaded = json.load(f)

        restored = MergeQAReport.from_dict(loaded)
        assert restored.pair_risk == qa.pair_risk
        assert restored.dominant_issue == qa.dominant_issue
        assert restored.adapter_a.rank == qa.adapter_a.rank
        assert restored.caveats == qa.caveats
        assert restored.compatibility_score == pytest.approx(qa.compatibility_score, abs=1e-3)

    def test_from_dict_roundtrip(self):
        report = _FakeReport([_make_lv_dict("safe")])
        qa = build_qa_report(report)

        d = qa.to_dict()
        restored = MergeQAReport.from_dict(d)

        assert restored.pair_risk == qa.pair_risk
        assert restored.recommended_strategy == qa.recommended_strategy
        assert restored.adapter_a.path == qa.adapter_a.path


# ---------------------------------------------------------------------------
# Tests — strategy derivation
# ---------------------------------------------------------------------------


class TestStrategyDerivation:
    def test_low_risk_no_compression_is_linear(self):
        """Low-risk, no compression -> linear."""
        report = _FakeReport(
            [
                _make_lv_dict("safe", layer_name="model.layers.0.self_attn.q_proj"),
                _make_lv_dict("safe", layer_name="model.layers.0.self_attn.v_proj"),
            ]
        )
        qa = build_qa_report(report)
        assert qa.recommended_strategy == "linear"

    def test_high_risk_is_audit_aware(self):
        """High-risk -> audit_aware."""
        report = _FakeReport(
            [
                _make_lv_dict(
                    "conflicting",
                    mean_overlap=0.6,
                    directional_agreement=-0.5,
                    conflict_dimensions=3,
                    layer_name="model.layers.0.self_attn.q_proj",
                ),
            ]
        )
        qa = build_qa_report(report)
        assert qa.recommended_strategy == "audit_aware"


# ---------------------------------------------------------------------------
# Tests — from_dict validation
# ---------------------------------------------------------------------------


def _minimal_report_dict(**overrides: Any) -> dict[str, Any]:
    """Return a minimal valid merge_qa_report/v1 dict, with optional overrides."""
    d: dict[str, Any] = {
        "schema": SCHEMA_ID,
        "adapter_a": {
            "path": "/tmp/a",
            "rank": 8,
            "alpha": 16.0,
            "n_layers": 32,
            "base_model": "llama",
            "eligibility_status": "eligible",
        },
        "adapter_b": {
            "path": "/tmp/b",
            "rank": 16,
            "alpha": 32.0,
            "n_layers": 32,
            "base_model": "llama",
            "eligibility_status": None,
        },
        "pair_risk": "low",
        "dominant_issue": "none",
        "dominant_issue_detail": "adapters are spectrally compatible",
        "recommended_action": "Merge is safe.",
        "recommended_strategy": "linear",
        "confidence": "high",
        "confidence_note": "High spectral compatibility",
        "caveats": ["caveat one"],
        "verdict_distribution": {"safe": 30, "redundant": 2, "conflicting": 0, "imbalanced": 0},
        "compatibility_score": 0.85,
    }
    d.update(overrides)
    return d


class TestFromDictValidation:
    """Strict validation tests for MergeQAReport.from_dict()."""

    def test_valid_roundtrip(self):
        """Valid dict round-trips through from_dict -> to_dict."""
        d = _minimal_report_dict()
        report = MergeQAReport.from_dict(d)
        assert report.pair_risk == "low"
        assert report.adapter_a.path == "/tmp/a"
        assert report.adapter_a.rank == 8
        assert report.adapter_b.eligibility_status is None
        assert report.dominant_issue == "none"
        assert report.recommended_strategy == "linear"
        assert report.confidence == "high"
        assert report.compatibility_score == pytest.approx(0.85)
        assert report.caveats == ("caveat one",)
        assert report.verdict_distribution["safe"] == 30

        # Round-trip through to_dict
        d2 = report.to_dict()
        report2 = MergeQAReport.from_dict(d2)
        assert report2.pair_risk == report.pair_risk
        assert report2.adapter_a.rank == report.adapter_a.rank

    def test_missing_schema(self):
        d = _minimal_report_dict()
        del d["schema"]
        with pytest.raises(QASchemaError, match="Missing required field: schema"):
            MergeQAReport.from_dict(d)

    def test_wrong_schema(self):
        d = _minimal_report_dict(schema="gradience.wrong/v99")
        with pytest.raises(QASchemaError, match="Expected schema"):
            MergeQAReport.from_dict(d)

    def test_missing_adapter_section(self):
        d = _minimal_report_dict()
        del d["adapter_a"]
        with pytest.raises(QASchemaError, match="Missing required section: adapter_a"):
            MergeQAReport.from_dict(d)

    def test_bad_pair_risk(self):
        d = _minimal_report_dict(pair_risk="extreme")
        with pytest.raises(QASchemaError, match="Invalid pair_risk"):
            MergeQAReport.from_dict(d)

    def test_unknown_dominant_issue(self):
        d = _minimal_report_dict(dominant_issue="cosmic_rays")
        with pytest.raises(QASchemaError, match="Unknown dominant_issue"):
            MergeQAReport.from_dict(d)

    def test_unknown_eligibility_status(self):
        d = _minimal_report_dict()
        d["adapter_a"]["eligibility_status"] = "super_eligible"
        with pytest.raises(QASchemaError, match="Unknown eligibility_status"):
            MergeQAReport.from_dict(d)

    def test_bad_confidence(self):
        d = _minimal_report_dict(confidence="very_high")
        with pytest.raises(QASchemaError, match="Invalid confidence"):
            MergeQAReport.from_dict(d)

    def test_caveats_must_be_list_of_str(self):
        d = _minimal_report_dict(caveats=[1, 2, 3])
        with pytest.raises(QASchemaError, match="caveats.*list of strings"):
            MergeQAReport.from_dict(d)

    def test_verdict_distribution_values_must_be_int(self):
        d = _minimal_report_dict(verdict_distribution={"safe": 1.5})
        with pytest.raises(QASchemaError, match="verdict_distribution.*must be int"):
            MergeQAReport.from_dict(d)

    def test_extra_keys_ignored(self):
        d = _minimal_report_dict(future_field="hello", another_extra=42)
        report = MergeQAReport.from_dict(d)
        assert report.pair_risk == "low"

    def test_null_eligibility_accepted(self):
        d = _minimal_report_dict()
        d["adapter_a"]["eligibility_status"] = None
        d["adapter_b"]["eligibility_status"] = None
        report = MergeQAReport.from_dict(d)
        assert report.adapter_a.eligibility_status is None
        assert report.adapter_b.eligibility_status is None

    def test_missing_optional_fields_backfilled(self):
        d = _minimal_report_dict()
        del d["dominant_issue_detail"]
        del d["confidence_note"]
        del d["recommended_action"]
        del d["caveats"]
        del d["verdict_distribution"]
        report = MergeQAReport.from_dict(d)
        assert report.dominant_issue_detail == ""
        assert report.confidence_note == ""
        assert report.recommended_action == ""
        assert report.caveats == ()
        assert report.verdict_distribution == {}
        assert report.core_space is None

    def test_numeric_rank_normalized_to_int(self):
        d = _minimal_report_dict()
        d["adapter_a"]["rank"] = 8.0  # float instead of int
        report = MergeQAReport.from_dict(d)
        assert report.adapter_a.rank == 8
        assert isinstance(report.adapter_a.rank, int)

    def test_task_relationship_advisory_roundtrip(self):
        """Advisory survives serialization round-trip."""
        d = _minimal_report_dict(
            task_relationship_advisory="Cross-task merge: adapters were evaluated on different tasks (qnli vs rte)."
        )
        report = MergeQAReport.from_dict(d)
        assert report.task_relationship_advisory is not None
        assert "qnli" in report.task_relationship_advisory

        d2 = report.to_dict()
        assert d2["task_relationship_advisory"] == report.task_relationship_advisory
        report2 = MergeQAReport.from_dict(d2)
        assert report2.task_relationship_advisory == report.task_relationship_advisory

    def test_task_relationship_advisory_absent_roundtrip(self):
        """No advisory → not in serialized dict, round-trips as None."""
        d = _minimal_report_dict()
        report = MergeQAReport.from_dict(d)
        assert report.task_relationship_advisory is None
        d2 = report.to_dict()
        assert "task_relationship_advisory" not in d2

    def test_unknown_strategy_accepted(self):
        """recommended_strategy is lenient for forward compatibility."""
        d = _minimal_report_dict(recommended_strategy="future_strategy_v3")
        report = MergeQAReport.from_dict(d)
        assert report.recommended_strategy == "future_strategy_v3"

    def test_core_space_optional_block_validates(self):
        d = _minimal_report_dict(
            core_space={
                "shared_basis_score": 0.66,
                "basis_distortion": 0.20,
                "effective_shared_rank": 5,
                "status": "marginal",
            }
        )
        report = MergeQAReport.from_dict(d)
        assert report.core_space is not None
        assert isinstance(report.core_space, CoreSpaceSummary)
        assert report.core_space.status == "marginal"

    def test_core_space_status_must_be_known(self):
        d = _minimal_report_dict(
            core_space={
                "shared_basis_score": 0.66,
                "basis_distortion": 0.20,
                "effective_shared_rank": 5,
                "status": "mystery",
            }
        )
        with pytest.raises(QASchemaError, match="core_space.status"):
            MergeQAReport.from_dict(d)

    def test_core_space_rank_must_be_int(self):
        d = _minimal_report_dict(
            core_space={
                "shared_basis_score": 0.66,
                "basis_distortion": 0.20,
                "effective_shared_rank": 5.0,
                "status": "marginal",
            }
        )
        with pytest.raises(QASchemaError, match="effective_shared_rank"):
            MergeQAReport.from_dict(d)

    def test_core_space_status_vocabulary_frozen(self):
        assert CORE_SPACE_STATUS_VALUES == {"compatible", "marginal", "incompatible", "not_applicable"}


# ---------------------------------------------------------------------------
# Tests — example files
# ---------------------------------------------------------------------------


class TestStrictReloadInvariant:
    """from_dict(to_dict(obj)) must produce an identical object."""

    def test_roundtrip_identity_safe(self) -> None:
        p = Path("examples/reports/safe_merge_report.json")
        with open(p) as f:
            d = json.load(f)
        original = MergeQAReport.from_dict(d)
        reloaded = MergeQAReport.from_dict(original.to_dict())
        assert reloaded == original

    def test_roundtrip_identity_high_risk(self) -> None:
        p = Path("examples/reports/high_risk_warn_report.json")
        with open(p) as f:
            d = json.load(f)
        original = MergeQAReport.from_dict(d)
        reloaded = MergeQAReport.from_dict(original.to_dict())
        assert reloaded == original


class TestExampleFiles:
    @pytest.mark.parametrize("path", sorted(glob.glob("examples/reports/*.json")))
    def test_example_loads_via_from_dict(self, path):
        with open(path) as f:
            d = json.load(f)
        report = MergeQAReport.from_dict(d)
        assert report.pair_risk in ("low", "medium", "high")
        assert report.dominant_issue in DOMINANT_ISSUE_LABELS
