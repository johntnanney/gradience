"""Tests for the honesty update: decoder-scale provenance warnings in QA reports."""

from __future__ import annotations

from typing import Any

import pytest

from gradience.exceptions import QASchemaError
from gradience.vnext.merge.containers import AdapterMetadata, AggregateResult
from gradience.vnext.merge.qa_report import (
    SCHEMA_ID,
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
        "suggested_coefficients": None,
        "over_accumulation_score": 0.0,
        "over_accumulation_band": "low",
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

    def __init__(self, layer_verdicts, base_model: str = "meta-llama/Llama-2-7b", source_qa=None):
        self.layer_verdicts = layer_verdicts
        self.source_qa = source_qa
        self.core_space = None
        n_safe = sum(1 for lv in layer_verdicts if lv["verdict"] == "safe")
        n_redundant = sum(1 for lv in layer_verdicts if lv["verdict"] == "redundant")
        n_conflicting = sum(1 for lv in layer_verdicts if lv["verdict"] == "conflicting")
        n_imbalanced = sum(1 for lv in layer_verdicts if lv["verdict"] == "imbalanced")
        overlaps = [lv["metrics"]["mean_overlap"] for lv in layer_verdicts]
        mag_ratios = [lv["metrics"]["magnitude_ratio"] for lv in layer_verdicts]
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
            base_model=base_model,
            n_layers=len(layer_verdicts),
        )
        self.adapter_b = AdapterMetadata(
            path="/tmp/adapter_b",
            rank=8,
            alpha=16.0,
            base_model=base_model,
            n_layers=len(layer_verdicts),
        )


# ---------------------------------------------------------------------------
# Tests — decoder-scale provenance caveat
# ---------------------------------------------------------------------------


class TestDecoderScaleCaveat:
    def test_decoder_scale_caveat_present(self):
        """Decoder-scale base model triggers provenance caveat."""
        report = _FakeReport(
            [_make_lv_dict("safe")],
            base_model="mistralai/Mistral-7B-v0.3",
        )
        qa = build_qa_report(report)
        assert any("DECODER-SCALE PROVENANCE" in c for c in qa.caveats)

    def test_encoder_scale_no_caveat(self):
        """Encoder-scale base model does NOT trigger provenance caveat."""
        report = _FakeReport(
            [_make_lv_dict("safe")],
            base_model="distilbert-base-uncased",
        )
        qa = build_qa_report(report)
        assert not any("DECODER-SCALE PROVENANCE" in c for c in qa.caveats)

    def test_decoder_scale_caveat_is_first(self):
        """Provenance caveat is at index 0 (most important)."""
        report = _FakeReport(
            [_make_lv_dict("safe")],
            base_model="mistralai/Mistral-7B-v0.3",
        )
        qa = build_qa_report(report)
        assert len(qa.caveats) > 0
        assert "DECODER-SCALE PROVENANCE" in qa.caveats[0]


# ---------------------------------------------------------------------------
# Tests — scale_validation block in JSON
# ---------------------------------------------------------------------------


class TestScaleValidationJSON:
    def test_json_output_includes_scale_validation(self):
        """Decoder-scale reports include scale_validation block in JSON."""
        report = _FakeReport(
            [_make_lv_dict("safe")],
            base_model="mistralai/Mistral-7B-v0.3",
        )
        qa = build_qa_report(report)
        d = qa.to_dict()
        assert "scale_validation" in d
        sv = d["scale_validation"]
        assert sv["risk_prediction_status"] == "unvalidated"
        assert sv["source_geometry_status"] == "validated"
        assert "decoder" in sv["detected_scale"]

    def test_json_output_no_scale_validation_for_encoder(self):
        """Encoder-scale reports do NOT include scale_validation block."""
        report = _FakeReport(
            [_make_lv_dict("safe")],
            base_model="distilbert-base-uncased",
        )
        qa = build_qa_report(report)
        d = qa.to_dict()
        assert "scale_validation" not in d


# ---------------------------------------------------------------------------
# Tests — softened recommendation language
# ---------------------------------------------------------------------------


class TestSoftenedRecommendations:
    def test_decoder_low_risk_softened(self):
        """Low-risk decoder-scale report uses softened language."""
        report = _FakeReport(
            [_make_lv_dict("safe")],
            base_model="mistralai/Mistral-7B-v0.3",
        )
        qa = build_qa_report(report)
        assert "not validated at this scale" in qa.recommended_action

    def test_encoder_low_risk_unchanged(self):
        """Low-risk encoder-scale report uses original language."""
        report = _FakeReport(
            [_make_lv_dict("safe")],
            base_model="distilbert-base-uncased",
        )
        qa = build_qa_report(report)
        assert "Merge is safe" in qa.recommended_action

    def test_decoder_high_risk_softened(self):
        """High-risk decoder-scale report uses softened language."""
        report = _FakeReport(
            [
                _make_lv_dict(
                    "conflicting",
                    mean_overlap=0.6,
                    directional_agreement=-0.5,
                    conflict_dimensions=3,
                ),
            ],
            base_model="mistralai/Mistral-7B-v0.3",
        )
        qa = build_qa_report(report)
        assert "unvalidated" in qa.recommended_action.lower()


# ---------------------------------------------------------------------------
# Tests — schema backward compatibility
# ---------------------------------------------------------------------------


def _minimal_report_dict(**overrides: Any) -> dict[str, Any]:
    """Return a minimal valid merge_qa_report dict."""
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


class TestSchemaBackwardCompat:
    def test_schema_v1_1_round_trip(self):
        """v1.1 schema with scale_validation round-trips correctly."""
        d = _minimal_report_dict(
            scale_validation={
                "detected_scale": "decoder_7b",
                "risk_prediction_status": "unvalidated",
                "source_geometry_status": "validated",
                "note": "Test note",
            }
        )
        report = MergeQAReport.from_dict(d)
        assert report.scale_validation is not None
        assert report.scale_validation["detected_scale"] == "decoder_7b"

        d2 = report.to_dict()
        report2 = MergeQAReport.from_dict(d2)
        assert report2.scale_validation == report.scale_validation

    def test_schema_v1_backward_compat(self):
        """v1 schema (without scale_validation) still deserializes correctly."""
        d = _minimal_report_dict(schema="gradience.merge_qa_report/v1")
        report = MergeQAReport.from_dict(d)
        assert report.pair_risk == "low"
        assert report.scale_validation is None

    def test_v1_1_schema_accepted(self):
        """v1.1 schema string is accepted."""
        d = _minimal_report_dict(schema="gradience.merge_qa_report/v1.1")
        report = MergeQAReport.from_dict(d)
        assert report.pair_risk == "low"

    def test_unknown_schema_rejected(self):
        """Unknown schema version is rejected."""
        d = _minimal_report_dict(schema="gradience.merge_qa_report/v99")
        with pytest.raises(QASchemaError, match="Expected schema"):
            MergeQAReport.from_dict(d)


# ---------------------------------------------------------------------------
# Tests — formatted output includes provenance
# ---------------------------------------------------------------------------


class TestFormattedOutputProvenance:
    def test_decoder_scale_provenance_in_formatted_text(self):
        """Formatted output includes provenance caveat for decoder-scale."""
        report = _FakeReport(
            [_make_lv_dict("safe")],
            base_model="mistralai/Mistral-7B-v0.3",
        )
        qa = build_qa_report(report)
        text = format_qa_report(qa)
        assert "DECODER-SCALE PROVENANCE" in text

    def test_encoder_scale_no_provenance_in_formatted_text(self):
        """Formatted output does NOT include provenance for encoder-scale."""
        report = _FakeReport(
            [_make_lv_dict("safe")],
            base_model="distilbert-base-uncased",
        )
        qa = build_qa_report(report)
        text = format_qa_report(qa)
        assert "DECODER-SCALE PROVENANCE" not in text
