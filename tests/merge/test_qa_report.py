"""Tests for gradience.vnext.merge.qa_report — QA report builder and formatter."""

from __future__ import annotations

import json

import pytest

from gradience.vnext.merge.qa_report import (
    AdapterSummary,
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
        self.aggregate = aggregate or {
            "overall_verdict": "safe",
            "compatibility_score": 0.75,
            "mean_overlap": sum(overlaps) / len(overlaps) if overlaps else 0.0,
            "median_overlap": sorted(overlaps)[len(overlaps) // 2] if overlaps else 0.0,
            "max_overlap": max(overlaps) if overlaps else 0.0,
            "mean_agreement": 0.5,
            "n_safe": n_safe,
            "n_redundant": n_redundant,
            "n_conflicting": n_conflicting,
            "n_imbalanced": n_imbalanced,
            "mean_magnitude_ratio": sum(mag_ratios) / len(mag_ratios) if mag_ratios else 1.0,
        }
        self.adapter_a = {
            "path": "/tmp/adapter_a",
            "rank": 8,
            "alpha": 16.0,
            "base_model": "meta-llama/Llama-2-7b",
            "n_layers": len(layer_verdicts),
        }
        self.adapter_b = {
            "path": "/tmp/adapter_b",
            "rank": 8,
            "alpha": 16.0,
            "base_model": "meta-llama/Llama-2-7b",
            "n_layers": len(layer_verdicts),
        }


# ---------------------------------------------------------------------------
# Tests — AdapterSummary
# ---------------------------------------------------------------------------


class TestAdapterSummary:
    def test_to_dict_roundtrip(self):
        s = AdapterSummary(
            path="/tmp/a", rank=8, alpha=16.0,
            n_layers=32, base_model="llama", eligibility="eligible",
        )
        d = s.to_dict()
        assert d["path"] == "/tmp/a"
        assert d["rank"] == 8
        assert d["eligibility"] == "eligible"


# ---------------------------------------------------------------------------
# Tests — build_qa_report
# ---------------------------------------------------------------------------


class TestBuildQAReport:
    def test_all_safe_layers(self):
        """All-safe pair produces low risk, no dominant issue."""
        report = _FakeReport([
            _make_lv_dict("safe", layer_name="model.layers.0.self_attn.q_proj"),
            _make_lv_dict("safe", layer_name="model.layers.0.self_attn.v_proj"),
        ])
        qa = build_qa_report(report)

        assert qa.pair_risk == "low"
        assert "none" in qa.dominant_issue.lower() or "compatible" in qa.dominant_issue.lower()
        assert qa.adapter_a.rank == 8
        assert qa.adapter_b.rank == 8
        assert qa.verdict_distribution["safe"] == 2
        assert qa.verdict_distribution["conflicting"] == 0

    def test_conflicting_layers_high_risk(self):
        """Conflicting layers produce high risk."""
        report = _FakeReport([
            _make_lv_dict(
                "conflicting",
                mean_overlap=0.6,
                directional_agreement=-0.5,
                conflict_dimensions=3,
                layer_name="model.layers.0.self_attn.q_proj",
            ),
        ])
        qa = build_qa_report(report)

        assert qa.pair_risk == "high"
        assert "conflict" in qa.dominant_issue.lower()
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

        assert "imbalance" in qa.dominant_issue.lower()
        assert qa.verdict_distribution["imbalanced"] == 1

    def test_redundant_layers(self):
        report = _FakeReport([
            _make_lv_dict("redundant", mean_overlap=0.8, directional_agreement=0.7,
                          layer_name="model.layers.0.self_attn.q_proj"),
            _make_lv_dict("redundant", mean_overlap=0.9, directional_agreement=0.8,
                          layer_name="model.layers.0.self_attn.v_proj"),
        ])
        qa = build_qa_report(report)

        assert "redundan" in qa.dominant_issue.lower()

    def test_eligibility_not_provided(self):
        """No source QA → eligibility shows 'not provided'."""
        report = _FakeReport([_make_lv_dict("safe")])
        qa = build_qa_report(report)

        assert qa.adapter_a.eligibility == "not provided"
        assert qa.adapter_b.eligibility == "not provided"
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

        assert qa.adapter_a.eligibility == "eligible"
        assert qa.adapter_b.eligibility == "eligible"
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

        assert qa.adapter_b.eligibility == "flagged_weak"
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


# ---------------------------------------------------------------------------
# Tests — format_qa_report
# ---------------------------------------------------------------------------


class TestFormatQAReport:
    def test_contains_all_sections(self):
        """Formatted output contains all key sections."""
        report = _FakeReport([
            _make_lv_dict("safe", layer_name="model.layers.0.self_attn.q_proj"),
            _make_lv_dict("conflicting", mean_overlap=0.6, directional_agreement=-0.5,
                          conflict_dimensions=2, layer_name="model.layers.1.self_attn.q_proj"),
        ])
        qa = build_qa_report(report)
        text = format_qa_report(qa)

        assert "MERGE QA REPORT" in text
        assert "Adapter A" in text
        assert "Adapter B" in text
        assert "Pair risk:" in text
        assert "Dominant issue:" in text
        assert "Recommended action" in text
        assert "Confidence" in text

    def test_caveats_section_present_when_needed(self):
        """Caveats section appears when there are caveats."""
        report = _FakeReport([_make_lv_dict("safe")])
        qa = build_qa_report(report)
        text = format_qa_report(qa)

        # No source QA → should have caveat about it
        assert "Caveats" in text
        assert "source-eligibility" in text.lower()

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
        assert "recommended_action" in d
        assert "confidence_note" in d
        assert "caveats" in d

    def test_json_roundtrip(self, tmp_path):
        report = _FakeReport([
            _make_lv_dict("safe", layer_name="model.layers.0.self_attn.q_proj"),
            _make_lv_dict("imbalanced", magnitude_ratio=5.0,
                          layer_name="model.layers.1.self_attn.q_proj"),
        ])
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
