"""Tests for gradience.vnext.audit.qa_artifact — single-adapter QA schema v1."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import pytest

from gradience.vnext.audit.qa_artifact import (
    CONFIDENCE_HIGH,
    CONFIDENCE_LOW,
    CONFIDENCE_MEDIUM,
    SCHEMA_VERSION,
    AdapterQAArtifact,
    build_qa_artifact,
    build_reasons,
    derive_confidence,
    derive_structural_flags,
)
from gradience.exceptions import QASchemaError
from gradience.vnext.merge.eligibility import AdapterQAResult, EligibilityStatus

# ---------------------------------------------------------------------------
# Helpers — lightweight stub for LoRAAuditResult
# ---------------------------------------------------------------------------


@dataclass
class _StubLayer:
    r: int = 16
    utilization: float = 0.08


@dataclass
class _StubAuditResult:
    peft_dir: str | None = "./adapters/test-adapter"
    n_layers: int = 32
    utilization_mean: float = 0.08
    stable_rank_mean: float = 1.9
    energy_rank_90_p50: float | None = 2.3
    layers: list[Any] = None  # type: ignore[assignment]

    def __post_init__(self):
        if self.layers is None:
            self.layers = [_StubLayer(r=16, utilization=self.utilization_mean) for _ in range(self.n_layers)]


# ---------------------------------------------------------------------------
# Schema output tests
# ---------------------------------------------------------------------------


class TestSchemaOutput:
    def test_schema_version(self):
        art = _make_artifact()
        d = art.to_dict()
        assert d["schema"] == SCHEMA_VERSION
        assert d["schema"] == "gradience.adapter_qa/v1"

    def test_required_top_level_keys(self):
        d = _make_artifact().to_dict()
        for key in ("schema", "adapter", "structural_summary", "behavioral_summary", "eligibility"):
            assert key in d, f"Missing top-level key: {key}"

    def test_adapter_fields(self):
        art = _make_artifact(adapter_name="my-adapter", adapter_path="./my-adapter", rank_nominal=16, n_layers=32)
        adapter = art.to_dict()["adapter"]
        assert adapter["name"] == "my-adapter"
        assert adapter["path"] == "./my-adapter"
        assert isinstance(adapter["rank_nominal"], int)
        assert isinstance(adapter["n_layers"], int)
        assert adapter["rank_nominal"] == 16
        assert adapter["n_layers"] == 32
        assert "base_model" in adapter

    def test_roundtrip(self):
        art = _make_artifact(
            status=EligibilityStatus.FLAGGED_WEAK,
            confidence=CONFIDENCE_MEDIUM,
            reasons=["adapter underperforms base on perplexity (oasst2)", "low utilization across layers"],
        )
        d = art.to_dict()
        restored = AdapterQAArtifact.from_dict(d)
        assert restored.status == art.status
        assert restored.confidence == art.confidence
        assert restored.reasons == art.reasons
        assert restored.adapter_name == art.adapter_name
        assert restored.utilization_mean == art.utilization_mean
        assert restored.eval_available == art.eval_available

    def test_json_serializable(self):
        d = _make_artifact().to_dict()
        # Must not raise
        serialized = json.dumps(d)
        assert isinstance(serialized, str)
        # And must roundtrip through JSON
        parsed = json.loads(serialized)
        assert parsed["schema"] == SCHEMA_VERSION

    def test_effective_rank_90_median_key(self):
        """JSON output uses 'effective_rank_90_median', not 'energy_rank_90_p50'."""
        d = _make_artifact().to_dict()
        assert "effective_rank_90_median" in d["structural_summary"]
        assert "energy_rank_90_p50" not in d["structural_summary"]

    def test_metric_name_null_when_no_eval(self):
        """metric_name should be null (not empty string) when no eval."""
        art = _make_artifact(eval_available=False, metric_name=None)
        d = art.to_dict()
        assert d["behavioral_summary"]["metric_name"] is None

    def test_lower_is_better_null_when_no_eval(self):
        """lower_is_better should be null when no eval."""
        art = _make_artifact(eval_available=False, lower_is_better=None)
        d = art.to_dict()
        assert d["behavioral_summary"]["lower_is_better"] is None

    def test_notes_field_present(self):
        """notes must always be present in output."""
        d = _make_artifact().to_dict()
        assert "notes" in d
        assert isinstance(d["notes"], list)

    def test_notes_roundtrip(self):
        art = _make_artifact()
        d = art.to_dict()
        d["notes"] = ["structural audit only"]
        restored = AdapterQAArtifact.from_dict(d)
        assert restored.notes == ["structural audit only"]

    def test_from_dict_empty_raises(self):
        with pytest.raises(QASchemaError):
            AdapterQAArtifact.from_dict({})


# ---------------------------------------------------------------------------
# Eligibility status assignment tests
# ---------------------------------------------------------------------------


class TestEligibilityAssignment:
    def test_eligible_when_beats_base(self):
        result = _StubAuditResult(utilization_mean=0.5, stable_rank_mean=8.0)
        art = build_qa_artifact(
            result,
            adapter_score=3.21,
            base_score=4.66,
            metric_name="perplexity",
            lower_is_better=True,
        )
        assert art.status == EligibilityStatus.ELIGIBLE
        assert art.beats_base is True

    def test_flagged_weak_when_worse_than_base(self):
        result = _StubAuditResult()
        art = build_qa_artifact(
            result,
            adapter_score=6.81,
            base_score=4.66,
            metric_name="perplexity",
            lower_is_better=True,
        )
        assert art.status == EligibilityStatus.FLAGGED_WEAK
        assert art.beats_base is False

    def test_uncertain_within_margin(self):
        result = _StubAuditResult(utilization_mean=0.5, stable_rank_mean=8.0)
        art = build_qa_artifact(
            result,
            adapter_score=4.60,
            base_score=4.66,
            metric_name="perplexity",
            lower_is_better=True,
            margin=0.1,
        )
        # delta = 4.66 - 4.60 = 0.06, positive but < margin → uncertain
        assert art.status == EligibilityStatus.UNCERTAIN

    def test_unknown_no_behavioral(self):
        result = _StubAuditResult()
        art = build_qa_artifact(result)
        assert art.status == EligibilityStatus.UNKNOWN_NO_BEHAVIORAL_EVAL
        assert art.eval_available is False
        assert art.beats_base is None


# ---------------------------------------------------------------------------
# Structural flag tests
# ---------------------------------------------------------------------------


class TestStructuralFlags:
    def test_low_utilization_flag(self):
        flags = derive_structural_flags(
            utilization_mean=0.10,
            rank_waste_ratio=0.90,
            energy_rank_90_p50=5.0,
            stable_rank_mean=4.0,
            rank_nominal=16,
        )
        assert "low_utilization" in flags

    def test_high_rank_waste_flag(self):
        flags = derive_structural_flags(
            utilization_mean=0.10,
            rank_waste_ratio=0.90,
            energy_rank_90_p50=5.0,
            stable_rank_mean=4.0,
            rank_nominal=16,
        )
        assert "high_rank_waste" in flags

    def test_no_flags_when_healthy(self):
        flags = derive_structural_flags(
            utilization_mean=0.60,
            rank_waste_ratio=0.40,
            energy_rank_90_p50=8.0,
            stable_rank_mean=10.0,
            rank_nominal=16,
        )
        assert flags == []

    def test_concentrated_spectrum_flag(self):
        flags = derive_structural_flags(
            utilization_mean=0.30,
            rank_waste_ratio=0.70,
            energy_rank_90_p50=1.5,
            stable_rank_mean=5.0,
            rank_nominal=16,
        )
        assert "concentrated_spectrum" in flags

    def test_concentrated_spectrum_not_triggered_low_rank(self):
        # rank_nominal < 8 should not trigger concentrated_spectrum
        flags = derive_structural_flags(
            utilization_mean=0.30,
            rank_waste_ratio=0.70,
            energy_rank_90_p50=1.5,
            stable_rank_mean=2.0,
            rank_nominal=4,
        )
        assert "concentrated_spectrum" not in flags

    def test_underutilized_capacity_flag(self):
        flags = derive_structural_flags(
            utilization_mean=0.10,
            rank_waste_ratio=0.90,
            energy_rank_90_p50=5.0,
            stable_rank_mean=1.5,  # < 16 * 0.2 = 3.2
            rank_nominal=16,
        )
        assert "underutilized_capacity" in flags

    def test_energy_rank_none_skips_concentrated(self):
        flags = derive_structural_flags(
            utilization_mean=0.10,
            rank_waste_ratio=0.90,
            energy_rank_90_p50=None,
            stable_rank_mean=1.5,
            rank_nominal=16,
        )
        assert "concentrated_spectrum" not in flags


# ---------------------------------------------------------------------------
# Confidence tests
# ---------------------------------------------------------------------------


class TestConfidence:
    def test_high_confidence_requires_behavioral(self):
        conf = derive_confidence(eval_available=True, status=EligibilityStatus.ELIGIBLE, delta=2.0, margin=0.0)
        assert conf == CONFIDENCE_HIGH

    def test_low_confidence_structural_only(self):
        conf = derive_confidence(eval_available=False, status=EligibilityStatus.UNKNOWN_NO_BEHAVIORAL_EVAL)
        assert conf == CONFIDENCE_LOW

    def test_never_high_without_behavioral(self):
        # Even with a status that would otherwise be clear, no behavioral → low
        conf = derive_confidence(eval_available=False, status=EligibilityStatus.ELIGIBLE)
        assert conf == CONFIDENCE_LOW

    def test_medium_confidence_uncertain_status(self):
        conf = derive_confidence(eval_available=True, status=EligibilityStatus.UNCERTAIN, delta=0.05, margin=0.1)
        assert conf == CONFIDENCE_MEDIUM

    def test_medium_confidence_marginal_behavioral(self):
        # delta is small relative to margin → medium even though status is clear
        conf = derive_confidence(eval_available=True, status=EligibilityStatus.FLAGGED_WEAK, delta=-0.05, margin=0.1)
        assert conf == CONFIDENCE_MEDIUM


# ---------------------------------------------------------------------------
# Reason list tests
# ---------------------------------------------------------------------------


class TestReasons:
    def test_reasons_include_behavioral_eligible(self):
        reasons = build_reasons(
            eval_available=True,
            status=EligibilityStatus.ELIGIBLE,
            metric_name="perplexity",
            eval_dataset="oasst2",
            structural_flags=[],
        )
        assert any("outperforms" in r for r in reasons)
        assert any("oasst2" in r for r in reasons)

    def test_reasons_include_behavioral_weak(self):
        reasons = build_reasons(
            eval_available=True,
            status=EligibilityStatus.FLAGGED_WEAK,
            metric_name="perplexity",
            eval_dataset="oasst2",
            structural_flags=["low_utilization"],
        )
        assert any("underperforms" in r for r in reasons)
        assert any("utilization" in r for r in reasons)

    def test_reasons_mention_no_eval(self):
        reasons = build_reasons(
            eval_available=False,
            status=EligibilityStatus.UNKNOWN_NO_BEHAVIORAL_EVAL,
            metric_name="",
            eval_dataset=None,
            structural_flags=["low_utilization"],
        )
        assert reasons[0] == "no behavioral evaluation available"
        assert len(reasons) == 2  # no eval + one structural flag

    def test_reasons_structural_flags_appended(self):
        reasons = build_reasons(
            eval_available=False,
            status=EligibilityStatus.UNKNOWN_NO_BEHAVIORAL_EVAL,
            metric_name="",
            eval_dataset=None,
            structural_flags=["low_utilization", "high_rank_waste"],
        )
        assert "low utilization across layers" in reasons
        assert "high rank waste" in reasons


# ---------------------------------------------------------------------------
# Bridge tests — to_qa_result
# ---------------------------------------------------------------------------


class TestBridgeToQAResult:
    def test_to_qa_result_preserves_status(self):
        art = _make_artifact(status=EligibilityStatus.FLAGGED_WEAK)
        qa = art.to_qa_result()
        assert isinstance(qa, AdapterQAResult)
        assert qa.status == EligibilityStatus.FLAGGED_WEAK

    def test_to_qa_result_preserves_scores(self):
        art = _make_artifact(adapter_score=6.81, base_score=4.66, metric_name="perplexity")
        qa = art.to_qa_result()
        assert qa.adapter_metric == 6.81
        assert qa.base_metric == 4.66
        assert qa.metric_name == "perplexity"

    def test_to_qa_result_preserves_eval_dataset(self):
        art = _make_artifact(eval_dataset="oasst2")
        qa = art.to_qa_result()
        assert qa.eval_dataset == "oasst2"

    def test_to_qa_result_evidence_contains_confidence(self):
        art = _make_artifact(confidence=CONFIDENCE_HIGH, structural_flags=["low_utilization"])
        qa = art.to_qa_result()
        assert qa.evidence["confidence"] == CONFIDENCE_HIGH
        assert "low_utilization" in qa.evidence["structural_flags"]

    def test_to_qa_result_notes_from_reasons(self):
        art = _make_artifact(reasons=["reason one", "reason two"])
        qa = art.to_qa_result()
        assert qa.notes == "reason one; reason two"

    def test_to_qa_result_notes_none_when_no_reasons(self):
        art = _make_artifact(reasons=[])
        qa = art.to_qa_result()
        assert qa.notes is None


# ---------------------------------------------------------------------------
# _load_source_qa backward compatibility
# ---------------------------------------------------------------------------


class TestLoadSourceQA:
    def test_load_v1_artifact(self, tmp_path):
        """_load_source_qa auto-detects v1 artifact and returns AdapterQAResult."""
        art = _make_artifact(
            status=EligibilityStatus.FLAGGED_WEAK,
            adapter_score=6.81,
            base_score=4.66,
        )
        p = tmp_path / "qa.json"
        p.write_text(json.dumps(art.to_dict()))

        from gradience.cli import _load_source_qa

        qa = _load_source_qa(str(p))
        assert isinstance(qa, AdapterQAResult)
        assert qa.status == EligibilityStatus.FLAGGED_WEAK
        assert qa.adapter_metric == 6.81

    def test_load_legacy_format(self, tmp_path):
        """_load_source_qa still works with legacy flat AdapterQAResult dicts."""
        legacy = {
            "adapter_path": "./test",
            "status": "eligible",
            "adapter_metric": 3.0,
            "base_metric": 4.0,
            "metric_name": "perplexity",
            "lower_is_better": True,
        }
        p = tmp_path / "legacy.json"
        p.write_text(json.dumps(legacy))

        from gradience.cli import _load_source_qa

        qa = _load_source_qa(str(p))
        assert isinstance(qa, AdapterQAResult)
        assert qa.status == EligibilityStatus.ELIGIBLE
        assert qa.adapter_metric == 3.0

    def test_load_returns_none_for_none(self):
        from gradience.cli import _load_source_qa

        assert _load_source_qa(None) is None

    def test_load_rejects_wrong_schema(self, tmp_path):
        """Wrong schema key should error, not fall back to legacy."""
        d = {"schema": "gradience.adapter_qa/v99", "adapter": {}}
        p = tmp_path / "bad_schema.json"
        p.write_text(json.dumps(d))

        from gradience.cli import _load_source_qa

        with pytest.raises(SystemExit):
            _load_source_qa(str(p))


# ---------------------------------------------------------------------------
# Missing behavioral data tests
# ---------------------------------------------------------------------------


class TestMissingBehavioral:
    def test_behavioral_summary_eval_not_available(self):
        art = _make_artifact(eval_available=False)
        d = art.to_dict()
        assert d["behavioral_summary"]["eval_available"] is False
        assert d["behavioral_summary"]["adapter_score"] is None
        assert d["behavioral_summary"]["base_score"] is None
        assert d["behavioral_summary"]["beats_base"] is None

    def test_builder_no_behavioral_args(self):
        result = _StubAuditResult()
        art = build_qa_artifact(result)
        assert art.eval_available is False
        assert art.adapter_score is None
        assert art.base_score is None
        assert art.beats_base is None
        assert art.status == EligibilityStatus.UNKNOWN_NO_BEHAVIORAL_EVAL
        assert art.confidence == CONFIDENCE_LOW
        assert any("no behavioral" in r for r in art.reasons)


# ---------------------------------------------------------------------------
# build_qa_artifact integration
# ---------------------------------------------------------------------------


class TestBuildQAArtifact:
    def test_adapter_name_from_path(self):
        result = _StubAuditResult(peft_dir="./adapters/catsubcat-r16")
        art = build_qa_artifact(result, adapter_path="./adapters/catsubcat-r16")
        assert art.adapter_name == "catsubcat-r16"

    def test_adapter_name_falls_back_to_peft_dir(self):
        result = _StubAuditResult(peft_dir="./adapters/fallback-name")
        art = build_qa_artifact(result)
        assert art.adapter_name == "fallback-name"

    def test_rank_nominal_from_layers(self):
        result = _StubAuditResult()
        result.layers = [_StubLayer(r=16) for _ in range(4)]
        art = build_qa_artifact(result)
        assert art.rank_nominal == 16

    def test_rank_nominal_zero_when_no_layers(self):
        result = _StubAuditResult()
        result.layers = []
        art = build_qa_artifact(result)
        assert art.rank_nominal == 0

    def test_rank_waste_ratio_derived(self):
        result = _StubAuditResult(utilization_mean=0.30)
        art = build_qa_artifact(result)
        assert abs(art.rank_waste_ratio - 0.70) < 1e-9

    def test_structural_flags_populated(self):
        result = _StubAuditResult(utilization_mean=0.08, stable_rank_mean=1.9)
        art = build_qa_artifact(result)
        assert "low_utilization" in art.structural_flags
        assert "high_rank_waste" in art.structural_flags


# ---------------------------------------------------------------------------
# Example file validation
# ---------------------------------------------------------------------------


class TestExampleFiles:
    @pytest.mark.parametrize(
        "filename,expected_status",
        [
            ("catsubcat_r16_qa.json", "flagged_weak"),
            ("btgenbot_r8_qa.json", "flagged_weak"),
            ("eligible_adapter_qa.json", "eligible"),
            ("structural_only_qa.json", "unknown_no_behavioral_eval"),
        ],
    )
    def test_example_file_valid(self, filename, expected_status):
        import pathlib

        p = pathlib.Path("examples/qa") / filename
        if not p.exists():
            pytest.skip(f"Example file not found: {p}")
        with open(p) as f:
            d = json.load(f)
        assert d["schema"] == SCHEMA_VERSION
        art = AdapterQAArtifact.from_dict(d)
        assert art.status.value == expected_status
        # Roundtrip
        d2 = art.to_dict()
        assert d2["eligibility"]["status"] == expected_status


# ---------------------------------------------------------------------------
# from_dict validation tests
# ---------------------------------------------------------------------------


class TestFromDictValidation:
    def test_rejects_missing_schema(self):
        with pytest.raises(QASchemaError, match="Missing required field: schema"):
            AdapterQAArtifact.from_dict({"adapter": {}})

    def test_rejects_wrong_schema(self):
        d = _make_artifact().to_dict()
        d["schema"] = "gradience.adapter_qa/v2"
        with pytest.raises(QASchemaError, match="Expected schema"):
            AdapterQAArtifact.from_dict(d)

    def test_rejects_missing_section(self):
        d = _make_artifact().to_dict()
        del d["eligibility"]
        with pytest.raises(QASchemaError, match="Missing required section: eligibility"):
            AdapterQAArtifact.from_dict(d)

    def test_rejects_section_not_dict(self):
        d = _make_artifact().to_dict()
        d["adapter"] = "not a dict"
        with pytest.raises(QASchemaError, match="must be a dict"):
            AdapterQAArtifact.from_dict(d)

    def test_rejects_missing_required_field(self):
        d = _make_artifact().to_dict()
        del d["adapter"]["rank_nominal"]
        with pytest.raises(QASchemaError, match="rank_nominal"):
            AdapterQAArtifact.from_dict(d)

    def test_rejects_unknown_status(self):
        d = _make_artifact().to_dict()
        d["eligibility"]["status"] = "experimental_new_status"
        with pytest.raises(QASchemaError, match="Unknown eligibility status"):
            AdapterQAArtifact.from_dict(d)

    def test_rejects_bad_confidence(self):
        d = _make_artifact().to_dict()
        d["eligibility"]["confidence"] = "super_high"
        with pytest.raises(QASchemaError, match="confidence"):
            AdapterQAArtifact.from_dict(d)

    def test_rejects_flags_not_list_of_str(self):
        d = _make_artifact().to_dict()
        d["structural_summary"]["flags"] = [1, 2, 3]
        with pytest.raises(QASchemaError, match="flags"):
            AdapterQAArtifact.from_dict(d)

    def test_rejects_notes_not_list_of_str(self):
        d = _make_artifact().to_dict()
        d["notes"] = "not a list"
        with pytest.raises(QASchemaError, match="notes"):
            AdapterQAArtifact.from_dict(d)

    def test_accepts_numeric_as_float(self):
        """Integer values for float fields should be accepted and normalized."""
        d = _make_artifact().to_dict()
        d["structural_summary"]["utilization_mean"] = 0  # int, not float
        d["structural_summary"]["rank_waste_ratio"] = 1  # int, not float
        art = AdapterQAArtifact.from_dict(d)
        assert isinstance(art.utilization_mean, float)
        assert isinstance(art.rank_waste_ratio, float)

    def test_missing_notes_backfilled(self):
        d = _make_artifact().to_dict()
        del d["notes"]
        art = AdapterQAArtifact.from_dict(d)
        assert art.notes == []

    def test_missing_reasons_backfilled(self):
        d = _make_artifact().to_dict()
        del d["eligibility"]["reasons"]
        art = AdapterQAArtifact.from_dict(d)
        assert art.reasons == []

    def test_extra_keys_ignored(self):
        d = _make_artifact().to_dict()
        d["provenance"] = {"generated_by": "test"}
        d["adapter"]["extra_field"] = "ignored"
        art = AdapterQAArtifact.from_dict(d)
        assert art.adapter_name == "test-adapter"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_artifact(**overrides: Any) -> AdapterQAArtifact:
    """Create an AdapterQAArtifact with sensible defaults, overridable by kwargs."""
    defaults = dict(
        adapter_name="test-adapter",
        adapter_path="./adapters/test-adapter",
        base_model="meta-llama/Llama-2-7b-hf",
        rank_nominal=16,
        n_layers=32,
        utilization_mean=0.08,
        utilization_median=0.05,
        stable_rank_mean=1.9,
        energy_rank_90_p50=2.3,
        rank_waste_ratio=0.92,
        structural_flags=["low_utilization", "high_rank_waste"],
        eval_available=False,
        eval_dataset=None,
        metric_name=None,
        adapter_score=None,
        base_score=None,
        lower_is_better=None,
        beats_base=None,
        status=EligibilityStatus.UNKNOWN_NO_BEHAVIORAL_EVAL,
        confidence=CONFIDENCE_LOW,
        reasons=["no behavioral evaluation available"],
    )
    defaults.update(overrides)
    return AdapterQAArtifact(**defaults)
