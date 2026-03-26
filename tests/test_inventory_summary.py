"""Tests for InventorySummary dataclass, from_dict validation, and build_inventory_summary."""

from __future__ import annotations

import glob
import json
import tempfile
from pathlib import Path

import pytest

from gradience.exceptions import QASchemaError
from gradience.vnext.audit.qa_artifact import AdapterQAArtifact
from gradience.vnext.inventory.summary import (
    SCHEMA_ID,
    InventoryActionPlan,
    InventorySummary,
    build_action_plan,
    build_inventory_summary,
    format_action_plan,
    format_inventory_summary,
)
from gradience.vnext.merge.eligibility import EligibilityStatus
from gradience.vnext.merge.qa_report import MergeQAReport


def _valid_dict() -> dict:
    """Return a minimal valid inventory_summary/v1 dict."""
    return {
        "schema": SCHEMA_ID,
        "sources": {"adapter_qa": 3, "merge_report": 2},
        "adapter_status_counts": {"eligible": 2, "flagged_weak": 1},
        "adapter_flag_counts": {"low_utilization": 1},
        "pair_risk_counts": {"low": 1, "high": 1},
        "recommended_strategy_counts": {"linear": 1, "audit_aware": 1},
        "dominant_issue_counts": {"none": 1, "subspace_conflict": 1},
        "strict_qa_block_candidates": 1,
        "notes": ["Example note"],
    }


class TestFromDictValidation:
    """Validate from_dict schema enforcement."""

    def test_valid_roundtrip(self) -> None:
        d = _valid_dict()
        obj = InventorySummary.from_dict(d)
        assert obj.sources == d["sources"]
        assert obj.adapter_status_counts == d["adapter_status_counts"]
        assert obj.adapter_flag_counts == d["adapter_flag_counts"]
        assert obj.pair_risk_counts == d["pair_risk_counts"]
        assert obj.recommended_strategy_counts == d["recommended_strategy_counts"]
        assert obj.dominant_issue_counts == d["dominant_issue_counts"]
        assert obj.strict_qa_block_candidates == 1
        assert obj.notes == ("Example note",)

    def test_to_dict_roundtrip(self) -> None:
        d = _valid_dict()
        obj = InventorySummary.from_dict(d)
        reconstructed = InventorySummary.from_dict(obj.to_dict())
        assert reconstructed == obj

    def test_to_json_roundtrip(self) -> None:
        d = _valid_dict()
        obj = InventorySummary.from_dict(d)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "summary.json"
            obj.to_json(path)
            with open(path) as f:
                loaded = json.load(f)
        reconstructed = InventorySummary.from_dict(loaded)
        assert reconstructed == obj

    def test_missing_schema_raises(self) -> None:
        d = _valid_dict()
        del d["schema"]
        with pytest.raises(QASchemaError, match="Missing required field: schema"):
            InventorySummary.from_dict(d)

    def test_wrong_schema_raises(self) -> None:
        d = _valid_dict()
        d["schema"] = "gradience.other/v1"
        with pytest.raises(QASchemaError, match="Expected schema"):
            InventorySummary.from_dict(d)

    def test_missing_sources_raises(self) -> None:
        d = _valid_dict()
        del d["sources"]
        with pytest.raises(QASchemaError, match="Missing required section: sources"):
            InventorySummary.from_dict(d)

    def test_missing_adapter_status_counts_raises(self) -> None:
        d = _valid_dict()
        del d["adapter_status_counts"]
        with pytest.raises(QASchemaError, match="Missing required section: adapter_status_counts"):
            InventorySummary.from_dict(d)

    def test_missing_pair_risk_counts_raises(self) -> None:
        d = _valid_dict()
        del d["pair_risk_counts"]
        with pytest.raises(QASchemaError, match="Missing required section: pair_risk_counts"):
            InventorySummary.from_dict(d)

    def test_missing_strict_qa_block_candidates_raises(self) -> None:
        d = _valid_dict()
        del d["strict_qa_block_candidates"]
        with pytest.raises(QASchemaError, match="Missing required field: strict_qa_block_candidates"):
            InventorySummary.from_dict(d)

    def test_non_int_count_value_raises(self) -> None:
        d = _valid_dict()
        d["adapter_status_counts"]["eligible"] = 2.5
        with pytest.raises(QASchemaError, match="Values in 'adapter_status_counts' must be int"):
            InventorySummary.from_dict(d)

    def test_non_int_source_count_raises(self) -> None:
        d = _valid_dict()
        d["sources"]["adapter_qa"] = "three"
        with pytest.raises(QASchemaError, match="Values in 'sources' must be int"):
            InventorySummary.from_dict(d)

    def test_non_int_strict_qa_raises(self) -> None:
        d = _valid_dict()
        d["strict_qa_block_candidates"] = 1.0
        with pytest.raises(QASchemaError, match="must be int"):
            InventorySummary.from_dict(d)

    def test_notes_must_be_list_of_str(self) -> None:
        d = _valid_dict()
        d["notes"] = [1, 2]
        with pytest.raises(QASchemaError, match="Each note must be a str"):
            InventorySummary.from_dict(d)

    def test_notes_non_list_raises(self) -> None:
        d = _valid_dict()
        d["notes"] = "not a list"
        with pytest.raises(QASchemaError, match="must be a list of str"):
            InventorySummary.from_dict(d)

    def test_notes_backfilled_when_absent(self) -> None:
        d = _valid_dict()
        del d["notes"]
        obj = InventorySummary.from_dict(d)
        assert obj.notes == ()

    def test_extra_keys_ignored(self) -> None:
        d = _valid_dict()
        d["extra_field"] = "should be ignored"
        d["another"] = 42
        obj = InventorySummary.from_dict(d)
        assert obj.sources == d["sources"]

    def test_empty_count_dicts_accepted(self) -> None:
        d = _valid_dict()
        d["sources"] = {}
        d["adapter_status_counts"] = {}
        d["adapter_flag_counts"] = {}
        d["pair_risk_counts"] = {}
        d["recommended_strategy_counts"] = {}
        d["dominant_issue_counts"] = {}
        d["strict_qa_block_candidates"] = 0
        obj = InventorySummary.from_dict(d)
        assert obj.sources == {}
        assert obj.strict_qa_block_candidates == 0


# ---------------------------------------------------------------------------
# build_inventory_summary helpers & tests
# ---------------------------------------------------------------------------


def _make_qa_artifact(status: str = "eligible", flags: list[str] | None = None) -> AdapterQAArtifact:
    """Create a minimal AdapterQAArtifact for testing."""
    return AdapterQAArtifact(
        adapter_name="test",
        adapter_path="/tmp/test",
        base_model="llama",
        rank_nominal=8,
        n_layers=32,
        utilization_mean=0.5,
        utilization_median=0.5,
        stable_rank_mean=4.0,
        energy_rank_90_p50=4.0,
        rank_waste_ratio=0.5,
        structural_flags=flags or [],
        eval_available=True,
        status=EligibilityStatus(status),
    )


def _make_merge_report_dict(
    pair_risk: str = "low",
    dominant_issue: str = "none",
    strategy: str = "linear",
    eligibility_a: str | None = "eligible",
    eligibility_b: str | None = "eligible",
) -> dict:
    """Create a minimal valid MergeQAReport dict for from_dict loading."""
    return {
        "schema": "gradience.merge_qa_report/v1",
        "adapter_a": {"path": "/tmp/a", "rank": 8, "eligibility_status": eligibility_a},
        "adapter_b": {"path": "/tmp/b", "rank": 8, "eligibility_status": eligibility_b},
        "pair_risk": pair_risk,
        "dominant_issue": dominant_issue,
        "recommended_strategy": strategy,
        "confidence": "high",
        "compatibility_score": 0.9,
    }


def _make_merge_report(**kwargs: str | None) -> MergeQAReport:
    return MergeQAReport.from_dict(_make_merge_report_dict(**kwargs))


class TestBuildInventorySummary:
    """Tests for the build_inventory_summary aggregation function."""

    def test_empty_inputs(self) -> None:
        result = build_inventory_summary([], [])
        assert result.sources == {"qa_artifact_count": 0, "merge_report_count": 0}
        assert result.adapter_status_counts == {}
        assert result.adapter_flag_counts == {}
        assert result.pair_risk_counts == {}
        assert result.recommended_strategy_counts == {}
        assert result.dominant_issue_counts == {}
        assert result.strict_qa_block_candidates == 0

    def test_adapter_status_counts(self) -> None:
        artifacts = [
            _make_qa_artifact("eligible"),
            _make_qa_artifact("eligible"),
            _make_qa_artifact("flagged_weak"),
        ]
        result = build_inventory_summary(artifacts, [])
        assert result.adapter_status_counts == {"eligible": 2, "flagged_weak": 1}
        assert result.sources["qa_artifact_count"] == 3

    def test_adapter_flag_counts(self) -> None:
        artifacts = [
            _make_qa_artifact(flags=["low_utilization", "high_rank_waste"]),
            _make_qa_artifact(flags=["low_utilization"]),
        ]
        result = build_inventory_summary(artifacts, [])
        assert result.adapter_flag_counts == {"low_utilization": 2, "high_rank_waste": 1}

    def test_pair_risk_counts(self) -> None:
        reports = [
            _make_merge_report(pair_risk="low"),
            _make_merge_report(pair_risk="low"),
            _make_merge_report(pair_risk="high"),
        ]
        result = build_inventory_summary([], reports)
        assert result.pair_risk_counts == {"low": 2, "high": 1}
        assert result.sources["merge_report_count"] == 3

    def test_strategy_counts(self) -> None:
        reports = [
            _make_merge_report(strategy="linear"),
            _make_merge_report(strategy="audit_aware"),
            _make_merge_report(strategy="audit_aware"),
        ]
        result = build_inventory_summary([], reports)
        assert result.recommended_strategy_counts == {"linear": 1, "audit_aware": 2}

    def test_dominant_issue_counts(self) -> None:
        reports = [
            _make_merge_report(dominant_issue="none"),
            _make_merge_report(dominant_issue="subspace_conflict"),
            _make_merge_report(dominant_issue="subspace_conflict"),
        ]
        result = build_inventory_summary([], reports)
        assert result.dominant_issue_counts == {"none": 1, "subspace_conflict": 2}

    def test_strict_qa_block_candidates_flagged_weak(self) -> None:
        reports = [_make_merge_report(eligibility_a="flagged_weak")]
        result = build_inventory_summary([], reports)
        assert result.strict_qa_block_candidates == 1

    def test_strict_qa_block_candidates_null(self) -> None:
        reports = [_make_merge_report(eligibility_a=None)]
        result = build_inventory_summary([], reports)
        assert result.strict_qa_block_candidates == 1

    def test_strict_qa_block_candidates_unknown(self) -> None:
        reports = [_make_merge_report(eligibility_a="unknown_no_behavioral_eval")]
        result = build_inventory_summary([], reports)
        assert result.strict_qa_block_candidates == 1

    def test_strict_qa_block_not_double_counted(self) -> None:
        reports = [_make_merge_report(eligibility_a="flagged_weak", eligibility_b=None)]
        result = build_inventory_summary([], reports)
        assert result.strict_qa_block_candidates == 1

    def test_to_dict_produces_valid_schema(self) -> None:
        artifacts = [_make_qa_artifact("eligible"), _make_qa_artifact("flagged_weak")]
        reports = [_make_merge_report(pair_risk="low", dominant_issue="none", strategy="linear")]
        result = build_inventory_summary(artifacts, reports)
        d = result.to_dict()
        roundtripped = InventorySummary.from_dict(d)
        assert roundtripped == result


# ---------------------------------------------------------------------------
# format_inventory_summary tests
# ---------------------------------------------------------------------------


def _summary_for_format() -> InventorySummary:
    """Return a populated InventorySummary for formatter tests."""
    return InventorySummary(
        sources={"qa_artifact_count": 5, "merge_report_count": 3},
        adapter_status_counts={"eligible": 2, "flagged_weak": 1},
        adapter_flag_counts={"low_utilization": 3, "high_rank_waste": 2},
        pair_risk_counts={"low": 1, "high": 2},
        recommended_strategy_counts={"linear": 1, "audit_aware": 1},
        dominant_issue_counts={"none": 1, "norm_imbalance": 1},
        strict_qa_block_candidates=2,
    )


class TestFormatInventorySummary:
    """Tests for the format_inventory_summary terminal formatter."""

    def test_contains_header(self) -> None:
        output = format_inventory_summary(_summary_for_format())
        assert "INVENTORY OVERVIEW" in output

    def test_contains_sources(self) -> None:
        output = format_inventory_summary(_summary_for_format())
        assert "Merge reports:" in output
        assert "QA artifacts:" in output

    def test_contains_adapter_status(self) -> None:
        output = format_inventory_summary(_summary_for_format())
        assert "eligible:" in output
        assert "flagged_weak:" in output

    def test_contains_strict_qa_count(self) -> None:
        output = format_inventory_summary(_summary_for_format())
        assert "Strict-QA block candidates" in output or "strict_qa_block_candidates" in output.lower()
        assert "2" in output  # strict-QA count appears somewhere

    def test_sources_label_qa_artifacts(self) -> None:
        """Source label for QA artifact count should read 'QA artifacts:'."""
        text = format_inventory_summary(_summary_for_format())
        assert "QA artifacts:" in text

    def test_sources_label_merge_reports(self) -> None:
        """Source label for merge report count should read 'Merge reports:'."""
        text = format_inventory_summary(_summary_for_format())
        assert "Merge reports:" in text

    def test_sources_label_not_mangled(self) -> None:
        """Source labels should NOT contain the old mangled forms."""
        text = format_inventory_summary(_summary_for_format())
        assert "Qa artifact" not in text
        assert "Merge report:" not in text or "Merge reports:" in text

    def test_empty_section_omitted(self) -> None:
        summary = InventorySummary(
            sources={"qa_artifact_count": 1, "merge_report_count": 0},
            adapter_status_counts={"eligible": 1},
            adapter_flag_counts={},
            pair_risk_counts={},
            recommended_strategy_counts={},
            dominant_issue_counts={},
            strict_qa_block_candidates=0,
        )
        output = format_inventory_summary(summary)
        # Empty structural detail sections should not appear
        assert "Flags" not in output or "Flags:" not in output
        # But SOURCE QA SNAPSHOT should still be present when adapters exist
        assert "SOURCE QA SNAPSHOT" in output


# ---------------------------------------------------------------------------
# Example file smoke tests
# ---------------------------------------------------------------------------


class TestExampleFiles:
    @pytest.mark.parametrize("path", sorted(glob.glob("examples/inventory/*.json")))
    def test_example_loads_via_from_dict(self, path: str) -> None:
        with open(path) as f:
            d = json.load(f)
        summary = InventorySummary.from_dict(d)
        assert summary.sources["qa_artifact_count"] >= 0


class TestStrictReloadInvariant:
    """from_dict(to_dict(obj)) must produce an identical object."""

    def test_roundtrip_identity(self) -> None:
        d = _valid_dict()
        original = InventorySummary.from_dict(d)
        reloaded = InventorySummary.from_dict(original.to_dict())
        assert reloaded == original

    def test_roundtrip_with_empty_counts(self) -> None:
        d = _valid_dict()
        d["adapter_flag_counts"] = {}
        d["pair_risk_counts"] = {}
        original = InventorySummary.from_dict(d)
        reloaded = InventorySummary.from_dict(original.to_dict())
        assert reloaded == original

    def test_roundtrip_with_notes(self) -> None:
        d = _valid_dict()
        d["notes"] = ["note one", "note two"]
        original = InventorySummary.from_dict(d)
        reloaded = InventorySummary.from_dict(original.to_dict())
        assert reloaded == original


class TestIntegrationWithExampleFiles:
    def test_summarize_from_existing_examples(self) -> None:
        """Build summary from the existing QA and report example files."""
        from gradience.api import summarize_inventory

        summary = summarize_inventory(
            qa_dir="examples/qa",
            report_dir="examples/reports",
        )
        assert summary.sources["qa_artifact_count"] >= 1
        assert summary.sources["merge_report_count"] >= 1
        # Round-trip
        d = summary.to_dict()
        summary2 = InventorySummary.from_dict(d)
        assert summary2 == summary


# ---------------------------------------------------------------------------
# Inventory action plan tests
# ---------------------------------------------------------------------------


def _make_qa_with_name(name: str, status: str = "eligible", eval_dataset: str = "qnli_dev") -> AdapterQAArtifact:
    """Create a QA artifact with a specific name and eval_dataset."""
    return AdapterQAArtifact(
        adapter_name=name,
        adapter_path=f"/tmp/{name}",
        base_model="distilbert-base-uncased",
        rank_nominal=16,
        n_layers=24,
        utilization_mean=0.5,
        utilization_median=0.5,
        stable_rank_mean=4.0,
        energy_rank_90_p50=4.0,
        rank_waste_ratio=0.5,
        structural_flags=[],
        eval_available=True,
        eval_dataset=eval_dataset,
        status=EligibilityStatus(status),
    )


def _make_report_with_names(
    name_a: str,
    name_b: str,
    pair_risk: str = "medium",
    advisory: str | None = None,
    eligibility_a: str = "eligible",
    eligibility_b: str = "eligible",
) -> MergeQAReport:
    """Create a merge report with specific adapter names and optional advisory."""
    d = {
        "schema": "gradience.merge_qa_report/v1",
        "adapter_a": {"path": f"/tmp/{name_a}", "rank": 16, "eligibility_status": eligibility_a},
        "adapter_b": {"path": f"/tmp/{name_b}", "rank": 16, "eligibility_status": eligibility_b},
        "pair_risk": pair_risk,
        "dominant_issue": "none",
        "recommended_strategy": "linear",
        "confidence": "high",
        "compatibility_score": 0.5,
    }
    if advisory:
        d["task_relationship_advisory"] = advisory
    return MergeQAReport.from_dict(d)


class TestActionPlanSameTask:
    """Same-task control inventory: advisory silent, confirmatory."""

    def test_no_cross_task_caution(self):
        qas = [_make_qa_with_name("a"), _make_qa_with_name("b")]
        reports = [_make_report_with_names("a", "b")]
        plan = build_action_plan(qas, reports)
        assert plan.cross_task_caution == ()
        assert len(plan.same_task_priority) == 1

    def test_summary_confirmatory(self):
        qas = [_make_qa_with_name("a"), _make_qa_with_name("b")]
        reports = [_make_report_with_names("a", "b")]
        plan = build_action_plan(qas, reports)
        assert "confirmatory" in plan.summary_line.lower()

    def test_format_no_caution(self):
        qas = [_make_qa_with_name("a"), _make_qa_with_name("b")]
        reports = [_make_report_with_names("a", "b")]
        plan = build_action_plan(qas, reports)
        text = format_action_plan(plan)
        assert "INVENTORY ACTION PLAN" in text
        assert "Cross-task caution" in text


class TestActionPlanMixedTask:
    """Mixed-task inventory: advisory fires, candidate reduction."""

    def test_cross_task_caution_present(self):
        qas = [
            _make_qa_with_name("sst2_a", eval_dataset="sst2_dev"),
            _make_qa_with_name("qnli_a", eval_dataset="qnli_dev"),
        ]
        reports = [
            _make_report_with_names("sst2_a", "qnli_a", advisory="Cross-task merge detected"),
        ]
        plan = build_action_plan(qas, reports)
        assert len(plan.cross_task_caution) > 0
        assert plan.same_task_priority == ()

    def test_mixed_inventory_with_same_and_cross(self):
        qas = [
            _make_qa_with_name("sst2_a", eval_dataset="sst2_dev"),
            _make_qa_with_name("sst2_b", eval_dataset="sst2_dev"),
            _make_qa_with_name("qnli_a", eval_dataset="qnli_dev"),
        ]
        reports = [
            _make_report_with_names("sst2_a", "sst2_b"),  # same-task, no advisory
            _make_report_with_names("sst2_a", "qnli_a", advisory="Cross-task merge"),
            _make_report_with_names("sst2_b", "qnli_a", advisory="Cross-task merge"),
        ]
        plan = build_action_plan(qas, reports)
        assert len(plan.same_task_priority) == 1
        assert len(plan.cross_task_caution) > 0
        assert plan.retained_count == 1
        assert plan.total_pairs == 3
        assert "reduction" in plan.summary_line.lower() or "boundary" in plan.summary_line.lower()

    def test_summary_mentions_reduction(self):
        qas = [
            _make_qa_with_name("sst2_a", eval_dataset="sst2_dev"),
            _make_qa_with_name("sst2_b", eval_dataset="sst2_dev"),
            _make_qa_with_name("qnli_a", eval_dataset="qnli_dev"),
            _make_qa_with_name("qnli_b", eval_dataset="qnli_dev"),
        ]
        reports = [
            _make_report_with_names("sst2_a", "sst2_b"),
            _make_report_with_names("qnli_a", "qnli_b"),
            _make_report_with_names("sst2_a", "qnli_a", advisory="Cross-task"),
            _make_report_with_names("sst2_a", "qnli_b", advisory="Cross-task"),
            _make_report_with_names("sst2_b", "qnli_a", advisory="Cross-task"),
            _make_report_with_names("sst2_b", "qnli_b", advisory="Cross-task"),
        ]
        plan = build_action_plan(qas, reports)
        assert plan.retained_count == 2
        assert plan.total_pairs == 6
        assert "reduction" in plan.summary_line.lower()


class TestActionPlanMessyInventory:
    """Messy inventory: weak sources dominate."""

    def test_exclude_weak_sources(self):
        qas = [
            _make_qa_with_name("good_a"),
            _make_qa_with_name("weak_b", status="flagged_weak"),
        ]
        reports = [_make_report_with_names("good_a", "weak_b", eligibility_b="flagged_weak")]
        plan = build_action_plan(qas, reports)
        assert len(plan.exclude) == 1
        assert "weak_b" in plan.exclude[0]

    def test_exclude_unknown_sources(self):
        qas = [
            _make_qa_with_name("good_a"),
            _make_qa_with_name("unknown_b", status="unknown_no_behavioral_eval"),
        ]
        reports = [_make_report_with_names("good_a", "unknown_b", eligibility_b="unknown_no_behavioral_eval")]
        plan = build_action_plan(qas, reports)
        assert len(plan.exclude) == 1
        assert "unknown_b" in plan.exclude[0]

    def test_format_shows_all_sections(self):
        qas = [_make_qa_with_name("a"), _make_qa_with_name("b", status="flagged_weak")]
        reports = [_make_report_with_names("a", "b", eligibility_b="flagged_weak")]
        plan = build_action_plan(qas, reports)
        text = format_action_plan(plan)
        assert "Exclude / deprioritize" in text
        assert "Same-task safe zone" in text
        assert "Cross-task caution" in text
        assert "Evaluate first" in text
        assert "Summary" in text


class TestActionPlanGuardrails:
    """Ensure no severity grading or hidden logic."""

    def test_no_severity_language(self):
        qas = [
            _make_qa_with_name("sst2_a", eval_dataset="sst2_dev"),
            _make_qa_with_name("qnli_a", eval_dataset="qnli_dev"),
        ]
        reports = [_make_report_with_names("sst2_a", "qnli_a", advisory="Cross-task")]
        plan = build_action_plan(qas, reports)
        text = format_action_plan(plan)
        assert "catastrophic" not in text.lower()
        assert "severity" not in text.lower()
        assert "optimal" not in text.lower()

    def test_action_plan_structure_always_present(self):
        """Even empty inventories produce a valid action plan."""
        plan = build_action_plan([], [])
        text = format_action_plan(plan)
        assert "INVENTORY ACTION PLAN" in text
