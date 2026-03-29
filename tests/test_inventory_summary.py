"""Tests for InventorySummary dataclass, from_dict validation, and build_inventory_summary."""

from __future__ import annotations

import glob
import json
import tempfile
from pathlib import Path
from typing import Any

import pytest

from gradience.exceptions import QASchemaError
from gradience.vnext.audit.qa_artifact import AdapterQAArtifact
from gradience.vnext.inventory.batch import build_batch_summary, format_batch_summary
from gradience.vnext.inventory.run_bundle import (
    build_comparison_md,
    derive_drift_summary,
    format_drift_summary_md,
)
from gradience.vnext.inventory.summary import (
    LARGE_INVENTORY_ADAPTER_THRESHOLD,
    LARGE_INVENTORY_PAIR_THRESHOLD,
    SCHEMA_ID,
    InventoryActionPlan,
    InventorySummary,
    build_action_plan,
    build_inventory_summary,
    derive_inventory_policy_summary,
    derive_region_summary,
    format_action_plan,
    format_candidate_space_map,
    format_inventory_summary,
    format_policy_summary,
    format_region_candidate_table,
    format_region_summary,
    is_large_inventory,
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


# ---------------------------------------------------------------------------
# Near-miss candidate tests
# ---------------------------------------------------------------------------


class TestActionPlanNearMiss:
    """Near-miss: same-task, structurally clean, but one source is weak."""

    def test_near_miss_detected_for_weak_same_task_pair(self):
        """A same-task pair with one flagged_weak source is a near-miss."""
        qas = [
            _make_qa_with_name("good_a"),
            _make_qa_with_name("weak_b", status="flagged_weak"),
        ]
        reports = [_make_report_with_names("good_a", "weak_b", pair_risk="low", eligibility_b="flagged_weak")]
        plan = build_action_plan(qas, reports)
        assert len(plan.near_miss_candidates) == 1
        assert "good_a" in plan.near_miss_candidates[0]
        assert "weak_b" in plan.near_miss_candidates[0]
        # Should NOT be in same_task_priority (retained)
        assert len(plan.same_task_priority) == 0
        # Should NOT be in evaluate_first
        assert len(plan.evaluate_first) == 0

    def test_near_miss_includes_detail(self):
        """Near-miss detail includes risk, strategy, and reason."""
        qas = [
            _make_qa_with_name("good_a"),
            _make_qa_with_name("weak_b", status="flagged_weak"),
        ]
        reports = [_make_report_with_names("good_a", "weak_b", pair_risk="low", eligibility_b="flagged_weak")]
        plan = build_action_plan(qas, reports)
        assert len(plan.near_miss_detail) == 1
        label, risk, strategy, reason = plan.near_miss_detail[0]
        assert risk == "low"
        assert "evidence-constrained" in reason
        assert "weak_b" in reason

    def test_near_miss_not_triggered_for_high_risk(self):
        """High-risk pairs with weak sources are NOT near-miss — just excluded."""
        qas = [
            _make_qa_with_name("good_a"),
            _make_qa_with_name("weak_b", status="flagged_weak"),
        ]
        reports = [_make_report_with_names("good_a", "weak_b", pair_risk="high", eligibility_b="flagged_weak")]
        plan = build_action_plan(qas, reports)
        assert len(plan.near_miss_candidates) == 0

    def test_near_miss_not_triggered_for_cross_task(self):
        """Cross-task pairs with weak sources are NOT near-miss."""
        qas = [
            _make_qa_with_name("good_a", eval_dataset="sst2"),
            _make_qa_with_name("weak_b", status="flagged_weak", eval_dataset="ag_news"),
        ]
        reports = [
            _make_report_with_names(
                "good_a", "weak_b", pair_risk="low", eligibility_b="flagged_weak", advisory="Cross-task"
            )
        ]
        plan = build_action_plan(qas, reports)
        assert len(plan.near_miss_candidates) == 0

    def test_near_miss_medium_risk_accepted(self):
        """Medium-risk same-task pairs with weak sources are near-miss."""
        qas = [
            _make_qa_with_name("good_a"),
            _make_qa_with_name("weak_b", status="flagged_weak"),
        ]
        reports = [_make_report_with_names("good_a", "weak_b", pair_risk="medium", eligibility_b="flagged_weak")]
        plan = build_action_plan(qas, reports)
        assert len(plan.near_miss_candidates) == 1

    def test_near_miss_shown_in_terminal_format(self):
        """Near-miss section appears in formatted output."""
        qas = [
            _make_qa_with_name("good_a"),
            _make_qa_with_name("weak_b", status="flagged_weak"),
        ]
        reports = [_make_report_with_names("good_a", "weak_b", pair_risk="low", eligibility_b="flagged_weak")]
        plan = build_action_plan(qas, reports)
        text = format_action_plan(plan)
        assert "Near-miss candidates" in text
        assert "evidence-constrained" in text

    def test_near_miss_absent_when_no_candidates(self):
        """No near-miss section when all pairs are cleanly retained or excluded."""
        qas = [_make_qa_with_name("a"), _make_qa_with_name("b")]
        reports = [_make_report_with_names("a", "b")]
        plan = build_action_plan(qas, reports)
        assert len(plan.near_miss_candidates) == 0
        text = format_action_plan(plan)
        assert "Near-miss" not in text

    def test_near_miss_with_unknown_status(self):
        """Pairs with unknown_no_behavioral_eval source qualify as near-miss."""
        qas = [
            _make_qa_with_name("good_a"),
            _make_qa_with_name("unknown_b", status="unknown_no_behavioral_eval"),
        ]
        reports = [
            _make_report_with_names("good_a", "unknown_b", pair_risk="low", eligibility_b="unknown_no_behavioral_eval")
        ]
        plan = build_action_plan(qas, reports)
        assert len(plan.near_miss_candidates) == 1

    def test_near_miss_coexists_with_retained(self):
        """Near-miss pairs coexist alongside retained same-task pairs."""
        qas = [
            _make_qa_with_name("good_a"),
            _make_qa_with_name("good_b"),
            _make_qa_with_name("weak_c", status="flagged_weak"),
        ]
        reports = [
            _make_report_with_names("good_a", "good_b", pair_risk="low"),
            _make_report_with_names("good_a", "weak_c", pair_risk="low", eligibility_b="flagged_weak"),
        ]
        plan = build_action_plan(qas, reports)
        assert len(plan.same_task_priority) == 1  # good_a × good_b
        assert len(plan.near_miss_candidates) == 1  # good_a × weak_c
        assert len(plan.evaluate_first) == 1


# ---------------------------------------------------------------------------
# Evidence tier tests (Project C)
# ---------------------------------------------------------------------------


class TestEvidenceTierCounts:
    """Evidence tier tracking in InventorySummary."""

    def test_all_eligible_behavioral_reported(self):
        """Eligible adapters with eval = behavioral_reported."""
        artifacts = [_make_qa_artifact("eligible"), _make_qa_artifact("eligible")]
        result = build_inventory_summary(artifacts, [])
        assert result.evidence_tier_counts.get("behavioral_reported", 0) == 2
        assert result.evidence_tier_counts.get("behavioral_missing", 0) == 0

    def test_no_eval_behavioral_missing(self):
        """Adapter with eval_available=False = behavioral_missing."""
        qa = AdapterQAArtifact(
            adapter_name="no_eval",
            adapter_path="/tmp/no_eval",
            base_model="llama",
            rank_nominal=8,
            n_layers=32,
            utilization_mean=0.5,
            utilization_median=0.5,
            stable_rank_mean=4.0,
            energy_rank_90_p50=4.0,
            rank_waste_ratio=0.5,
            structural_flags=[],
            eval_available=False,
            status=EligibilityStatus("unknown_no_behavioral_eval"),
        )
        result = build_inventory_summary([qa], [])
        assert result.evidence_tier_counts.get("behavioral_missing", 0) == 1

    def test_flagged_weak_with_eval_behavioral_weak(self):
        """Flagged-weak adapter with eval = behavioral_weak."""
        qa = _make_qa_artifact("flagged_weak")
        result = build_inventory_summary([qa], [])
        assert result.evidence_tier_counts.get("behavioral_weak", 0) == 1

    def test_evidence_tier_roundtrip(self):
        """Evidence tiers survive to_dict/from_dict roundtrip."""
        artifacts = [_make_qa_artifact("eligible"), _make_qa_artifact("flagged_weak")]
        result = build_inventory_summary(artifacts, [])
        d = result.to_dict()
        reloaded = InventorySummary.from_dict(d)
        assert reloaded.evidence_tier_counts == result.evidence_tier_counts

    def test_evidence_tier_backfill_from_old_schema(self):
        """Old v1 dicts without evidence_tier_counts get empty dict."""
        d = _valid_dict()
        assert "evidence_tier_counts" not in d
        obj = InventorySummary.from_dict(d)
        assert obj.evidence_tier_counts == {}


class TestFormatInventorySummaryEvidenceTier:
    """Evidence tier display in terminal formatter."""

    def test_evidence_tier_section_present(self):
        """Summary with evidence tiers should show them."""
        summary = InventorySummary(
            sources={"qa_artifact_count": 3, "merge_report_count": 1},
            adapter_status_counts={"eligible": 2, "flagged_weak": 1},
            adapter_flag_counts={},
            pair_risk_counts={"low": 1},
            recommended_strategy_counts={"linear": 1},
            dominant_issue_counts={"none": 1},
            strict_qa_block_candidates=0,
            evidence_tier_counts={"behavioral_reported": 2, "behavioral_weak": 1},
        )
        text = format_inventory_summary(summary)
        assert "Evidence tier:" in text
        assert "behavioral_reported" in text
        assert "behavioral_weak" in text

    def test_non_verification_disclaimer(self):
        """Should use 'user-reported', not 'verified'."""
        summary = InventorySummary(
            sources={"qa_artifact_count": 2, "merge_report_count": 0},
            adapter_status_counts={"eligible": 2},
            adapter_flag_counts={},
            pair_risk_counts={},
            recommended_strategy_counts={},
            dominant_issue_counts={},
            strict_qa_block_candidates=0,
            evidence_tier_counts={"behavioral_reported": 2},
        )
        text = format_inventory_summary(summary)
        assert "user-reported" in text
        assert "verified" not in text.lower()


class TestStructuralDetailMultiLine:
    """STRUCTURAL DETAIL should break into multi-line when > 3 items."""

    def test_many_issues_multi_line(self):
        summary = InventorySummary(
            sources={"qa_artifact_count": 5, "merge_report_count": 10},
            adapter_status_counts={"eligible": 5},
            adapter_flag_counts={},
            pair_risk_counts={"low": 3, "medium": 4, "high": 3},
            recommended_strategy_counts={"linear": 3, "norm_equalized": 4, "audit_aware": 3},
            dominant_issue_counts={
                "none": 2,
                "norm_imbalance": 3,
                "subspace_conflict": 2,
                "high_redundancy": 2,
                "partial_redundancy": 1,
            },
            strict_qa_block_candidates=0,
        )
        text = format_inventory_summary(summary)
        # Issues has 5 items, should be multi-line (indented sub-items)
        assert "Issues:" in text
        # Should have individual lines for each issue
        assert "norm_imbalance:" in text
        assert "subspace_conflict:" in text


# ---------------------------------------------------------------------------
# Inventory policy summary tests
# ---------------------------------------------------------------------------


def _plan_for_policy(
    *,
    total_pairs: int = 6,
    retained_count: int = 3,
    cross_task_count: int = 3,
    exclude: tuple[str, ...] = (),
) -> InventoryActionPlan:
    """Build a minimal InventoryActionPlan for policy summary tests."""
    return InventoryActionPlan(
        exclude=exclude,
        same_task_priority=tuple(f"p{i}" for i in range(retained_count)),
        cross_task_caution=tuple(f"region_{i}" for i in range(min(cross_task_count, 2))),
        evaluate_first=tuple(f"p{i}" for i in range(min(retained_count, 4))),
        summary_line="test summary",
        total_pairs=total_pairs,
        retained_count=retained_count,
        cross_task_count=cross_task_count,
    )


class TestPolicySummarySameTaskControl:
    """Same-task control: all same-task, all eligible, all low risk."""

    def _make(self):
        summary = InventorySummary(
            sources={"qa_artifact_count": 4, "merge_report_count": 6},
            adapter_status_counts={"eligible": 4},
            adapter_flag_counts={},
            pair_risk_counts={"low": 6},
            recommended_strategy_counts={"linear": 6},
            dominant_issue_counts={"none": 6},
            strict_qa_block_candidates=0,
        )
        plan = _plan_for_policy(total_pairs=6, retained_count=6, cross_task_count=0)
        return derive_inventory_policy_summary(summary, plan)

    def test_inventory_type(self):
        assert self._make()["inventory_type"] == "same_task"

    def test_dominant_driver(self):
        assert self._make()["dominant_driver"] == "none"

    def test_exploration_posture(self):
        assert self._make()["exploration_posture"] == "confirmatory"

    def test_constraint_wording(self):
        ps = self._make()
        assert ps["constraint"] == ("This same-task inventory is mostly confirmatory; low-risk merges can proceed.")


class TestPolicySummaryMixedTask:
    """Standard mixed-task: some same-task, some cross-task, clean sources."""

    def _make(self):
        summary = InventorySummary(
            sources={"qa_artifact_count": 5, "merge_report_count": 10},
            adapter_status_counts={"eligible": 5},
            adapter_flag_counts={},
            pair_risk_counts={"low": 4, "medium": 4, "high": 2},
            recommended_strategy_counts={"linear": 4, "norm_equalized": 4, "audit_aware": 2},
            dominant_issue_counts={"none": 4, "norm_imbalance": 4, "subspace_conflict": 2},
            strict_qa_block_candidates=0,
        )
        plan = _plan_for_policy(total_pairs=10, retained_count=6, cross_task_count=4)
        return derive_inventory_policy_summary(summary, plan)

    def test_inventory_type(self):
        assert self._make()["inventory_type"] == "mixed_task"

    def test_dominant_driver(self):
        assert self._make()["dominant_driver"] == "task_boundary"

    def test_exploration_posture(self):
        assert self._make()["exploration_posture"] == "moderate"

    def test_constraint_wording(self):
        ps = self._make()
        assert ps["constraint"] == (
            "Task boundary is the binding constraint; same-task pairs are the credible exploration space."
        )


class TestPolicySummaryMessyMixedQuality:
    """Messy mixed-quality: some weak sources, narrow retained set."""

    def _make(self):
        summary = InventorySummary(
            sources={"qa_artifact_count": 6, "merge_report_count": 15},
            adapter_status_counts={"eligible": 4, "flagged_weak": 2},
            adapter_flag_counts={"low_utilization": 2},
            pair_risk_counts={"low": 5, "medium": 5, "high": 5},
            recommended_strategy_counts={"linear": 5, "norm_equalized": 5, "audit_aware": 5},
            dominant_issue_counts={"none": 5, "norm_imbalance": 5, "subspace_conflict": 5},
            strict_qa_block_candidates=4,
        )
        plan = _plan_for_policy(
            total_pairs=15,
            retained_count=1,
            cross_task_count=6,
            exclude=("weak_a: weak source", "weak_b: weak source"),
        )
        return derive_inventory_policy_summary(summary, plan)

    def test_inventory_type(self):
        assert self._make()["inventory_type"] == "mixed_quality"

    def test_dominant_driver(self):
        assert self._make()["dominant_driver"] == "source_qa"

    def test_exploration_posture(self):
        assert self._make()["exploration_posture"] == "narrow"

    def test_constraint_wording(self):
        ps = self._make()
        assert ps["constraint"] == (
            "Source QA is the binding constraint; resolve weak evidence before exploring merges."
        )


class TestPolicySummaryWeakEvidence:
    """Weak-evidence: all sources are unknown_no_behavioral_eval."""

    def _make(self):
        summary = InventorySummary(
            sources={"qa_artifact_count": 3, "merge_report_count": 3},
            adapter_status_counts={"unknown_no_behavioral_eval": 3},
            adapter_flag_counts={},
            pair_risk_counts={"low": 3},
            recommended_strategy_counts={"linear": 3},
            dominant_issue_counts={"none": 3},
            strict_qa_block_candidates=3,
        )
        plan = _plan_for_policy(
            total_pairs=3,
            retained_count=0,
            cross_task_count=0,
            exclude=("a: missing", "b: missing", "c: missing"),
        )
        return derive_inventory_policy_summary(summary, plan)

    def test_inventory_type(self):
        assert self._make()["inventory_type"] == "weak_evidence"

    def test_dominant_driver(self):
        assert self._make()["dominant_driver"] == "source_qa"

    def test_exploration_posture(self):
        assert self._make()["exploration_posture"] == "narrow"


class TestPolicySummaryEmpty:
    """Empty: QA artifacts exist but no pair reports."""

    def _make(self):
        summary = InventorySummary(
            sources={"qa_artifact_count": 2, "merge_report_count": 0},
            adapter_status_counts={"eligible": 2},
            adapter_flag_counts={},
            pair_risk_counts={},
            recommended_strategy_counts={},
            dominant_issue_counts={},
            strict_qa_block_candidates=0,
        )
        plan = _plan_for_policy(total_pairs=0, retained_count=0, cross_task_count=0)
        return derive_inventory_policy_summary(summary, plan)

    def test_inventory_type(self):
        assert self._make()["inventory_type"] == "empty"

    def test_dominant_driver(self):
        assert self._make()["dominant_driver"] == "none"

    def test_constraint_wording(self):
        ps = self._make()
        assert ps["constraint"] == ("No pair reports available; run pairwise merge audits to populate this inventory.")


class TestPolicySummaryFormat:
    """Terminal formatting of the policy summary block."""

    def test_format_contains_all_components(self):
        ps = {
            "inventory_type": "mixed_task",
            "dominant_driver": "task_boundary",
            "exploration_posture": "moderate",
            "constraint": "Task boundary is the binding constraint; "
            "same-task pairs are the credible exploration space.",
        }
        text = format_policy_summary(ps)
        assert "INVENTORY POLICY SUMMARY" in text
        assert "Type:        mixed_task" in text
        assert "Driver:      task_boundary" in text
        assert "Posture:     moderate" in text
        assert "Task boundary" in text

    def test_format_integrated_in_inventory_summary(self):
        """Policy summary block appears between STRUCTURAL DETAIL and INTERPRETATION."""
        summary = InventorySummary(
            sources={"qa_artifact_count": 2, "merge_report_count": 1},
            adapter_status_counts={"eligible": 2},
            adapter_flag_counts={},
            pair_risk_counts={"low": 1},
            recommended_strategy_counts={"linear": 1},
            dominant_issue_counts={"none": 1},
            strict_qa_block_candidates=0,
        )
        ps = {
            "inventory_type": "same_task",
            "dominant_driver": "none",
            "exploration_posture": "confirmatory",
            "constraint": "This same-task inventory is mostly confirmatory; low-risk merges can proceed.",
        }
        text = format_inventory_summary(summary, policy_summary=ps)
        assert "INVENTORY POLICY SUMMARY" in text
        # Policy summary should appear before INTERPRETATION
        ps_pos = text.index("INVENTORY POLICY SUMMARY")
        interp_pos = text.index("INTERPRETATION")
        assert ps_pos < interp_pos

    def test_format_without_policy_summary_unchanged(self):
        """Calling format_inventory_summary without policy_summary produces no block."""
        summary = InventorySummary(
            sources={"qa_artifact_count": 2, "merge_report_count": 1},
            adapter_status_counts={"eligible": 2},
            adapter_flag_counts={},
            pair_risk_counts={"low": 1},
            recommended_strategy_counts={"linear": 1},
            dominant_issue_counts={"none": 1},
            strict_qa_block_candidates=0,
        )
        text = format_inventory_summary(summary)
        assert "INVENTORY POLICY SUMMARY" not in text


class TestPolicySummaryJSONMirror:
    """JSON mirror consistency: derive → serialize → verify keys."""

    def test_json_has_all_keys(self):
        summary = InventorySummary(
            sources={"qa_artifact_count": 4, "merge_report_count": 6},
            adapter_status_counts={"eligible": 4},
            adapter_flag_counts={},
            pair_risk_counts={"low": 4, "high": 2},
            recommended_strategy_counts={"linear": 4, "audit_aware": 2},
            dominant_issue_counts={"none": 4, "subspace_conflict": 2},
            strict_qa_block_candidates=0,
        )
        plan = _plan_for_policy(total_pairs=6, retained_count=2, cross_task_count=4)
        ps = derive_inventory_policy_summary(summary, plan)

        assert "inventory_type" in ps
        assert "dominant_driver" in ps
        assert "exploration_posture" in ps
        assert "constraint" in ps
        assert len(ps) == 4

    def test_json_values_from_controlled_vocabulary(self):
        """All values must come from the allowed enums."""
        valid_types = {"same_task", "cross_task", "mixed_task", "mixed_quality", "weak_evidence", "empty"}
        valid_drivers = {"source_qa", "task_boundary", "structural_risk", "none"}
        valid_postures = {"narrow", "moderate", "broad", "confirmatory"}

        # Test across several regimes
        configs = [
            ({"eligible": 4}, {"low": 6}, 6, 6, 0, ()),
            ({"eligible": 5}, {"low": 4, "high": 2}, 10, 6, 4, ()),
            ({"eligible": 4, "flagged_weak": 2}, {"low": 5, "high": 5}, 15, 1, 6, ("w",)),
            ({"unknown_no_behavioral_eval": 3}, {"low": 3}, 3, 0, 0, ("a",)),
        ]
        for status, risk, tp, ret, ct, excl in configs:
            summary = InventorySummary(
                sources={"qa_artifact_count": sum(status.values()), "merge_report_count": tp},
                adapter_status_counts=status,
                adapter_flag_counts={},
                pair_risk_counts=risk,
                recommended_strategy_counts={"linear": sum(risk.values())},
                dominant_issue_counts={"none": sum(risk.values())},
                strict_qa_block_candidates=0,
            )
            plan = _plan_for_policy(total_pairs=tp, retained_count=ret, cross_task_count=ct, exclude=excl)
            ps = derive_inventory_policy_summary(summary, plan)

            assert ps["inventory_type"] in valid_types, f"Bad type: {ps['inventory_type']}"
            assert ps["dominant_driver"] in valid_drivers, f"Bad driver: {ps['dominant_driver']}"
            assert ps["exploration_posture"] in valid_postures, f"Bad posture: {ps['exploration_posture']}"
            assert isinstance(ps["constraint"], str) and len(ps["constraint"]) > 10


class TestPolicySummaryGuardrails:
    """Guardrails: no free-text, no severity language, deterministic."""

    def test_no_severity_language(self):
        """Policy summary must never use severity grading language."""
        summary = InventorySummary(
            sources={"qa_artifact_count": 4, "merge_report_count": 6},
            adapter_status_counts={"eligible": 4},
            adapter_flag_counts={},
            pair_risk_counts={"high": 6},
            recommended_strategy_counts={"audit_aware": 6},
            dominant_issue_counts={"subspace_conflict": 6},
            strict_qa_block_candidates=0,
        )
        plan = _plan_for_policy(total_pairs=6, retained_count=6, cross_task_count=0)
        ps = derive_inventory_policy_summary(summary, plan)
        all_values = " ".join(str(v) for v in ps.values())
        assert "catastrophic" not in all_values.lower()
        assert "severity" not in all_values.lower()
        assert "optimal" not in all_values.lower()

    def test_deterministic(self):
        """Same inputs always produce same output."""
        summary = InventorySummary(
            sources={"qa_artifact_count": 3, "merge_report_count": 3},
            adapter_status_counts={"eligible": 2, "flagged_weak": 1},
            adapter_flag_counts={},
            pair_risk_counts={"low": 2, "high": 1},
            recommended_strategy_counts={"linear": 2, "audit_aware": 1},
            dominant_issue_counts={"none": 2, "subspace_conflict": 1},
            strict_qa_block_candidates=1,
        )
        plan = _plan_for_policy(total_pairs=3, retained_count=1, cross_task_count=1, exclude=("w",))
        ps1 = derive_inventory_policy_summary(summary, plan)
        ps2 = derive_inventory_policy_summary(summary, plan)
        assert ps1 == ps2


# ---------------------------------------------------------------------------
# Inventory drift summary tests
# ---------------------------------------------------------------------------


def _preflight_json(
    *,
    adapter_count: int = 4,
    pair_count: int = 6,
    retained: int = 3,
    advisory: int = 2,
    bev: int = 2,
    tsc: int = 4,
    excluded: list[str] | None = None,
    policy: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Build a minimal preflight_summary.json dict for drift tests."""
    d: dict[str, Any] = {
        "run_id": "test-run",
        "adapter_count": adapter_count,
        "pair_count": pair_count,
        "retained_candidate_count": retained,
        "advisory_pair_count": advisory,
        "behavioral_evidence_count": bev,
        "total_source_count": tsc,
        "excluded_sources": excluded or [],
    }
    if policy is not None:
        d["inventory_policy_summary"] = policy
    return d


class TestDriftNarrowing:
    """F1: Reduced candidate subset shrinks → narrowing."""

    def test_status_is_narrowing(self):
        prev = _preflight_json(retained=4)
        curr = _preflight_json(retained=2)
        drift = derive_drift_summary(curr, prev)
        assert drift["status"] == "narrowing"

    def test_retained_delta_negative(self):
        prev = _preflight_json(retained=4)
        curr = _preflight_json(retained=2)
        drift = derive_drift_summary(curr, prev)
        assert drift["retained_candidate_delta"] == -2

    def test_safe_zone_shrunk(self):
        prev = _preflight_json(retained=4)
        curr = _preflight_json(retained=2)
        drift = derive_drift_summary(curr, prev)
        assert drift["same_task_safe_zone_change"] == "shrunk"

    def test_implication_narrower(self):
        prev = _preflight_json(retained=4)
        curr = _preflight_json(retained=2)
        drift = derive_drift_summary(curr, prev)
        assert drift["implication"] == "current_run_materially_narrower"


class TestDriftBroadening:
    """F2: Candidate subset expands → broadening."""

    def test_status_is_broadening(self):
        prev = _preflight_json(retained=2)
        curr = _preflight_json(retained=6)
        drift = derive_drift_summary(curr, prev)
        assert drift["status"] == "broadening"

    def test_safe_zone_expanded(self):
        prev = _preflight_json(retained=2)
        curr = _preflight_json(retained=6)
        drift = derive_drift_summary(curr, prev)
        assert drift["same_task_safe_zone_change"] == "expanded"

    def test_implication_broader(self):
        prev = _preflight_json(retained=2)
        curr = _preflight_json(retained=6)
        drift = derive_drift_summary(curr, prev)
        assert drift["implication"] == "current_run_materially_broader"


class TestDriftStable:
    """F3: Key summary fields unchanged → stable."""

    def test_status_is_stable(self):
        prev = _preflight_json(retained=3)
        curr = _preflight_json(retained=3)
        drift = derive_drift_summary(curr, prev)
        assert drift["status"] == "stable"

    def test_safe_zone_unchanged(self):
        prev = _preflight_json(retained=3)
        curr = _preflight_json(retained=3)
        drift = derive_drift_summary(curr, prev)
        assert drift["same_task_safe_zone_change"] == "unchanged"

    def test_implication_no_change(self):
        prev = _preflight_json(retained=3, bev=2, tsc=4)
        curr = _preflight_json(retained=3, bev=2, tsc=4)
        drift = derive_drift_summary(curr, prev)
        assert drift["implication"] == "no_substantial_change"


class TestDriftEvidenceImproved:
    """F4: Evidence tier shift → improved."""

    def test_evidence_improved(self):
        prev = _preflight_json(retained=3, bev=2, tsc=4)
        curr = _preflight_json(retained=3, bev=4, tsc=4)
        drift = derive_drift_summary(curr, prev)
        assert drift["evidence_profile_change"] == "improved"

    def test_implication_evidence_improved(self):
        prev = _preflight_json(retained=3, bev=2, tsc=4)
        curr = _preflight_json(retained=3, bev=4, tsc=4)
        drift = derive_drift_summary(curr, prev)
        assert drift["implication"] == "evidence_improved_posture_unchanged"


class TestDriftEvidenceDegraded:
    """F5: Evidence profile worsens → degraded."""

    def test_evidence_degraded(self):
        prev = _preflight_json(retained=3, bev=4, tsc=4)
        curr = _preflight_json(retained=3, bev=2, tsc=4)
        drift = derive_drift_summary(curr, prev)
        assert drift["evidence_profile_change"] == "degraded"

    def test_implication_evidence_degraded(self):
        prev = _preflight_json(retained=3, bev=4, tsc=4)
        curr = _preflight_json(retained=3, bev=2, tsc=4)
        drift = derive_drift_summary(curr, prev)
        assert drift["implication"] == "evidence_degraded"


class TestDriftPolicyChange:
    """F6: Policy summary changes are detected."""

    def test_policy_change_detected(self):
        prev_policy = {
            "inventory_type": "same_task",
            "dominant_driver": "none",
            "exploration_posture": "confirmatory",
            "constraint": "This same-task inventory is mostly confirmatory; low-risk merges can proceed.",
        }
        curr_policy = {
            "inventory_type": "mixed_task",
            "dominant_driver": "task_boundary",
            "exploration_posture": "moderate",
            "constraint": "Task boundary is the binding constraint; "
            "same-task pairs are the credible exploration space.",
        }
        prev = _preflight_json(retained=3, policy=prev_policy)
        curr = _preflight_json(retained=3, policy=curr_policy)
        drift = derive_drift_summary(curr, prev)
        assert drift["policy_change"]["inventory_type_changed"] is True
        assert drift["policy_change"]["dominant_driver_changed"] is True
        assert drift["policy_change"]["exploration_posture_changed"] is True
        assert drift["policy_change"]["constraint_changed"] is True

    def test_policy_unchanged(self):
        policy = {
            "inventory_type": "same_task",
            "dominant_driver": "none",
            "exploration_posture": "confirmatory",
            "constraint": "This same-task inventory is mostly confirmatory; low-risk merges can proceed.",
        }
        prev = _preflight_json(retained=3, policy=policy)
        curr = _preflight_json(retained=3, policy=policy)
        drift = derive_drift_summary(curr, prev)
        assert drift["policy_change"]["inventory_type_changed"] is False
        assert drift["policy_change"]["dominant_driver_changed"] is False

    def test_no_policy_in_either_run(self):
        prev = _preflight_json(retained=3)
        curr = _preflight_json(retained=3)
        drift = derive_drift_summary(curr, prev)
        assert "policy_change" not in drift


class TestDriftJSONMirror:
    """F7: Machine-readable drift object matches human-readable summary."""

    def test_json_has_required_keys(self):
        prev = _preflight_json(retained=4)
        curr = _preflight_json(retained=2)
        drift = derive_drift_summary(curr, prev)
        required = {
            "status",
            "adapter_count_delta",
            "pair_count_delta",
            "retained_candidate_delta",
            "same_task_safe_zone_change",
            "cross_task_caution_zone_change",
            "evidence_profile_change",
            "implication",
        }
        assert required.issubset(drift.keys())

    def test_values_from_controlled_vocabulary(self):
        valid_statuses = {"narrowing", "broadening", "stable"}
        valid_zones = {"expanded", "shrunk", "unchanged"}
        valid_evidence = {"improved", "degraded", "unchanged"}
        valid_implications = {
            "current_run_materially_narrower",
            "current_run_materially_broader",
            "source_composition_changed",
            "evidence_improved_posture_unchanged",
            "evidence_degraded",
            "no_substantial_change",
        }

        configs = [
            (_preflight_json(retained=4), _preflight_json(retained=2)),
            (_preflight_json(retained=2), _preflight_json(retained=6)),
            (_preflight_json(retained=3), _preflight_json(retained=3)),
            (_preflight_json(retained=3, bev=2, tsc=4), _preflight_json(retained=3, bev=4, tsc=4)),
        ]
        for prev, curr in configs:
            drift = derive_drift_summary(curr, prev)
            assert drift["status"] in valid_statuses
            assert drift["same_task_safe_zone_change"] in valid_zones
            assert drift["cross_task_caution_zone_change"] in valid_zones
            assert drift["evidence_profile_change"] in valid_evidence
            assert drift["implication"] in valid_implications

    def test_deterministic(self):
        prev = _preflight_json(retained=4, bev=2, tsc=4)
        curr = _preflight_json(retained=2, bev=4, tsc=5)
        d1 = derive_drift_summary(curr, prev)
        d2 = derive_drift_summary(curr, prev)
        assert d1 == d2


class TestDriftInComparisonMd:
    """Drift summary appears in compare_to_previous.md output."""

    def test_comparison_md_contains_drift_section(self):
        prev = _preflight_json(retained=4)
        curr = _preflight_json(retained=2)
        md = build_comparison_md(current_summary=curr, previous_summary=prev)
        assert "## History / Drift Summary" in md
        assert "narrowing" in md

    def test_comparison_md_contains_implication(self):
        prev = _preflight_json(retained=4)
        curr = _preflight_json(retained=2)
        md = build_comparison_md(current_summary=curr, previous_summary=prev)
        assert "materially narrower" in md


class TestDriftFormatMd:
    """Human-readable drift summary markdown formatting."""

    def test_format_contains_all_sections(self):
        prev = _preflight_json(retained=4, bev=2, tsc=4)
        curr = _preflight_json(retained=2, bev=4, tsc=5)
        drift = derive_drift_summary(curr, prev)
        md = format_drift_summary_md(drift, curr, prev)
        assert "**Status:**" in md
        assert "**Key changes:**" in md
        assert "**Implication:**" in md

    def test_format_with_policy_change(self):
        policy_prev = {
            "inventory_type": "same_task",
            "dominant_driver": "none",
            "exploration_posture": "confirmatory",
            "constraint": "c1",
        }
        policy_curr = {
            "inventory_type": "mixed_task",
            "dominant_driver": "task_boundary",
            "exploration_posture": "moderate",
            "constraint": "c2",
        }
        prev = _preflight_json(retained=3, policy=policy_prev)
        curr = _preflight_json(retained=3, policy=policy_curr)
        drift = derive_drift_summary(curr, prev)
        md = format_drift_summary_md(drift, curr, prev)
        assert "**Policy change:**" in md
        assert "Dominant driver changed" in md

    def test_cross_task_caution_zone_label(self):
        prev = _preflight_json(advisory=2)
        curr = _preflight_json(advisory=5)
        drift = derive_drift_summary(curr, prev)
        md = format_drift_summary_md(drift, curr, prev)
        assert "Cross-task caution zone: expanded" in md


class TestBatchSummaryWithDrift:
    """Batch summary includes per-run drift and extended trend fields."""

    def test_batch_drift_status_per_run(self):
        runs = [
            _preflight_json(retained=4),
            _preflight_json(retained=2),
            _preflight_json(retained=6),
        ]
        batch = build_batch_summary(runs)
        assert batch["runs"][0]["drift_status"] is None  # first run
        assert batch["runs"][1]["drift_status"] == "narrowing"
        assert batch["runs"][2]["drift_status"] == "broadening"

    def test_batch_evidence_trend(self):
        runs = [
            _preflight_json(bev=1, tsc=4),
            _preflight_json(bev=3, tsc=4),
        ]
        batch = build_batch_summary(runs)
        assert batch["evidence_trend"] == "improved"

    def test_batch_policy_changed(self):
        p1 = {
            "inventory_type": "same_task",
            "dominant_driver": "none",
            "exploration_posture": "confirmatory",
            "constraint": "c",
        }
        p2 = {
            "inventory_type": "mixed_task",
            "dominant_driver": "task_boundary",
            "exploration_posture": "moderate",
            "constraint": "d",
        }
        runs = [
            _preflight_json(policy=p1),
            _preflight_json(policy=p2),
        ]
        batch = build_batch_summary(runs)
        assert batch["policy_changed"] is True

    def test_batch_format_has_drift_column(self):
        runs = [
            _preflight_json(retained=4),
            _preflight_json(retained=2),
        ]
        batch = build_batch_summary(runs)
        text = format_batch_summary(batch)
        assert "Drift" in text
        assert "narrowing" in text


# ---------------------------------------------------------------------------
# Large-inventory ergonomics tests
# ---------------------------------------------------------------------------


def _make_region_qa(name: str, eval_dataset: str = "qnli_dev") -> AdapterQAArtifact:
    """Build a minimal QA artifact for region tests."""
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
        status=EligibilityStatus("eligible"),
    )


def _make_region_report(
    name_a: str,
    name_b: str,
    pair_risk: str = "low",
    strategy: str = "linear",
    advisory: str | None = None,
) -> MergeQAReport:
    """Build a minimal merge report for region tests."""
    d: dict[str, Any] = {
        "schema": "gradience.merge_qa_report/v1",
        "adapter_a": {"path": f"/tmp/{name_a}", "rank": 16, "eligibility_status": "eligible"},
        "adapter_b": {"path": f"/tmp/{name_b}", "rank": 16, "eligibility_status": "eligible"},
        "pair_risk": pair_risk,
        "dominant_issue": "none",
        "recommended_strategy": strategy,
        "confidence": "high",
        "compatibility_score": 0.85,
    }
    if advisory:
        d["task_relationship_advisory"] = advisory
    return MergeQAReport.from_dict(d)


def _large_mixed_task_fixtures():
    """8 adapters across 3 tasks: QNLI (3), SST-2 (3), RTE (2).

    Same-task pairs: C(3,2)+C(3,2)+C(2,2) = 3+3+1 = 7
    Cross-task pairs: 3*3 + 3*2 + 3*2 = 9+6+6 = 21
    Total pairs: 28
    """
    qas = [
        _make_region_qa("qnli_a", "qnli_dev"),
        _make_region_qa("qnli_b", "qnli_dev"),
        _make_region_qa("qnli_c", "qnli_dev"),
        _make_region_qa("sst2_a", "sst2_dev"),
        _make_region_qa("sst2_b", "sst2_dev"),
        _make_region_qa("sst2_c", "sst2_dev"),
        _make_region_qa("rte_a", "rte_dev"),
        _make_region_qa("rte_b", "rte_dev"),
    ]
    reports = []
    # Same-task QNLI pairs
    reports.append(_make_region_report("qnli_a", "qnli_b"))
    reports.append(_make_region_report("qnli_a", "qnli_c"))
    reports.append(_make_region_report("qnli_b", "qnli_c"))
    # Same-task SST-2 pairs
    reports.append(_make_region_report("sst2_a", "sst2_b"))
    reports.append(_make_region_report("sst2_a", "sst2_c"))
    reports.append(_make_region_report("sst2_b", "sst2_c"))
    # Same-task RTE pair
    reports.append(_make_region_report("rte_a", "rte_b"))
    # Cross-task pairs (QNLI × SST-2)
    for q in ("qnli_a", "qnli_b", "qnli_c"):
        for s in ("sst2_a", "sst2_b", "sst2_c"):
            reports.append(_make_region_report(q, s, pair_risk="high", strategy="audit_aware", advisory="Cross-task"))
    # Cross-task pairs (QNLI × RTE)
    for q in ("qnli_a", "qnli_b", "qnli_c"):
        for r in ("rte_a", "rte_b"):
            reports.append(_make_region_report(q, r, pair_risk="high", strategy="audit_aware", advisory="Cross-task"))
    # Cross-task pairs (SST-2 × RTE)
    for s in ("sst2_a", "sst2_b", "sst2_c"):
        for r in ("rte_a", "rte_b"):
            reports.append(_make_region_report(s, r, pair_risk="high", strategy="audit_aware", advisory="Cross-task"))

    plan = build_action_plan(qas, reports)
    return qas, reports, plan


def _small_same_task_fixtures():
    """3 adapters, 3 pairs — below threshold."""
    qas = [
        _make_region_qa("qnli_a", "qnli_dev"),
        _make_region_qa("qnli_b", "qnli_dev"),
        _make_region_qa("qnli_c", "qnli_dev"),
    ]
    reports = [
        _make_region_report("qnli_a", "qnli_b"),
        _make_region_report("qnli_a", "qnli_c"),
        _make_region_report("qnli_b", "qnli_c"),
    ]
    plan = build_action_plan(qas, reports)
    return qas, reports, plan


class TestIsLargeInventory:
    """Threshold activation rules."""

    def test_below_both_thresholds(self):
        assert is_large_inventory(7, 19) is False

    def test_at_adapter_threshold(self):
        assert is_large_inventory(8, 10) is True

    def test_at_pair_threshold(self):
        assert is_large_inventory(5, 20) is True

    def test_above_both(self):
        assert is_large_inventory(10, 30) is True

    def test_constants_match_design_doc(self):
        assert LARGE_INVENTORY_ADAPTER_THRESHOLD == 8
        assert LARGE_INVENTORY_PAIR_THRESHOLD == 20


class TestRegionSummaryLargeMixedTask:
    """Region derivation for an 8-adapter, 3-task inventory."""

    def test_enabled(self):
        qas, _, plan = _large_mixed_task_fixtures()
        rs = derive_region_summary(plan, qas)
        assert rs["enabled"] is True

    def test_threshold_trigger(self):
        qas, _, plan = _large_mixed_task_fixtures()
        rs = derive_region_summary(plan, qas)
        assert rs["threshold_trigger"] == "8_adapters"

    def test_same_task_safe_regions(self):
        qas, _, plan = _large_mixed_task_fixtures()
        rs = derive_region_summary(plan, qas)
        assert rs["region_counts"]["same_task_safe_regions"] == 3  # QNLI, SST-2, RTE

    def test_cross_task_caution_regions(self):
        qas, _, plan = _large_mixed_task_fixtures()
        rs = derive_region_summary(plan, qas)
        assert rs["region_counts"]["cross_task_caution_regions"] == 3  # Q×S, Q×R, R×S

    def test_no_weak_evidence_regions(self):
        qas, _, plan = _large_mixed_task_fixtures()
        rs = derive_region_summary(plan, qas)
        assert rs["region_counts"]["weak_evidence_regions"] == 0

    def test_priority_evaluation_regions(self):
        qas, _, plan = _large_mixed_task_fixtures()
        rs = derive_region_summary(plan, qas)
        # evaluate_first is capped at 4 pairs → should overlap with at least 1 region
        assert rs["region_counts"]["priority_evaluation_regions"] >= 1

    def test_candidate_space_map_has_all_regions(self):
        qas, _, plan = _large_mixed_task_fixtures()
        rs = derive_region_summary(plan, qas)
        csmap = rs["candidate_space_map"]
        types = {e["type"] for e in csmap}
        assert "same_task_safe_region" in types
        assert "cross_task_caution_region" in types

    def test_region_types_are_controlled_vocabulary(self):
        qas, _, plan = _large_mixed_task_fixtures()
        rs = derive_region_summary(plan, qas)
        valid_types = {"same_task_safe_region", "cross_task_caution_region", "weak_evidence_region"}
        for entry in rs["candidate_space_map"]:
            assert entry["type"] in valid_types, f"Unexpected type: {entry['type']}"

    def test_status_labels_are_controlled_vocabulary(self):
        qas, _, plan = _large_mixed_task_fixtures()
        rs = derive_region_summary(plan, qas)
        valid_statuses = {"evaluate_first", "same_task_safe", "do_not_explore_casually", "excluded"}
        for entry in rs["candidate_space_map"]:
            assert entry["status"] in valid_statuses, f"Unexpected status: {entry['status']}"

    def test_same_task_regions_have_pairs(self):
        qas, _, plan = _large_mixed_task_fixtures()
        rs = derive_region_summary(plan, qas)
        for entry in rs["candidate_space_map"]:
            if entry["type"] == "same_task_safe_region":
                assert len(entry["pairs"]) > 0

    def test_deterministic(self):
        qas, _, plan = _large_mixed_task_fixtures()
        rs1 = derive_region_summary(plan, qas)
        rs2 = derive_region_summary(plan, qas)
        assert rs1 == rs2


class TestRegionSummaryBelowThreshold:
    """Below threshold: region summary disabled."""

    def test_disabled(self):
        qas, _, plan = _small_same_task_fixtures()
        rs = derive_region_summary(plan, qas)
        assert rs["enabled"] is False

    def test_empty_map(self):
        qas, _, plan = _small_same_task_fixtures()
        rs = derive_region_summary(plan, qas)
        assert rs["candidate_space_map"] == []

    def test_zero_counts(self):
        qas, _, plan = _small_same_task_fixtures()
        rs = derive_region_summary(plan, qas)
        rc = rs["region_counts"]
        assert all(v == 0 for v in rc.values())


class TestFormatRegionSummary:
    """Human-readable region summary block."""

    def test_contains_header(self):
        qas, _, plan = _large_mixed_task_fixtures()
        rs = derive_region_summary(plan, qas)
        text = format_region_summary(rs)
        assert "LARGE INVENTORY REGION SUMMARY" in text

    def test_contains_region_counts(self):
        qas, _, plan = _large_mixed_task_fixtures()
        rs = derive_region_summary(plan, qas)
        text = format_region_summary(rs)
        assert "Same-task safe regions:" in text
        assert "Cross-task caution regions:" in text
        assert "Priority evaluation regions:" in text

    def test_contains_interpretation(self):
        qas, _, plan = _large_mixed_task_fixtures()
        rs = derive_region_summary(plan, qas)
        text = format_region_summary(rs)
        assert "grouped regions" in text

    def test_empty_when_disabled(self):
        qas, _, plan = _small_same_task_fixtures()
        rs = derive_region_summary(plan, qas)
        text = format_region_summary(rs)
        assert text == ""


class TestFormatCandidateSpaceMap:
    """Candidate-space map table."""

    def test_contains_header(self):
        qas, _, plan = _large_mixed_task_fixtures()
        rs = derive_region_summary(plan, qas)
        text = format_candidate_space_map(rs)
        assert "CANDIDATE-SPACE MAP" in text

    def test_contains_table_header(self):
        qas, _, plan = _large_mixed_task_fixtures()
        rs = derive_region_summary(plan, qas)
        text = format_candidate_space_map(rs)
        assert "Region" in text
        assert "Type" in text
        assert "Pairs" in text
        assert "Status" in text

    def test_contains_task_labels(self):
        qas, _, plan = _large_mixed_task_fixtures()
        rs = derive_region_summary(plan, qas)
        text = format_candidate_space_map(rs)
        assert "QNLI" in text
        assert "SST-2" in text
        assert "RTE" in text

    def test_contains_status_labels(self):
        qas, _, plan = _large_mixed_task_fixtures()
        rs = derive_region_summary(plan, qas)
        text = format_candidate_space_map(rs)
        # Should have at least some evaluate first and caution
        assert "evaluate first" in text or "same-task safe" in text
        assert "do not explore casually" in text

    def test_empty_when_disabled(self):
        qas, _, plan = _small_same_task_fixtures()
        rs = derive_region_summary(plan, qas)
        text = format_candidate_space_map(rs)
        assert text == ""


class TestFormatRegionCandidateTable:
    """Region-aware evaluate-first table."""

    def test_groups_by_region(self):
        qas, _, plan = _large_mixed_task_fixtures()
        rs = derive_region_summary(plan, qas)
        text = format_region_candidate_table(rs)
        if text:  # only if there are priority regions
            assert "EVALUATE FIRST (by region)" in text
            assert "QNLI" in text or "SST-2" in text or "RTE" in text

    def test_empty_when_disabled(self):
        qas, _, plan = _small_same_task_fixtures()
        rs = derive_region_summary(plan, qas)
        text = format_region_candidate_table(rs)
        assert text == ""


class TestRegionSummaryJSONMirror:
    """JSON mirror matches design doc contract."""

    def test_json_serializable(self):
        import json as json_mod

        qas, _, plan = _large_mixed_task_fixtures()
        rs = derive_region_summary(plan, qas)
        # Should not raise
        json_mod.dumps(rs)

    def test_enabled_key(self):
        qas, _, plan = _large_mixed_task_fixtures()
        rs = derive_region_summary(plan, qas)
        assert rs["enabled"] is True

    def test_required_keys(self):
        qas, _, plan = _large_mixed_task_fixtures()
        rs = derive_region_summary(plan, qas)
        assert "enabled" in rs
        assert "threshold_trigger" in rs
        assert "region_counts" in rs
        assert "candidate_space_map" in rs

    def test_region_count_keys(self):
        qas, _, plan = _large_mixed_task_fixtures()
        rs = derive_region_summary(plan, qas)
        rc = rs["region_counts"]
        assert "same_task_safe_regions" in rc
        assert "cross_task_caution_regions" in rc
        assert "weak_evidence_regions" in rc
        assert "priority_evaluation_regions" in rc

    def test_map_entry_keys(self):
        qas, _, plan = _large_mixed_task_fixtures()
        rs = derive_region_summary(plan, qas)
        for entry in rs["candidate_space_map"]:
            assert "region" in entry
            assert "type" in entry
            assert "pair_count" in entry
            assert "status" in entry
            assert "pairs" in entry


class TestRegionGuardrails:
    """No new scoring, no free text, deterministic."""

    def test_no_scoring_language_in_region_summary(self):
        qas, _, plan = _large_mixed_task_fixtures()
        rs = derive_region_summary(plan, qas)
        text = format_region_summary(rs)
        for forbidden in ("score", "grade", "severity", "critical", "warning"):
            assert forbidden not in text.lower(), f"Forbidden: {forbidden}"

    def test_no_scoring_language_in_candidate_map(self):
        qas, _, plan = _large_mixed_task_fixtures()
        rs = derive_region_summary(plan, qas)
        text = format_candidate_space_map(rs)
        for forbidden in ("score", "grade", "severity", "critical", "warning"):
            assert forbidden not in text.lower(), f"Forbidden: {forbidden}"

    def test_all_counts_are_integers(self):
        qas, _, plan = _large_mixed_task_fixtures()
        rs = derive_region_summary(plan, qas)
        for v in rs["region_counts"].values():
            assert isinstance(v, int)
        for entry in rs["candidate_space_map"]:
            assert isinstance(entry["pair_count"], int)
