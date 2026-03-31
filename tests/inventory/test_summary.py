"""Tests for gradience.vnext.inventory.summary — schema, builders, formatters."""

from __future__ import annotations

import json

import pytest

from gradience.exceptions import QASchemaError
from gradience.vnext.inventory.summary import (
    InventorySummary,
    build_action_plan,
    build_inventory_summary,
    derive_inventory_policy_summary,
    format_action_plan,
    format_inventory_summary,
    is_large_inventory,
)
from tests.inventory.conftest import (
    StubAdapterRef,
    StubEligibility,
    StubQA,
    StubReport,
    _valid_summary_dict,
)

# ===================================================================
# from_dict validation
# ===================================================================


class TestFromDictValidation:
    """Schema gatekeeper rejects invalid v1 dicts."""

    def test_missing_schema(self) -> None:
        d = _valid_summary_dict()
        del d["schema"]
        with pytest.raises(QASchemaError, match="schema"):
            InventorySummary.from_dict(d)

    def test_wrong_schema(self) -> None:
        d = _valid_summary_dict()
        d["schema"] = "gradience.inventory_summary/v99"
        with pytest.raises(QASchemaError, match="Expected schema"):
            InventorySummary.from_dict(d)

    @pytest.mark.parametrize(
        "section",
        [
            "sources",
            "adapter_status_counts",
            "adapter_flag_counts",
            "pair_risk_counts",
            "recommended_strategy_counts",
            "dominant_issue_counts",
        ],
    )
    def test_missing_required_section(self, section: str) -> None:
        d = _valid_summary_dict()
        del d[section]
        with pytest.raises(QASchemaError):
            InventorySummary.from_dict(d)

    def test_non_dict_section(self) -> None:
        d = _valid_summary_dict()
        d["sources"] = "not_a_dict"
        with pytest.raises(QASchemaError, match="must be a dict"):
            InventorySummary.from_dict(d)

    def test_non_int_value_in_section(self) -> None:
        d = _valid_summary_dict()
        d["adapter_status_counts"] = {"eligible": "two"}
        with pytest.raises(QASchemaError, match="must be int"):
            InventorySummary.from_dict(d)

    def test_missing_strict_qa_block_candidates(self) -> None:
        d = _valid_summary_dict()
        del d["strict_qa_block_candidates"]
        with pytest.raises(QASchemaError):
            InventorySummary.from_dict(d)

    def test_non_int_sqbc(self) -> None:
        d = _valid_summary_dict()
        d["strict_qa_block_candidates"] = "zero"
        with pytest.raises(QASchemaError):
            InventorySummary.from_dict(d)

    def test_non_list_notes(self) -> None:
        d = _valid_summary_dict()
        d["notes"] = "not_a_list"
        with pytest.raises(QASchemaError, match="must be a list"):
            InventorySummary.from_dict(d)

    def test_non_string_note_item(self) -> None:
        d = _valid_summary_dict()
        d["notes"] = [123]
        with pytest.raises(QASchemaError, match="must be a str"):
            InventorySummary.from_dict(d)

    def test_evidence_tier_counts_backfill(self) -> None:
        d = _valid_summary_dict()
        del d["evidence_tier_counts"]
        summary = InventorySummary.from_dict(d)
        assert summary.evidence_tier_counts == {}

    def test_extra_keys_ignored(self) -> None:
        d = _valid_summary_dict()
        d["future_field"] = "hello"
        summary = InventorySummary.from_dict(d)
        assert summary.sources["qa_artifact_count"] == 2


# ===================================================================
# Roundtrip serialization
# ===================================================================


class TestRoundtrip:
    """Verify dict and JSON round-trips preserve data."""

    def test_dict_roundtrip(self) -> None:
        original = InventorySummary.from_dict(_valid_summary_dict())
        reloaded = InventorySummary.from_dict(original.to_dict())
        assert reloaded == original

    def test_json_roundtrip(self, tmp_path) -> None:
        original = InventorySummary.from_dict(_valid_summary_dict())
        out = tmp_path / "summary.json"
        original.to_json(out)
        with open(out, encoding="utf-8") as f:
            loaded = json.load(f)
        reloaded = InventorySummary.from_dict(loaded)
        assert reloaded == original

    def test_notes_preserved(self) -> None:
        d = _valid_summary_dict()
        d["notes"] = ["alpha", "beta"]
        original = InventorySummary.from_dict(d)
        reloaded = InventorySummary.from_dict(original.to_dict())
        assert reloaded.notes == ("alpha", "beta")


# ===================================================================
# build_inventory_summary
# ===================================================================


class TestBuildInventorySummary:
    """Unit tests for the counting builder."""

    def test_empty_inputs(self) -> None:
        summary = build_inventory_summary([], [])
        assert summary.sources == {"qa_artifact_count": 0, "merge_report_count": 0}
        assert summary.strict_qa_block_candidates == 0

    def test_status_counting(self) -> None:
        artifacts = [
            StubQA(status=StubEligibility("eligible")),
            StubQA(status=StubEligibility("eligible")),
            StubQA(status=StubEligibility("flagged_weak")),
        ]
        summary = build_inventory_summary(artifacts, [])
        assert summary.adapter_status_counts["eligible"] == 2
        assert summary.adapter_status_counts["flagged_weak"] == 1

    def test_flag_counting(self) -> None:
        artifacts = [
            StubQA(structural_flags=["low_rank", "high_condition"]),
            StubQA(structural_flags=["low_rank"]),
        ]
        summary = build_inventory_summary(artifacts, [])
        assert summary.adapter_flag_counts["low_rank"] == 2
        assert summary.adapter_flag_counts["high_condition"] == 1

    def test_evidence_tier_reported(self) -> None:
        artifacts = [StubQA(eval_available=True, status=StubEligibility("eligible"))]
        summary = build_inventory_summary(artifacts, [])
        assert summary.evidence_tier_counts.get("behavioral_reported") == 1

    def test_evidence_tier_weak(self) -> None:
        artifacts = [StubQA(eval_available=True, status=StubEligibility("flagged_weak"))]
        summary = build_inventory_summary(artifacts, [])
        assert summary.evidence_tier_counts.get("behavioral_weak") == 1

    def test_evidence_tier_missing(self) -> None:
        artifacts = [StubQA(eval_available=False)]
        summary = build_inventory_summary(artifacts, [])
        assert summary.evidence_tier_counts.get("behavioral_missing") == 1

    def test_strict_qa_block_none_eligibility(self) -> None:
        report = StubReport(adapter_a=StubAdapterRef(eligibility_status=None))
        summary = build_inventory_summary([], [report])
        assert summary.strict_qa_block_candidates == 1

    def test_strict_qa_block_both_weak_counts_once(self) -> None:
        report = StubReport(
            adapter_a=StubAdapterRef(eligibility_status="flagged_weak"),
            adapter_b=StubAdapterRef(eligibility_status="flagged_weak"),
        )
        summary = build_inventory_summary([], [report])
        assert summary.strict_qa_block_candidates == 1


# ===================================================================
# build_action_plan
# ===================================================================


class TestBuildActionPlan:
    """Unit tests for the action plan builder."""

    def test_empty_inputs(self) -> None:
        plan = build_action_plan([], [])
        assert plan.total_pairs == 0

    def test_same_task_pairs(self) -> None:
        qa = [StubQA(adapter_name="a"), StubQA(adapter_name="b")]
        report = StubReport(
            adapter_a=StubAdapterRef(path="/adapters/a"),
            adapter_b=StubAdapterRef(path="/adapters/b"),
            task_relationship_advisory=None,
        )
        plan = build_action_plan(qa, [report])
        assert len(plan.same_task_priority) == 1
        assert plan.cross_task_count == 0

    def test_cross_task_pairs(self) -> None:
        qa = [StubQA(adapter_name="a"), StubQA(adapter_name="b")]
        report = StubReport(
            adapter_a=StubAdapterRef(path="/adapters/a"),
            adapter_b=StubAdapterRef(path="/adapters/b"),
            task_relationship_advisory={"note": "different tasks"},
            task_relationship="cross_task",
        )
        plan = build_action_plan(qa, [report])
        assert plan.cross_task_count == 1
        assert len(plan.same_task_priority) == 0

    def test_same_family_routes_to_same_task(self) -> None:
        qa = [StubQA(adapter_name="a"), StubQA(adapter_name="b")]
        report = StubReport(
            adapter_a=StubAdapterRef(path="/adapters/a"),
            adapter_b=StubAdapterRef(path="/adapters/b"),
            task_relationship_advisory={"note": "same family"},
            task_relationship="same_family",
        )
        plan = build_action_plan(qa, [report])
        assert len(plan.same_task_priority) == 1
        assert plan.cross_task_count == 0

    def test_weak_source_excluded(self) -> None:
        qa = [
            StubQA(adapter_name="a", status=StubEligibility("flagged_weak")),
            StubQA(adapter_name="b"),
        ]
        report = StubReport(
            adapter_a=StubAdapterRef(eligibility_status="flagged_weak", path="/adapters/a"),
            adapter_b=StubAdapterRef(path="/adapters/b"),
            task_relationship_advisory=None,
        )
        plan = build_action_plan(qa, [report])
        assert len(plan.exclude) > 0
        assert plan.retained_count == 0

    def test_near_miss_structurally_clean(self) -> None:
        qa = [
            StubQA(adapter_name="a", status=StubEligibility("flagged_weak")),
            StubQA(adapter_name="b"),
        ]
        report = StubReport(
            pair_risk="low",
            adapter_a=StubAdapterRef(eligibility_status="flagged_weak", path="/adapters/a"),
            adapter_b=StubAdapterRef(path="/adapters/b"),
            task_relationship_advisory=None,
        )
        plan = build_action_plan(qa, [report])
        assert len(plan.near_miss_candidates) == 1

    def test_near_miss_severity_marginal(self) -> None:
        qa = [
            StubQA(
                adapter_name="a",
                status=StubEligibility("flagged_weak"),
                adapter_score=0.501,
                base_score=0.500,
                lower_is_better=False,
            ),
            StubQA(adapter_name="b"),
        ]
        report = StubReport(
            pair_risk="low",
            adapter_a=StubAdapterRef(eligibility_status="flagged_weak", path="/adapters/a"),
            adapter_b=StubAdapterRef(path="/adapters/b"),
            task_relationship_advisory=None,
        )
        plan = build_action_plan(qa, [report])
        sev_map = dict(plan.near_miss_severity)
        assert sev_map["a × b"] == "marginal"

    def test_near_miss_severity_moderate(self) -> None:
        # delta = 0.490 - 0.500 = -0.010 -> moderate (boundary)
        qa = [
            StubQA(
                adapter_name="a",
                status=StubEligibility("flagged_weak"),
                adapter_score=0.490,
                base_score=0.500,
                lower_is_better=False,
            ),
            StubQA(adapter_name="b"),
        ]
        report = StubReport(
            pair_risk="low",
            adapter_a=StubAdapterRef(eligibility_status="flagged_weak", path="/adapters/a"),
            adapter_b=StubAdapterRef(path="/adapters/b"),
            task_relationship_advisory=None,
        )
        plan = build_action_plan(qa, [report])
        sev_map = dict(plan.near_miss_severity)
        assert sev_map["a × b"] == "moderate"

    def test_near_miss_severity_substantial(self) -> None:
        # delta = 0.449 - 0.500 = -0.051 -> substantial
        qa = [
            StubQA(
                adapter_name="a",
                status=StubEligibility("flagged_weak"),
                adapter_score=0.449,
                base_score=0.500,
                lower_is_better=False,
            ),
            StubQA(adapter_name="b"),
        ]
        report = StubReport(
            pair_risk="low",
            adapter_a=StubAdapterRef(eligibility_status="flagged_weak", path="/adapters/a"),
            adapter_b=StubAdapterRef(path="/adapters/b"),
            task_relationship_advisory=None,
        )
        plan = build_action_plan(qa, [report])
        sev_map = dict(plan.near_miss_severity)
        assert sev_map["a × b"] == "substantial"

    def test_near_miss_severity_unknown(self) -> None:
        qa = [
            StubQA(adapter_name="a", status=StubEligibility("flagged_weak")),
            StubQA(adapter_name="b"),
        ]
        report = StubReport(
            pair_risk="low",
            adapter_a=StubAdapterRef(eligibility_status="flagged_weak", path="/adapters/a"),
            adapter_b=StubAdapterRef(path="/adapters/b"),
            task_relationship_advisory=None,
        )
        plan = build_action_plan(qa, [report])
        sev_map = dict(plan.near_miss_severity)
        assert sev_map["a × b"] == "unknown"

    def test_evaluate_first_cap(self) -> None:
        qa = [StubQA(adapter_name=f"a{i}") for i in range(10)]
        reports = [
            StubReport(
                adapter_a=StubAdapterRef(path=f"/adapters/a{i}"),
                adapter_b=StubAdapterRef(path=f"/adapters/a{i + 1}"),
                task_relationship_advisory=None,
            )
            for i in range(0, 9)
        ]
        plan = build_action_plan(qa, reports)
        # Same-task priority should have all 9, but evaluate_first capped at 4
        assert len(plan.evaluate_first) == 4
        assert len(plan.same_task_priority) == 9

    def test_summary_line_no_pairs(self) -> None:
        plan = build_action_plan([], [])
        assert "No pair reports available" in plan.summary_line

    def test_summary_line_all_same_task(self) -> None:
        qa = [StubQA(adapter_name="a"), StubQA(adapter_name="b")]
        report = StubReport(
            adapter_a=StubAdapterRef(path="/adapters/a"),
            adapter_b=StubAdapterRef(path="/adapters/b"),
            task_relationship_advisory=None,
        )
        plan = build_action_plan(qa, [report])
        assert "confirmatory" in plan.summary_line

    def test_summary_line_all_cross_task(self) -> None:
        qa = [StubQA(adapter_name="a"), StubQA(adapter_name="b")]
        report = StubReport(
            adapter_a=StubAdapterRef(path="/adapters/a"),
            adapter_b=StubAdapterRef(path="/adapters/b"),
            task_relationship_advisory={"note": "different"},
            task_relationship="cross_task",
        )
        plan = build_action_plan(qa, [report])
        assert "cross-task" in plan.summary_line.lower() or "All pairs are cross-task" in plan.summary_line


# ===================================================================
# derive_inventory_policy_summary
# ===================================================================


class TestDeriveInventoryPolicySummary:
    """Tests for the four-component policy derivation."""

    def test_empty_inventory(self) -> None:
        summary = build_inventory_summary([], [])
        plan = build_action_plan([], [])
        policy = derive_inventory_policy_summary(summary, plan)
        assert policy["inventory_type"] == "empty"

    def test_all_weak(self) -> None:
        artifacts = [
            StubQA(adapter_name="a", status=StubEligibility("flagged_weak")),
            StubQA(adapter_name="b", status=StubEligibility("flagged_weak")),
        ]
        # Need at least one report so n_reports > 0 (else "empty" wins).
        report = StubReport(
            adapter_a=StubAdapterRef(eligibility_status="flagged_weak", path="/adapters/a"),
            adapter_b=StubAdapterRef(eligibility_status="flagged_weak", path="/adapters/b"),
            task_relationship_advisory=None,
        )
        summary = build_inventory_summary(artifacts, [report])
        plan = build_action_plan(artifacts, [report])
        policy = derive_inventory_policy_summary(summary, plan)
        assert policy["inventory_type"] == "weak_evidence"
        assert policy["dominant_driver"] == "source_qa"

    def test_same_task_no_driver(self) -> None:
        qa = [StubQA(adapter_name="a"), StubQA(adapter_name="b")]
        report = StubReport(
            adapter_a=StubAdapterRef(path="/adapters/a"),
            adapter_b=StubAdapterRef(path="/adapters/b"),
            task_relationship_advisory=None,
        )
        summary = build_inventory_summary(qa, [report])
        plan = build_action_plan(qa, [report])
        policy = derive_inventory_policy_summary(summary, plan)
        assert policy["inventory_type"] == "same_task"
        assert policy["dominant_driver"] == "none"
        assert policy["exploration_posture"] == "confirmatory"

    def test_cross_task(self) -> None:
        qa = [StubQA(adapter_name="a"), StubQA(adapter_name="b")]
        report = StubReport(
            adapter_a=StubAdapterRef(path="/adapters/a"),
            adapter_b=StubAdapterRef(path="/adapters/b"),
            task_relationship_advisory={"note": "different"},
            task_relationship="cross_task",
        )
        summary = build_inventory_summary(qa, [report])
        plan = build_action_plan(qa, [report])
        policy = derive_inventory_policy_summary(summary, plan)
        assert policy["inventory_type"] in ("mixed_task", "cross_task")
        assert policy["dominant_driver"] == "task_boundary"

    def test_structural_risk_dominates(self) -> None:
        qa = [StubQA(adapter_name="a"), StubQA(adapter_name="b")]
        # All same-task, but all high-risk
        reports = [
            StubReport(
                pair_risk="high",
                adapter_a=StubAdapterRef(path="/adapters/a"),
                adapter_b=StubAdapterRef(path="/adapters/b"),
                task_relationship_advisory=None,
            ),
            StubReport(
                pair_risk="high",
                adapter_a=StubAdapterRef(path="/adapters/a"),
                adapter_b=StubAdapterRef(path="/adapters/b"),
                task_relationship_advisory=None,
            ),
        ]
        summary = build_inventory_summary(qa, reports)
        plan = build_action_plan(qa, reports)
        policy = derive_inventory_policy_summary(summary, plan)
        assert policy["dominant_driver"] == "structural_risk"


# ===================================================================
# is_large_inventory
# ===================================================================


class TestIsLargeInventory:
    """Threshold tests for large-inventory detection."""

    def test_below_both(self) -> None:
        assert is_large_inventory(7, 19) is False

    def test_exactly_8_adapters(self) -> None:
        assert is_large_inventory(8, 10) is True

    def test_exactly_20_pairs(self) -> None:
        assert is_large_inventory(5, 20) is True


# ===================================================================
# Formatters (smoke tests)
# ===================================================================


class TestFormatters:
    """Smoke tests — verify formatters run without error and contain key headings."""

    def test_format_inventory_summary_runs(self) -> None:
        summary = InventorySummary.from_dict(_valid_summary_dict())
        output = format_inventory_summary(summary)
        assert isinstance(output, str)
        assert "INVENTORY OVERVIEW" in output

    def test_format_action_plan_runs(self) -> None:
        plan = build_action_plan([], [])
        output = format_action_plan(plan)
        assert isinstance(output, str)
        assert "ACTION PLAN" in output
