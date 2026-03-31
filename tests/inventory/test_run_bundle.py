"""Tests for gradience.vnext.inventory.run_bundle — preflight run bundle packaging."""

from __future__ import annotations

import json
from pathlib import Path

from gradience.vnext.inventory.run_bundle import (
    build_action_plan_md,
    build_preflight_summary_json,
    derive_drift_summary,
    emit_run_bundle,
)
from gradience.vnext.inventory.summary import InventoryActionPlan
from tests.inventory.conftest import StubQA, StubReport

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_summary(
    *,
    run_id: str = "run_001",
    adapter_count: int = 4,
    pair_count: int = 6,
    retained_candidate_count: int = 3,
    advisory_pair_count: int = 1,
    behavioral_evidence_count: int = 2,
    total_source_count: int = 4,
    excluded_sources: list[str] | None = None,
    inventory_policy_summary: dict[str, str] | None = None,
) -> dict:
    d: dict = {
        "run_id": run_id,
        "adapter_count": adapter_count,
        "pair_count": pair_count,
        "retained_candidate_count": retained_candidate_count,
        "advisory_pair_count": advisory_pair_count,
        "behavioral_evidence_count": behavioral_evidence_count,
        "total_source_count": total_source_count,
        "excluded_sources": excluded_sources or [],
    }
    if inventory_policy_summary is not None:
        d["inventory_policy_summary"] = inventory_policy_summary
    return d


def _minimal_action_plan(**overrides) -> InventoryActionPlan:
    defaults = dict(
        exclude=(),
        same_task_priority=(),
        cross_task_caution=(),
        evaluate_first=(),
        summary_line="Test summary.",
        total_pairs=0,
        retained_count=0,
        cross_task_count=0,
        behavioral_evidence_count=0,
        total_source_count=0,
        retained_pair_detail=(),
        near_miss_candidates=(),
        near_miss_detail=(),
        near_miss_severity=(),
    )
    defaults.update(overrides)
    return InventoryActionPlan(**defaults)


# ---------------------------------------------------------------------------
# TestDeriveDriftSummary
# ---------------------------------------------------------------------------


class TestDeriveDriftSummary:
    def test_narrowing(self) -> None:
        curr = _make_summary(retained_candidate_count=3)
        prev = _make_summary(retained_candidate_count=5)
        drift = derive_drift_summary(curr, prev)
        assert drift["status"] == "narrowing"
        assert drift["implication"] == "current_run_materially_narrower"

    def test_broadening(self) -> None:
        curr = _make_summary(retained_candidate_count=7)
        prev = _make_summary(retained_candidate_count=5)
        drift = derive_drift_summary(curr, prev)
        assert drift["status"] == "broadening"
        assert drift["implication"] == "current_run_materially_broader"

    def test_stable(self) -> None:
        curr = _make_summary(retained_candidate_count=5)
        prev = _make_summary(retained_candidate_count=5)
        drift = derive_drift_summary(curr, prev)
        assert drift["status"] == "stable"

    def test_evidence_improved(self) -> None:
        curr = _make_summary(
            retained_candidate_count=5,
            behavioral_evidence_count=3, total_source_count=4,
        )
        prev = _make_summary(
            retained_candidate_count=5,
            behavioral_evidence_count=1, total_source_count=4,
        )
        drift = derive_drift_summary(curr, prev)
        assert drift["status"] == "stable"
        assert drift["implication"] == "evidence_improved_posture_unchanged"

    def test_evidence_degraded(self) -> None:
        curr = _make_summary(
            retained_candidate_count=5,
            behavioral_evidence_count=1, total_source_count=4,
        )
        prev = _make_summary(
            retained_candidate_count=5,
            behavioral_evidence_count=3, total_source_count=4,
        )
        drift = derive_drift_summary(curr, prev)
        assert drift["status"] == "stable"
        assert drift["implication"] == "evidence_degraded"

    def test_source_composition_changed(self) -> None:
        curr = _make_summary(
            retained_candidate_count=5,
            excluded_sources=["adapter_x"],
            behavioral_evidence_count=2, total_source_count=4,
        )
        prev = _make_summary(
            retained_candidate_count=5,
            excluded_sources=["adapter_y"],
            behavioral_evidence_count=2, total_source_count=4,
        )
        drift = derive_drift_summary(curr, prev)
        assert drift["status"] == "stable"
        assert drift["implication"] == "source_composition_changed"

    def test_narrowing_beats_source_change(self) -> None:
        curr = _make_summary(
            retained_candidate_count=3,
            excluded_sources=["adapter_x"],
        )
        prev = _make_summary(
            retained_candidate_count=5,
            excluded_sources=["adapter_y"],
        )
        drift = derive_drift_summary(curr, prev)
        # Narrowing takes priority over source composition change
        assert drift["implication"] == "current_run_materially_narrower"

    def test_no_substantial_change(self) -> None:
        s = _make_summary(
            retained_candidate_count=5,
            behavioral_evidence_count=2, total_source_count=4,
            excluded_sources=["adapter_x"],
        )
        drift = derive_drift_summary(s, s)
        assert drift["implication"] == "no_substantial_change"

    def test_policy_change_detected(self) -> None:
        policy_a = {"inventory_type": "focused", "dominant_driver": "risk", "exploration_posture": "narrow", "constraint": "strict"}
        policy_b = {"inventory_type": "broad", "dominant_driver": "coverage", "exploration_posture": "wide", "constraint": "relaxed"}
        curr = _make_summary(retained_candidate_count=5, inventory_policy_summary=policy_b)
        prev = _make_summary(retained_candidate_count=5, inventory_policy_summary=policy_a)
        drift = derive_drift_summary(curr, prev)
        assert "policy_change" in drift
        assert drift["policy_change"]["inventory_type_changed"] is True
        assert drift["policy_change"]["dominant_driver_changed"] is True

    def test_no_policy_either(self) -> None:
        curr = _make_summary(retained_candidate_count=5)
        prev = _make_summary(retained_candidate_count=5)
        drift = derive_drift_summary(curr, prev)
        assert "policy_change" not in drift


# ---------------------------------------------------------------------------
# TestBuildPreflightSummaryJson
# ---------------------------------------------------------------------------


class TestBuildPreflightSummaryJson:
    def test_basic_counts(self) -> None:
        qa_list = [
            StubQA(adapter_name="a"),
            StubQA(adapter_name="b"),
        ]
        report = StubReport()
        plan = _minimal_action_plan(
            total_pairs=1,
            retained_count=1,
            evaluate_first=("a+b",),
        )
        result = build_preflight_summary_json(
            inventory_id="inv_001",
            run_id="run_001",
            qa_artifacts=qa_list,
            merge_reports=[report],
            action_plan=plan,
        )
        assert result["adapter_count"] == 2
        assert result["pair_count"] == 1

    def test_zero_reports(self) -> None:
        qa_list = [StubQA(adapter_name="a")]
        plan = _minimal_action_plan(total_pairs=0, retained_count=0)
        result = build_preflight_summary_json(
            inventory_id="inv_001",
            run_id="run_001",
            qa_artifacts=qa_list,
            merge_reports=[],
            action_plan=plan,
        )
        assert result["reduction_ratio"] == 0.0


# ---------------------------------------------------------------------------
# TestBuildActionPlanMd
# ---------------------------------------------------------------------------


class TestBuildActionPlanMd:
    def test_empty_sections(self) -> None:
        plan = _minimal_action_plan()
        md = build_action_plan_md(plan)
        assert "none" in md.lower()

    def test_with_pairs(self) -> None:
        plan = _minimal_action_plan(
            evaluate_first=("adapter_a+adapter_b",),
            retained_pair_detail=(("adapter_a+adapter_b", "low", "linear"),),
        )
        md = build_action_plan_md(plan)
        assert "adapter_a+adapter_b" in md


# ---------------------------------------------------------------------------
# TestEmitRunBundle
# ---------------------------------------------------------------------------


class TestEmitRunBundle:
    def test_all_files_created(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run_001"
        qa_list = [StubQA(adapter_name="a")]
        report = StubReport()
        plan = _minimal_action_plan(total_pairs=1, retained_count=1)

        emit_run_bundle(
            inventory_id="inv_001",
            run_id="run_001",
            run_dir=run_dir,
            qa_artifacts=qa_list,
            merge_reports=[report],
            action_plan=plan,
        )

        expected_files = [
            "preflight_summary.json",
            "run_manifest.json",
            "preflight_summary.md",
            "inventory_action_plan.md",
            "review_packet.md",
            "review_packet.json",
        ]
        for fname in expected_files:
            assert (run_dir / fname).exists(), f"Missing expected file: {fname}"

        # compare_to_previous.md should NOT exist without previous_run_dir
        assert not (run_dir / "compare_to_previous.md").exists()

    def test_compare_to_previous_when_previous_exists(self, tmp_path: Path) -> None:
        # Set up previous run directory with a preflight_summary.json
        prev_dir = tmp_path / "run_prev"
        prev_dir.mkdir()
        prev_summary = {
            "run_id": "run_prev",
            "adapter_count": 2,
            "pair_count": 1,
            "retained_candidate_count": 1,
            "advisory_pair_count": 0,
            "behavioral_evidence_count": 1,
            "total_source_count": 2,
            "excluded_sources": [],
        }
        (prev_dir / "preflight_summary.json").write_text(json.dumps(prev_summary))

        run_dir = tmp_path / "run_001"
        qa_list = [StubQA(adapter_name="a"), StubQA(adapter_name="b")]
        report = StubReport()
        plan = _minimal_action_plan(total_pairs=1, retained_count=1)

        emit_run_bundle(
            inventory_id="inv_001",
            run_id="run_001",
            run_dir=run_dir,
            qa_artifacts=qa_list,
            merge_reports=[report],
            action_plan=plan,
            previous_run_dir=prev_dir,
        )

        assert (run_dir / "compare_to_previous.md").exists()
