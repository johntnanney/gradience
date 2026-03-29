"""Tests for the preflight run bundle emitter."""

from __future__ import annotations

import json

from gradience.vnext.audit.qa_artifact import AdapterQAArtifact
from gradience.vnext.inventory.run_bundle import (
    build_action_plan_md,
    build_comparison_md,
    build_preflight_summary_json,
    build_preflight_summary_md,
    build_review_packet_json,
    build_review_packet_md,
    build_run_manifest,
    emit_run_bundle,
    update_latest_pointer,
)
from gradience.vnext.inventory.summary import build_action_plan
from gradience.vnext.merge.eligibility import EligibilityStatus
from gradience.vnext.merge.qa_report import MergeQAReport

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _make_qa(name: str, eval_dataset: str = "qnli_dev", status: str = "eligible") -> AdapterQAArtifact:
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


def _make_report(
    name_a: str,
    name_b: str,
    pair_risk: str = "low",
    strategy: str = "linear",
    advisory: str | None = None,
) -> MergeQAReport:
    d: dict = {
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


def _same_task_fixtures():
    """Same-task control: 2 adapters, 1 pair, no advisory."""
    qas = [_make_qa("sst2_a", "sst2_dev"), _make_qa("sst2_b", "sst2_dev")]
    reports = [_make_report("sst2_a", "sst2_b")]
    plan = build_action_plan(qas, reports)
    return qas, reports, plan


def _mixed_task_fixtures():
    """Mixed-task: 3 adapters, 3 pairs (1 same-task, 2 cross-task)."""
    qas = [
        _make_qa("sst2_a", "sst2_dev"),
        _make_qa("sst2_b", "sst2_dev"),
        _make_qa("qnli_a", "qnli_dev"),
    ]
    reports = [
        _make_report("sst2_a", "sst2_b", pair_risk="low", strategy="linear"),
        _make_report("sst2_a", "qnli_a", pair_risk="high", strategy="audit_aware", advisory="Cross-task"),
        _make_report("sst2_b", "qnli_a", pair_risk="high", strategy="audit_aware", advisory="Cross-task"),
    ]
    plan = build_action_plan(qas, reports)
    return qas, reports, plan


# ---------------------------------------------------------------------------
# Tests: build_preflight_summary_json
# ---------------------------------------------------------------------------


class TestPreflightSummaryJson:
    def test_keys_present(self):
        qas, reports, plan = _same_task_fixtures()
        result = build_preflight_summary_json(
            inventory_id="inv",
            run_id="r1",
            qa_artifacts=qas,
            merge_reports=reports,
            action_plan=plan,
        )
        expected_keys = {
            "inventory_id",
            "run_id",
            "timestamp",
            "adapter_count",
            "pair_count",
            "excluded_sources",
            "same_task_priority_pairs",
            "cross_task_caution_regions",
            "reduced_candidate_subset",
            "retained_candidate_count",
            "reduction_ratio",
            "advisory_pair_count",
            "qa_summary",
            "summary_line",
            "behavioral_evidence_count",
            "total_source_count",
            "near_miss_candidates",
        }
        assert set(result.keys()) == expected_keys

    def test_adapter_count(self):
        qas, reports, plan = _mixed_task_fixtures()
        result = build_preflight_summary_json(
            inventory_id="inv",
            run_id="r1",
            qa_artifacts=qas,
            merge_reports=reports,
            action_plan=plan,
        )
        assert result["adapter_count"] == 3
        assert result["pair_count"] == 3
        assert result["advisory_pair_count"] == 2

    def test_json_serializable(self):
        qas, reports, plan = _mixed_task_fixtures()
        result = build_preflight_summary_json(
            inventory_id="inv",
            run_id="r1",
            qa_artifacts=qas,
            merge_reports=reports,
            action_plan=plan,
        )
        # Should not raise
        json.dumps(result)


# ---------------------------------------------------------------------------
# Tests: build_run_manifest
# ---------------------------------------------------------------------------


class TestRunManifest:
    def test_manifest_fields(self, tmp_path):
        manifest = build_run_manifest(
            inventory_id="inv",
            run_id="r1",
            run_dir=tmp_path,
            adapter_count=4,
            pair_count=6,
            advisory_pair_count=2,
            base_model="distilbert",
        )
        assert manifest["inventory_id"] == "inv"
        assert manifest["adapter_count"] == 4
        assert manifest["base_model"] == "distilbert"
        assert manifest["previous_run_id"] is None

    def test_previous_run_id(self, tmp_path):
        manifest = build_run_manifest(
            inventory_id="inv",
            run_id="r2",
            run_dir=tmp_path,
            adapter_count=4,
            pair_count=6,
            advisory_pair_count=2,
            previous_run_id="r1",
        )
        assert manifest["previous_run_id"] == "r1"


# ---------------------------------------------------------------------------
# Tests: build_preflight_summary_md
# ---------------------------------------------------------------------------


class TestPreflightSummaryMd:
    def test_contains_sections(self):
        qas, reports, plan = _mixed_task_fixtures()
        md = build_preflight_summary_md(
            inventory_id="test_inv",
            run_id="r1",
            qa_artifacts=qas,
            merge_reports=reports,
            action_plan=plan,
        )
        assert "# Preflight Summary" in md
        assert "## Source QA" in md
        assert "## Task-boundary partition" in md
        assert "## Reduced candidate set" in md

    def test_previous_run_reference(self):
        qas, reports, plan = _same_task_fixtures()
        md = build_preflight_summary_md(
            inventory_id="inv",
            run_id="r2",
            qa_artifacts=qas,
            merge_reports=reports,
            action_plan=plan,
            previous_run_id="r1",
        )
        assert "compare_to_previous.md" in md
        assert "r1" in md


# ---------------------------------------------------------------------------
# Tests: build_action_plan_md
# ---------------------------------------------------------------------------


class TestActionPlanMd:
    def test_markdown_sections(self):
        _, _, plan = _mixed_task_fixtures()
        md = build_action_plan_md(plan)
        assert "# Inventory Action Plan" in md
        assert "## Exclude / deprioritize" in md
        assert "## Same-task safe zone" in md
        assert "## Cross-task caution" in md
        assert "## Evaluate first" in md
        assert "## Provenance" in md
        assert "## Summary" in md

    def test_per_pair_detail_in_md(self):
        _, _, plan = _mixed_task_fixtures()
        md = build_action_plan_md(plan)
        # Same-task pair should show risk and strategy
        assert "low risk" in md
        assert "linear" in md


# ---------------------------------------------------------------------------
# Tests: build_comparison_md
# ---------------------------------------------------------------------------


class TestComparisonMd:
    def test_no_change(self):
        summary = {
            "run_id": "r1",
            "adapter_count": 3,
            "pair_count": 6,
            "excluded_sources": [],
            "retained_candidate_count": 2,
            "advisory_pair_count": 4,
            "reduced_candidate_subset": ["a × b", "c × d"],
        }
        md = build_comparison_md(current_summary=summary, previous_summary=summary)
        assert "unchanged" in md.lower()
        assert "No substantial preflight change" in md

    def test_narrower_run(self):
        prev = {
            "run_id": "r1",
            "adapter_count": 3,
            "pair_count": 6,
            "excluded_sources": [],
            "retained_candidate_count": 4,
            "advisory_pair_count": 2,
            "reduced_candidate_subset": ["a × b"],
        }
        curr = {
            "run_id": "r2",
            "adapter_count": 3,
            "pair_count": 6,
            "excluded_sources": ["weak_c"],
            "retained_candidate_count": 2,
            "advisory_pair_count": 4,
            "reduced_candidate_subset": ["a × b"],
        }
        md = build_comparison_md(current_summary=curr, previous_summary=prev)
        assert "narrower" in md.lower()

    def test_broader_run(self):
        prev = {
            "run_id": "r1",
            "adapter_count": 2,
            "pair_count": 3,
            "excluded_sources": [],
            "retained_candidate_count": 1,
            "advisory_pair_count": 2,
            "reduced_candidate_subset": ["a × b"],
        }
        curr = {
            "run_id": "r2",
            "adapter_count": 4,
            "pair_count": 6,
            "excluded_sources": [],
            "retained_candidate_count": 3,
            "advisory_pair_count": 3,
            "reduced_candidate_subset": ["a × b", "c × d"],
        }
        md = build_comparison_md(current_summary=curr, previous_summary=prev)
        assert "broader" in md.lower()


# ---------------------------------------------------------------------------
# Tests: emit_run_bundle
# ---------------------------------------------------------------------------


class TestEmitRunBundle:
    def test_creates_all_files(self, tmp_path):
        qas, reports, plan = _same_task_fixtures()
        run_dir = tmp_path / "run_001"
        emit_run_bundle(
            inventory_id="test",
            run_id="run_001",
            run_dir=run_dir,
            qa_artifacts=qas,
            merge_reports=reports,
            action_plan=plan,
        )
        assert (run_dir / "preflight_summary.md").exists()
        assert (run_dir / "preflight_summary.json").exists()
        assert (run_dir / "run_manifest.json").exists()
        assert (run_dir / "inventory_action_plan.md").exists()

    def test_no_comparison_without_previous(self, tmp_path):
        qas, reports, plan = _same_task_fixtures()
        run_dir = tmp_path / "run_001"
        emit_run_bundle(
            inventory_id="test",
            run_id="run_001",
            run_dir=run_dir,
            qa_artifacts=qas,
            merge_reports=reports,
            action_plan=plan,
        )
        assert not (run_dir / "compare_to_previous.md").exists()

    def test_comparison_with_previous(self, tmp_path):
        qas, reports, plan = _same_task_fixtures()

        # First run
        run1 = tmp_path / "run_001"
        emit_run_bundle(
            inventory_id="test",
            run_id="run_001",
            run_dir=run1,
            qa_artifacts=qas,
            merge_reports=reports,
            action_plan=plan,
        )

        # Second run referencing first
        run2 = tmp_path / "run_002"
        emit_run_bundle(
            inventory_id="test",
            run_id="run_002",
            run_dir=run2,
            qa_artifacts=qas,
            merge_reports=reports,
            action_plan=plan,
            previous_run_dir=run1,
        )
        assert (run2 / "compare_to_previous.md").exists()
        comparison = (run2 / "compare_to_previous.md").read_text()
        assert "run_001" in comparison

    def test_json_outputs_parseable(self, tmp_path):
        qas, reports, plan = _mixed_task_fixtures()
        run_dir = tmp_path / "run_001"
        emit_run_bundle(
            inventory_id="test",
            run_id="run_001",
            run_dir=run_dir,
            qa_artifacts=qas,
            merge_reports=reports,
            action_plan=plan,
        )
        summary = json.loads((run_dir / "preflight_summary.json").read_text())
        manifest = json.loads((run_dir / "run_manifest.json").read_text())
        assert summary["adapter_count"] == 3
        assert manifest["run_id"] == "run_001"

    def test_returns_run_dir(self, tmp_path):
        qas, reports, plan = _same_task_fixtures()
        run_dir = tmp_path / "run_001"
        result = emit_run_bundle(
            inventory_id="test",
            run_id="run_001",
            run_dir=run_dir,
            qa_artifacts=qas,
            merge_reports=reports,
            action_plan=plan,
        )
        assert result == run_dir


# ---------------------------------------------------------------------------
# Tests: update_latest_pointer
# ---------------------------------------------------------------------------


class TestLatestPointer:
    def test_creates_symlink(self, tmp_path):
        run_dir = tmp_path / "run_001"
        run_dir.mkdir()
        update_latest_pointer(tmp_path, run_dir)
        latest = tmp_path / "latest"
        assert latest.is_symlink()
        assert latest.resolve() == run_dir.resolve()

    def test_updates_existing_symlink(self, tmp_path):
        run1 = tmp_path / "run_001"
        run1.mkdir()
        run2 = tmp_path / "run_002"
        run2.mkdir()

        update_latest_pointer(tmp_path, run1)
        update_latest_pointer(tmp_path, run2)

        latest = tmp_path / "latest"
        assert latest.resolve() == run2.resolve()


# ---------------------------------------------------------------------------
# Tests: Project C — Provenance / Trust Language
# ---------------------------------------------------------------------------


class TestProvenanceInJson:
    """Preflight summary JSON should include evidence counts."""

    def test_evidence_counts_present(self):
        qas, reports, plan = _mixed_task_fixtures()
        result = build_preflight_summary_json(
            inventory_id="inv",
            run_id="r1",
            qa_artifacts=qas,
            merge_reports=reports,
            action_plan=plan,
        )
        assert "behavioral_evidence_count" in result
        assert "total_source_count" in result
        assert result["behavioral_evidence_count"] == 3
        assert result["total_source_count"] == 3


class TestProvenanceInSummaryMd:
    """Preflight summary MD should include provenance section."""

    def test_provenance_section_present(self):
        qas, reports, plan = _mixed_task_fixtures()
        md = build_preflight_summary_md(
            inventory_id="inv",
            run_id="r1",
            qa_artifacts=qas,
            merge_reports=reports,
            action_plan=plan,
        )
        assert "## Provenance" in md
        assert "user-reported" in md

    def test_pair_detail_in_reduced_set(self):
        qas, reports, plan = _mixed_task_fixtures()
        md = build_preflight_summary_md(
            inventory_id="inv",
            run_id="r1",
            qa_artifacts=qas,
            merge_reports=reports,
            action_plan=plan,
        )
        # Same-task pair should show risk/strategy
        assert "low risk" in md or "medium risk" in md


class TestProvenanceInComparison:
    """Comparison MD should track evidence changes."""

    def test_provenance_changes_section(self):
        s1 = {
            "run_id": "r1",
            "adapter_count": 3,
            "pair_count": 6,
            "excluded_sources": [],
            "retained_candidate_count": 2,
            "advisory_pair_count": 4,
            "reduced_candidate_subset": ["a × b"],
            "behavioral_evidence_count": 2,
            "total_source_count": 3,
        }
        s2 = {
            "run_id": "r2",
            "adapter_count": 4,
            "pair_count": 10,
            "excluded_sources": [],
            "retained_candidate_count": 3,
            "advisory_pair_count": 7,
            "reduced_candidate_subset": ["a × b", "c × d"],
            "behavioral_evidence_count": 3,
            "total_source_count": 4,
        }
        md = build_comparison_md(current_summary=s2, previous_summary=s1)
        assert "## Provenance changes" in md
        assert "2/3" in md
        assert "3/4" in md

    def test_provenance_unchanged(self):
        s1 = {
            "run_id": "r1",
            "adapter_count": 3,
            "pair_count": 3,
            "excluded_sources": [],
            "retained_candidate_count": 2,
            "advisory_pair_count": 1,
            "reduced_candidate_subset": [],
            "behavioral_evidence_count": 3,
            "total_source_count": 3,
        }
        md = build_comparison_md(current_summary=s1, previous_summary=s1)
        assert "unchanged" in md.lower()


class TestNoVerifiedLanguage:
    """Ensure 'verified' does not appear in non-verification contexts."""

    def test_action_plan_md_no_verified(self):
        _, _, plan = _mixed_task_fixtures()
        md = build_action_plan_md(plan)
        assert "verified" not in md.lower()

    def test_preflight_md_no_verified(self):
        qas, reports, plan = _mixed_task_fixtures()
        md = build_preflight_summary_md(
            inventory_id="inv",
            run_id="r1",
            qa_artifacts=qas,
            merge_reports=reports,
            action_plan=plan,
        )
        assert "verified" not in md.lower()


# ---------------------------------------------------------------------------
# Tests: Review Packet
# ---------------------------------------------------------------------------


def _summary_json_for_review(*, policy: dict | None = None, drift: dict | None = None) -> dict:
    """Minimal preflight_summary.json dict for review packet tests."""
    d: dict = {
        "run_id": "r1",
        "adapter_count": 3,
        "pair_count": 6,
        "retained_candidate_count": 2,
        "advisory_pair_count": 4,
        "behavioral_evidence_count": 2,
        "total_source_count": 3,
        "excluded_sources": ["weak_c"],
        "qa_summary": {"eligible": 2, "flagged_weak": 1},
    }
    if policy is not None:
        d["inventory_policy_summary"] = policy
    if drift is not None:
        d["inventory_drift_summary"] = drift
    return d


_SAMPLE_POLICY = {
    "inventory_type": "mixed_task",
    "dominant_driver": "task_boundary",
    "exploration_posture": "moderate",
    "constraint": "Task boundary is the binding constraint.",
}

_SAMPLE_DRIFT = {
    "status": "narrowing",
    "adapter_count_delta": 1,
    "pair_count_delta": 3,
    "retained_candidate_delta": -2,
    "same_task_safe_zone_change": "shrunk",
    "cross_task_caution_zone_change": "expanded",
    "evidence_profile_change": "improved",
    "policy_change": {
        "inventory_type_changed": False,
        "dominant_driver_changed": True,
        "exploration_posture_changed": False,
        "constraint_changed": True,
    },
    "implication": "current_run_materially_narrower",
}


class TestReviewPacketMdSections:
    """Review packet markdown contains all expected sections."""

    def test_all_seven_sections_present(self):
        _, _, plan = _mixed_task_fixtures()
        summary_json = _summary_json_for_review(policy=_SAMPLE_POLICY, drift=_SAMPLE_DRIFT)
        prev_summary = _summary_json_for_review()
        prev_summary["run_id"] = "r0"

        md = build_review_packet_md(
            inventory_id="inv",
            run_id="r1",
            summary_json=summary_json,
            action_plan=plan,
            policy_summary=_SAMPLE_POLICY,
            drift_summary=_SAMPLE_DRIFT,
            previous_summary=prev_summary,
        )
        assert "# Review Packet" in md
        assert "## Inventory Policy Summary" in md
        assert "## Source QA / Trust Snapshot" in md
        assert "## Action Plan" in md
        assert "## Drift Summary" in md
        assert "## Compare to Previous" in md
        assert "## Artifacts" in md

    def test_header_contains_run_metadata(self):
        _, _, plan = _same_task_fixtures()
        summary_json = _summary_json_for_review()
        md = build_review_packet_md(
            inventory_id="my_inv",
            run_id="run_42",
            summary_json=summary_json,
            action_plan=plan,
        )
        assert "my_inv" in md
        assert "run_42" in md
        assert "**Adapters:** 3" in md
        assert "**Pairs:** 6" in md

    def test_no_drift_section_without_drift(self):
        _, _, plan = _same_task_fixtures()
        summary_json = _summary_json_for_review()
        md = build_review_packet_md(
            inventory_id="inv",
            run_id="r1",
            summary_json=summary_json,
            action_plan=plan,
        )
        assert "## Drift Summary" not in md

    def test_no_compare_section_without_previous(self):
        _, _, plan = _same_task_fixtures()
        summary_json = _summary_json_for_review()
        md = build_review_packet_md(
            inventory_id="inv",
            run_id="r1",
            summary_json=summary_json,
            action_plan=plan,
        )
        assert "## Compare to Previous" not in md

    def test_policy_fallback_when_absent(self):
        _, _, plan = _same_task_fixtures()
        summary_json = _summary_json_for_review()
        md = build_review_packet_md(
            inventory_id="inv",
            run_id="r1",
            summary_json=summary_json,
            action_plan=plan,
            policy_summary=None,
        )
        assert "Not available for this run" in md

    def test_trust_snapshot_qa_counts(self):
        _, _, plan = _same_task_fixtures()
        summary_json = _summary_json_for_review()
        md = build_review_packet_md(
            inventory_id="inv",
            run_id="r1",
            summary_json=summary_json,
            action_plan=plan,
        )
        assert "eligible: 2" in md
        assert "flagged_weak: 1" in md

    def test_excluded_sources_listed(self):
        _, _, plan = _same_task_fixtures()
        summary_json = _summary_json_for_review()
        md = build_review_packet_md(
            inventory_id="inv",
            run_id="r1",
            summary_json=summary_json,
            action_plan=plan,
        )
        assert "weak_c" in md

    def test_artifact_links_include_comparison_when_previous(self):
        _, _, plan = _same_task_fixtures()
        summary_json = _summary_json_for_review()
        prev_summary = _summary_json_for_review()
        prev_summary["run_id"] = "r0"
        md = build_review_packet_md(
            inventory_id="inv",
            run_id="r1",
            summary_json=summary_json,
            action_plan=plan,
            previous_summary=prev_summary,
        )
        assert "compare_to_previous.md" in md

    def test_section_order(self):
        _, _, plan = _mixed_task_fixtures()
        summary_json = _summary_json_for_review(policy=_SAMPLE_POLICY, drift=_SAMPLE_DRIFT)
        prev_summary = _summary_json_for_review()
        prev_summary["run_id"] = "r0"

        md = build_review_packet_md(
            inventory_id="inv",
            run_id="r1",
            summary_json=summary_json,
            action_plan=plan,
            policy_summary=_SAMPLE_POLICY,
            drift_summary=_SAMPLE_DRIFT,
            previous_summary=prev_summary,
        )
        # Sections must appear in design-doc order
        pos_policy = md.index("## Inventory Policy Summary")
        pos_trust = md.index("## Source QA / Trust Snapshot")
        pos_action = md.index("## Action Plan")
        pos_drift = md.index("## Drift Summary")
        pos_compare = md.index("## Compare to Previous")
        pos_artifacts = md.index("## Artifacts")
        assert pos_policy < pos_trust < pos_action < pos_drift < pos_compare < pos_artifacts


class TestReviewPacketJson:
    """Review packet JSON companion matches schema contract."""

    def test_schema_key(self):
        _, _, plan = _same_task_fixtures()
        summary_json = _summary_json_for_review()
        result = build_review_packet_json(
            inventory_id="inv",
            run_id="r1",
            summary_json=summary_json,
            action_plan=plan,
        )
        assert result["schema"] == "gradience.review_packet/v1"

    def test_required_keys(self):
        _, _, plan = _same_task_fixtures()
        summary_json = _summary_json_for_review()
        result = build_review_packet_json(
            inventory_id="inv",
            run_id="r1",
            summary_json=summary_json,
            action_plan=plan,
        )
        for key in (
            "schema",
            "inventory_id",
            "run_id",
            "timestamp",
            "sections_present",
            "adapter_count",
            "pair_count",
            "retained_candidate_count",
            "trust_snapshot",
            "action_plan_summary",
        ):
            assert key in result, f"Missing key: {key}"

    def test_sections_present_minimal(self):
        _, _, plan = _same_task_fixtures()
        summary_json = _summary_json_for_review()
        result = build_review_packet_json(
            inventory_id="inv",
            run_id="r1",
            summary_json=summary_json,
            action_plan=plan,
        )
        assert "header" in result["sections_present"]
        assert "trust_snapshot" in result["sections_present"]
        assert "action_plan" in result["sections_present"]
        assert "artifact_links" in result["sections_present"]
        # No drift or compare without previous
        assert "drift_summary" not in result["sections_present"]
        assert "compare_to_previous" not in result["sections_present"]

    def test_sections_present_full(self):
        _, _, plan = _same_task_fixtures()
        summary_json = _summary_json_for_review()
        result = build_review_packet_json(
            inventory_id="inv",
            run_id="r1",
            summary_json=summary_json,
            action_plan=plan,
            policy_summary=_SAMPLE_POLICY,
            drift_summary=_SAMPLE_DRIFT,
            previous_run_id="r0",
        )
        assert "policy_summary" in result["sections_present"]
        assert "drift_summary" in result["sections_present"]
        assert "compare_to_previous" in result["sections_present"]

    def test_trust_snapshot_structure(self):
        _, _, plan = _same_task_fixtures()
        summary_json = _summary_json_for_review()
        result = build_review_packet_json(
            inventory_id="inv",
            run_id="r1",
            summary_json=summary_json,
            action_plan=plan,
        )
        ts = result["trust_snapshot"]
        assert ts["behavioral_evidence_count"] == 2
        assert ts["total_source_count"] == 3
        assert ts["excluded_source_count"] == 1

    def test_action_plan_summary_structure(self):
        _, _, plan = _mixed_task_fixtures()
        summary_json = _summary_json_for_review()
        result = build_review_packet_json(
            inventory_id="inv",
            run_id="r1",
            summary_json=summary_json,
            action_plan=plan,
        )
        aps = result["action_plan_summary"]
        assert "total_pairs" in aps
        assert "retained_count" in aps
        assert "summary_line" in aps

    def test_json_serializable(self):
        _, _, plan = _mixed_task_fixtures()
        summary_json = _summary_json_for_review(policy=_SAMPLE_POLICY, drift=_SAMPLE_DRIFT)
        result = build_review_packet_json(
            inventory_id="inv",
            run_id="r1",
            summary_json=summary_json,
            action_plan=plan,
            policy_summary=_SAMPLE_POLICY,
            drift_summary=_SAMPLE_DRIFT,
            previous_run_id="r0",
        )
        # Should not raise
        json.dumps(result)

    def test_null_optional_fields_when_absent(self):
        _, _, plan = _same_task_fixtures()
        summary_json = _summary_json_for_review()
        result = build_review_packet_json(
            inventory_id="inv",
            run_id="r1",
            summary_json=summary_json,
            action_plan=plan,
        )
        assert result["policy_summary"] is None
        assert result["drift_summary"] is None
        assert result["previous_run_id"] is None


class TestReviewPacketInBundle:
    """Review packet files are emitted as part of emit_run_bundle."""

    def test_review_packet_files_created(self, tmp_path):
        qas, reports, plan = _same_task_fixtures()
        run_dir = tmp_path / "run_001"
        emit_run_bundle(
            inventory_id="test",
            run_id="run_001",
            run_dir=run_dir,
            qa_artifacts=qas,
            merge_reports=reports,
            action_plan=plan,
        )
        assert (run_dir / "review_packet.md").exists()
        assert (run_dir / "review_packet.json").exists()

    def test_review_packet_json_parseable(self, tmp_path):
        qas, reports, plan = _mixed_task_fixtures()
        run_dir = tmp_path / "run_001"
        emit_run_bundle(
            inventory_id="test",
            run_id="run_001",
            run_dir=run_dir,
            qa_artifacts=qas,
            merge_reports=reports,
            action_plan=plan,
        )
        pkt = json.loads((run_dir / "review_packet.json").read_text())
        assert pkt["schema"] == "gradience.review_packet/v1"
        assert pkt["inventory_id"] == "test"
        assert pkt["run_id"] == "run_001"

    def test_review_packet_with_previous_run(self, tmp_path):
        qas, reports, plan = _same_task_fixtures()

        run1 = tmp_path / "run_001"
        emit_run_bundle(
            inventory_id="test",
            run_id="run_001",
            run_dir=run1,
            qa_artifacts=qas,
            merge_reports=reports,
            action_plan=plan,
        )

        run2 = tmp_path / "run_002"
        emit_run_bundle(
            inventory_id="test",
            run_id="run_002",
            run_dir=run2,
            qa_artifacts=qas,
            merge_reports=reports,
            action_plan=plan,
            previous_run_dir=run1,
        )

        md = (run2 / "review_packet.md").read_text()
        assert "## Drift Summary" in md
        assert "## Compare to Previous" in md

        pkt = json.loads((run2 / "review_packet.json").read_text())
        assert pkt["previous_run_id"] == "run_001"
        assert "drift_summary" in pkt["sections_present"]

    def test_review_packet_with_policy(self, tmp_path):
        qas, reports, plan = _mixed_task_fixtures()
        run_dir = tmp_path / "run_001"
        emit_run_bundle(
            inventory_id="test",
            run_id="run_001",
            run_dir=run_dir,
            qa_artifacts=qas,
            merge_reports=reports,
            action_plan=plan,
            policy_summary=_SAMPLE_POLICY,
        )
        md = (run_dir / "review_packet.md").read_text()
        assert "mixed_task" in md
        assert "task_boundary" in md

        pkt = json.loads((run_dir / "review_packet.json").read_text())
        assert pkt["policy_summary"] == _SAMPLE_POLICY


class TestReviewPacketGuardrails:
    """Review packet follows the guardrail contracts."""

    def test_no_new_scoring_language(self):
        _, _, plan = _same_task_fixtures()
        summary_json = _summary_json_for_review(policy=_SAMPLE_POLICY, drift=_SAMPLE_DRIFT)
        prev = _summary_json_for_review()
        prev["run_id"] = "r0"
        md = build_review_packet_md(
            inventory_id="inv",
            run_id="r1",
            summary_json=summary_json,
            action_plan=plan,
            policy_summary=_SAMPLE_POLICY,
            drift_summary=_SAMPLE_DRIFT,
            previous_summary=prev,
        )
        # No severity / scoring language (exclude the standing trust disclaimer)
        md_check = md.lower().replace("behavioral scores are user-reported", "")
        for forbidden in ("score", "grade", "severity", "critical", "warning"):
            assert forbidden not in md_check, f"Forbidden term '{forbidden}' found in review packet"

    def test_deterministic(self):
        """Same inputs produce identical structure (ignoring timestamp)."""
        _, _, plan = _same_task_fixtures()
        summary_json = _summary_json_for_review()
        j1 = build_review_packet_json(
            inventory_id="inv",
            run_id="r1",
            summary_json=summary_json,
            action_plan=plan,
        )
        j2 = build_review_packet_json(
            inventory_id="inv",
            run_id="r1",
            summary_json=summary_json,
            action_plan=plan,
        )
        # Same keys and values except timestamp
        assert j1["sections_present"] == j2["sections_present"]
        assert j1["trust_snapshot"] == j2["trust_snapshot"]
        assert j1["action_plan_summary"] == j2["action_plan_summary"]


# ---------------------------------------------------------------------------
# Tests: Region Summary Integration
# ---------------------------------------------------------------------------

_SAMPLE_REGION_SUMMARY = {
    "enabled": True,
    "threshold_trigger": "8_adapters",
    "region_counts": {
        "same_task_safe_regions": 3,
        "cross_task_caution_regions": 2,
        "weak_evidence_regions": 0,
        "priority_evaluation_regions": 2,
    },
    "candidate_space_map": [
        {
            "region": "QNLI",
            "type": "same_task_safe_region",
            "pair_count": 3,
            "status": "evaluate_first",
            "pairs": ["qnli_a × qnli_b", "qnli_a × qnli_c"],
        },
        {
            "region": "SST-2",
            "type": "same_task_safe_region",
            "pair_count": 3,
            "status": "evaluate_first",
            "pairs": ["sst2_a × sst2_b"],
        },
        {
            "region": "QNLI × SST-2",
            "type": "cross_task_caution_region",
            "pair_count": 9,
            "status": "do_not_explore_casually",
            "pairs": [],
        },
    ],
}

_SAMPLE_REGION_DISABLED = {
    "enabled": False,
    "threshold_trigger": None,
    "region_counts": {
        "same_task_safe_regions": 0,
        "cross_task_caution_regions": 0,
        "weak_evidence_regions": 0,
        "priority_evaluation_regions": 0,
    },
    "candidate_space_map": [],
}


class TestRegionInPreflightJson:
    """Region summary flows into preflight_summary.json."""

    def test_region_present_when_provided(self):
        qas, reports, plan = _mixed_task_fixtures()
        result = build_preflight_summary_json(
            inventory_id="inv",
            run_id="r1",
            qa_artifacts=qas,
            merge_reports=reports,
            action_plan=plan,
            region_summary=_SAMPLE_REGION_SUMMARY,
        )
        assert "large_inventory_region_summary" in result
        assert result["large_inventory_region_summary"]["enabled"] is True

    def test_region_absent_when_not_provided(self):
        qas, reports, plan = _same_task_fixtures()
        result = build_preflight_summary_json(
            inventory_id="inv",
            run_id="r1",
            qa_artifacts=qas,
            merge_reports=reports,
            action_plan=plan,
        )
        assert "large_inventory_region_summary" not in result


class TestRegionInPreflightMd:
    """Region summary appears in preflight_summary.md for large inventories."""

    def test_region_section_present(self):
        qas, reports, plan = _mixed_task_fixtures()
        md = build_preflight_summary_md(
            inventory_id="inv",
            run_id="r1",
            qa_artifacts=qas,
            merge_reports=reports,
            action_plan=plan,
            region_summary=_SAMPLE_REGION_SUMMARY,
        )
        assert "## Large Inventory Region Summary" in md
        assert "### Candidate-Space Map" in md

    def test_region_section_absent_when_disabled(self):
        qas, reports, plan = _same_task_fixtures()
        md = build_preflight_summary_md(
            inventory_id="inv",
            run_id="r1",
            qa_artifacts=qas,
            merge_reports=reports,
            action_plan=plan,
            region_summary=_SAMPLE_REGION_DISABLED,
        )
        assert "## Large Inventory Region Summary" not in md

    def test_region_section_absent_when_none(self):
        qas, reports, plan = _same_task_fixtures()
        md = build_preflight_summary_md(
            inventory_id="inv",
            run_id="r1",
            qa_artifacts=qas,
            merge_reports=reports,
            action_plan=plan,
        )
        assert "## Large Inventory Region Summary" not in md


class TestRegionInReviewPacket:
    """Region summary flows into review packet."""

    def test_review_packet_md_has_region_section(self):
        _, _, plan = _mixed_task_fixtures()
        summary_json = _summary_json_for_review()
        md = build_review_packet_md(
            inventory_id="inv",
            run_id="r1",
            summary_json=summary_json,
            action_plan=plan,
            region_summary=_SAMPLE_REGION_SUMMARY,
        )
        assert "## Large Inventory Region Summary" in md
        assert "QNLI" in md
        assert "SST-2" in md

    def test_review_packet_md_no_region_when_disabled(self):
        _, _, plan = _same_task_fixtures()
        summary_json = _summary_json_for_review()
        md = build_review_packet_md(
            inventory_id="inv",
            run_id="r1",
            summary_json=summary_json,
            action_plan=plan,
            region_summary=_SAMPLE_REGION_DISABLED,
        )
        assert "## Large Inventory Region Summary" not in md

    def test_review_packet_json_has_region(self):
        _, _, plan = _mixed_task_fixtures()
        summary_json = _summary_json_for_review()
        result = build_review_packet_json(
            inventory_id="inv",
            run_id="r1",
            summary_json=summary_json,
            action_plan=plan,
            region_summary=_SAMPLE_REGION_SUMMARY,
        )
        assert result["large_inventory_region_summary"]["enabled"] is True
        assert "region_summary" in result["sections_present"]

    def test_review_packet_json_null_region_when_disabled(self):
        _, _, plan = _same_task_fixtures()
        summary_json = _summary_json_for_review()
        result = build_review_packet_json(
            inventory_id="inv",
            run_id="r1",
            summary_json=summary_json,
            action_plan=plan,
            region_summary=_SAMPLE_REGION_DISABLED,
        )
        assert result["large_inventory_region_summary"] is None
        assert "region_summary" not in result["sections_present"]


class TestRegionInBundle:
    """Region summary emitted through emit_run_bundle."""

    def test_bundle_with_region(self, tmp_path):
        qas, reports, plan = _mixed_task_fixtures()
        run_dir = tmp_path / "run_001"
        emit_run_bundle(
            inventory_id="test",
            run_id="run_001",
            run_dir=run_dir,
            qa_artifacts=qas,
            merge_reports=reports,
            action_plan=plan,
            region_summary=_SAMPLE_REGION_SUMMARY,
        )
        # Check preflight JSON
        summary = json.loads((run_dir / "preflight_summary.json").read_text())
        assert "large_inventory_region_summary" in summary

        # Check review packet
        pkt = json.loads((run_dir / "review_packet.json").read_text())
        assert pkt["large_inventory_region_summary"]["enabled"] is True

        # Check markdown
        md = (run_dir / "review_packet.md").read_text()
        assert "## Large Inventory Region Summary" in md
