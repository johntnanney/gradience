"""Tests for the static HTML preflight report generator."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from gradience.vnext.inventory.html_report import render_html_report

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def minimal_bundle(tmp_path: Path) -> Path:
    """Create a minimal valid bundle directory."""
    summary = {
        "inventory_id": "test-inv",
        "run_id": "run-001",
        "timestamp": "2026-03-28T12:00:00+00:00",
        "adapter_count": 3,
        "pair_count": 3,
        "excluded_sources": ["bad_adapter"],
        "same_task_priority_pairs": ["a_vs_b"],
        "cross_task_caution_regions": ["QNLI region"],
        "reduced_candidate_subset": ["a_vs_b", "a_vs_c"],
        "retained_candidate_count": 2,
        "reduction_ratio": 0.333,
        "advisory_pair_count": 1,
        "qa_summary": {"eligible": 2, "flagged_weak": 1},
        "behavioral_evidence_count": 2,
        "total_source_count": 3,
        "summary_line": "3 adapters, 3 pairs, 2 retained (33% reduction).",
    }
    (tmp_path / "preflight_summary.json").write_text(json.dumps(summary))

    manifest = {
        "inventory_id": "test-inv",
        "run_id": "run-001",
        "timestamp": "2026-03-28T12:00:00+00:00",
        "adapter_count": 3,
        "pair_count": 3,
    }
    (tmp_path / "run_manifest.json").write_text(json.dumps(manifest))

    return tmp_path


@pytest.fixture()
def bundle_with_qa(minimal_bundle: Path) -> Path:
    """Add QA artifacts to the bundle."""
    qa_dir = minimal_bundle / "qa"
    qa_dir.mkdir()
    qa = {
        "schema": "gradience.adapter_qa/v1",
        "adapter_name": "sst2_lora",
        "adapter_path": "/models/sst2_lora",
        "base_model": "distilbert-base-uncased",
        "rank_nominal": 16,
        "n_layers": 6,
        "utilization_mean": 0.72,
        "utilization_median": 0.71,
        "stable_rank_mean": 8.5,
        "energy_rank_90_p50": None,
        "rank_waste_ratio": 0.28,
        "structural_flags": ["low_utilization"],
        "eval_available": True,
        "eval_dataset": "sst2-val",
        "metric_name": "accuracy",
        "adapter_score": 0.91,
        "base_score": 0.82,
        "lower_is_better": False,
        "beats_base": True,
        "status": "eligible",
        "confidence": "high",
        "reasons": ["Adapter beats base by 9pp on sst2-val"],
        "notes": [],
    }
    (qa_dir / "sst2_lora.json").write_text(json.dumps(qa))
    return minimal_bundle


@pytest.fixture()
def bundle_with_reports(bundle_with_qa: Path) -> Path:
    """Add pair reports to the bundle."""
    reports_dir = bundle_with_qa / "pair_reports"
    reports_dir.mkdir()
    report = {
        "schema": "gradience.merge_qa_report/v1",
        "adapter_a": {
            "path": "/models/sst2_lora",
            "rank": 16,
            "alpha": 16.0,
            "n_layers": 6,
            "base_model": "distilbert-base-uncased",
            "eligibility_status": "eligible",
        },
        "adapter_b": {
            "path": "/models/mrpc_lora",
            "rank": 16,
            "alpha": 16.0,
            "n_layers": 6,
            "base_model": "distilbert-base-uncased",
            "eligibility_status": "eligible",
        },
        "pair_risk": "low",
        "dominant_issue": "none",
        "dominant_issue_detail": "No dominant issue detected.",
        "recommended_action": "Proceed with linear merge.",
        "recommended_strategy": "linear",
        "confidence": "high",
        "confidence_note": "Both adapters have behavioral evidence.",
        "caveats": [],
        "verdict_distribution": {"safe": 5, "redundant": 1},
        "compatibility_score": 0.87,
        "task_relationship_advisory": None,
    }
    (reports_dir / "sst2_vs_mrpc.json").write_text(json.dumps(report))
    return bundle_with_qa


@pytest.fixture()
def bundle_with_policy(bundle_with_reports: Path) -> Path:
    """Add policy summary to the preflight JSON."""
    summary_path = bundle_with_reports / "preflight_summary.json"
    summary = json.loads(summary_path.read_text())
    summary["inventory_policy_summary"] = {
        "inventory_type": "mixed_task",
        "dominant_driver": "task_boundary",
        "exploration_posture": "conservative",
        "constraint": "Behavioral evidence required for all sources.",
    }
    summary_path.write_text(json.dumps(summary))
    return bundle_with_reports


@pytest.fixture()
def bundle_with_regions(bundle_with_policy: Path) -> Path:
    """Add large inventory region summary."""
    summary_path = bundle_with_policy / "preflight_summary.json"
    summary = json.loads(summary_path.read_text())
    summary["large_inventory_region_summary"] = {
        "enabled": True,
        "region_counts": {
            "same_task_safe_regions": 2,
            "cross_task_caution_regions": 1,
            "weak_evidence_regions": 0,
            "priority_evaluation_regions": 1,
        },
        "candidate_space_map": [
            {"region": "SST-2 cluster", "type": "same_task_safe_region", "pair_count": 3, "status": "same_task_safe"},
            {
                "region": "QNLI × SST-2",
                "type": "cross_task_caution_region",
                "pair_count": 2,
                "status": "evaluate_first",
            },
        ],
    }
    summary_path.write_text(json.dumps(summary))
    return bundle_with_policy


@pytest.fixture()
def bundle_with_drift(bundle_with_reports: Path) -> Path:
    """Add drift summary via review_packet.json."""
    review = {
        "schema": "gradience.review_packet/v1",
        "drift_summary": {
            "status": "narrowing",
            "adapter_count_delta": 1,
            "pair_count_delta": 2,
            "retained_candidate_delta": -1,
            "same_task_safe_zone_change": "unchanged",
            "cross_task_caution_zone_change": "shrunk",
            "evidence_profile_change": "improved",
            "implication": "current_run_materially_narrower",
        },
    }
    (bundle_with_reports / "review_packet.json").write_text(json.dumps(review))
    return bundle_with_reports


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRenderBasic:
    """Core rendering tests."""

    def test_minimal_bundle_produces_valid_html(self, minimal_bundle: Path) -> None:
        html = render_html_report(minimal_bundle)
        assert html.startswith("<!DOCTYPE html>")
        assert "</html>" in html
        assert "test-inv" in html
        assert "run-001" in html

    def test_stat_boxes_present(self, minimal_bundle: Path) -> None:
        html = render_html_report(minimal_bundle)
        assert "stat-value" in html
        assert ">3<" in html  # adapter count
        assert ">2<" in html  # retained count

    def test_trust_snapshot_rendered(self, minimal_bundle: Path) -> None:
        html = render_html_report(minimal_bundle)
        assert "Trust Snapshot" in html
        assert "eligible" in html.lower()
        assert "flagged weak" in html.lower()

    def test_trust_snapshot_split_sections(self, minimal_bundle: Path) -> None:
        """Trust snapshot should have separate status and evidence subsections."""
        html = render_html_report(minimal_bundle)
        assert "Eligibility Status" in html
        assert "Evidence Profile" in html
        assert "2</strong> of <strong>3</strong>" in html  # behavioral evidence

    def test_action_plan_sections(self, minimal_bundle: Path) -> None:
        html = render_html_report(minimal_bundle)
        assert "Exclude" in html
        assert "bad_adapter" in html
        assert "Same-task safe zone" in html
        assert "Cross-task caution" in html
        assert "Evaluate first" in html
        assert "33% reduction" in html

    def test_title_override(self, minimal_bundle: Path) -> None:
        html = render_html_report(minimal_bundle, title="Custom Title")
        assert "<title>Custom Title</title>" in html

    def test_no_crash_on_empty_bundle(self, tmp_path: Path) -> None:
        (tmp_path / "preflight_summary.json").write_text("{}")
        html = render_html_report(tmp_path)
        assert "<!DOCTYPE html>" in html

    def test_executive_interpretation(self, minimal_bundle: Path) -> None:
        """Executive interpretation line should appear below stat grid."""
        html = render_html_report(minimal_bundle)
        assert "exec-line" in html
        assert "2 of 3 candidate pairs survived preflight" in html

    def test_jump_nav_present(self, minimal_bundle: Path) -> None:
        """Jump navigation bar should appear."""
        html = render_html_report(minimal_bundle)
        assert "jump-nav" in html
        assert 'href="#sec-trust"' in html
        assert 'href="#sec-action"' in html

    def test_retained_stat_highlighted(self, minimal_bundle: Path) -> None:
        """Retained stat box should be visually emphasized."""
        html = render_html_report(minimal_bundle)
        assert "highlight" in html
        assert "2 of 3 retained" in html  # subtitle


class TestSectionRendering:
    """Tests for individual sections."""

    def test_qa_artifacts_rendered(self, bundle_with_qa: Path) -> None:
        html = render_html_report(bundle_with_qa)
        assert "Source Adapter Details" in html
        assert "sst2_lora" in html
        assert "low utilization" in html  # humanized flag

    def test_pair_reports_rendered(self, bundle_with_reports: Path) -> None:
        html = render_html_report(bundle_with_reports)
        assert "Pair Summary" in html
        assert "sst2_lora" in html
        assert "mrpc_lora" in html
        assert "linear" in html.lower()

    def test_policy_summary_rendered(self, bundle_with_policy: Path) -> None:
        html = render_html_report(bundle_with_policy)
        assert "Inventory Policy Summary" in html
        # Human-facing labels instead of raw enum values
        assert "Mixed-task inventory" in html
        assert "Task-boundary risk" in html
        assert "Conservative" in html

    def test_policy_raw_enums_absent(self, bundle_with_policy: Path) -> None:
        """Internal enum values should not appear in the rendered HTML."""
        html = render_html_report(bundle_with_policy)
        # These raw values should be replaced by human-facing labels
        assert ">mixed_task<" not in html
        assert ">task_boundary<" not in html

    def test_region_summary_rendered(self, bundle_with_regions: Path) -> None:
        html = render_html_report(bundle_with_regions)
        assert "Region Summary" in html
        assert "SST-2 cluster" in html
        assert "Candidate-Space Map" in html

    def test_region_summary_promoted_when_enabled(self, bundle_with_regions: Path) -> None:
        """Region summary should appear before action plan in large inventories."""
        html = render_html_report(bundle_with_regions)
        # Check body section order (skip nav bar which lists all anchors)
        region_pos = html.index('id="sec-regions"')
        action_pos = html.index('id="sec-action"')
        assert region_pos < action_pos

    def test_drift_summary_rendered(self, bundle_with_drift: Path) -> None:
        html = render_html_report(bundle_with_drift)
        assert "History / Drift Summary" in html
        assert "narrowing" in html

    def test_drift_implication_concrete(self, bundle_with_drift: Path) -> None:
        """Drift implication should be a concrete sentence, not an enum key."""
        html = render_html_report(bundle_with_drift)
        assert "current_run_materially_narrower" not in html
        assert "fewer plausible merges" in html


class TestEdgeCases:
    """Edge case handling."""

    def test_missing_optional_files(self, minimal_bundle: Path) -> None:
        """No QA dir, no pair reports, no review packet — should not crash."""
        html = render_html_report(minimal_bundle)
        assert "<!DOCTYPE html>" in html
        # No pair reports section when no reports exist
        assert "Pair Summary" not in html

    def test_cross_task_advisory_marker(self, bundle_with_reports: Path) -> None:
        """Advisory marker appears for cross-task pairs."""
        # Modify report to have advisory
        report_path = bundle_with_reports / "pair_reports" / "sst2_vs_mrpc.json"
        report = json.loads(report_path.read_text())
        report["task_relationship_advisory"] = "Different eval datasets detected."
        report_path.write_text(json.dumps(report))

        html = render_html_report(bundle_with_reports)
        assert "&#9888;" in html  # warning sign

    def test_html_escaping(self, minimal_bundle: Path) -> None:
        """Ensure user-provided strings are escaped."""
        summary_path = minimal_bundle / "preflight_summary.json"
        summary = json.loads(summary_path.read_text())
        summary["inventory_id"] = '<script>alert("xss")</script>'
        summary_path.write_text(json.dumps(summary))

        html = render_html_report(minimal_bundle)
        assert "<script>" not in html
        assert "&lt;script&gt;" in html

    def test_pair_grouping_with_mixed_risks(self, bundle_with_reports: Path) -> None:
        """Pairs should be grouped by risk/type when multiple groups present."""
        # Add a high-risk pair
        reports_dir = bundle_with_reports / "pair_reports"
        high_risk = {
            "adapter_a": {"path": "/models/a_lora"},
            "adapter_b": {"path": "/models/b_lora"},
            "pair_risk": "high",
            "dominant_issue": "subspace_conflict",
            "recommended_strategy": "audit_aware",
            "confidence": "low",
            "task_relationship_advisory": "Different tasks.",
        }
        (reports_dir / "a_vs_b.json").write_text(json.dumps(high_risk))

        html = render_html_report(bundle_with_reports)
        assert "group-header" in html
        assert "High-risk pairs" in html

    def test_footer_artifact_links(self, minimal_bundle: Path) -> None:
        """Footer should list discovered bundle artifacts."""
        html = render_html_report(minimal_bundle)
        # Minimal bundle has preflight_summary.json so at least that link exists
        assert "Preflight JSON" in html

        # Add a review packet — it should also appear
        review = {"schema": "gradience.review_packet/v1", "drift_summary": {"status": "stable"}}
        (minimal_bundle / "review_packet.json").write_text(json.dumps(review))
        html = render_html_report(minimal_bundle)
        assert "Review Packet" in html

    def test_dominant_issue_humanized(self, bundle_with_reports: Path) -> None:
        """Dominant issue should show human-facing label."""
        report_path = bundle_with_reports / "pair_reports" / "sst2_vs_mrpc.json"
        report = json.loads(report_path.read_text())
        report["dominant_issue"] = "norm_imbalance"
        report_path.write_text(json.dumps(report))

        html = render_html_report(bundle_with_reports)
        assert "Norm imbalance" in html
        assert "norm_imbalance" not in html


class TestCLIIntegration:
    """Test the CLI command wiring."""

    def test_cli_help(self) -> None:
        """preflight-report command should appear in help."""
        import subprocess  # noqa: F811

        result = subprocess.run(
            [sys.executable, "-m", "gradience.cli", "preflight-report", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "bundle-dir" in result.stdout

    def test_cli_generates_file(self, bundle_with_reports: Path, tmp_path: Path) -> None:
        """CLI should generate an HTML file."""
        import subprocess
        import sys

        output = tmp_path / "out.html"
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "gradience.cli",
                "preflight-report",
                "--bundle-dir",
                str(bundle_with_reports),
                "-o",
                str(output),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert output.exists()
        html = output.read_text()
        assert "<!DOCTYPE html>" in html
