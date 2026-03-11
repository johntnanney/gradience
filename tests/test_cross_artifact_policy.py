"""Cross-artifact policy regression tests.

Verifies that eligibility status flows correctly from AdapterQAArtifact
through MergeQAReport to InventorySummary, and that strategy/risk
alignment is consistent.
"""

from __future__ import annotations

import json
from pathlib import Path

from gradience.vnext.audit.qa_artifact import AdapterQAArtifact
from gradience.vnext.inventory.summary import build_inventory_summary
from gradience.vnext.merge.qa_report import MergeQAReport

# ---------------------------------------------------------------------------
# Helpers — load canonical examples
# ---------------------------------------------------------------------------


def _load_qa(name: str) -> AdapterQAArtifact:
    """Load a QA artifact from examples/qa/{name}.json."""
    p = Path(f"examples/qa/{name}.json")
    with open(p) as f:
        return AdapterQAArtifact.from_dict(json.load(f))


def _load_report(name: str) -> MergeQAReport:
    """Load a merge report from examples/reports/{name}.json."""
    p = Path(f"examples/reports/{name}.json")
    with open(p) as f:
        return MergeQAReport.from_dict(json.load(f))


# ---------------------------------------------------------------------------
# Eligibility flow tests
# ---------------------------------------------------------------------------


class TestEligibilityFlow:
    """Verify eligibility_status propagates from QA -> merge -> inventory."""

    def test_eligible_adapters_zero_block_candidates(self) -> None:
        """Two eligible adapters produce zero strict-QA block candidates."""
        report = _load_report("safe_merge_report")
        assert report.adapter_a.eligibility_status == "eligible"
        assert report.adapter_b.eligibility_status == "eligible"

        summary = build_inventory_summary([], [report])
        assert summary.strict_qa_block_candidates == 0

    def test_flagged_weak_adapter_counted_as_block_candidate(self) -> None:
        """A report with a flagged_weak adapter is a block candidate."""
        report = _load_report("strict_blocked_report")
        a_elig = report.adapter_a.eligibility_status
        b_elig = report.adapter_b.eligibility_status
        blocked_statuses = {"flagged_weak", "unknown_no_behavioral_eval", None}
        assert a_elig in blocked_statuses or b_elig in blocked_statuses

        summary = build_inventory_summary([], [report])
        assert summary.strict_qa_block_candidates >= 1

    def test_null_eligibility_counted_as_block_candidate(self) -> None:
        """A report with null eligibility_status is a block candidate."""
        report = _load_report("safe_merge_report")
        d = report.to_dict()
        d["adapter_a"]["eligibility_status"] = None
        patched = MergeQAReport.from_dict(d)

        summary = build_inventory_summary([], [patched])
        assert summary.strict_qa_block_candidates == 1

    def test_adapter_status_counts_match_qa_artifacts(self) -> None:
        """Inventory adapter_status_counts matches the statuses from QA artifacts."""
        eligible = _load_qa("eligible_adapter_qa")
        structural = _load_qa("structural_only_qa")

        summary = build_inventory_summary([eligible, structural], [])
        assert summary.adapter_status_counts.get("eligible", 0) >= 1
        assert summary.adapter_status_counts.get("unknown_no_behavioral_eval", 0) >= 1
        total = sum(summary.adapter_status_counts.values())
        assert total == 2


# ---------------------------------------------------------------------------
# Strategy/risk alignment tests
# ---------------------------------------------------------------------------


class TestStrategyAlignment:
    """Verify pair_risk -> recommended_strategy alignment in canonical examples."""

    def test_low_risk_maps_to_linear(self) -> None:
        """Low-risk merge reports should recommend 'linear' strategy."""
        report = _load_report("safe_merge_report")
        assert report.pair_risk == "low"
        assert report.recommended_strategy == "linear"

    def test_high_risk_maps_to_audit_aware(self) -> None:
        """High-risk merge reports should recommend 'audit_aware' strategy."""
        report = _load_report("high_risk_warn_report")
        assert report.pair_risk == "high"
        assert report.recommended_strategy == "audit_aware"

    def test_strategy_counts_match_risk_counts_in_inventory(self) -> None:
        """Each merge report contributes exactly one risk and one strategy count."""
        safe = _load_report("safe_merge_report")
        risky = _load_report("high_risk_warn_report")

        summary = build_inventory_summary([], [safe, risky])

        total_risk = sum(summary.pair_risk_counts.values())
        total_strategy = sum(summary.recommended_strategy_counts.values())
        assert total_risk == 2
        assert total_strategy == 2


# ---------------------------------------------------------------------------
# End-to-end spine test
# ---------------------------------------------------------------------------


class TestFullSpine:
    """Load all canonical examples, build inventory, verify consistency."""

    def test_all_examples_produce_valid_inventory(self) -> None:
        """Loading all example QA + reports produces a valid InventorySummary."""
        qa_dir = Path("examples/qa")
        report_dir = Path("examples/reports")

        qa_artifacts = []
        for p in sorted(qa_dir.glob("*.json")):
            with open(p) as f:
                qa_artifacts.append(AdapterQAArtifact.from_dict(json.load(f)))

        merge_reports = []
        for p in sorted(report_dir.glob("*.json")):
            with open(p) as f:
                merge_reports.append(MergeQAReport.from_dict(json.load(f)))

        summary = build_inventory_summary(qa_artifacts, merge_reports)

        assert summary.sources["qa_artifact_count"] == len(qa_artifacts)
        assert summary.sources["merge_report_count"] == len(merge_reports)
        assert sum(summary.adapter_status_counts.values()) == len(qa_artifacts)
        assert sum(summary.pair_risk_counts.values()) == len(merge_reports)
        assert sum(summary.recommended_strategy_counts.values()) == len(merge_reports)
        assert summary.strict_qa_block_candidates <= len(merge_reports)
