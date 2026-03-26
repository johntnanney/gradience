"""Tests for rule-based neighborhood suggestions."""

from __future__ import annotations

from gradience.vnext.audit.qa_artifact import AdapterQAArtifact
from gradience.vnext.inventory.merge_neighborhood_report import MergeNeighborhoodReport
from gradience.vnext.inventory.neighborhoods import (
    build_compatibility_matrix,
    suggest_merge_neighborhoods,
)
from gradience.vnext.merge.eligibility import EligibilityStatus
from gradience.vnext.merge.qa_report import MergeQAReport


def _qa(adapter_path: str, status: str) -> AdapterQAArtifact:
    return AdapterQAArtifact(
        adapter_name=adapter_path.split("/")[-1],
        adapter_path=adapter_path,
        base_model="llama",
        rank_nominal=8,
        n_layers=4,
        utilization_mean=0.5,
        utilization_median=0.5,
        stable_rank_mean=3.0,
        energy_rank_90_p50=3.0,
        rank_waste_ratio=0.5,
        structural_flags=[],
        eval_available=status != "unknown_no_behavioral_eval",
        status=EligibilityStatus(status),
    )


def _report(
    adapter_a: str,
    adapter_b: str,
    *,
    pair_risk: str,
    strategy: str,
    score: float,
    dominant_issue: str = "none",
    eligibility_a: str | None = "eligible",
    eligibility_b: str | None = "eligible",
) -> MergeQAReport:
    return MergeQAReport.from_dict(
        {
            "schema": "gradience.merge_qa_report/v1",
            "adapter_a": {"path": adapter_a, "rank": 8, "eligibility_status": eligibility_a},
            "adapter_b": {"path": adapter_b, "rank": 8, "eligibility_status": eligibility_b},
            "pair_risk": pair_risk,
            "dominant_issue": dominant_issue,
            "recommended_strategy": strategy,
            "confidence": "medium",
            "compatibility_score": score,
        }
    )


class TestCompatibilityMatrix:
    def test_bucket_assignment(self) -> None:
        reports = [
            _report("a", "b", pair_risk="low", strategy="linear", score=0.9),
            _report("a", "c", pair_risk="medium", strategy="norm_equalized", score=0.6),
            _report("b", "c", pair_risk="high", strategy="audit_aware", score=0.4),
        ]
        edges = build_compatibility_matrix(reports)
        buckets = {tuple(sorted((e.adapter_a, e.adapter_b))): e.compatibility_bucket for e in edges}
        assert buckets[("a", "b")] == "high_compatibility"
        assert buckets[("a", "c")] == "moderate_compatibility"
        assert buckets[("b", "c")] == "conditional_compatibility"

    def test_min_compatibility_threshold(self) -> None:
        reports = [_report("a", "b", pair_risk="low", strategy="linear", score=0.2)]
        edge = build_compatibility_matrix(reports, min_compatibility=0.5)[0]
        assert edge.compatibility_bucket == "incompatible"


class TestNeighborhoodSuggestions:
    def test_excluded_weak_adapter_case(self) -> None:
        qa = [_qa("a", "eligible"), _qa("b", "flagged_weak"), _qa("c", "eligible")]
        reports = [
            _report("a", "c", pair_risk="low", strategy="linear", score=0.9),
            _report("a", "b", pair_risk="low", strategy="linear", score=0.9, eligibility_b="flagged_weak"),
        ]
        result = suggest_merge_neighborhoods(qa, reports)

        excluded = {e.adapter: e.reason for e in result.excluded}
        assert excluded["b"] == "flagged_weak"
        all_members = {m for g in result.groups for m in g.members}
        assert "b" not in all_members

    def test_all_safe_inventory_case(self) -> None:
        qa = [_qa("a", "eligible"), _qa("b", "eligible"), _qa("c", "eligible")]
        reports = [
            _report("a", "b", pair_risk="low", strategy="linear", score=0.9),
            _report("a", "c", pair_risk="low", strategy="linear", score=0.92),
            _report("b", "c", pair_risk="low", strategy="linear", score=0.91),
        ]
        result = suggest_merge_neighborhoods(qa, reports)
        assert len(result.groups) == 1
        assert result.groups[0].characterization == "likely-safe neighborhood"
        assert len(result.excluded) == 0
        assert len(result.boundary_warnings) == 0

    def test_fragmented_inventory_case(self) -> None:
        qa = [_qa("a", "eligible"), _qa("b", "eligible"), _qa("c", "eligible"), _qa("d", "eligible")]
        reports = [
            _report("a", "b", pair_risk="low", strategy="linear", score=0.9),
            _report("c", "d", pair_risk="low", strategy="linear", score=0.88),
            _report("a", "c", pair_risk="high", strategy="audit_aware", score=0.35, dominant_issue="subspace_conflict"),
            _report("b", "d", pair_risk="high", strategy="audit_aware", score=0.33, dominant_issue="norm_imbalance"),
        ]
        result = suggest_merge_neighborhoods(qa, reports)
        assert len(result.groups) >= 2
        assert len(result.boundary_warnings) >= 1

    def test_mixed_risk_inventory_case(self) -> None:
        qa = [_qa("a", "eligible"), _qa("b", "eligible"), _qa("c", "eligible")]
        reports = [
            _report("a", "b", pair_risk="low", strategy="linear", score=0.9),
            _report("b", "c", pair_risk="medium", strategy="norm_equalized", score=0.63, dominant_issue="norm_imbalance"),
            _report("a", "c", pair_risk="high", strategy="audit_aware", score=0.34, dominant_issue="subspace_conflict"),
        ]
        result = suggest_merge_neighborhoods(qa, reports)
        assert len(result.groups) >= 2
        assert any(g.characterization in {"caution neighborhood", "audit-aware neighborhood"} for g in result.groups)

    def test_strict_qa_exclusion_case(self) -> None:
        qa = [_qa("a", "eligible"), _qa("b", "unknown_no_behavioral_eval")]
        reports = [
            _report(
                "a",
                "b",
                pair_risk="low",
                strategy="linear",
                score=0.8,
                eligibility_b="unknown_no_behavioral_eval",
            )
        ]
        result = suggest_merge_neighborhoods(qa, reports, strict_qa=True)
        excluded = {e.adapter: e.reason for e in result.excluded}
        assert excluded["b"] == "unknown_no_behavioral_eval"

    def test_missing_qa_artifact_handling_under_strict_mode(self) -> None:
        # No QA artifacts provided; merge report still references adapters.
        reports = [
            _report(
                "a",
                "b",
                pair_risk="medium",
                strategy="norm_equalized",
                score=0.6,
                eligibility_a=None,
                eligibility_b="eligible",
            )
        ]
        result = suggest_merge_neighborhoods([], reports, strict_qa=True)
        excluded = {e.adapter: e.reason for e in result.excluded}
        assert excluded["a"] == "missing_qa_artifact"

    def test_cross_artifact_integration_and_reload(self) -> None:
        qa = [_qa("a", "eligible"), _qa("b", "eligible"), _qa("c", "flagged_weak")]
        reports = [
            _report("a", "b", pair_risk="low", strategy="linear", score=0.91),
            _report("a", "c", pair_risk="high", strategy="audit_aware", score=0.3, eligibility_b="flagged_weak"),
        ]
        result = suggest_merge_neighborhoods(qa, reports)
        assert isinstance(result, MergeNeighborhoodReport)
        reloaded = MergeNeighborhoodReport.from_dict(result.to_dict())
        assert reloaded == result
