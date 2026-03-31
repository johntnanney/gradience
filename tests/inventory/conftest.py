"""Lightweight stubs and factories for inventory tests."""

from __future__ import annotations

from dataclasses import dataclass, field

from gradience.vnext.inventory.summary import SCHEMA_ID

# ---------------------------------------------------------------------------
# Stub dataclasses — minimal duck-typed stand-ins for production objects
# ---------------------------------------------------------------------------


@dataclass
class StubEligibility:
    """Mimics the `.status` attribute of an AdapterQAArtifact."""

    value: str = "eligible"


@dataclass
class StubQA:
    """Minimal stand-in for AdapterQAArtifact.

    Attributes match the duck-typed access patterns used in
    ``build_inventory_summary`` and ``build_action_plan``.
    """

    status: StubEligibility = field(default_factory=StubEligibility)
    structural_flags: list[str] = field(default_factory=list)
    eval_available: bool = False
    adapter_name: str = "test_adapter"
    eval_dataset: str | None = None
    adapter_score: float | None = None
    base_score: float | None = None
    lower_is_better: bool | None = None


@dataclass
class StubAdapterRef:
    """Minimal stand-in for adapter references inside a MergeQAReport."""

    eligibility_status: str | None = "eligible"
    path: str = "/adapters/test_adapter"


@dataclass
class StubReport:
    """Minimal stand-in for MergeQAReport.

    Attributes match the duck-typed access patterns used in
    ``build_inventory_summary`` and ``build_action_plan``.
    """

    pair_risk: str = "low"
    recommended_strategy: str = "linear"
    dominant_issue: str = "none"
    adapter_a: StubAdapterRef = field(default_factory=StubAdapterRef)
    adapter_b: StubAdapterRef = field(default_factory=lambda: StubAdapterRef(path="/adapters/test_adapter_b"))
    task_relationship_advisory: dict | None = None
    task_relationship: str | None = None
    compatibility_score: float = 0.95


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------


def _valid_summary_dict() -> dict:
    """Return a minimal valid ``gradience.inventory_summary/v1`` dict."""
    return {
        "schema": SCHEMA_ID,
        "sources": {"qa_artifact_count": 2, "merge_report_count": 1},
        "adapter_status_counts": {"eligible": 2},
        "adapter_flag_counts": {},
        "pair_risk_counts": {"low": 1},
        "recommended_strategy_counts": {"linear": 1},
        "dominant_issue_counts": {"none": 1},
        "strict_qa_block_candidates": 0,
        "evidence_tier_counts": {"behavioral_reported": 2},
        "notes": ["test fixture"],
    }
