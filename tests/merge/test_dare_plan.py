"""Tests for DARE plan strategy generation."""

from __future__ import annotations

import pytest

from gradience.vnext.merge.containers import AdapterMetadata, AggregateResult, MatchingSummary
from gradience.vnext.merge.plan import (
    PLAN_STRATEGIES,
    plan_from_audit,
)
from gradience.vnext.merge.report import MergeAuditReport


@pytest.fixture
def mock_report():
    """Minimal MergeAuditReport for plan generation tests."""
    return MergeAuditReport(
        adapter_a=AdapterMetadata(
            path="/tmp/adapter_a",
            rank=32,
            alpha=32.0,
            n_layers=2,
        ),
        adapter_b=AdapterMetadata(
            path="/tmp/adapter_b",
            rank=32,
            alpha=32.0,
            n_layers=2,
        ),
        matching=MatchingSummary(
            n_shared=2,
            n_only_a=0,
            n_only_b=0,
        ),
        layer_verdicts=[
            {
                "layer_name": "model.layers.0.self_attn.q_proj",
                "verdict": "safe",
                "metrics": {"mean_overlap": 0.3},
                "suggested_coefficients": None,
            },
            {
                "layer_name": "model.layers.0.self_attn.v_proj",
                "verdict": "conflicting",
                "metrics": {"mean_overlap": 0.7},
                "suggested_coefficients": None,
            },
        ],
        aggregate=AggregateResult(
            overall_verdict="safe",
            compatibility_score=0.65,
        ),
        recommendations=[],
    )


class TestDARELinearPlan:
    def test_registered(self):
        assert "dare_linear" in PLAN_STRATEGIES

    def test_generates_plan(self, mock_report):
        plan = plan_from_audit(
            "dare_linear",
            mock_report,
            "/tmp/adapter_a",
            "/tmp/adapter_b",
            output_rank=32,
            dare_drop_fraction=0.3,
        )
        assert plan.strategy_name == "dare_linear"
        assert len(plan.layer_configs) == 2
        for lc in plan.layer_configs:
            assert lc.strategy == "dare_linear"
            assert lc.trim_fraction == 0.3

    def test_default_drop_fraction(self, mock_report):
        plan = plan_from_audit(
            "dare_linear",
            mock_report,
            "/tmp/adapter_a",
            "/tmp/adapter_b",
        )
        for lc in plan.layer_configs:
            assert lc.trim_fraction == 0.3  # default


class TestDARETIESPlan:
    def test_registered(self):
        assert "dare_ties" in PLAN_STRATEGIES

    def test_generates_plan(self, mock_report):
        plan = plan_from_audit(
            "dare_ties",
            mock_report,
            "/tmp/adapter_a",
            "/tmp/adapter_b",
            output_rank=32,
            dare_drop_fraction=0.5,
        )
        assert plan.strategy_name == "dare_ties"
        assert len(plan.layer_configs) == 2
        for lc in plan.layer_configs:
            assert lc.strategy == "dare_ties"
            assert lc.trim_fraction == 0.5

    def test_default_drop_fraction(self, mock_report):
        plan = plan_from_audit(
            "dare_ties",
            mock_report,
            "/tmp/adapter_a",
            "/tmp/adapter_b",
        )
        for lc in plan.layer_configs:
            assert lc.trim_fraction == 0.5  # default
