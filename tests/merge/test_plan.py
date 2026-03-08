"""Tests for merge plan generation from audit reports.

Validates that the three planning strategies (uniform_linear, audit_aware,
overlap_ties) correctly map audit verdicts to per-layer merge configurations.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from gradience.vnext.merge.plan import (
    MergePlan,
    plan_from_audit,
    plan_uniform_linear,
    plan_audit_aware,
    plan_norm_equalized,
    plan_overlap_ties,
    PLAN_STRATEGIES,
)
from gradience.vnext.merge.report import MergeAuditReport
from gradience.vnext.merge.strategies import LayerMergeConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_layer_verdict(
    layer_name: str,
    verdict: str,
    mean_overlap: float = 0.3,
    directional_agreement: float = 0.5,
    suggested_strategy: str = "linear",
    suggested_coefficients: list | None = None,
) -> Dict[str, Any]:
    """Build a minimal layer_verdict dict matching MergeAuditReport schema."""
    return {
        "layer_name": layer_name,
        "module_type": "attn",
        "metrics": {
            "principal_angle_cosines": [mean_overlap],
            "mean_overlap": mean_overlap,
            "max_overlap": mean_overlap,
            "directional_agreement": directional_agreement,
            "magnitude_ratio": 1.0,
            "frobenius_ratio": 1.0,
            "frobenius_norm_a": 1.0,
            "frobenius_norm_b": 1.0,
            "effective_rank_a": 4,
            "effective_rank_b": 4,
            "nominal_rank_a": 8,
            "nominal_rank_b": 8,
            "stable_rank_a": 3.5,
            "stable_rank_b": 3.5,
        },
        "verdict": verdict,
        "confidence": 0.8,
        "recommendation": f"Test recommendation for {verdict}",
        "conflict_dimensions": 0,
        "safe_merge_rank": 8,
        "suggested_strategy": suggested_strategy,
        "suggested_coefficients": suggested_coefficients,
    }


def _make_report(
    layer_verdicts: List[Dict[str, Any]],
    overall_verdict: str = "safe",
) -> MergeAuditReport:
    """Build a minimal MergeAuditReport for testing."""
    return MergeAuditReport(
        adapter_a={"path": "/tmp/adapter_a", "rank": 8, "alpha": 16.0, "base_model": "test", "n_layers": 2},
        adapter_b={"path": "/tmp/adapter_b", "rank": 8, "alpha": 16.0, "base_model": "test", "n_layers": 2},
        matching={
            "n_shared": len(layer_verdicts),
            "n_only_a": 0,
            "n_only_b": 0,
            "shared_layers": [lv["layer_name"] for lv in layer_verdicts],
            "only_a_layers": [],
            "only_b_layers": [],
        },
        layer_verdicts=layer_verdicts,
        aggregate={
            "overall_verdict": overall_verdict,
            "compatibility_score": 0.5,
            "mean_overlap": 0.3,
            "median_overlap": 0.3,
            "max_overlap": 0.5,
            "mean_agreement": 0.5,
            "n_safe": sum(1 for lv in layer_verdicts if lv["verdict"] == "safe"),
            "n_redundant": sum(1 for lv in layer_verdicts if lv["verdict"] == "redundant"),
            "n_conflicting": sum(1 for lv in layer_verdicts if lv["verdict"] == "conflicting"),
            "n_imbalanced": sum(1 for lv in layer_verdicts if lv["verdict"] == "imbalanced"),
        },
        recommendations=["Test recommendation"],
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPlanUniformLinear:
    def test_all_layers_get_linear(self):
        """uniform_linear assigns 'linear' strategy to every layer."""
        report = _make_report([
            _make_layer_verdict("layer.0.q_proj", "safe"),
            _make_layer_verdict("layer.0.v_proj", "redundant"),
        ])
        plan = plan_uniform_linear(report, "/tmp/a", "/tmp/b")

        assert plan.strategy_name == "uniform_linear"
        assert len(plan.layer_configs) == 2
        for lc in plan.layer_configs:
            assert lc.strategy == "linear"
            assert lc.coefficients == (0.5, 0.5)
            assert lc.trim_fraction == 0.0

    def test_custom_coefficients(self):
        """Custom coefficients are applied to all layers."""
        report = _make_report([
            _make_layer_verdict("layer.0.q_proj", "safe"),
        ])
        plan = plan_uniform_linear(
            report, "/tmp/a", "/tmp/b",
            coefficients=(0.7, 0.3),
        )

        for lc in plan.layer_configs:
            assert lc.coefficients == (0.7, 0.3)

    def test_output_rank_and_alpha(self):
        """output_rank and output_alpha are correctly stored."""
        report = _make_report([_make_layer_verdict("layer.0.q_proj", "safe")])
        plan = plan_uniform_linear(
            report, "/tmp/a", "/tmp/b",
            output_rank=16, output_alpha=32.0,
        )

        assert plan.output_rank == 16
        assert plan.output_alpha == 32.0
        assert plan.layer_configs[0].target_rank == 16


class TestPlanAuditAware:
    def test_safe_layers_get_linear(self):
        """SAFE verdict → linear strategy."""
        report = _make_report([
            _make_layer_verdict("layer.0.q_proj", "safe"),
        ])
        plan = plan_audit_aware(report, "/tmp/a", "/tmp/b")

        assert plan.layer_configs[0].strategy == "linear"
        assert plan.layer_configs[0].coefficients == (0.5, 0.5)

    def test_redundant_layers_get_ties(self):
        """REDUNDANT verdict → ties strategy."""
        report = _make_report([
            _make_layer_verdict("layer.0.q_proj", "redundant"),
        ])
        plan = plan_audit_aware(report, "/tmp/a", "/tmp/b")

        assert plan.layer_configs[0].strategy == "ties"
        assert plan.layer_configs[0].trim_fraction == 0.0

    def test_conflicting_layers_get_dare_ties(self):
        """CONFLICTING verdict → dare_ties with computed drop rate."""
        report = _make_report([
            _make_layer_verdict("layer.0.q_proj", "conflicting"),
        ])
        plan = plan_audit_aware(report, "/tmp/a", "/tmp/b")

        assert plan.layer_configs[0].strategy == "dare_ties"
        assert plan.layer_configs[0].trim_fraction >= 0.15  # minimum DARE drop rate

    def test_imbalanced_layers_get_linear_rebalanced(self):
        """IMBALANCED verdict → linear with rebalanced coefficients."""
        report = _make_report([
            _make_layer_verdict(
                "layer.0.q_proj", "imbalanced",
                suggested_coefficients=[0.2, 0.8],
            ),
        ])
        plan = plan_audit_aware(report, "/tmp/a", "/tmp/b")

        assert plan.layer_configs[0].strategy == "linear"
        assert plan.layer_configs[0].coefficients == (0.2, 0.8)

    def test_mixed_verdicts(self):
        """Mixed verdicts produce appropriate per-layer strategies."""
        report = _make_report([
            _make_layer_verdict("layer.0.q_proj", "safe"),
            _make_layer_verdict("layer.0.v_proj", "redundant"),
            _make_layer_verdict("layer.1.q_proj", "conflicting"),
            _make_layer_verdict("layer.1.v_proj", "imbalanced", suggested_coefficients=[0.3, 0.7]),
        ])
        plan = plan_audit_aware(report, "/tmp/a", "/tmp/b")

        assert plan.layer_configs[0].strategy == "linear"      # safe
        assert plan.layer_configs[1].strategy == "ties"         # redundant
        assert plan.layer_configs[2].strategy == "dare_ties"    # conflicting → dare_ties
        assert plan.layer_configs[2].trim_fraction >= 0.15      # DARE drop rate
        assert plan.layer_configs[3].strategy == "linear"       # imbalanced
        assert plan.layer_configs[3].coefficients == (0.3, 0.7)


class TestPlanOverlapTies:
    def test_all_layers_get_ties(self):
        """overlap_ties assigns 'ties' strategy to every layer."""
        report = _make_report([
            _make_layer_verdict("layer.0.q_proj", "safe", mean_overlap=0.1),
            _make_layer_verdict("layer.0.v_proj", "redundant", mean_overlap=0.8),
        ])
        plan = plan_overlap_ties(report, "/tmp/a", "/tmp/b", trim_fraction=0.3)

        for lc in plan.layer_configs:
            assert lc.strategy == "ties"

    def test_trim_scales_with_overlap(self):
        """Trim fraction is proportional to mean_overlap."""
        report = _make_report([
            _make_layer_verdict("layer.0.q_proj", "safe", mean_overlap=0.1),
            _make_layer_verdict("layer.0.v_proj", "redundant", mean_overlap=0.9),
        ])
        plan = plan_overlap_ties(report, "/tmp/a", "/tmp/b", trim_fraction=0.5)

        # Low overlap: 0.5 * 0.1 = 0.05
        assert abs(plan.layer_configs[0].trim_fraction - 0.05) < 1e-4
        # High overlap: 0.5 * 0.9 = 0.45
        assert abs(plan.layer_configs[1].trim_fraction - 0.45) < 1e-4


class TestPlanJsonRoundtrip:
    def test_serialize_deserialize(self, tmp_path):
        """Plan survives JSON round-trip with all fields preserved."""
        report = _make_report([
            _make_layer_verdict("layer.0.q_proj", "safe"),
            _make_layer_verdict("layer.0.v_proj", "redundant"),
        ])
        plan = plan_uniform_linear(report, "/tmp/a", "/tmp/b", output_rank=16)

        # Write to JSON
        json_path = tmp_path / "test_plan.json"
        plan.to_json(json_path)

        # Read back
        loaded = MergePlan.from_json(json_path)

        assert loaded.plan_id == plan.plan_id
        assert loaded.strategy_name == plan.strategy_name
        assert loaded.adapter_a_dir == plan.adapter_a_dir
        assert loaded.adapter_b_dir == plan.adapter_b_dir
        assert loaded.output_rank == plan.output_rank
        assert loaded.output_alpha == plan.output_alpha
        assert len(loaded.layer_configs) == len(plan.layer_configs)

        for orig, rest in zip(plan.layer_configs, loaded.layer_configs):
            assert orig == rest

    def test_dict_roundtrip(self):
        """Plan survives dict round-trip."""
        report = _make_report([
            _make_layer_verdict("layer.0.q_proj", "safe"),
        ])
        plan = plan_audit_aware(report, "/tmp/a", "/tmp/b")

        d = plan.to_dict()
        restored = MergePlan.from_dict(d)

        assert restored.plan_id == plan.plan_id
        assert restored.strategy_name == plan.strategy_name
        assert restored.layer_configs == plan.layer_configs


class TestPlanFromAudit:
    def test_dispatch_all_strategies(self):
        """plan_from_audit correctly dispatches to all 3 strategies."""
        report = _make_report([_make_layer_verdict("layer.0.q_proj", "safe")])

        for name in ["uniform_linear", "audit_aware", "overlap_ties"]:
            plan = plan_from_audit(name, report, "/tmp/a", "/tmp/b")
            assert plan.strategy_name == name

    def test_unknown_strategy_raises(self):
        """Unknown strategy name raises ValueError."""
        report = _make_report([_make_layer_verdict("layer.0.q_proj", "safe")])
        with pytest.raises(ValueError, match="Unknown plan strategy"):
            plan_from_audit("nonexistent", report, "/tmp/a", "/tmp/b")

    def test_kwargs_forwarded(self):
        """Extra kwargs are forwarded to the strategy function."""
        report = _make_report([_make_layer_verdict("layer.0.q_proj", "safe")])
        plan = plan_from_audit(
            "uniform_linear", report, "/tmp/a", "/tmp/b",
            output_rank=32, output_alpha=64.0,
        )
        assert plan.output_rank == 32
        assert plan.output_alpha == 64.0


class TestPlanNormEqualized:
    def test_all_layers_get_norm_equalized(self):
        """norm_equalized assigns 'norm_equalized' strategy to every layer."""
        report = _make_report([
            _make_layer_verdict("layer.0.q_proj", "safe"),
            _make_layer_verdict("layer.0.v_proj", "imbalanced"),
        ])
        plan = plan_norm_equalized(report, "/tmp/a", "/tmp/b")

        assert plan.strategy_name == "norm_equalized"
        assert len(plan.layer_configs) == 2
        for lc in plan.layer_configs:
            assert lc.strategy == "norm_equalized"
            assert lc.coefficients == (0.5, 0.5)
            assert lc.trim_fraction == 0.0

    def test_custom_coefficients(self):
        """Custom coefficients are applied to all layers."""
        report = _make_report([
            _make_layer_verdict("layer.0.q_proj", "safe"),
        ])
        plan = plan_norm_equalized(
            report, "/tmp/a", "/tmp/b",
            coefficients=(0.7, 0.3),
        )

        for lc in plan.layer_configs:
            assert lc.coefficients == (0.7, 0.3)

    def test_via_plan_from_audit(self):
        """plan_from_audit dispatches to norm_equalized correctly."""
        report = _make_report([_make_layer_verdict("layer.0.q_proj", "safe")])
        plan = plan_from_audit("norm_equalized", report, "/tmp/a", "/tmp/b")
        assert plan.strategy_name == "norm_equalized"
        assert plan.layer_configs[0].strategy == "norm_equalized"


class TestPlanStrategiesRegistry:
    def test_all_strategies_registered(self):
        """All expected strategies are in the PLAN_STRATEGIES registry."""
        assert set(PLAN_STRATEGIES.keys()) == {
            "uniform_linear", "audit_aware", "norm_equalized", "overlap_ties",
            "dare_linear", "dare_ties",
        }
