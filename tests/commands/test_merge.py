"""Unit tests for merge commands — error paths and validation."""

from __future__ import annotations

import argparse
import json

import pytest

from gradience.commands.merge import cmd_explain, cmd_merge, cmd_merge_audit, cmd_merge_plan
from gradience.exceptions import ConfigError


def _make_merge_audit_args(**kwargs) -> argparse.Namespace:
    defaults = {
        "adapter_a": None,
        "adapter_b": None,
        "thresholds": "default",
        "output_dir": None,
        "energy_threshold": 0.90,
        "compute_dtype": "float64",
        "verbose": False,
        "source_a_qa": None,
        "source_b_qa": None,
        "compute_core_space": False,
        "strict_qa": False,
        "qa_report": False,
        "emit_report": None,
        "json": False,
        "strategy": None,
    }
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


def _make_merge_plan_args(**kwargs) -> argparse.Namespace:
    defaults = {
        "adapter_a": None,
        "adapter_b": None,
        "strategy": "uniform_linear",
        "output_dir": "/tmp/merge_out",
        "output_rank": 8,
        "output_alpha": 16.0,
        "verbose": False,
    }
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


def _make_merge_args(**kwargs) -> argparse.Namespace:
    defaults = {
        "plan": None,
        "output_dir": "/tmp/merge_out",
        "compute_dtype": "float64",
        "verbose": False,
    }
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


def _make_explain_args(**kwargs) -> argparse.Namespace:
    defaults = {
        "audit_json": None,
        "layer": None,
        "verbose": False,
    }
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


class TestMergeAuditValidation:
    def test_missing_adapters(self):
        args = _make_merge_audit_args()
        with pytest.raises(ConfigError, match="required"):
            cmd_merge_audit(args)

    def test_missing_adapter_a_only(self, tmp_path):
        b = tmp_path / "b"
        b.mkdir()
        args = _make_merge_audit_args(adapter_b=str(b))
        with pytest.raises(ConfigError, match="required"):
            cmd_merge_audit(args)

    def test_adapter_a_not_a_directory(self, tmp_path):
        a_file = tmp_path / "a.txt"
        a_file.write_text("not a dir")
        b = tmp_path / "b"
        b.mkdir()
        args = _make_merge_audit_args(adapter_a=str(a_file), adapter_b=str(b))
        with pytest.raises(ConfigError, match="not .* directory"):
            cmd_merge_audit(args)

    def test_valid_pair(self, orthogonal_pair):
        """cmd_merge_audit should complete successfully on a valid adapter pair."""
        a, b = orthogonal_pair
        args = _make_merge_audit_args(adapter_a=str(a), adapter_b=str(b))
        # Should not raise
        cmd_merge_audit(args)

    def test_json_output(self, orthogonal_pair, capsys):
        """--json should emit parseable JSON."""
        a, b = orthogonal_pair
        args = _make_merge_audit_args(adapter_a=str(a), adapter_b=str(b), json=True)
        cmd_merge_audit(args)
        out = capsys.readouterr().out
        payload = json.loads(out)
        assert "pair_risk" in payload or "aggregate" in payload


class TestMergePlanValidation:
    def test_missing_adapters(self):
        args = _make_merge_plan_args()
        with pytest.raises(ConfigError, match="required"):
            cmd_merge_plan(args)

    def test_adapter_not_directory(self, tmp_path):
        a_file = tmp_path / "a.txt"
        a_file.write_text("not a dir")
        b = tmp_path / "b"
        b.mkdir()
        args = _make_merge_plan_args(adapter_a=str(a_file), adapter_b=str(b))
        with pytest.raises(ConfigError, match="not .* directory"):
            cmd_merge_plan(args)


class TestMergeValidation:
    def test_missing_plan(self):
        args = _make_merge_args()
        with pytest.raises(ConfigError, match="required"):
            cmd_merge(args)

    def test_plan_file_not_found(self):
        args = _make_merge_args(plan="/nonexistent/plan.json")
        with pytest.raises(ConfigError, match="not found"):
            cmd_merge(args)


class TestExplainValidation:
    def test_missing_audit_json(self):
        args = _make_explain_args(layer="some.layer")
        with pytest.raises(ConfigError, match="required"):
            cmd_explain(args)

    def test_missing_layer(self):
        args = _make_explain_args(audit_json="/some/file.json")
        with pytest.raises(ConfigError, match="required"):
            cmd_explain(args)

    def test_audit_json_not_found(self, tmp_path):
        args = _make_explain_args(
            audit_json=str(tmp_path / "nonexistent.json"),
            layer="some.layer",
        )
        with pytest.raises(ConfigError, match="[Ee]rror loading"):
            cmd_explain(args)

    def test_audit_json_no_disagreement(self, tmp_path):
        """Audit JSON without policy_disagreement_analysis should raise ConfigError."""
        audit_file = tmp_path / "audit.json"
        audit_file.write_text(json.dumps({"layers": [], "summary": {}}))
        args = _make_explain_args(audit_json=str(audit_file), layer="some.layer")
        with pytest.raises(ConfigError, match="disagreement"):
            cmd_explain(args)
