"""Unit tests for cmd_audit and cmd_audit_adapter — error paths and validation."""

from __future__ import annotations

import argparse

import pytest

from gradience.commands.audit import cmd_audit, cmd_audit_adapter
from gradience.exceptions import ConfigError


def _make_audit_args(**kwargs) -> argparse.Namespace:
    defaults = {
        "peft_dir": None,
        "adapter_config": None,
        "weights": None,
        "top_wasteful": 0,
        "top_singular_values": 0,
        "json": False,
        "layers": False,
        "suggest_per_layer": False,
        "base_model": None,
        "base_norms_cache": None,
        "no_udr": True,
        "rank_policies": "energy@0.90,knee,erank",
        "importance_quantile": 0.75,
        "importance_uniform_mult_gate": 1.5,
        "importance_metric": "energy_share",
        "disagreement_rationale": "flagged_only",
        "append": None,
        "verbose": False,
    }
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


def _make_audit_adapter_args(**kwargs) -> argparse.Namespace:
    defaults = {
        "peft_dir": None,
        "adapter_config": None,
        "weights": None,
        "base_model": None,
        "base_norms_cache": None,
        "no_udr": True,
        "eval_dataset": None,
        "metric_name": None,
        "adapter_score": None,
        "base_score": None,
        "lower_is_better": True,
        "higher_is_better": False,
        "margin": 0.0,
        "out": None,
        "json": False,
    }
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


class TestAuditValidation:
    def test_missing_peft_dir(self):
        args = _make_audit_args()
        with pytest.raises(ConfigError, match="peft-dir"):
            cmd_audit(args)


class TestAuditAdapterValidation:
    def test_missing_peft_dir(self):
        args = _make_audit_adapter_args()
        with pytest.raises(ConfigError, match="peft-dir"):
            cmd_audit_adapter(args)
