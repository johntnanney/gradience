"""Unit tests for cmd_check — error paths and config merging."""

from __future__ import annotations

import argparse
import json

import pytest

from gradience.commands.check import cmd_check
from gradience.exceptions import ConfigError


def _make_args(**kwargs) -> argparse.Namespace:
    defaults = {
        "config": None,
        "peft": None,
        "peft_dir": None,
        "training": None,
        "training_dir": None,
        "task": None,
        "dataset": None,
        "model": None,
        "task_profile": None,
        "optimizer": None,
        "lr": None,
        "weight_decay": None,
        "r": None,
        "alpha": None,
        "targets": None,
        "notes": None,
        "verbose": False,
        "json": False,
    }
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


class TestCheckValidation:
    def test_no_inputs_raises(self):
        """cmd_check with no config/peft/training should raise ConfigError."""
        args = _make_args()
        with pytest.raises(ConfigError, match="provide either"):
            cmd_check(args)

    def test_config_file_not_found(self):
        args = _make_args(config="/nonexistent/config.yaml")
        with pytest.raises(ConfigError, match="not found"):
            cmd_check(args)

    def test_peft_file_not_found(self):
        args = _make_args(peft="/nonexistent/adapter_config.json")
        with pytest.raises(ConfigError, match="not found"):
            cmd_check(args)

    def test_training_file_not_found(self):
        args = _make_args(training="/nonexistent/training_args.json")
        with pytest.raises(ConfigError, match="not found"):
            cmd_check(args)

    def test_task_aliases_to_dataset(self, tmp_path):
        """--task should set dataset when dataset is None."""
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps({
            "model_name": "test-model",
            "dataset_name": None,
            "task_profile": "easy_classification",
            "optimizer": {"type": "AdamW", "lr": 5e-5, "weight_decay": 0.01},
            "lora": {"r": 8, "alpha": 16, "target_modules": ["q_proj"]},
        }))
        args = _make_args(config=str(config_file), task="gsm8k")
        # Should not raise — task becomes dataset
        cmd_check(args)
        assert args.dataset == "gsm8k"

    def test_peft_dir_auto_detect_missing_dir(self):
        """--peft-dir with non-existent directory should raise ConfigError."""
        args = _make_args(peft_dir="/no/such/peft_dir")
        with pytest.raises(ConfigError):
            cmd_check(args)

    def test_malformed_json_config(self, tmp_path):
        config_file = tmp_path / "bad.json"
        config_file.write_text("{ this is not json")
        args = _make_args(config=str(config_file))
        with pytest.raises(ConfigError, match="parse"):
            cmd_check(args)

    def test_valid_config_runs(self, tmp_path):
        """A minimal valid config should run without error."""
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps({
            "model_name": "test-model",
            "dataset_name": "sst2",
            "task_profile": "easy_classification",
            "optimizer": {"type": "AdamW", "lr": 5e-5, "weight_decay": 0.01},
            "lora": {"r": 8, "alpha": 16, "target_modules": ["q_proj", "v_proj"]},
        }))
        args = _make_args(config=str(config_file))
        # Should complete without exception
        cmd_check(args)

    def test_json_output(self, tmp_path, capsys):
        """--json should emit parseable JSON."""
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps({
            "model_name": "test-model",
            "dataset_name": "sst2",
            "task_profile": "easy_classification",
            "optimizer": {"type": "AdamW", "lr": 5e-5, "weight_decay": 0.01},
            "lora": {"r": 8, "alpha": 16, "target_modules": ["q_proj"]},
        }))
        args = _make_args(config=str(config_file), json=True)
        cmd_check(args)
        out = capsys.readouterr().out
        payload = json.loads(out)
        assert "config" in payload
        assert "recommendations" in payload
