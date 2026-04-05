"""Unit tests for cmd_truncate — error paths and validation."""

from __future__ import annotations

import argparse

import pytest

from gradience.commands.truncate import cmd_truncate
from gradience.exceptions import ConfigError


def _make_args(**kwargs) -> argparse.Namespace:
    defaults = {
        "peft_dir": "/nonexistent",
        "out_dir": "/tmp/out",
        "rank": 4,
        "alpha_mode": "keep_ratio",
        "dtype": "fp16",
        "report": None,
        "json": False,
        "verbose": False,
    }
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


class TestTruncateValidation:
    def test_missing_peft_dir(self):
        args = _make_args(peft_dir="/no/such/dir")
        with pytest.raises(ConfigError, match="not found"):
            cmd_truncate(args)

    def test_zero_rank(self, tmp_path):
        # Create a real directory so the path check passes
        peft_dir = tmp_path / "adapter"
        peft_dir.mkdir()
        args = _make_args(peft_dir=str(peft_dir), rank=0)
        with pytest.raises(ConfigError, match="positive"):
            cmd_truncate(args)

    def test_negative_rank(self, tmp_path):
        peft_dir = tmp_path / "adapter"
        peft_dir.mkdir()
        args = _make_args(peft_dir=str(peft_dir), rank=-1)
        with pytest.raises(ConfigError, match="positive"):
            cmd_truncate(args)
