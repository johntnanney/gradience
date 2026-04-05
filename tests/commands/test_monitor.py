"""Unit tests for cmd_report and cmd_monitor — error paths and validation."""

from __future__ import annotations

import argparse

import pytest

from gradience.commands.monitor import cmd_monitor, cmd_report
from gradience.exceptions import TelemetryError


def _make_report_args(**kwargs) -> argparse.Namespace:
    defaults = {
        "file": "/nonexistent/telemetry.jsonl",
    }
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


def _make_monitor_args(**kwargs) -> argparse.Namespace:
    defaults = {
        "file": "/nonexistent/telemetry.jsonl",
        "gap_threshold": 1.5,
        "strict_schema": False,
        "verbose": False,
        "json": False,
    }
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


class TestReportValidation:
    def test_file_not_found(self):
        args = _make_report_args(file="/no/such/file.jsonl")
        with pytest.raises(TelemetryError, match="not found|No such"):
            cmd_report(args)


class TestMonitorValidation:
    def test_file_not_found(self):
        args = _make_monitor_args(file="/no/such/file.jsonl")
        with pytest.raises(TelemetryError, match="not found|No such"):
            cmd_monitor(args)
