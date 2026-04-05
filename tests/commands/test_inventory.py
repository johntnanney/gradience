"""Unit tests for inventory commands — error paths and validation."""

from __future__ import annotations

import argparse
import json

import pytest

from gradience.commands.inventory import cmd_batch_summary, cmd_preflight_report, cmd_summarize_inventory
from gradience.exceptions import ConfigError


def _make_summarize_args(**kwargs) -> argparse.Namespace:
    defaults = {
        "qa_dir": None,
        "report_dir": None,
        "strict_input": False,
        "emit_report": None,
        "emit_bundle": None,
        "previous_run": None,
        "inventory_id": None,
        "run_id": None,
    }
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


def _make_batch_summary_args(**kwargs) -> argparse.Namespace:
    defaults = {
        "run_dir": ".",
        "emit_report": None,
    }
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


def _make_preflight_report_args(**kwargs) -> argparse.Namespace:
    defaults = {
        "bundle_dir": "/nonexistent",
        "output": None,
        "title": None,
    }
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


class TestSummarizeInventoryValidation:
    def test_no_dirs_raises(self):
        """Must provide at least one of --qa-dir or --report-dir."""
        args = _make_summarize_args()
        with pytest.raises(ConfigError, match="qa-dir.*report-dir|report-dir.*qa-dir"):
            cmd_summarize_inventory(args)

    def test_empty_dirs(self, tmp_path):
        """Empty qa and report dirs should still run (producing empty summary)."""
        qa = tmp_path / "qa"
        qa.mkdir()
        reports = tmp_path / "reports"
        reports.mkdir()
        args = _make_summarize_args(qa_dir=str(qa), report_dir=str(reports))
        # Should not raise — empty inventory is valid
        cmd_summarize_inventory(args)

    def test_emit_report(self, tmp_path):
        """--emit-report should write summary JSON to disk."""
        qa = tmp_path / "qa"
        qa.mkdir()
        out_path = tmp_path / "summary.json"
        args = _make_summarize_args(qa_dir=str(qa), emit_report=str(out_path))
        cmd_summarize_inventory(args)
        assert out_path.exists()
        data = json.loads(out_path.read_text())
        assert "schema" in data


class TestBatchSummaryValidation:
    def test_nonexistent_dir(self):
        args = _make_batch_summary_args(run_dir="/no/such/dir")
        with pytest.raises(ConfigError, match="not a directory"):
            cmd_batch_summary(args)


class TestPreflightReportValidation:
    def test_nonexistent_bundle_dir(self):
        args = _make_preflight_report_args(bundle_dir="/no/such/dir")
        with pytest.raises(ConfigError, match="not a directory"):
            cmd_preflight_report(args)

    def test_missing_summary_json(self, tmp_path):
        """Bundle dir without preflight_summary.json should raise."""
        args = _make_preflight_report_args(bundle_dir=str(tmp_path))
        with pytest.raises(ConfigError, match="preflight_summary"):
            cmd_preflight_report(args)
