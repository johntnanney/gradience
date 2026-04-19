"""CLI exit code tests for artifact-producing commands.

Verifies that audit-adapter, merge-audit, and summarize-inventory
return correct exit codes on success and failure.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path


def _run_gradience(*args: str, check: bool = False) -> subprocess.CompletedProcess[str]:
    """Run 'python -m gradience <args>' and capture output."""
    return subprocess.run(
        [sys.executable, "-m", "gradience", *args],
        capture_output=True,
        text=True,
        check=check,
        timeout=60,
    )


# ---------------------------------------------------------------------------
# summarize-inventory
# ---------------------------------------------------------------------------


class TestSummarizeInventoryExitCodes:
    """Exit code tests for 'gradience summarize-inventory'."""

    def test_success_with_valid_dirs(self) -> None:
        """Exit 0 when given valid QA and report directories."""
        result = _run_gradience(
            "summarize-inventory",
            "--qa-dir",
            "examples/qa",
            "--report-dir",
            "examples/reports",
        )
        assert result.returncode == 0

    def test_success_emits_valid_json(self) -> None:
        """--emit-report writes valid JSON and exits 0."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            out_path = tmp.name
        try:
            result = _run_gradience(
                "summarize-inventory",
                "--qa-dir",
                "examples/qa",
                "--report-dir",
                "examples/reports",
                "--emit-report",
                out_path,
            )
            assert result.returncode == 0
            with open(out_path) as f:
                data = json.load(f)
            assert data["schema"] == "gradience.inventory_summary/v1"
        finally:
            Path(out_path).unlink(missing_ok=True)

    def test_strict_input_fails_on_malformed(self) -> None:
        """--strict-input causes non-zero exit on malformed JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            bad_file = Path(tmpdir) / "bad.json"
            bad_file.write_text('{"schema": "gradience.adapter_qa/v1"}')
            result = _run_gradience(
                "summarize-inventory",
                "--qa-dir",
                tmpdir,
                "--strict-input",
            )
            assert result.returncode != 0

    def test_no_args_fails(self) -> None:
        """No arguments at all -> non-zero exit."""
        result = _run_gradience("summarize-inventory")
        assert result.returncode != 0


# ---------------------------------------------------------------------------
# audit-adapter
# ---------------------------------------------------------------------------


class TestAuditAdapterExitCodes:
    """Exit code tests for 'gradience audit-adapter'."""

    def test_success_with_real_adapter(self) -> None:
        """Exit 0 when auditing a real PEFT adapter directory."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            out_path = tmp.name
        try:
            result = _run_gradience(
                "audit-adapter",
                "--peft-dir",
                "examples/adapters/tiny_lora",
                "--out",
                out_path,
            )
            assert result.returncode == 0, f"stderr: {result.stderr}"
            with open(out_path) as f:
                data = json.load(f)
            assert data["schema"] == "gradience.adapter_qa/v1"
        finally:
            Path(out_path).unlink(missing_ok=True)

    def test_missing_required_arg_fails(self) -> None:
        """Missing required --peft-dir argument -> non-zero exit."""
        result = _run_gradience("audit-adapter")
        assert result.returncode != 0


# ---------------------------------------------------------------------------
# merge-audit
# ---------------------------------------------------------------------------


class TestMergeAuditExitCodes:
    """Exit code tests for 'gradience merge-audit'."""

    def test_success_with_real_adapters(self) -> None:
        """Exit 0 when merging two copies of the same adapter."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            out_path = tmp.name
        try:
            result = _run_gradience(
                "merge-audit",
                "--adapter-a",
                "examples/adapters/tiny_lora",
                "--adapter-b",
                "examples/adapters/tiny_lora",
                "--emit-report",
                out_path,
            )
            assert result.returncode == 0, f"stderr: {result.stderr}"
            with open(out_path) as f:
                data = json.load(f)
            assert data["schema"].startswith("gradience.merge_qa_report/v1")
        finally:
            Path(out_path).unlink(missing_ok=True)

    def test_nonexistent_adapter_fails(self) -> None:
        """Non-existent adapter directory -> non-zero exit."""
        result = _run_gradience(
            "merge-audit",
            "--adapter-a",
            "/nonexistent/path",
            "--adapter-b",
            "examples/adapters/tiny_lora",
        )
        assert result.returncode != 0


# ---------------------------------------------------------------------------
# suggest-neighborhoods
# ---------------------------------------------------------------------------


class TestSuggestNeighborhoodsExitCodes:
    """Exit code tests for 'gradience suggest-neighborhoods'."""

    def test_success_with_example_dirs(self) -> None:
        result = _run_gradience(
            "suggest-neighborhoods",
            "--qa-dir",
            "examples/qa",
            "--report-dir",
            "examples/reports",
        )
        assert result.returncode == 0

    def test_emit_report_schema(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            out_path = tmp.name
        try:
            result = _run_gradience(
                "suggest-neighborhoods",
                "--qa-dir",
                "examples/qa",
                "--report-dir",
                "examples/reports",
                "--emit-report",
                out_path,
            )
            assert result.returncode == 0
            with open(out_path) as f:
                data = json.load(f)
            assert data["schema"] == "gradience.merge_neighborhoods/v1"
        finally:
            Path(out_path).unlink(missing_ok=True)

    def test_missing_report_dir_fails(self) -> None:
        result = _run_gradience("suggest-neighborhoods")
        assert result.returncode != 0
