"""Unit tests for cli_format — output formatting utilities."""

from __future__ import annotations

from gradience.cli_format import _fmt, _fmt_params, _severity_rank


class TestFmt:
    def test_none_returns_dash(self):
        assert _fmt(None) == "-"

    def test_integer(self):
        assert _fmt(8) == "8"

    def test_float(self):
        assert _fmt(0.123) == "0.123"

    def test_percentage(self):
        assert _fmt(0.85, pct=True) == "85.0%"

    def test_percentage_none(self):
        assert _fmt(None, pct=True) == "-"

    def test_small_scientific(self):
        result = _fmt(0.000123)
        assert "e" in result

    def test_large_scientific(self):
        result = _fmt(12345.0)
        assert "e" in result

    def test_string_passthrough(self):
        assert _fmt("hello") == "hello"

    def test_zero(self):
        assert _fmt(0) == "0"


class TestFmtParams:
    def test_none(self):
        assert _fmt_params(None) == "n/a"

    def test_billions(self):
        assert _fmt_params(1.5e9) == "1.5B"

    def test_millions(self):
        assert _fmt_params(2.5e6) == "2.5M"

    def test_thousands(self):
        assert _fmt_params(1500) == "1.5K"

    def test_small_integer(self):
        assert _fmt_params(42) == "42"

    def test_string_passthrough(self):
        assert _fmt_params("unknown") == "unknown"


class TestSeverityRank:
    def test_critical_is_lowest(self):
        assert _severity_rank("critical") < _severity_rank("error")

    def test_error_before_warning(self):
        assert _severity_rank("error") < _severity_rank("warning")

    def test_warning_before_info(self):
        assert _severity_rank("warning") < _severity_rank("info")

    def test_unknown_is_last(self):
        assert _severity_rank("unknown_severity") > _severity_rank("info")

    def test_case_insensitive(self):
        assert _severity_rank("CRITICAL") == _severity_rank("critical")
