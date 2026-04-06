"""Tests for the portfolio view module and CLI command."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from gradience.exceptions import QASchemaError
from gradience.vnext.inventory.portfolio import (
    PortfolioView,
    _derive_dominant_driver,
    _derive_evidence_profile,
    _derive_exploration_posture,
    format_portfolio_html,
    format_portfolio_table,
    scan_portfolio,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_INVENTORY_SUMMARY_V1 = "gradience.inventory_summary/v1"


def _write_inventory_summary(path: Path, **overrides: object) -> None:
    """Write a minimal valid inventory summary JSON to *path*."""
    data: dict = {
        "schema": _INVENTORY_SUMMARY_V1,
        "sources": {"qa_artifact_count": 3, "merge_report_count": 2},
        "adapter_status_counts": {"eligible": 2, "flagged_weak": 1},
        "adapter_flag_counts": {"low_utilization": 1},
        "pair_risk_counts": {"low": 1, "high": 1},
        "recommended_strategy_counts": {"linear": 1, "audit_aware": 1},
        "dominant_issue_counts": {"subspace_conflict": 1, "none": 1},
        "strict_qa_block_candidates": 1,
        "notes": [],
    }
    data.update(overrides)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# ---------------------------------------------------------------------------
# Unit tests: derivation helpers
# ---------------------------------------------------------------------------


class TestDeriveDominantDriver:
    def test_returns_top_non_none_issue(self) -> None:
        counts = {"none": 5, "subspace_conflict": 3, "norm_imbalance": 1}
        assert _derive_dominant_driver(counts) == "subspace_conflict"

    def test_returns_none_when_only_none(self) -> None:
        counts = {"none": 2}
        assert _derive_dominant_driver(counts) == "none"

    def test_returns_dash_when_empty(self) -> None:
        assert _derive_dominant_driver({}) == "—"

    def test_ignores_unknown_issue(self) -> None:
        counts = {"unknown": 5, "norm_imbalance": 2}
        assert _derive_dominant_driver(counts) == "norm_imbalance"

    def test_ties_resolved_by_max(self) -> None:
        counts = {"subspace_conflict": 5, "norm_imbalance": 3}
        assert _derive_dominant_driver(counts) == "subspace_conflict"


class TestDeriveExplorationPosture:
    def test_linear_gives_standard(self) -> None:
        assert _derive_exploration_posture({"linear": 3}) == "standard"

    def test_audit_aware_gives_high_scrutiny(self) -> None:
        assert _derive_exploration_posture({"audit_aware": 2, "linear": 1}) == "high-scrutiny"

    def test_norm_equalized_gives_balanced(self) -> None:
        assert _derive_exploration_posture({"norm_equalized": 4}) == "balanced"

    def test_empty_gives_dash(self) -> None:
        assert _derive_exploration_posture({}) == "—"

    def test_unknown_strategy_returned_as_is(self) -> None:
        result = _derive_exploration_posture({"unknown_strategy": 1})
        assert result == "unknown_strategy"


class TestDeriveEvidenceProfile:
    def test_all_eligible_gives_full(self) -> None:
        assert _derive_evidence_profile({"eligible": 3}) == "full"

    def test_all_unknown_gives_blind(self) -> None:
        assert _derive_evidence_profile({"unknown_no_behavioral_eval": 3}) == "blind"

    def test_partial_when_mix_with_eligible(self) -> None:
        result = _derive_evidence_profile({"eligible": 2, "unknown_no_behavioral_eval": 1})
        assert result == "partial"

    def test_weak_when_flagged_but_no_eligible(self) -> None:
        result = _derive_evidence_profile({"flagged_weak": 2, "unknown_no_behavioral_eval": 1})
        assert result == "weak"

    def test_empty_gives_dash(self) -> None:
        assert _derive_evidence_profile({}) == "—"


# ---------------------------------------------------------------------------
# Integration tests: scan_portfolio
# ---------------------------------------------------------------------------


class TestScanPortfolio:
    def test_empty_directory_returns_empty_view(self, tmp_path: Path) -> None:
        view = scan_portfolio(tmp_path)
        assert isinstance(view, PortfolioView)
        assert view.rows == ()
        assert str(tmp_path) in view.scanned_dir

    def test_nonexistent_directory_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="does not exist"):
            scan_portfolio(tmp_path / "nonexistent")

    def test_file_path_raises(self, tmp_path: Path) -> None:
        fp = tmp_path / "file.json"
        fp.write_text("{}")
        with pytest.raises(ValueError, match="not a directory"):
            scan_portfolio(fp)

    def test_root_level_summary_produces_row(self, tmp_path: Path) -> None:
        _write_inventory_summary(tmp_path / "my_inventory.json")
        view = scan_portfolio(tmp_path)
        assert len(view.rows) == 1
        row = view.rows[0]
        assert row.name == "my_inventory"
        assert row.adapter_count == 3
        assert row.pair_count == 2

    def test_subdir_summary_uses_subdir_name(self, tmp_path: Path) -> None:
        _write_inventory_summary(tmp_path / "math_loras" / "inventory_summary.json")
        view = scan_portfolio(tmp_path)
        assert len(view.rows) == 1
        assert view.rows[0].name == "math_loras"

    def test_multiple_inventories(self, tmp_path: Path) -> None:
        _write_inventory_summary(tmp_path / "inv_a" / "summary.json")
        _write_inventory_summary(tmp_path / "inv_b" / "summary.json")
        _write_inventory_summary(tmp_path / "root_inv.json")
        view = scan_portfolio(tmp_path)
        names = {r.name for r in view.rows}
        assert names == {"inv_a", "inv_b", "root_inv"}

    def test_non_inventory_json_ignored(self, tmp_path: Path) -> None:
        (tmp_path / "random.json").write_text('{"schema": "some.other/v1"}')
        view = scan_portfolio(tmp_path)
        assert view.rows == ()

    def test_retained_candidates_from_eligible(self, tmp_path: Path) -> None:
        _write_inventory_summary(
            tmp_path / "inv.json",
            adapter_status_counts={"eligible": 4, "flagged_weak": 1},
        )
        view = scan_portfolio(tmp_path)
        assert view.rows[0].retained_candidates == 4

    def test_reduction_pct_computed(self, tmp_path: Path) -> None:
        _write_inventory_summary(
            tmp_path / "inv.json",
            adapter_status_counts={"eligible": 3, "flagged_weak": 1, "uncertain": 1},
        )
        view = scan_portfolio(tmp_path)
        row = view.rows[0]
        assert row.reduction_pct is not None
        # 2 removed out of 5 total → 40%
        assert row.reduction_pct == pytest.approx(40.0, abs=0.1)

    def test_reduction_pct_none_when_no_adapters(self, tmp_path: Path) -> None:
        _write_inventory_summary(tmp_path / "inv.json", adapter_status_counts={})
        view = scan_portfolio(tmp_path)
        assert view.rows[0].reduction_pct is None

    def test_dominant_driver_derived(self, tmp_path: Path) -> None:
        _write_inventory_summary(
            tmp_path / "inv.json",
            dominant_issue_counts={"norm_imbalance": 3, "none": 1},
        )
        view = scan_portfolio(tmp_path)
        assert view.rows[0].dominant_driver == "norm_imbalance"

    def test_html_link_found(self, tmp_path: Path) -> None:
        _write_inventory_summary(tmp_path / "inv.json")
        (tmp_path / "report.html").write_text("<html/>")
        view = scan_portfolio(tmp_path)
        assert view.rows[0].html_report_link is not None
        assert "report.html" in view.rows[0].html_report_link

    def test_no_html_link_when_none(self, tmp_path: Path) -> None:
        _write_inventory_summary(tmp_path / "inv.json")
        view = scan_portfolio(tmp_path)
        assert view.rows[0].html_report_link is None

    def test_drift_status_found(self, tmp_path: Path) -> None:
        _write_inventory_summary(tmp_path / "inv.json")
        (tmp_path / "spectral_drift_report.json").write_text('{"schema": "other/v1"}')
        view = scan_portfolio(tmp_path)
        assert "drift" in view.rows[0].drift_status

    def test_drift_status_dash_when_none(self, tmp_path: Path) -> None:
        _write_inventory_summary(tmp_path / "inv.json")
        view = scan_portfolio(tmp_path)
        assert view.rows[0].drift_status == "—"

    def test_malformed_json_skipped_by_default(self, tmp_path: Path) -> None:
        (tmp_path / "bad.json").write_text("{not valid json")
        _write_inventory_summary(tmp_path / "good.json")
        view = scan_portfolio(tmp_path)
        assert len(view.rows) == 1

    def test_strict_input_raises_on_malformed_schema(self, tmp_path: Path) -> None:
        """strict_input should propagate exceptions from from_dict."""
        (tmp_path / "bad.json").write_text(
            json.dumps({"schema": _INVENTORY_SUMMARY_V1, "sources": "not-a-dict"})
        )
        with pytest.raises(QASchemaError):
            scan_portfolio(tmp_path, strict_input=True)

    def test_duplicate_names_uses_newest(self, tmp_path: Path) -> None:
        """When a subdir contains multiple JSONs, only the newest is kept."""
        import time

        subdir = tmp_path / "inv"
        subdir.mkdir()
        old = subdir / "old.json"
        _write_inventory_summary(old, sources={"qa_artifact_count": 1, "merge_report_count": 0})
        time.sleep(0.01)
        new = subdir / "new.json"
        _write_inventory_summary(new, sources={"qa_artifact_count": 99, "merge_report_count": 0})

        view = scan_portfolio(tmp_path)
        assert len(view.rows) == 1
        assert view.rows[0].adapter_count == 99

    def test_artifact_link_is_absolute_path(self, tmp_path: Path) -> None:
        _write_inventory_summary(tmp_path / "inv.json")
        view = scan_portfolio(tmp_path)
        assert Path(view.rows[0].artifact_link).is_absolute()


# ---------------------------------------------------------------------------
# Formatter tests
# ---------------------------------------------------------------------------


class TestFormatPortfolioTable:
    def test_empty_view_produces_no_inventory_message(self, tmp_path: Path) -> None:
        view = PortfolioView(rows=(), scanned_dir=str(tmp_path), generated_at="2024-01-01 00:00 UTC")
        out = format_portfolio_table(view)
        assert "No inventory summaries found" in out

    def test_table_contains_column_headers(self, tmp_path: Path) -> None:
        _write_inventory_summary(tmp_path / "inv.json")
        view = scan_portfolio(tmp_path)
        out = format_portfolio_table(view)
        for header in ("Inventory", "Adapters", "Pairs", "Retained", "Reduction", "Dominant Driver"):
            assert header in out

    def test_table_contains_row_data(self, tmp_path: Path) -> None:
        _write_inventory_summary(tmp_path / "my_project.json")
        view = scan_portfolio(tmp_path)
        out = format_portfolio_table(view)
        assert "my_project" in out

    def test_table_shows_reduction_pct(self, tmp_path: Path) -> None:
        _write_inventory_summary(
            tmp_path / "inv.json",
            adapter_status_counts={"eligible": 1, "flagged_weak": 1},
        )
        view = scan_portfolio(tmp_path)
        out = format_portfolio_table(view)
        assert "50%" in out

    def test_generated_at_in_output(self, tmp_path: Path) -> None:
        _write_inventory_summary(tmp_path / "inv.json")
        view = scan_portfolio(tmp_path)
        out = format_portfolio_table(view)
        assert "Generated:" in out


class TestFormatPortfolioHtml:
    def test_produces_html_doctype(self, tmp_path: Path) -> None:
        _write_inventory_summary(tmp_path / "inv.json")
        view = scan_portfolio(tmp_path)
        html = format_portfolio_html(view)
        assert "<!DOCTYPE html>" in html

    def test_contains_table_element(self, tmp_path: Path) -> None:
        _write_inventory_summary(tmp_path / "inv.json")
        view = scan_portfolio(tmp_path)
        html = format_portfolio_html(view)
        assert "<table>" in html

    def test_contains_inventory_name(self, tmp_path: Path) -> None:
        _write_inventory_summary(tmp_path / "math_loras.json")
        view = scan_portfolio(tmp_path)
        html = format_portfolio_html(view)
        assert "math_loras" in html

    def test_empty_view_has_no_inventory_message(self, tmp_path: Path) -> None:
        view = PortfolioView(rows=(), scanned_dir=str(tmp_path), generated_at="2024-01-01 00:00 UTC")
        html = format_portfolio_html(view)
        assert "No inventories found" in html

    def test_json_link_present(self, tmp_path: Path) -> None:
        _write_inventory_summary(tmp_path / "inv.json")
        view = scan_portfolio(tmp_path)
        html = format_portfolio_html(view)
        assert "JSON" in html
        assert "inv.json" in html

    def test_html_link_present_when_report_exists(self, tmp_path: Path) -> None:
        _write_inventory_summary(tmp_path / "inv.json")
        (tmp_path / "report.html").write_text("<html/>")
        view = scan_portfolio(tmp_path)
        html = format_portfolio_html(view)
        assert "HTML" in html
        assert "report.html" in html

    def test_custom_title_used(self, tmp_path: Path) -> None:
        view = PortfolioView(rows=(), scanned_dir=str(tmp_path), generated_at="2024-01-01 00:00 UTC")
        html = format_portfolio_html(view, title="My Custom Portfolio")
        assert "My Custom Portfolio" in html

    def test_badge_classes_present(self, tmp_path: Path) -> None:
        _write_inventory_summary(
            tmp_path / "inv.json",
            adapter_status_counts={"eligible": 2, "unknown_no_behavioral_eval": 1},
        )
        view = scan_portfolio(tmp_path)
        html = format_portfolio_html(view)
        assert "badge-partial" in html


# ---------------------------------------------------------------------------
# API wrapper test
# ---------------------------------------------------------------------------


def test_api_scan_portfolio(tmp_path: Path) -> None:
    _write_inventory_summary(tmp_path / "inv.json")
    from gradience.api import scan_portfolio as api_scan

    view = api_scan(tmp_path)
    assert len(view.rows) == 1


# ---------------------------------------------------------------------------
# CLI integration test
# ---------------------------------------------------------------------------


class TestCLIPortfolio:
    def test_cli_terminal_output(self, tmp_path: Path) -> None:
        _write_inventory_summary(tmp_path / "test_inv.json")
        result = subprocess.run(
            [sys.executable, "-m", "gradience.cli", "portfolio", str(tmp_path)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "test_inv" in result.stdout

    def test_cli_html_output(self, tmp_path: Path) -> None:
        _write_inventory_summary(tmp_path / "test_inv.json")
        out_html = tmp_path / "portfolio.html"
        result = subprocess.run(
            [
                sys.executable, "-m", "gradience.cli",
                "portfolio", str(tmp_path),
                "--html", str(out_html),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert out_html.exists()
        content = out_html.read_text()
        assert "<!DOCTYPE html>" in content
        assert "test_inv" in content

    def test_cli_nonexistent_dir_exits_nonzero(self, tmp_path: Path) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "gradience.cli", "portfolio", str(tmp_path / "nope")],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0

    def test_cli_empty_dir_exits_zero(self, tmp_path: Path) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "gradience.cli", "portfolio", str(tmp_path)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "No inventory summaries found" in result.stdout
