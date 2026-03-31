"""Tests for gradience.vnext.inventory.portfolio."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gradience.exceptions import QASchemaError
from gradience.vnext.inventory.portfolio import (
    PortfolioRow,
    PortfolioView,
    _derive_dominant_driver,
    _derive_evidence_profile,
    _derive_exploration_posture,
    format_portfolio_html,
    format_portfolio_table,
    scan_portfolio,
)

# ---------------------------------------------------------------------------
# _derive_dominant_driver
# ---------------------------------------------------------------------------


class TestDeriveDominantDriver:
    def test_empty(self) -> None:
        assert _derive_dominant_driver({}) == "\u2014"

    def test_none_only(self) -> None:
        assert _derive_dominant_driver({"none": 5}) == "none"

    def test_non_trivial_winner(self) -> None:
        result = _derive_dominant_driver({"none": 2, "norm_imbalance": 3, "unknown": 1})
        assert result == "norm_imbalance"

    def test_unknown_filtered(self) -> None:
        # "unknown" is in _NONE_ISSUES frozenset, so it is filtered out
        assert _derive_dominant_driver({"unknown": 5}) == "\u2014"


# ---------------------------------------------------------------------------
# _derive_exploration_posture
# ---------------------------------------------------------------------------


class TestDeriveExplorationPosture:
    def test_zero_total(self) -> None:
        assert _derive_exploration_posture({}) == "\u2014"

    def test_linear_dominant(self) -> None:
        assert _derive_exploration_posture({"linear": 5, "audit_aware": 1}) == "standard"

    def test_audit_aware_dominant(self) -> None:
        assert _derive_exploration_posture({"audit_aware": 5}) == "high-scrutiny"

    def test_unknown_strategy(self) -> None:
        # Strategies not in _POSTURE_MAP pass through as-is
        assert _derive_exploration_posture({"some_new_strategy": 3}) == "some_new_strategy"


# ---------------------------------------------------------------------------
# _derive_evidence_profile
# ---------------------------------------------------------------------------


class TestDeriveEvidenceProfile:
    def test_zero_total(self) -> None:
        assert _derive_evidence_profile({}) == "\u2014"

    def test_all_eligible(self) -> None:
        assert _derive_evidence_profile({"eligible": 5}) == "full"

    def test_all_unknown(self) -> None:
        assert _derive_evidence_profile({"unknown_no_behavioral_eval": 5}) == "blind"

    def test_mixed_with_eligible(self) -> None:
        assert _derive_evidence_profile({"eligible": 2, "flagged_weak": 1}) == "partial"

    def test_only_weak(self) -> None:
        assert _derive_evidence_profile({"flagged_weak": 3}) == "weak"


# ---------------------------------------------------------------------------
# Helpers — valid inventory summary dict
# ---------------------------------------------------------------------------


def _valid_summary_dict() -> dict:
    """Return a minimal valid ``gradience.inventory_summary/v1`` dict."""
    return {
        "schema": "gradience.inventory_summary/v1",
        "sources": {"qa_artifact_count": 2, "merge_report_count": 1},
        "adapter_status_counts": {"eligible": 2},
        "adapter_flag_counts": {},
        "pair_risk_counts": {"low": 1},
        "recommended_strategy_counts": {"linear": 1},
        "dominant_issue_counts": {"none": 1},
        "strict_qa_block_candidates": 0,
        "evidence_tier_counts": {"behavioral_reported": 2},
    }


# ---------------------------------------------------------------------------
# scan_portfolio
# ---------------------------------------------------------------------------


class TestScanPortfolio:
    def test_nonexistent_directory(self, tmp_path: Path) -> None:
        missing = tmp_path / "does_not_exist"
        with pytest.raises(ValueError, match="does not exist"):
            scan_portfolio(missing)

    def test_valid_inventory_found(self, tmp_path: Path) -> None:
        subdir = tmp_path / "inv_alpha"
        subdir.mkdir()
        (subdir / "preflight_summary.json").write_text(json.dumps(_valid_summary_dict()))

        view = scan_portfolio(tmp_path)

        assert len(view.rows) == 1
        row = view.rows[0]
        assert row.name == "inv_alpha"
        assert row.adapter_count == 2
        assert row.pair_count == 1
        assert row.retained_candidates == 2

    def test_strict_input_rejects_malformed(self, tmp_path: Path) -> None:
        subdir = tmp_path / "bad_inv"
        subdir.mkdir()
        # Write a file that has the right schema key but invalid structure
        (subdir / "summary.json").write_text(
            json.dumps({"schema": "gradience.inventory_summary/v1", "bad_key": True})
        )
        with pytest.raises(QASchemaError):
            scan_portfolio(tmp_path, strict_input=True)

    def test_strict_input_false_skips_malformed(self, tmp_path: Path) -> None:
        # Malformed inventory in one subdir
        bad_dir = tmp_path / "bad_inv"
        bad_dir.mkdir()
        (bad_dir / "summary.json").write_text(
            json.dumps({"schema": "gradience.inventory_summary/v1", "bad_key": True})
        )

        # Valid inventory in another subdir
        good_dir = tmp_path / "good_inv"
        good_dir.mkdir()
        (good_dir / "summary.json").write_text(json.dumps(_valid_summary_dict()))

        view = scan_portfolio(tmp_path, strict_input=False)

        assert len(view.rows) == 1
        assert view.rows[0].name == "good_inv"


# ---------------------------------------------------------------------------
# format_portfolio_table
# ---------------------------------------------------------------------------


def _make_row(**overrides: object) -> PortfolioRow:
    """Create a PortfolioRow with sensible defaults, accepting overrides."""
    defaults = dict(
        name="test_inv",
        timestamp="2025-01-15 10:30 UTC",
        adapter_count=2,
        pair_count=1,
        retained_candidates=2,
        reduction_pct=0.0,
        dominant_driver="none",
        exploration_posture="standard",
        evidence_profile="full",
        drift_status="\u2014",
        html_report_link=None,
        artifact_link="/tmp/test.json",
    )
    defaults.update(overrides)
    return PortfolioRow(**defaults)  # type: ignore[arg-type]


class TestFormatPortfolioTable:
    def test_empty_view(self) -> None:
        view = PortfolioView(
            rows=(),
            scanned_dir="/tmp/empty",
            generated_at="2025-01-15 10:30 UTC",
        )
        output = format_portfolio_table(view)
        assert "No inventory summaries found" in output

    def test_with_rows(self) -> None:
        row = _make_row(name="my_inventory")
        view = PortfolioView(
            rows=(row,),
            scanned_dir="/tmp/test",
            generated_at="2025-01-15 10:30 UTC",
        )
        output = format_portfolio_table(view)
        assert "my_inventory" in output


# ---------------------------------------------------------------------------
# format_portfolio_html
# ---------------------------------------------------------------------------


class TestFormatPortfolioHtml:
    def test_empty_view(self) -> None:
        view = PortfolioView(
            rows=(),
            scanned_dir="/tmp/empty",
            generated_at="2025-01-15 10:30 UTC",
        )
        html = format_portfolio_html(view)
        assert "No inventories found" in html

    def test_contains_title(self) -> None:
        row = _make_row(name="alpha")
        view = PortfolioView(
            rows=(row,),
            scanned_dir="/tmp/test",
            generated_at="2025-01-15 10:30 UTC",
        )
        html = format_portfolio_html(view, title="My Custom Title")
        assert "My Custom Title" in html
