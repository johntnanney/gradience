"""Tests for gradience.vnext.inventory.html_report."""

from __future__ import annotations

import json
from pathlib import Path

from gradience.vnext.inventory.html_report import _e, render_html_report

# ---------------------------------------------------------------------------
# Helpers — minimal bundle data
# ---------------------------------------------------------------------------


def _minimal_preflight_summary() -> dict:
    """Return a minimal preflight_summary.json payload."""
    return {
        "inventory_id": "test_inv",
        "run_id": "test_run_001",
        "adapter_count": 2,
        "pair_count": 1,
        "retained_candidate_count": 1,
        "reduction_ratio": 0.5,
        "advisory_pair_count": 0,
        "qa_summary": {"eligible": 2},
        "behavioral_evidence_count": 2,
        "total_source_count": 2,
        "excluded_sources": [],
        "same_task_safe_pairs": ["a x b"],
        "cross_task_pairs": [],
        "evaluate_first": ["a x b"],
    }


# ---------------------------------------------------------------------------
# render_html_report
# ---------------------------------------------------------------------------


class TestRenderHtmlReport:
    def test_minimal_bundle(self, tmp_path: Path) -> None:
        (tmp_path / "preflight_summary.json").write_text(
            json.dumps(_minimal_preflight_summary())
        )
        html = render_html_report(tmp_path)
        assert "<!DOCTYPE html>" in html

    def test_custom_title(self, tmp_path: Path) -> None:
        (tmp_path / "preflight_summary.json").write_text(
            json.dumps(_minimal_preflight_summary())
        )
        html = render_html_report(tmp_path, title="My Custom Title")
        assert "My Custom Title" in html

    def test_missing_all_files(self, tmp_path: Path) -> None:
        # Empty directory — should degrade gracefully, not crash
        html = render_html_report(tmp_path)
        assert "<!DOCTYPE html>" in html

    def test_full_bundle(self, tmp_path: Path) -> None:
        (tmp_path / "preflight_summary.json").write_text(
            json.dumps(_minimal_preflight_summary())
        )
        manifest = {
            "run_id": "test_run_001",
            "generated_at": "2025-01-15T10:30:00Z",
            "qa_artifacts": ["a.json", "b.json"],
            "merge_reports": ["ab.json"],
        }
        (tmp_path / "run_manifest.json").write_text(json.dumps(manifest))

        html = render_html_report(tmp_path)
        # Both the summary and manifest content should be reflected
        assert "<!DOCTYPE html>" in html
        assert "test_run_001" in html


# ---------------------------------------------------------------------------
# _e helper
# ---------------------------------------------------------------------------


class TestHtmlHelpers:
    def test_html_escape(self) -> None:
        result = _e("<script>")
        assert "&lt;" in result
        assert "<script>" not in result

    def test_none_input(self) -> None:
        # _e coerces via str(), so None becomes the string "None"
        result = _e(None)
        assert isinstance(result, str)

    def test_empty_string(self) -> None:
        result = _e("")
        assert result == ""

    def test_ampersand_escape(self) -> None:
        result = _e("a & b")
        assert "&amp;" in result
