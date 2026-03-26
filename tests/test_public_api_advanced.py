"""Tests for advanced public API wrappers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gradience.api import (
    compute_core_space_diagnostic,
    merge_risk_report,
    suggest_neighborhoods,
)


def test_compute_core_space_diagnostic_self_pair() -> None:
    tiny = Path("examples/adapters/tiny_lora")
    diag = compute_core_space_diagnostic(adapter_a=tiny, adapter_b=tiny)

    assert diag is not None
    assert 0.0 <= float(diag.shared_basis_score) <= 1.0
    assert float(diag.basis_distortion) >= 0.0
    assert int(diag.effective_shared_rank) >= 0
    assert diag.status in {"compatible", "marginal", "incompatible", "not_applicable"}


def test_merge_risk_report_compute_core_space_optional_block() -> None:
    tiny = Path("examples/adapters/tiny_lora")
    report = merge_risk_report(
        adapter_a=tiny,
        adapter_b=tiny,
        compute_core_space=True,
    )
    assert report.core_space is not None
    assert report.core_space.status in {"compatible", "marginal", "incompatible", "not_applicable"}


def test_suggest_neighborhoods_from_fixture_inventory() -> None:
    fixture_root = Path("examples/inventories/inventory_with_weak_sources")
    report = suggest_neighborhoods(
        qa_dir=fixture_root / "qa",
        report_dir=fixture_root / "reports",
    )

    assert report.to_dict()["schema"] == "gradience.merge_neighborhoods/v1"
    assert any(set(group.members) == {"weak_a", "weak_b"} for group in report.groups)
    assert any(item.adapter == "weak_flagged" and item.reason == "flagged_weak" for item in report.excluded)


def test_suggest_neighborhoods_strict_input_behavior(tmp_path: Path) -> None:
    qa_dir = tmp_path / "qa"
    qa_dir.mkdir(parents=True, exist_ok=True)
    (qa_dir / "bad.json").write_text("{not-valid-json", encoding="utf-8")

    report_dir = Path("examples/inventories/inventory_safe_small/reports")

    # Non-strict mode skips bad JSON and still returns a report from merge reports.
    report = suggest_neighborhoods(
        qa_dir=qa_dir,
        report_dir=report_dir,
        strict_input=False,
    )
    assert report.to_dict()["schema"] == "gradience.merge_neighborhoods/v1"

    # Strict mode fails on malformed inputs.
    with pytest.raises(json.JSONDecodeError):
        suggest_neighborhoods(
            qa_dir=qa_dir,
            report_dir=report_dir,
            strict_input=True,
        )


def test_suggest_neighborhoods_requires_reports_source() -> None:
    with pytest.raises(ValueError, match="report_dir or report_paths"):
        suggest_neighborhoods(qa_paths=[Path("examples/qa/eligible_adapter_qa.json")])


def test_suggest_neighborhoods_report_paths_mode(tmp_path: Path) -> None:
    report_src = Path("examples/inventories/inventory_safe_small/reports/safe_ab.json")
    report_payload = json.loads(report_src.read_text(encoding="utf-8"))
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report_payload), encoding="utf-8")

    report = suggest_neighborhoods(report_paths=[report_path])
    assert report.to_dict()["schema"] == "gradience.merge_neighborhoods/v1"
