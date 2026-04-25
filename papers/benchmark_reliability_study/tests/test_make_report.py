"""Tests for 99_make_report.py (SPEC §8.13).

Exercises the report assembler end-to-end via subprocess against
synthesized paper-root layouts in tmp_path. Each test isolates one
artifact-presence/absence behavior of the report.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


PAPER_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = PAPER_ROOT / "scripts"
FIXTURES_DIR = PAPER_ROOT / "tests" / "fixtures"
REPORT_SCRIPT = SCRIPTS_DIR / "99_make_report.py"


# -- Fixture build -----------------------------------------------------------


@pytest.fixture(scope="session", autouse=True)
def ensure_fixtures_built():
    cfg = FIXTURES_DIR / "configs" / "study_config.yaml"
    if not cfg.exists():
        result = subprocess.run(
            [sys.executable, str(FIXTURES_DIR / "make_fixtures.py")],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, f"Fixture build failed: {result.stderr}"


# -- Layout helpers ----------------------------------------------------------


@pytest.fixture
def paper_root(tmp_path):
    """Construct an empty paper-root with copied fixture configs."""
    root = tmp_path / "paper_root"
    (root / "configs").mkdir(parents=True)
    for src in (FIXTURES_DIR / "configs").glob("*.yaml"):
        shutil.copy(src, root / "configs" / src.name)
    for sub in ("manifests", "runs/raw", "runs/normalized",
                "analysis", "tables", "figures", "reports"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    return root


def _run_report(paper_root: Path,
                with_config: bool = True,
                with_manifests: bool = True,
                with_reports: bool = True) -> tuple[int, str, Path]:
    out = paper_root / "reports" / "cpu_pipeline_report.md"
    cmd = [
        sys.executable, str(REPORT_SCRIPT),
        "--analysis-dir", str(paper_root / "analysis"),
        "--tables-dir", str(paper_root / "tables"),
        "--figures-dir", str(paper_root / "figures"),
        "--out", str(out),
    ]
    if with_config:
        cmd.extend(["--config", str(paper_root / "configs" / "study_config.yaml")])
    if with_manifests:
        cmd.extend(["--manifests-dir", str(paper_root / "manifests")])
    if with_reports:
        cmd.extend(["--reports-dir", str(paper_root / "reports")])
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode, (out.read_text() if out.exists() else ""), out


# -- Tests -------------------------------------------------------------------


def test_report_runs_with_only_configs(paper_root):
    """All optional artifacts missing — report generated with 'not yet
    available' annotations and exit 0."""
    rc, md, _ = _run_report(paper_root)
    assert rc == 0, f"Expected exit 0, got {rc}. Output:\n{md}"
    # All five sections should appear by title.
    for title in (
        "1. Pipeline execution summary",
        "2. Headline numbers",
        "3. Artifact links",
        "4. Deviations",
        "5. Reproducibility trace status",
    ):
        assert title in md, f"Missing section title: {title}"
    assert "not yet available" in md
    assert "No deviations recorded." in md
    assert "Reproducibility trace not yet generated." in md


def test_report_links_present_artifacts(paper_root):
    """Place mock files in analysis/tables/figures dirs; confirm the report
    links them all."""
    # Drop a couple of mock files.
    (paper_root / "analysis" / "variance_components").mkdir(parents=True, exist_ok=True)
    (paper_root / "analysis" / "variance_components" / "aggregate_vc.csv").write_text(
        "benchmark_id,model_id,source\nb1,m1,prompt\n", encoding="utf-8"
    )
    (paper_root / "tables" / "table_4_tolerance_schedule.md").write_text(
        "# Table 4\n", encoding="utf-8"
    )
    (paper_root / "figures" / "fig_2_tolerance_by_benchmark.png").write_bytes(
        b"\x89PNG\r\n\x1a\n"
    )

    rc, md, _ = _run_report(paper_root)
    assert rc == 0, md
    assert "aggregate_vc.csv" in md
    assert "table_4_tolerance_schedule.md" in md
    assert "fig_2_tolerance_by_benchmark.png" in md


def test_report_includes_h1_when_present(paper_root):
    """Place mock h1_test.json with h1_confirmed: true; report shows decision."""
    h1_dir = paper_root / "analysis" / "tolerance_schedules"
    h1_dir.mkdir(parents=True, exist_ok=True)
    (h1_dir / "h1_test.json").write_text(json.dumps({
        "h1_hypothesis": "test hypothesis",
        "threshold": 0.005,
        "n_required": 3,
        "n_exceeding": 4,
        "benchmarks_exceeding": ["bench_a", "bench_b", "bench_c", "bench_d"],
        "h1_confirmed": True,
    }), encoding="utf-8")

    rc, md, _ = _run_report(paper_root)
    assert rc == 0, md
    assert "**CONFIRMED**" in md
    assert "bench_a" in md
    assert "Threshold: 0.005" in md


def test_report_embeds_deviations_when_present(paper_root):
    """deviations.md content should be embedded inline."""
    body = (
        "# Deviations log\n\n"
        "## DEV-001: prompt P3 swap\n\n"
        "Rationale: source URL became unavailable; switched to "
        "Wayback snapshot.\n"
    )
    (paper_root / "reports" / "deviations.md").write_text(body, encoding="utf-8")
    rc, md, _ = _run_report(paper_root)
    assert rc == 0, md
    assert "DEV-001: prompt P3 swap" in md
    assert "Wayback snapshot" in md
    # Header contributed by the section renderer should still be there.
    assert "4. Deviations" in md


def test_report_reproducibility_status_when_present(paper_root):
    """When reproducibility_trace.md exists, report reflects its status."""
    trace_path = paper_root / "reports" / "reproducibility_trace.md"
    trace_path.write_text(
        "# Reproducibility Trace\n\n"
        "## 1. Config state\n\n"
        "- foo\n\n"
        "## 7. Summary\n\n"
        "- Total reproducibility-critical artifacts checked: 12\n"
        "- Total reproducibility failures: 0\n"
        "- **Trace status: `pass`**\n",
        encoding="utf-8",
    )
    rc, md, _ = _run_report(paper_root)
    assert rc == 0, md
    assert "Trace status:" in md
    # The status word "pass" should appear in the embedded summary excerpt.
    assert "`pass`" in md
    assert "Total reproducibility-critical artifacts checked: 12" in md


def test_report_reflects_failed_repro_status(paper_root):
    """A failing trace should be reflected verbatim."""
    trace_path = paper_root / "reports" / "reproducibility_trace.md"
    trace_path.write_text(
        "# Reproducibility Trace\n\n"
        "## 7. Summary\n\n"
        "- Trace status: `fail`\n",
        encoding="utf-8",
    )
    rc, md, _ = _run_report(paper_root)
    assert rc == 0, md
    assert "**`fail`**" in md or "`fail`" in md
