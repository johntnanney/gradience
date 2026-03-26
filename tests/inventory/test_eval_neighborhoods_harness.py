"""Integration tests for scripts/eval_neighborhoods.py."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _latest_run_dir(out_root: Path) -> Path:
    runs = sorted([p for p in out_root.iterdir() if p.is_dir()])
    assert runs, f"No run directories found in {out_root}"
    return runs[-1]


class TestNeighborhoodEvalHarness:
    def test_runs_all_fixtures_and_emits_bundle(self, tmp_path: Path) -> None:
        out_root = tmp_path / "eval_out"
        p = subprocess.run(
            [
                sys.executable,
                "scripts/eval_neighborhoods.py",
                "--fixtures-root",
                "examples/inventories",
                "--out-root",
                str(out_root),
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert p.returncode == 0, p.stderr
        run_dir = _latest_run_dir(out_root)
        summary_json = run_dir / "summary.json"
        summary_md = run_dir / "summary.md"
        assert summary_json.exists()
        assert summary_md.exists()

        with open(summary_json) as f:
            summary = json.load(f)

        fixtures = summary["fixtures"]
        fixture_names = {f["fixture"] for f in fixtures}
        assert "inventory_safe_small" in fixture_names
        assert "inventory_mixed_small" in fixture_names
        assert "inventory_fragmented_small" in fixture_names
        assert "inventory_with_weak_sources" in fixture_names
        assert "inventory_large_realistic" in fixture_names

        # All fixed fixtures in this baseline should pass as authored.
        for row in fixtures:
            assert row["verdict"] == "passed", row

        # Each fixture bundle should contain the required outputs.
        for name in fixture_names:
            fixture_dir = run_dir / name
            assert (fixture_dir / "neighborhood_report.json").exists()
            assert (fixture_dir / "terminal_summary.txt").exists()
            assert (fixture_dir / "comparison.json").exists()
            assert (fixture_dir / "verdict.txt").exists()
            assert (fixture_dir / "expected_notes.md").exists()

    def test_fixture_subset_option(self, tmp_path: Path) -> None:
        out_root = tmp_path / "subset_out"
        p = subprocess.run(
            [
                sys.executable,
                "scripts/eval_neighborhoods.py",
                "--fixtures-root",
                "examples/inventories",
                "--out-root",
                str(out_root),
                "--fixture",
                "inventory_safe_small",
                "--fixture",
                "inventory_with_weak_sources",
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert p.returncode == 0, p.stderr
        run_dir = _latest_run_dir(out_root)
        with open(run_dir / "summary.json") as f:
            summary = json.load(f)
        names = sorted([row["fixture"] for row in summary["fixtures"]])
        assert names == ["inventory_safe_small", "inventory_with_weak_sources"]
