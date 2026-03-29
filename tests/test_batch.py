"""Tests for batch preflight ergonomics (Project E).

Tests auto-discovery of previous runs, cross-run summary collection,
batch summary building, formatting, and emission.
"""

from __future__ import annotations

import json
from pathlib import Path

from gradience.vnext.inventory.batch import (
    build_batch_summary,
    collect_run_summaries,
    discover_previous_run,
    emit_batch_summary,
    format_batch_summary,
)


def _write_summary(run_dir: Path, run_id: str, **overrides) -> Path:
    """Write a minimal preflight_summary.json to a run directory."""
    run_dir.mkdir(parents=True, exist_ok=True)
    data = {
        "run_id": run_id,
        "timestamp": f"2026-03-26T00:00:{run_id[-2:]}Z",
        "adapter_count": 4,
        "pair_count": 6,
        "retained_candidate_count": 2,
        "reduction_ratio": 0.667,
        "advisory_pair_count": 4,
        "behavioral_evidence_count": 3,
        "total_source_count": 4,
    }
    data.update(overrides)
    with open(run_dir / "preflight_summary.json", "w") as f:
        json.dump(data, f)
    return run_dir


# ---------------------------------------------------------------------------
# discover_previous_run
# ---------------------------------------------------------------------------


class TestDiscoverPreviousRun:
    def test_returns_none_for_empty_dir(self, tmp_path):
        assert discover_previous_run(tmp_path) is None

    def test_finds_latest_symlink(self, tmp_path):
        r1 = _write_summary(tmp_path / "run_01", "run_01")
        latest = tmp_path / "latest"
        latest.symlink_to(r1.resolve(), target_is_directory=True)
        result = discover_previous_run(tmp_path)
        assert result is not None
        assert result.resolve() == r1.resolve()

    def test_finds_most_recent_sibling(self, tmp_path):
        _write_summary(tmp_path / "run_01", "run_01")
        r2 = _write_summary(tmp_path / "run_02", "run_02")
        # Touch r2's summary to make it newer
        (r2 / "preflight_summary.json").touch()
        result = discover_previous_run(tmp_path)
        assert result is not None
        assert result.resolve() == r2.resolve()

    def test_ignores_dirs_without_summary(self, tmp_path):
        (tmp_path / "empty_dir").mkdir()
        r1 = _write_summary(tmp_path / "run_01", "run_01")
        result = discover_previous_run(tmp_path)
        assert result is not None
        assert result.resolve() == r1.resolve()

    def test_returns_none_for_nonexistent_dir(self, tmp_path):
        assert discover_previous_run(tmp_path / "no_such_dir") is None


# ---------------------------------------------------------------------------
# collect_run_summaries
# ---------------------------------------------------------------------------


class TestCollectRunSummaries:
    def test_collects_all_valid_runs(self, tmp_path):
        _write_summary(tmp_path / "run_01", "run_01")
        _write_summary(tmp_path / "run_02", "run_02")
        _write_summary(tmp_path / "run_03", "run_03")
        result = collect_run_summaries(tmp_path)
        assert len(result) == 3
        assert result[0]["run_id"] == "run_01"
        assert result[2]["run_id"] == "run_03"

    def test_skips_invalid_json(self, tmp_path):
        _write_summary(tmp_path / "run_01", "run_01")
        bad = tmp_path / "bad_run"
        bad.mkdir()
        (bad / "preflight_summary.json").write_text("not json")
        result = collect_run_summaries(tmp_path)
        assert len(result) == 1

    def test_skips_dirs_without_run_id(self, tmp_path):
        _write_summary(tmp_path / "run_01", "run_01")
        no_id = tmp_path / "no_id_run"
        no_id.mkdir()
        with open(no_id / "preflight_summary.json", "w") as f:
            json.dump({"adapter_count": 1}, f)
        result = collect_run_summaries(tmp_path)
        assert len(result) == 1

    def test_empty_dir_returns_empty(self, tmp_path):
        assert collect_run_summaries(tmp_path) == []

    def test_nonexistent_dir_returns_empty(self, tmp_path):
        assert collect_run_summaries(tmp_path / "nope") == []


# ---------------------------------------------------------------------------
# build_batch_summary
# ---------------------------------------------------------------------------


class TestBuildBatchSummary:
    def test_basic_structure(self, tmp_path):
        _write_summary(tmp_path / "r1", "r1")
        _write_summary(tmp_path / "r2", "r2", retained_candidate_count=3)
        summaries = collect_run_summaries(tmp_path)
        batch = build_batch_summary(summaries)
        assert batch["schema"] == "gradience.batch_summary/v1"
        assert batch["run_count"] == 2
        assert len(batch["runs"]) == 2

    def test_trend_narrowing(self):
        s1 = {"run_id": "r1", "retained_candidate_count": 5}
        s2 = {"run_id": "r2", "retained_candidate_count": 2}
        batch = build_batch_summary([s1, s2])
        assert batch["trend"] == "narrowing"

    def test_trend_broadening(self):
        s1 = {"run_id": "r1", "retained_candidate_count": 2}
        s2 = {"run_id": "r2", "retained_candidate_count": 5}
        batch = build_batch_summary([s1, s2])
        assert batch["trend"] == "broadening"

    def test_trend_stable(self):
        s1 = {"run_id": "r1", "retained_candidate_count": 3}
        s2 = {"run_id": "r2", "retained_candidate_count": 3}
        batch = build_batch_summary([s1, s2])
        assert batch["trend"] == "stable"

    def test_trend_insufficient_single_run(self):
        batch = build_batch_summary([{"run_id": "r1"}])
        assert batch["trend"] == "insufficient data"

    def test_evidence_ratio(self):
        s = {"run_id": "r1", "behavioral_evidence_count": 3, "total_source_count": 4}
        batch = build_batch_summary([s])
        assert batch["runs"][0]["evidence_ratio"] == "3/4"

    def test_evidence_ratio_na_when_missing(self):
        batch = build_batch_summary([{"run_id": "r1"}])
        assert batch["runs"][0]["evidence_ratio"] == "n/a"


# ---------------------------------------------------------------------------
# format_batch_summary
# ---------------------------------------------------------------------------


class TestFormatBatchSummary:
    def test_header_present(self):
        batch = build_batch_summary([{"run_id": "r1", "retained_candidate_count": 2}])
        text = format_batch_summary(batch)
        assert "BATCH PREFLIGHT SUMMARY" in text

    def test_trend_shown(self):
        batch = {"run_count": 1, "trend": "stable", "runs": [{"run_id": "r1"}]}
        text = format_batch_summary(batch)
        assert "stable" in text

    def test_empty_runs(self):
        text = format_batch_summary({"runs": [], "run_count": 0, "trend": "none"})
        assert "No runs found" in text


# ---------------------------------------------------------------------------
# emit_batch_summary
# ---------------------------------------------------------------------------


class TestEmitBatchSummary:
    def test_creates_both_files(self, tmp_path):
        batch = build_batch_summary([
            {"run_id": "r1", "adapter_count": 3, "retained_candidate_count": 2},
        ])
        out = tmp_path / "output"
        emit_batch_summary(batch=batch, output_dir=out)
        assert (out / "batch_summary.json").exists()
        assert (out / "batch_summary.md").exists()

    def test_json_is_parseable(self, tmp_path):
        batch = build_batch_summary([
            {"run_id": "r1", "adapter_count": 3, "retained_candidate_count": 2},
        ])
        out = tmp_path / "output"
        emit_batch_summary(batch=batch, output_dir=out)
        with open(out / "batch_summary.json") as f:
            loaded = json.load(f)
        assert loaded["run_count"] == 1

    def test_md_contains_table(self, tmp_path):
        batch = build_batch_summary([
            {"run_id": "r1", "adapter_count": 3, "retained_candidate_count": 2},
            {"run_id": "r2", "adapter_count": 4, "retained_candidate_count": 1},
        ])
        out = tmp_path / "output"
        emit_batch_summary(batch=batch, output_dir=out)
        md = (out / "batch_summary.md").read_text()
        assert "# Batch Preflight Summary" in md
        assert "| r1" in md
        assert "| r2" in md
        assert "Trend" in md
