"""Tests for gradience.vnext.inventory.batch — cross-run summary and discovery."""

from __future__ import annotations

import json
import time
from pathlib import Path

from gradience.vnext.inventory.batch import (
    build_batch_summary,
    collect_run_summaries,
    discover_previous_run,
    emit_batch_summary,
    format_batch_summary,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_summary(
    *,
    run_id: str = "run_001",
    timestamp: str = "2026-03-01T00:00:00Z",
    adapter_count: int = 4,
    pair_count: int = 6,
    retained_candidate_count: int = 3,
    advisory_pair_count: int = 1,
    behavioral_evidence_count: int = 2,
    total_source_count: int = 4,
    excluded_sources: list[str] | None = None,
    inventory_policy_summary: dict[str, str] | None = None,
) -> dict:
    d: dict = {
        "run_id": run_id,
        "timestamp": timestamp,
        "adapter_count": adapter_count,
        "pair_count": pair_count,
        "retained_candidate_count": retained_candidate_count,
        "advisory_pair_count": advisory_pair_count,
        "behavioral_evidence_count": behavioral_evidence_count,
        "total_source_count": total_source_count,
        "excluded_sources": excluded_sources or [],
    }
    if inventory_policy_summary is not None:
        d["inventory_policy_summary"] = inventory_policy_summary
    return d


def _write_summary(directory: Path, summary: dict) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "preflight_summary.json").write_text(json.dumps(summary))


# ---------------------------------------------------------------------------
# TestDiscoverPreviousRun
# ---------------------------------------------------------------------------


class TestDiscoverPreviousRun:
    def test_empty_dir(self, tmp_path: Path) -> None:
        assert discover_previous_run(tmp_path) is None

    def test_symlink_with_summary(self, tmp_path: Path) -> None:
        target = tmp_path / "run_001"
        _write_summary(target, _make_summary(run_id="run_001"))
        symlink = tmp_path / "latest"
        symlink.symlink_to(target)

        result = discover_previous_run(tmp_path)
        assert result is not None
        assert result == target.resolve()

    def test_no_symlink_mtime_fallback(self, tmp_path: Path) -> None:
        older = tmp_path / "run_old"
        _write_summary(older, _make_summary(run_id="run_old"))
        # Ensure different mtime
        time.sleep(0.05)
        newer = tmp_path / "run_new"
        _write_summary(newer, _make_summary(run_id="run_new"))

        result = discover_previous_run(tmp_path)
        assert result is not None
        assert result == newer

    def test_hidden_dirs_skipped(self, tmp_path: Path) -> None:
        hidden = tmp_path / ".hidden_run"
        _write_summary(hidden, _make_summary(run_id="hidden"))

        assert discover_previous_run(tmp_path) is None


# ---------------------------------------------------------------------------
# TestCollectRunSummaries
# ---------------------------------------------------------------------------


class TestCollectRunSummaries:
    def test_valid_json_collected(self, tmp_path: Path) -> None:
        _write_summary(
            tmp_path / "run_a",
            _make_summary(run_id="run_a", timestamp="2026-03-01T00:00:00Z"),
        )
        _write_summary(
            tmp_path / "run_b",
            _make_summary(run_id="run_b", timestamp="2026-03-02T00:00:00Z"),
        )

        results = collect_run_summaries(tmp_path)
        assert len(results) == 2
        # Sorted by timestamp (oldest first)
        assert results[0]["run_id"] == "run_a"
        assert results[1]["run_id"] == "run_b"

    def test_malformed_json_skipped(self, tmp_path: Path) -> None:
        good = tmp_path / "run_good"
        _write_summary(good, _make_summary(run_id="run_good"))
        bad = tmp_path / "run_bad"
        bad.mkdir()
        (bad / "preflight_summary.json").write_text("{invalid json")

        results = collect_run_summaries(tmp_path)
        assert len(results) == 1
        assert results[0]["run_id"] == "run_good"

    def test_missing_run_id_skipped(self, tmp_path: Path) -> None:
        no_id = tmp_path / "run_no_id"
        no_id.mkdir()
        (no_id / "preflight_summary.json").write_text(json.dumps({"adapter_count": 1}))

        good = tmp_path / "run_good"
        _write_summary(good, _make_summary(run_id="run_good"))

        results = collect_run_summaries(tmp_path)
        assert len(results) == 1
        assert results[0]["run_id"] == "run_good"


# ---------------------------------------------------------------------------
# TestBuildBatchSummary
# ---------------------------------------------------------------------------


class TestBuildBatchSummary:
    def test_fewer_than_two_runs(self) -> None:
        batch = build_batch_summary([_make_summary()])
        assert batch["trend"] == "insufficient data"

    def test_narrowing_trend(self) -> None:
        summaries = [
            _make_summary(run_id="r1", timestamp="2026-03-01", retained_candidate_count=10),
            _make_summary(run_id="r2", timestamp="2026-03-02", retained_candidate_count=5),
        ]
        batch = build_batch_summary(summaries)
        assert batch["trend"] == "narrowing"

    def test_broadening_trend(self) -> None:
        summaries = [
            _make_summary(run_id="r1", timestamp="2026-03-01", retained_candidate_count=5),
            _make_summary(run_id="r2", timestamp="2026-03-02", retained_candidate_count=10),
        ]
        batch = build_batch_summary(summaries)
        assert batch["trend"] == "broadening"

    def test_stable_trend(self) -> None:
        summaries = [
            _make_summary(run_id="r1", timestamp="2026-03-01", retained_candidate_count=5),
            _make_summary(run_id="r2", timestamp="2026-03-02", retained_candidate_count=5),
        ]
        batch = build_batch_summary(summaries)
        assert batch["trend"] == "stable"

    def test_evidence_trend_improved(self) -> None:
        summaries = [
            _make_summary(
                run_id="r1", timestamp="2026-03-01",
                behavioral_evidence_count=1, total_source_count=4,
            ),
            _make_summary(
                run_id="r2", timestamp="2026-03-02",
                behavioral_evidence_count=3, total_source_count=4,
            ),
        ]
        batch = build_batch_summary(summaries)
        assert batch["evidence_trend"] == "improved"

    def test_evidence_trend_degraded(self) -> None:
        summaries = [
            _make_summary(
                run_id="r1", timestamp="2026-03-01",
                behavioral_evidence_count=3, total_source_count=4,
            ),
            _make_summary(
                run_id="r2", timestamp="2026-03-02",
                behavioral_evidence_count=1, total_source_count=4,
            ),
        ]
        batch = build_batch_summary(summaries)
        assert batch["evidence_trend"] == "degraded"

    def test_zero_source_count(self) -> None:
        summaries = [
            _make_summary(
                run_id="r1", timestamp="2026-03-01",
                behavioral_evidence_count=0, total_source_count=0,
            ),
            _make_summary(
                run_id="r2", timestamp="2026-03-02",
                behavioral_evidence_count=0, total_source_count=0,
            ),
        ]
        batch = build_batch_summary(summaries)
        # evidence_ratio should be "n/a" for each row
        for row in batch["runs"]:
            assert row["evidence_ratio"] == "n/a"
        # When both total_source_count == 0, ratio comparison is skipped
        assert batch["evidence_trend"] == "unchanged"

    def test_policy_evolution_detected(self) -> None:
        policy_a = {"inventory_type": "focused", "dominant_driver": "risk", "exploration_posture": "narrow", "constraint": "strict"}
        policy_b = {"inventory_type": "broad", "dominant_driver": "coverage", "exploration_posture": "wide", "constraint": "relaxed"}
        summaries = [
            _make_summary(run_id="r1", timestamp="2026-03-01", inventory_policy_summary=policy_a),
            _make_summary(run_id="r2", timestamp="2026-03-02", inventory_policy_summary=policy_b),
        ]
        batch = build_batch_summary(summaries)
        assert batch["policy_changed"] is True


# ---------------------------------------------------------------------------
# TestFormatBatchSummary
# ---------------------------------------------------------------------------


class TestFormatBatchSummary:
    def test_empty_runs(self) -> None:
        batch = {"runs": [], "run_count": 0, "trend": "unknown"}
        result = format_batch_summary(batch)
        assert "No runs found" in result

    def test_format_with_runs(self) -> None:
        summaries = [
            _make_summary(run_id="r1", timestamp="2026-03-01"),
            _make_summary(run_id="r2", timestamp="2026-03-02"),
        ]
        batch = build_batch_summary(summaries)
        result = format_batch_summary(batch)
        assert "BATCH PREFLIGHT SUMMARY" in result


# ---------------------------------------------------------------------------
# TestEmitBatchSummary
# ---------------------------------------------------------------------------


class TestEmitBatchSummary:
    def test_writes_json_and_md(self, tmp_path: Path) -> None:
        summaries = [
            _make_summary(run_id="r1", timestamp="2026-03-01"),
            _make_summary(run_id="r2", timestamp="2026-03-02"),
        ]
        batch = build_batch_summary(summaries)
        emit_batch_summary(batch=batch, output_dir=tmp_path)

        assert (tmp_path / "batch_summary.json").exists()
        assert (tmp_path / "batch_summary.md").exists()

        # JSON should be valid
        with open(tmp_path / "batch_summary.json") as f:
            loaded = json.load(f)
        assert loaded["run_count"] == 2
