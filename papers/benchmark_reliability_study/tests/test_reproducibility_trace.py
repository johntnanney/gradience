"""Tests for 98_reproducibility_trace.py (SPEC §8.12, §13).

Exercises the trace generator end-to-end via subprocess against
synthesized paper-root layouts in tmp_path. Each test isolates one
behavior of the trace-status decision logic.
"""

from __future__ import annotations

import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from gradience_study.config import compute_config_hash, load_config


PAPER_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = PAPER_ROOT / "scripts"
FIXTURES_DIR = PAPER_ROOT / "tests" / "fixtures"
TRACE_SCRIPT = SCRIPTS_DIR / "98_reproducibility_trace.py"


CONDITION_FIELDNAMES = [
    "condition_id",
    "tier",
    "benchmark_id",
    "subject_id",
    "model_id",
    "prompt_id",
    "seed_id",
    "fewshot_k",
    "scoring_rule_id",
    "task_type",
    "expected_num_items",
    "condition_status",
    "created_at",
    "config_hash",
]

CONDITION_LEVEL_FIELDNAMES = [
    "condition_id", "tier", "benchmark_id", "subject_id", "model_id",
    "prompt_id", "seed_id", "fewshot_k", "scoring_rule_id",
    "num_items", "num_correct", "accuracy", "parseability_rate",
    "mean_generation_length_tokens", "condition_status",
]


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
    # Empty pipeline subdirs (consumers must tolerate missing).
    for sub in ("manifests", "runs/raw", "runs/normalized", "analysis", "reports"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    return root


def _run_trace(paper_root: Path, *, sample_n: int = 5,
               seed: int = 20260424,
               compare_env: Path | None = None) -> tuple[int, str, Path]:
    out = paper_root / "reports" / "reproducibility_trace.md"
    cmd = [
        sys.executable, str(TRACE_SCRIPT),
        "--config", str(paper_root / "configs" / "study_config.yaml"),
        "--manifests-dir", str(paper_root / "manifests"),
        "--raw-dir", str(paper_root / "runs" / "raw"),
        "--normalized-dir", str(paper_root / "runs" / "normalized"),
        "--analysis-dir", str(paper_root / "analysis"),
        "--sample-n", str(sample_n),
        "--seed", str(seed),
        "--out", str(out),
    ]
    if compare_env is not None:
        cmd.extend(["--compare-env", str(compare_env)])
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode, (out.read_text() if out.exists() else ""), out


def _write_conditions_primary(
    path: Path,
    rows: list[dict],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CONDITION_FIELDNAMES, lineterminator="\n")
        w.writeheader()
        for r in rows:
            full = {k: "" for k in CONDITION_FIELDNAMES}
            full.update(r)
            w.writerow(full)


def _write_condition_level(
    path: Path,
    rows: list[dict],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CONDITION_LEVEL_FIELDNAMES,
                           lineterminator="\n")
        w.writeheader()
        for r in rows:
            full = {k: "" for k in CONDITION_LEVEL_FIELDNAMES}
            full.update(r)
            w.writerow(full)


def _gp_condition_id() -> str:
    return "arc_challenge__pythia_410m__P1_original__s42__generate_parse"


def _make_complete_pipeline(paper_root: Path,
                            stored_accuracy: float = 0.5) -> str:
    """Set up: 1 complete condition + raw outputs + condition-level CSV.

    Uses the existing G&P fixture (4 items, 2 correct → recomputed accuracy
    = 0.5). The stored-accuracy parameter lets us inject a synthetic
    mismatch by pinning a different number.
    """
    cid = _gp_condition_id()
    # Copy raw run dir into runs/raw/<cid>/.
    src = FIXTURES_DIR / "raw" / cid
    dst = paper_root / "runs" / "raw" / cid
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)

    # Manifest row marking this condition complete.
    _write_conditions_primary(
        paper_root / "manifests" / "conditions_primary.csv",
        [{
            "condition_id": cid,
            "tier": "primary",
            "benchmark_id": "arc_challenge",
            "subject_id": "",
            "model_id": "pythia_410m",
            "prompt_id": "P1_original",
            "seed_id": "42",
            "fewshot_k": 5,
            "scoring_rule_id": "generate_parse",
            "task_type": "constrained_choice",
            "expected_num_items": 4,
            "condition_status": "complete",
            "created_at": "2026-04-24T00:00:00+00:00",
            "config_hash": "deadbeef",
        }],
    )
    # Condition-level CSV with the chosen stored accuracy.
    _write_condition_level(
        paper_root / "runs" / "normalized" / "condition_level_primary.csv",
        [{
            "condition_id": cid,
            "tier": "primary",
            "benchmark_id": "arc_challenge",
            "subject_id": "",
            "model_id": "pythia_410m",
            "prompt_id": "P1_original",
            "seed_id": "42",
            "fewshot_k": 5,
            "scoring_rule_id": "generate_parse",
            "num_items": 4,
            "num_correct": 2,
            "accuracy": stored_accuracy,
            "parseability_rate": 1.0,
            "mean_generation_length_tokens": 5.0,
            "condition_status": "complete",
        }],
    )
    return cid


# -- Tests -------------------------------------------------------------------


def test_trace_runs_with_empty_pipeline(paper_root):
    """Brand-new pipeline (configs only) — trace status pass, exit 0."""
    rc, md, _ = _run_trace(paper_root)
    assert rc == 0, f"Expected exit 0, got {rc}. Trace:\n{md}"
    assert "Trace status: `pass`" in md
    # Most sections should report "not yet runnable" or similar.
    assert "not yet runnable" in md or "missing" in md
    # Config hash should be present (configs loaded successfully).
    assert "`config_hash_full`" in md


def test_trace_detects_manifest_row_mismatch(paper_root):
    """Manifest says complete but no raw_dir → flagged in Section 3 → fail."""
    cid = _gp_condition_id()
    # Manifest says complete...
    _write_conditions_primary(
        paper_root / "manifests" / "conditions_primary.csv",
        [{
            "condition_id": cid,
            "tier": "primary",
            "benchmark_id": "arc_challenge",
            "subject_id": "",
            "model_id": "pythia_410m",
            "prompt_id": "P1_original",
            "seed_id": "42",
            "fewshot_k": 5,
            "scoring_rule_id": "generate_parse",
            "task_type": "constrained_choice",
            "expected_num_items": 4,
            "condition_status": "complete",
            "created_at": "2026-04-24T00:00:00+00:00",
            "config_hash": "deadbeef",
        }],
    )
    # ...but a different raw_dir exists (so raw_subdir_count > 0 and the
    # mismatch lists trigger).
    extra = paper_root / "runs" / "raw" / "ghost_condition__model__P1__s42__rule"
    extra.mkdir(parents=True)
    (extra / "run_metadata.json").write_text("{}", encoding="utf-8")

    rc, md, _ = _run_trace(paper_root)
    # Failure expected because the manifest's complete cid is missing on disk
    # AND there's a raw dir without a matching manifest row.
    assert rc == 5, f"Expected exit 5 (fail), got {rc}. Trace:\n{md}"
    assert "Trace status: `fail`" in md
    assert "Manifest says complete" in md
    assert "without a matching manifest row" in md


def test_trace_recompute_sample_matches(paper_root):
    """Item-level + condition-level consistent: recompute sample passes."""
    _make_complete_pipeline(paper_root, stored_accuracy=0.5)
    rc, md, _ = _run_trace(paper_root, sample_n=1)
    assert rc == 0, f"Expected pass, got {rc}. Trace:\n{md}"
    assert "Trace status: `pass`" in md
    # The recompute table should show pass status.
    assert "| pass |" in md
    # And the recomputed value should be present.
    assert "0.500000" in md


def test_trace_recompute_sample_detects_mismatch(paper_root):
    """Inject a synthetic mismatch: stored accuracy != recomputed → fail, exit 5."""
    _make_complete_pipeline(paper_root, stored_accuracy=0.99)
    rc, md, _ = _run_trace(paper_root, sample_n=1)
    assert rc == 5, f"Expected exit 5 (fail), got {rc}. Trace:\n{md}"
    assert "Trace status: `fail`" in md
    # Failed row in the recompute table.
    assert "| fail |" in md


def test_trace_config_hash_present(paper_root):
    """Config hash in trace matches compute_config_hash() output."""
    rc, md, _ = _run_trace(paper_root)
    assert rc == 0, md
    merged = load_config(paper_root / "configs" / "study_config.yaml")
    full, short = compute_config_hash(merged.raw_dict)
    assert full in md
    assert short in md


def test_trace_summary_section_consistent(paper_root):
    """Summary failure count equals sum of per-section failures (= 0 here)."""
    rc, md, _ = _run_trace(paper_root)
    assert rc == 0
    # On an empty pipeline, total failures = 0 across all sections.
    assert "Total reproducibility failures: 0" in md


def test_trace_compare_env_check(paper_root, tmp_path):
    """--compare-env activates Section 6 with byte-level diff."""
    # Set up identical analysis dirs for self-comparison: copy the mock
    # analysis tree into both the local and the compare-env locations.
    for sub in ("variance_components", "tolerance_schedules", "mmlu_subjects"):
        src = FIXTURES_DIR / "mock_analysis" / sub
        if src.exists():
            dst_local = paper_root / "analysis" / sub
            dst_local.mkdir(parents=True, exist_ok=True)
            for f in src.iterdir():
                if f.is_file():
                    shutil.copy(f, dst_local / f.name)

    compare_env = tmp_path / "env_b"
    shutil.copytree(paper_root / "analysis", compare_env)

    rc, md, _ = _run_trace(paper_root, compare_env=compare_env)
    # Self-comparison: every artifact byte-equal → pass overall.
    # (analysis re-derivation may report "not yet implemented" since
    # 06_/07_ scripts are absent; that's not a failure under graceful
    # degradation, so trace status remains pass.)
    assert rc == 0, f"Expected pass, got {rc}. Trace:\n{md}"
    assert "compare-env" in md


def test_trace_compare_env_detects_mismatch(paper_root, tmp_path):
    """If compare-env analysis differs, cross-env section flags failure."""
    src = FIXTURES_DIR / "mock_analysis" / "tolerance_schedules" / "h1_test.json"
    if not src.exists():
        pytest.skip("mock h1_test.json not available")
    local_h1 = paper_root / "analysis" / "tolerance_schedules"
    local_h1.mkdir(parents=True, exist_ok=True)
    shutil.copy(src, local_h1 / "h1_test.json")

    compare_env = tmp_path / "env_b"
    (compare_env / "tolerance_schedules").mkdir(parents=True)
    # Write a divergent h1_test.json.
    altered = json.loads(src.read_text(encoding="utf-8"))
    altered["h1_confirmed"] = not altered["h1_confirmed"]
    altered["benchmarks_exceeding"] = []
    (compare_env / "tolerance_schedules" / "h1_test.json").write_text(
        json.dumps(altered, indent=2, sort_keys=True), encoding="utf-8"
    )

    rc, md, _ = _run_trace(paper_root, compare_env=compare_env)
    assert rc == 5, f"Expected exit 5 from cross-env mismatch, got {rc}. Trace:\n{md}"
    assert "Trace status: `fail`" in md
