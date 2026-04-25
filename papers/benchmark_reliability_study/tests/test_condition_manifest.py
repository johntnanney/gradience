"""Tests for 01_build_manifests.py — SPEC_CPU_v0_2 §8.2.

Eight tests cover:
    1. Determinism: byte-identical CSVs across two runs with --fixed-timestamp.
    2. Nominal condition count for the fixture configs.
    3. Secondary tier emits header-only when there are no secondary benchmarks.
    4. Condition ID scheme matches SPEC §7.2.
    5. Every row's config_hash is an 8-char hex string.
    6. Every row's condition_status == "pending".
    7. fewshot_k == 0 benchmark uses seed_id == "0shot".
    8. scoring_manifest.csv contains the expected scoring rules.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


PAPER_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = PAPER_ROOT / "scripts"
FIXTURES_DIR = Path(__file__).parent / "fixtures"
SCRIPT_PATH = SCRIPTS_DIR / "01_build_manifests.py"

FIXED_TIMESTAMP = "2026-04-24T00:00:00Z"


# -- Module loader (script filename starts with a digit) ---------------------


def _load_script_module():
    spec = importlib.util.spec_from_file_location(
        "build_manifests", SCRIPT_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


build_module = _load_script_module()
CONDITION_FIELDNAMES = build_module.CONDITION_FIELDNAMES
SCORING_FIELDNAMES = build_module.SCORING_FIELDNAMES


# -- Fixtures ---------------------------------------------------------------


@pytest.fixture(scope="session", autouse=True)
def ensure_fixtures_built():
    """Build fixture configs if missing. Idempotent."""
    cfg_dir = FIXTURES_DIR / "configs"
    if not (cfg_dir / "study_config.yaml").exists():
        result = subprocess.run(
            [sys.executable, str(FIXTURES_DIR / "make_fixtures.py")],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"Fixture build failed: {result.stderr}"


@pytest.fixture
def paper_root(tmp_path):
    """A minimal paper-root layout copying the fixture configs."""
    root = tmp_path / "paper_root"
    (root / "configs").mkdir(parents=True)
    for src in (FIXTURES_DIR / "configs").glob("*.yaml"):
        shutil.copy(src, root / "configs" / src.name)
    (root / "manifests").mkdir()
    return root


def _run_script(paper_root: Path, out_dir: Path) -> subprocess.CompletedProcess:
    out_dir.mkdir(parents=True, exist_ok=True)
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--config",
            str(paper_root / "configs" / "study_config.yaml"),
            "--out-dir",
            str(out_dir),
            "--fixed-timestamp",
            FIXED_TIMESTAMP,
        ],
        capture_output=True,
        text=True,
        cwd=str(paper_root),
    )


def _file_sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _read_rows(path: Path) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


# ============================================================================
# Test 1 — determinism
# ============================================================================


def test_deterministic_condition_ids(paper_root, tmp_path):
    """Two runs with --fixed-timestamp produce byte-identical CSVs."""
    out_a = tmp_path / "out_a"
    out_b = tmp_path / "out_b"
    rc_a = _run_script(paper_root, out_a)
    rc_b = _run_script(paper_root, out_b)
    assert rc_a.returncode == 0, rc_a.stderr
    assert rc_b.returncode == 0, rc_b.stderr

    for name in (
        "conditions_primary.csv",
        "conditions_gsm8k.csv",
        "scoring_manifest.csv",
    ):
        sha_a = _file_sha(out_a / name)
        sha_b = _file_sha(out_b / name)
        assert sha_a == sha_b, f"{name} not byte-identical across runs."


# ============================================================================
# Test 2 — nominal condition count
# ============================================================================


def test_nominal_condition_count(paper_root, tmp_path):
    """Fixture: 3 models × 4 prompts × 2 rules × ((4 benchmarks × 3 seeds) + (1 × 1)) = 312."""
    out_dir = tmp_path / "out"
    result = _run_script(paper_root, out_dir)
    assert result.returncode == 0, result.stderr

    rows = _read_rows(out_dir / "conditions_primary.csv")
    expected = 3 * (4 * 3 * 2 * 4 + 4 * 1 * 2 * 1)  # = 3 * (96 + 8) = 312
    assert len(rows) == expected, (
        f"Expected {expected} primary rows, got {len(rows)}."
    )


# ============================================================================
# Test 3 — secondary tier header-only
# ============================================================================


def test_secondary_tier_emitted_when_gsm8k_present(paper_root, tmp_path):
    """Fixture has no secondary benchmark: secondary CSV exists but is header-only."""
    out_dir = tmp_path / "out"
    result = _run_script(paper_root, out_dir)
    assert result.returncode == 0, result.stderr

    secondary = out_dir / "conditions_gsm8k.csv"
    assert secondary.exists(), "conditions_gsm8k.csv must always be emitted."

    rows = _read_rows(secondary)
    assert len(rows) == 0, (
        f"Expected header-only secondary CSV, got {len(rows)} data rows."
    )

    # Header should still match the spec.
    with open(secondary, encoding="utf-8", newline="") as f:
        header = next(csv.reader(f))
    assert header == CONDITION_FIELDNAMES


# ============================================================================
# Test 4 — condition ID scheme
# ============================================================================


def test_condition_id_scheme(paper_root, tmp_path):
    """Sample 10 rows; each condition_id parses into the 6-segment scheme."""
    out_dir = tmp_path / "out"
    result = _run_script(paper_root, out_dir)
    assert result.returncode == 0, result.stderr

    rows = _read_rows(out_dir / "conditions_primary.csv")
    assert len(rows) >= 10

    # Take a deterministic sample (first 10 rows by sorted condition_id).
    sample = sorted(rows, key=lambda r: r["condition_id"])[:10]

    seed_pat = re.compile(r"^(s\d+|0shot)$")

    for row in sample:
        # Six segments separated by `__`. Subject is empty for all
        # fixture benchmarks (no subjects), so segment 1 is "".
        segments = row["condition_id"].split("__")
        assert len(segments) == 6, (
            f"condition_id has {len(segments)} segments, expected 6: "
            f"{row['condition_id']}"
        )
        bench, subj, model, prompt, seed, rule = segments

        assert bench == row["benchmark_id"]
        assert subj == row["subject_id"]
        assert model == row["model_id"]
        assert prompt == row["prompt_id"]
        assert seed == row["seed_id"]
        assert rule == row["scoring_rule_id"]

        assert seed_pat.match(seed), (
            f"seed segment '{seed}' does not match s<int>|0shot pattern"
        )

    # condition_id uniqueness across the entire manifest.
    all_ids = {r["condition_id"] for r in rows}
    assert len(all_ids) == len(rows), "condition_id values must be unique."


# ============================================================================
# Test 5 — config_hash present and 8-char hex on every row
# ============================================================================


def test_config_hash_populated(paper_root, tmp_path):
    out_dir = tmp_path / "out"
    result = _run_script(paper_root, out_dir)
    assert result.returncode == 0, result.stderr

    rows = _read_rows(out_dir / "conditions_primary.csv")
    pat = re.compile(r"^[0-9a-f]{8}$")
    for row in rows:
        assert pat.match(row["config_hash"]), (
            f"Bad config_hash for row {row['condition_id']}: "
            f"'{row['config_hash']}'"
        )

    # All rows share the same config_hash (single config state).
    assert len({r["config_hash"] for r in rows}) == 1


# ============================================================================
# Test 6 — every row pending
# ============================================================================


def test_status_all_pending(paper_root, tmp_path):
    out_dir = tmp_path / "out"
    result = _run_script(paper_root, out_dir)
    assert result.returncode == 0, result.stderr

    rows = _read_rows(out_dir / "conditions_primary.csv")
    statuses = {r["condition_status"] for r in rows}
    assert statuses == {"pending"}, (
        f"Expected condition_status=='pending' for every row; got {statuses}"
    )


# ============================================================================
# Test 7 — 0-shot benchmark uses seed_id "0shot" and fewshot_k == 0
# ============================================================================


def test_fewshot_k_zero_uses_0shot_seed(paper_root, tmp_path):
    """test_bench_3 is 0-shot: every row should use seed_id == '0shot'."""
    out_dir = tmp_path / "out"
    result = _run_script(paper_root, out_dir)
    assert result.returncode == 0, result.stderr

    rows = _read_rows(out_dir / "conditions_primary.csv")
    bench_3_rows = [r for r in rows if r["benchmark_id"] == "test_bench_3"]
    assert bench_3_rows, "test_bench_3 should produce some rows."

    for r in bench_3_rows:
        assert r["seed_id"] == "0shot", (
            f"Expected seed_id=='0shot' for test_bench_3 row; "
            f"got '{r['seed_id']}'"
        )
        assert r["fewshot_k"] == "0", (
            f"Expected fewshot_k=='0' for test_bench_3 row; "
            f"got '{r['fewshot_k']}'"
        )

    # Sanity: 3 models × 4 prompts × 1 seed × 2 rules = 24
    assert len(bench_3_rows) == 24, (
        f"Expected 24 rows for test_bench_3 (0-shot); got {len(bench_3_rows)}."
    )


# ============================================================================
# Test 8 — scoring_manifest.csv shape
# ============================================================================


def test_scoring_manifest_emitted(paper_root, tmp_path):
    out_dir = tmp_path / "out"
    result = _run_script(paper_root, out_dir)
    assert result.returncode == 0, result.stderr

    scoring = out_dir / "scoring_manifest.csv"
    assert scoring.exists()

    with open(scoring, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == SCORING_FIELDNAMES
        rows = list(reader)

    assert len(rows) == 2, (
        f"Expected 2 scoring rules (ll_norm, generate_parse); got {len(rows)}."
    )
    rule_ids = {r["scoring_rule_id"] for r in rows}
    assert rule_ids == {"ll_norm", "generate_parse"}

    # applies_to_task_types is semicolon-separated.
    for r in rows:
        assert ";" in r["applies_to_task_types"] or r["applies_to_task_types"], (
            f"applies_to_task_types should be non-empty: {r}"
        )
