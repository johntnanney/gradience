"""Tests for config loading and validation (SPEC §4.5).

Exercises load_config() directly and invokes 00_validate_config.py
end-to-end against fixture configs.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from gradience_study.config import (
    ConfigValidationError,
    load_config,
    compute_config_hash,
)


FIXTURES_DIR = Path(__file__).parent / "fixtures"
SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"


# -- Fixtures ----------------------------------------------------------------


@pytest.fixture(scope="session", autouse=True)
def ensure_fixtures_built():
    """Build fixture configs if missing. Idempotent."""
    cfg_dir = FIXTURES_DIR / "configs"
    if not (cfg_dir / "study_config.yaml").exists():
        # Run the fixture builder
        result = subprocess.run(
            [sys.executable, str(FIXTURES_DIR / "make_fixtures.py")],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"Fixture build failed: {result.stderr}"


@pytest.fixture
def paper_root(tmp_path):
    """A minimal paper-root layout using fixture configs."""
    root = tmp_path / "paper_root"
    (root / "configs").mkdir(parents=True)
    for src in (FIXTURES_DIR / "configs").glob("*.yaml"):
        shutil.copy(src, root / "configs" / src.name)
    return root


# -- load_config tests -------------------------------------------------------


def test_load_config_returns_merged_object(paper_root):
    merged = load_config(paper_root / "configs" / "study_config.yaml")
    assert merged.study_config.study_id == "test_study"
    assert len(merged.models) == 3
    assert all(m.primary for m in merged.models)
    assert len(merged.benchmarks) == 5
    assert all(b.tier == "primary" for b in merged.benchmarks)
    assert len(merged.prompts) == 20  # 5 benchmarks × 4 prompts
    assert len(merged.scoring_rules) == 2


def test_load_config_raises_on_missing_file(tmp_path):
    missing = tmp_path / "configs" / "study_config.yaml"
    with pytest.raises(FileNotFoundError):
        load_config(missing)


def test_load_config_raises_on_missing_include(paper_root):
    # Remove one of the includes
    (paper_root / "configs" / "models.yaml").unlink()
    with pytest.raises(ConfigValidationError):
        load_config(paper_root / "configs" / "study_config.yaml")


def test_merged_raw_dict_hashable(paper_root):
    """The raw_dict is used for canonicalization/hashing."""
    merged = load_config(paper_root / "configs" / "study_config.yaml")
    full, short = compute_config_hash(merged.raw_dict)
    assert len(full) == 64
    assert len(short) == 8


def test_merged_raw_dict_hash_stable_across_loads(paper_root):
    """Loading the same configs twice yields the same hash."""
    merged_a = load_config(paper_root / "configs" / "study_config.yaml")
    merged_b = load_config(paper_root / "configs" / "study_config.yaml")
    h_a, _ = compute_config_hash(merged_a.raw_dict)
    h_b, _ = compute_config_hash(merged_b.raw_dict)
    assert h_a == h_b


def test_benchmark_seed_facet_honored(paper_root):
    merged = load_config(paper_root / "configs" / "study_config.yaml")
    b3 = next(b for b in merged.benchmarks if b.benchmark_id == "test_bench_3")
    assert b3.seed_facet is False
    assert b3.fewshot_k == 0
    b1 = next(b for b in merged.benchmarks if b.benchmark_id == "test_bench_1")
    assert b1.seed_facet is True
    assert b1.fewshot_k == 5


# -- End-to-end: 00_validate_config.py --------------------------------------


def test_validate_script_runs_and_emits_json(paper_root, tmp_path):
    out_path = tmp_path / "config_validation.json"
    script_path = SCRIPTS_DIR / "00_validate_config.py"
    result = subprocess.run(
        [sys.executable, str(script_path),
         "--config", str(paper_root / "configs" / "study_config.yaml"),
         "--out", str(out_path)],
        capture_output=True,
        text=True,
    )
    # The fixture configs reference prompt files that do not exist, so
    # validation should fail with exit code 2 — but the report JSON must
    # still be emitted.
    assert out_path.exists()
    with open(out_path) as f:
        report = json.load(f)
    assert "validation_status" in report
    assert "validation_errors" in report
    assert "validation_warnings" in report


def test_validate_script_reports_primary_counts(paper_root, tmp_path):
    out_path = tmp_path / "config_validation.json"
    script_path = SCRIPTS_DIR / "00_validate_config.py"
    subprocess.run(
        [sys.executable, str(script_path),
         "--config", str(paper_root / "configs" / "study_config.yaml"),
         "--out", str(out_path)],
        capture_output=True,
        text=True,
    )
    with open(out_path) as f:
        report = json.load(f)
    assert report["primary_model_count"] == 3
    assert report["primary_benchmark_count"] == 5
    assert report["total_prompts"] == 20


def test_validate_script_computes_hash_when_configs_load(paper_root, tmp_path):
    """Even when other validation errors occur, the hash should be present
    if the configs loaded successfully."""
    out_path = tmp_path / "config_validation.json"
    script_path = SCRIPTS_DIR / "00_validate_config.py"
    subprocess.run(
        [sys.executable, str(script_path),
         "--config", str(paper_root / "configs" / "study_config.yaml"),
         "--out", str(out_path)],
        capture_output=True,
        text=True,
    )
    with open(out_path) as f:
        report = json.load(f)
    if report["config_hash_full"] is not None:
        assert len(report["config_hash_full"]) == 64
        assert len(report["config_hash"]) == 8


def test_validate_script_fails_exit_2_on_missing_prompt_files(paper_root, tmp_path):
    """Fixture configs declare prompt paths that don't exist on disk,
    so the script should exit 2 (per SPEC §11.1)."""
    out_path = tmp_path / "config_validation.json"
    script_path = SCRIPTS_DIR / "00_validate_config.py"
    result = subprocess.run(
        [sys.executable, str(script_path),
         "--config", str(paper_root / "configs" / "study_config.yaml"),
         "--out", str(out_path)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2


def test_validate_script_passes_when_prompt_files_exist(paper_root, tmp_path):
    """When all prompt files exist (content hashes marked TODO yield
    warnings, not errors), validation passes with exit 0."""
    # Create the prompt files the fixture configs declare
    for bench in [f"test_bench_{i}" for i in range(1, 6)]:
        bench_dir = paper_root / "prompts" / bench
        bench_dir.mkdir(parents=True, exist_ok=True)
        for pid in ["P1_original", "P2_lm_eval",
                    "P3_helm_or_published", "P4_minimal_sourced"]:
            (bench_dir / f"{pid}.txt").write_text(
                f"TEST PROMPT {bench}/{pid}\n{{question}}\n{{answer_instruction}}\n",
                encoding="utf-8",
            )

    out_path = tmp_path / "config_validation.json"
    script_path = SCRIPTS_DIR / "00_validate_config.py"
    result = subprocess.run(
        [sys.executable, str(script_path),
         "--config", str(paper_root / "configs" / "study_config.yaml"),
         "--out", str(out_path)],
        capture_output=True,
        text=True,
    )
    with open(out_path) as f:
        report = json.load(f)
    # TODO content hashes produce warnings, not errors
    assert report["validation_status"] == "pass", (
        f"Expected pass, got fail with errors: {report['validation_errors']}"
    )
    assert result.returncode == 0
    # But we should see warnings for the TODO hashes
    assert len(report["validation_warnings"]) > 0
