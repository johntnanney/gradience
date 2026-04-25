"""Tests for 03_validate_prompts.py — SPEC_CPU_v0_2 §8.4.

Six tests cover:
    1. Missing prompt file → exit 2.
    2. Content-hash mismatch (when declared hash is not a placeholder) → exit 2.
    3. content_hash == "TODO" → exit 0 with a warning row.
    4. Missing placeholder → exit 2.
    5. admissibility_status == "draft" → exit 0 with a warning.
    6. All-pass: every row pass or warn; exit 0; one row per prompt.

Each test builds its own minimal paper-root in tmp_path so we can vary the
prompts.yaml content and the on-disk prompt files independently.
"""

from __future__ import annotations

import csv
import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


PAPER_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = PAPER_ROOT / "scripts"
FIXTURES_DIR = Path(__file__).parent / "fixtures"
SCRIPT_PATH = SCRIPTS_DIR / "03_validate_prompts.py"


# -- Fixtures ---------------------------------------------------------------


@pytest.fixture(scope="session", autouse=True)
def ensure_fixtures_built():
    cfg_dir = FIXTURES_DIR / "configs"
    if not (cfg_dir / "study_config.yaml").exists():
        result = subprocess.run(
            [sys.executable, str(FIXTURES_DIR / "make_fixtures.py")],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"Fixture build failed: {result.stderr}"


def _copy_base_configs(root: Path) -> None:
    (root / "configs").mkdir(parents=True, exist_ok=True)
    for src in (FIXTURES_DIR / "configs").glob("*.yaml"):
        if src.name == "prompts.yaml":
            continue  # tests will write their own
        shutil.copy(src, root / "configs" / src.name)
    (root / "manifests").mkdir(exist_ok=True)


def _make_prompt_file(root: Path, rel_path: str, content: str) -> Path:
    """Create a prompt file under ``root`` at ``rel_path`` with given content."""
    p = root / rel_path
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return p


def _sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _write_single_prompt_yaml(
    root: Path,
    benchmark_id: str = "test_bench_1",
    prompt_id: str = "P1_original",
    *,
    prompt_path: str,
    content_hash: str,
    source_type: str = "benchmark_authors",
    admissibility_status: str = "locked",
    placeholders_used: list[str] | None = None,
) -> None:
    """Write a single-prompt prompts.yaml to root/configs/.

    Replaces the multi-prompt fixture so tests can vary one prompt at a time.
    """
    placeholders_used = placeholders_used or ["question", "answer_instruction"]
    placeholders_block = "\n".join(f"      - {p}" for p in placeholders_used)
    yaml_text = (
        "prompts:\n"
        f"  - benchmark_id: {benchmark_id}\n"
        f"    prompt_id: {prompt_id}\n"
        f"    prompt_path: {prompt_path}\n"
        f"    content_hash: {content_hash}\n"
        f"    source_type: {source_type}\n"
        f"    source_url_or_ref: test://source/{benchmark_id}/{prompt_id}\n"
        f"    source_commit: null\n"
        f"    admissibility_status: {admissibility_status}\n"
        f"    placeholders_used:\n"
        f"{placeholders_block}\n"
        f"    notes: test fixture\n"
    )
    (root / "configs" / "prompts.yaml").write_text(yaml_text, encoding="utf-8")


def _run_script(root: Path, out_path: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--config",
            str(root / "configs" / "study_config.yaml"),
            "--out",
            str(out_path),
        ],
        capture_output=True,
        text=True,
        cwd=str(root),
    )


def _read_rows(path: Path) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


# ============================================================================
# Test 1 — missing prompt file is an error
# ============================================================================


def test_missing_prompt_file_is_error(tmp_path):
    root = tmp_path / "paper_root"
    _copy_base_configs(root)
    _write_single_prompt_yaml(
        root,
        prompt_path="prompts/test_bench_1/P1_original.txt",
        content_hash="TODO",
    )
    # Deliberately do NOT create the prompt file on disk.

    out_csv = tmp_path / "prompt_manifest.csv"
    result = _run_script(root, out_csv)

    assert result.returncode == 2, (
        f"Expected exit 2; got {result.returncode}.\nstderr: {result.stderr}"
    )
    assert out_csv.exists()
    rows = _read_rows(out_csv)
    assert len(rows) == 1
    assert rows[0]["validation_status"] == "fail"
    assert "not found" in rows[0]["warnings"].lower()


# ============================================================================
# Test 2 — content hash mismatch is an error
# ============================================================================


def test_content_hash_mismatch_is_error(tmp_path):
    root = tmp_path / "paper_root"
    _copy_base_configs(root)

    content = "PROMPT BODY {{question}}\n{{answer_instruction}}\n"
    _make_prompt_file(root, "prompts/test_bench_1/P1_original.txt", content)

    # Declare a wrong (non-placeholder) hash.
    wrong_hash = "0" * 64
    _write_single_prompt_yaml(
        root,
        prompt_path="prompts/test_bench_1/P1_original.txt",
        content_hash=wrong_hash,
    )

    out_csv = tmp_path / "prompt_manifest.csv"
    result = _run_script(root, out_csv)

    assert result.returncode == 2, (
        f"Expected exit 2; got {result.returncode}.\nstderr: {result.stderr}"
    )
    rows = _read_rows(out_csv)
    assert rows[0]["validation_status"] == "fail"
    assert "mismatch" in rows[0]["warnings"].lower()
    assert rows[0]["actual_content_hash"] == _sha256_text(content)


# ============================================================================
# Test 3 — content_hash == TODO is a warning, not an error
# ============================================================================


def test_content_hash_todo_is_warning(tmp_path):
    root = tmp_path / "paper_root"
    _copy_base_configs(root)

    content = "PROMPT BODY {{question}}\n{{answer_instruction}}\n"
    _make_prompt_file(root, "prompts/test_bench_1/P1_original.txt", content)

    _write_single_prompt_yaml(
        root,
        prompt_path="prompts/test_bench_1/P1_original.txt",
        content_hash="TODO",
    )

    out_csv = tmp_path / "prompt_manifest.csv"
    result = _run_script(root, out_csv)

    assert result.returncode == 0, (
        f"Expected exit 0; got {result.returncode}.\nstderr: {result.stderr}"
    )
    rows = _read_rows(out_csv)
    assert rows[0]["validation_status"] == "warn"
    assert "placeholder" in rows[0]["warnings"].lower()
    # actual_content_hash must be the computed SHA-256.
    assert rows[0]["actual_content_hash"] == _sha256_text(content)


# ============================================================================
# Test 4 — missing placeholder is an error
# ============================================================================


def test_missing_placeholder_is_error(tmp_path):
    root = tmp_path / "paper_root"
    _copy_base_configs(root)

    # File contains {{answer_instruction}} but not {{question}}.
    content = "PROMPT BODY without question\n{{answer_instruction}}\n"
    _make_prompt_file(root, "prompts/test_bench_1/P1_original.txt", content)

    _write_single_prompt_yaml(
        root,
        prompt_path="prompts/test_bench_1/P1_original.txt",
        content_hash="TODO",
        placeholders_used=["question", "answer_instruction"],
    )

    out_csv = tmp_path / "prompt_manifest.csv"
    result = _run_script(root, out_csv)

    assert result.returncode == 2, (
        f"Expected exit 2; got {result.returncode}.\nstderr: {result.stderr}"
    )
    rows = _read_rows(out_csv)
    assert rows[0]["validation_status"] == "fail"
    assert "{{question}}" in rows[0]["warnings"]


# ============================================================================
# Test 5 — admissibility_status == "draft" is a warning
# ============================================================================


def test_admissibility_draft_is_warning(tmp_path):
    root = tmp_path / "paper_root"
    _copy_base_configs(root)

    content = "PROMPT BODY {{question}}\n{{answer_instruction}}\n"
    _make_prompt_file(root, "prompts/test_bench_1/P1_original.txt", content)

    _write_single_prompt_yaml(
        root,
        prompt_path="prompts/test_bench_1/P1_original.txt",
        content_hash="TODO",
        admissibility_status="draft",
    )

    out_csv = tmp_path / "prompt_manifest.csv"
    result = _run_script(root, out_csv)

    assert result.returncode == 0, (
        f"Expected exit 0; got {result.returncode}.\nstderr: {result.stderr}"
    )
    rows = _read_rows(out_csv)
    assert rows[0]["validation_status"] == "warn"
    assert "draft" in rows[0]["warnings"].lower()


# ============================================================================
# Test 6 — all-pass produces a clean manifest
# ============================================================================


def test_all_pass_emits_clean_manifest(tmp_path):
    """All prompts have correct hashes, all placeholders, locked status."""
    root = tmp_path / "paper_root"
    _copy_base_configs(root)

    # Five benchmarks × four prompts = 20 prompts in the standard fixture
    # set, but we'll write a smaller scoped prompts.yaml to keep this test
    # focused. Use 2 benchmarks × 2 prompts = 4 prompts.
    prompts_lines = ["prompts:"]
    bench_ids = ["test_bench_1", "test_bench_2"]
    prompt_ids = ["P1_original", "P2_lm_eval"]
    file_specs: list[tuple[str, str, str]] = []  # (rel_path, content, hash)

    for b in bench_ids:
        for p in prompt_ids:
            rel = f"prompts/{b}/{p}.txt"
            content = (
                f"PROMPT {b}/{p}\n"
                "{{question}}\n"
                "{{answer_instruction}}\n"
            )
            _make_prompt_file(root, rel, content)
            ch = _sha256_text(content)
            file_specs.append((rel, content, ch))
            prompts_lines.append(
                f"  - benchmark_id: {b}\n"
                f"    prompt_id: {p}\n"
                f"    prompt_path: {rel}\n"
                f"    content_hash: {ch}\n"
                f"    source_type: benchmark_authors\n"
                f"    source_url_or_ref: test://source/{b}/{p}\n"
                f"    source_commit: null\n"
                f"    admissibility_status: locked\n"
                f"    placeholders_used:\n"
                f"      - question\n"
                f"      - answer_instruction\n"
                f"    notes: test fixture"
            )
    (root / "configs" / "prompts.yaml").write_text(
        "\n".join(prompts_lines) + "\n", encoding="utf-8"
    )

    out_csv = tmp_path / "prompt_manifest.csv"
    result = _run_script(root, out_csv)

    assert result.returncode == 0, (
        f"Expected exit 0; got {result.returncode}.\nstderr: {result.stderr}"
    )
    rows = _read_rows(out_csv)
    assert len(rows) == 4

    for row in rows:
        assert row["validation_status"] in ("pass", "warn"), (
            f"Row {row['benchmark_id']}/{row['prompt_id']} status="
            f"{row['validation_status']} (warnings={row['warnings']})"
        )
        # No 'fail' rows in an all-pass manifest.
        assert row["validation_status"] != "fail"
        # actual_content_hash matches what we wrote.
        assert row["actual_content_hash"] == row["declared_content_hash"]
