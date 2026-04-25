"""Tests for 04_normalize_outputs.py and 05_make_condition_scores.py.

Exercises both scripts via subprocess against minimal paper-root layouts
constructed in tmp_path from the canonical fixtures.
"""

from __future__ import annotations

import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import pytest

PAPER_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = PAPER_ROOT / "scripts"
SCHEMAS_DIR = PAPER_ROOT / "schemas"
FIXTURES_DIR = PAPER_ROOT / "tests" / "fixtures"

NORMALIZE_SCRIPT = SCRIPTS_DIR / "04_normalize_outputs.py"
AGGREGATE_SCRIPT = SCRIPTS_DIR / "05_make_condition_scores.py"


# -- Fixture build -----------------------------------------------------------


@pytest.fixture(scope="session", autouse=True)
def ensure_fixtures_built():
    """Build all fixture artifacts (including raw run dirs). Idempotent."""
    needed = [
        FIXTURES_DIR / "tiny_item_outputs.jsonl",
        FIXTURES_DIR / "tiny_item_scores.jsonl",
        FIXTURES_DIR / "tiny_run_metadata_gp.json",
        FIXTURES_DIR / "tiny_run_metadata_ll.json",
        FIXTURES_DIR / "raw"
            / "arc_challenge__pythia_410m__P1_original__s42__generate_parse"
            / "run_metadata.json",
        FIXTURES_DIR / "raw"
            / "arc_challenge__pythia_410m__P1_original__s42__ll_norm"
            / "run_metadata.json",
    ]
    if any(not p.exists() for p in needed):
        result = subprocess.run(
            [sys.executable, str(FIXTURES_DIR / "make_fixtures.py")],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"Fixture build failed: {result.stderr}"


# -- Shared constants --------------------------------------------------------


GP_RUN_ID = "arc_challenge__pythia_410m__P1_original__s42__generate_parse"
LL_RUN_ID = "arc_challenge__pythia_410m__P1_original__s42__ll_norm"

CONDITION_FIELDS = [
    "condition_id", "tier", "benchmark_id", "subject_id", "model_id",
    "prompt_id", "seed_id", "fewshot_k", "scoring_rule_id", "task_type",
    "expected_num_items", "condition_status", "created_at", "config_hash",
]


# -- Helpers -----------------------------------------------------------------


def _gp_condition_row(condition_status: str = "complete",
                      expected_num_items: int = 4) -> dict:
    return {
        "condition_id": GP_RUN_ID,
        "tier": "primary",
        "benchmark_id": "arc_challenge",
        "subject_id": "",
        "model_id": "pythia_410m",
        "prompt_id": "P1_original",
        "seed_id": "42",
        "fewshot_k": 5,
        "scoring_rule_id": "generate_parse",
        "task_type": "constrained_choice",
        "expected_num_items": expected_num_items,
        "condition_status": condition_status,
        "created_at": "2026-04-24T00:00:00+00:00",
        "config_hash": "deadbeef",
    }


def _ll_condition_row(condition_status: str = "complete",
                      expected_num_items: int = 4) -> dict:
    return {
        "condition_id": LL_RUN_ID,
        "tier": "primary",
        "benchmark_id": "arc_challenge",
        "subject_id": "",
        "model_id": "pythia_410m",
        "prompt_id": "P1_original",
        "seed_id": "42",
        "fewshot_k": 5,
        "scoring_rule_id": "ll_norm",
        "task_type": "constrained_choice",
        "expected_num_items": expected_num_items,
        "condition_status": condition_status,
        "created_at": "2026-04-24T00:00:00+00:00",
        "config_hash": "deadbeef",
    }


def _write_conditions_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CONDITION_FIELDS, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _copy_run(src_run_dir: Path, dst_root: Path) -> Path:
    """Copy a fixture run-dir into dst_root/<run_id>/, return new path."""
    dst = dst_root / src_run_dir.name
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src_run_dir, dst)
    return dst


def _run_normalize(conditions_csv: Path, raw_dir: Path, out_path: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            sys.executable, str(NORMALIZE_SCRIPT),
            "--conditions", str(conditions_csv),
            "--raw-dir", str(raw_dir),
            "--schemas-dir", str(SCHEMAS_DIR),
            "--out", str(out_path),
        ],
        capture_output=True,
        text=True,
    )


def _run_aggregate(item_level: Path, out_path: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            sys.executable, str(AGGREGATE_SCRIPT),
            "--item-level", str(item_level),
            "--out", str(out_path),
        ],
        capture_output=True,
        text=True,
    )


# -- Tests: end-to-end normalization ----------------------------------------


def test_end_to_end_gp_normalization(tmp_path):
    """G&P normalization: 4 rows in, 4 rows out, parseable=True for all."""
    raw_dir = tmp_path / "runs" / "raw"
    raw_dir.mkdir(parents=True)
    _copy_run(FIXTURES_DIR / "raw" / GP_RUN_ID, raw_dir)

    conditions_csv = tmp_path / "manifests" / "conditions_primary.csv"
    _write_conditions_csv(conditions_csv, [_gp_condition_row()])

    out_path = tmp_path / "runs" / "normalized" / "item_level_primary.parquet"
    result = _run_normalize(conditions_csv, raw_dir, out_path)

    assert result.returncode == 0, f"stderr: {result.stderr}"
    assert out_path.exists()

    df = pq.read_table(out_path).to_pandas()
    assert len(df) == 4
    assert (df["condition_id"] == GP_RUN_ID).all()
    assert (df["scoring_rule_id"] == "generate_parse").all()
    # All G&P fixture rows have parse_status="parsed".
    assert df["parseable"].all()
    # LL-specific columns must all be null.
    assert df["selected_answer"].isna().all()
    assert df["ll_margin_top2"].isna().all()
    assert df["gold_choice_score"].isna().all()
    assert df["selected_choice_score"].isna().all()
    # G&P-specific columns must be populated.
    assert df["raw_generation"].notna().all()
    assert df["parse_status"].notna().all()


def test_end_to_end_ll_normalization(tmp_path):
    """LL normalization: ll_margin_top2 computed correctly; G&P fields null."""
    raw_dir = tmp_path / "runs" / "raw"
    raw_dir.mkdir(parents=True)
    _copy_run(FIXTURES_DIR / "raw" / LL_RUN_ID, raw_dir)

    conditions_csv = tmp_path / "manifests" / "conditions_primary.csv"
    _write_conditions_csv(conditions_csv, [_ll_condition_row()])

    out_path = tmp_path / "runs" / "normalized" / "item_level_primary.parquet"
    result = _run_normalize(conditions_csv, raw_dir, out_path)

    assert result.returncode == 0, f"stderr: {result.stderr}"
    df = pq.read_table(out_path).to_pandas()
    assert len(df) == 4

    # Recompute ll_margin_top2 from the source fixture and compare.
    src_rows: list[dict] = []
    with open(FIXTURES_DIR / "tiny_item_scores.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                src_rows.append(json.loads(line))
    expected_margins: dict[str, float] = {}
    for row in src_rows:
        scores = sorted(row["choice_scores"].values(), reverse=True)
        expected_margins[row["item_id"]] = float(scores[0] - scores[1])

    for _, df_row in df.iterrows():
        item_id = df_row["item_id"]
        assert item_id in expected_margins
        assert abs(float(df_row["ll_margin_top2"]) - expected_margins[item_id]) < 1e-5

    # G&P-specific columns must all be null on LL rows.
    assert df["raw_generation"].isna().all()
    assert df["parsed_answer"].isna().all()
    assert df["generation_length_tokens"].isna().all()
    assert df["parse_status"].isna().all()
    assert df["parseable"].isna().all()
    # LL-specific columns must be populated.
    assert df["selected_answer"].notna().all()
    assert df["choice_count"].notna().all()
    assert (df["choice_count"] == 4).all()


# -- Tests: hard-fail paths --------------------------------------------------


def test_run_id_mismatch_hard_fail(tmp_path):
    """Manifest condition_id disagrees with run_metadata.condition_id."""
    raw_dir = tmp_path / "runs" / "raw"
    raw_dir.mkdir(parents=True)

    # Build a manifest that references the GP run_id, but rewrite metadata
    # to claim a different condition_id ("foo") under the same directory.
    run_dir = raw_dir / GP_RUN_ID
    src_run_dir = FIXTURES_DIR / "raw" / GP_RUN_ID
    shutil.copytree(src_run_dir, run_dir)
    md_path = run_dir / "run_metadata.json"
    with open(md_path, "r", encoding="utf-8") as f:
        md = json.load(f)
    md["condition_id"] = "foo"
    md["run_id"] = "foo"
    with open(md_path, "w", encoding="utf-8") as f:
        json.dump(md, f)

    conditions_csv = tmp_path / "manifests" / "conditions_primary.csv"
    _write_conditions_csv(conditions_csv, [_gp_condition_row()])

    out_path = tmp_path / "runs" / "normalized" / "item_level_primary.parquet"
    result = _run_normalize(conditions_csv, raw_dir, out_path)

    assert result.returncode == 3, f"stderr: {result.stderr}"
    # Parquet still written (possibly empty), but contains no rows for that condition.
    if out_path.exists():
        df = pq.read_table(out_path).to_pandas()
        assert (df["condition_id"] != GP_RUN_ID).all() if not df.empty else True


def test_item_count_mismatch_hard_fail(tmp_path):
    """Manifest expected_num_items=4 but JSONL has only 3 rows."""
    raw_dir = tmp_path / "runs" / "raw"
    raw_dir.mkdir(parents=True)
    run_dir = _copy_run(FIXTURES_DIR / "raw" / GP_RUN_ID, raw_dir)
    # Drop the last row of item_outputs.jsonl.
    jsonl_path = run_dir / "item_outputs.jsonl"
    with open(jsonl_path, "r", encoding="utf-8") as f:
        lines = [ln for ln in f if ln.strip()]
    with open(jsonl_path, "w", encoding="utf-8") as f:
        f.writelines(lines[:3])

    # Rewrite metadata so the partial-completion check doesn't fire first.
    md_path = run_dir / "run_metadata.json"
    with open(md_path, "r", encoding="utf-8") as f:
        md = json.load(f)
    md["num_items_expected"] = 3
    md["num_items_completed"] = 3
    with open(md_path, "w", encoding="utf-8") as f:
        json.dump(md, f)

    conditions_csv = tmp_path / "manifests" / "conditions_primary.csv"
    _write_conditions_csv(conditions_csv, [_gp_condition_row(expected_num_items=4)])

    out_path = tmp_path / "runs" / "normalized" / "item_level_primary.parquet"
    result = _run_normalize(conditions_csv, raw_dir, out_path)

    assert result.returncode == 3
    # The condition's id should appear in the error stream.
    assert GP_RUN_ID in result.stderr


def test_partial_run_completion_hard_fail(tmp_path):
    """run_metadata.num_items_completed (3) != num_items_expected (4)."""
    raw_dir = tmp_path / "runs" / "raw"
    raw_dir.mkdir(parents=True)
    run_dir = _copy_run(FIXTURES_DIR / "raw" / GP_RUN_ID, raw_dir)
    md_path = run_dir / "run_metadata.json"
    with open(md_path, "r", encoding="utf-8") as f:
        md = json.load(f)
    md["num_items_completed"] = 3
    with open(md_path, "w", encoding="utf-8") as f:
        json.dump(md, f)

    conditions_csv = tmp_path / "manifests" / "conditions_primary.csv"
    _write_conditions_csv(conditions_csv, [_gp_condition_row()])

    out_path = tmp_path / "runs" / "normalized" / "item_level_primary.parquet"
    result = _run_normalize(conditions_csv, raw_dir, out_path)

    assert result.returncode == 3
    assert "PartialRunCompletion" in result.stderr or "partial" in result.stderr.lower()


def test_schema_invalid_row_hard_fail(tmp_path):
    """A single row with a missing required field fails the whole condition."""
    raw_dir = tmp_path / "runs" / "raw"
    raw_dir.mkdir(parents=True)
    run_dir = _copy_run(FIXTURES_DIR / "raw" / GP_RUN_ID, raw_dir)
    jsonl_path = run_dir / "item_outputs.jsonl"
    with open(jsonl_path, "r", encoding="utf-8") as f:
        rows = [json.loads(ln) for ln in f if ln.strip()]
    # Remove a required field from the first row.
    del rows[0]["parse_status"]
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in rows:
            json.dump(r, f, sort_keys=True)
            f.write("\n")

    conditions_csv = tmp_path / "manifests" / "conditions_primary.csv"
    _write_conditions_csv(conditions_csv, [_gp_condition_row()])

    out_path = tmp_path / "runs" / "normalized" / "item_level_primary.parquet"
    result = _run_normalize(conditions_csv, raw_dir, out_path)

    assert result.returncode == 3
    if out_path.exists():
        df = pq.read_table(out_path).to_pandas()
        # No rows from the failed condition were emitted.
        if not df.empty:
            assert (df["condition_id"] != GP_RUN_ID).all()


def test_pending_conditions_skipped(tmp_path):
    """A pending condition is skipped — no row, no raw_dir read attempted."""
    raw_dir = tmp_path / "runs" / "raw"
    raw_dir.mkdir(parents=True)
    # No raw dir copied; the script must not look for it.
    conditions_csv = tmp_path / "manifests" / "conditions_primary.csv"
    _write_conditions_csv(
        conditions_csv,
        [_gp_condition_row(condition_status="pending")],
    )

    out_path = tmp_path / "runs" / "normalized" / "item_level_primary.parquet"
    result = _run_normalize(conditions_csv, raw_dir, out_path)

    assert result.returncode == 0, f"stderr: {result.stderr}"
    df = pq.read_table(out_path).to_pandas()
    assert len(df) == 0


def test_failed_conditions_skipped(tmp_path):
    """A failed condition is skipped — no row emitted, exit 0."""
    raw_dir = tmp_path / "runs" / "raw"
    raw_dir.mkdir(parents=True)
    conditions_csv = tmp_path / "manifests" / "conditions_primary.csv"
    _write_conditions_csv(
        conditions_csv,
        [_gp_condition_row(condition_status="failed")],
    )

    out_path = tmp_path / "runs" / "normalized" / "item_level_primary.parquet"
    result = _run_normalize(conditions_csv, raw_dir, out_path)

    assert result.returncode == 0, f"stderr: {result.stderr}"
    df = pq.read_table(out_path).to_pandas()
    assert len(df) == 0


# -- Tests: aggregator -------------------------------------------------------


def test_condition_aggregator_end_to_end(tmp_path):
    """Aggregator on a 4-row item-level parquet emits one condition row."""
    # Reuse the GP normalization to produce input.
    raw_dir = tmp_path / "runs" / "raw"
    raw_dir.mkdir(parents=True)
    _copy_run(FIXTURES_DIR / "raw" / GP_RUN_ID, raw_dir)
    conditions_csv = tmp_path / "manifests" / "conditions_primary.csv"
    _write_conditions_csv(conditions_csv, [_gp_condition_row()])
    item_level = tmp_path / "runs" / "normalized" / "item_level_primary.parquet"
    norm_result = _run_normalize(conditions_csv, raw_dir, item_level)
    assert norm_result.returncode == 0, f"normalize stderr: {norm_result.stderr}"

    out_csv = tmp_path / "runs" / "normalized" / "condition_level_primary.csv"
    agg_result = _run_aggregate(item_level, out_csv)
    assert agg_result.returncode == 0, f"aggregate stderr: {agg_result.stderr}"

    df = pd.read_csv(out_csv)
    assert len(df) == 1
    row = df.iloc[0]
    assert row["condition_id"] == GP_RUN_ID
    assert row["num_items"] == 4
    # 2 of 4 G&P fixture items are correct.
    assert row["num_correct"] == 2
    assert abs(float(row["accuracy"]) - (row["num_correct"] / row["num_items"])) < 1e-9


def test_aggregator_parseability_computed_for_gp_only(tmp_path):
    """LL conditions get null parseability_rate; G&P conditions get a number."""
    raw_dir = tmp_path / "runs" / "raw"
    raw_dir.mkdir(parents=True)
    _copy_run(FIXTURES_DIR / "raw" / GP_RUN_ID, raw_dir)
    _copy_run(FIXTURES_DIR / "raw" / LL_RUN_ID, raw_dir)

    conditions_csv = tmp_path / "manifests" / "conditions_primary.csv"
    _write_conditions_csv(
        conditions_csv,
        [_gp_condition_row(), _ll_condition_row()],
    )
    item_level = tmp_path / "runs" / "normalized" / "item_level_primary.parquet"
    norm_result = _run_normalize(conditions_csv, raw_dir, item_level)
    assert norm_result.returncode == 0, f"normalize stderr: {norm_result.stderr}"

    out_csv = tmp_path / "runs" / "normalized" / "condition_level_primary.csv"
    agg_result = _run_aggregate(item_level, out_csv)
    assert agg_result.returncode == 0, f"aggregate stderr: {agg_result.stderr}"

    df = pd.read_csv(out_csv)
    assert len(df) == 2

    gp_row = df[df["scoring_rule_id"] == "generate_parse"].iloc[0]
    ll_row = df[df["scoring_rule_id"] == "ll_norm"].iloc[0]

    # G&P parseability is populated and equal to 1.0 for the all-parsed fixture.
    assert pd.notna(gp_row["parseability_rate"])
    assert abs(float(gp_row["parseability_rate"]) - 1.0) < 1e-9
    # LL parseability is null in the CSV (read as NaN).
    assert pd.isna(ll_row["parseability_rate"])
    # LL mean_generation_length_tokens is null too.
    assert pd.isna(ll_row["mean_generation_length_tokens"])
    # G&P mean_generation_length_tokens is populated.
    assert pd.notna(gp_row["mean_generation_length_tokens"])
