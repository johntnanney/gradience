"""Tests for raw-output JSON Schemas (SPEC §5.2–5.4).

Pure schema-validation tests; do not invoke any pipeline scripts. Each
schema is loaded from `schemas/`, then exercised against the canonical
fixtures from `tests/fixtures/`.
"""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import jsonschema
import pytest

PAPER_ROOT = Path(__file__).resolve().parent.parent
SCHEMAS_DIR = PAPER_ROOT / "schemas"
FIXTURES_DIR = PAPER_ROOT / "tests" / "fixtures"


# -- Fixtures ----------------------------------------------------------------


@pytest.fixture(scope="module", autouse=True)
def ensure_fixtures_built():
    """Build fixture artifacts if missing. Idempotent."""
    needed = [
        FIXTURES_DIR / "tiny_item_outputs.jsonl",
        FIXTURES_DIR / "tiny_item_scores.jsonl",
        FIXTURES_DIR / "tiny_run_metadata_gp.json",
        FIXTURES_DIR / "tiny_run_metadata_ll.json",
        FIXTURES_DIR / "pathological_all_parse_fail.jsonl",
    ]
    if any(not p.exists() for p in needed):
        result = subprocess.run(
            [sys.executable, str(FIXTURES_DIR / "make_fixtures.py")],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"Fixture build failed: {result.stderr}"


def _load_schema(name: str) -> dict:
    with open(SCHEMAS_DIR / name, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


# -- Schema instances --------------------------------------------------------


@pytest.fixture(scope="module")
def run_metadata_schema() -> dict:
    return _load_schema("run_metadata.schema.json")


@pytest.fixture(scope="module")
def item_outputs_schema() -> dict:
    return _load_schema("item_outputs.schema.json")


@pytest.fixture(scope="module")
def item_scores_schema() -> dict:
    return _load_schema("item_scores.schema.json")


# -- Tests -------------------------------------------------------------------


def test_valid_run_metadata_passes(run_metadata_schema):
    """`tiny_run_metadata_gp.json` validates against the run_metadata schema."""
    with open(FIXTURES_DIR / "tiny_run_metadata_gp.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)
    jsonschema.validate(metadata, run_metadata_schema)  # raises on failure


def test_run_metadata_missing_required_field_fails(run_metadata_schema):
    """Removing a required field (status) makes the metadata invalid."""
    with open(FIXTURES_DIR / "tiny_run_metadata_gp.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)
    bad = copy.deepcopy(metadata)
    del bad["status"]
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(bad, run_metadata_schema)


def test_valid_item_outputs_pass(item_outputs_schema):
    """Every row in tiny_item_outputs.jsonl validates against the G&P schema."""
    rows = _load_jsonl(FIXTURES_DIR / "tiny_item_outputs.jsonl")
    assert len(rows) > 0
    for i, row in enumerate(rows):
        try:
            jsonschema.validate(row, item_outputs_schema)
        except jsonschema.ValidationError as e:
            pytest.fail(f"Row {i} failed schema validation: {e.message}")


def test_valid_item_scores_pass(item_scores_schema):
    """Every row in tiny_item_scores.jsonl validates against the LL schema."""
    rows = _load_jsonl(FIXTURES_DIR / "tiny_item_scores.jsonl")
    assert len(rows) > 0
    for i, row in enumerate(rows):
        try:
            jsonschema.validate(row, item_scores_schema)
        except jsonschema.ValidationError as e:
            pytest.fail(f"Row {i} failed schema validation: {e.message}")


def test_pathological_parse_fail_fixtures_are_valid(item_outputs_schema):
    """Parse-fail rows have valid parse_status enum values; schema permits them."""
    rows = _load_jsonl(FIXTURES_DIR / "pathological_all_parse_fail.jsonl")
    assert len(rows) > 0
    seen_statuses = {r["parse_status"] for r in rows}
    # Must include several non-parsed statuses (the fixture's whole point).
    assert "parsed" not in seen_statuses
    for i, row in enumerate(rows):
        try:
            jsonschema.validate(row, item_outputs_schema)
        except jsonschema.ValidationError as e:
            pytest.fail(f"Pathological row {i} failed validation: {e.message}")


def test_item_outputs_invalid_parse_status_fails(item_outputs_schema):
    """An invented parse_status value must be rejected by the enum constraint."""
    rows = _load_jsonl(FIXTURES_DIR / "tiny_item_outputs.jsonl")
    bad = copy.deepcopy(rows[0])
    bad["parse_status"] = "invented_status"
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(bad, item_outputs_schema)


def test_item_scores_requires_two_choices(item_scores_schema):
    """choices array with only 1 element violates minItems=2."""
    rows = _load_jsonl(FIXTURES_DIR / "tiny_item_scores.jsonl")
    bad = copy.deepcopy(rows[0])
    bad["choices"] = ["A"]
    bad["choice_scores"] = {"A": -1.0}
    bad["choice_token_counts"] = {"A": 1}
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(bad, item_scores_schema)


def test_hash_pattern_enforced(item_outputs_schema):
    """prompt_text_hash must be 64 hex chars; non-hex strings are rejected."""
    rows = _load_jsonl(FIXTURES_DIR / "tiny_item_outputs.jsonl")
    bad = copy.deepcopy(rows[0])
    bad["prompt_text_hash"] = "notahex"
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(bad, item_outputs_schema)
