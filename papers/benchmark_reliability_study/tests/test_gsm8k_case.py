"""Tests for 10_gsm8k_case.py — SPEC_CPU_v0_2 §8.11.

Covers:
    1. Tolerance computation (sem, tolerance = 2*sem) on a hand-built case.
    2. Decimal-place licensing categorical (§9.1) for all three branches.
    3. Extraction sensitivity delta = permissive - strict.
    4. Parseability aggregation per (model, extraction_variant).
    5. End-to-end CLI emits all three CSVs.
    6. Tolerance schedule output schema conformance.
"""

from __future__ import annotations

import csv
import importlib.util
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest


PAPER_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = PAPER_ROOT / "scripts"
FIXTURES_DIR = Path(__file__).parent / "fixtures"
SCRIPT_PATH = SCRIPTS_DIR / "10_gsm8k_case.py"
ANALYSIS_CONFIG = FIXTURES_DIR / "configs" / "analysis_config.yaml"
GSM8K_ITEM = FIXTURES_DIR / "gsm8k_item_level.parquet"
GSM8K_COND = FIXTURES_DIR / "gsm8k_condition_level.csv"


# -- Build / import the script as a module ----------------------------------


@pytest.fixture(scope="session", autouse=True)
def ensure_fixtures_built():
    """Ensure GSM8K fixtures are present."""
    if not GSM8K_ITEM.exists() or not GSM8K_COND.exists():
        result = subprocess.run(
            [sys.executable, str(FIXTURES_DIR / "make_fixtures.py")],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"Fixture build failed: {result.stderr}"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("gsm8k_case", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


mod = _load_script_module()


# -- Tests ------------------------------------------------------------------


def test_licensed_precision_three_levels():
    """SPEC §9.1 mapping function exhausts three branches."""
    assert mod.licensed_precision(0.0001) == "three_decimal_accuracy"
    assert mod.licensed_precision(0.0004999) == "three_decimal_accuracy"
    assert mod.licensed_precision(0.0005) == "two_decimal_accuracy"
    assert mod.licensed_precision(0.001) == "two_decimal_accuracy"
    assert mod.licensed_precision(0.0049999) == "two_decimal_accuracy"
    assert mod.licensed_precision(0.005) == "interval_required"
    assert mod.licensed_precision(0.05) == "interval_required"


def test_tolerance_computed_from_condition_std():
    """Hand-built (model, rule) cell: tolerance equals 2 * std(scores, ddof=1)."""
    rows = [
        {
            "condition_id": "c1", "tier": "secondary", "benchmark_id": "gsm8k",
            "subject_id": "", "model_id": "M1", "prompt_id": "P1",
            "seed_id": "42", "fewshot_k": 5,
            "scoring_rule_id": "generate_parse_strict",
            "num_items": 100, "num_correct": 50, "accuracy": 0.50,
            "parseability_rate": 1.0, "mean_generation_length_tokens": 0.0,
            "condition_status": "complete",
        },
        {
            "condition_id": "c2", "tier": "secondary", "benchmark_id": "gsm8k",
            "subject_id": "", "model_id": "M1", "prompt_id": "P2",
            "seed_id": "42", "fewshot_k": 5,
            "scoring_rule_id": "generate_parse_strict",
            "num_items": 100, "num_correct": 60, "accuracy": 0.60,
            "parseability_rate": 1.0, "mean_generation_length_tokens": 0.0,
            "condition_status": "complete",
        },
        {
            "condition_id": "c3", "tier": "secondary", "benchmark_id": "gsm8k",
            "subject_id": "", "model_id": "M1", "prompt_id": "P3",
            "seed_id": "42", "fewshot_k": 5,
            "scoring_rule_id": "generate_parse_strict",
            "num_items": 100, "num_correct": 70, "accuracy": 0.70,
            "parseability_rate": 1.0, "mean_generation_length_tokens": 0.0,
            "condition_status": "complete",
        },
    ]
    df = pd.DataFrame(rows)
    result = mod.compute_tolerance_schedule(
        df, n_resamples=200, seed=42, ci_lower_pct=2.5, ci_upper_pct=97.5
    )
    assert len(result) == 1
    r = result[0]
    # std([0.5, 0.6, 0.7], ddof=1) = 0.1; tolerance = 2 * 0.1 = 0.2.
    assert abs(r["sem"] - 0.1) < 1e-9
    assert abs(r["tolerance"] - 0.2) < 1e-9
    # 0.2 >= 0.005 → interval_required.
    assert r["licensed_precision"] == "interval_required"


def test_extraction_sensitivity_delta():
    """Delta = accuracy_permissive - accuracy_strict per (model, prompt, seed)."""
    rows = [
        {
            "condition_id": "c1_strict", "tier": "secondary", "benchmark_id": "gsm8k",
            "subject_id": "", "model_id": "M1", "prompt_id": "P1",
            "seed_id": "42", "fewshot_k": 5,
            "scoring_rule_id": "generate_parse_strict",
            "num_items": 100, "num_correct": 40, "accuracy": 0.40,
            "parseability_rate": 0.5, "mean_generation_length_tokens": 0.0,
            "condition_status": "complete",
        },
        {
            "condition_id": "c1_perm", "tier": "secondary", "benchmark_id": "gsm8k",
            "subject_id": "", "model_id": "M1", "prompt_id": "P1",
            "seed_id": "42", "fewshot_k": 5,
            "scoring_rule_id": "generate_parse_permissive",
            "num_items": 100, "num_correct": 70, "accuracy": 0.70,
            "parseability_rate": 0.9, "mean_generation_length_tokens": 0.0,
            "condition_status": "complete",
        },
    ]
    df = pd.DataFrame(rows)
    result = mod.compute_extraction_sensitivity(
        df, n_resamples=100, seed=42, ci_lower_pct=2.5, ci_upper_pct=97.5
    )
    assert len(result) == 1
    r = result[0]
    assert r["model_id"] == "M1"
    assert r["prompt_id"] == "P1"
    assert abs(float(r["accuracy_strict"]) - 0.40) < 1e-9
    assert abs(float(r["accuracy_permissive"]) - 0.70) < 1e-9
    assert abs(float(r["delta"]) - 0.30) < 1e-9


def test_parseability_aggregation():
    """Mean parseability per (model, extraction_variant) aggregates correctly."""
    df = pd.read_csv(GSM8K_COND)
    rows = mod.compute_parseability(df)
    # 2 models × 2 scoring rules → 4 rows.
    assert len(rows) == 4
    # By construction: strict parseability = 0.5, permissive = 0.75.
    by_key = {(r["model_id"], r["extraction_variant"]): r for r in rows}
    for model_id in ("test_model_1", "test_model_2"):
        s = by_key[(model_id, "generate_parse_strict")]
        p = by_key[(model_id, "generate_parse_permissive")]
        assert abs(float(s["mean_parseability"]) - 0.5) < 1e-9
        assert abs(float(p["mean_parseability"]) - 0.75) < 1e-9
        # 4 conditions per (model, rule) cell in the fixture: 2 prompts × 2 seeds.
        assert int(s["n_conditions"]) == 4
        assert int(p["n_conditions"]) == 4


def test_end_to_end_cli(tmp_path):
    """CLI invocation emits all three artifacts with valid schemas."""
    out_dir = tmp_path / "gsm8k_case"
    result = subprocess.run(
        [
            sys.executable, str(SCRIPT_PATH),
            "--item-level", str(GSM8K_ITEM),
            "--condition-level", str(GSM8K_COND),
            "--analysis-config", str(ANALYSIS_CONFIG),
            "--out-dir", str(out_dir),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"

    tol_path = out_dir / "gsm8k_tolerance_schedule.csv"
    ext_path = out_dir / "gsm8k_extraction_sensitivity.csv"
    pars_path = out_dir / "gsm8k_parseability.csv"
    assert tol_path.exists()
    assert ext_path.exists()
    assert pars_path.exists()

    # Schema checks.
    tol = pd.read_csv(tol_path)
    assert list(tol.columns) == [
        "model_id", "extraction_variant", "sem", "tolerance",
        "tolerance_ci_lower", "tolerance_ci_upper", "licensed_precision",
    ]
    # Each row's licensed_precision is one of the three canonical levels.
    for v in tol["licensed_precision"]:
        assert v in {
            "three_decimal_accuracy",
            "two_decimal_accuracy",
            "interval_required",
        }

    ext = pd.read_csv(ext_path)
    assert list(ext.columns) == [
        "model_id", "prompt_id", "seed_id", "accuracy_strict",
        "accuracy_permissive", "delta", "delta_ci_lower", "delta_ci_upper",
    ]
    # By fixture construction, delta = 0.25 everywhere.
    for d in ext["delta"]:
        assert abs(float(d) - 0.25) < 1e-9

    pars = pd.read_csv(pars_path)
    assert list(pars.columns) == [
        "model_id", "extraction_variant", "mean_parseability", "n_conditions",
    ]


def test_extraction_sensitivity_skips_when_one_rule_missing():
    """If only one scoring rule is present, extraction_sensitivity returns []."""
    rows = [
        {
            "condition_id": "c1_strict", "tier": "secondary", "benchmark_id": "gsm8k",
            "subject_id": "", "model_id": "M1", "prompt_id": "P1",
            "seed_id": "42", "fewshot_k": 5,
            "scoring_rule_id": "generate_parse_strict",
            "num_items": 100, "num_correct": 40, "accuracy": 0.40,
            "parseability_rate": 0.5, "mean_generation_length_tokens": 0.0,
            "condition_status": "complete",
        },
    ]
    df = pd.DataFrame(rows)
    result = mod.compute_extraction_sensitivity(
        df, n_resamples=10, seed=0, ci_lower_pct=2.5, ci_upper_pct=97.5
    )
    assert result == []
