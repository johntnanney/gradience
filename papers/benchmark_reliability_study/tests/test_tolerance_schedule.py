"""Tests for 07_tolerance_schedule.py (SPEC §8.8, §9.1, §9.2, §9.3).

Strategy:
    - Unit tests for licensed_precision, sem_from_variance_components,
      bootstrap_tolerance_ci with known-answer fixtures.
    - Integration tests for compute_per_cell_tolerance + test_h1 with
      constructed inputs.
    - End-to-end CLI tests via subprocess.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PAPER_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = PAPER_ROOT / "scripts" / "07_tolerance_schedule.py"
FIXTURES_DIR = PAPER_ROOT / "tests" / "fixtures"


def _import_script():
    spec = importlib.util.spec_from_file_location("tol_script", SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# -- licensed_precision (§9.1) ------------------------------------------------


def test_licensed_precision_three_decimal_boundary():
    mod = _import_script()
    # Just under 0.0005 → three decimals
    assert mod.licensed_precision(0.0004999) == "three_decimal_accuracy"
    # Exactly 0.0005 → two decimals (boundary goes to the next tier)
    assert mod.licensed_precision(0.0005) == "two_decimal_accuracy"


def test_licensed_precision_two_decimal_boundary():
    mod = _import_script()
    assert mod.licensed_precision(0.004999) == "two_decimal_accuracy"
    assert mod.licensed_precision(0.005) == "interval_required"


def test_licensed_precision_extreme_values():
    mod = _import_script()
    assert mod.licensed_precision(0.0) == "three_decimal_accuracy"
    assert mod.licensed_precision(1.0) == "interval_required"


def test_licensed_precision_nan_returns_interval_required():
    mod = _import_script()
    assert mod.licensed_precision(float("nan")) == "interval_required"


def test_licensed_precision_all_three_levels_reachable():
    mod = _import_script()
    assert mod.licensed_precision(0.0001) == "three_decimal_accuracy"
    assert mod.licensed_precision(0.001) == "two_decimal_accuracy"
    assert mod.licensed_precision(0.05) == "interval_required"


# -- SEM math (§9.2) ---------------------------------------------------------


def test_sem_single_sum_of_components():
    mod = _import_script()
    sem = mod.sem_from_variance_components(
        var_prompt=0.01, var_seed=0.02, var_scoring_rule=0.03,
        var_residual=0.04, averaging="single", n_conditions=24,
    )
    expected = np.sqrt(0.01 + 0.02 + 0.03 + 0.04)
    assert sem == pytest.approx(expected, rel=1e-9)


def test_sem_within_rule_excludes_scoring_rule():
    mod = _import_script()
    sem = mod.sem_from_variance_components(
        var_prompt=0.01, var_seed=0.02, var_scoring_rule=0.03,
        var_residual=0.04, averaging="within_rule", n_conditions=24,
    )
    expected = np.sqrt(0.01 + 0.02 + 0.04)
    assert sem == pytest.approx(expected, rel=1e-9)


def test_sem_prompt_avg_shrinks_by_sqrt_4():
    mod = _import_script()
    sem_single = mod.sem_from_variance_components(
        var_prompt=0.1, var_seed=0.0, var_scoring_rule=0.0,
        var_residual=0.0, averaging="single", n_conditions=24,
    )
    sem_pavg = mod.sem_from_variance_components(
        var_prompt=0.1, var_seed=0.0, var_scoring_rule=0.0,
        var_residual=0.0, averaging="prompt_averaged", n_conditions=24,
    )
    assert sem_pavg == pytest.approx(sem_single / np.sqrt(4), rel=1e-9)


def test_sem_full_design_shrinks_by_sqrt_n_conditions():
    mod = _import_script()
    sem_single = mod.sem_from_variance_components(
        var_prompt=0.1, var_seed=0.0, var_scoring_rule=0.0,
        var_residual=0.0, averaging="single", n_conditions=24,
    )
    sem_full = mod.sem_from_variance_components(
        var_prompt=0.1, var_seed=0.0, var_scoring_rule=0.0,
        var_residual=0.0, averaging="full_design", n_conditions=24,
    )
    assert sem_full == pytest.approx(sem_single / np.sqrt(24), rel=1e-9)


def test_sem_invalid_averaging_raises():
    mod = _import_script()
    with pytest.raises(ValueError):
        mod.sem_from_variance_components(
            0.01, 0.01, 0.01, 0.01, "unknown_scheme", 24
        )


def test_tolerance_from_sem_is_double():
    mod = _import_script()
    assert mod.tolerance_from_sem(0.01) == pytest.approx(0.02, rel=1e-9)


# -- Bootstrap CI (§9.3) -----------------------------------------------------


def test_bootstrap_determinism_same_seed():
    mod = _import_script()
    scores = np.array([0.5, 0.52, 0.48, 0.51, 0.49])
    tol_a, lo_a, up_a = mod.bootstrap_tolerance_ci(
        scores, n_resamples=1000, seed=42
    )
    tol_b, lo_b, up_b = mod.bootstrap_tolerance_ci(
        scores, n_resamples=1000, seed=42
    )
    assert tol_a == pytest.approx(tol_b, rel=1e-12)
    assert lo_a == pytest.approx(lo_b, rel=1e-12)
    assert up_a == pytest.approx(up_b, rel=1e-12)


def test_bootstrap_different_seeds_differ():
    mod = _import_script()
    # Use richer sample (24 scores) to avoid degenerate bootstrap
    # distribution where the 2.5 percentile lands on the same discrete
    # value across seeds.
    rng = np.random.default_rng(100)
    scores = rng.normal(0.5, 0.08, size=24)
    _, lo_a, _ = mod.bootstrap_tolerance_ci(
        scores, n_resamples=2000, seed=42
    )
    _, lo_b, _ = mod.bootstrap_tolerance_ci(
        scores, n_resamples=2000, seed=12345
    )
    # Very small chance of exact equality with 2000 resamples on 24
    # continuous scores.
    assert lo_a != lo_b


def test_bootstrap_ci_covers_point_estimate():
    mod = _import_script()
    rng = np.random.default_rng(0)
    scores = rng.normal(0.5, 0.05, size=20)
    tol, lo, up = mod.bootstrap_tolerance_ci(
        scores, n_resamples=2000, seed=42
    )
    # 95% CI should contain the point estimate for well-behaved samples
    assert lo <= tol <= up


def test_bootstrap_tolerance_scales_with_sd():
    mod = _import_script()
    # Higher-variance scores should produce larger tolerance
    low_var = np.array([0.50, 0.51, 0.49, 0.50])
    high_var = np.array([0.30, 0.70, 0.20, 0.80])
    tol_low, _, _ = mod.bootstrap_tolerance_ci(
        low_var, n_resamples=500, seed=42
    )
    tol_high, _, _ = mod.bootstrap_tolerance_ci(
        high_var, n_resamples=500, seed=42
    )
    assert tol_high > tol_low


def test_bootstrap_single_condition_returns_nan():
    mod = _import_script()
    tol, lo, up = mod.bootstrap_tolerance_ci(
        np.array([0.5]), n_resamples=100, seed=42
    )
    assert np.isnan(tol)
    assert np.isnan(lo)
    assert np.isnan(up)


# -- H1 decision rule (§8.8) -------------------------------------------------


def _build_analysis_config(threshold=0.005, n_required=3):
    from gradience_study.config import (
        AnalysisConfig, BootstrapConfig, ConvergenceTriggersConfig,
        MixedEffectsCascadeConfig, ToleranceConfig,
    )
    return AnalysisConfig(
        bootstrap=BootstrapConfig(
            n_resamples=1000, random_seed=42,
            ci_lower_percentile=2.5, ci_upper_percentile=97.5,
        ),
        mixed_effects_cascade=MixedEffectsCascadeConfig(levels={}),
        convergence_triggers=ConvergenceTriggersConfig(
            max_singular_warning="trigger_fallback",
            hessian_non_pd="trigger_fallback",
            iteration_limit_exceeded="trigger_fallback",
            gradient_norm_above=1e-3,
        ),
        tolerance=ToleranceConfig(
            decision_rule_threshold_h1=threshold,
            benchmarks_required_for_h1=n_required,
        ),
        h2_generalizability_threshold=0.80,
        h3_ranking_reversal_threshold=0.10,
        h4_mmlu_interaction_threshold=0.10,
    )


def test_h1_confirmed_when_enough_benchmarks_exceed():
    mod = _import_script()
    # 4 benchmarks exceed, threshold is 3
    summary = pd.DataFrame([
        {"benchmark_id": "b1",
         "exceeds_h1_threshold_ci_lower": True},
        {"benchmark_id": "b2",
         "exceeds_h1_threshold_ci_lower": True},
        {"benchmark_id": "b3",
         "exceeds_h1_threshold_ci_lower": True},
        {"benchmark_id": "b4",
         "exceeds_h1_threshold_ci_lower": True},
        {"benchmark_id": "b5",
         "exceeds_h1_threshold_ci_lower": False},
    ])
    result = mod.test_h1(summary, _build_analysis_config())
    assert result["h1_confirmed"] is True
    assert result["n_exceeding"] == 4
    assert result["n_required"] == 3


def test_h1_not_confirmed_when_insufficient_benchmarks():
    mod = _import_script()
    summary = pd.DataFrame([
        {"benchmark_id": "b1",
         "exceeds_h1_threshold_ci_lower": True},
        {"benchmark_id": "b2",
         "exceeds_h1_threshold_ci_lower": True},
        {"benchmark_id": "b3",
         "exceeds_h1_threshold_ci_lower": False},
        {"benchmark_id": "b4",
         "exceeds_h1_threshold_ci_lower": False},
        {"benchmark_id": "b5",
         "exceeds_h1_threshold_ci_lower": False},
    ])
    result = mod.test_h1(summary, _build_analysis_config())
    assert result["h1_confirmed"] is False
    assert result["n_exceeding"] == 2


def test_h1_result_has_complete_schema():
    mod = _import_script()
    summary = pd.DataFrame([
        {"benchmark_id": "b1", "exceeds_h1_threshold_ci_lower": False},
    ])
    result = mod.test_h1(summary, _build_analysis_config())
    required_keys = {
        "h1_hypothesis", "threshold", "n_required", "n_exceeding",
        "benchmarks_exceeding", "h1_confirmed",
    }
    assert required_keys.issubset(set(result.keys()))


# -- End-to-end CLI ----------------------------------------------------------


def _make_vc_df():
    """Construct a minimal aggregate_vc.csv DataFrame for 2 benchmarks × 2 models."""
    rows = []
    for b in ["b1", "b2"]:
        for m in ["m1", "m2"]:
            rows.extend([
                {"benchmark_id": b, "model_id": m, "source": "prompt",
                 "variance": 0.003, "proportion": 0.3, "df": 23},
                {"benchmark_id": b, "model_id": m, "source": "seed",
                 "variance": 0.001, "proportion": 0.1, "df": 23},
                {"benchmark_id": b, "model_id": m, "source": "scoring_rule",
                 "variance": 0.002, "proportion": 0.2, "df": 23},
                {"benchmark_id": b, "model_id": m, "source": "residual",
                 "variance": 0.004, "proportion": 0.4, "df": 23},
            ])
    return pd.DataFrame(rows)


def _make_cond_df():
    """Condition-level data for the same 2 benchmarks × 2 models."""
    rng = np.random.default_rng(42)
    rows = []
    for b in ["b1", "b2"]:
        for m in ["m1", "m2"]:
            for p in ["P1", "P2", "P3", "P4"]:
                for s in ["s42", "s123", "s2024"]:
                    for r in ["ll_norm", "generate_parse"]:
                        acc = float(np.clip(rng.normal(0.6, 0.1), 0, 1))
                        rows.append({
                            "condition_id": f"{b}_{m}_{p}_{s}_{r}",
                            "tier": "primary",
                            "benchmark_id": b,
                            "subject_id": "",
                            "model_id": m,
                            "prompt_id": p,
                            "seed_id": s,
                            "fewshot_k": 5,
                            "scoring_rule_id": r,
                            "num_items": 100,
                            "num_correct": int(acc * 100),
                            "accuracy": acc,
                            "parseability_rate": 1.0,
                            "mean_generation_length_tokens": 10.0,
                            "condition_status": "complete",
                        })
    return pd.DataFrame(rows)


@pytest.fixture
def paper_with_vc_and_conditions(tmp_path):
    root = tmp_path / "paper_root"
    root.mkdir()
    (root / "configs").mkdir()
    (root / "analysis" / "variance_components").mkdir(parents=True)
    (root / "runs" / "normalized").mkdir(parents=True)

    import shutil
    for src in (FIXTURES_DIR / "configs").glob("*.yaml"):
        shutil.copy(src, root / "configs" / src.name)

    _make_vc_df().to_csv(
        root / "analysis" / "variance_components" / "aggregate_vc.csv",
        index=False,
    )
    _make_cond_df().to_csv(
        root / "runs" / "normalized" / "condition_level_primary.csv",
        index=False,
    )
    return root


def test_cli_end_to_end_produces_all_outputs(paper_with_vc_and_conditions):
    root = paper_with_vc_and_conditions
    out_dir = root / "analysis" / "tolerance_schedules"

    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH),
         "--condition-level",
         str(root / "runs/normalized/condition_level_primary.csv"),
         "--variance-components",
         str(root / "analysis/variance_components/aggregate_vc.csv"),
         "--analysis-config", str(root / "configs/study_config.yaml"),
         "--out-dir", str(out_dir)],
        capture_output=True, text=True,
    )

    assert result.returncode == 0, (
        f"Unexpected exit {result.returncode}\nSTDERR:\n{result.stderr}"
    )
    assert (out_dir / "tolerance_by_cell.csv").exists()
    assert (out_dir / "tolerance_by_benchmark_summary.csv").exists()
    assert (out_dir / "h1_test.json").exists()


def test_cli_tolerance_by_cell_schema(paper_with_vc_and_conditions):
    root = paper_with_vc_and_conditions
    out_dir = root / "analysis" / "tolerance_schedules"
    subprocess.run(
        [sys.executable, str(SCRIPT_PATH),
         "--condition-level",
         str(root / "runs/normalized/condition_level_primary.csv"),
         "--variance-components",
         str(root / "analysis/variance_components/aggregate_vc.csv"),
         "--analysis-config", str(root / "configs/study_config.yaml"),
         "--out-dir", str(out_dir)],
        capture_output=True, text=True,
    )
    cells = pd.read_csv(out_dir / "tolerance_by_cell.csv")
    required = {
        "benchmark_id", "model_id", "scoring_rule_id",
        "sem_single", "tolerance_single",
        "tolerance_single_ci_lower", "tolerance_single_ci_upper",
        "licensed_precision_single", "licensed_precision_full_design",
    }
    assert required.issubset(set(cells.columns))
    # Every licensed_precision value must be one of the three valid levels
    valid = {"three_decimal_accuracy", "two_decimal_accuracy", "interval_required"}
    assert set(cells["licensed_precision_single"].unique()).issubset(valid)


def test_cli_h1_test_json_schema(paper_with_vc_and_conditions):
    root = paper_with_vc_and_conditions
    out_dir = root / "analysis" / "tolerance_schedules"
    subprocess.run(
        [sys.executable, str(SCRIPT_PATH),
         "--condition-level",
         str(root / "runs/normalized/condition_level_primary.csv"),
         "--variance-components",
         str(root / "analysis/variance_components/aggregate_vc.csv"),
         "--analysis-config", str(root / "configs/study_config.yaml"),
         "--out-dir", str(out_dir)],
        capture_output=True, text=True,
    )
    with open(out_dir / "h1_test.json") as f:
        h1 = json.load(f)
    required = {"h1_hypothesis", "threshold", "n_required",
                "n_exceeding", "benchmarks_exceeding", "h1_confirmed"}
    assert required.issubset(set(h1.keys()))
    assert isinstance(h1["h1_confirmed"], bool)
    assert isinstance(h1["benchmarks_exceeding"], list)


def test_cli_missing_variance_components_fails_2(tmp_path):
    """Missing VC CSV should exit 2."""
    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH),
         "--condition-level", "/nonexistent/cond.csv",
         "--variance-components", str(tmp_path / "nonexistent_vc.csv"),
         "--analysis-config", str(tmp_path / "nonexistent.yaml"),
         "--out-dir", str(tmp_path / "out")],
        capture_output=True, text=True,
    )
    assert result.returncode == 2
