"""Tests for 08_ranking_stability.py — SPEC_CPU_v0_2 §8.9, §9.4.

Covers:
    1. Kendall tau computation across condition pairs (known answer).
    2. Reversal fraction under a constructed scenario where two
       conditions disagree on the model ranking.
    3. Bootstrap determinism (same seed → identical CSV bytes).
    4. H3 threshold flag column.
    5. Output schema conformance for all three CSVs.
    6. End-to-end CLI invocation produces a figure file.
    7. Empty-input degeneracy (single benchmark, fewer than 2 models).
"""

from __future__ import annotations

import csv
import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


PAPER_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = PAPER_ROOT / "scripts"
FIXTURES_DIR = Path(__file__).parent / "fixtures"
SCRIPT_PATH = SCRIPTS_DIR / "08_ranking_stability.py"
ANALYSIS_CONFIG = FIXTURES_DIR / "configs" / "analysis_config.yaml"


# -- Importing the digit-prefixed script as a module ------------------------


def _load_script_module():
    spec = importlib.util.spec_from_file_location("ranking_stability", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


rs = _load_script_module()


# -- Helpers ----------------------------------------------------------------


CONDITION_FIELDS = [
    "condition_id", "tier", "benchmark_id", "subject_id", "model_id",
    "prompt_id", "seed_id", "fewshot_k", "scoring_rule_id",
    "num_items", "num_correct", "accuracy",
    "parseability_rate", "mean_generation_length_tokens", "condition_status",
]


def _make_condition_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CONDITION_FIELDS, lineterminator="\n")
        writer.writeheader()
        for r in rows:
            full = {k: "" for k in CONDITION_FIELDS}
            full.update(r)
            writer.writerow(full)


def _make_three_model_benchmark(
    benchmark_id: str,
    accuracies: dict[str, list[float]],
) -> list[dict]:
    """Build condition-level rows where the same conditions are evaluated
    on every model. Each list in `accuracies` must have the same length;
    they are aligned by index = condition number.
    """
    rows: list[dict] = []
    n_conditions = len(next(iter(accuracies.values())))
    for c in range(n_conditions):
        cond_id = f"{benchmark_id}__c{c:03d}"
        for model_id, scores in accuracies.items():
            rows.append({
                "condition_id": cond_id,
                "tier": "primary",
                "benchmark_id": benchmark_id,
                "subject_id": "",
                "model_id": model_id,
                "prompt_id": f"P{(c % 4) + 1}",
                "seed_id": str(42 + c),
                "fewshot_k": 5,
                "scoring_rule_id": "ll_norm",
                "num_items": 100,
                "num_correct": int(round(scores[c] * 100)),
                "accuracy": scores[c],
                "parseability_rate": 1.0,
                "mean_generation_length_tokens": 0.0,
                "condition_status": "complete",
            })
    return rows


# -- Tests ------------------------------------------------------------------


def test_kendall_tau_known_concordant():
    """Two condition rows that agree on ranking yield tau = 1."""
    pivot = pd.DataFrame({
        "model_a": [0.40, 0.50],
        "model_b": [0.50, 0.60],
        "model_c": [0.60, 0.70],
    }, index=["c1", "c2"])
    mean, median, lo, hi, n = rs.kendall_tau_across_condition_pairs(pivot)
    assert n == 1
    assert abs(mean - 1.0) < 1e-9
    assert abs(median - 1.0) < 1e-9


def test_kendall_tau_known_discordant():
    """Reverse rankings yield tau = -1."""
    pivot = pd.DataFrame({
        "model_a": [0.40, 0.70],
        "model_b": [0.50, 0.60],
        "model_c": [0.60, 0.50],
    }, index=["c1", "c2"])
    mean, median, _, _, n = rs.kendall_tau_across_condition_pairs(pivot)
    assert n == 1
    assert abs(mean + 1.0) < 1e-9


def test_pairwise_reversal_fraction_under_construction():
    """Condition c1 favors A, condition c2 favors B by the same magnitude;
    overall mean is tied, so every condition is a reversal relative to
    the zero-mean overall.

    We test the simpler, more realistic case where overall A > B and one
    condition reverses that ranking.
    """
    # Three conditions: c1, c2 favor A; c3 favors B by a small margin.
    pivot = pd.DataFrame({
        "model_a": [0.70, 0.65, 0.45],
        "model_b": [0.50, 0.55, 0.60],
    }, index=["c1", "c2", "c3"])
    rows = rs.pairwise_reversal_fraction(pivot, h3_threshold=0.10)
    assert len(rows) == 1
    r = rows[0]
    assert r["model_a"] == "model_a"
    assert r["model_b"] == "model_b"
    # overall mean diff = (0.70+0.65+0.45)/3 - (0.50+0.55+0.60)/3 = 0.6 - 0.55 = 0.05
    assert abs(r["overall_mean_diff"] - 0.05) < 1e-9
    # 1 of 3 conditions has sign(a-b) != sign(overall) → c3 is a reversal.
    assert abs(r["condition_reversal_fraction"] - (1.0 / 3.0)) < 1e-9
    assert r["n_conditions"] == 3
    # 0.333 > 0.10 → exceeds H3 threshold.
    assert r["exceeds_h3_threshold"] is True


def test_h3_threshold_below():
    """Reversal fraction below threshold yields exceeds_h3_threshold=False."""
    # Five conditions, all favor A.
    pivot = pd.DataFrame({
        "model_a": [0.70, 0.65, 0.62, 0.60, 0.58],
        "model_b": [0.50, 0.55, 0.52, 0.45, 0.48],
    }, index=[f"c{i}" for i in range(5)])
    rows = rs.pairwise_reversal_fraction(pivot, h3_threshold=0.10)
    assert len(rows) == 1
    r = rows[0]
    assert r["condition_reversal_fraction"] == 0.0
    assert r["exceeds_h3_threshold"] is False


def test_bootstrap_determinism_same_seed():
    """Same seed → identical bootstrap CIs."""
    pivot = pd.DataFrame({
        "model_a": np.linspace(0.4, 0.7, 8),
        "model_b": np.linspace(0.5, 0.6, 8),
    })
    r1 = rs.bootstrap_pairwise_win_prob(
        pivot, n_resamples=200, seed=42,
        ci_lower_pct=2.5, ci_upper_pct=97.5,
    )
    r2 = rs.bootstrap_pairwise_win_prob(
        pivot, n_resamples=200, seed=42,
        ci_lower_pct=2.5, ci_upper_pct=97.5,
    )
    assert r1 == r2

    r3 = rs.bootstrap_pairwise_win_prob(
        pivot, n_resamples=200, seed=43,
        ci_lower_pct=2.5, ci_upper_pct=97.5,
    )
    # Different seed → likely different mean (probabilistic, but with 200
    # resamples the chance of exact equality is vanishing).
    assert r3 != r1


def test_end_to_end_cli(tmp_path):
    """Smoke-test: invoke the script via CLI; check all four output files exist."""
    cond_csv = tmp_path / "condition_level.csv"
    rows = _make_three_model_benchmark(
        "arc_challenge",
        {
            "model_a": [0.70, 0.65, 0.62, 0.60, 0.58, 0.55, 0.52, 0.50],
            "model_b": [0.50, 0.55, 0.52, 0.45, 0.48, 0.50, 0.49, 0.51],
            "model_c": [0.40, 0.42, 0.44, 0.46, 0.48, 0.50, 0.52, 0.54],
        },
    )
    rows.extend(_make_three_model_benchmark(
        "hellaswag",
        {
            "model_a": [0.60, 0.61, 0.59, 0.58, 0.62],
            "model_b": [0.55, 0.54, 0.56, 0.57, 0.53],
            "model_c": [0.50, 0.49, 0.48, 0.51, 0.52],
        },
    ))
    _make_condition_csv(cond_csv, rows)

    out_dir = tmp_path / "ranking_stability"
    fig_dir = tmp_path / "figures"
    result = subprocess.run(
        [
            sys.executable, str(SCRIPT_PATH),
            "--condition-level", str(cond_csv),
            "--analysis-config", str(ANALYSIS_CONFIG),
            "--out-dir", str(out_dir),
            "--figures-dir", str(fig_dir),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"

    # All three CSVs and the figure must exist.
    assert (out_dir / "ranking_reversals.csv").exists()
    assert (out_dir / "pairwise_win_probabilities.csv").exists()
    assert (out_dir / "kendall_tau_by_benchmark.csv").exists()
    assert (fig_dir / "ranking_stability_by_benchmark.png").exists()

    # Schema check.
    rev = pd.read_csv(out_dir / "ranking_reversals.csv")
    assert list(rev.columns) == [
        "benchmark_id", "model_a", "model_b", "overall_mean_diff",
        "condition_reversal_fraction", "n_conditions", "exceeds_h3_threshold",
    ]
    # Two benchmarks × C(3,2) = 6 model pairs total.
    assert len(rev) == 6
    # Both benchmarks represented.
    assert set(rev["benchmark_id"].unique()) == {"arc_challenge", "hellaswag"}

    wp = pd.read_csv(out_dir / "pairwise_win_probabilities.csv")
    assert list(wp.columns) == [
        "benchmark_id", "model_a", "model_b", "p_a_gt_b_mean",
        "p_a_gt_b_ci_lower", "p_a_gt_b_ci_upper", "n_bootstrap",
    ]
    assert len(wp) == 6

    tau = pd.read_csv(out_dir / "kendall_tau_by_benchmark.csv")
    assert list(tau.columns) == [
        "benchmark_id", "tau_mean", "tau_median",
        "tau_ci_lower", "tau_ci_upper", "n_condition_pairs",
    ]
    assert len(tau) == 2


def test_too_few_models_skipped(tmp_path):
    """A benchmark with fewer than 2 models with data is skipped, not crashed."""
    cond_csv = tmp_path / "condition_level.csv"
    # Only one model per benchmark.
    rows = _make_three_model_benchmark(
        "single_model_bench",
        {"model_a": [0.5, 0.55, 0.6]},
    )
    _make_condition_csv(cond_csv, rows)

    out_dir = tmp_path / "ranking_stability"
    fig_dir = tmp_path / "figures"
    result = subprocess.run(
        [
            sys.executable, str(SCRIPT_PATH),
            "--condition-level", str(cond_csv),
            "--analysis-config", str(ANALYSIS_CONFIG),
            "--out-dir", str(out_dir),
            "--figures-dir", str(fig_dir),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"
    rev = pd.read_csv(out_dir / "ranking_reversals.csv")
    assert len(rev) == 0
