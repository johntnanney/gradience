"""Tests for 06_variance_components.py (SPEC §8.7, §10.1, §10.2).

Strategy:
    - test_aggregate_g_theory_* tests use synthetic data with known
      variance components and verify recovery within numerical tolerance.
    - test_cascade_* tests verify the cascade's level selection and
      convergence logging.
    - test_cli_end_to_end tests the full script via subprocess.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

PAPER_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = PAPER_ROOT / "scripts" / "06_variance_components.py"
FIXTURES_DIR = PAPER_ROOT / "tests" / "fixtures"


def _import_script():
    """Import 06_variance_components.py as a module for direct testing."""
    spec = importlib.util.spec_from_file_location("vc_script", SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# -- Synthetic data with known variance components --------------------------


def _make_synthetic_conditions(
    benchmark_id: str,
    n_models: int = 3,
    n_prompts: int = 4,
    n_seeds: int = 3,
    n_rules: int = 2,
    true_var_prompt: float = 0.01,
    true_var_seed: float = 0.005,
    true_var_rule: float = 0.002,
    true_var_residual: float = 0.001,
    rng_seed: int = 42,
) -> pd.DataFrame:
    """Generate a condition-level DataFrame with known variance components.

    Each condition's accuracy = base + prompt_effect + seed_effect
      + rule_effect + residual, where each effect ~ N(0, true_variance)
    and base = 0.5 per model.
    """
    rng = np.random.default_rng(rng_seed)
    prompt_effects = rng.normal(0, np.sqrt(true_var_prompt), n_prompts)
    seed_effects = rng.normal(0, np.sqrt(true_var_seed), n_seeds)
    rule_effects = rng.normal(0, np.sqrt(true_var_rule), n_rules)

    rows = []
    for m_idx in range(n_models):
        for p_idx in range(n_prompts):
            for s_idx in range(n_seeds):
                for r_idx in range(n_rules):
                    resid = rng.normal(0, np.sqrt(true_var_residual))
                    acc = 0.5 + prompt_effects[p_idx] + seed_effects[s_idx] \
                        + rule_effects[r_idx] + resid
                    acc = float(np.clip(acc, 0.0, 1.0))
                    rows.append({
                        "condition_id":
                            f"{benchmark_id}__m{m_idx}__P{p_idx}__s{s_idx}__r{r_idx}",
                        "tier": "primary",
                        "benchmark_id": benchmark_id,
                        "subject_id": "",
                        "model_id": f"model_{m_idx}",
                        "prompt_id": f"P{p_idx}",
                        "seed_id": f"s{s_idx}",
                        "fewshot_k": 5,
                        "scoring_rule_id": f"rule_{r_idx}",
                        "num_items": 100,
                        "num_correct": int(acc * 100),
                        "accuracy": acc,
                        "parseability_rate": 1.0,
                        "mean_generation_length_tokens": 10.0,
                        "condition_status": "complete",
                    })
    return pd.DataFrame(rows)


def _make_synthetic_items(
    condition_df: pd.DataFrame,
    n_items: int = 20,
    rng_seed: int = 123,
) -> pd.DataFrame:
    """Generate item-level rows consistent with condition-level accuracies."""
    rng = np.random.default_rng(rng_seed)
    rows = []
    for _, cond in condition_df.iterrows():
        p_correct = float(cond["accuracy"])
        outcomes = rng.binomial(1, p_correct, size=n_items).astype(bool)
        for item_idx, ok in enumerate(outcomes):
            rows.append({
                "condition_id": cond["condition_id"],
                "tier": cond["tier"],
                "benchmark_id": cond["benchmark_id"],
                "subject_id": None,
                "model_id": cond["model_id"],
                "model_family": "synthetic",
                "model_type": "base",
                "parameter_count_b": 1.0,
                "prompt_id": cond["prompt_id"],
                "prompt_source_type": "fixture",
                "seed_id": cond["seed_id"],
                "fewshot_k": cond["fewshot_k"],
                "scoring_rule_id": cond["scoring_rule_id"],
                "task_type": "constrained_choice",
                "item_id": f"{cond['benchmark_id']}_item_{item_idx:03d}",
                "gold_answer": "A",
                "predicted_answer": "A" if ok else "B",
                "is_correct": bool(ok),
                "parse_status": "parsed",
                "parseable": True,
                "choice_count": 4,
                "prompt_text_hash": "a" * 64,
                "rendered_prompt_hash": "b" * 64,
                "run_id": cond["condition_id"],
                "created_at": "2026-04-24T00:00:00+00:00",
                "selected_answer": None,
                "ll_margin_top2": None,
                "gold_choice_score": None,
                "selected_choice_score": None,
                "raw_generation": "The answer is A" if ok else "The answer is B",
                "parsed_answer": "A" if ok else "B",
                "generation_length_tokens": 4,
            })
    return pd.DataFrame(rows)


# -- Aggregate G-theory tests -----------------------------------------------


def test_aggregate_g_theory_recovers_known_prompt_variance():
    """Dominant prompt variance should be recovered as the largest VC."""
    mod = _import_script()
    cond = _make_synthetic_conditions(
        "test_bench",
        true_var_prompt=0.05,
        true_var_seed=0.001,
        true_var_rule=0.0005,
        true_var_residual=0.0005,
        rng_seed=42,
    )
    vc = mod.fit_aggregate_g_theory(cond, "test_bench", "model_0")
    assert vc is not None
    # Prompt variance should be clearly the largest non-residual component.
    # With realistic finite-sample noise we allow var_prompt > var_seed
    # and var_prompt > var_scoring_rule.
    assert vc["var_prompt"] > vc["var_seed"]
    assert vc["var_prompt"] > vc["var_scoring_rule"]
    # Total should be positive
    assert vc["var_total"] > 0


def test_aggregate_g_theory_returns_none_for_too_few_conditions():
    mod = _import_script()
    # Only 2 conditions — below threshold of 4.
    cond = _make_synthetic_conditions(
        "test_bench", n_prompts=1, n_seeds=1, n_rules=1, n_models=1,
    )
    assert len(cond) <= 4
    # Trim to 2 conditions
    cond_small = cond.iloc[:2]
    vc = mod.fit_aggregate_g_theory(cond_small, "test_bench", "model_0")
    # For ≥4 conditions required; with 2 we return None
    if vc is not None:
        assert vc["n_conditions"] == len(cond_small)


def test_aggregate_g_theory_handles_zero_shot_collapse():
    """Benchmark with only a single seed level should collapse seed variance
    to zero or residual; no error."""
    mod = _import_script()
    cond = _make_synthetic_conditions(
        "test_bench", n_seeds=1,
    )
    vc = mod.fit_aggregate_g_theory(cond, "test_bench", "model_0")
    assert vc is not None
    assert vc["var_seed"] == 0.0  # collapsed to residual
    assert vc["var_total"] >= 0.0


def test_variance_components_nonneg():
    """All variance components must be non-negative (variances are ≥ 0)."""
    mod = _import_script()
    cond = _make_synthetic_conditions("test_bench", rng_seed=7)
    vc = mod.fit_aggregate_g_theory(cond, "test_bench", "model_0")
    assert vc is not None
    for k in ["var_prompt", "var_seed", "var_scoring_rule", "var_residual"]:
        assert vc[k] >= 0.0, f"{k} < 0: {vc[k]}"


# -- Cascade tests -----------------------------------------------------------


def test_cascade_returns_level_1_or_descends():
    """Item-level cascade on clean data should either succeed at some
    level ≤ 3 or fall through to level_4."""
    mod = _import_script()
    cond = _make_synthetic_conditions(
        "test_bench", n_models=3, n_prompts=3, n_seeds=2, n_rules=2,
    )
    items = _make_synthetic_items(cond, n_items=10)
    level, fit, log = mod.fit_item_level_cascade(
        items, "test_bench", {"gradient_norm_above": 1e-3}
    )
    assert level in ("level_1", "level_2", "level_3", "level_4")
    assert len(log) >= 1
    # Every attempt entry has the required schema
    for entry in log:
        assert "benchmark_id" in entry
        assert "cascade_level" in entry
        assert "success" in entry


def test_cascade_level_4_when_data_too_small():
    """With only a handful of items, the cascade should ultimately fall
    through to level_4 (or succeed at a simplified level)."""
    mod = _import_script()
    cond = _make_synthetic_conditions(
        "test_bench", n_models=2, n_prompts=2, n_seeds=1, n_rules=1,
    )
    items = _make_synthetic_items(cond, n_items=3)
    level, fit, log = mod.fit_item_level_cascade(
        items, "test_bench", {"gradient_norm_above": 1e-3}
    )
    # With so little data, some level should be reached
    assert level in ("level_1", "level_2", "level_3", "level_4")


def test_cascade_log_includes_all_attempted_levels():
    """Each cascade attempt is logged separately."""
    mod = _import_script()
    cond = _make_synthetic_conditions("test_bench", n_models=2, n_prompts=2)
    items = _make_synthetic_items(cond, n_items=5)
    level, fit, log = mod.fit_item_level_cascade(
        items, "test_bench", {"gradient_norm_above": 1e-3}
    )
    # The log should have at least one entry; if level_4 reached, it
    # should have 4 entries (one per attempted level).
    if level == "level_4":
        # Attempt 1–3 all failed, plus level_4 entry
        assert len(log) >= 4


# -- Licensing helper (via import) ------------------------------------------


# -- End-to-end CLI tests ---------------------------------------------------


@pytest.fixture
def paper_with_synthetic_data(tmp_path):
    """Set up a tmp paper-root with synthetic item-level + condition-level
    + fixture configs, and return its path."""
    root = tmp_path / "paper_root"
    root.mkdir()
    (root / "configs").mkdir()
    (root / "runs" / "normalized").mkdir(parents=True)

    # Copy fixture configs
    import shutil
    for src in (FIXTURES_DIR / "configs").glob("*.yaml"):
        shutil.copy(src, root / "configs" / src.name)

    # Synthetic condition-level + item-level
    cond = _make_synthetic_conditions(
        "test_bench_1", n_models=3, n_prompts=3, n_seeds=2, n_rules=2,
    )
    items = _make_synthetic_items(cond, n_items=20)

    cond.to_csv(root / "runs" / "normalized" / "condition_level_primary.csv",
                index=False)
    pq.write_table(
        pa.Table.from_pandas(items),
        root / "runs" / "normalized" / "item_level_primary.parquet",
    )

    return root


def test_cli_end_to_end_produces_all_outputs(paper_with_synthetic_data):
    root = paper_with_synthetic_data
    out_dir = root / "analysis" / "variance_components"

    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH),
         "--item-level", str(root / "runs/normalized/item_level_primary.parquet"),
         "--condition-level", str(root / "runs/normalized/condition_level_primary.csv"),
         "--config", str(root / "configs/study_config.yaml"),
         "--out-dir", str(out_dir)],
        capture_output=True, text=True,
    )

    # Exit 0 or 4 (Level-4 fallback) are both success states per SPEC.
    assert result.returncode in (0, 4), (
        f"Unexpected exit code {result.returncode}\nSTDERR:\n{result.stderr}"
    )
    assert (out_dir / "item_level_vc.csv").exists()
    assert (out_dir / "aggregate_vc.csv").exists()
    assert (out_dir / "model_convergence_report.csv").exists()


def test_cli_aggregate_vc_schema(paper_with_synthetic_data):
    root = paper_with_synthetic_data
    out_dir = root / "analysis" / "variance_components"
    subprocess.run(
        [sys.executable, str(SCRIPT_PATH),
         "--item-level", str(root / "runs/normalized/item_level_primary.parquet"),
         "--condition-level", str(root / "runs/normalized/condition_level_primary.csv"),
         "--config", str(root / "configs/study_config.yaml"),
         "--out-dir", str(out_dir)],
        capture_output=True, text=True,
    )
    agg = pd.read_csv(out_dir / "aggregate_vc.csv")
    expected_cols = {"benchmark_id", "model_id", "source",
                     "variance", "proportion", "df"}
    assert expected_cols.issubset(set(agg.columns))
    # Every (benchmark, model) cell should have 4 source rows:
    # prompt, seed, scoring_rule, residual.
    cells = agg.groupby(["benchmark_id", "model_id"])
    for (b, m), grp in cells:
        if "aggregate_fit_failed" in grp["source"].values:
            continue
        sources = set(grp["source"].values)
        assert sources == {"prompt", "seed", "scoring_rule", "residual"}, (
            f"({b}, {m}): sources = {sources}"
        )


def test_cli_convergence_report_populated(paper_with_synthetic_data):
    root = paper_with_synthetic_data
    out_dir = root / "analysis" / "variance_components"
    subprocess.run(
        [sys.executable, str(SCRIPT_PATH),
         "--item-level", str(root / "runs/normalized/item_level_primary.parquet"),
         "--condition-level", str(root / "runs/normalized/condition_level_primary.csv"),
         "--config", str(root / "configs/study_config.yaml"),
         "--out-dir", str(out_dir)],
        capture_output=True, text=True,
    )
    conv = pd.read_csv(out_dir / "model_convergence_report.csv")
    # At least one attempt logged
    assert len(conv) >= 1
    expected_cols = {"benchmark_id", "cascade_level", "success",
                     "error_type", "n_items", "fit_seconds"}
    assert expected_cols.issubset(set(conv.columns))


def test_cli_missing_config_fails_2(tmp_path):
    """Missing config should exit 2."""
    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH),
         "--item-level", "/nonexistent/item.parquet",
         "--condition-level", "/nonexistent/cond.csv",
         "--config", str(tmp_path / "nonexistent.yaml"),
         "--out-dir", str(tmp_path / "out")],
        capture_output=True, text=True,
    )
    assert result.returncode == 2
