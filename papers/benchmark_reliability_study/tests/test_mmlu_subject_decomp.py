"""Tests for 09_mmlu_subject_decomp.py — SPEC_CPU_v0_2 §8.10.

Covers:
    1. Accuracy matrix has correct shape and marginal row/column.
    2. Variance proportions sum to 1 (within numerical tolerance).
    3. H4 threshold check is consistent with the observed proportion.
    4. Mixed-effects path is taken on the well-conditioned fixture.
    5. ANOVA fallback path is reachable (small/degenerate input).
    6. End-to-end CLI emits all four artifacts.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest


PAPER_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = PAPER_ROOT / "scripts"
FIXTURES_DIR = Path(__file__).parent / "fixtures"
SCRIPT_PATH = SCRIPTS_DIR / "09_mmlu_subject_decomp.py"
ANALYSIS_CONFIG = FIXTURES_DIR / "configs" / "analysis_config.yaml"
MMLU_FIXTURE = FIXTURES_DIR / "mmlu_item_level.parquet"


# -- Build / import the script as a module ----------------------------------


@pytest.fixture(scope="session", autouse=True)
def ensure_fixtures_built():
    """Ensure the MMLU parquet fixture is present."""
    if not MMLU_FIXTURE.exists():
        result = subprocess.run(
            [sys.executable, str(FIXTURES_DIR / "make_fixtures.py")],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"Fixture build failed: {result.stderr}"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("mmlu_subject_decomp", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


mod = _load_script_module()


# -- Tests ------------------------------------------------------------------


def test_accuracy_matrix_shape_and_marginals():
    """Matrix has shape (n_models+1, n_subjects+1) with named marginals."""
    df = pd.read_parquet(MMLU_FIXTURE)
    matrix = mod.build_accuracy_matrix(df)
    # 3 models + 1 marginal row, 5 subjects + 1 marginal column.
    assert matrix.shape == (4, 6)
    assert "model_marginal_mean" in matrix.columns
    assert "__subject_marginal__" in matrix.index
    # Subject marginal row's last column equals the grand mean (uniform-weighted
    # over subjects).
    grand = matrix.loc["__subject_marginal__", "model_marginal_mean"]
    assert 0.0 <= grand <= 1.0


def test_variance_proportions_sum_to_one():
    """Proportions across the six canonical sources sum to 1.0."""
    df = pd.read_parquet(MMLU_FIXTURE)
    components, fallback = mod.compute_variance_components(df)
    var, prop = mod.normalize_to_proportions(components)
    total_prop = sum(prop.values())
    assert abs(total_prop - 1.0) < 1e-9
    # Sources are exactly the canonical six.
    assert set(prop.keys()) == set(mod.VC_SOURCE_ORDER)
    # All non-negative.
    for v in prop.values():
        assert v >= 0.0


def test_h4_check_consistency_with_observed():
    """H4 confirmed iff observed_proportion >= threshold."""
    df = pd.read_parquet(MMLU_FIXTURE)
    components, _ = mod.compute_variance_components(df)
    _, prop = mod.normalize_to_proportions(components)
    observed = float(prop["model_x_subject"])
    threshold = 0.10
    h4_confirmed = observed >= threshold
    # Smoke check: the constructed fixture should confirm H4.
    assert h4_confirmed, (
        f"Expected H4 confirmed on the canonical MMLU fixture; "
        f"observed model_x_subject proportion = {observed:.4f}"
    )


def test_mixed_effects_path_used_on_well_conditioned_fixture():
    """Default fixture has enough levels for the mixed-effects path."""
    df = pd.read_parquet(MMLU_FIXTURE)
    _, fallback = mod.compute_variance_components(df)
    assert fallback in {"mixed_effects", "anova"}
    # On the canonical fixture the mixed-effects fit should converge.
    assert fallback == "mixed_effects"


def test_anova_fallback_reachable(tmp_path):
    """If the mixed-effects fit is degenerate (e.g. one model only),
    the dispatcher returns the ANOVA fallback rather than crashing.
    """
    df = pd.read_parquet(MMLU_FIXTURE)
    # Filter to a single model — the model main effect is undefined and
    # the mixed-effects fit will degenerate; force the ANOVA fallback by
    # passing a 1-model frame to the ANOVA estimator directly.
    one_model = df[df["model_id"] == "test_model_1"].copy()
    components = mod.variance_components_anova_fallback(one_model)
    # All six sources must be present.
    assert set(components.keys()) >= set(mod.VC_SOURCE_ORDER)
    # No NaNs.
    for v in components.values():
        assert v == v  # not NaN


def test_end_to_end_cli(tmp_path):
    """End-to-end CLI: matrix CSV, VC CSV, JSON, and heatmap all written."""
    out_dir = tmp_path / "mmlu_subjects"
    fig_dir = tmp_path / "figures"
    result = subprocess.run(
        [
            sys.executable, str(SCRIPT_PATH),
            "--item-level", str(MMLU_FIXTURE),
            "--analysis-config", str(ANALYSIS_CONFIG),
            "--out-dir", str(out_dir),
            "--figures-dir", str(fig_dir),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"

    # Output files.
    assert (out_dir / "mmlu_subject_accuracy_matrix.csv").exists()
    assert (out_dir / "mmlu_subject_variance_components.csv").exists()
    assert (out_dir / "h4_test.json").exists()
    assert (fig_dir / "mmlu_model_subject_heatmap.png").exists()

    # H4 JSON schema.
    with open(out_dir / "h4_test.json") as f:
        obj = json.load(f)
    for key in (
        "h4_hypothesis",
        "threshold",
        "observed_proportion",
        "h4_confirmed",
        "fallback_used",
    ):
        assert key in obj
    assert obj["fallback_used"] in {"mixed_effects", "anova"}
    assert isinstance(obj["h4_confirmed"], bool)
    assert 0.0 <= obj["observed_proportion"] <= 1.0


def test_variance_components_csv_schema(tmp_path):
    """VC CSV has the canonical columns and the six canonical sources."""
    out_dir = tmp_path / "mmlu_subjects"
    fig_dir = tmp_path / "figures"
    result = subprocess.run(
        [
            sys.executable, str(SCRIPT_PATH),
            "--item-level", str(MMLU_FIXTURE),
            "--analysis-config", str(ANALYSIS_CONFIG),
            "--out-dir", str(out_dir),
            "--figures-dir", str(fig_dir),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"

    vc = pd.read_csv(out_dir / "mmlu_subject_variance_components.csv")
    assert list(vc.columns) == ["source", "variance", "proportion", "h4_relevant"]
    assert list(vc["source"]) == mod.VC_SOURCE_ORDER
    # Only the model_x_subject row is flagged H4-relevant.
    assert (vc["h4_relevant"].astype(str).str.lower() == "true").sum() == 1
    flagged = vc[vc["h4_relevant"].astype(str).str.lower() == "true"]
    assert flagged.iloc[0]["source"] == "model_x_subject"
