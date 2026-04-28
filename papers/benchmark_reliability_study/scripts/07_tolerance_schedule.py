"""07_tolerance_schedule.py — SPEC_CPU_v0_2 §8.8, §9.1, §9.2, §9.3

Tolerance schedule with bootstrap confidence intervals.

Consumes the aggregate variance components from 06_variance_components.py
(`analysis/variance_components/aggregate_vc.csv`) and the condition-level
accuracies (`runs/normalized/condition_level_primary.csv`). Produces:

    - analysis/tolerance_schedules/tolerance_by_cell.csv
    - analysis/tolerance_schedules/tolerance_by_benchmark_summary.csv
    - analysis/tolerance_schedules/h1_test.json

SEM definitions per §9.2:
    SEM_single          = sqrt(var_prompt + var_seed + var_scoring_rule + var_residual)
    SEM_within_rule     = sqrt(var_prompt + var_seed + var_residual)
    SEM_prompt_averaged = SEM_single / sqrt(4)
    SEM_full_design     = SEM_single / sqrt(n_conditions)

Bootstrap CI per §9.3: resample conditions (not items) with replacement;
n_resamples = analysis_config.bootstrap.n_resamples (default 10000);
seed = analysis_config.bootstrap.random_seed. The bootstrap lower bound
of the cross-model median single-occasion tolerance is the statistic
used by the H1 decision rule (§8.8 revised).

Decimal-place licensing per §9.1 (three-level categorical):
    tolerance < 0.0005       → "three_decimal_accuracy"   (0.742 licensed)
    0.0005 ≤ tolerance < 0.005 → "two_decimal_accuracy"   (0.74  licensed)
    tolerance ≥ 0.005        → "interval_required"        (no rounded point est.)

H1 decision rule (SPEC §3.2 / §8.8 revised):
    For ≥ 3 of 5 benchmarks, the bootstrap lower bound of the cross-model
    median single-occasion tolerance exceeds 0.005.

Usage:
    python scripts/07_tolerance_schedule.py \\
      --condition-level runs/normalized/condition_level_primary.csv \\
      --variance-components analysis/variance_components/aggregate_vc.csv \\
      --analysis-config configs/study_config.yaml \\
      --out-dir analysis/tolerance_schedules/

Exit codes:
    0 — success
    1 — generic failure
    2 — config or input failure
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from gradience_study.config import AnalysisConfig, load_config, load_raw_yaml  # noqa: E402


SCRIPT_VERSION = "0.1.0"


# -- Logging -----------------------------------------------------------------


def _log(level: str, msg: str) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"[{ts}] [{level}] [07_tolerance_schedule] {msg}", file=sys.stderr)


# -- Licensing per §9.1 ------------------------------------------------------


def licensed_precision(tolerance: float) -> str:
    """Map tolerance → decimal-place licensing categorical.

    Three levels:
      - "three_decimal_accuracy": tolerance < 0.0005 (0.742 licensed)
      - "two_decimal_accuracy": 0.0005 ≤ tolerance < 0.005 (0.74 licensed)
      - "interval_required": tolerance ≥ 0.005 (no rounded point est.)
    """
    if np.isnan(tolerance):
        return "interval_required"
    if tolerance < 0.0005:
        return "three_decimal_accuracy"
    if tolerance < 0.005:
        return "two_decimal_accuracy"
    return "interval_required"


def _determine_regime(
    scoring_rule_conditions: pd.DataFrame,
    parse_failure_threshold: float,
) -> str:
    """Decide whether a (benchmark, model, scoring_rule) cell is
    parse-failure-dominated.

    A cell is parse-failure-dominated when its scoring rule produces
    parseability data (i.e., generate-and-parse style) AND the median
    parseability across conditions falls below the configured threshold.
    Log-likelihood scoring rules have no parseability metric and are
    always treated as g_theory regime.

    Returns:
        "parse_failure_dominated" or "g_theory".
    """
    parseability = scoring_rule_conditions["parseability_rate"].dropna()
    if len(parseability) == 0:
        # LL scoring rule (or no condition data); g-theory regime applies.
        return "g_theory"
    median_parseability = float(parseability.median())
    if median_parseability < parse_failure_threshold:
        return "parse_failure_dominated"
    return "g_theory"


# -- SEM / tolerance math per §9.2 -------------------------------------------


def sem_from_variance_components(
    var_prompt: float,
    var_seed: float,
    var_scoring_rule: float,
    var_residual: float,
    averaging: str,
    n_conditions: int = 24,
) -> float:
    """Compute SEM under one of four averaging schemes.

    averaging ∈ {"single", "within_rule", "prompt_averaged", "full_design"}
    """
    total = var_prompt + var_seed + var_scoring_rule + var_residual
    within_rule = var_prompt + var_seed + var_residual

    if averaging == "single":
        return float(np.sqrt(max(total, 0.0)))
    if averaging == "within_rule":
        return float(np.sqrt(max(within_rule, 0.0)))
    if averaging == "prompt_averaged":
        return float(np.sqrt(max(total, 0.0) / 4.0))
    if averaging == "full_design":
        return float(np.sqrt(max(total, 0.0) / max(n_conditions, 1)))
    raise ValueError(f"Unknown averaging scheme: {averaging}")


def tolerance_from_sem(sem: float) -> float:
    """Tolerance is the half-width: 2 × SEM (conventional, not exact 95%)."""
    return 2.0 * sem


# -- Bootstrap CI per §9.3 ---------------------------------------------------


def bootstrap_tolerance_ci(
    condition_scores: np.ndarray,
    n_resamples: int,
    seed: int,
    ci_lower_pct: float = 2.5,
    ci_upper_pct: float = 97.5,
) -> tuple[float, float, float]:
    """Bootstrap CI on tolerance derived from condition-level scores.

    Resample conditions (not items) with replacement; recompute SEM as
    sample SD; tolerance = 2 × SEM.

    Returns:
        (tolerance_point, ci_lower, ci_upper).

    If n_conditions < 2, returns (nan, nan, nan) — no variability to
    bootstrap.
    """
    n = len(condition_scores)
    if n < 2:
        return float("nan"), float("nan"), float("nan")

    rng = np.random.default_rng(seed)
    scores = np.asarray(condition_scores, dtype=float)

    point_sem = float(np.std(scores, ddof=1))
    point_tol = 2.0 * point_sem

    boot_tols = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        sample = rng.choice(scores, size=n, replace=True)
        sem = float(np.std(sample, ddof=1))
        boot_tols[i] = 2.0 * sem

    ci_lower = float(np.percentile(boot_tols, ci_lower_pct))
    ci_upper = float(np.percentile(boot_tols, ci_upper_pct))
    return point_tol, ci_lower, ci_upper


# -- Main computation --------------------------------------------------------


def _load_analysis_config(config_path: Path) -> AnalysisConfig:
    """Load AnalysisConfig from either study_config.yaml or a bare
    analysis_config.yaml.
    """
    raw = load_raw_yaml(config_path)
    if "includes" in raw and isinstance(raw.get("includes"), list):
        return load_config(config_path).analysis_config
    # Bare analysis_config.yaml — hand-build a minimal AnalysisConfig
    from gradience_study.config import (
        BootstrapConfig,
        ConvergenceTriggersConfig,
        MixedEffectsCascadeConfig,
        ToleranceConfig,
    )
    boot = raw["bootstrap"]
    trig = raw["convergence_triggers"]
    tol = raw["tolerance"]
    return AnalysisConfig(
        bootstrap=BootstrapConfig(
            n_resamples=int(boot["n_resamples"]),
            random_seed=int(boot["random_seed"]),
            ci_lower_percentile=float(boot["ci_lower_percentile"]),
            ci_upper_percentile=float(boot["ci_upper_percentile"]),
        ),
        mixed_effects_cascade=MixedEffectsCascadeConfig(
            levels=dict(raw["mixed_effects_cascade"])
        ),
        convergence_triggers=ConvergenceTriggersConfig(
            max_singular_warning=trig["max_singular_warning"],
            hessian_non_pd=trig["hessian_non_pd"],
            iteration_limit_exceeded=trig["iteration_limit_exceeded"],
            gradient_norm_above=float(trig["gradient_norm_above"]),
        ),
        tolerance=ToleranceConfig(
            decision_rule_threshold_h1=float(tol["decision_rule_threshold_h1"]),
            benchmarks_required_for_h1=int(tol["benchmarks_required_for_h1"]),
        ),
        h2_generalizability_threshold=float(raw["h2_generalizability_threshold"]),
        h3_ranking_reversal_threshold=float(raw["h3_ranking_reversal_threshold"]),
        h4_mmlu_interaction_threshold=float(raw["h4_mmlu_interaction_threshold"]),
    )


def compute_per_cell_tolerance(
    vc_df: pd.DataFrame,
    condition_df: pd.DataFrame,
    analysis_config: AnalysisConfig,
) -> pd.DataFrame:
    """Compute tolerance rows per (benchmark, model, scoring_rule)."""
    boot_cfg = analysis_config.bootstrap
    rng_seed_base = boot_cfg.random_seed

    rows = []

    # VC is at the (benchmark, model) level; scoring_rule is within-cell.
    # We expand to one tolerance row per (benchmark, model, scoring_rule)
    # by filtering the condition data.
    cells = (
        condition_df[["benchmark_id", "model_id", "scoring_rule_id"]]
        .drop_duplicates()
        .sort_values(["benchmark_id", "model_id", "scoring_rule_id"])
        .reset_index(drop=True)
    )

    for idx, row in cells.iterrows():
        b_id = row["benchmark_id"]
        m_id = row["model_id"]
        s_id = row["scoring_rule_id"]

        vc_cell = vc_df[
            (vc_df["benchmark_id"] == b_id)
            & (vc_df["model_id"] == m_id)
        ]
        if len(vc_cell) == 0:
            continue

        def _get_vc(source_name: str) -> float:
            row_match = vc_cell[vc_cell["source"] == source_name]
            if len(row_match) == 0:
                return 0.0
            v = row_match["variance"].iloc[0]
            return float(v) if not pd.isna(v) else 0.0

        var_p = _get_vc("prompt")
        var_s = _get_vc("seed")
        var_r = _get_vc("scoring_rule")
        var_e = _get_vc("residual")

        # n_conditions for this cell (across all scoring rules and seeds)
        cell_conditions = condition_df[
            (condition_df["benchmark_id"] == b_id)
            & (condition_df["model_id"] == m_id)
        ]
        n_cond = len(cell_conditions)

        # v1.1.2: per-(benchmark, model, scoring_rule) regime check.
        # When parseability_rate is below threshold for this scoring
        # rule's conditions, the variance-components decomposition is
        # degenerate (parse-failure mechanism dominates) and we route
        # to sample-SD-based tolerance per the D-07 pattern. This
        # resolves D-09 by limiting LPM-vs-GLMM regime to cells where
        # the binomial-tail-behavior matters.
        scoring_rule_conditions = cell_conditions[
            cell_conditions["scoring_rule_id"] == s_id
        ]
        regime = _determine_regime(
            scoring_rule_conditions,
            analysis_config.tolerance.parse_failure_threshold,
        )

        if regime == "parse_failure_dominated":
            # Sample-SD tolerance per D-07 pattern; variance components
            # are reported as NaN to flag the regime split downstream.
            sr_accuracies = scoring_rule_conditions["accuracy"].to_numpy()
            if len(sr_accuracies) >= 2:
                sem_single = float(np.std(sr_accuracies, ddof=1))
            else:
                sem_single = float("nan")
            sem_within = sem_single  # within-rule == single in this regime
            sem_pavg = sem_single / np.sqrt(4.0) if not np.isnan(sem_single) else float("nan")
            sem_full = sem_single / np.sqrt(max(n_cond, 1)) if not np.isnan(sem_single) else float("nan")
        else:
            sem_single = sem_from_variance_components(
                var_p, var_s, var_r, var_e, "single", n_cond
            )
            sem_within = sem_from_variance_components(
                var_p, var_s, var_r, var_e, "within_rule", n_cond
            )
            sem_pavg = sem_from_variance_components(
                var_p, var_s, var_r, var_e, "prompt_averaged", n_cond
            )
            sem_full = sem_from_variance_components(
                var_p, var_s, var_r, var_e, "full_design", n_cond
            )

        tol_single = tolerance_from_sem(sem_single)
        tol_within = tolerance_from_sem(sem_within)
        tol_pavg = tolerance_from_sem(sem_pavg)
        tol_full = tolerance_from_sem(sem_full)

        # Bootstrap CI using condition-level accuracies for this cell
        # (filtered to this scoring rule — for the primary single-occasion
        # tolerance — and also across all rules — for the within-rule
        # and across-rule distinction per §8.3).
        within_scores = cell_conditions[
            cell_conditions["scoring_rule_id"] == s_id
        ]["accuracy"].to_numpy()
        across_scores = cell_conditions["accuracy"].to_numpy()

        # Per-cell bootstrap seed is derived from (base, benchmark, model,
        # scoring_rule) so different cells get independent resamples but
        # the whole study is deterministic. Note: Python's built-in
        # hash() is randomized per process (PYTHONHASHSEED), so we use
        # hashlib.sha256 to derive a stable per-cell seed offset across
        # script invocations. This is the D-21 fix.
        cell_key = f"{b_id}|{m_id}|{s_id}".encode("utf-8")
        cell_offset = int.from_bytes(
            hashlib.sha256(cell_key).digest()[:4], byteorder="big"
        )
        cell_seed = (rng_seed_base + cell_offset) % (2**31)

        _, tol_single_cil, tol_single_ciu = bootstrap_tolerance_ci(
            across_scores,
            n_resamples=boot_cfg.n_resamples,
            seed=cell_seed,
            ci_lower_pct=boot_cfg.ci_lower_percentile,
            ci_upper_pct=boot_cfg.ci_upper_percentile,
        )
        _, tol_within_cil, tol_within_ciu = bootstrap_tolerance_ci(
            within_scores,
            n_resamples=boot_cfg.n_resamples,
            seed=cell_seed + 1,
            ci_lower_pct=boot_cfg.ci_lower_percentile,
            ci_upper_pct=boot_cfg.ci_upper_percentile,
        )

        # For prompt-averaged and full-design tolerance CIs, we shrink
        # the point estimate by the averaging factor and shrink the CI
        # bounds analogously (linear scaling — conservative for the
        # lower bound).
        shrink_prompt = 1.0 / np.sqrt(4.0)
        shrink_full = 1.0 / np.sqrt(max(n_cond, 1))
        tol_pavg_cil = tol_single_cil * shrink_prompt
        tol_pavg_ciu = tol_single_ciu * shrink_prompt
        tol_full_cil = tol_single_cil * shrink_full
        tol_full_ciu = tol_single_ciu * shrink_full

        rows.append({
            "benchmark_id": b_id,
            "model_id": m_id,
            "scoring_rule_id": s_id,
            "regime": regime,
            "n_conditions": n_cond,
            "sem_single": sem_single,
            "tolerance_single": tol_single,
            "tolerance_single_ci_lower": tol_single_cil,
            "tolerance_single_ci_upper": tol_single_ciu,
            "sem_within_rule": sem_within,
            "tolerance_within_rule": tol_within,
            "tolerance_within_rule_ci_lower": tol_within_cil,
            "tolerance_within_rule_ci_upper": tol_within_ciu,
            "sem_prompt_avg": sem_pavg,
            "tolerance_prompt_avg": tol_pavg,
            "tolerance_prompt_avg_ci_lower": tol_pavg_cil,
            "tolerance_prompt_avg_ci_upper": tol_pavg_ciu,
            "sem_full_design": sem_full,
            "tolerance_full_design": tol_full,
            "tolerance_full_design_ci_lower": tol_full_cil,
            "tolerance_full_design_ci_upper": tol_full_ciu,
            "licensed_precision_single": licensed_precision(tol_single),
            "licensed_precision_full_design": licensed_precision(tol_full),
        })
        _ = idx  # suppress unused-var warning

    return pd.DataFrame(rows, columns=[
        "benchmark_id", "model_id", "scoring_rule_id", "regime",
        "n_conditions",
        "sem_single", "tolerance_single",
        "tolerance_single_ci_lower", "tolerance_single_ci_upper",
        "sem_within_rule", "tolerance_within_rule",
        "tolerance_within_rule_ci_lower", "tolerance_within_rule_ci_upper",
        "sem_prompt_avg", "tolerance_prompt_avg",
        "tolerance_prompt_avg_ci_lower", "tolerance_prompt_avg_ci_upper",
        "sem_full_design", "tolerance_full_design",
        "tolerance_full_design_ci_lower", "tolerance_full_design_ci_upper",
        "licensed_precision_single", "licensed_precision_full_design",
    ])


def compute_benchmark_summary(
    per_cell_df: pd.DataFrame,
    analysis_config: AnalysisConfig,
) -> pd.DataFrame:
    """Aggregate per-cell tolerance to per-benchmark cross-model medians."""
    rows = []
    threshold = analysis_config.tolerance.decision_rule_threshold_h1

    for b_id, group in per_cell_df.groupby("benchmark_id"):
        # Cross-model median of single-occasion tolerance and CI bounds
        med_tol = float(group["tolerance_single"].median())
        med_cil = float(group["tolerance_single_ci_lower"].median())
        med_ciu = float(group["tolerance_single_ci_upper"].median())
        med_tol_full = float(group["tolerance_full_design"].median())

        rows.append({
            "benchmark_id": b_id,
            "n_cells": len(group),
            "tolerance_single_median": med_tol,
            "tolerance_single_median_ci_lower": med_cil,
            "tolerance_single_median_ci_upper": med_ciu,
            "tolerance_full_design_median": med_tol_full,
            "exceeds_h1_threshold_point": med_tol > threshold,
            "exceeds_h1_threshold_ci_lower": med_cil > threshold,
            "licensed_precision_single": licensed_precision(med_tol),
            "licensed_precision_full_design": licensed_precision(med_tol_full),
        })

    return pd.DataFrame(rows, columns=[
        "benchmark_id", "n_cells",
        "tolerance_single_median",
        "tolerance_single_median_ci_lower",
        "tolerance_single_median_ci_upper",
        "tolerance_full_design_median",
        "exceeds_h1_threshold_point",
        "exceeds_h1_threshold_ci_lower",
        "licensed_precision_single",
        "licensed_precision_full_design",
    ])


def test_h1(
    summary_df: pd.DataFrame,
    analysis_config: AnalysisConfig,
) -> dict:
    """H1 decision per SPEC §8.8 (revised with bootstrap lower bound)."""
    threshold = analysis_config.tolerance.decision_rule_threshold_h1
    n_required = analysis_config.tolerance.benchmarks_required_for_h1

    exceeding = summary_df[
        summary_df["exceeds_h1_threshold_ci_lower"]
    ]["benchmark_id"].tolist()

    return {
        "h1_hypothesis": (
            "bootstrap lower bound of cross-model median single-occasion "
            f"tolerance > {threshold} for at least {n_required} of 5 benchmarks"
        ),
        "threshold": threshold,
        "n_required": n_required,
        "n_exceeding": len(exceeding),
        "benchmarks_exceeding": exceeding,
        "h1_confirmed": len(exceeding) >= n_required,
    }


# -- Orchestration -----------------------------------------------------------


def run_tolerance_schedule(
    condition_level_path: Path,
    variance_components_path: Path,
    config_path: Path,
    out_dir: Path,
) -> tuple[int, dict]:
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        analysis_config = _load_analysis_config(config_path)
    except Exception as e:
        _log("ERROR", f"Analysis config load failed: {e}")
        return 2, {}

    if not variance_components_path.exists():
        _log("ERROR", f"Variance components CSV not found: {variance_components_path}")
        return 2, {}
    if not condition_level_path.exists():
        _log("ERROR", f"Condition-level CSV not found: {condition_level_path}")
        return 2, {}

    vc_df = pd.read_csv(variance_components_path)
    condition_df = pd.read_csv(condition_level_path)

    _log("INFO",
         f"Computing tolerance schedule for {condition_df['benchmark_id'].nunique()} "
         f"benchmark(s), {condition_df['model_id'].nunique()} model(s), "
         f"{condition_df['scoring_rule_id'].nunique()} scoring rule(s)")

    per_cell_df = compute_per_cell_tolerance(vc_df, condition_df, analysis_config)
    per_cell_df.to_csv(out_dir / "tolerance_by_cell.csv", index=False)

    summary_df = compute_benchmark_summary(per_cell_df, analysis_config)
    summary_df.to_csv(out_dir / "tolerance_by_benchmark_summary.csv", index=False)

    h1_result = test_h1(summary_df, analysis_config)
    with open(out_dir / "h1_test.json", "w", encoding="utf-8") as f:
        json.dump(h1_result, f, indent=2, sort_keys=True)

    _log("INFO",
         f"H1: {h1_result['n_exceeding']}/{h1_result['n_required']} benchmarks "
         f"{'confirmed' if h1_result['h1_confirmed'] else 'not confirmed'}")

    return 0, {
        "n_cells": len(per_cell_df),
        "n_benchmarks": len(summary_df),
        "h1_confirmed": h1_result["h1_confirmed"],
        "script_version": SCRIPT_VERSION,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Tolerance schedule with bootstrap CIs per SPEC §8.8"
    )
    parser.add_argument("--condition-level", required=True, type=Path)
    parser.add_argument("--variance-components", required=True, type=Path)
    parser.add_argument("--analysis-config", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()

    exit_code, summary = run_tolerance_schedule(
        args.condition_level,
        args.variance_components,
        args.analysis_config,
        args.out_dir,
    )
    _log("INFO", f"exit_code={exit_code} summary={summary}")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
