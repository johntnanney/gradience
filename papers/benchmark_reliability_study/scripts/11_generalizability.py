"""Generalizability coefficients + H2 test (SPEC §7.2; pre-reg §3.3 H2).

H2 hypothesis (pre-reg §3.3, line 93):
    For at least three of the five primary benchmarks, the
    generalizability coefficient under single-occasion measurement
    (single prompt × single seed × single scoring rule) is less than
    0.80, indicating that single-occasion evaluation is an unreliable
    estimator of the generalized score.

Computation per pre-reg §7.2:
    For each benchmark, compute G under multiple averaging schemes:

        G(N) = var_systematic / (var_systematic + var_error / N)

    where:
        var_systematic = between-model variance (variance of the 3
            models' mean accuracies across conditions; the "true" score
            we're trying to estimate)
        var_error      = average within-model variance across the
            error sources (prompt, seed, scoring_rule, residual)
        N              = number of averaged conditions
            - N = 1: single-occasion (single prompt × seed × rule)
            - N = 4: prompt-averaged (1 seed × 1 rule)
            - N = 8: prompt × seed averaged (1 rule)
            - N = 24: full design (all 4 prompts × 3 seeds × 2 rules)

H2 decision rule:
    Count benchmarks where G_single < 0.80; H2 confirmed if count >= 3.

This is the third pre-registered hypothesis test alongside H1 (script 07)
and H4 (script 09). H3 ranking-stability is in script 08.

Usage (from papers/benchmark_reliability_study/):
    python scripts/11_generalizability.py \\
        --condition-level runs/normalized/condition_level_primary.csv \\
        --variance-components analysis/variance_components/aggregate_vc.csv \\
        --analysis-config configs/study_config.yaml \\
        --out-dir analysis/generalizability/
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PAPER_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PAPER_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from gradience_study.config import load_config  # noqa: E402


def _ts() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _log(level: str, msg: str) -> None:
    print(f"[{_ts()}] [{level}] [11_generalizability] {msg}", file=sys.stderr)


def compute_generalizability_per_benchmark(
    condition_level: pd.DataFrame,
    aggregate_vc: pd.DataFrame,
) -> pd.DataFrame:
    """For each benchmark, compute G under 4 averaging schemes.

    Returns a dataframe with columns:
        benchmark_id, var_between_models, var_within_model_mean,
        G_single, G_prompt_avg, G_within_rule, G_full_design,
        n_conditions_per_cell
    """
    out_rows = []

    error_sources = {"prompt", "seed", "scoring_rule", "residual"}

    for benchmark_id in sorted(condition_level["benchmark_id"].unique()):
        bench_cond = condition_level[
            condition_level["benchmark_id"] == benchmark_id
        ]
        # Mean accuracy per model
        per_model_means = bench_cond.groupby("model_id")["accuracy"].mean()
        n_models = len(per_model_means)
        if n_models < 2:
            _log("WARNING", f"{benchmark_id}: only {n_models} models; "
                 "skipping G computation")
            continue
        # Between-model variance (sample variance with ddof=1)
        var_between = float(per_model_means.var(ddof=1))

        # Within-model variance per cell:
        # average of (var_prompt + var_seed + var_scoring_rule + var_residual)
        # across the n_models models for this benchmark.
        bench_vc = aggregate_vc[
            (aggregate_vc["benchmark_id"] == benchmark_id) &
            (aggregate_vc["source"].isin(error_sources))
        ]
        within_per_model = (
            bench_vc.groupby("model_id")["variance"].sum()
        )
        var_within_mean = float(within_per_model.mean())

        # n_conditions per cell varies by benchmark
        n_per_cell = int(bench_cond.groupby("model_id").size().median())

        # G under different averaging schemes.
        # SPEC §9.2 SEM_full_design uses N = n_conditions = 24 (or 8 for
        # zero-shot). SEM_prompt_avg uses N = 4 (prompts averaged, single
        # seed × rule). SEM_within_rule uses var without scoring_rule
        # term (we keep one scoring rule fixed).
        def _g(error_var: float, n: int) -> float:
            if (var_between + error_var / n) <= 0:
                return float("nan")
            return var_between / (var_between + error_var / n)

        # var_within_rule: drop scoring_rule contribution
        bench_vc_within_rule = aggregate_vc[
            (aggregate_vc["benchmark_id"] == benchmark_id) &
            (aggregate_vc["source"].isin({"prompt", "seed", "residual"}))
        ]
        within_rule_per_model = (
            bench_vc_within_rule.groupby("model_id")["variance"].sum()
        )
        var_within_rule_mean = float(within_rule_per_model.mean())

        # Detect zero-shot benchmark (no seed facet → n_per_cell = 8 not 24)
        n_full = n_per_cell  # use observed
        n_within_rule = max(n_full // 2, 1)  # fixed one scoring rule
        n_prompt_avg = max(n_within_rule // 3, 1) if n_within_rule >= 6 else 4

        out_rows.append({
            "benchmark_id": benchmark_id,
            "n_models": n_models,
            "n_conditions_per_cell": n_per_cell,
            "var_between_models": var_between,
            "var_within_model_mean": var_within_mean,
            "var_within_rule_mean": var_within_rule_mean,
            "G_single": _g(var_within_mean, 1),
            "G_within_rule": _g(var_within_rule_mean, 1),
            "G_prompt_avg": _g(var_within_mean, n_prompt_avg),
            "G_full_design": _g(var_within_mean, n_full),
        })

    return pd.DataFrame(out_rows)


def test_h2(g_table: pd.DataFrame, threshold: float, n_required: int) -> dict:
    """Apply pre-reg §3.3 H2 decision rule.

    H2 confirmed if at least n_required benchmarks have G_single < threshold.
    """
    g_table = g_table.dropna(subset=["G_single"])
    benchmarks_below = g_table[g_table["G_single"] < threshold]
    n_below = int(len(benchmarks_below))
    return {
        "h2_hypothesis": (
            "for at least N benchmarks, generalizability coefficient under "
            "single-occasion measurement is less than the threshold "
            "(indicating single-occasion evaluation is unreliable)"
        ),
        "threshold": threshold,
        "n_required": n_required,
        "n_below": n_below,
        "benchmarks_below": sorted(benchmarks_below["benchmark_id"].tolist()),
        "benchmarks_above_or_equal": sorted(
            g_table[g_table["G_single"] >= threshold]["benchmark_id"].tolist()
        ),
        "h2_confirmed": n_below >= n_required,
        "g_single_per_benchmark": dict(
            zip(g_table["benchmark_id"], g_table["G_single"].round(4))
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generalizability coefficients + H2 test (SPEC §7.2)"
    )
    parser.add_argument("--condition-level", type=Path, required=True)
    parser.add_argument("--variance-components", type=Path, required=True)
    parser.add_argument("--analysis-config", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    cond = pd.read_csv(args.condition_level)
    vc = pd.read_csv(args.variance_components)
    cfg = load_config(args.analysis_config)

    threshold = cfg.analysis_config.h2_generalizability_threshold
    # Pre-reg §3.3 says "at least three of the five primary benchmarks"
    n_required = 3

    _log("INFO", f"Computing G coefficients for "
         f"{len(cond['benchmark_id'].unique())} benchmark(s); "
         f"H2 threshold = {threshold}")

    g_table = compute_generalizability_per_benchmark(cond, vc)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    g_table.to_csv(args.out_dir / "generalizability_coefficients.csv",
                   index=False)
    _log("INFO", f"Wrote generalizability_coefficients.csv "
         f"({len(g_table)} rows)")

    h2 = test_h2(g_table, threshold, n_required)
    (args.out_dir / "h2_test.json").write_text(
        json.dumps(h2, indent=2, sort_keys=True) + "\n"
    )
    _log(
        "INFO",
        f"H2: {h2['n_below']}/{h2['n_required']} benchmarks below "
        f"threshold; confirmed={h2['h2_confirmed']}",
    )

    summary = {
        "n_benchmarks_analyzed": int(len(g_table)),
        "h2_confirmed": bool(h2["h2_confirmed"]),
        "h2_n_below_threshold": int(h2["n_below"]),
        "script_version": "0.1.0",
    }
    _log("INFO", f"exit_code=0 summary={summary}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
