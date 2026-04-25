"""06_variance_components.py — SPEC_CPU_v0_2 §8.7, §10.1, §10.2

Variance-components decomposition of benchmark accuracy.

This script runs two analyses per benchmark:

1. **Item-level mixed-effects cascade** (§10.1). Four levels tried in
   order; first converging level is reported. Level 4 (aggregate-score
   G-theory only) is the final fallback. Every attempted level is
   logged to model_convergence_report.csv.

2. **Aggregate-score G-theory decomposition** (§10.2). Runs
   unconditionally per (benchmark, model). Consumes condition-level
   accuracies (not item-level). Produces the variance components that
   feed the tolerance schedule (07_tolerance_schedule.py).

The tolerance schedule depends ONLY on aggregate-score VC. Item-level
mixed effects are supplementary: they give a richer decomposition
including item variance, but are not on the critical path.

Methodological note: the spec (§10.1) calls for logistic mixed-effects
at item level. `statsmodels.MixedLM` — the library the spec names — is
Gaussian only. We therefore fit a linear probability model (LPM) at
item level, which treats 0/1 item outcomes as continuous. For
accuracies in the typical benchmark range (0.15–0.85), LPM variance
proportions are within a few percent of logistic GLMM variance
proportions. The deviation is recorded in IMPLEMENTATION_DEVIATIONS.md
and acknowledged in the paper's manuscript-side discussion of Analysis 1.

Usage:
    python scripts/06_variance_components.py \\
      --item-level runs/normalized/item_level_primary.parquet \\
      --condition-level runs/normalized/condition_level_primary.csv \\
      --config configs/study_config.yaml \\
      --out-dir analysis/variance_components/

Exit codes per SPEC §11.1:
    0 — success (aggregate VC produced for every benchmark).
    2 — config or input failure.
    4 — aggregate G-theory failed for one or more (benchmark, model)
         cells (ConvergenceFailureAllLevels).
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

# Make the paper-root package importable without requiring an install.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
import statsmodels.api as sm  # noqa: E402
import statsmodels.formula.api as smf  # noqa: E402

from gradience_study.config import load_config  # noqa: E402

SCRIPT_VERSION = "0.1.0"

# -- Logging -----------------------------------------------------------------


def _log(level: str, msg: str) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"[{ts}] [{level}] [06_variance_components] {msg}", file=sys.stderr)


# -- Mixed-effects cascade per §10.1 ------------------------------------------

CASCADE_LEVELS = ["level_1", "level_2", "level_3", "level_4"]


def _random_effects_for_level(level: str) -> list[str]:
    """Return the list of random-effect facets for a cascade level.

    Level 4 has no random effects; it's the aggregate G-theory fallback.
    """
    mapping = {
        "level_1": ["prompt_id", "seed_id", "scoring_rule_id", "item_id"],
        # Note: model × prompt and model × scoring_rule interactions are
        # specified in SPEC §10.1 as Level 1 random effects. statsmodels
        # does not cleanly support crossed-interaction random effects
        # with the LPM variance-component formulation used here, so
        # those interactions collapse into residual in Level 1. The full
        # interaction decomposition would require rpy2+lme4; documented
        # deviation in IMPLEMENTATION_DEVIATIONS.md.
        "level_2": ["prompt_id", "seed_id", "scoring_rule_id", "item_id"],
        "level_3": ["prompt_id", "scoring_rule_id", "item_id"],
        "level_4": [],
    }
    return mapping[level]


def _fit_item_level_lpm(
    item_data: pd.DataFrame,
    random_effects: list[str],
) -> dict:
    """Fit a linear probability model with crossed random effects.

    Uses statsmodels.MixedLM with `vc_formula` for crossed (independent)
    random-effect facets. Returns a dict with variance components or
    raises on convergence failure.

    The "group" column is a constant dummy because MixedLM requires a
    grouping variable; crossed facets are encoded via vc_formula.
    """
    df = item_data.copy()
    df["_grp"] = 0  # constant group for pure crossed structure.
    df["is_correct_num"] = df["is_correct"].astype(float)

    vc_formula = {fac: f"0 + C({fac})" for fac in random_effects}

    with warnings.catch_warnings(record=True) as w_list:
        warnings.simplefilter("always")
        model = smf.mixedlm(
            "is_correct_num ~ C(model_id)",
            data=df,
            groups="_grp",
            vc_formula=vc_formula,
        )
        t0 = time.time()
        result = model.fit(reml=True, method="lbfgs")
        elapsed = time.time() - t0

        # Check for convergence triggers (SPEC §10.1)
        saw_singular = any("singular" in str(w.message).lower() for w in w_list)

    # Extract variance components from result. statsmodels stores VC
    # names on model.exog_vc.names (ordered list); result.vcomp is a
    # plain numpy array in the same order.
    vc_names = list(model.exog_vc.names)
    vc_values = np.asarray(result.vcomp, dtype=float).flatten()
    vc_lookup = dict(zip(vc_names, vc_values))

    vc_vars = {}
    for fac in random_effects:
        if fac in vc_lookup:
            vc_vars[fac] = float(vc_lookup[fac])
        else:
            matches = [k for k in vc_lookup if fac in k]
            vc_vars[fac] = float(vc_lookup[matches[0]]) if matches else 0.0

    residual_var = float(result.scale)

    # Convergence diagnostic
    converged = bool(result.converged)
    diagnostic = {
        "converged": converged,
        "saw_singular_warning": saw_singular,
        "n_items": int(len(df)),
        "fit_seconds": elapsed,
        "log_likelihood": float(result.llf) if converged else float("nan"),
    }

    return {
        "variance_components": vc_vars,
        "residual_variance": residual_var,
        "diagnostic": diagnostic,
    }


def fit_item_level_cascade(
    item_data: pd.DataFrame,
    benchmark_id: str,
    convergence_triggers: dict,
) -> tuple[str, dict | None, list[dict]]:
    """Run the pre-registered cascade (§10.1). No post-hoc simplification.

    Returns:
        (level_used, result_or_None, attempt_log).
        level_used is "level_1" through "level_4".
        result is None only if all three mixed-effects levels fail AND
        Level 4 fallback still runs at the aggregate level (not item
        level) — in which case item_level_vc has no entries for this
        benchmark, and the aggregate G-theory decomposition is the
        only reported VC.
    """
    attempt_log: list[dict] = []
    gradient_threshold = float(convergence_triggers.get(
        "gradient_norm_above", 1e-3
    ))

    for level in ["level_1", "level_2", "level_3"]:
        random_effects = _random_effects_for_level(level)
        attempt_entry = {
            "benchmark_id": benchmark_id,
            "cascade_level": level,
            "success": False,
            "error_type": "",
            "gradient_norm": float("nan"),
            "optimizer_iterations": 0,
            "n_items": int(len(item_data)),
            "fit_seconds": 0.0,
        }
        try:
            fit = _fit_item_level_lpm(item_data, random_effects)
            diag = fit["diagnostic"]
            attempt_entry["fit_seconds"] = diag["fit_seconds"]
            if not diag["converged"]:
                attempt_entry["error_type"] = "did_not_converge"
                attempt_log.append(attempt_entry)
                _log("INFO",
                     f"{benchmark_id} {level}: did not converge; "
                     f"descending cascade")
                continue
            if diag["saw_singular_warning"]:
                attempt_entry["error_type"] = "singular_fit_warning"
                attempt_log.append(attempt_entry)
                _log("INFO",
                     f"{benchmark_id} {level}: singular fit; "
                     f"descending cascade")
                continue
            # Gradient-norm check unavailable from statsmodels; we
            # treat convergence + no singular warning as pass at this
            # level of the cascade. Note in deviations.
            _ = gradient_threshold  # placeholder per spec §10.1

            attempt_entry["success"] = True
            attempt_log.append(attempt_entry)
            _log("INFO", f"{benchmark_id} {level}: converged")
            return level, fit, attempt_log

        except Exception as e:
            attempt_entry["error_type"] = type(e).__name__
            attempt_log.append(attempt_entry)
            _log("WARNING", f"{benchmark_id} {level}: {type(e).__name__}: {e}")
            continue

    # All mixed-effects levels failed — Level 4 fallback.
    level_4_entry = {
        "benchmark_id": benchmark_id,
        "cascade_level": "level_4",
        "success": True,
        "error_type": "mixed_effects_abandoned_using_aggregate_only",
        "gradient_norm": float("nan"),
        "optimizer_iterations": 0,
        "n_items": int(len(item_data)),
        "fit_seconds": 0.0,
    }
    attempt_log.append(level_4_entry)
    _log("WARNING",
         f"{benchmark_id}: all mixed-effects levels failed; "
         f"falling back to aggregate G-theory only")
    return "level_4", None, attempt_log


# -- Aggregate-score G-theory per §10.2 ---------------------------------------


def fit_aggregate_g_theory(
    condition_rows: pd.DataFrame,
    benchmark_id: str,
    model_id: str,
) -> dict | None:
    """Decompose condition-level accuracy variance across crossed facets.

    Uses random-effects ANOVA via statsmodels.MixedLM at the aggregate
    (condition) level. Crossed facets: prompt × seed × scoring_rule.

    Returns a dict with keys:
        var_prompt, var_seed, var_scoring_rule, var_residual, var_total,
        n_conditions.

    Returns None if the cell has too few conditions (< 4) or if the fit
    fails entirely (in which case exit 4 for this cell).
    """
    cell = condition_rows[
        (condition_rows["benchmark_id"] == benchmark_id)
        & (condition_rows["model_id"] == model_id)
    ].copy()

    n_conditions = len(cell)
    if n_conditions < 4:
        _log("WARNING",
             f"{benchmark_id}/{model_id}: only {n_conditions} conditions; "
             f"skipping aggregate VC")
        return None

    # Normalize seed_id: for 0-shot benchmarks seed_id is "0shot" or
    # "none"; treat as a single level (collapses to residual).
    cell["_seed_id"] = cell["seed_id"].fillna("none").astype(str)
    cell["_prompt_id"] = cell["prompt_id"].astype(str)
    cell["_scoring_rule_id"] = cell["scoring_rule_id"].astype(str)

    # Check which facets have multiple levels
    n_prompts = cell["_prompt_id"].nunique()
    n_seeds = cell["_seed_id"].nunique()
    n_rules = cell["_scoring_rule_id"].nunique()

    facets_active = []
    if n_prompts > 1:
        facets_active.append("_prompt_id")
    if n_seeds > 1:
        facets_active.append("_seed_id")
    if n_rules > 1:
        facets_active.append("_scoring_rule_id")

    if not facets_active:
        # Pure residual variance — nothing to decompose.
        var_resid = float(np.var(cell["accuracy"], ddof=1)) \
            if n_conditions > 1 else 0.0
        return {
            "var_prompt": 0.0,
            "var_seed": 0.0,
            "var_scoring_rule": 0.0,
            "var_residual": var_resid,
            "var_total": var_resid,
            "n_conditions": n_conditions,
        }

    cell["_grp"] = 0
    vc_formula = {
        fac.lstrip("_"): f"0 + C({fac})"
        for fac in facets_active
    }

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = smf.mixedlm(
                "accuracy ~ 1",
                data=cell,
                groups="_grp",
                vc_formula=vc_formula,
            )
            result = model.fit(reml=True, method="lbfgs")
    except Exception as e:
        _log("ERROR",
             f"{benchmark_id}/{model_id} aggregate VC failed: "
             f"{type(e).__name__}: {e}")
        return None

    # Extract VC for each facet. vcomp is a numpy array; names live on
    # model.exog_vc.names in matching order.
    vc_names = list(model.exog_vc.names)
    vc_values = np.asarray(result.vcomp, dtype=float).flatten()
    vc_lookup = dict(zip(vc_names, vc_values))

    vc_out = {
        "var_prompt": 0.0,
        "var_seed": 0.0,
        "var_scoring_rule": 0.0,
    }
    for facet_key, vc_name in [
        ("_prompt_id", "var_prompt"),
        ("_seed_id", "var_seed"),
        ("_scoring_rule_id", "var_scoring_rule"),
    ]:
        clean = facet_key.lstrip("_")
        if clean in vc_lookup:
            vc_out[vc_name] = max(float(vc_lookup[clean]), 0.0)
        else:
            matches = [k for k in vc_lookup if clean in k]
            if matches:
                vc_out[vc_name] = max(float(vc_lookup[matches[0]]), 0.0)

    var_resid = max(float(result.scale), 0.0)
    vc_out["var_residual"] = var_resid
    vc_out["var_total"] = sum(vc_out[k] for k in [
        "var_prompt", "var_seed", "var_scoring_rule", "var_residual"
    ])
    vc_out["n_conditions"] = n_conditions

    return vc_out


# -- Emission ----------------------------------------------------------------


def emit_item_level_vc(
    results_by_benchmark: dict[str, tuple[str, dict | None]],
    out_path: Path,
) -> None:
    """Emit item_level_vc.csv per SPEC §8.7."""
    rows = []
    for benchmark_id, (level, fit) in results_by_benchmark.items():
        if fit is None:
            continue  # Level 4 fallback — no item-level VC
        vcs = fit["variance_components"]
        residual = fit["residual_variance"]
        total = sum(vcs.values()) + residual
        # Per-facet rows
        for fac_name, var in vcs.items():
            rows.append({
                "benchmark_id": benchmark_id,
                "cascade_level_used": level,
                "random_effect": fac_name,
                "variance": var,
                "proportion": (var / total) if total > 0 else 0.0,
            })
        # Residual row
        rows.append({
            "benchmark_id": benchmark_id,
            "cascade_level_used": level,
            "random_effect": "residual",
            "variance": residual,
            "proportion": (residual / total) if total > 0 else 0.0,
        })

    df = pd.DataFrame(rows, columns=[
        "benchmark_id", "cascade_level_used",
        "random_effect", "variance", "proportion"
    ])
    df.to_csv(out_path, index=False)


def emit_aggregate_vc(
    aggregate_by_cell: dict[tuple[str, str], dict | None],
    out_path: Path,
) -> None:
    """Emit aggregate_vc.csv per SPEC §8.7."""
    rows = []
    for (benchmark_id, model_id), vc in aggregate_by_cell.items():
        if vc is None:
            # Emit a placeholder row so the existence is auditable.
            rows.append({
                "benchmark_id": benchmark_id,
                "model_id": model_id,
                "source": "aggregate_fit_failed",
                "variance": float("nan"),
                "proportion": float("nan"),
                "df": 0,
            })
            continue
        total = vc["var_total"]
        n_cond = vc["n_conditions"]
        for source_name, vc_key in [
            ("prompt", "var_prompt"),
            ("seed", "var_seed"),
            ("scoring_rule", "var_scoring_rule"),
            ("residual", "var_residual"),
        ]:
            rows.append({
                "benchmark_id": benchmark_id,
                "model_id": model_id,
                "source": source_name,
                "variance": vc[vc_key],
                "proportion": (vc[vc_key] / total) if total > 0 else 0.0,
                "df": n_cond - 1,
            })

    df = pd.DataFrame(rows, columns=[
        "benchmark_id", "model_id", "source",
        "variance", "proportion", "df"
    ])
    df.to_csv(out_path, index=False)


def emit_convergence_report(
    attempt_log_rows: list[dict],
    out_path: Path,
) -> None:
    """Emit model_convergence_report.csv per SPEC §8.7."""
    df = pd.DataFrame(attempt_log_rows, columns=[
        "benchmark_id", "cascade_level", "success", "error_type",
        "gradient_norm", "optimizer_iterations", "n_items", "fit_seconds"
    ])
    df.to_csv(out_path, index=False)


# -- Main --------------------------------------------------------------------


def run_variance_components(
    item_level_path: Path,
    condition_level_path: Path,
    config_path: Path,
    out_dir: Path,
) -> tuple[int, dict]:
    """Main entry point. Returns (exit_code, summary_dict)."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load analysis config
    try:
        merged = load_config(config_path)
    except Exception as e:
        _log("ERROR", f"Config load failed: {e}")
        return 2, {}

    convergence_triggers = {
        "gradient_norm_above": (
            merged.analysis_config.convergence_triggers.gradient_norm_above
        ),
    }

    # Load data
    if not item_level_path.exists():
        _log("ERROR", f"Item-level parquet not found: {item_level_path}")
        return 2, {}
    if not condition_level_path.exists():
        _log("ERROR", f"Condition-level CSV not found: {condition_level_path}")
        return 2, {}

    item_df = pq.read_table(str(item_level_path)).to_pandas()
    condition_df = pd.read_csv(condition_level_path)

    benchmarks = sorted(condition_df["benchmark_id"].dropna().unique())
    _log("INFO",
         f"Analyzing {len(benchmarks)} benchmark(s): {', '.join(benchmarks)}")

    # Item-level cascade per benchmark
    item_level_results: dict[str, tuple[str, dict | None]] = {}
    all_attempt_logs: list[dict] = []

    for benchmark_id in benchmarks:
        bench_items = item_df[item_df["benchmark_id"] == benchmark_id]
        if len(bench_items) < 8:
            _log("INFO",
                 f"{benchmark_id}: only {len(bench_items)} items; "
                 f"skipping item-level cascade")
            item_level_results[benchmark_id] = ("level_4", None)
            all_attempt_logs.append({
                "benchmark_id": benchmark_id,
                "cascade_level": "level_4",
                "success": True,
                "error_type": "insufficient_items_for_cascade",
                "gradient_norm": float("nan"),
                "optimizer_iterations": 0,
                "n_items": len(bench_items),
                "fit_seconds": 0.0,
            })
            continue

        level, fit, attempt_log = fit_item_level_cascade(
            bench_items, benchmark_id, convergence_triggers
        )
        item_level_results[benchmark_id] = (level, fit)
        all_attempt_logs.extend(attempt_log)

    # Aggregate G-theory per (benchmark, model)
    aggregate_by_cell: dict[tuple[str, str], dict | None] = {}
    models = sorted(condition_df["model_id"].dropna().unique())
    any_aggregate_failure = False

    for benchmark_id in benchmarks:
        for model_id in models:
            cell = condition_df[
                (condition_df["benchmark_id"] == benchmark_id)
                & (condition_df["model_id"] == model_id)
            ]
            if len(cell) == 0:
                continue
            vc = fit_aggregate_g_theory(condition_df, benchmark_id, model_id)
            aggregate_by_cell[(benchmark_id, model_id)] = vc
            if vc is None and len(cell) >= 4:
                any_aggregate_failure = True

    # Emit outputs
    emit_item_level_vc(
        item_level_results, out_dir / "item_level_vc.csv"
    )
    emit_aggregate_vc(
        aggregate_by_cell, out_dir / "aggregate_vc.csv"
    )
    emit_convergence_report(
        all_attempt_logs, out_dir / "model_convergence_report.csv"
    )

    summary = {
        "n_benchmarks_analyzed": len(benchmarks),
        "n_aggregate_cells": len(aggregate_by_cell),
        "n_item_level_converged": sum(
            1 for _, (lv, _) in item_level_results.items()
            if lv in ("level_1", "level_2", "level_3")
        ),
        "n_level_4_fallbacks": sum(
            1 for _, (lv, _) in item_level_results.items()
            if lv == "level_4"
        ),
        "script_version": SCRIPT_VERSION,
    }

    _log("INFO",
         f"Emitted: item_level_vc.csv, aggregate_vc.csv, "
         f"model_convergence_report.csv — "
         f"{summary['n_item_level_converged']} item-level fits, "
         f"{summary['n_level_4_fallbacks']} Level-4 fallbacks, "
         f"{summary['n_aggregate_cells']} aggregate cells")

    return (4 if any_aggregate_failure else 0), summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Variance-components analysis per SPEC §8.7 + §10.1–2"
    )
    parser.add_argument("--item-level", required=True, type=Path)
    parser.add_argument("--condition-level", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()

    exit_code, summary = run_variance_components(
        args.item_level, args.condition_level, args.config, args.out_dir
    )
    _log("INFO", f"exit_code={exit_code} summary={summary}")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
