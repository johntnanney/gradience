"""
Retrospective early-stopping simulation.

Simulates what would have happened if training had stopped based on
geometric signals rather than running to completion.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class StoppingRule:
    """A stopping rule with parameters."""

    name: str
    trigger_fn: Callable[[pd.DataFrame], int | None]
    params: dict[str, Any]


@dataclass
class StoppingResult:
    """Result of applying a stopping rule to a run."""

    rule_name: str
    stop_step: int | None
    stop_index: int | None
    total_steps: int
    pct_training_saved: float
    metric_at_stop: float | None
    metric_at_end: float
    metric_delta: float  # metric_at_stop - metric_at_end


def plateau_rule(
    aligned_df: pd.DataFrame,
    series_col: str,
    delta_threshold: float,
    patience: int,
) -> int | None:
    """
    Detect when a series has plateaued.

    Returns the step at which the series has been within delta_threshold
    (relative change) of its value `patience` intervals ago,
    for `patience` consecutive intervals.

    Returns None if the rule never triggers.
    """
    series = aligned_df[series_col].values
    steps = aligned_df["step"].values

    if len(series) < patience + 1:
        return None

    consecutive = 0
    for i in range(1, len(series)):
        prev = series[i - 1]
        curr = series[i]

        if abs(prev) < 1e-12:
            rel_change = abs(curr) if abs(curr) > delta_threshold else 0.0
        else:
            rel_change = abs((curr - prev) / prev)

        if rel_change < delta_threshold:
            consecutive += 1
            if consecutive >= patience:
                return int(steps[i])
        else:
            consecutive = 0

    return None


def loss_patience_rule(
    aligned_df: pd.DataFrame,
    patience: int,
) -> int | None:
    """
    Standard loss-based early stopping: stop when eval_loss hasn't improved
    for `patience` consecutive eval intervals.
    """
    if "eval_loss" not in aligned_df.columns:
        return None

    losses = aligned_df["eval_loss"].values
    steps = aligned_df["step"].values

    if len(losses) < patience + 1:
        return None

    best_loss = losses[0]
    wait = 0

    for i in range(1, len(losses)):
        if losses[i] < best_loss - 1e-8:
            best_loss = losses[i]
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                return int(steps[i])

    return None


def make_stopping_rules(
    delta_thresholds: list[float] | None = None,
    patience_values: list[int] | None = None,
) -> list[StoppingRule]:
    """
    Generate a set of stopping rules for simulation.

    Returns geometric plateau rules + loss-based baselines.
    """
    if delta_thresholds is None:
        delta_thresholds = [0.01, 0.02, 0.05, 0.10]
    if patience_values is None:
        patience_values = [2, 3]

    rules = []

    # Geometric plateau rules
    geo_cols = ["grad_norm_mean", "grad_norm_last", "train_loss_mean"]

    for col in geo_cols:
        for delta in delta_thresholds:
            for pat in patience_values:
                name = f"{col}_plateau_d{delta}_p{pat}"
                rules.append(
                    StoppingRule(
                        name=name,
                        trigger_fn=lambda df, c=col, d=delta, p=pat: plateau_rule(df, c, d, p),
                        params={"series": col, "delta": delta, "patience": pat},
                    )
                )

    # Loss-based baselines
    for pat in [2, 3, 5]:
        rules.append(
            StoppingRule(
                name=f"loss_patience_p{pat}",
                trigger_fn=lambda df, p=pat: loss_patience_rule(df, p),
                params={"patience": pat},
            )
        )

    return rules


def simulate_early_stopping(
    aligned_df: pd.DataFrame,
    rules: list[StoppingRule] | None = None,
    final_eval_metric: str = "eval_accuracy",
    fallback_metric: str = "eval_loss",
) -> list[StoppingResult]:
    """
    For each rule, determine when it would have stopped training and
    what the impact on the final metric would be.
    """
    if rules is None:
        rules = make_stopping_rules()

    # Determine which metric to use
    metric_col = final_eval_metric if final_eval_metric in aligned_df.columns else fallback_metric
    if metric_col not in aligned_df.columns:
        return []

    steps = aligned_df["step"].values
    total_steps = int(steps[-1]) if len(steps) > 0 else 0
    metric_at_end = float(aligned_df[metric_col].iloc[-1]) if len(aligned_df) > 0 else np.nan

    results = []

    for rule in rules:
        # Check if required columns exist
        series_col = rule.params.get("series", "")
        if series_col and series_col not in aligned_df.columns:
            continue

        try:
            stop_step = rule.trigger_fn(aligned_df)
        except (KeyError, ValueError):
            continue

        if stop_step is None:
            results.append(
                StoppingResult(
                    rule_name=rule.name,
                    stop_step=None,
                    stop_index=None,
                    total_steps=total_steps,
                    pct_training_saved=0.0,
                    metric_at_stop=metric_at_end,
                    metric_at_end=metric_at_end,
                    metric_delta=0.0,
                )
            )
            continue

        # Find the eval index closest to (but not after) stop_step
        stop_mask = aligned_df["step"] <= stop_step
        if not stop_mask.any():
            continue

        stop_idx = int(stop_mask.sum() - 1)
        metric_at_stop = float(aligned_df[metric_col].iloc[stop_idx])

        pct_saved = (total_steps - stop_step) / total_steps * 100 if total_steps > 0 else 0.0
        metric_delta = metric_at_stop - metric_at_end

        results.append(
            StoppingResult(
                rule_name=rule.name,
                stop_step=stop_step,
                stop_index=stop_idx,
                total_steps=total_steps,
                pct_training_saved=max(0.0, pct_saved),
                metric_at_stop=metric_at_stop,
                metric_at_end=metric_at_end,
                metric_delta=metric_delta,
            )
        )

    return results


def aggregate_stopping_results(
    all_results: dict[str, list[StoppingResult]],
) -> pd.DataFrame:
    """
    Aggregate stopping results across runs into a summary table.

    Parameters
    ----------
    all_results : dict mapping run_label -> list of StoppingResult

    Returns
    -------
    DataFrame with one row per rule, showing mean stats across runs.
    """
    rows = []
    rule_names: set[str] = set()
    for results in all_results.values():
        for r in results:
            rule_names.add(r.rule_name)

    for rule_name in sorted(rule_names):
        savings = []
        deltas = []
        triggered_count = 0
        total_count = 0

        for _run_label, results in all_results.items():
            for r in results:
                if r.rule_name != rule_name:
                    continue
                total_count += 1
                if r.stop_step is not None:
                    triggered_count += 1
                    savings.append(r.pct_training_saved)
                    deltas.append(r.metric_delta)

        if not savings:
            rows.append(
                {
                    "rule": rule_name,
                    "triggered": f"{triggered_count}/{total_count}",
                    "mean_pct_saved": 0.0,
                    "std_pct_saved": 0.0,
                    "mean_metric_delta": 0.0,
                    "std_metric_delta": 0.0,
                    "runs_within_1pct": total_count,
                }
            )
            continue

        savings_arr = np.array(savings)
        deltas_arr = np.array(deltas)

        rows.append(
            {
                "rule": rule_name,
                "triggered": f"{triggered_count}/{total_count}",
                "mean_pct_saved": float(savings_arr.mean()),
                "std_pct_saved": float(savings_arr.std()),
                "mean_metric_delta": float(deltas_arr.mean()),
                "std_metric_delta": float(deltas_arr.std()),
                "runs_within_1pct": int(np.sum(np.abs(deltas_arr) <= 0.01)),
            }
        )

    return pd.DataFrame(rows)
