"""
Extract and align time series from Gradience vNext telemetry JSONL files.

Parses train_step, eval, and metrics events into aligned DataFrames
suitable for lead-lag analysis.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class RunMeta:
    """Metadata extracted from a telemetry run."""

    task: str = "unknown"
    seed: int = 0
    model: str = "unknown"
    lora_r: int = 0
    lora_alpha: float = 0.0
    total_train_steps: int = 0
    total_eval_events: int = 0
    eval_interval: int = 0  # inferred from eval step spacing
    train_log_interval: int = 0  # inferred from train step spacing
    source_path: str = ""


@dataclass
class ExtractedRun:
    """Parsed time series from a single training run."""

    meta: RunMeta
    train: pd.DataFrame  # columns: step, loss, lr, grad_norm, epoch
    eval: pd.DataFrame  # columns: step, eval_loss, eval_accuracy, ...
    aligned: pd.DataFrame  # eval-step indexed, with aggregated geometric features


def _parse_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read all JSON lines from a file."""
    events = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events


def _extract_meta(events: list[dict], path: Path) -> RunMeta:
    """Extract run metadata from run_start event."""
    meta = RunMeta(source_path=str(path))

    for evt in events:
        if evt.get("event") == "run_start":
            cfg = evt.get("config", {})
            meta.task = cfg.get("task_profile", "unknown")
            meta.model = cfg.get("model_name", "unknown")
            training = cfg.get("training", {})
            meta.seed = training.get("seed", 0)
            lora = cfg.get("lora", {})
            meta.lora_r = lora.get("r", 0)
            meta.lora_alpha = lora.get("alpha", 0.0)
            break

    # Count events
    train_steps = [e["step"] for e in events if e.get("event") == "train_step"]
    eval_steps = [e["step"] for e in events if e.get("event") == "eval"]

    meta.total_train_steps = len(train_steps)
    meta.total_eval_events = len(eval_steps)

    # Infer intervals
    if len(train_steps) >= 2:
        diffs = np.diff(sorted(train_steps))
        meta.train_log_interval = int(np.median(diffs))
    if len(eval_steps) >= 2:
        diffs = np.diff(sorted(eval_steps))
        meta.eval_interval = int(np.median(diffs))

    return meta


def _build_train_df(events: list[dict]) -> pd.DataFrame:
    """Build DataFrame from train_step events."""
    rows = []
    for evt in events:
        if evt.get("event") != "train_step":
            continue
        rows.append(
            {
                "step": evt["step"],
                "loss": evt.get("loss"),
                "lr": evt.get("lr"),
                "grad_norm": evt.get("grad_norm"),
                "epoch": evt.get("epoch"),
            }
        )

    if not rows:
        return pd.DataFrame(columns=["step", "loss", "lr", "grad_norm", "epoch"])

    df = pd.DataFrame(rows).sort_values("step").reset_index(drop=True)
    return df


def _build_eval_df(events: list[dict]) -> pd.DataFrame:
    """Build DataFrame from eval events."""
    rows = []
    for evt in events:
        if evt.get("event") != "eval":
            continue
        metrics = evt.get("metrics", {})
        row = {"step": evt["step"]}
        # Normalize metric names with eval_ prefix
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                key = k if k.startswith("eval_") else f"eval_{k}"
                row[key] = v
        rows.append(row)

    if not rows:
        return pd.DataFrame(columns=["step", "eval_loss", "eval_accuracy"])

    df = pd.DataFrame(rows).sort_values("step").reset_index(drop=True)
    return df


def _align_to_eval_grid(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    eval_interval: int,
) -> pd.DataFrame:
    """
    Align geometric features to the eval-step grid.

    For each eval step t, aggregate geometric features from the preceding
    interval (t - eval_interval, t].

    Returns one row per eval step with both geometric aggregates and eval metrics.
    """
    if eval_df.empty or train_df.empty:
        return pd.DataFrame()

    eval_steps = sorted(eval_df["step"].values)
    rows = []

    for i, t in enumerate(eval_steps):
        # Preceding interval: (prev_eval, t]
        prev_t = eval_steps[i - 1] if i > 0 else 0
        mask = (train_df["step"] > prev_t) & (train_df["step"] <= t)
        window = train_df[mask]

        row: dict[str, Any] = {"step": t, "eval_index": i}

        # Aggregate grad_norm in the window
        if "grad_norm" in window.columns and not window["grad_norm"].dropna().empty:
            gn = window["grad_norm"].dropna()
            row["grad_norm_mean"] = float(gn.mean())
            row["grad_norm_max"] = float(gn.max())
            row["grad_norm_min"] = float(gn.min())
            row["grad_norm_std"] = float(gn.std()) if len(gn) > 1 else 0.0
            row["grad_norm_last"] = float(gn.iloc[-1])
        else:
            row["grad_norm_mean"] = np.nan
            row["grad_norm_max"] = np.nan
            row["grad_norm_min"] = np.nan
            row["grad_norm_std"] = np.nan
            row["grad_norm_last"] = np.nan

        # Aggregate loss in the window (training loss, distinct from eval_loss)
        if "loss" in window.columns and not window["loss"].dropna().empty:
            tl = window["loss"].dropna()
            row["train_loss_mean"] = float(tl.mean())
            row["train_loss_last"] = float(tl.iloc[-1])
        else:
            row["train_loss_mean"] = np.nan
            row["train_loss_last"] = np.nan

        # Add eval metrics for this step
        eval_row = eval_df[eval_df["step"] == t]
        if not eval_row.empty:
            for col in eval_row.columns:
                if col != "step":
                    row[col] = float(eval_row[col].iloc[0])

        rows.append(row)

    aligned = pd.DataFrame(rows)

    # Compute delta features (change from previous eval interval)
    for col in ["grad_norm_mean", "grad_norm_max", "grad_norm_last", "train_loss_mean"]:
        if col in aligned.columns:
            aligned[f"{col}_delta"] = aligned[col].diff()

    # Compute acceleration (second difference)
    if "grad_norm_mean" in aligned.columns:
        aligned["grad_norm_mean_accel"] = aligned["grad_norm_mean_delta"].diff()

    return aligned


def extract_timeseries(jsonl_path: str | Path) -> ExtractedRun:
    """
    Parse a JSONL telemetry file into aligned time series.

    Parameters
    ----------
    jsonl_path : path to a Gradience vNext telemetry JSONL file

    Returns
    -------
    ExtractedRun with meta, train, eval, and aligned DataFrames
    """
    path = Path(jsonl_path)
    events = _parse_jsonl(path)

    meta = _extract_meta(events, path)
    train_df = _build_train_df(events)
    eval_df = _build_eval_df(events)

    aligned = _align_to_eval_grid(train_df, eval_df, meta.eval_interval)

    return ExtractedRun(
        meta=meta,
        train=train_df,
        eval=eval_df,
        aligned=aligned,
    )


def extract_all_runs(
    bench_runs_dir: str | Path,
    min_eval_events: int = 4,
    min_train_steps: int = 10,
) -> list[ExtractedRun]:
    """
    Extract time series from all run.jsonl files under a bench_runs directory.

    Filters to runs meeting minimum data requirements.
    """
    base = Path(bench_runs_dir)
    runs = []

    for jsonl_path in sorted(base.rglob("run.jsonl")):
        try:
            run = extract_timeseries(jsonl_path)
            if (
                run.meta.total_eval_events >= min_eval_events
                and run.meta.total_train_steps >= min_train_steps
            ):
                runs.append(run)
        except Exception:
            continue

    return runs


def save_aligned(runs: list[ExtractedRun], output_dir: str | Path) -> list[Path]:
    """Save aligned DataFrames as parquet files."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = []

    for run in runs:
        if run.aligned.empty:
            continue
        # Create a descriptive filename
        src = Path(run.meta.source_path)
        # Use parent directory names for identification
        parts = src.relative_to(src.parents[2]).parent.parts if len(src.parts) > 3 else src.parent.parts[-2:]
        name = "__".join(parts) + f"_seed{run.meta.seed}_r{run.meta.lora_r}.parquet"
        dest = out / name
        run.aligned.to_parquet(dest, index=False)
        paths.append(dest)

    return paths
