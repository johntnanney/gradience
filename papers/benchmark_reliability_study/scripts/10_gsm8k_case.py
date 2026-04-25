"""10_gsm8k_case.py — SPEC_CPU_v0_2 §8.11

GSM8K secondary-tier case study with extraction-variant facet.

Three analyses:

    1. Tolerance schedule per (model, extraction_variant) cell, where
       extraction_variant is the GSM8K scoring rule
       ('generate_parse_strict' or 'generate_parse_permissive').
       Computes SEM as std-across-conditions, tolerance = 2*SEM, with
       bootstrap CI on the tolerance and §9.1 decimal-place licensing.

    2. Extraction sensitivity: per (model, prompt, seed), the accuracy
       delta between strict and permissive extraction. Reports per-cell
       deltas with bootstrap CI.

    3. Parseability aggregated by (model, extraction_variant) — the
       parseability_rate column already on the condition-level CSV.

Usage:
    python scripts/10_gsm8k_case.py \\
      --item-level runs/normalized/item_level_gsm8k.parquet \\
      --condition-level runs/normalized/condition_level_gsm8k.csv \\
      --analysis-config configs/analysis_config.yaml \\
      --out-dir analysis/gsm8k_case/

Optional:
    --benchmark-id gsm8k
        Override the benchmark filter (default 'gsm8k').

Exit codes per SPEC §11.1:
    0 — success
    1 — generic failure (implementation error)
"""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Make the paper-root package importable without requiring an install.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from gradience_study.config import load_config  # noqa: E402


# -- Logging -----------------------------------------------------------------


def _log(level: str, msg: str) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"[{ts}] [{level}] [10_gsm8k_case] {msg}", file=sys.stderr)


# -- Output schemas ----------------------------------------------------------


TOLERANCE_FIELDS: list[str] = [
    "model_id",
    "extraction_variant",
    "sem",
    "tolerance",
    "tolerance_ci_lower",
    "tolerance_ci_upper",
    "licensed_precision",
]

EXTRACTION_FIELDS: list[str] = [
    "model_id",
    "prompt_id",
    "seed_id",
    "accuracy_strict",
    "accuracy_permissive",
    "delta",
    "delta_ci_lower",
    "delta_ci_upper",
]

PARSEABILITY_FIELDS: list[str] = [
    "model_id",
    "extraction_variant",
    "mean_parseability",
    "n_conditions",
]


STRICT_RULE_ID = "generate_parse_strict"
PERMISSIVE_RULE_ID = "generate_parse_permissive"


# -- Decimal-place licensing (SPEC §9.1) -------------------------------------


def licensed_precision(tolerance: float) -> str:
    if tolerance != tolerance:  # NaN
        return "interval_required"
    if tolerance < 0.0005:
        return "three_decimal_accuracy"
    if tolerance < 0.005:
        return "two_decimal_accuracy"
    return "interval_required"


# -- Tolerance schedule ------------------------------------------------------


def compute_tolerance_schedule(
    condition_level: pd.DataFrame,
    n_resamples: int,
    seed: int,
    ci_lower_pct: float,
    ci_upper_pct: float,
) -> list[dict[str, Any]]:
    """For each (model_id, extraction_variant) cell, compute tolerance and
    bootstrap CI. extraction_variant is the GSM8K scoring_rule_id.
    """
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    grouped = condition_level.groupby(["model_id", "scoring_rule_id"], dropna=True)
    for (model_id, rule_id), group in grouped:
        scores = group["accuracy"].to_numpy(dtype=float)
        n = int(len(scores))
        if n < 2:
            sem_val = 0.0
            tolerance_point = 0.0
            ci_lo = 0.0
            ci_hi = 0.0
        else:
            sem_val = float(np.std(scores, ddof=1))
            tolerance_point = float(2.0 * sem_val)
            boot = np.empty(n_resamples, dtype=float)
            for k in range(n_resamples):
                idx = rng.integers(0, n, size=n)
                resampled = scores[idx]
                # Edge case: a resample of constants gives std=0; keep it.
                boot[k] = float(2.0 * np.std(resampled, ddof=1)) if n > 1 else 0.0
            ci_lo = float(np.percentile(boot, ci_lower_pct))
            ci_hi = float(np.percentile(boot, ci_upper_pct))

        rows.append(
            {
                "model_id": model_id,
                "extraction_variant": rule_id,
                "sem": sem_val,
                "tolerance": tolerance_point,
                "tolerance_ci_lower": ci_lo,
                "tolerance_ci_upper": ci_hi,
                "licensed_precision": licensed_precision(tolerance_point),
            }
        )
    return rows


# -- Extraction sensitivity --------------------------------------------------


def compute_extraction_sensitivity(
    condition_level: pd.DataFrame,
    n_resamples: int,
    seed: int,
    ci_lower_pct: float,
    ci_upper_pct: float,
) -> list[dict[str, Any]]:
    """Per (model, prompt, seed) cell: delta = accuracy_permissive - accuracy_strict.

    Bootstrap CI is computed across all (model, prompt, seed) cells —
    i.e. a cross-cell sampling of deltas — and the same CI is attached
    to every row. (Single-cell bootstrap would resample a 1-d delta value
    and return a degenerate CI, which is uninformative.)
    """
    # Pivot to wide form on scoring_rule_id.
    keys = ["model_id", "prompt_id", "seed_id"]
    pivot = (
        condition_level.pivot_table(
            index=keys,
            columns="scoring_rule_id",
            values="accuracy",
            aggfunc="mean",
        )
        .reset_index()
    )
    if STRICT_RULE_ID not in pivot.columns or PERMISSIVE_RULE_ID not in pivot.columns:
        _log(
            "WARNING",
            f"Extraction sensitivity: missing scoring rules in pivot "
            f"(have {list(pivot.columns)}). Returning empty.",
        )
        return []
    pivot = pivot.dropna(subset=[STRICT_RULE_ID, PERMISSIVE_RULE_ID])
    if pivot.empty:
        return []

    pivot["delta"] = pivot[PERMISSIVE_RULE_ID] - pivot[STRICT_RULE_ID]
    deltas = pivot["delta"].to_numpy(dtype=float)

    rng = np.random.default_rng(seed)
    n = len(deltas)
    if n >= 2:
        boot_means = np.empty(n_resamples, dtype=float)
        for k in range(n_resamples):
            idx = rng.integers(0, n, size=n)
            boot_means[k] = float(np.mean(deltas[idx]))
        ci_lo = float(np.percentile(boot_means, ci_lower_pct))
        ci_hi = float(np.percentile(boot_means, ci_upper_pct))
    else:
        ci_lo = float(deltas[0])
        ci_hi = float(deltas[0])

    rows: list[dict[str, Any]] = []
    for _, r in pivot.iterrows():
        rows.append(
            {
                "model_id": r["model_id"],
                "prompt_id": r["prompt_id"],
                "seed_id": r["seed_id"],
                "accuracy_strict": float(r[STRICT_RULE_ID]),
                "accuracy_permissive": float(r[PERMISSIVE_RULE_ID]),
                "delta": float(r["delta"]),
                "delta_ci_lower": ci_lo,
                "delta_ci_upper": ci_hi,
            }
        )
    return rows


# -- Parseability aggregation ------------------------------------------------


def compute_parseability(
    condition_level: pd.DataFrame,
) -> list[dict[str, Any]]:
    if "parseability_rate" not in condition_level.columns:
        return []
    sub = condition_level.dropna(subset=["parseability_rate"])
    if sub.empty:
        return []
    grouped = sub.groupby(["model_id", "scoring_rule_id"], dropna=True)["parseability_rate"]
    rows: list[dict[str, Any]] = []
    for (model_id, rule_id), values in grouped:
        rows.append(
            {
                "model_id": model_id,
                "extraction_variant": rule_id,
                "mean_parseability": float(values.mean()),
                "n_conditions": int(len(values)),
            }
        )
    return rows


# -- CSV writer --------------------------------------------------------------


def _format_value(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        if v != v:
            return ""
        return repr(v)
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v)


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(fields)
        for r in rows:
            writer.writerow([_format_value(r.get(c)) for c in fields])


# -- Main pipeline -----------------------------------------------------------


def _load_bootstrap_config(
    analysis_config_path: Path,
) -> tuple[int, int, float, float]:
    try:
        merged = load_config(analysis_config_path)
        boot = merged.analysis_config.bootstrap
        return (
            int(boot.n_resamples),
            int(boot.random_seed),
            float(boot.ci_lower_percentile),
            float(boot.ci_upper_percentile),
        )
    except Exception:  # noqa: BLE001
        import yaml  # noqa: PLC0415

        with open(analysis_config_path, "r", encoding="utf-8") as f:
            ac = yaml.safe_load(f)
        boot = ac["bootstrap"]
        return (
            int(boot["n_resamples"]),
            int(boot["random_seed"]),
            float(boot["ci_lower_percentile"]),
            float(boot["ci_upper_percentile"]),
        )


def run(
    item_level_path: Path,
    condition_level_path: Path,
    analysis_config_path: Path,
    out_dir: Path,
    benchmark_id: str,
) -> int:
    if not condition_level_path.exists():
        _log("ERROR", f"Condition-level CSV not found: {condition_level_path}")
        return 1
    # item-level parquet is currently used only for forward-compatibility
    # (e.g., per-item parseability). The current spec computes everything
    # required from condition-level rows, but we still error if the file
    # is missing — its presence is a contract.
    if not item_level_path.exists():
        _log("ERROR", f"Item-level parquet not found: {item_level_path}")
        return 1

    boot_n, boot_seed, ci_lo, ci_hi = _load_bootstrap_config(analysis_config_path)

    cond = pd.read_csv(condition_level_path)
    _log(
        "INFO",
        f"Loaded {len(cond)} condition rows from {condition_level_path} "
        f"({cond['benchmark_id'].nunique() if not cond.empty else 0} benchmarks).",
    )
    cond = cond[cond["benchmark_id"] == benchmark_id].copy()
    if cond.empty:
        _log("ERROR", f"No condition rows for benchmark_id={benchmark_id!r}.")
        return 1

    # Tolerance schedule.
    tolerance_rows = compute_tolerance_schedule(
        cond, n_resamples=boot_n, seed=boot_seed,
        ci_lower_pct=ci_lo, ci_upper_pct=ci_hi,
    )
    write_csv(out_dir / "gsm8k_tolerance_schedule.csv", tolerance_rows, TOLERANCE_FIELDS)
    _log("INFO", f"Wrote gsm8k_tolerance_schedule.csv ({len(tolerance_rows)} rows).")

    # Extraction sensitivity.
    extraction_rows = compute_extraction_sensitivity(
        cond, n_resamples=boot_n, seed=boot_seed,
        ci_lower_pct=ci_lo, ci_upper_pct=ci_hi,
    )
    write_csv(
        out_dir / "gsm8k_extraction_sensitivity.csv",
        extraction_rows,
        EXTRACTION_FIELDS,
    )
    _log(
        "INFO",
        f"Wrote gsm8k_extraction_sensitivity.csv ({len(extraction_rows)} rows).",
    )

    # Parseability.
    parseability_rows = compute_parseability(cond)
    write_csv(
        out_dir / "gsm8k_parseability.csv", parseability_rows, PARSEABILITY_FIELDS
    )
    _log("INFO", f"Wrote gsm8k_parseability.csv ({len(parseability_rows)} rows).")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="GSM8K case-study analysis (SPEC_CPU_v0_2 §8.11)."
    )
    parser.add_argument(
        "--item-level",
        required=True,
        type=Path,
        help="Path to GSM8K item-level parquet.",
    )
    parser.add_argument(
        "--condition-level",
        required=True,
        type=Path,
        help="Path to GSM8K condition-level CSV.",
    )
    parser.add_argument(
        "--analysis-config",
        required=True,
        type=Path,
        help="Path to study_config.yaml or bare analysis_config.yaml.",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        type=Path,
        help="Directory for analysis output (created if missing).",
    )
    parser.add_argument(
        "--benchmark-id",
        type=str,
        default="gsm8k",
        help="Benchmark to filter on. Default: gsm8k.",
    )
    args = parser.parse_args()

    try:
        return run(
            args.item_level,
            args.condition_level,
            args.analysis_config,
            args.out_dir,
            args.benchmark_id,
        )
    except Exception as e:  # noqa: BLE001
        _log("CRITICAL", f"Unhandled exception: {type(e).__name__}: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
