"""05_make_condition_scores.py — SPEC_CPU_v0_2 §8.6

Aggregate the item-level parquet (output of 04_normalize_outputs.py) to a
condition-level CSV (one row per condition_id) conforming to SPEC §5.6.

Usage:
    python scripts/05_make_condition_scores.py \\
      --item-level runs/normalized/item_level_primary.parquet \\
      --out runs/normalized/condition_level_primary.csv

Aggregation rules (SPEC §8.6):
    For each condition_id (group of item-level rows):
        num_items     = len(group)
        num_correct   = group["is_correct"].sum()
        accuracy      = num_correct / num_items
        parseability  = group["parseable"].sum() / num_items   if any non-null
                         else None
        mean_gen_len  = group["generation_length_tokens"].mean() if any non-null
                         else None

Carry-over columns from item-level (constant within a condition_id): tier,
benchmark_id, subject_id, model_id, prompt_id, seed_id, fewshot_k,
scoring_rule_id. condition_status is fixed to 'complete' since only complete
conditions reach this stage.

Exit codes per SPEC §11.1:
    0 — success
    1 — generic failure
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

import pandas as pd  # noqa: E402


CONDITION_FIELDNAMES: list[str] = [
    "condition_id",
    "tier",
    "benchmark_id",
    "subject_id",
    "model_id",
    "prompt_id",
    "seed_id",
    "fewshot_k",
    "scoring_rule_id",
    "num_items",
    "num_correct",
    "accuracy",
    "parseability_rate",
    "mean_generation_length_tokens",
    "condition_status",
]

CARRY_OVER_COLUMNS: list[str] = [
    "tier",
    "benchmark_id",
    "subject_id",
    "model_id",
    "prompt_id",
    "seed_id",
    "fewshot_k",
    "scoring_rule_id",
]


# -- Logging -----------------------------------------------------------------


def _log(level: str, msg: str) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"[{ts}] [{level}] [05_make_condition_scores] {msg}", file=sys.stderr)


# -- Aggregation -------------------------------------------------------------


def _format_value(col: str, v: Any) -> str:
    """Stringify a CSV cell. Null/NaN -> empty string."""
    if v is None:
        return ""
    if isinstance(v, float):
        # NaN check: NaN != NaN.
        if v != v:  # noqa: PLR0124
            return ""
        return repr(v) if not v.is_integer() else str(v)
    return str(v)


def aggregate(item_level: pd.DataFrame) -> list[dict[str, Any]]:
    """Group item-level rows by condition_id and compute per-condition stats.

    Returns a list of dicts in CONDITION_FIELDNAMES order. Output ordering
    is deterministic: groups appear in sorted condition_id order.
    """
    rows: list[dict[str, Any]] = []
    if item_level.empty:
        return rows

    for condition_id in sorted(item_level["condition_id"].unique()):
        group = item_level[item_level["condition_id"] == condition_id]
        num_items = int(len(group))
        num_correct = int(group["is_correct"].sum())
        accuracy = float(num_correct) / float(num_items) if num_items else 0.0

        # Parseability — only meaningful for G&P (parseable column populated).
        parseability_rate: float | None
        if "parseable" in group.columns and group["parseable"].notna().any():
            # Sum True values among non-null entries; rate over num_items.
            parseability_rate = float(group["parseable"].fillna(False).sum()) / float(num_items)
        else:
            parseability_rate = None

        # Mean generation length — only for G&P.
        mean_gen_length: float | None
        if (
            "generation_length_tokens" in group.columns
            and group["generation_length_tokens"].notna().any()
        ):
            mean_gen_length = float(group["generation_length_tokens"].mean())
        else:
            mean_gen_length = None

        # Carry-over columns: take the first row's value, normalizing NaN to None.
        carry: dict[str, Any] = {}
        first = group.iloc[0]
        for col in CARRY_OVER_COLUMNS:
            v = first.get(col, None)
            if v is None:
                carry[col] = None
            elif isinstance(v, float) and v != v:  # noqa: PLR0124 — NaN
                carry[col] = None
            else:
                carry[col] = v

        rows.append({
            "condition_id": condition_id,
            "tier": carry["tier"],
            "benchmark_id": carry["benchmark_id"],
            "subject_id": carry["subject_id"],
            "model_id": carry["model_id"],
            "prompt_id": carry["prompt_id"],
            "seed_id": carry["seed_id"],
            "fewshot_k": int(carry["fewshot_k"]) if carry["fewshot_k"] is not None else None,
            "scoring_rule_id": carry["scoring_rule_id"],
            "num_items": num_items,
            "num_correct": num_correct,
            "accuracy": accuracy,
            "parseability_rate": parseability_rate,
            "mean_generation_length_tokens": mean_gen_length,
            "condition_status": "complete",
        })

    return rows


def write_condition_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(CONDITION_FIELDNAMES)
        for r in rows:
            writer.writerow([_format_value(c, r.get(c)) for c in CONDITION_FIELDNAMES])


# -- Main --------------------------------------------------------------------


def run(item_level_path: Path, out_path: Path) -> int:
    if not item_level_path.exists():
        _log("ERROR", f"Item-level parquet not found: {item_level_path}")
        return 1

    df = pd.read_parquet(item_level_path)
    _log("INFO",
         f"Loaded {len(df)} item rows from {item_level_path} "
         f"({df['condition_id'].nunique() if not df.empty else 0} conditions).")

    rows = aggregate(df)
    write_condition_csv(out_path, rows)
    _log("INFO", f"Wrote {len(rows)} condition rows to {out_path}.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Aggregate item-level normalized parquet to condition-level CSV "
                    "(SPEC_CPU_v0_2 §8.6)."
    )
    parser.add_argument("--item-level", required=True, type=Path,
                        help="Path to item-level parquet from 04_normalize_outputs.py.")
    parser.add_argument("--out", required=True, type=Path,
                        help="Output condition-level CSV path.")
    args = parser.parse_args()

    try:
        return run(args.item_level, args.out)
    except Exception as e:  # noqa: BLE001
        _log("CRITICAL", f"Unhandled exception: {type(e).__name__}: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
