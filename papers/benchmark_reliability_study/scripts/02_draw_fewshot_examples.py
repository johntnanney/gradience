"""02_draw_fewshot_examples.py — SPEC_CPU_v0_2 §8.3

Deterministically draw few-shot exemplars per (benchmark, subject, seed)
and verify no leakage between the few-shot pool and the eval split.

Usage:
    python scripts/02_draw_fewshot_examples.py \\
      --config configs/study_config.yaml \\
      --seeds 42 123 2024 \\
      --out manifests/fewshot_manifest.csv \\
      --lock-out preregistration/appendices/fewshot_draws_LOCKED.json

Exit codes per SPEC §11.1:
    0  — success
    2  — configuration validation failure
    3  — data contract violation (few-shot leakage)

Deferred execution against real data
------------------------------------
Real Hugging Face dataset access is deferred pending v1.1 pre-registration
lock. The default loader raises NotImplementedError; tests inject a fixture
loader directly into ``main()``.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

# Make the paper-root package importable without requiring an install.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gradience_study.config import (  # noqa: E402
    BenchmarkSpec,
    ConfigValidationError,
    FewshotLeakageError,
    MergedConfig,
    compute_config_hash,
    load_config,
)


DRAW_SCRIPT_VERSION = "0.1.0"
DRAW_ALGORITHM_DESCRIPTION = (
    "uniform random choice without replacement via numpy.random.default_rng(seed)"
)

# Column order for fewshot_manifest.csv. Tests verify this order.
MANIFEST_FIELDNAMES: list[str] = [
    "benchmark_id",
    "subject_id",
    "seed_id",
    "fewshot_k",
    "example_rank",
    "example_item_id",
    "example_source_split",
    "draw_timestamp",
    "draw_script_version",
]


# Loader signature: (hf_dataset, hf_config, split, dataset_version_hash) -> list[dict]
Loader = Callable[[str, str | None, str, str], list[dict]]


def _default_loader(
    hf_dataset: str,
    hf_config: str | None,
    split: str,
    dataset_version_hash: str,
) -> list[dict]:
    """Real-data loader stub.

    Raised at call time so the script's CLI surface and argument parsing
    work fine; tests inject a fixture loader before any data fetch.
    """
    raise NotImplementedError(
        "Real Hugging Face dataset access is deferred pending v1.1 "
        "pre-registration lock. Inject a loader function for testing."
    )


# -- Logging -----------------------------------------------------------------


def _log(level: str, msg: str) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"[{ts}] [{level}] [02_draw_fewshot_examples] {msg}", file=sys.stderr)


# -- Draw algorithm ----------------------------------------------------------


def _draw_fewshot(
    benchmark: BenchmarkSpec,
    subject_id: str | None,
    seed: int,
    loader: Loader,
) -> list[Any]:
    """Draw ``benchmark.fewshot_k`` exemplar IDs for one (subject, seed).

    Algorithm per SPEC §8.3:
        rng = numpy.random.default_rng(seed)
        indices = rng.choice(len(source), size=k, replace=False)
        return [source[i][item_id_field] for i in indices]
    """
    if benchmark.fewshot_source_split is None:
        raise ValueError(
            f"Benchmark '{benchmark.benchmark_id}' has fewshot_k > 0 but "
            f"fewshot_source_split is null. Cannot draw."
        )
    source = loader(
        benchmark.hf_dataset,
        benchmark.hf_config,
        benchmark.fewshot_source_split,
        benchmark.dataset_version_hash,
    )
    if subject_id is not None:
        source = [item for item in source if item.get("subject") == subject_id]

    n = len(source)
    k = benchmark.fewshot_k
    if k > n:
        raise ValueError(
            f"Benchmark '{benchmark.benchmark_id}' subject={subject_id}: "
            f"fewshot_k={k} exceeds source split size {n}."
        )

    rng = np.random.default_rng(seed)
    indices = rng.choice(n, size=k, replace=False)
    field = benchmark.item_id_field
    return [source[int(i)][field] for i in indices]


# -- Manifest construction ---------------------------------------------------


def _placeholder_row(benchmark: BenchmarkSpec, draw_timestamp: str) -> dict[str, Any]:
    """One row marker that a 0-shot benchmark exists in the manifest.

    Per SPEC §3.4 / §8.3: 0-shot benchmarks generate exactly one
    ``seed_id=none`` condition; this row lets downstream pipeline stages
    enumerate them without performing a draw.
    """
    return {
        "benchmark_id": benchmark.benchmark_id,
        "subject_id": "",
        "seed_id": "none",
        "fewshot_k": 0,
        "example_rank": 0,
        "example_item_id": "",
        "example_source_split": "",
        "draw_timestamp": draw_timestamp,
        "draw_script_version": DRAW_SCRIPT_VERSION,
    }


def _build_manifest(
    merged: MergedConfig,
    seeds: list[int],
    loader: Loader,
    draw_timestamp: str,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, dict[str, list[Any]]]]]:
    """Iterate primary benchmarks; build CSV rows and the LOCKED.json draws map.

    Returns:
        (rows, draws) where:
            rows is the list of CSV row dicts (in MANIFEST_FIELDNAMES order),
            draws is a nested dict: benchmark_id -> subject_or_'null' ->
                seed_id -> [item_id, ...].
    """
    rows: list[dict[str, Any]] = []
    draws: dict[str, dict[str, dict[str, list[Any]]]] = {}

    for benchmark in merged.benchmarks:
        # Iterate both primary and secondary tier benchmarks: secondary
        # benchmarks (e.g. GSM8K) also need few-shot draws and they must
        # land in the same locked manifest. SPEC §11's minimal-command
        # sequence does not have a separate fewshot drawer for the
        # secondary tier; this is the single drawer for the whole study.

        if benchmark.fewshot_k == 0:
            # Marker row only; no draws populated.
            rows.append(_placeholder_row(benchmark, draw_timestamp))
            _log(
                "INFO",
                f"Benchmark '{benchmark.benchmark_id}': 0-shot "
                f"(placeholder row, no draw).",
            )
            continue

        per_benchmark: dict[str, dict[str, list[Any]]] = {}
        subjects: tuple[str | None, ...]
        if benchmark.subjects:
            subjects = tuple(benchmark.subjects)
        else:
            subjects = (None,)

        for subject_id in subjects:
            subj_key = subject_id if subject_id is not None else "null"
            per_subject: dict[str, list[Any]] = {}
            for seed in seeds:
                item_ids = _draw_fewshot(benchmark, subject_id, seed, loader)
                seed_str = str(seed)
                per_subject[seed_str] = list(item_ids)
                for rank, item_id in enumerate(item_ids):
                    rows.append(
                        {
                            "benchmark_id": benchmark.benchmark_id,
                            "subject_id": subject_id if subject_id is not None else "",
                            "seed_id": seed_str,
                            "fewshot_k": benchmark.fewshot_k,
                            "example_rank": rank,
                            "example_item_id": item_id,
                            "example_source_split": benchmark.fewshot_source_split or "",
                            "draw_timestamp": draw_timestamp,
                            "draw_script_version": DRAW_SCRIPT_VERSION,
                        }
                    )
            per_benchmark[subj_key] = per_subject
            _log(
                "INFO",
                f"Benchmark '{benchmark.benchmark_id}' subject={subj_key}: "
                f"drew {benchmark.fewshot_k} exemplars × {len(seeds)} seed(s).",
            )

        draws[benchmark.benchmark_id] = per_benchmark

    return rows, draws


# -- Leakage check -----------------------------------------------------------


def _validate_no_leakage(
    rows: list[dict[str, Any]],
    merged: MergedConfig,
    loader: Loader,
) -> None:
    """Hard-fail if any drawn fewshot ID appears in the eval split.

    Per SPEC §3.7 / §8.3. Raises FewshotLeakageError on any intersection.
    """
    fewshot_ids_by_bench: dict[str, set[Any]] = {}
    for row in rows:
        if not row["example_item_id"]:
            # Skip placeholder rows for 0-shot benchmarks.
            continue
        fewshot_ids_by_bench.setdefault(row["benchmark_id"], set()).add(
            row["example_item_id"]
        )

    for benchmark in merged.benchmarks:
        # Iterate both primary and secondary tier benchmarks: secondary
        # benchmarks (e.g. GSM8K) also need few-shot draws and they must
        # land in the same locked manifest. SPEC §11's minimal-command
        # sequence does not have a separate fewshot drawer for the
        # secondary tier; this is the single drawer for the whole study.
        fewshot_ids = fewshot_ids_by_bench.get(benchmark.benchmark_id, set())
        if not fewshot_ids:
            continue

        eval_items = loader(
            benchmark.hf_dataset,
            benchmark.hf_config,
            benchmark.eval_split,
            benchmark.dataset_version_hash,
        )
        eval_ids = {item[benchmark.item_id_field] for item in eval_items}
        intersection = fewshot_ids & eval_ids
        if intersection:
            leaked = sorted(intersection, key=lambda x: str(x))[:10]
            raise FewshotLeakageError(
                f"Benchmark {benchmark.benchmark_id}: "
                f"{len(intersection)} item(s) in both fewshot and eval "
                f"sets. Leaked IDs (first 10): {leaked}"
            )


# -- Output writers ----------------------------------------------------------


def _write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=MANIFEST_FIELDNAMES, lineterminator="\n"
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_lock(
    path: Path,
    draws: dict[str, dict[str, dict[str, list[Any]]]],
    config_hash_short: str,
    seeds: list[int],
) -> None:
    payload = {
        "draw_script_version": DRAW_SCRIPT_VERSION,
        "draw_algorithm_description": DRAW_ALGORITHM_DESCRIPTION,
        "config_hash": config_hash_short,
        "seeds": list(seeds),
        "draws": draws,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


# -- Main --------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Draw few-shot exemplars per SPEC_CPU_v0_2 §8.3"
    )
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Path to study_config.yaml",
    )
    parser.add_argument(
        "--seeds",
        required=True,
        type=int,
        nargs="+",
        help="Integer seeds for deterministic draw (e.g. 42 123 2024)",
    )
    parser.add_argument(
        "--out",
        required=True,
        type=Path,
        help="Output path for fewshot_manifest.csv",
    )
    parser.add_argument(
        "--lock-out",
        required=True,
        type=Path,
        help="Output path for fewshot_draws_LOCKED.json",
    )
    parser.add_argument(
        "--fixed-timestamp",
        default=None,
        help=(
            "Override the draw_timestamp with a fixed ISO-8601 UTC value "
            "for reproducibility testing. If omitted, current UTC time is used."
        ),
    )
    return parser.parse_args(argv)


def main(
    argv: list[str] | None = None,
    loader: Loader | None = None,
) -> int:
    args = _parse_args(argv)
    loader = loader or _default_loader

    # --- Load configs ------------------------------------------------------

    try:
        merged = load_config(args.config)
    except FileNotFoundError as e:
        _log("ERROR", f"Config file not found: {e}")
        return 2
    except ConfigValidationError as e:
        _log("ERROR", f"Config validation error: {e}")
        return 2
    except KeyError as e:
        _log("ERROR", f"Missing required config field: {e}")
        return 2

    _, config_hash_short = compute_config_hash(merged.raw_dict)

    draw_timestamp = args.fixed_timestamp or datetime.now(timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )

    # --- Draw --------------------------------------------------------------

    try:
        rows, draws = _build_manifest(merged, list(args.seeds), loader, draw_timestamp)
    except NotImplementedError as e:
        # Real-loader stub or test misconfiguration.
        _log("ERROR", str(e))
        return 1

    # --- Leakage check -----------------------------------------------------

    try:
        _validate_no_leakage(rows, merged, loader)
    except FewshotLeakageError as e:
        _log("ERROR", f"Few-shot leakage detected: {e}")
        return 3

    # --- Emit --------------------------------------------------------------

    _write_manifest(args.out, rows)
    _write_lock(args.lock_out, draws, config_hash_short, list(args.seeds))

    _log(
        "INFO",
        f"Wrote {len(rows)} manifest row(s) to {args.out} "
        f"(config_hash={config_hash_short}); "
        f"lock file at {args.lock_out}.",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
