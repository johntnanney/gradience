"""01_build_manifests.py — SPEC_CPU_v0_2 §8.2

Generate condition and scoring manifests from the merged study config.

Usage:
    python scripts/01_build_manifests.py \\
      --config configs/study_config.yaml \\
      --out-dir manifests/

Outputs:
    manifests/conditions_primary.csv  -- primary-tier conditions
    manifests/conditions_gsm8k.csv    -- secondary-tier (GSM8K) conditions
    manifests/scoring_manifest.csv    -- one row per scoring-rule declaration

Exit codes per SPEC §11.1:
    0  -- success
    2  -- configuration validation failure
    1  -- generic failure

Determinism
-----------
Iteration order is fully sorted (benchmarks, models, subjects, prompts,
seeds, scoring_rules) before enumeration so that re-running with the
same configs produces byte-identical CSVs. The ``--fixed-timestamp``
flag pins ``created_at`` so tests can verify byte-level reproducibility.

Note
----
This script does NOT emit ``manifests/prompt_manifest.csv`` — that is the
output of ``scripts/03_validate_prompts.py`` (SPEC §8.4).
"""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime, timezone
from pathlib import Path

# Make the paper-root package importable without requiring an install.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gradience_study.config import (  # noqa: E402
    BenchmarkSpec,
    ConfigValidationError,
    MergedConfig,
    ModelSpec,
    PromptSpec,
    ScoringRuleSpec,
    compute_config_hash,
    load_config,
)


# Seeds used when ``benchmark.seed_facet == True``. Order matters for
# determinism: the script sorts seeds lexicographically by their string
# form before enumerating, so any fixed list yields a deterministic order
# regardless of the literal Python list order written here.
DEFAULT_SEEDS: tuple[str, ...] = ("42", "123", "2024")

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
    "task_type",
    "expected_num_items",
    "condition_status",
    "created_at",
    "config_hash",
]

SCORING_FIELDNAMES: list[str] = [
    "scoring_rule_id",
    "display_name",
    "applies_to_task_types",
    "normalization",
    "selection_rule",
    "parse_strategy",
]


# -- Logging -----------------------------------------------------------------


def _log(level: str, msg: str) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"[{ts}] [{level}] [01_build_manifests] {msg}", file=sys.stderr)


# -- Helpers -----------------------------------------------------------------


def _expected_num_items(benchmark: BenchmarkSpec) -> int:
    """Resolve the per-condition expected item count.

    For benchmarks with subjects, prefer ``expected_num_items_per_subject``
    if set, then fall back to ``expected_num_items``. For benchmarks
    without subjects, use ``expected_num_items``. If neither is populated,
    return 0 with a warning.
    """
    if benchmark.subjects:
        if benchmark.expected_num_items_per_subject is not None:
            return int(benchmark.expected_num_items_per_subject)
        if benchmark.expected_num_items is not None:
            return int(benchmark.expected_num_items)
    else:
        if benchmark.expected_num_items is not None:
            return int(benchmark.expected_num_items)
    _log(
        "WARNING",
        f"Benchmark '{benchmark.benchmark_id}': neither expected_num_items "
        f"nor expected_num_items_per_subject is populated; emitting 0.",
    )
    return 0


def _seeds_for(benchmark: BenchmarkSpec) -> tuple[str, ...]:
    """Return the seed_id strings used when enumerating ``benchmark``.

    Per SPEC §7.2: ``s{seed}`` for seed-faceted benchmarks; ``0shot`` for
    ``fewshot_k == 0``. Sorted lexicographically for deterministic order.
    """
    if benchmark.fewshot_k == 0:
        return ("0shot",)
    if benchmark.seed_facet:
        return tuple(sorted(f"s{s}" for s in DEFAULT_SEEDS))
    # Seed-facet disabled but fewshot_k > 0 — treat as a single canonical
    # seed slot. Use the first DEFAULT_SEEDS value for stability.
    return (f"s{DEFAULT_SEEDS[0]}",)


def _condition_id(
    benchmark_id: str,
    subject_id: str | None,
    model_id: str,
    prompt_id: str,
    seed_id: str,
    scoring_rule_id: str,
) -> str:
    """Construct a condition_id per SPEC §7.2.

    Subject is "" when the benchmark has no subjects; the literal
    ``__`` delimiter still surrounds it so the segment count is stable
    across all rows.
    """
    subj = subject_id if subject_id is not None else ""
    return (
        f"{benchmark_id}__{subj}__{model_id}__{prompt_id}__{seed_id}"
        f"__{scoring_rule_id}"
    )


def _select_models(merged: MergedConfig) -> list[ModelSpec]:
    """Select models eligible for the primary manifest.

    Primary models are always included. Extension (7B) models are added
    only if ``study_config.extension_7b_enabled`` is true. Sorted by
    ``model_id`` for determinism.
    """
    eligible_ids: set[str] = set()
    eligible: list[ModelSpec] = []
    for m in merged.models:
        if m.primary:
            if m.model_id not in eligible_ids:
                eligible.append(m)
                eligible_ids.add(m.model_id)
    if merged.study_config.extension_7b_enabled:
        for m in merged.models:
            if m.optional_extension and m.model_id not in eligible_ids:
                eligible.append(m)
                eligible_ids.add(m.model_id)
    return sorted(eligible, key=lambda m: m.model_id)


def _prompts_for_benchmark(
    merged: MergedConfig, benchmark_id: str
) -> list[PromptSpec]:
    return sorted(
        (p for p in merged.prompts if p.benchmark_id == benchmark_id),
        key=lambda p: p.prompt_id,
    )


# -- Row enumeration ---------------------------------------------------------


def _enumerate_conditions(
    merged: MergedConfig,
    config_hash_short: str,
    created_at: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Walk the configured measurement universe and emit condition rows.

    Returns:
        (primary_rows, secondary_rows). Each row is a dict keyed by
        ``CONDITION_FIELDNAMES``. Primary rows include the extension
        models when ``extension_7b_enabled``.
    """
    primary_rows: list[dict[str, object]] = []
    secondary_rows: list[dict[str, object]] = []

    models = _select_models(merged)
    benchmarks_sorted = sorted(merged.benchmarks, key=lambda b: b.benchmark_id)
    rules_by_id: dict[str, ScoringRuleSpec] = {
        s.scoring_rule_id: s for s in merged.scoring_rules
    }

    for benchmark in benchmarks_sorted:
        if benchmark.tier not in ("primary", "secondary"):
            _log(
                "WARNING",
                f"Benchmark '{benchmark.benchmark_id}': unknown tier "
                f"'{benchmark.tier}'; skipping.",
            )
            continue

        prompts = _prompts_for_benchmark(merged, benchmark.benchmark_id)
        if not prompts:
            _log(
                "WARNING",
                f"Benchmark '{benchmark.benchmark_id}': no prompts declared; "
                f"no conditions will be emitted.",
            )
            continue

        if benchmark.subjects:
            subjects: tuple[str | None, ...] = tuple(sorted(benchmark.subjects))
        else:
            subjects = (None,)

        seeds = _seeds_for(benchmark)
        n_items = _expected_num_items(benchmark)
        rule_ids_sorted = tuple(sorted(benchmark.scoring_rules))

        for subject in subjects:
            for model in models:
                for prompt in prompts:
                    for seed_id in seeds:
                        for rule_id in rule_ids_sorted:
                            if rule_id not in rules_by_id:
                                # Cross-ref check is the responsibility of
                                # 00_validate_config.py; here we simply do
                                # not silently drop the row but flag it.
                                _log(
                                    "WARNING",
                                    f"Benchmark '{benchmark.benchmark_id}': "
                                    f"references unknown scoring_rule "
                                    f"'{rule_id}'; row still emitted.",
                                )
                            row = _build_row(
                                benchmark=benchmark,
                                subject=subject,
                                model=model,
                                prompt=prompt,
                                seed_id=seed_id,
                                rule_id=rule_id,
                                config_hash_short=config_hash_short,
                                created_at=created_at,
                            )
                            if benchmark.tier == "primary":
                                primary_rows.append(row)
                            else:
                                secondary_rows.append(row)

    return primary_rows, secondary_rows


def _build_row(
    *,
    benchmark: BenchmarkSpec,
    subject: str | None,
    model: ModelSpec,
    prompt: PromptSpec,
    seed_id: str,
    rule_id: str,
    config_hash_short: str,
    created_at: str,
) -> dict[str, object]:
    cid = _condition_id(
        benchmark_id=benchmark.benchmark_id,
        subject_id=subject,
        model_id=model.model_id,
        prompt_id=prompt.prompt_id,
        seed_id=seed_id,
        scoring_rule_id=rule_id,
    )
    return {
        "condition_id": cid,
        "tier": benchmark.tier,
        "benchmark_id": benchmark.benchmark_id,
        "subject_id": subject if subject is not None else "",
        "model_id": model.model_id,
        "prompt_id": prompt.prompt_id,
        "seed_id": seed_id,
        "fewshot_k": int(benchmark.fewshot_k),
        "scoring_rule_id": rule_id,
        "task_type": benchmark.task_type,
        "expected_num_items": _expected_num_items(benchmark),
        "condition_status": "pending",
        "created_at": created_at,
        "config_hash": config_hash_short,
    }


def _build_scoring_rows(merged: MergedConfig) -> list[dict[str, object]]:
    """One row per declared scoring rule; sorted by scoring_rule_id."""
    rules = sorted(merged.scoring_rules, key=lambda s: s.scoring_rule_id)
    rows: list[dict[str, object]] = []
    for s in rules:
        rows.append(
            {
                "scoring_rule_id": s.scoring_rule_id,
                "display_name": s.display_name,
                "applies_to_task_types": ";".join(s.applies_to_task_types),
                "normalization": s.normalization or "",
                "selection_rule": s.selection_rule or "",
                "parse_strategy": s.parse_strategy or "",
            }
        )
    return rows


# -- IO ----------------------------------------------------------------------


def _write_csv(
    path: Path, rows: list[dict[str, object]], fieldnames: list[str]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=fieldnames, lineterminator="\n"
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


# -- Main --------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build condition and scoring manifests per SPEC_CPU_v0_2 §8.2"
    )
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Path to study_config.yaml",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        type=Path,
        help="Directory to write manifest CSVs into.",
    )
    parser.add_argument(
        "--fixed-timestamp",
        default=None,
        help=(
            "Override the created_at timestamp with a fixed ISO-8601 UTC "
            "value for reproducibility testing. If omitted, current UTC "
            "time is used."
        ),
    )
    return parser.parse_args(argv)


def _nominal_count(merged: MergedConfig, models: list[ModelSpec]) -> int:
    """Compute the script's predicted total row count across both manifests.

    Logged before enumeration as a sanity check vs the actual emitted count.
    """
    n_models = len(models)
    total = 0
    for b in merged.benchmarks:
        if b.tier not in ("primary", "secondary"):
            continue
        prompts_n = sum(1 for p in merged.prompts if p.benchmark_id == b.benchmark_id)
        if prompts_n == 0:
            continue
        n_subjects = len(b.subjects) if b.subjects else 1
        if b.fewshot_k == 0:
            n_seeds = 1
        elif b.seed_facet:
            n_seeds = len(DEFAULT_SEEDS)
        else:
            n_seeds = 1
        n_rules = len(b.scoring_rules)
        total += n_models * n_subjects * prompts_n * n_seeds * n_rules
    return total


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    # --- Load configs -----------------------------------------------------

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

    created_at = args.fixed_timestamp or datetime.now(timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )

    # --- Sanity-log the predicted nominal count ---------------------------

    models = _select_models(merged)
    nominal = _nominal_count(merged, models)
    _log(
        "INFO",
        f"Computed nominal condition count = {nominal} "
        f"(across primary + secondary tiers; "
        f"models in scope = {len(models)}, "
        f"benchmarks = {len(merged.benchmarks)}).",
    )

    # --- Enumerate --------------------------------------------------------

    try:
        primary_rows, secondary_rows = _enumerate_conditions(
            merged, config_hash_short, created_at
        )
    except Exception as e:  # pragma: no cover — defensive
        _log("ERROR", f"Failed to enumerate conditions: {e}")
        return 1

    # --- Emit -------------------------------------------------------------

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    primary_path = out_dir / "conditions_primary.csv"
    secondary_path = out_dir / "conditions_gsm8k.csv"
    scoring_path = out_dir / "scoring_manifest.csv"

    _write_csv(primary_path, primary_rows, CONDITION_FIELDNAMES)
    _write_csv(secondary_path, secondary_rows, CONDITION_FIELDNAMES)

    scoring_rows = _build_scoring_rows(merged)
    _write_csv(scoring_path, scoring_rows, SCORING_FIELDNAMES)

    _log(
        "INFO",
        f"Wrote {len(primary_rows)} primary row(s) to {primary_path}, "
        f"{len(secondary_rows)} secondary row(s) to {secondary_path}, "
        f"{len(scoring_rows)} scoring rule(s) to {scoring_path}. "
        f"(config_hash={config_hash_short})",
    )

    if (len(primary_rows) + len(secondary_rows)) != nominal:
        _log(
            "WARNING",
            f"Emitted row count "
            f"({len(primary_rows) + len(secondary_rows)}) "
            f"differs from nominal ({nominal}). Likely cause: a benchmark "
            f"has no prompts declared, or an unknown tier was skipped.",
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
