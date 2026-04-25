"""00_validate_config.py — SPEC_CPU_v0_2 §8.1

Validate all six config files; compute and record config hash.

Usage:
    python scripts/00_validate_config.py \\
      --config configs/study_config.yaml \\
      --out reports/config_validation.json

Exit codes per SPEC §11.1:
    0  — validation passed
    2  — configuration validation failure
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# Make the paper-root package importable without requiring an install.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gradience_study.config import (  # noqa: E402
    ConfigValidationError,
    MergedConfig,
    compute_config_hash,
    load_config,
)


EXPECTED_PROMPT_IDS: frozenset[str] = frozenset(
    {"P1_original", "P2_lm_eval", "P3_helm_or_published", "P4_minimal_sourced"}
)

HASH_PLACEHOLDER_VALUES: frozenset[str] = frozenset(
    {"TODO", "", "<commit-hash-locked-at-v1.1>", "<sha256-locked-at-v1.1>"}
)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _compute_nominal_conditions(merged: MergedConfig) -> int:
    """Compute nominal condition count for the primary tier.

    For each primary benchmark: n_primary_models × n_subjects × n_prompts
      × (3 if seed_facet else 1) × n_scoring_rules.
    MMLU-panel subjects expand within the benchmark.
    """
    n_primary_models = sum(1 for m in merged.models if m.primary)
    total = 0
    for b in merged.benchmarks:
        if b.tier != "primary":
            continue
        n_subjects = len(b.subjects) if b.subjects else 1
        n_seeds = 3 if b.seed_facet else 1
        n_rules = len(b.scoring_rules)
        n_prompts = len(EXPECTED_PROMPT_IDS)
        total += n_primary_models * n_subjects * n_prompts * n_seeds * n_rules
    return total


def validate(config_path: Path) -> dict:
    """Run all validation checks per SPEC §4.5; return a validation report."""
    errors: list[str] = []
    warnings: list[str] = []

    try:
        merged = load_config(config_path)
    except FileNotFoundError as e:
        errors.append(f"Config file not found: {e}")
        return _build_report(config_path, errors, warnings, None, None,
                             primary_model_count=0, primary_benchmark_count=0,
                             total_prompts=0, total_nominal_conditions=0)
    except ConfigValidationError as e:
        errors.append(f"Config validation error: {e}")
        return _build_report(config_path, errors, warnings, None, None,
                             primary_model_count=0, primary_benchmark_count=0,
                             total_prompts=0, total_nominal_conditions=0)
    except KeyError as e:
        errors.append(f"Missing required field: {e}")
        return _build_report(config_path, errors, warnings, None, None,
                             primary_model_count=0, primary_benchmark_count=0,
                             total_prompts=0, total_nominal_conditions=0)

    # --- Primary-tier structural checks -------------------------------------

    primary_models = [m for m in merged.models if m.primary]
    if len(primary_models) != 3:
        errors.append(
            f"Expected 3 primary models, found {len(primary_models)}. "
            f"Any deviation must be declared explicitly in deviations.md."
        )

    primary_benchmarks = [b for b in merged.benchmarks if b.tier == "primary"]
    if len(primary_benchmarks) != 5:
        errors.append(
            f"Expected 5 primary benchmarks, found {len(primary_benchmarks)}."
        )

    # --- Prompt coverage per benchmark --------------------------------------

    prompts_by_benchmark: dict[str, list] = {}
    for p in merged.prompts:
        prompts_by_benchmark.setdefault(p.benchmark_id, []).append(p)

    for b in merged.benchmarks:
        entries = prompts_by_benchmark.get(b.benchmark_id, [])
        ids_present = {p.prompt_id for p in entries}
        missing = EXPECTED_PROMPT_IDS - ids_present
        if missing:
            errors.append(
                f"Benchmark '{b.benchmark_id}': missing prompt IDs "
                f"{sorted(missing)}"
            )
        extras = ids_present - EXPECTED_PROMPT_IDS
        if extras:
            warnings.append(
                f"Benchmark '{b.benchmark_id}': unexpected prompt IDs "
                f"{sorted(extras)}"
            )

    # --- Prompt file existence and content-hash verification ----------------

    paper_root = config_path.parent.parent
    for p in merged.prompts:
        prompt_file = paper_root / p.prompt_path
        if not prompt_file.exists():
            errors.append(f"Prompt file not found: {p.prompt_path}")
            continue
        actual_hash = _sha256_file(prompt_file)
        if p.content_hash in HASH_PLACEHOLDER_VALUES:
            warnings.append(
                f"Prompt {p.benchmark_id}/{p.prompt_id}: content_hash is a "
                f"placeholder ('{p.content_hash}'); actual hash is "
                f"{actual_hash[:16]}...  Update at v1.1 lock."
            )
        elif p.content_hash != actual_hash:
            errors.append(
                f"Prompt {p.benchmark_id}/{p.prompt_id}: content hash "
                f"mismatch (declared {p.content_hash[:16]}..., "
                f"actual {actual_hash[:16]}...)"
            )

    # --- Scoring rule cross-reference ---------------------------------------

    declared_rule_ids = {s.scoring_rule_id for s in merged.scoring_rules}
    for b in merged.benchmarks:
        for rule_id in b.scoring_rules:
            if rule_id not in declared_rule_ids:
                errors.append(
                    f"Benchmark '{b.benchmark_id}': references unknown "
                    f"scoring rule '{rule_id}' (not in scoring_rules.yaml)"
                )

    # --- Pre-lock state warnings --------------------------------------------

    for p in merged.prompts:
        if p.admissibility_status != "locked":
            warnings.append(
                f"Prompt {p.benchmark_id}/{p.prompt_id}: admissibility_status "
                f"is '{p.admissibility_status}', not 'locked'. "
                f"Must be 'locked' before data collection."
            )
    for m in merged.models:
        if m.hf_revision in HASH_PLACEHOLDER_VALUES:
            warnings.append(
                f"Model {m.model_id}: hf_revision is a placeholder "
                f"('{m.hf_revision}'). Pin at v1.1 lock."
            )
    for b in merged.benchmarks:
        if b.dataset_version_hash in HASH_PLACEHOLDER_VALUES:
            warnings.append(
                f"Benchmark {b.benchmark_id}: dataset_version_hash is a "
                f"placeholder ('{b.dataset_version_hash}'). Pin at v1.1 lock."
            )

    # --- Config hash --------------------------------------------------------

    full_hash, short_hash = compute_config_hash(merged.raw_dict)
    nominal_conditions = _compute_nominal_conditions(merged)

    return _build_report(
        config_path,
        errors,
        warnings,
        full_hash,
        short_hash,
        primary_model_count=len(primary_models),
        primary_benchmark_count=len(primary_benchmarks),
        total_prompts=len(merged.prompts),
        total_nominal_conditions=nominal_conditions,
    )


def _build_report(
    config_path: Path,
    errors: list[str],
    warnings: list[str],
    full_hash: str | None,
    short_hash: str | None,
    primary_model_count: int,
    primary_benchmark_count: int,
    total_prompts: int,
    total_nominal_conditions: int,
) -> dict:
    return {
        "config_path": str(config_path),
        "config_hash_full": full_hash,
        "config_hash": short_hash,
        "validation_status": "pass" if not errors else "fail",
        "validation_errors": errors,
        "validation_warnings": warnings,
        "primary_model_count": primary_model_count,
        "primary_benchmark_count": primary_benchmark_count,
        "total_prompts": total_prompts,
        "total_nominal_conditions": total_nominal_conditions,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def _log(level: str, msg: str) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"[{ts}] [{level}] [00_validate_config] {msg}", file=sys.stderr)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate study configs per SPEC_CPU_v0_2 §4.5"
    )
    parser.add_argument("--config", required=True, type=Path,
                        help="Path to study_config.yaml")
    parser.add_argument("--out", required=True, type=Path,
                        help="Path to output config_validation.json")
    args = parser.parse_args()

    report = validate(args.config)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True)

    for warning in report["validation_warnings"]:
        _log("WARNING", warning)

    if report["validation_status"] == "pass":
        _log("INFO",
             f"Config validation: PASS (config_hash={report['config_hash']}, "
             f"{report['total_nominal_conditions']} nominal conditions)")
        return 0
    else:
        _log("ERROR", f"Config validation: FAIL ({len(report['validation_errors'])} errors)")
        for err in report["validation_errors"]:
            _log("ERROR", err)
        return 2


if __name__ == "__main__":
    sys.exit(main())
