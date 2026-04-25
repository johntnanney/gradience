"""04_normalize_outputs.py — SPEC_CPU_v0_2 §8.5

Ingest raw GPU-side outputs; validate against JSON Schemas; emit a
normalized item-level parquet conforming to SPEC §5.5.

Usage:
    python scripts/04_normalize_outputs.py \\
      --conditions manifests/conditions_primary.csv \\
      --raw-dir runs/raw/ \\
      --schemas-dir schemas/ \\
      --out runs/normalized/item_level_primary.parquet

Optional:
    --config configs/study_config.yaml
        If provided, columns sourced from the configs (model_family,
        model_type, parameter_count_b, prompt_source_type) are populated
        by joining on model_id and prompt_id. If omitted, those columns
        are emitted as null. The required join keys (tier, benchmark_id,
        model_id, prompt_id, seed_id, fewshot_k, scoring_rule_id, etc.)
        are read from the condition manifest, which always carries them.

Exit codes per SPEC §11.1:
    0  — success; every complete condition normalized cleanly
    1  — generic failure (implementation error)
    3  — data contract violation: any of
            - run_metadata.json missing
            - schema validation failure (run_metadata, item_outputs, item_scores)
            - run_id / condition_id mismatch with the manifest
            - partial run completion (num_items_completed != num_items_expected)
            - condition-count mismatch (manifest expected_num_items != JSONL row count)

No silent failures: any condition that fails normalization is logged
loudly to stderr and contributes no rows to the output parquet, and
the script's exit code reflects the failure even if other conditions
were normalized cleanly.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Make the paper-root package importable without requiring an install.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import jsonschema  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from gradience_study.config import (  # noqa: E402
    ConditionCountMismatch,
    MergedConfig,
    PartialRunCompletion,
    SchemaValidationError,
    load_config,
)


# -- Logging -----------------------------------------------------------------


def _log(level: str, msg: str) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"[{ts}] [{level}] [04_normalize_outputs] {msg}", file=sys.stderr)


# -- Item-level Arrow schema (per SPEC §5.5) --------------------------------


ITEM_LEVEL_SCHEMA = pa.schema(
    [
        pa.field("condition_id", pa.string(), nullable=False),
        pa.field("tier", pa.string(), nullable=False),
        pa.field("benchmark_id", pa.string(), nullable=False),
        pa.field("subject_id", pa.string(), nullable=True),
        pa.field("model_id", pa.string(), nullable=False),
        pa.field("model_family", pa.string(), nullable=True),
        pa.field("model_type", pa.string(), nullable=True),
        pa.field("parameter_count_b", pa.float32(), nullable=True),
        pa.field("prompt_id", pa.string(), nullable=False),
        pa.field("prompt_source_type", pa.string(), nullable=True),
        pa.field("seed_id", pa.string(), nullable=True),
        pa.field("fewshot_k", pa.int32(), nullable=False),
        pa.field("scoring_rule_id", pa.string(), nullable=False),
        pa.field("task_type", pa.string(), nullable=False),
        pa.field("item_id", pa.string(), nullable=False),
        pa.field("gold_answer", pa.string(), nullable=False),
        pa.field("predicted_answer", pa.string(), nullable=True),
        pa.field("is_correct", pa.bool_(), nullable=False),
        pa.field("parse_status", pa.string(), nullable=True),
        pa.field("parseable", pa.bool_(), nullable=True),
        pa.field("choice_count", pa.int32(), nullable=True),
        pa.field("prompt_text_hash", pa.string(), nullable=False),
        pa.field("rendered_prompt_hash", pa.string(), nullable=False),
        pa.field("run_id", pa.string(), nullable=False),
        pa.field("created_at", pa.timestamp("us", tz="UTC"), nullable=False),
        # LL-only
        pa.field("selected_answer", pa.string(), nullable=True),
        pa.field("ll_margin_top2", pa.float32(), nullable=True),
        pa.field("gold_choice_score", pa.float32(), nullable=True),
        pa.field("selected_choice_score", pa.float32(), nullable=True),
        # G&P-only
        pa.field("raw_generation", pa.string(), nullable=True),
        pa.field("parsed_answer", pa.string(), nullable=True),
        pa.field("generation_length_tokens", pa.int32(), nullable=True),
    ]
)


# -- Schema loading ----------------------------------------------------------


def _load_schema(schemas_dir: Path, name: str) -> dict[str, Any]:
    path = schemas_dir / name
    if not path.exists():
        raise FileNotFoundError(f"JSON Schema not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@dataclass(frozen=True)
class Schemas:
    run_metadata: dict[str, Any]
    item_outputs: dict[str, Any]
    item_scores: dict[str, Any]


def load_schemas(schemas_dir: Path) -> Schemas:
    return Schemas(
        run_metadata=_load_schema(schemas_dir, "run_metadata.schema.json"),
        item_outputs=_load_schema(schemas_dir, "item_outputs.schema.json"),
        item_scores=_load_schema(schemas_dir, "item_scores.schema.json"),
    )


# -- Helpers -----------------------------------------------------------------


def _parse_str_field(v: Any) -> str | None:
    """Treat empty strings as null for nullable string columns."""
    if v is None:
        return None
    s = str(v)
    if s == "":
        return None
    return s


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.rstrip("\n")
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise SchemaValidationError(
                    f"{path}: line {line_no}: invalid JSON ({e})"
                ) from e
            rows.append(obj)
    return rows


def _validate_jsonl_rows(
    rows: list[dict[str, Any]],
    schema: dict[str, Any],
    source_label: str,
) -> None:
    """Validate every row in a JSONL file against the schema.

    Raises SchemaValidationError on the first invalid row, with enough
    context to locate the bad row.
    """
    validator = jsonschema.Draft7Validator(schema)
    for i, row in enumerate(rows, start=1):
        errors = sorted(validator.iter_errors(row), key=lambda e: e.path)
        if errors:
            err = errors[0]
            path = "/".join(str(p) for p in err.absolute_path) or "<root>"
            raise SchemaValidationError(
                f"{source_label}: row {i}: schema violation at '{path}': {err.message}"
            )


def _validate_run_metadata(
    metadata: dict[str, Any], schema: dict[str, Any], source_label: str
) -> None:
    validator = jsonschema.Draft7Validator(schema)
    errors = sorted(validator.iter_errors(metadata), key=lambda e: e.path)
    if errors:
        err = errors[0]
        path = "/".join(str(p) for p in err.absolute_path) or "<root>"
        raise SchemaValidationError(
            f"{source_label}: schema violation at '{path}': {err.message}"
        )


# -- Condition manifest reader ----------------------------------------------


def _read_conditions_manifest(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Condition manifest not found: {path}")
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


# -- Per-condition normalization --------------------------------------------


@dataclass
class NormalizationResult:
    """Outcome of normalizing one condition row."""

    condition_id: str
    status: str  # "ok" | "skipped" | "failed"
    reason: str = ""
    rows: list[dict[str, Any]] | None = None


def _scoring_files_for_run(run_dir: Path) -> tuple[Path | None, Path | None]:
    """Return (item_outputs_path, item_scores_path) — either or both may be None."""
    out = run_dir / "item_outputs.jsonl"
    scores = run_dir / "item_scores.jsonl"
    return (out if out.exists() else None,
            scores if scores.exists() else None)


def _build_lookup_maps(
    merged: MergedConfig | None,
) -> tuple[dict[str, Any], dict[tuple[str, str], Any]]:
    """Return (model_id → ModelSpec, (benchmark_id, prompt_id) → PromptSpec)."""
    if merged is None:
        return {}, {}
    models = {m.model_id: m for m in merged.models}
    prompts = {(p.benchmark_id, p.prompt_id): p for p in merged.prompts}
    return models, prompts


def _normalize_gp_row(
    row: dict[str, Any],
    condition: dict[str, str],
    model_lookup: dict[str, Any],
    prompt_lookup: dict[tuple[str, str], Any],
    created_at: datetime,
) -> dict[str, Any]:
    """Build one item-level row from a G&P raw row."""
    model_spec = model_lookup.get(condition["model_id"])
    prompt_spec = prompt_lookup.get(
        (condition["benchmark_id"], condition["prompt_id"])
    )
    parse_status = row["parse_status"]
    parsed = row.get("parsed_answer")
    raw_generation = row["raw_generation"]
    # Token-length proxy: whitespace word count of raw generation.
    gen_len = len(raw_generation.split()) if raw_generation else 0

    return {
        "condition_id": condition["condition_id"],
        "tier": condition["tier"],
        "benchmark_id": condition["benchmark_id"],
        "subject_id": _parse_str_field(condition.get("subject_id")),
        "model_id": condition["model_id"],
        "model_family": getattr(model_spec, "model_family", None) if model_spec else None,
        "model_type": getattr(model_spec, "model_type", None) if model_spec else None,
        "parameter_count_b": (
            float(model_spec.parameter_count_b) if model_spec is not None else None
        ),
        "prompt_id": condition["prompt_id"],
        "prompt_source_type": (
            getattr(prompt_spec, "source_type", None) if prompt_spec else None
        ),
        "seed_id": _parse_str_field(condition.get("seed_id")),
        "fewshot_k": int(condition["fewshot_k"]),
        "scoring_rule_id": condition["scoring_rule_id"],
        "task_type": condition["task_type"],
        "item_id": row["item_id"],
        "gold_answer": row["gold_answer"],
        "predicted_answer": parsed,
        "is_correct": bool(row["is_correct"]),
        "parse_status": parse_status,
        "parseable": parse_status == "parsed",
        "choice_count": None,  # not derivable from G&P without external context
        "prompt_text_hash": row["prompt_text_hash"],
        "rendered_prompt_hash": row["rendered_prompt_hash"],
        "run_id": condition["condition_id"],
        "created_at": created_at,
        # LL-only — null for G&P
        "selected_answer": None,
        "ll_margin_top2": None,
        "gold_choice_score": None,
        "selected_choice_score": None,
        # G&P-only
        "raw_generation": raw_generation,
        "parsed_answer": parsed,
        "generation_length_tokens": gen_len,
    }


def _normalize_ll_row(
    row: dict[str, Any],
    condition: dict[str, str],
    model_lookup: dict[str, Any],
    prompt_lookup: dict[tuple[str, str], Any],
    created_at: datetime,
) -> dict[str, Any]:
    """Build one item-level row from an LL raw row."""
    model_spec = model_lookup.get(condition["model_id"])
    prompt_spec = prompt_lookup.get(
        (condition["benchmark_id"], condition["prompt_id"])
    )
    choice_scores: dict[str, float] = row["choice_scores"]
    # Compute ll_margin_top2 = top-1 - top-2 (per SPEC §5.5).
    sorted_scores = sorted(choice_scores.values(), reverse=True)
    if len(sorted_scores) >= 2:
        ll_margin = float(sorted_scores[0] - sorted_scores[1])
    else:
        ll_margin = 0.0

    selected = row["selected_answer"]
    gold = row["gold_answer"]
    gold_score = (
        float(choice_scores[gold]) if gold in choice_scores else None
    )
    selected_score = (
        float(choice_scores[selected]) if selected in choice_scores else None
    )

    return {
        "condition_id": condition["condition_id"],
        "tier": condition["tier"],
        "benchmark_id": condition["benchmark_id"],
        "subject_id": _parse_str_field(condition.get("subject_id")),
        "model_id": condition["model_id"],
        "model_family": getattr(model_spec, "model_family", None) if model_spec else None,
        "model_type": getattr(model_spec, "model_type", None) if model_spec else None,
        "parameter_count_b": (
            float(model_spec.parameter_count_b) if model_spec is not None else None
        ),
        "prompt_id": condition["prompt_id"],
        "prompt_source_type": (
            getattr(prompt_spec, "source_type", None) if prompt_spec else None
        ),
        "seed_id": _parse_str_field(condition.get("seed_id")),
        "fewshot_k": int(condition["fewshot_k"]),
        "scoring_rule_id": condition["scoring_rule_id"],
        "task_type": condition["task_type"],
        "item_id": row["item_id"],
        "gold_answer": gold,
        "predicted_answer": selected,
        "is_correct": bool(row["is_correct"]),
        "parse_status": None,
        "parseable": None,
        "choice_count": int(len(row["choices"])),
        "prompt_text_hash": row["prompt_text_hash"],
        "rendered_prompt_hash": row["rendered_prompt_hash"],
        "run_id": condition["condition_id"],
        "created_at": created_at,
        # LL-only
        "selected_answer": selected,
        "ll_margin_top2": ll_margin,
        "gold_choice_score": gold_score,
        "selected_choice_score": selected_score,
        # G&P-only — null for LL
        "raw_generation": None,
        "parsed_answer": None,
        "generation_length_tokens": None,
    }


def _normalize_condition(
    condition: dict[str, str],
    raw_dir: Path,
    schemas: Schemas,
    model_lookup: dict[str, Any],
    prompt_lookup: dict[tuple[str, str], Any],
    created_at: datetime,
) -> NormalizationResult:
    cond_id = condition["condition_id"]
    status = (condition.get("condition_status") or "").strip()

    if status != "complete":
        # SPEC §8.5 step 1: only process complete conditions.
        return NormalizationResult(
            condition_id=cond_id,
            status="skipped",
            reason=f"condition_status={status!r} (only 'complete' is normalized)",
        )

    run_dir = raw_dir / cond_id
    metadata_path = run_dir / "run_metadata.json"
    if not metadata_path.exists():
        raise SchemaValidationError(
            f"Condition '{cond_id}': run_metadata.json missing at {metadata_path}"
        )

    with open(metadata_path, "r", encoding="utf-8") as f:
        try:
            metadata = json.load(f)
        except json.JSONDecodeError as e:
            raise SchemaValidationError(
                f"Condition '{cond_id}': invalid JSON in {metadata_path}: {e}"
            ) from e

    # Step 2: schema-validate run_metadata.json.
    _validate_run_metadata(
        metadata, schemas.run_metadata, source_label=str(metadata_path)
    )

    # Step 3: condition_id consistency.
    if metadata.get("condition_id") != cond_id:
        raise SchemaValidationError(
            f"Condition '{cond_id}': run_metadata.condition_id="
            f"{metadata.get('condition_id')!r} disagrees with manifest."
        )
    if metadata.get("run_id") != cond_id:
        raise SchemaValidationError(
            f"Condition '{cond_id}': run_metadata.run_id="
            f"{metadata.get('run_id')!r} disagrees with condition_id (SPEC §5.2)."
        )

    # Step 4: partial-completion check.
    n_expected_md = int(metadata["num_items_expected"])
    n_completed_md = int(metadata["num_items_completed"])
    if n_completed_md != n_expected_md:
        raise PartialRunCompletion(
            f"Condition '{cond_id}': run_metadata reports "
            f"num_items_completed={n_completed_md} but "
            f"num_items_expected={n_expected_md}. Partial runs "
            f"are not normalized (SPEC §8.5)."
        )

    # Step 5: manifest expected_num_items consistency.
    n_expected_manifest = int(condition["expected_num_items"])
    if n_expected_md != n_expected_manifest:
        raise ConditionCountMismatch(
            f"Condition '{cond_id}': manifest expected_num_items="
            f"{n_expected_manifest} disagrees with run_metadata "
            f"num_items_expected={n_expected_md}."
        )

    # Step 6: locate the JSONL file for the scoring rule type.
    out_path, scores_path = _scoring_files_for_run(run_dir)
    if out_path is not None and scores_path is not None:
        raise SchemaValidationError(
            f"Condition '{cond_id}': both item_outputs.jsonl and "
            f"item_scores.jsonl are present; a single condition is one "
            f"scoring rule (SPEC §5.1)."
        )
    if out_path is None and scores_path is None:
        raise SchemaValidationError(
            f"Condition '{cond_id}': neither item_outputs.jsonl nor "
            f"item_scores.jsonl found in {run_dir}."
        )

    is_ll = scores_path is not None
    jsonl_path = scores_path if is_ll else out_path
    assert jsonl_path is not None
    raw_rows = _read_jsonl(jsonl_path)

    # Step 7: row count must match expected.
    if len(raw_rows) != n_expected_manifest:
        raise ConditionCountMismatch(
            f"Condition '{cond_id}': {jsonl_path.name} has {len(raw_rows)} "
            f"rows but manifest expected_num_items={n_expected_manifest}."
        )

    # Step 8: schema-validate every row.
    schema = schemas.item_scores if is_ll else schemas.item_outputs
    _validate_jsonl_rows(raw_rows, schema, source_label=str(jsonl_path))

    # Step 9: condition_id must match on every row.
    for i, r in enumerate(raw_rows, start=1):
        if r.get("condition_id") != cond_id:
            raise SchemaValidationError(
                f"Condition '{cond_id}': {jsonl_path.name} row {i} "
                f"condition_id={r.get('condition_id')!r} disagrees with manifest."
            )

    # Step 10: build normalized rows.
    if is_ll:
        normed = [
            _normalize_ll_row(r, condition, model_lookup, prompt_lookup, created_at)
            for r in raw_rows
        ]
    else:
        normed = [
            _normalize_gp_row(r, condition, model_lookup, prompt_lookup, created_at)
            for r in raw_rows
        ]

    return NormalizationResult(
        condition_id=cond_id, status="ok", rows=normed
    )


# -- Output writer -----------------------------------------------------------


def _write_parquet(out_path: Path, rows: list[dict[str, Any]]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        # Emit an empty parquet with the correct schema so downstream
        # consumers can still open the file.
        empty_table = pa.Table.from_pylist([], schema=ITEM_LEVEL_SCHEMA)
        pq.write_table(empty_table, out_path)
        return
    table = pa.Table.from_pylist(rows, schema=ITEM_LEVEL_SCHEMA)
    pq.write_table(table, out_path)


# -- Main entry point --------------------------------------------------------


def run(
    conditions_path: Path,
    raw_dir: Path,
    schemas_dir: Path,
    out_path: Path,
    config_path: Path | None = None,
) -> int:
    schemas = load_schemas(schemas_dir)

    merged: MergedConfig | None = None
    if config_path is not None:
        try:
            merged = load_config(config_path)
        except Exception as e:  # pragma: no cover — config loader has its own tests
            _log("WARNING", f"Failed to load config from {config_path}: {e}; "
                 f"continuing without config-derived enrichment.")
            merged = None

    model_lookup, prompt_lookup = _build_lookup_maps(merged)

    conditions = _read_conditions_manifest(conditions_path)
    _log("INFO", f"Loaded {len(conditions)} condition rows from {conditions_path}")

    # Use a single created_at for the run for deterministic output.
    created_at = datetime.now(timezone.utc).replace(microsecond=0)

    normalized_rows: list[dict[str, Any]] = []
    n_complete = 0
    n_ok = 0
    n_failed = 0
    n_skipped = 0
    failed_ids: list[str] = []

    for condition in conditions:
        cond_id = condition.get("condition_id", "<unknown>")
        status = (condition.get("condition_status") or "").strip()
        if status == "complete":
            n_complete += 1
        try:
            result = _normalize_condition(
                condition, raw_dir, schemas,
                model_lookup, prompt_lookup, created_at,
            )
        except (SchemaValidationError, PartialRunCompletion,
                ConditionCountMismatch) as e:
            _log("ERROR", f"Condition '{cond_id}': {type(e).__name__}: {e}")
            n_failed += 1
            failed_ids.append(cond_id)
            continue
        except Exception as e:  # noqa: BLE001
            _log("CRITICAL",
                 f"Condition '{cond_id}': unexpected error during "
                 f"normalization: {type(e).__name__}: {e}")
            return 1

        if result.status == "ok":
            assert result.rows is not None
            normalized_rows.extend(result.rows)
            n_ok += 1
        elif result.status == "skipped":
            n_skipped += 1
            _log("INFO", f"Condition '{cond_id}': skipped ({result.reason}).")

    # Write normalized parquet (always — partial completion is informative).
    _write_parquet(out_path, normalized_rows)
    _log("INFO",
         f"Wrote {len(normalized_rows)} item rows to {out_path} "
         f"(complete={n_complete}, normalized_ok={n_ok}, "
         f"failed={n_failed}, skipped={n_skipped}).")

    if n_failed > 0:
        _log("ERROR",
             f"{n_failed} condition(s) failed normalization: "
             f"{failed_ids[:10]}{'...' if len(failed_ids) > 10 else ''}")
        return 3

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Normalize raw GPU-side outputs to item-level parquet "
                    "(SPEC_CPU_v0_2 §8.5)."
    )
    parser.add_argument("--conditions", required=True, type=Path,
                        help="Path to condition manifest CSV.")
    parser.add_argument("--raw-dir", required=True, type=Path,
                        help="Directory containing per-run output subdirectories.")
    parser.add_argument("--schemas-dir", required=True, type=Path,
                        help="Directory containing JSON Schema files.")
    parser.add_argument("--out", required=True, type=Path,
                        help="Output item-level parquet path.")
    parser.add_argument("--config", type=Path, default=None,
                        help=("Optional study_config.yaml; enables "
                              "config-derived column enrichment."))
    args = parser.parse_args()

    try:
        return run(args.conditions, args.raw_dir, args.schemas_dir,
                   args.out, args.config)
    except FileNotFoundError as e:
        _log("ERROR", f"Input not found: {e}")
        return 3
    except Exception as e:  # noqa: BLE001
        _log("CRITICAL", f"Unhandled exception: {type(e).__name__}: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
