"""03_validate_prompts.py — SPEC_CPU_v0_2 §8.4

Validate prompt provenance, content hashes, and placeholder coverage.

Usage:
    python scripts/03_validate_prompts.py \\
      --config configs/study_config.yaml \\
      --out manifests/prompt_manifest.csv

Validation checks per SPEC §8.4:

  1. Every prompt file exists on disk at the resolved ``prompt_path``
     (relative to the paper root).
  2. SHA-256 of file content matches ``content_hash`` in prompts.yaml.
     Placeholder content_hash values (TODO, "", <sha256-locked-at-v1.1>)
     produce a warning rather than an error.
  3. Every prompt file contains all placeholders declared in its
     ``placeholders_used`` list, via literal string search for the
     pattern ``{{placeholder_name}}``. Missing placeholder → error.
  4. ``admissibility_status == "locked"`` for all prompts → warning if
     not. Becomes a hard error only if ``study_config.yaml`` sets
     ``require_locked_prompts: true``.
  5. P4 prompts (``source_type == "author_minimal_declared"``) should
     have a corresponding entry in
     ``preregistration/appendices/admissibility_sources_LOCKED.md`` →
     warning if not present (the appendix file may not exist pre-lock).

Exit codes per SPEC §11.1:
    0  -- all-pass (warnings permitted)
    2  -- any error (missing file, hash mismatch when content_hash is
          not a TODO placeholder, missing placeholder, or unlocked
          prompt when ``require_locked_prompts`` is true)
    1  -- generic failure
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from datetime import datetime, timezone
from pathlib import Path

# Make the paper-root package importable without requiring an install.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gradience_study.config import (  # noqa: E402
    ConfigValidationError,
    MergedConfig,
    PromptSpec,
    load_config,
)


HASH_PLACEHOLDER_VALUES: frozenset[str] = frozenset(
    {"TODO", "", "<commit-hash-locked-at-v1.1>", "<sha256-locked-at-v1.1>"}
)

ADMISSIBILITY_APPENDIX_REL: str = (
    "preregistration/appendices/admissibility_sources_LOCKED.md"
)

MANIFEST_FIELDNAMES: list[str] = [
    "benchmark_id",
    "prompt_id",
    "prompt_path",
    "declared_content_hash",
    "actual_content_hash",
    "source_type",
    "source_url_or_ref",
    "admissibility_status",
    "validation_status",
    "warnings",
]


# -- Logging -----------------------------------------------------------------


def _log(level: str, msg: str) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"[{ts}] [{level}] [03_validate_prompts] {msg}", file=sys.stderr)


# -- Helpers -----------------------------------------------------------------


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _placeholder_present(text: str, name: str) -> bool:
    """SPEC §8.4 #3: literal string search for ``{{<name>}}``."""
    return f"{{{{{name}}}}}" in text


def _read_admissibility_appendix(paper_root: Path) -> str | None:
    path = paper_root / ADMISSIBILITY_APPENDIX_REL
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8")


# -- Per-prompt validation ---------------------------------------------------


def _validate_one(
    prompt: PromptSpec,
    paper_root: Path,
    *,
    require_locked: bool,
    appendix_text: str | None,
) -> tuple[dict[str, object], bool, bool]:
    """Validate a single prompt; build the manifest row.

    Returns:
        (row, has_error, has_warning).
    """
    warnings: list[str] = []
    has_error = False

    prompt_path = paper_root / prompt.prompt_path
    declared = prompt.content_hash
    actual = ""

    # 1. File existence.
    if not prompt_path.exists():
        warnings.append(f"prompt file not found at {prompt.prompt_path}")
        has_error = True
    else:
        actual = _sha256_file(prompt_path)

        # 2. Content-hash check. Cast declared to str defensively: YAML
        #    may deserialize a malformed hash value as int/float, which
        #    would break both the placeholder comparison and slicing.
        declared_str = str(declared) if declared is not None else ""
        if declared_str in HASH_PLACEHOLDER_VALUES:
            warnings.append(
                f"content_hash is a placeholder "
                f"('{declared_str}'); actual={actual[:16]}..."
            )
        elif declared_str != actual:
            warnings.append(
                f"content hash mismatch "
                f"(declared={declared_str[:16]}..., actual={actual[:16]}...)"
            )
            has_error = True

        # 3. Placeholder coverage.
        try:
            text = prompt_path.read_text(encoding="utf-8")
        except Exception as e:
            warnings.append(f"could not read prompt file: {e}")
            has_error = True
            text = ""
        for ph in prompt.placeholders_used:
            if not _placeholder_present(text, ph):
                warnings.append(f"missing placeholder {{{{{ph}}}}}")
                has_error = True

    # 4. Admissibility status.
    if prompt.admissibility_status != "locked":
        msg = (
            f"admissibility_status='{prompt.admissibility_status}', "
            f"not 'locked'"
        )
        warnings.append(msg)
        if require_locked:
            has_error = True

    # 5. P4 / author_minimal_declared appendix presence.
    if prompt.source_type == "author_minimal_declared":
        if appendix_text is None:
            warnings.append(
                f"author_minimal_declared prompt requires entry in "
                f"{ADMISSIBILITY_APPENDIX_REL} (appendix file missing)"
            )
        else:
            # Heuristic match: the appendix mentions either the prompt_id
            # or the prompt_path.
            if (
                prompt.prompt_id not in appendix_text
                and prompt.prompt_path not in appendix_text
            ):
                warnings.append(
                    f"author_minimal_declared prompt not referenced in "
                    f"{ADMISSIBILITY_APPENDIX_REL}"
                )

    if has_error:
        validation_status = "fail"
    elif warnings:
        validation_status = "warn"
    else:
        validation_status = "pass"

    row: dict[str, object] = {
        "benchmark_id": prompt.benchmark_id,
        "prompt_id": prompt.prompt_id,
        "prompt_path": prompt.prompt_path,
        "declared_content_hash": declared,
        "actual_content_hash": actual,
        "source_type": prompt.source_type,
        "source_url_or_ref": prompt.source_url_or_ref,
        "admissibility_status": prompt.admissibility_status,
        "validation_status": validation_status,
        "warnings": ";".join(warnings),
    }
    return row, has_error, bool(warnings)


# -- Top-level walk ----------------------------------------------------------


def _validate_all(
    merged: MergedConfig, paper_root: Path
) -> tuple[list[dict[str, object]], int, int]:
    """Walk all prompts; return (rows, error_count, warning_count)."""
    require_locked = bool(
        merged.raw_dict.get("study_config", {}).get(
            "require_locked_prompts", False
        )
    )

    appendix_text = _read_admissibility_appendix(paper_root)

    rows: list[dict[str, object]] = []
    error_count = 0
    warning_count = 0

    prompts_sorted = sorted(
        merged.prompts, key=lambda p: (p.benchmark_id, p.prompt_id)
    )
    for prompt in prompts_sorted:
        row, has_error, has_warning = _validate_one(
            prompt,
            paper_root,
            require_locked=require_locked,
            appendix_text=appendix_text,
        )
        rows.append(row)
        if has_error:
            error_count += 1
        if has_warning:
            warning_count += 1

    return rows, error_count, warning_count


# -- IO ----------------------------------------------------------------------


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=MANIFEST_FIELDNAMES, lineterminator="\n"
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


# -- Main --------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate prompt provenance per SPEC_CPU_v0_2 §8.4"
    )
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help=(
            "Path to study_config.yaml. The paper root (used to resolve "
            "prompt_path entries) is derived as study_config.parent.parent."
        ),
    )
    parser.add_argument(
        "--out",
        required=True,
        type=Path,
        help="Output path for prompt_manifest.csv",
    )
    return parser.parse_args(argv)


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

    paper_root = args.config.resolve().parent.parent

    # --- Validate ---------------------------------------------------------

    rows, n_errors, n_warnings = _validate_all(merged, paper_root)

    # --- Emit manifest ----------------------------------------------------

    _write_csv(args.out, rows)

    # --- Log warnings and errors ------------------------------------------

    for row in rows:
        if not row["warnings"]:
            continue
        for warn in str(row["warnings"]).split(";"):
            level = "ERROR" if row["validation_status"] == "fail" else "WARNING"
            _log(
                level,
                f"{row['benchmark_id']}/{row['prompt_id']}: {warn}",
            )

    _log(
        "INFO",
        f"Wrote {len(rows)} prompt manifest row(s) to {args.out} "
        f"(errors={n_errors}, warnings={n_warnings}).",
    )

    if n_errors > 0:
        _log(
            "ERROR",
            f"Prompt validation: FAIL ({n_errors} prompt(s) with errors).",
        )
        return 2

    _log("INFO", "Prompt validation: PASS (warnings permitted).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
