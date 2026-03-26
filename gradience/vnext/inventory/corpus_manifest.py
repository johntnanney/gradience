"""Corpus manifest schema (v1).

Records one validated corpus entry that links QA artifacts, pair reports,
and one neighborhood report for a single inventory/run.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

from gradience.exceptions import QASchemaError

SCHEMA_ID = "gradience.corpus_manifest/v1"


def _require_non_empty_str(value: Any, field_name: str) -> str:
    if not isinstance(value, str):
        raise QASchemaError(f"Field '{field_name}' must be str, got {type(value).__name__}")
    stripped = value.strip()
    if not stripped:
        raise QASchemaError(f"Field '{field_name}' must be non-empty")
    return stripped


def _require_list_of_non_empty_str(value: Any, field_name: str, *, min_len: int = 0) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise QASchemaError(f"Field '{field_name}' must be a list of strings")
    items: list[str] = []
    for i, item in enumerate(value):
        if not isinstance(item, str):
            raise QASchemaError(f"Field '{field_name}[{i}]' must be str, got {type(item).__name__}")
        stripped = item.strip()
        if not stripped:
            raise QASchemaError(f"Field '{field_name}[{i}]' must be non-empty")
        items.append(stripped)
    if len(items) < min_len:
        raise QASchemaError(f"Field '{field_name}' must contain at least {min_len} entries")
    if len(set(items)) != len(items):
        raise QASchemaError(f"Field '{field_name}' must not contain duplicates")
    return tuple(items)


def _require_iso_date_or_datetime(value: Any, field_name: str) -> str:
    raw = _require_non_empty_str(value, field_name)
    parsed = raw.replace("Z", "+00:00")
    try:
        datetime.fromisoformat(parsed)
        return raw
    except ValueError:
        pass
    try:
        date.fromisoformat(raw)
        return raw
    except ValueError as exc:
        raise QASchemaError(
            f"Field '{field_name}' must be ISO date/time (e.g., '2026-03-17' or '2026-03-17T12:34:56Z')"
        ) from exc


@dataclass(frozen=True)
class CorpusManifest:
    run_id: str
    date: str
    base_model: str
    adapter_names: tuple[str, ...]
    qa_artifact_paths: tuple[str, ...]
    pair_report_paths: tuple[str, ...]
    neighborhood_report_path: str
    downstream_outcome_paths: tuple[str, ...] = field(default_factory=tuple)
    notes: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA_ID,
            "run_id": self.run_id,
            "date": self.date,
            "base_model": self.base_model,
            "adapter_names": list(self.adapter_names),
            "qa_artifact_paths": list(self.qa_artifact_paths),
            "pair_report_paths": list(self.pair_report_paths),
            "neighborhood_report_path": self.neighborhood_report_path,
            "downstream_outcome_paths": list(self.downstream_outcome_paths),
            "notes": list(self.notes),
        }

    def to_json(self, path: Path | str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
            f.write("\n")

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CorpusManifest:
        if "schema" not in d:
            raise QASchemaError("Missing required field: schema")
        if d["schema"] != SCHEMA_ID:
            raise QASchemaError(f"Expected schema '{SCHEMA_ID}', got '{d['schema']}'")

        required = (
            "run_id",
            "date",
            "base_model",
            "adapter_names",
            "qa_artifact_paths",
            "pair_report_paths",
            "neighborhood_report_path",
        )
        for field_name in required:
            if field_name not in d:
                raise QASchemaError(f"Missing required field: {field_name}")

        run_id = _require_non_empty_str(d["run_id"], "run_id")
        iso_date = _require_iso_date_or_datetime(d["date"], "date")
        base_model = _require_non_empty_str(d["base_model"], "base_model")
        adapter_names = _require_list_of_non_empty_str(d["adapter_names"], "adapter_names", min_len=1)
        qa_artifact_paths = _require_list_of_non_empty_str(d["qa_artifact_paths"], "qa_artifact_paths", min_len=1)
        pair_report_paths = _require_list_of_non_empty_str(d["pair_report_paths"], "pair_report_paths", min_len=1)
        neighborhood_report_path = _require_non_empty_str(d["neighborhood_report_path"], "neighborhood_report_path")

        downstream_outcome_paths = _require_list_of_non_empty_str(
            d.get("downstream_outcome_paths", []),
            "downstream_outcome_paths",
        )
        notes = _require_list_of_non_empty_str(
            d.get("notes", []),
            "notes",
        )

        return cls(
            run_id=run_id,
            date=iso_date,
            base_model=base_model,
            adapter_names=adapter_names,
            qa_artifact_paths=qa_artifact_paths,
            pair_report_paths=pair_report_paths,
            neighborhood_report_path=neighborhood_report_path,
            downstream_outcome_paths=downstream_outcome_paths,
            notes=notes,
        )
