"""Append one validated corpus manifest entry.

The script validates referenced artifacts first and only writes a manifest when
every required input is present and schema-valid.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from gradience.vnext.audit.qa_artifact import AdapterQAArtifact
from gradience.vnext.inventory.corpus_manifest import CorpusManifest
from gradience.vnext.inventory.merge_neighborhood_report import MergeNeighborhoodReport
from gradience.vnext.merge.qa_report import MergeQAReport

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Missing required file: {path}") from exc
    except Exception as exc:
        raise ValueError(f"Unreadable JSON file: {path} ({exc})") from exc

    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return data


def _repo_relative_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def _collect_input_files(dir_path: str | None, explicit_paths: list[str] | None) -> list[Path]:
    paths: list[Path] = []
    if dir_path:
        d = Path(dir_path)
        if not d.is_dir():
            raise NotADirectoryError(f"Expected directory, got: {d}")
        paths.extend(sorted(d.glob("*.json")))
    if explicit_paths:
        paths.extend(Path(p) for p in explicit_paths)

    deduped: dict[Path, None] = {}
    for p in paths:
        deduped[p.resolve()] = None
    return list(deduped.keys())


def _validate_qa_artifacts(paths: list[Path]) -> list[AdapterQAArtifact]:
    if not paths:
        raise ValueError("No QA artifacts provided. Use --qa-dir and/or --qa-artifact.")

    artifacts: list[AdapterQAArtifact] = []
    for path in paths:
        data = _load_json_object(path)
        schema = data.get("schema", "")
        if not isinstance(schema, str) or not schema.startswith("gradience.adapter_qa/"):
            raise ValueError(f"File is not an adapter QA artifact: {path}")
        try:
            artifacts.append(AdapterQAArtifact.from_dict(data))
        except Exception as exc:
            raise ValueError(f"Malformed adapter QA artifact: {path} ({exc})") from exc
    return artifacts


def _validate_pair_reports(paths: list[Path]) -> list[MergeQAReport]:
    if not paths:
        raise ValueError("No pair reports provided. Use --report-dir and/or --pair-report.")

    reports: list[MergeQAReport] = []
    for path in paths:
        data = _load_json_object(path)
        schema = data.get("schema", "")
        if not isinstance(schema, str) or not schema.startswith("gradience.merge_qa_report/"):
            raise ValueError(f"File is not a merge QA report: {path}")
        try:
            reports.append(MergeQAReport.from_dict(data))
        except Exception as exc:
            raise ValueError(f"Malformed merge QA report: {path} ({exc})") from exc
    return reports


def _validate_neighborhood_report(path: Path) -> MergeNeighborhoodReport:
    data = _load_json_object(path)
    schema = data.get("schema", "")
    if schema != "gradience.merge_neighborhoods/v1":
        raise ValueError(f"File is not a merge neighborhoods report: {path}")
    try:
        return MergeNeighborhoodReport.from_dict(data)
    except Exception as exc:
        raise ValueError(f"Malformed merge neighborhoods report: {path} ({exc})") from exc


def _validated_existing_files(paths: list[str] | None) -> list[Path]:
    out: list[Path] = []
    for raw in paths or []:
        p = Path(raw).resolve()
        if not p.is_file():
            raise FileNotFoundError(f"Missing downstream outcome file: {p}")
        out.append(p)
    return out


def _resolve_base_model(
    qa_artifacts: list[AdapterQAArtifact],
    override: str | None,
) -> str:
    derived_values = sorted({a.base_model.strip() for a in qa_artifacts if a.base_model and a.base_model.strip()})
    if len(derived_values) > 1:
        raise ValueError(
            "Inconsistent base_model values in QA artifacts: " + ", ".join(derived_values)
        )
    derived = derived_values[0] if derived_values else None

    if override:
        override = override.strip()
        if not override:
            raise ValueError("Provided --base-model is empty")
        if derived and override != derived:
            raise ValueError(
                f"--base-model '{override}' does not match QA-derived base_model '{derived}'"
            )
        return override

    if derived:
        return derived
    raise ValueError("Could not derive base_model from QA artifacts; pass --base-model explicitly")


def _resolve_adapter_names(
    qa_artifacts: list[AdapterQAArtifact],
    overrides: list[str] | None,
) -> list[str]:
    derived = sorted({a.adapter_name.strip() for a in qa_artifacts if a.adapter_name and a.adapter_name.strip()})
    if not derived:
        raise ValueError("Could not derive adapter names from QA artifacts")

    if not overrides:
        return derived

    provided = sorted({name.strip() for name in overrides if name.strip()})
    if not provided:
        raise ValueError("Provided --adapter-name values are empty")
    if set(provided) != set(derived):
        raise ValueError(
            "Provided --adapter-name values do not match QA artifacts. "
            f"provided={provided}, derived={derived}"
        )
    return provided


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Append one validated entry to the corpus manifest set.")
    parser.add_argument("--run-id", required=True, help="Stable id for this corpus entry")
    parser.add_argument(
        "--date",
        default=None,
        help="ISO date/time for this entry (default: current UTC timestamp)",
    )
    parser.add_argument(
        "--base-model",
        default=None,
        help="Base model id/name; validated against QA artifacts when they include base_model",
    )
    parser.add_argument(
        "--adapter-name",
        action="append",
        default=None,
        help="Optional explicit adapter name (repeat flag). Must match QA-derived names.",
    )
    parser.add_argument("--qa-dir", default=None, help="Directory of adapter QA artifacts (*.json)")
    parser.add_argument(
        "--qa-artifact",
        action="append",
        default=None,
        help="Explicit adapter QA artifact path (repeat flag)",
    )
    parser.add_argument("--report-dir", default=None, help="Directory of merge QA reports (*.json)")
    parser.add_argument(
        "--pair-report",
        action="append",
        default=None,
        help="Explicit merge QA report path (repeat flag)",
    )
    parser.add_argument(
        "--neighborhood-report",
        required=True,
        help="Path to gradience.merge_neighborhoods/v1 report JSON",
    )
    parser.add_argument(
        "--downstream-outcome",
        action="append",
        default=None,
        help="Optional downstream outcome file path (repeat flag)",
    )
    parser.add_argument("--note", action="append", default=None, help="Optional note line (repeat flag)")
    parser.add_argument(
        "--corpus-root",
        default="results/corpus",
        help="Corpus root directory (default: results/corpus)",
    )
    parser.add_argument(
        "--emit-manifest",
        default=None,
        help="Explicit output path for manifest JSON (default: <corpus-root>/manifests/<run-id>.json)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing manifest file when path already exists",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        qa_paths = _collect_input_files(args.qa_dir, args.qa_artifact)
        report_paths = _collect_input_files(args.report_dir, args.pair_report)
        qa_artifacts = _validate_qa_artifacts(qa_paths)
        _validate_pair_reports(report_paths)
        _validate_neighborhood_report(Path(args.neighborhood_report).resolve())
        downstream_files = _validated_existing_files(args.downstream_outcome)

        manifest_date = args.date or datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
        base_model = _resolve_base_model(qa_artifacts, args.base_model)
        adapter_names = _resolve_adapter_names(qa_artifacts, args.adapter_name)
        notes = tuple(args.note or [])

        if args.emit_manifest:
            manifest_path = Path(args.emit_manifest)
        else:
            manifest_path = Path(args.corpus_root) / "manifests" / f"{args.run_id}.json"

        if manifest_path.exists() and not args.overwrite:
            raise FileExistsError(f"Manifest already exists: {manifest_path} (use --overwrite to replace)")

        manifest = CorpusManifest(
            run_id=args.run_id.strip(),
            date=manifest_date,
            base_model=base_model,
            adapter_names=tuple(adapter_names),
            qa_artifact_paths=tuple(sorted(_repo_relative_path(p) for p in qa_paths)),
            pair_report_paths=tuple(sorted(_repo_relative_path(p) for p in report_paths)),
            neighborhood_report_path=_repo_relative_path(Path(args.neighborhood_report).resolve()),
            downstream_outcome_paths=tuple(sorted(_repo_relative_path(p) for p in downstream_files)),
            notes=notes,
        )
        # Enforce the same strict loader used for future reads before writing.
        manifest = CorpusManifest.from_dict(manifest.to_dict())
        manifest.to_json(manifest_path)
    except Exception as exc:
        print(f"Error: failed to append corpus entry: {exc}", file=sys.stderr)
        return 1

    print(f"Wrote corpus manifest: {manifest_path}")
    print(f"- run_id: {manifest.run_id}")
    print(f"- adapter_count: {len(manifest.adapter_names)}")
    print(f"- pair_report_count: {len(manifest.pair_report_paths)}")
    print(f"- base_model: {manifest.base_model}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

