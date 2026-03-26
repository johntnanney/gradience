"""Summarize an accumulated corpus of validated manifest entries."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from gradience.vnext.audit.qa_artifact import AdapterQAArtifact
from gradience.vnext.inventory.corpus_manifest import CorpusManifest
from gradience.vnext.inventory.merge_neighborhood_report import MergeNeighborhoodReport
from gradience.vnext.merge.qa_report import MergeQAReport

REPO_ROOT = Path(__file__).resolve().parents[1]
_STRICT_BLOCK_STATUSES = frozenset({"flagged_weak", "unknown_no_behavioral_eval", None})


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


def _resolve_ref(ref: str, manifest_path: Path) -> Path:
    raw = Path(ref)
    if raw.is_absolute():
        return raw.resolve()

    repo_candidate = (REPO_ROOT / raw).resolve()
    if repo_candidate.exists():
        return repo_candidate

    manifest_relative = (manifest_path.parent / raw).resolve()
    if manifest_relative.exists():
        return manifest_relative

    return repo_candidate


def _discover_manifest_paths(corpus_root: Path, explicit_paths: list[str] | None) -> list[Path]:
    if explicit_paths:
        return [Path(p).resolve() for p in explicit_paths]
    manifest_dir = corpus_root / "manifests"
    return sorted(p.resolve() for p in manifest_dir.glob("*.json"))


def _load_manifest(path: Path) -> CorpusManifest:
    data = _load_json_object(path)
    try:
        return CorpusManifest.from_dict(data)
    except Exception as exc:
        raise ValueError(f"Malformed corpus manifest: {path} ({exc})") from exc


def _load_qa(path: Path) -> AdapterQAArtifact:
    data = _load_json_object(path)
    try:
        return AdapterQAArtifact.from_dict(data)
    except Exception as exc:
        raise ValueError(f"Malformed QA artifact: {path} ({exc})") from exc


def _load_pair_report(path: Path) -> MergeQAReport:
    data = _load_json_object(path)
    try:
        return MergeQAReport.from_dict(data)
    except Exception as exc:
        raise ValueError(f"Malformed merge QA report: {path} ({exc})") from exc


def _load_neighborhood(path: Path) -> MergeNeighborhoodReport:
    data = _load_json_object(path)
    try:
        return MergeNeighborhoodReport.from_dict(data)
    except Exception as exc:
        raise ValueError(f"Malformed merge neighborhoods report: {path} ({exc})") from exc


def _canonicalize_identity_path(raw: str, manifest_path: Path) -> str:
    """Resolve a stable canonical path string for identity keys."""
    p = Path(raw)
    if p.is_absolute():
        return str(p.resolve())

    repo_candidate = (REPO_ROOT / p).resolve()
    if repo_candidate.exists():
        return str(repo_candidate)

    manifest_candidate = (manifest_path.parent / p).resolve()
    if manifest_candidate.exists():
        return str(manifest_candidate)

    # Deterministic fallback for non-existent relative paths.
    return str(repo_candidate)


def resolve_adapter_instance_key(
    *,
    manifest_data: dict[str, Any],
    manifest_path: Path,
    qa_ref: str,
    qa_path: Path,
    qa_data: dict[str, Any],
    qa_artifact: AdapterQAArtifact,
    index: int,
) -> str:
    """Resolve a deterministic identity-safe key for one adapter instance.

    Source precedence (first available wins):
    1) explicit manifest-level instance id (`adapter_instance_ids[index]`)
    2) explicit artifact-level instance id (`adapter.instance_id`)
    3) normalized adapter path from QA artifact metadata (`adapter.path`)
    4) canonical resolved QA artifact path
    5) stable hash fallback of canonical QA path payload
    """
    raw_manifest_ids = manifest_data.get("adapter_instance_ids")
    if isinstance(raw_manifest_ids, list) and index < len(raw_manifest_ids):
        candidate = raw_manifest_ids[index]
        if isinstance(candidate, str) and candidate.strip():
            return f"manifest_id:{candidate.strip()}"

    adapter_section = qa_data.get("adapter")
    if isinstance(adapter_section, dict):
        artifact_instance_id = adapter_section.get("instance_id")
        if isinstance(artifact_instance_id, str) and artifact_instance_id.strip():
            return f"artifact_id:{artifact_instance_id.strip()}"

    if qa_artifact.adapter_path and qa_artifact.adapter_path.strip():
        canonical_adapter_path = _canonicalize_identity_path(qa_artifact.adapter_path, manifest_path)
        return f"adapter_path:{canonical_adapter_path}"

    canonical_qa_path = _canonicalize_identity_path(qa_ref, manifest_path)
    if canonical_qa_path:
        return f"qa_artifact_path:{canonical_qa_path}"

    fallback_hash = hashlib.sha256(str(qa_path.resolve()).encode("utf-8")).hexdigest()
    return f"qa_path_hash:{fallback_hash}"


def _render_terminal(summary: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("CORPUS SUMMARY")
    lines.append("")
    lines.append(f"inventories: {summary['inventory_count']}")
    lines.append(f"adapter instances: {summary['adapter_instance_count']}")
    lines.append(f"unique adapters: {summary['unique_adapter_count']}")
    lines.append(f"unique adapter display names: {summary['unique_adapter_display_name_count']}")
    lines.append(f"pair reports: {summary['pair_report_count']}")
    lines.append(f"strict block candidate pairs: {summary['strict_block_candidate_pairs']}")
    lines.append("")
    n = summary["neighborhood_counts"]
    lines.append("neighborhood totals:")
    lines.append(f"- groups: {n['groups_total']}")
    lines.append(f"- excluded: {n['excluded_total']}")
    lines.append(f"- boundary warnings: {n['boundary_warnings_total']}")
    lines.append("")

    def _render_counter(title: str, values: dict[str, int]) -> None:
        lines.append(title + ":")
        if not values:
            lines.append("- none")
            lines.append("")
            return
        for key in sorted(values):
            lines.append(f"- {key}: {values[key]}")
        lines.append("")

    _render_counter("strategy distribution", summary["strategy_counts"])
    _render_counter("issue distribution", summary["dominant_issue_counts"])
    _render_counter("base model distribution", summary["base_model_counts"])
    _render_counter("neighborhood characterization distribution", summary["neighborhood_characterization_counts"])

    return "\n".join(lines).rstrip() + "\n"


def _render_markdown(summary: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Corpus Summary")
    lines.append("")
    lines.append(f"- Inventories: **{summary['inventory_count']}**")
    lines.append(f"- Adapter instances: **{summary['adapter_instance_count']}**")
    lines.append(f"- Unique adapters: **{summary['unique_adapter_count']}**")
    lines.append(f"- Unique adapter display names: **{summary['unique_adapter_display_name_count']}**")
    lines.append(f"- Pair reports: **{summary['pair_report_count']}**")
    lines.append(f"- Strict block candidate pairs: **{summary['strict_block_candidate_pairs']}**")
    lines.append("")

    n = summary["neighborhood_counts"]
    lines.append("## Neighborhood Counts")
    lines.append("")
    lines.append(f"- Groups total: **{n['groups_total']}**")
    lines.append(f"- Excluded total: **{n['excluded_total']}**")
    lines.append(f"- Boundary warnings total: **{n['boundary_warnings_total']}**")
    lines.append("")

    def _counter_table(title: str, values: dict[str, int]) -> None:
        lines.append(f"## {title}")
        lines.append("")
        if not values:
            lines.append("_none_")
            lines.append("")
            return
        lines.append("| Key | Count |")
        lines.append("|---|---:|")
        for key in sorted(values):
            lines.append(f"| {key} | {values[key]} |")
        lines.append("")

    _counter_table("Strategy Distribution", summary["strategy_counts"])
    _counter_table("Issue Distribution", summary["dominant_issue_counts"])
    _counter_table("Base Models", summary["base_model_counts"])
    _counter_table("Neighborhood Characterizations", summary["neighborhood_characterization_counts"])

    lines.append("## Per-Inventory")
    lines.append("")
    lines.append("| Run ID | Date | Adapters | Pair Reports | Groups | Excluded | Boundaries |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for row in summary["inventories"]:
        lines.append(
            f"| {row['run_id']} | {row['date']} | {row['adapter_count']} | {row['pair_report_count']} | "
            f"{row['group_count']} | {row['excluded_count']} | {row['boundary_warning_count']} |"
        )
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize the current Gradience corpus manifest set.")
    parser.add_argument(
        "--corpus-root",
        default="results/corpus",
        help="Corpus root directory (default: results/corpus)",
    )
    parser.add_argument(
        "--manifest",
        action="append",
        default=None,
        help="Explicit corpus manifest path (repeat flag). Defaults to <corpus-root>/manifests/*.json",
    )
    parser.add_argument(
        "--emit-json",
        default=None,
        help="Optional output path for summary JSON",
    )
    parser.add_argument(
        "--emit-md",
        default=None,
        help="Optional output path for summary markdown",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    corpus_root = Path(args.corpus_root)

    try:
        manifest_paths = _discover_manifest_paths(corpus_root, args.manifest)
        if not manifest_paths:
            raise FileNotFoundError(
                f"No corpus manifests found under {corpus_root / 'manifests'}"
            )

        strategy_counts: Counter[str] = Counter()
        issue_counts: Counter[str] = Counter()
        base_model_counts: Counter[str] = Counter()
        characterization_counts: Counter[str] = Counter()

        inventory_rows: list[dict[str, Any]] = []
        unique_adapter_display_names: set[str] = set()
        unique_adapter_instance_keys: set[str] = set()
        pair_report_count = 0
        strict_block_candidates = 0
        groups_total = 0
        excluded_total = 0
        boundary_total = 0

        for manifest_path in manifest_paths:
            manifest_data = _load_json_object(manifest_path)
            try:
                manifest = CorpusManifest.from_dict(manifest_data)
            except Exception as exc:
                raise ValueError(f"Malformed corpus manifest: {manifest_path} ({exc})") from exc

            qa_artifacts: list[AdapterQAArtifact] = []
            manifest_instance_keys: set[str] = set()
            for i, ref in enumerate(manifest.qa_artifact_paths):
                qa_path = _resolve_ref(ref, manifest_path)
                qa_data = _load_json_object(qa_path)
                try:
                    qa_artifact = AdapterQAArtifact.from_dict(qa_data)
                except Exception as exc:
                    raise ValueError(f"Malformed QA artifact: {qa_path} ({exc})") from exc
                qa_artifacts.append(qa_artifact)
                manifest_instance_keys.add(
                    resolve_adapter_instance_key(
                        manifest_data=manifest_data,
                        manifest_path=manifest_path,
                        qa_ref=ref,
                        qa_path=qa_path,
                        qa_data=qa_data,
                        qa_artifact=qa_artifact,
                        index=i,
                    )
                )

            pair_reports = [
                _load_pair_report(_resolve_ref(ref, manifest_path))
                for ref in manifest.pair_report_paths
            ]
            neighborhood = _load_neighborhood(_resolve_ref(manifest.neighborhood_report_path, manifest_path))

            qa_adapter_names = sorted({a.adapter_name for a in qa_artifacts})
            if set(qa_adapter_names) != set(manifest.adapter_names):
                raise ValueError(
                    f"Manifest adapter_names mismatch for run_id={manifest.run_id}: "
                    f"manifest={sorted(manifest.adapter_names)} qa={qa_adapter_names}"
                )

            qa_base_models = sorted({a.base_model for a in qa_artifacts if a.base_model})
            if len(qa_base_models) > 1:
                raise ValueError(
                    f"Inconsistent QA base_model values for run_id={manifest.run_id}: {qa_base_models}"
                )
            if qa_base_models and qa_base_models[0] != manifest.base_model:
                raise ValueError(
                    f"Manifest base_model mismatch for run_id={manifest.run_id}: "
                    f"manifest={manifest.base_model} qa={qa_base_models[0]}"
                )

            base_model_counts[manifest.base_model] += 1
            unique_adapter_instance_keys.update(manifest_instance_keys)
            unique_adapter_display_names.update(manifest.adapter_names)
            pair_report_count += len(pair_reports)

            for report in pair_reports:
                strategy_counts[report.recommended_strategy] += 1
                issue_counts[report.dominant_issue] += 1
                a_status = report.adapter_a.eligibility_status
                b_status = report.adapter_b.eligibility_status
                if a_status in _STRICT_BLOCK_STATUSES or b_status in _STRICT_BLOCK_STATUSES:
                    strict_block_candidates += 1

            groups_total += len(neighborhood.groups)
            excluded_total += len(neighborhood.excluded)
            boundary_total += len(neighborhood.boundary_warnings)
            for group in neighborhood.groups:
                characterization_counts[group.characterization] += 1

            inventory_rows.append(
                {
                    "run_id": manifest.run_id,
                    "date": manifest.date,
                    "base_model": manifest.base_model,
                    "adapter_count": len(manifest_instance_keys),
                    "pair_report_count": len(pair_reports),
                    "group_count": len(neighborhood.groups),
                    "excluded_count": len(neighborhood.excluded),
                    "boundary_warning_count": len(neighborhood.boundary_warnings),
                    "manifest_path": str(manifest_path),
                }
            )

        summary: dict[str, Any] = {
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
            "inventory_count": len(inventory_rows),
            # Identity-safe unique instance count across manifests.
            "adapter_instance_count": len(unique_adapter_instance_keys),
            # Backward-compatible alias of identity-safe unique adapter count.
            "unique_adapter_count": len(unique_adapter_instance_keys),
            # Human-readable label diversity, kept separate from identity counting.
            "unique_adapter_display_name_count": len(unique_adapter_display_names),
            "pair_report_count": pair_report_count,
            "strict_block_candidate_pairs": strict_block_candidates,
            "strategy_counts": dict(strategy_counts),
            "dominant_issue_counts": dict(issue_counts),
            "base_model_counts": dict(base_model_counts),
            "neighborhood_characterization_counts": dict(characterization_counts),
            "neighborhood_counts": {
                "groups_total": groups_total,
                "excluded_total": excluded_total,
                "boundary_warnings_total": boundary_total,
            },
            "inventories": sorted(inventory_rows, key=lambda x: x["run_id"]),
        }

        terminal = _render_terminal(summary)
        sys.stdout.write(terminal)

        if args.emit_json:
            out_json = Path(args.emit_json)
            out_json.parent.mkdir(parents=True, exist_ok=True)
            with open(out_json, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
                f.write("\n")

        if args.emit_md:
            out_md = Path(args.emit_md)
            out_md.parent.mkdir(parents=True, exist_ok=True)
            out_md.write_text(_render_markdown(summary), encoding="utf-8")

    except Exception as exc:
        print(f"Error: failed to summarize corpus: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
