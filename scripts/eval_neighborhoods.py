"""Evaluate suggest-neighborhoods behavior on named inventory fixtures.

Produces a compact evaluation bundle per fixture:
- emitted gradience.merge_neighborhoods/v1 JSON
- terminal summary capture
- comparison against expected outcomes
- verdict file (passed | partially passed | unexpected grouping)
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from gradience.vnext.audit.qa_artifact import AdapterQAArtifact
from gradience.vnext.inventory.merge_neighborhood_report import (
    MergeNeighborhoodReport,
    format_merge_neighborhood_report,
)
from gradience.vnext.inventory.neighborhoods import suggest_merge_neighborhoods
from gradience.vnext.merge.qa_report import MergeQAReport


@dataclass(frozen=True)
class FixtureExpected:
    groups: list[set[str]]
    excluded: set[str]
    boundary_pairs: list[set[frozenset[str]]]
    characterizations: dict[frozenset[str], str]


@dataclass(frozen=True)
class ComparisonResult:
    groups_match: bool
    excluded_match: bool
    boundaries_match: bool
    characterizations_match: bool
    verdict: str
    details: dict[str, Any]


def _load_json(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return data


def _load_fixture_expected(path: Path) -> FixtureExpected:
    if not path.is_file():
        return FixtureExpected(groups=[], excluded=set(), boundary_pairs=[], characterizations={})

    d = _load_json(path)

    groups: list[set[str]] = []
    for raw_group in d.get("expected_groups", []):
        if not isinstance(raw_group, list) or not all(isinstance(x, str) for x in raw_group):
            raise ValueError(f"expected_groups must be list[list[str]] in {path}")
        groups.append(set(raw_group))

    excluded_raw = d.get("expected_excluded", [])
    if not isinstance(excluded_raw, list) or not all(isinstance(x, str) for x in excluded_raw):
        raise ValueError(f"expected_excluded must be list[str] in {path}")
    excluded = set(excluded_raw)

    boundary_pairs: list[set[frozenset[str]]] = []
    for raw_pair in d.get("expected_boundary_warnings", []):
        if (
            not isinstance(raw_pair, list)
            or len(raw_pair) != 2
            or not all(isinstance(side, list) and all(isinstance(x, str) for x in side) for side in raw_pair)
        ):
            raise ValueError(f"expected_boundary_warnings must be list[[list[str], list[str]]] in {path}")
        boundary_pairs.append({frozenset(raw_pair[0]), frozenset(raw_pair[1])})

    characterizations: dict[frozenset[str], str] = {}
    for raw_item in d.get("expected_characterizations", []):
        if not isinstance(raw_item, dict):
            raise ValueError(f"expected_characterizations items must be dicts in {path}")
        members = raw_item.get("members")
        characterization = raw_item.get("characterization")
        if not isinstance(members, list) or not all(isinstance(x, str) for x in members):
            raise ValueError(f"expected_characterizations[].members must be list[str] in {path}")
        if not isinstance(characterization, str):
            raise ValueError(f"expected_characterizations[].characterization must be str in {path}")
        characterizations[frozenset(members)] = characterization

    return FixtureExpected(
        groups=groups,
        excluded=excluded,
        boundary_pairs=boundary_pairs,
        characterizations=characterizations,
    )


def _actual_groups(report: MergeNeighborhoodReport) -> list[set[str]]:
    return [set(g.members) for g in report.groups]


def _actual_excluded(report: MergeNeighborhoodReport) -> set[str]:
    return {e.adapter for e in report.excluded}


def _actual_boundary_pairs(report: MergeNeighborhoodReport) -> list[set[frozenset[str]]]:
    group_map = {g.group_id: frozenset(g.members) for g in report.groups}
    pairs: list[set[frozenset[str]]] = []
    for bw in report.boundary_warnings:
        left = group_map.get(bw.between[0])
        right = group_map.get(bw.between[1])
        if left is None or right is None:
            continue
        pairs.append({left, right})
    return pairs


def _compare_fixture(report: MergeNeighborhoodReport, expected: FixtureExpected) -> ComparisonResult:
    actual_groups = _actual_groups(report)
    actual_excluded = _actual_excluded(report)
    actual_boundary_pairs = _actual_boundary_pairs(report)
    actual_characterizations = {frozenset(g.members): g.characterization for g in report.groups}

    exp_groups_set = {frozenset(g) for g in expected.groups}
    act_groups_set = {frozenset(g) for g in actual_groups}
    groups_match = exp_groups_set == act_groups_set

    excluded_match = expected.excluded == actual_excluded

    exp_bounds_set = {frozenset(pair) for pair in expected.boundary_pairs}
    act_bounds_set = {frozenset(pair) for pair in actual_boundary_pairs}
    boundaries_match = exp_bounds_set == act_bounds_set

    char_mismatches: list[dict[str, Any]] = []
    for members, exp_char in expected.characterizations.items():
        act_char = actual_characterizations.get(members)
        if act_char != exp_char:
            char_mismatches.append(
                {
                    "members": sorted(members),
                    "expected": exp_char,
                    "actual": act_char,
                }
            )
    characterizations_match = len(char_mismatches) == 0

    if groups_match and excluded_match and boundaries_match and characterizations_match:
        verdict = "passed"
    elif not groups_match:
        verdict = "unexpected grouping"
    else:
        verdict = "partially passed"

    details = {
        "expected": {
            "groups": sorted([sorted(g) for g in expected.groups]),
            "excluded": sorted(expected.excluded),
            "boundary_pairs": sorted(
                [[sorted(list(side)) for side in pair] for pair in expected.boundary_pairs],
                key=lambda x: (x[0], x[1]),
            ),
            "characterizations": [
                {"members": sorted(k), "characterization": v} for k, v in sorted(expected.characterizations.items())
            ],
        },
        "actual": {
            "groups": sorted([sorted(g) for g in actual_groups]),
            "excluded": sorted(actual_excluded),
            "boundary_pairs": sorted(
                [[sorted(list(side)) for side in pair] for pair in actual_boundary_pairs],
                key=lambda x: (x[0], x[1]),
            ),
            "characterizations": [
                {"members": sorted(list(k)), "characterization": v}
                for k, v in sorted(actual_characterizations.items(), key=lambda x: sorted(list(x[0])))
            ],
        },
        "mismatches": {
            "groups_only_in_expected": sorted([sorted(g) for g in exp_groups_set - act_groups_set]),
            "groups_only_in_actual": sorted([sorted(g) for g in act_groups_set - exp_groups_set]),
            "excluded_only_in_expected": sorted(expected.excluded - actual_excluded),
            "excluded_only_in_actual": sorted(actual_excluded - expected.excluded),
            "boundaries_only_in_expected": sorted(
                [[sorted(list(side)) for side in pair] for pair in exp_bounds_set - act_bounds_set],
                key=lambda x: (x[0], x[1]),
            ),
            "boundaries_only_in_actual": sorted(
                [[sorted(list(side)) for side in pair] for pair in act_bounds_set - exp_bounds_set],
                key=lambda x: (x[0], x[1]),
            ),
            "characterization_mismatches": char_mismatches,
        },
    }

    return ComparisonResult(
        groups_match=groups_match,
        excluded_match=excluded_match,
        boundaries_match=boundaries_match,
        characterizations_match=characterizations_match,
        verdict=verdict,
        details=details,
    )


def _load_qa_artifacts(qa_dir: Path) -> list[AdapterQAArtifact]:
    artifacts: list[AdapterQAArtifact] = []
    for path in sorted(qa_dir.glob("*.json")):
        artifacts.append(AdapterQAArtifact.from_dict(_load_json(path)))
    return artifacts


def _load_reports(report_dir: Path) -> list[MergeQAReport]:
    reports: list[MergeQAReport] = []
    for path in sorted(report_dir.glob("*.json")):
        reports.append(MergeQAReport.from_dict(_load_json(path)))
    return reports


def _evaluate_fixture(
    fixture_dir: Path,
    out_root: Path,
    *,
    strict_qa: bool = False,
    min_compatibility: float = 0.0,
    exclude_unknown: bool = False,
) -> dict[str, Any]:
    qa_dir = fixture_dir / "qa"
    report_dir = fixture_dir / "reports"
    expected_path = fixture_dir / "expected.json"
    expected_notes_path = fixture_dir / "expected_notes.md"

    if not qa_dir.is_dir():
        raise FileNotFoundError(f"Missing qa directory in fixture: {fixture_dir}")
    if not report_dir.is_dir():
        raise FileNotFoundError(f"Missing reports directory in fixture: {fixture_dir}")

    artifacts = _load_qa_artifacts(qa_dir)
    reports = _load_reports(report_dir)
    expected = _load_fixture_expected(expected_path)

    neighborhood_report = suggest_merge_neighborhoods(
        artifacts,
        reports,
        strict_qa=strict_qa,
        min_compatibility=min_compatibility,
        exclude_unknown=exclude_unknown,
    )
    terminal_text = format_merge_neighborhood_report(neighborhood_report)
    comparison = _compare_fixture(neighborhood_report, expected)
    expected_notes = expected_notes_path.read_text(encoding="utf-8") if expected_notes_path.is_file() else None

    fixture_out = out_root / fixture_dir.name
    fixture_out.mkdir(parents=True, exist_ok=True)
    with open(fixture_out / "neighborhood_report.json", "w", encoding="utf-8") as f:
        json.dump(neighborhood_report.to_dict(), f, indent=2)
        f.write("\n")
    with open(fixture_out / "terminal_summary.txt", "w", encoding="utf-8") as f:
        f.write(terminal_text)
        if not terminal_text.endswith("\n"):
            f.write("\n")
    with open(fixture_out / "comparison.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "fixture": fixture_dir.name,
                "groups_match": comparison.groups_match,
                "excluded_match": comparison.excluded_match,
                "boundaries_match": comparison.boundaries_match,
                "characterizations_match": comparison.characterizations_match,
                "verdict": comparison.verdict,
                "details": comparison.details,
                "expected_notes": expected_notes,
            },
            f,
            indent=2,
        )
        f.write("\n")
    if expected_notes is not None:
        (fixture_out / "expected_notes.md").write_text(expected_notes, encoding="utf-8")
    with open(fixture_out / "verdict.txt", "w", encoding="utf-8") as f:
        f.write(comparison.verdict + "\n")

    return {
        "fixture": fixture_dir.name,
        "qa_count": len(artifacts),
        "report_count": len(reports),
        "groups_match": comparison.groups_match,
        "excluded_match": comparison.excluded_match,
        "boundaries_match": comparison.boundaries_match,
        "characterizations_match": comparison.characterizations_match,
        "verdict": comparison.verdict,
    }


def _write_markdown_summary(rows: list[dict[str, Any]], out_path: Path) -> None:
    lines = []
    lines.append("# Neighborhood Evaluation Summary")
    lines.append("")
    lines.append("| Fixture | QA | Reports | Groups | Excluded | Boundaries | Characterizations | Verdict |")
    lines.append("|---|---:|---:|---|---|---|---|---|")
    for row in rows:
        lines.append(
            f"| {row['fixture']} | {row['qa_count']} | {row['report_count']} | "
            f"{'OK' if row['groups_match'] else 'MISS'} | "
            f"{'OK' if row['excluded_match'] else 'MISS'} | "
            f"{'OK' if row['boundaries_match'] else 'MISS'} | "
            f"{'OK' if row['characterizations_match'] else 'MISS'} | "
            f"{row['verdict']} |"
        )
    lines.append("")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _discover_fixtures(fixtures_root: Path, selected: list[str] | None) -> list[Path]:
    all_fixtures = sorted([p for p in fixtures_root.iterdir() if p.is_dir()])
    if not selected:
        return all_fixtures
    wanted = set(selected)
    chosen = [p for p in all_fixtures if p.name in wanted]
    missing = sorted(wanted - {p.name for p in chosen})
    if missing:
        raise FileNotFoundError(f"Unknown fixture(s): {', '.join(missing)}")
    return chosen


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate suggest-neighborhoods across inventory fixtures.")
    parser.add_argument(
        "--fixtures-root",
        type=Path,
        default=Path("examples/inventories"),
        help="Root directory containing inventory fixtures (default: examples/inventories)",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("results/neighborhood_eval"),
        help="Root directory where evaluation bundles are written (default: results/neighborhood_eval)",
    )
    parser.add_argument(
        "--fixture",
        action="append",
        default=None,
        help="Specific fixture name to run. Repeatable. Default: run all fixtures.",
    )
    parser.add_argument("--strict-qa", action="store_true", help="Run suggester with strict_qa=True")
    parser.add_argument(
        "--min-compatibility",
        type=float,
        default=0.0,
        help="Minimum compatibility score passed to suggester (default: 0.0)",
    )
    parser.add_argument(
        "--exclude-unknown",
        action="store_true",
        help="Run suggester with exclude_unknown=True",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fixtures_root: Path = args.fixtures_root
    out_root: Path = args.out_root

    if not fixtures_root.is_dir():
        raise FileNotFoundError(f"Fixtures root does not exist: {fixtures_root}")

    run_label = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_out = out_root / run_label
    run_out.mkdir(parents=True, exist_ok=True)

    fixture_dirs = _discover_fixtures(fixtures_root, args.fixture)
    if not fixture_dirs:
        raise ValueError(f"No fixtures found under {fixtures_root}")

    rows: list[dict[str, Any]] = []
    for fixture_dir in fixture_dirs:
        row = _evaluate_fixture(
            fixture_dir,
            run_out,
            strict_qa=bool(args.strict_qa),
            min_compatibility=float(args.min_compatibility),
            exclude_unknown=bool(args.exclude_unknown),
        )
        rows.append(row)

    with open(run_out / "summary.json", "w", encoding="utf-8") as f:
        json.dump({"run_label": run_label, "fixtures": rows}, f, indent=2)
        f.write("\n")
    _write_markdown_summary(rows, run_out / "summary.md")

    print(f"Neighborhood evaluation run: {run_label}")
    print(f"Output directory: {run_out}")
    for row in rows:
        print(
            f"- {row['fixture']}: {row['verdict']} "
            f"(groups={'ok' if row['groups_match'] else 'miss'}, "
            f"excluded={'ok' if row['excluded_match'] else 'miss'}, "
            f"boundaries={'ok' if row['boundaries_match'] else 'miss'})"
        )


if __name__ == "__main__":
    main()
