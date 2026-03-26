"""Merge neighborhood artifact schema (v1).

This artifact is an inventory-level decision aid built from existing
AdapterQAArtifact and MergeQAReport objects. It is intentionally
diagnostic-first and rule-based.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gradience.exceptions import QASchemaError

SCHEMA_ID = "gradience.merge_neighborhoods/v1"

CHARACTERIZATION_VALUES = frozenset(
    {
        "likely-safe neighborhood",
        "caution neighborhood",
        "audit-aware neighborhood",
    }
)


def _require_list_of_str(value: Any, field_name: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(x, str) for x in value):
        raise QASchemaError(f"Field '{field_name}' must be a list of strings")
    return value


@dataclass(frozen=True)
class NeighborhoodGroup:
    group_id: str
    members: tuple[str, ...]
    characterization: str
    common_strategy: str
    dominant_issue: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "group_id": self.group_id,
            "members": list(self.members),
            "characterization": self.characterization,
            "common_strategy": self.common_strategy,
            "dominant_issue": self.dominant_issue,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> NeighborhoodGroup:
        for field_name in ("group_id", "members", "characterization", "common_strategy", "dominant_issue"):
            if field_name not in d:
                raise QASchemaError(f"Missing required field: groups[].{field_name}")

        members = tuple(_require_list_of_str(d["members"], "groups[].members"))
        if not members:
            raise QASchemaError("Field 'groups[].members' must contain at least one adapter")

        characterization = str(d["characterization"])
        if characterization not in CHARACTERIZATION_VALUES:
            raise QASchemaError(
                f"Invalid groups[].characterization '{characterization}'. "
                f"Must be one of: {sorted(CHARACTERIZATION_VALUES)}"
            )

        return cls(
            group_id=str(d["group_id"]),
            members=members,
            characterization=characterization,
            common_strategy=str(d["common_strategy"]),
            dominant_issue=str(d["dominant_issue"]),
        )


@dataclass(frozen=True)
class ExcludedAdapter:
    adapter: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "adapter": self.adapter,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ExcludedAdapter:
        if "adapter" not in d:
            raise QASchemaError("Missing required field: excluded[].adapter")
        if "reason" not in d:
            raise QASchemaError("Missing required field: excluded[].reason")
        return cls(adapter=str(d["adapter"]), reason=str(d["reason"]))


@dataclass(frozen=True)
class BoundaryWarning:
    between: tuple[str, str]
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "between": [self.between[0], self.between[1]],
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> BoundaryWarning:
        if "between" not in d:
            raise QASchemaError("Missing required field: boundary_warnings[].between")
        if "reason" not in d:
            raise QASchemaError("Missing required field: boundary_warnings[].reason")

        between = _require_list_of_str(d["between"], "boundary_warnings[].between")
        if len(between) != 2:
            raise QASchemaError("Field 'boundary_warnings[].between' must contain exactly two group ids")

        return cls(
            between=(between[0], between[1]),
            reason=str(d["reason"]),
        )


@dataclass(frozen=True)
class MergeNeighborhoodReport:
    groups: tuple[NeighborhoodGroup, ...]
    excluded: tuple[ExcludedAdapter, ...]
    boundary_warnings: tuple[BoundaryWarning, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA_ID,
            "groups": [g.to_dict() for g in self.groups],
            "excluded": [e.to_dict() for e in self.excluded],
            "boundary_warnings": [bw.to_dict() for bw in self.boundary_warnings],
        }

    def to_json(self, path: Path | str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> MergeNeighborhoodReport:
        if "schema" not in d:
            raise QASchemaError("Missing required field: schema")
        if d["schema"] != SCHEMA_ID:
            raise QASchemaError(f"Expected schema '{SCHEMA_ID}', got '{d['schema']}'")

        for section_name in ("groups", "excluded", "boundary_warnings"):
            if section_name not in d:
                raise QASchemaError(f"Missing required section: {section_name}")
            if not isinstance(d[section_name], list):
                raise QASchemaError(f"Section '{section_name}' must be a list")

        groups = tuple(NeighborhoodGroup.from_dict(x) for x in d["groups"])
        excluded = tuple(ExcludedAdapter.from_dict(x) for x in d["excluded"])
        boundary_warnings = tuple(BoundaryWarning.from_dict(x) for x in d["boundary_warnings"])

        return cls(
            groups=groups,
            excluded=excluded,
            boundary_warnings=boundary_warnings,
        )


def format_merge_neighborhood_report(report: MergeNeighborhoodReport) -> str:
    """Format a neighborhood report for terminal output."""
    lines: list[str] = []
    lines.append("")
    lines.append("MERGE NEIGHBORHOODS")
    lines.append("")

    if report.groups:
        for i, group in enumerate(report.groups, start=1):
            members = ", ".join(group.members)
            lines.append(f"Group {i}: {members}")
            lines.append(f"- characterization: {group.characterization}")
            lines.append(f"- common strategy: {group.common_strategy}")
            lines.append(f"- dominant issue: {group.dominant_issue}")
            lines.append("")
    else:
        lines.append("No merge neighborhoods were formed.")
        lines.append("")

    lines.append("Excluded:")
    if report.excluded:
        for item in report.excluded:
            lines.append(f"- {item.adapter} ({item.reason})")
    else:
        lines.append("- none")
    lines.append("")

    if report.boundary_warnings:
        lines.append("Boundary warnings:")
        for warning in report.boundary_warnings:
            lines.append(f"- {warning.between[0]} <-> {warning.between[1]}: {warning.reason}")
        lines.append("")

    return "\n".join(lines)
