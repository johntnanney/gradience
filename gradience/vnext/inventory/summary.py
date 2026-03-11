"""Inventory-level summary schema (v1).

Aggregates adapter QA artifacts and merge risk reports into counts and
distributions, giving operators a single object that answers "what does
my adapter fleet look like?"

Schema: ``gradience.inventory_summary/v1`` -- frozen, additive-only.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from gradience.exceptions import QASchemaError

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCHEMA_ID = "gradience.inventory_summary/v1"

_STRICT_QA_BLOCK_STATUSES = frozenset({"flagged_weak", "unknown_no_behavioral_eval"})

# Sections that must be present and must be dict[str, int].
_REQUIRED_COUNT_MAPS = (
    "sources",
    "adapter_status_counts",
    "adapter_flag_counts",
    "pair_risk_counts",
    "recommended_strategy_counts",
    "dominant_issue_counts",
)


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InventorySummary:
    """Frozen summary of an adapter inventory.

    All count-map values must be non-negative integers.  ``notes`` is a
    tuple of free-form strings (optional; backfilled to ``()`` when
    absent).
    """

    sources: dict[str, int]
    adapter_status_counts: dict[str, int]
    adapter_flag_counts: dict[str, int]
    pair_risk_counts: dict[str, int]
    recommended_strategy_counts: dict[str, int]
    dominant_issue_counts: dict[str, int]
    strict_qa_block_candidates: int
    notes: tuple[str, ...] = field(default_factory=tuple)

    # -- Serialization -----------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a v1 schema dict."""
        return {
            "schema": SCHEMA_ID,
            "sources": dict(self.sources),
            "adapter_status_counts": dict(self.adapter_status_counts),
            "adapter_flag_counts": dict(self.adapter_flag_counts),
            "pair_risk_counts": dict(self.pair_risk_counts),
            "recommended_strategy_counts": dict(self.recommended_strategy_counts),
            "dominant_issue_counts": dict(self.dominant_issue_counts),
            "strict_qa_block_candidates": self.strict_qa_block_candidates,
            "notes": list(self.notes),
        }

    def to_json(self, path: Path | str) -> None:
        """Write inventory summary to a JSON file."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    # -- Deserialization ---------------------------------------------------

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> InventorySummary:
        """Deserialize from a v1 schema dict.

        This is the single canonical gatekeeper for the
        ``inventory_summary/v1`` schema.  Validates schema identity,
        required sections, type enforcement, and notes format.  Raises
        :class:`~gradience.exceptions.QASchemaError` on contract
        violations.  Extra keys are silently ignored for forward
        compatibility.
        """
        # --- Schema identity ---
        if "schema" not in d:
            raise QASchemaError("Missing required field: schema")
        if d["schema"] != SCHEMA_ID:
            raise QASchemaError(f"Expected schema '{SCHEMA_ID}', got '{d['schema']}'")

        # --- Required count-map sections ---
        for section in _REQUIRED_COUNT_MAPS:
            if section not in d:
                raise QASchemaError(f"Missing required section: {section}")
            val = d[section]
            if not isinstance(val, dict):
                raise QASchemaError(f"Section '{section}' must be a dict")
            for k, v in val.items():
                if not isinstance(v, int):
                    raise QASchemaError(f"Values in '{section}' must be int, got {type(v).__name__} for key '{k}'")

        # --- strict_qa_block_candidates ---
        if "strict_qa_block_candidates" not in d:
            raise QASchemaError("Missing required field: strict_qa_block_candidates")
        sqbc = d["strict_qa_block_candidates"]
        if not isinstance(sqbc, int):
            raise QASchemaError(f"Field 'strict_qa_block_candidates' must be int, got {type(sqbc).__name__}")

        # --- notes (optional, backfill to ()) ---
        raw_notes = d.get("notes", ())
        if raw_notes != () and not isinstance(raw_notes, list):
            raise QASchemaError(f"Field 'notes' must be a list of str, got {type(raw_notes).__name__}")
        if isinstance(raw_notes, list):
            for i, item in enumerate(raw_notes):
                if not isinstance(item, str):
                    raise QASchemaError(f"Each note must be a str, got {type(item).__name__} at index {i}")
            notes = tuple(raw_notes)
        else:
            notes = ()

        return cls(
            sources=d["sources"],
            adapter_status_counts=d["adapter_status_counts"],
            adapter_flag_counts=d["adapter_flag_counts"],
            pair_risk_counts=d["pair_risk_counts"],
            recommended_strategy_counts=d["recommended_strategy_counts"],
            dominant_issue_counts=d["dominant_issue_counts"],
            strict_qa_block_candidates=sqbc,
            notes=notes,
        )


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


def build_inventory_summary(
    qa_artifacts: list[Any],
    merge_reports: list[Any],
) -> InventorySummary:
    """Aggregate QA artifacts and merge reports into an inventory summary.

    Pure counting — no policy decisions beyond identifying strict-QA block
    candidates.

    Parameters
    ----------
    qa_artifacts
        List of :class:`~gradience.vnext.audit.qa_artifact.AdapterQAArtifact`.
    merge_reports
        List of :class:`~gradience.vnext.merge.qa_report.MergeQAReport`.

    Returns
    -------
    InventorySummary
    """
    # --- Adapter-level counts ---
    status_counter: Counter[str] = Counter()
    flag_counter: Counter[str] = Counter()
    for artifact in qa_artifacts:
        status_counter[artifact.status.value] += 1
        for flag in artifact.structural_flags:
            flag_counter[flag] += 1

    # --- Merge-report-level counts ---
    risk_counter: Counter[str] = Counter()
    strategy_counter: Counter[str] = Counter()
    issue_counter: Counter[str] = Counter()
    strict_qa_block_candidates = 0

    for report in merge_reports:
        risk_counter[report.pair_risk] += 1
        strategy_counter[report.recommended_strategy] += 1
        issue_counter[report.dominant_issue] += 1

        # A report is a block candidate if *either* adapter has a
        # problematic eligibility status.  Both-bad still counts as 1.
        a_elig = report.adapter_a.eligibility_status
        b_elig = report.adapter_b.eligibility_status
        a_blocks = a_elig is None or a_elig in _STRICT_QA_BLOCK_STATUSES
        b_blocks = b_elig is None or b_elig in _STRICT_QA_BLOCK_STATUSES
        if a_blocks or b_blocks:
            strict_qa_block_candidates += 1

    return InventorySummary(
        sources={
            "qa_artifact_count": len(qa_artifacts),
            "merge_report_count": len(merge_reports),
        },
        adapter_status_counts=dict(status_counter),
        adapter_flag_counts=dict(flag_counter),
        pair_risk_counts=dict(risk_counter),
        recommended_strategy_counts=dict(strategy_counter),
        dominant_issue_counts=dict(issue_counter),
        strict_qa_block_candidates=strict_qa_block_candidates,
    )


# ---------------------------------------------------------------------------
# Terminal formatter
# ---------------------------------------------------------------------------

_SECTION_DEFS: tuple[tuple[str, str], ...] = (
    ("adapter_status_counts", "ADAPTER STATUS"),
    ("adapter_flag_counts", "STRUCTURAL FLAGS"),
    ("pair_risk_counts", "PAIR RISK"),
    ("recommended_strategy_counts", "RECOMMENDED STRATEGIES"),
    ("dominant_issue_counts", "DOMINANT ISSUES"),
)


def format_inventory_summary(summary: InventorySummary) -> str:
    """Format an :class:`InventorySummary` as clean, human-readable text.

    Sections whose count dicts are empty (all zeroes or no keys) are
    omitted.  The SOURCES and STRICT-QA BLOCK CANDIDATES sections are
    always shown.
    """
    lines: list[str] = []

    lines.append("")
    lines.append("  INVENTORY SUMMARY")
    lines.append("  " + "=" * 60)

    # --- SOURCES (always shown) ---
    lines.append("")
    lines.append("  SOURCES")
    lines.append("  " + "-" * 40)
    for key in sorted(summary.sources):
        label = key.replace("_", " ").replace("count", "").strip()
        # Capitalise first word only for readability
        label = label[0].upper() + label[1:] + ":" if label else key + ":"
        lines.append(f"  {label:<20s}{summary.sources[key]}")

    # --- Count-map sections (omitted when empty) ---
    for attr, header in _SECTION_DEFS:
        counts: dict[str, int] = getattr(summary, attr)
        if not counts or all(v == 0 for v in counts.values()):
            continue
        lines.append("")
        lines.append(f"  {header}")
        lines.append("  " + "-" * 40)
        for key in sorted(counts):
            lines.append(f"  {key}:  {counts[key]}")

    # --- STRICT-QA BLOCK CANDIDATES (always shown) ---
    lines.append("")
    lines.append(f"  STRICT-QA BLOCK CANDIDATES: {summary.strict_qa_block_candidates}")

    lines.append("")

    return "\n".join(lines)
