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

_SOURCE_LABELS: dict[str, str] = {
    "qa_artifact_count": "QA artifacts:",
    "merge_report_count": "Merge reports:",
}

_SECTION_DEFS: tuple[tuple[str, str], ...] = (
    ("adapter_status_counts", "ADAPTER STATUS"),
    ("adapter_flag_counts", "STRUCTURAL FLAGS"),
    ("pair_risk_counts", "PAIR RISK"),
    ("recommended_strategy_counts", "RECOMMENDED STRATEGIES"),
    ("dominant_issue_counts", "DOMINANT ISSUES"),
)


def _inventory_headline(summary: InventorySummary) -> str:
    """Generate a one-line inventory headline for quick scanning."""
    n_reports = summary.sources.get("merge_report_count", summary.sources.get("merge_reports", 0))
    n_qa = summary.sources.get("qa_artifact_count", summary.sources.get("qa_artifacts", 0))

    risk_counts = summary.pair_risk_counts or {}
    high_risk = risk_counts.get("high", 0)
    total_risk = sum(risk_counts.values()) if risk_counts else 0

    status_counts = summary.adapter_status_counts or {}
    weak = status_counts.get("flagged_weak", 0)
    unknown = status_counts.get("unknown_no_behavioral_eval", 0)

    if weak + unknown > 0:
        return f"Mixed-quality inventory — {weak + unknown} weak/unknown source(s) identified"
    if high_risk > total_risk // 2:
        return f"{n_qa} adapters, {n_reports} pairs — high structural risk dominates"
    return f"{n_qa} adapters, {n_reports} pairs"


def format_inventory_summary(summary: InventorySummary) -> str:
    """Format an :class:`InventorySummary` as clean, human-readable text.

    Output uses standardized blocks in a fixed order:

    1. INVENTORY OVERVIEW — headline and counts
    2. SOURCE QA SNAPSHOT — provenance/trust and eligibility
    3. STRUCTURAL DETAIL — pair risk, strategies, issues
    4. INTERPRETATION — plain-language guidance
    """
    lines: list[str] = []

    # ---------------------------------------------------------------
    # 1. INVENTORY OVERVIEW
    # ---------------------------------------------------------------
    lines.append("")
    lines.append("  INVENTORY OVERVIEW")
    lines.append("  " + "=" * 60)
    lines.append(f"  {_inventory_headline(summary)}")
    lines.append("")
    for key in sorted(summary.sources):
        label = _SOURCE_LABELS.get(key, key + ":")
        lines.append(f"  {label:<20s}{summary.sources[key]}")

    # ---------------------------------------------------------------
    # 2. SOURCE QA SNAPSHOT
    # ---------------------------------------------------------------
    status_counts = summary.adapter_status_counts or {}
    eligible = status_counts.get("eligible", 0)
    uncertain = status_counts.get("uncertain", 0)
    weak_count = status_counts.get("flagged_weak", 0)
    unknown_count = status_counts.get("unknown_no_behavioral_eval", 0)

    if any(v > 0 for v in status_counts.values()):
        lines.append("")
        lines.append("  SOURCE QA SNAPSHOT")
        lines.append("  " + "-" * 40)
        for key in sorted(status_counts):
            if status_counts[key] > 0:
                lines.append(f"  {key}:  {status_counts[key]}")

        # Provenance note
        if eligible + uncertain + weak_count + unknown_count > 0:
            lines.append("")
            if eligible > 0:
                lines.append(f"  {eligible} source(s) with behavioral evidence (user-provided)")
            if uncertain > 0:
                lines.append(f"  {uncertain} source(s) with uncertain behavioral evidence")
            if weak_count > 0:
                lines.append(f"  {weak_count} source(s) flagged weak")
            if unknown_count > 0:
                lines.append(f"  {unknown_count} source(s) with missing behavioral evidence")
            if weak_count + unknown_count > 0:
                lines.append("  Note: behavioral scores are user-provided; Gradience does not")
                lines.append("  independently verify claimed evaluation results.")

        if summary.strict_qa_block_candidates > 0:
            lines.append(f"  Strict-QA block candidates: {summary.strict_qa_block_candidates}")

    # ---------------------------------------------------------------
    # 3. STRUCTURAL DETAIL
    # ---------------------------------------------------------------
    has_structural = False
    for attr, header in _SECTION_DEFS:
        if attr == "adapter_status_counts":
            continue  # Already shown in SOURCE QA SNAPSHOT
        counts: dict[str, int] = getattr(summary, attr)
        if not counts or all(v == 0 for v in counts.values()):
            continue
        if not has_structural:
            lines.append("")
            lines.append("  STRUCTURAL DETAIL")
            lines.append("  " + "-" * 40)
            has_structural = True
        label_map = {
            "adapter_flag_counts": "Flags",
            "pair_risk_counts": "Pair risk",
            "recommended_strategy_counts": "Strategies",
            "dominant_issue_counts": "Issues",
        }
        section_label = label_map.get(attr, header)
        items = ", ".join(f"{k}: {counts[k]}" for k in sorted(counts) if counts[k] > 0)
        lines.append(f"  {section_label}: {items}")

    # ---------------------------------------------------------------
    # 4. INTERPRETATION
    # ---------------------------------------------------------------
    risk_counts = summary.pair_risk_counts or {}
    weak_total = weak_count + unknown_count
    total_pairs = sum(risk_counts.values()) if risk_counts else 0
    high_risk = risk_counts.get("high", 0)

    lines.append("")
    lines.append("  INTERPRETATION")
    lines.append("  " + "-" * 40)

    if weak_total > 0:
        lines.append(f"  {weak_total} adapter(s) have weak or missing behavioral evidence.")
        lines.append("  Source QA is likely the main narrowing step for this inventory.")
    elif total_pairs > 0 and high_risk > total_pairs // 2:
        lines.append("  Most pairs show high structural risk.")
        lines.append("  Check task-boundary advisories on individual pair reports to identify")
        lines.append("  same-task safe pairs, if any exist.")
    elif total_pairs > 0:
        lines.append("  Check task-boundary advisories on individual pair reports to partition")
        lines.append("  same-task safe pairs from cross-task caution pairs.")
    else:
        lines.append("  No pair reports available for interpretation.")

    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Inventory action plan
# ---------------------------------------------------------------------------

_WEAK_STATUSES = frozenset({"flagged_weak", "unknown_no_behavioral_eval"})


@dataclass(frozen=True)
class InventoryActionPlan:
    """Structured action plan derived from existing stable signals.

    Presentation only — no new scoring or recommendation logic.
    """

    exclude: tuple[str, ...]
    same_task_priority: tuple[str, ...]
    cross_task_caution: tuple[str, ...]
    evaluate_first: tuple[str, ...]
    summary_line: str
    total_pairs: int
    retained_count: int


def build_action_plan(
    qa_artifacts: list[Any],
    merge_reports: list[Any],
) -> InventoryActionPlan:
    """Build an action plan from raw QA artifacts and merge reports.

    Uses only existing stable signals: source QA status, pair-risk,
    and task-relationship advisory. No new logic or scoring.
    """
    from pathlib import PurePosixPath

    # --- Build adapter info maps ---
    adapter_status: dict[str, str] = {}  # name -> eligibility status
    adapter_eval_ds: dict[str, str] = {}  # name -> eval_dataset

    for qa in qa_artifacts:
        name = qa.adapter_name if hasattr(qa, "adapter_name") else getattr(qa, "name", "?")
        adapter_status[name] = qa.status.value if hasattr(qa.status, "value") else str(qa.status)
        eval_ds = getattr(qa, "eval_dataset", None)
        if eval_ds:
            adapter_eval_ds[name] = eval_ds

    # --- Classify sources ---
    exclude_names: list[str] = []
    for name, status in adapter_status.items():
        if status in _WEAK_STATUSES:
            if status == "flagged_weak":
                label = "weak source — low confidence"
            else:
                label = "missing behavioral evidence — low confidence"
            exclude_names.append(f"{name}: {label}")

    # --- Classify pairs ---
    same_task_pairs: list[str] = []
    cross_task_pairs: list[str] = []
    cross_task_regions: set[str] = set()

    for report in merge_reports:
        a_path = report.adapter_a.path if hasattr(report.adapter_a, "path") else ""
        b_path = report.adapter_b.path if hasattr(report.adapter_b, "path") else ""
        a_name = PurePosixPath(a_path).name if a_path else "?"
        b_name = PurePosixPath(b_path).name if b_path else "?"
        pair_label = f"{a_name} × {b_name}"

        has_advisory = report.task_relationship_advisory is not None

        # Check if either source is weak
        a_elig = report.adapter_a.eligibility_status
        b_elig = report.adapter_b.eligibility_status
        has_weak = (a_elig in _WEAK_STATUSES) or (b_elig in _WEAK_STATUSES) if a_elig and b_elig else False

        if has_weak:
            continue  # Already handled by exclude

        if has_advisory:
            cross_task_pairs.append(pair_label)
            # Extract task region names from eval_dataset
            a_ds = adapter_eval_ds.get(a_name, "")
            b_ds = adapter_eval_ds.get(b_name, "")
            if a_ds and b_ds:
                # Normalize: "qnli_dev" -> "QNLI", "sst2_dev" -> "SST-2"
                a_task = a_ds.replace("_dev", "").replace("_test", "").upper().replace("SST2", "SST-2")
                b_task = b_ds.replace("_dev", "").replace("_test", "").upper().replace("SST2", "SST-2")
                region = f"{min(a_task, b_task)} × {max(a_task, b_task)} region"
                cross_task_regions.add(region)
        else:
            same_task_pairs.append(pair_label)

    # --- Build evaluate-first list (same-task pairs minus weak sources) ---
    evaluate_first = same_task_pairs[:4]  # Cap at 4 for readability

    # --- Build summary line ---
    total_pairs = len(merge_reports)
    retained = len(same_task_pairs)

    if total_pairs == 0:
        summary_line = "No pair reports available for interpretation."
    elif retained == total_pairs:
        summary_line = "This same-task inventory is mostly confirmatory."
    elif retained == 0 and len(exclude_names) > 0:
        summary_line = "QA dominates this inventory; no credible same-task candidates remain."
    elif retained == 0:
        summary_line = "All pairs are cross-task; no same-task safe region exists in this inventory."
    elif len(exclude_names) > 0:
        summary_line = (
            f"QA and task boundary dominate this inventory. "
            f"Candidate space reduced from {total_pairs} pairs to {retained}."
        )
    else:
        pct = round(100 * (1 - retained / total_pairs)) if total_pairs > 0 else 0
        summary_line = (
            f"Inventory is mostly explained by task boundary. "
            f"Candidate space reduced from {total_pairs} pairs to {retained} ({pct}% reduction)."
        )

    # --- Cross-task caution entries ---
    caution_entries: list[str] = sorted(cross_task_regions)
    if cross_task_pairs and not caution_entries:
        caution_entries = ["cross-task pairs should not be prioritized for casual exploration"]

    return InventoryActionPlan(
        exclude=tuple(exclude_names),
        same_task_priority=tuple(same_task_pairs),
        cross_task_caution=tuple(caution_entries),
        evaluate_first=tuple(evaluate_first),
        summary_line=summary_line,
        total_pairs=total_pairs,
        retained_count=retained,
    )


def format_action_plan(plan: InventoryActionPlan) -> str:
    """Format an action plan as clean, human-readable text."""
    lines: list[str] = []

    lines.append("")
    lines.append("  INVENTORY ACTION PLAN")
    lines.append("  " + "=" * 60)

    # Reduced candidate set (visually primary)
    lines.append("")
    lines.append("  REDUCED CANDIDATE SET")
    lines.append("  " + "-" * 40)
    lines.append(f"  Starting pairs:      {plan.total_pairs}")
    lines.append(f"  Retained candidates: {plan.retained_count}")
    if plan.total_pairs > 0:
        pct = round(100 * (1 - plan.retained_count / plan.total_pairs))
        lines.append(f"  Reduction:           {pct}%")
    lines.append("")
    lines.append("  Evaluate first:")
    if plan.evaluate_first:
        for pair in plan.evaluate_first:
            lines.append(f"    - {pair}")
    else:
        lines.append("    - no clear priority candidates identified")

    # Exclude / deprioritize
    lines.append("")
    lines.append("  Exclude / deprioritize")
    lines.append("  " + "-" * 40)
    if plan.exclude:
        for entry in plan.exclude:
            lines.append(f"  - {entry}")
    else:
        lines.append("  - none")

    # Same-task safe zone
    lines.append("")
    lines.append("  Same-task safe zone")
    lines.append("  " + "-" * 40)
    if plan.same_task_priority:
        for pair in plan.same_task_priority:
            lines.append(f"  - {pair}")
    else:
        lines.append("  - none")

    # Cross-task caution zone
    lines.append("")
    lines.append("  Cross-task caution zone")
    lines.append("  " + "-" * 40)
    if plan.cross_task_caution:
        for entry in plan.cross_task_caution:
            lines.append(f"  - {entry}")
        if plan.same_task_priority:
            lines.append("  - do not prioritize these pairs for casual merge exploration")
    else:
        lines.append("  - none")

    # Summary
    lines.append("")
    lines.append("  Summary")
    lines.append("  " + "-" * 40)
    lines.append(f"  {plan.summary_line}")

    lines.append("")

    return "\n".join(lines)
