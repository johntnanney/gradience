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
    evidence_tier_counts: dict[str, int] = field(default_factory=dict)
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
            "evidence_tier_counts": dict(self.evidence_tier_counts),
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

        # --- evidence_tier_counts (additive, backfill to {}) ---
        raw_evidence = d.get("evidence_tier_counts", {})
        if not isinstance(raw_evidence, dict):
            raw_evidence = {}
        evidence_tier_counts: dict[str, int] = {}
        for k, v in raw_evidence.items():
            if isinstance(v, int):
                evidence_tier_counts[k] = v

        return cls(
            sources=d["sources"],
            adapter_status_counts=d["adapter_status_counts"],
            adapter_flag_counts=d["adapter_flag_counts"],
            pair_risk_counts=d["pair_risk_counts"],
            recommended_strategy_counts=d["recommended_strategy_counts"],
            dominant_issue_counts=d["dominant_issue_counts"],
            strict_qa_block_candidates=sqbc,
            evidence_tier_counts=evidence_tier_counts,
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
    evidence_counter: Counter[str] = Counter()
    for artifact in qa_artifacts:
        status_counter[artifact.status.value] += 1
        for flag in artifact.structural_flags:
            flag_counter[flag] += 1
        # Evidence tier: reported (eval available + eligible/uncertain),
        # uncertain (eval available + flagged_weak), missing (no eval).
        eval_avail = getattr(artifact, "eval_available", False)
        status_val = artifact.status.value if hasattr(artifact.status, "value") else str(artifact.status)
        if not eval_avail:
            evidence_counter["behavioral_missing"] += 1
        elif status_val in ("eligible", "uncertain"):
            evidence_counter["behavioral_reported"] += 1
        else:
            evidence_counter["behavioral_weak"] += 1

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
        evidence_tier_counts=dict(evidence_counter),
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


def format_inventory_summary(
    summary: InventorySummary,
    policy_summary: dict[str, str] | None = None,
) -> str:
    """Format an :class:`InventorySummary` as clean, human-readable text.

    Output uses standardized blocks in a fixed order:

    1. INVENTORY OVERVIEW — headline and counts
    2. SOURCE QA SNAPSHOT — provenance/trust and eligibility
    3. STRUCTURAL DETAIL — pair risk, strategies, issues
    4. INVENTORY POLICY SUMMARY — type, driver, posture, constraint (if provided)
    5. INTERPRETATION — plain-language guidance

    Parameters
    ----------
    summary
        The inventory summary to format.
    policy_summary
        Optional pre-computed policy summary dict from
        :func:`derive_inventory_policy_summary`.  When provided, the
        INVENTORY POLICY SUMMARY block is included.
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
    weak_count = status_counts.get("flagged_weak", 0)
    unknown_count = status_counts.get("unknown_no_behavioral_eval", 0)

    if any(v > 0 for v in status_counts.values()):
        lines.append("")
        lines.append("  SOURCE QA SNAPSHOT")
        lines.append("  " + "-" * 40)

        # Eligibility status
        lines.append("  Eligibility:")
        for key in sorted(status_counts):
            if status_counts[key] > 0:
                lines.append(f"    {key}:  {status_counts[key]}")

        # Evidence tier (from evidence_tier_counts if available, else derive from status)
        evidence = summary.evidence_tier_counts or {}
        reported = evidence.get("behavioral_reported", 0)
        ev_weak = evidence.get("behavioral_weak", 0)
        missing = evidence.get("behavioral_missing", 0)
        total_ev = reported + ev_weak + missing

        if total_ev > 0:
            lines.append("")
            lines.append("  Evidence tier:")
            if reported > 0:
                lines.append(f"    behavioral_reported:  {reported}")
            if ev_weak > 0:
                lines.append(f"    behavioral_weak:     {ev_weak}")
            if missing > 0:
                lines.append(f"    behavioral_missing:  {missing}")
            lines.append("")
            lines.append("  Note: behavioral scores are user-reported; Gradience does not")
            lines.append("  independently verify claimed evaluation results.")
        elif weak_count + unknown_count > 0:
            # Fallback for summaries built without evidence_tier_counts
            lines.append("")
            lines.append("  Note: behavioral scores are user-reported; Gradience does not")
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
        nonzero = [(k, counts[k]) for k in sorted(counts) if counts[k] > 0]
        if len(nonzero) <= 3:
            items = ", ".join(f"{k}: {v}" for k, v in nonzero)
            lines.append(f"  {section_label}: {items}")
        else:
            lines.append(f"  {section_label}:")
            for k, v in nonzero:
                lines.append(f"    {k}:  {v}")

    # ---------------------------------------------------------------
    # 4. INVENTORY POLICY SUMMARY (if provided)
    # ---------------------------------------------------------------
    if policy_summary is not None:
        lines.append(format_policy_summary(policy_summary))

    # ---------------------------------------------------------------
    # 5. INTERPRETATION
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
# Inventory policy summary — derived presentation layer
# ---------------------------------------------------------------------------

# Frozen constraint wording table.  These are the ONLY permitted strings.
# Keyed on (dominant_driver, inventory_type) with fallbacks.
_CONSTRAINT_TABLE: dict[tuple[str, str], str] = {
    ("source_qa", "mixed_quality"): (
        "Source QA is the binding constraint; resolve weak evidence before exploring merges."
    ),
    ("source_qa", "weak_evidence"): (
        "Source QA is the binding constraint; resolve weak evidence before exploring merges."
    ),
    ("task_boundary", "mixed_task"): (
        "Task boundary is the binding constraint; same-task pairs are the credible exploration space."
    ),
    ("task_boundary", "cross_task"): (
        "Task boundary is the binding constraint; same-task pairs are the credible exploration space."
    ),
    ("structural_risk", "mixed_task"): (
        "Structural risk dominates retained pairs; audit individual reports before merging."
    ),
    ("structural_risk", "same_task"): (
        "Structural risk dominates retained pairs; audit individual reports before merging."
    ),
    ("none", "same_task"): ("This same-task inventory is mostly confirmatory; low-risk merges can proceed."),
    ("none", "empty"): ("No pair reports available; run pairwise merge audits to populate this inventory."),
}

# Fallback for (dominant_driver, inventory_type) combos not explicitly listed.
_CONSTRAINT_FALLBACK: dict[str, str] = {
    "source_qa": "Source QA is the binding constraint; resolve weak evidence before exploring merges.",
    "task_boundary": "Task boundary is the binding constraint; same-task pairs are the credible exploration space.",
    "structural_risk": "Structural risk dominates retained pairs; audit individual reports before merging.",
    "none": "No single driver dominates; review individual pair reports for case-by-case guidance.",
}


def derive_inventory_policy_summary(
    summary: InventorySummary,
    action_plan: InventoryActionPlan,
) -> dict[str, str]:
    """Derive a four-component policy summary from existing stable signals.

    Pure derivation — no new scoring, no free-text, no access to raw
    artifacts.  Every output value comes from a controlled vocabulary.

    Parameters
    ----------
    summary
        The inventory summary (counts and distributions).
    action_plan
        The action plan (candidate-space reduction).

    Returns
    -------
    dict with keys: inventory_type, dominant_driver, exploration_posture, constraint.
    """
    # --- Count ingredients ---
    n_reports = summary.sources.get("merge_report_count", 0)
    n_qa = summary.sources.get("qa_artifact_count", 0)
    status_counts = summary.adapter_status_counts or {}
    weak_count = status_counts.get("flagged_weak", 0)
    unknown_count = status_counts.get("unknown_no_behavioral_eval", 0)
    weak_total = weak_count + unknown_count
    risk_counts = summary.pair_risk_counts or {}
    high_risk = risk_counts.get("high", 0)
    total_pairs = action_plan.total_pairs
    retained = action_plan.retained_count
    cross_task = action_plan.cross_task_count

    # --- inventory_type ---
    if n_reports == 0:
        inventory_type = "empty"
    elif n_qa > 0 and weak_total >= n_qa:
        inventory_type = "weak_evidence"
    elif weak_total > 0:
        inventory_type = "mixed_quality"
    elif cross_task == 0:
        inventory_type = "same_task"
    elif retained == 0:
        inventory_type = "cross_task"
    else:
        inventory_type = "mixed_task"

    # --- dominant_driver ---
    if weak_total > 0:
        dominant_driver = "source_qa"
    elif cross_task > 0:
        dominant_driver = "task_boundary"
    elif retained > 0 and high_risk > retained // 2:
        dominant_driver = "structural_risk"
    else:
        dominant_driver = "none"

    # --- exploration_posture ---
    if inventory_type == "same_task" and dominant_driver == "none":
        exploration_posture = "confirmatory"
    elif total_pairs == 0 or (total_pairs > 0 and retained / total_pairs < 0.25):
        exploration_posture = "narrow"
    elif retained / total_pairs < 0.75:
        exploration_posture = "moderate"
    else:
        exploration_posture = "broad"

    # --- constraint (frozen wording lookup) ---
    constraint = _CONSTRAINT_TABLE.get(
        (dominant_driver, inventory_type),
        _CONSTRAINT_FALLBACK.get(dominant_driver, _CONSTRAINT_FALLBACK["none"]),
    )

    return {
        "inventory_type": inventory_type,
        "dominant_driver": dominant_driver,
        "exploration_posture": exploration_posture,
        "constraint": constraint,
    }


def format_policy_summary(policy_summary: dict[str, str]) -> str:
    """Format a policy summary dict as a human-readable terminal block.

    Returns a multi-line string suitable for insertion into the
    inventory summary output.
    """
    lines: list[str] = []
    lines.append("")
    lines.append("  INVENTORY POLICY SUMMARY")
    lines.append("  " + "-" * 40)
    lines.append(f"  Type:        {policy_summary['inventory_type']}")
    lines.append(f"  Driver:      {policy_summary['dominant_driver']}")
    lines.append(f"  Posture:     {policy_summary['exploration_posture']}")

    # Wrap constraint at ~60 chars for readability
    constraint = policy_summary["constraint"]
    if len(constraint) <= 55:
        lines.append(f"  Constraint:  {constraint}")
    else:
        # Split at first semicolon or at word boundary near 55 chars
        split_at = constraint.find(";")
        if 0 < split_at < len(constraint) - 1:
            lines.append(f"  Constraint:  {constraint[: split_at + 1]}")
            lines.append(f"               {constraint[split_at + 2 :]}")
        else:
            lines.append(f"  Constraint:  {constraint}")

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
    cross_task_count: int = 0
    behavioral_evidence_count: int = 0
    total_source_count: int = 0
    # Per-pair detail: list of (pair_label, pair_risk, recommended_strategy)
    retained_pair_detail: tuple[tuple[str, str, str], ...] = ()
    # Near-miss: same-task pairs with clean structure but excluded by weak/uncertain source
    near_miss_candidates: tuple[str, ...] = ()
    near_miss_detail: tuple[tuple[str, str, str, str], ...] = ()  # (label, risk, strategy, reason)


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
    same_task_detail: list[tuple[str, str, str]] = []  # (label, risk, strategy)
    cross_task_pairs: list[str] = []
    cross_task_regions: set[str] = set()
    near_miss_pairs: list[str] = []
    near_miss_detail: list[tuple[str, str, str, str]] = []  # (label, risk, strategy, reason)

    # Structural risk levels considered "clean enough" for near-miss
    _NEAR_MISS_CLEAN_RISKS = frozenset({"low", "medium"})

    for report in merge_reports:
        a_path = report.adapter_a.path if hasattr(report.adapter_a, "path") else ""
        b_path = report.adapter_b.path if hasattr(report.adapter_b, "path") else ""
        a_name = PurePosixPath(a_path).name if a_path else "?"
        b_name = PurePosixPath(b_path).name if b_path else "?"
        pair_label = f"{a_name} × {b_name}"

        has_advisory = report.task_relationship_advisory is not None

        # Check if either source is weak/uncertain
        a_elig = report.adapter_a.eligibility_status
        b_elig = report.adapter_b.eligibility_status
        a_weak = a_elig in _WEAK_STATUSES if a_elig else False
        b_weak = b_elig in _WEAK_STATUSES if b_elig else False
        has_weak = a_weak or b_weak

        pair_risk = getattr(report, "pair_risk", "unknown")
        rec_strategy = getattr(report, "recommended_strategy", "unknown")

        if has_weak:
            # Near-miss: same-task, structurally clean, but one source is weak
            if not has_advisory and pair_risk in _NEAR_MISS_CLEAN_RISKS:
                # Build reason string
                weak_names = []
                if a_weak:
                    weak_names.append(a_name)
                if b_weak:
                    weak_names.append(b_name)
                reason = f"{', '.join(weak_names)} {'is' if len(weak_names) == 1 else 'are'} evidence-constrained"
                near_miss_pairs.append(pair_label)
                near_miss_detail.append((pair_label, pair_risk, rec_strategy, reason))
            continue  # Skip to next pair — not retained

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
            same_task_detail.append((pair_label, pair_risk, rec_strategy))

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

    # --- Provenance ---
    behavioral_count = sum(1 for qa in qa_artifacts if getattr(qa, "eval_available", False))

    return InventoryActionPlan(
        exclude=tuple(exclude_names),
        same_task_priority=tuple(same_task_pairs),
        cross_task_caution=tuple(caution_entries),
        evaluate_first=tuple(evaluate_first),
        summary_line=summary_line,
        total_pairs=total_pairs,
        retained_count=retained,
        cross_task_count=len(cross_task_pairs),
        behavioral_evidence_count=behavioral_count,
        total_source_count=len(qa_artifacts),
        retained_pair_detail=tuple(same_task_detail),
        near_miss_candidates=tuple(near_miss_pairs),
        near_miss_detail=tuple(near_miss_detail),
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
    if plan.cross_task_count > 0:
        lines.append(f"  Cross-task excluded: {plan.cross_task_count}")
    lines.append("")
    lines.append("  Evaluate first:")
    if plan.evaluate_first:
        # Show pair-risk and strategy when detail is available
        detail_map = {d[0]: (d[1], d[2]) for d in plan.retained_pair_detail}
        for pair in plan.evaluate_first:
            if pair in detail_map:
                risk, strategy = detail_map[pair]
                lines.append(f"    - {pair}  ({risk} risk, {strategy})")
            else:
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
        detail_map = {d[0]: (d[1], d[2]) for d in plan.retained_pair_detail}
        for pair in plan.same_task_priority:
            if pair in detail_map:
                risk, strategy = detail_map[pair]
                lines.append(f"  - {pair}  ({risk} risk, {strategy})")
            else:
                lines.append(f"  - {pair}")
    else:
        lines.append("  - none")

    # Near-miss candidates
    if plan.near_miss_candidates:
        lines.append("")
        lines.append("  Near-miss candidates")
        lines.append("  " + "-" * 40)
        lines.append("  Structurally plausible, evidence-constrained.")
        lines.append("  Optional if evaluation budget allows.")
        nm_detail_map = {d[0]: (d[1], d[2], d[3]) for d in plan.near_miss_detail}
        for pair in plan.near_miss_candidates:
            if pair in nm_detail_map:
                risk, strategy, reason = nm_detail_map[pair]
                lines.append(f"  - {pair}  ({risk} risk, {strategy} — {reason})")
            else:
                lines.append(f"  - {pair}")

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

    # Provenance
    if plan.total_source_count > 0:
        lines.append("")
        lines.append("  Provenance")
        lines.append("  " + "-" * 40)
        lines.append(f"  Sources with behavioral evidence: {plan.behavioral_evidence_count}/{plan.total_source_count}")

    # Summary
    lines.append("")
    lines.append("  Summary")
    lines.append("  " + "-" * 40)
    lines.append(f"  {plan.summary_line}")

    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Large-inventory ergonomics — region-grouped presentation layer
# ---------------------------------------------------------------------------

LARGE_INVENTORY_ADAPTER_THRESHOLD = 8
LARGE_INVENTORY_PAIR_THRESHOLD = 20


def is_large_inventory(adapter_count: int, pair_count: int) -> bool:
    """Return True when large-inventory rendering should activate."""
    return adapter_count >= LARGE_INVENTORY_ADAPTER_THRESHOLD or pair_count >= LARGE_INVENTORY_PAIR_THRESHOLD


def derive_region_summary(
    action_plan: InventoryActionPlan,
    qa_artifacts: list[Any],
) -> dict[str, Any]:
    """Derive a region-grouped summary from existing stable signals.

    Returns a compact dict suitable for both human-readable rendering and
    JSON serialisation.  No new analysis — grouping is by task label only.
    """
    adapter_count = action_plan.total_source_count
    pair_count = action_plan.total_pairs
    enabled = is_large_inventory(adapter_count, pair_count)

    if not enabled:
        return {
            "enabled": False,
            "threshold_trigger": None,
            "region_counts": {
                "same_task_safe_regions": 0,
                "cross_task_caution_regions": 0,
                "weak_evidence_regions": 0,
                "priority_evaluation_regions": 0,
            },
            "candidate_space_map": [],
        }

    threshold_trigger = "8_adapters" if adapter_count >= LARGE_INVENTORY_ADAPTER_THRESHOLD else "20_pairs"

    # --- Build adapter → task-label map ---
    adapter_task: dict[str, str] = {}
    for qa in qa_artifacts:
        name = qa.adapter_name if hasattr(qa, "adapter_name") else getattr(qa, "name", "?")
        eval_ds = getattr(qa, "eval_dataset", None) or ""
        task = eval_ds.replace("_dev", "").replace("_test", "").upper().replace("SST2", "SST-2")
        if task:
            adapter_task[name] = task

    # --- Group same-task pairs by task label ---
    same_task_groups: dict[str, list[str]] = {}

    for pair_label in action_plan.same_task_priority:
        # Extract adapter names from "adapter_a × adapter_b"
        parts = pair_label.split(" × ", 1)
        if len(parts) == 2:
            a_name, b_name = parts
            task_a = adapter_task.get(a_name, "")
            task_b = adapter_task.get(b_name, "")
            # Same-task → both should have the same task label
            task_label = task_a or task_b or "unknown"
        else:
            task_label = "unknown"
        same_task_groups.setdefault(task_label, []).append(pair_label)

    # --- Cross-task caution regions (already grouped in action plan) ---
    cross_task_regions: list[str] = []
    for entry in action_plan.cross_task_caution:
        # Strip trailing " region" if present for cleaner labels
        region_label = entry.removesuffix(" region").strip() if entry.endswith(" region") else entry
        cross_task_regions.append(region_label)

    # --- Evaluate-first set for priority tagging ---
    evaluate_first_set = frozenset(action_plan.evaluate_first)

    # --- Build candidate-space map ---
    candidate_map: list[dict[str, Any]] = []

    # Priority evaluation regions (same-task groups with evaluate_first overlap)
    priority_regions: set[str] = set()
    for task_label, pairs in sorted(same_task_groups.items()):
        if any(p in evaluate_first_set for p in pairs):
            priority_regions.add(task_label)

    # Same-task safe regions
    for task_label in sorted(same_task_groups):
        pairs = same_task_groups[task_label]
        is_priority = task_label in priority_regions
        candidate_map.append(
            {
                "region": task_label,
                "type": "same_task_safe_region",
                "pair_count": len(pairs),
                "status": "evaluate_first" if is_priority else "same_task_safe",
                "pairs": pairs,
            }
        )

    # Cross-task caution regions
    n_ct_regions = len(cross_task_regions)
    ct_total = action_plan.cross_task_count
    for region_label in sorted(cross_task_regions):
        est_count = ct_total // n_ct_regions if n_ct_regions > 0 else 0
        candidate_map.append(
            {
                "region": region_label,
                "type": "cross_task_caution_region",
                "pair_count": est_count,
                "status": "do_not_explore_casually",
                "pairs": [],
            }
        )

    # Weak-evidence region
    weak_count = len(action_plan.exclude)
    if weak_count > 0:
        weak_pair_count = action_plan.total_pairs - action_plan.retained_count - action_plan.cross_task_count
        if weak_pair_count < 0:
            weak_pair_count = 0
        candidate_map.append(
            {
                "region": "(weak evidence)",
                "type": "weak_evidence_region",
                "pair_count": weak_pair_count,
                "status": "excluded",
                "pairs": [],
            }
        )

    # --- Region counts ---
    n_same_task = len(same_task_groups)
    n_cross_task = n_ct_regions
    n_weak = 1 if weak_count > 0 else 0
    n_priority = len(priority_regions)

    return {
        "enabled": True,
        "threshold_trigger": threshold_trigger,
        "region_counts": {
            "same_task_safe_regions": n_same_task,
            "cross_task_caution_regions": n_cross_task,
            "weak_evidence_regions": n_weak,
            "priority_evaluation_regions": n_priority,
        },
        "candidate_space_map": candidate_map,
    }


_REGION_INTERPRETATION = (
    "This inventory is best understood as a small number of grouped regions rather than a flat list of pairs."
)


def format_region_summary(region_summary: dict[str, Any]) -> str:
    """Format the LARGE INVENTORY REGION SUMMARY block for terminal/markdown."""
    if not region_summary.get("enabled"):
        return ""

    lines: list[str] = []
    lines.append("")
    lines.append("  LARGE INVENTORY REGION SUMMARY")
    lines.append("  " + "=" * 60)
    lines.append("")

    rc = region_summary["region_counts"]
    lines.append(f"  Same-task safe regions:         {rc['same_task_safe_regions']}")
    lines.append(f"  Cross-task caution regions:     {rc['cross_task_caution_regions']}")
    lines.append(f"  Weak-evidence regions:          {rc['weak_evidence_regions']}")
    lines.append(f"  Priority evaluation regions:    {rc['priority_evaluation_regions']}")
    lines.append("")
    lines.append(f"  {_REGION_INTERPRETATION}")
    lines.append("")

    return "\n".join(lines)


# Status label display map
_STATUS_DISPLAY: dict[str, str] = {
    "evaluate_first": "evaluate first",
    "same_task_safe": "same-task safe",
    "do_not_explore_casually": "do not explore casually",
    "excluded": "excluded",
}

# Type label display map
_TYPE_DISPLAY: dict[str, str] = {
    "same_task_safe_region": "same-task safe",
    "cross_task_caution_region": "cross-task caution",
    "weak_evidence_region": "weak-evidence",
    "priority_evaluation_region": "priority evaluation",
}


def format_candidate_space_map(region_summary: dict[str, Any]) -> str:
    """Format the CANDIDATE-SPACE MAP as a compact markdown table."""
    if not region_summary.get("enabled"):
        return ""

    csmap = region_summary["candidate_space_map"]
    if not csmap:
        return ""

    lines: list[str] = []
    lines.append("")
    lines.append("  CANDIDATE-SPACE MAP")
    lines.append("  " + "=" * 60)
    lines.append("")

    # Compute column widths
    region_w = max(len(r["region"]) for r in csmap)
    region_w = max(region_w, 6)  # minimum for "Region"
    type_w = max(len(_TYPE_DISPLAY.get(r["type"], r["type"])) for r in csmap)
    type_w = max(type_w, 4)  # minimum for "Type"
    status_w = max(len(_STATUS_DISPLAY.get(r["status"], r["status"])) for r in csmap)
    status_w = max(status_w, 6)  # minimum for "Status"

    header = f"  | {'Region':<{region_w}} | {'Type':<{type_w}} | {'Pairs':>5} | {'Status':<{status_w}} |"
    sep = f"  |{'-' * (region_w + 2)}|{'-' * (type_w + 2)}|{'-' * 7}|{'-' * (status_w + 2)}|"
    lines.append(header)
    lines.append(sep)

    for entry in csmap:
        region = entry["region"]
        rtype = _TYPE_DISPLAY.get(entry["type"], entry["type"])
        pairs = entry["pair_count"]
        status = _STATUS_DISPLAY.get(entry["status"], entry["status"])
        lines.append(f"  | {region:<{region_w}} | {rtype:<{type_w}} | {pairs:>5} | {status:<{status_w}} |")

    lines.append("")

    return "\n".join(lines)


def format_region_candidate_table(region_summary: dict[str, Any]) -> str:
    """Format the region-aware reduced-candidate table."""
    if not region_summary.get("enabled"):
        return ""

    csmap = region_summary["candidate_space_map"]
    # Only show regions with evaluate_first status and actual pairs
    priority_regions = [r for r in csmap if r["status"] == "evaluate_first" and r["pairs"]]
    if not priority_regions:
        return ""

    lines: list[str] = []
    lines.append("")
    lines.append("  EVALUATE FIRST (by region)")
    lines.append("  " + "-" * 40)

    for r in priority_regions:
        lines.append(f"  {r['region']}:")
        for pair in r["pairs"]:
            lines.append(f"    - {pair}")

    lines.append("")

    return "\n".join(lines)
