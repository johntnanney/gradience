"""Preflight run bundle — packaging layer for inventory preflight results.

Assembles existing stable outputs (QA, pair reports, inventory summary,
neighborhoods, action plan) into a standard bundle with:
- preflight_summary.md
- inventory_action_plan.md
- preflight_summary.json
- run_manifest.json
- compare_to_previous.md (if prior run exists)
- review_packet.md
- review_packet.json

No new analysis logic. Presentation and packaging only.
"""

from __future__ import annotations

import contextlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _short_name(path: str) -> str:
    """Extract adapter short name from path."""
    return Path(path).name


def build_preflight_summary_json(
    *,
    inventory_id: str,
    run_id: str,
    qa_artifacts: list[Any],
    merge_reports: list[Any],
    action_plan: Any,
    policy_summary: dict[str, str] | None = None,
    region_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build machine-readable preflight summary from existing outputs."""
    excluded = [e for e in action_plan.exclude] if action_plan.exclude else []
    same_task = list(action_plan.same_task_priority)
    cross_task = list(action_plan.cross_task_caution)
    evaluate_first = list(action_plan.evaluate_first)

    # QA summary
    qa_summary: dict[str, int] = {}
    for qa in qa_artifacts:
        status = qa.status.value if hasattr(qa.status, "value") else str(qa.status)
        qa_summary[status] = qa_summary.get(status, 0) + 1

    # Advisory count
    advisory_count = sum(1 for r in merge_reports if r.task_relationship_advisory is not None)

    # Evidence tier counts
    behavioral_evidence_count = getattr(action_plan, "behavioral_evidence_count", 0)
    total_source_count = getattr(action_plan, "total_source_count", 0)

    result: dict[str, Any] = {
        "inventory_id": inventory_id,
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "adapter_count": len(qa_artifacts),
        "pair_count": len(merge_reports),
        "excluded_sources": excluded,
        "same_task_priority_pairs": same_task,
        "cross_task_caution_regions": cross_task,
        "reduced_candidate_subset": evaluate_first,
        "near_miss_candidates": list(getattr(action_plan, "near_miss_candidates", ())),
        "retained_candidate_count": action_plan.retained_count,
        "reduction_ratio": round(1 - action_plan.retained_count / len(merge_reports), 3) if merge_reports else 0.0,
        "advisory_pair_count": advisory_count,
        "qa_summary": qa_summary,
        "behavioral_evidence_count": behavioral_evidence_count,
        "total_source_count": total_source_count,
        "summary_line": action_plan.summary_line,
    }
    if policy_summary is not None:
        result["inventory_policy_summary"] = policy_summary
    if region_summary is not None:
        result["large_inventory_region_summary"] = region_summary
    return result


def build_run_manifest(
    *,
    inventory_id: str,
    run_id: str,
    run_dir: Path,
    adapter_count: int,
    pair_count: int,
    advisory_pair_count: int,
    previous_run_id: str | None = None,
    base_model: str | None = None,
) -> dict[str, Any]:
    """Build run manifest metadata."""
    return {
        "inventory_id": inventory_id,
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "base_model": base_model,
        "adapter_count": adapter_count,
        "pair_count": pair_count,
        "qa_ran": (run_dir / "qa").is_dir() and any((run_dir / "qa").glob("*.json")),
        "pair_reports_ran": (run_dir / "pair_reports").is_dir() and any((run_dir / "pair_reports").glob("*.json")),
        "inventory_summary_ran": (run_dir / "inventory" / "inventory_summary.json").exists(),
        "neighborhoods_ran": (run_dir / "neighborhoods" / "neighborhoods.json").exists(),
        "advisory_pair_count": advisory_pair_count,
        "primary_summary_path": "preflight_summary.md",
        "action_plan_path": "inventory_action_plan.md",
        "previous_run_id": previous_run_id,
    }


def build_preflight_summary_md(
    *,
    inventory_id: str,
    run_id: str,
    qa_artifacts: list[Any],
    merge_reports: list[Any],
    action_plan: Any,
    previous_run_id: str | None = None,
    policy_summary: dict[str, str] | None = None,
    region_summary: dict[str, Any] | None = None,
) -> str:
    """Build human-readable preflight summary."""
    lines: list[str] = []

    lines.append(f"# Preflight Summary — {inventory_id}")
    lines.append("")
    lines.append(f"**Run:** {run_id}")
    lines.append(f"**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    lines.append("")

    # Source QA summary
    lines.append("## Source QA")
    lines.append("")
    qa_counts: dict[str, int] = {}
    for qa in qa_artifacts:
        status = qa.status.value if hasattr(qa.status, "value") else str(qa.status)
        qa_counts[status] = qa_counts.get(status, 0) + 1
    for status, count in sorted(qa_counts.items()):
        lines.append(f"- {status}: {count}")
    lines.append("")

    # Policy summary (if available)
    if policy_summary is not None:
        lines.append("## Inventory Policy Summary")
        lines.append("")
        lines.append(f"- **Type:** {policy_summary['inventory_type']}")
        lines.append(f"- **Driver:** {policy_summary['dominant_driver']}")
        lines.append(f"- **Posture:** {policy_summary['exploration_posture']}")
        lines.append(f"- **Constraint:** {policy_summary['constraint']}")
        lines.append("")

    # Region summary (large inventories only)
    if region_summary and region_summary.get("enabled"):
        lines.append("## Large Inventory Region Summary")
        lines.append("")
        rc = region_summary["region_counts"]
        lines.append(f"- Same-task safe regions: {rc['same_task_safe_regions']}")
        lines.append(f"- Cross-task caution regions: {rc['cross_task_caution_regions']}")
        lines.append(f"- Weak-evidence regions: {rc['weak_evidence_regions']}")
        lines.append(f"- Priority evaluation regions: {rc['priority_evaluation_regions']}")
        lines.append("")
        # Candidate-space map table
        csmap = region_summary.get("candidate_space_map", [])
        if csmap:
            lines.append("### Candidate-Space Map")
            lines.append("")
            lines.append("| Region | Type | Pairs | Status |")
            lines.append("|---|---|---:|---|")
            _sd = {
                "evaluate_first": "evaluate first",
                "same_task_safe": "same-task safe",
                "do_not_explore_casually": "do not explore casually",
                "excluded": "excluded",
            }
            _td = {
                "same_task_safe_region": "same-task safe",
                "cross_task_caution_region": "cross-task caution",
                "weak_evidence_region": "weak-evidence",
            }
            for entry in csmap:
                r = entry["region"]
                t = _td.get(entry["type"], entry["type"])
                p = entry["pair_count"]
                s = _sd.get(entry["status"], entry["status"])
                lines.append(f"| {r} | {t} | {p} | {s} |")
            lines.append("")

    # Task-boundary partition
    advisory_count = sum(1 for r in merge_reports if r.task_relationship_advisory is not None)
    silent_count = len(merge_reports) - advisory_count
    lines.append("## Task-boundary partition")
    lines.append("")
    lines.append(f"- Same-task pairs (advisory silent): {silent_count}")
    lines.append(f"- Cross-task pairs (advisory active): {advisory_count}")
    lines.append(f"- Total pairs: {len(merge_reports)}")
    lines.append("")

    # Reduced candidate set
    lines.append("## Reduced candidate set")
    lines.append("")
    if action_plan.evaluate_first:
        detail_map = {d[0]: (d[1], d[2]) for d in getattr(action_plan, "retained_pair_detail", ())}
        for pair in action_plan.evaluate_first:
            if pair in detail_map:
                risk, strategy = detail_map[pair]
                lines.append(f"- {pair}  ({risk} risk, {strategy})")
            else:
                lines.append(f"- {pair}")
    else:
        lines.append("- no clear priority candidates identified")
    lines.append("")

    # Near-miss candidates (between reduced set and summary)
    near_miss = getattr(action_plan, "near_miss_candidates", ())
    if near_miss:
        nm_detail_map = {d[0]: (d[1], d[2], d[3]) for d in getattr(action_plan, "near_miss_detail", ())}
        lines.append("## Near-miss candidates")
        lines.append("")
        lines.append("Structurally plausible, evidence-constrained. Optional if evaluation budget allows.")
        lines.append("")
        for pair in near_miss:
            if pair in nm_detail_map:
                risk, strategy, reason = nm_detail_map[pair]
                lines.append(f"- {pair}  ({risk} risk, {strategy} — {reason})")
            else:
                lines.append(f"- {pair}")
        lines.append("")

    lines.append(f"**{action_plan.summary_line}**")
    lines.append("")

    # Provenance
    bev = getattr(action_plan, "behavioral_evidence_count", 0)
    tsc = getattr(action_plan, "total_source_count", 0)
    if tsc > 0:
        lines.append("## Provenance")
        lines.append("")
        lines.append(f"Sources with behavioral evidence: {bev}/{tsc}")
        lines.append("")
        lines.append("*Behavioral scores are user-reported; Gradience does not independently")
        lines.append("verify claimed evaluation results.*")
        lines.append("")

    # Inventory action plan
    lines.append("## Inventory action plan")
    lines.append("")
    lines.append("See `inventory_action_plan.md` for the full structured plan.")
    lines.append("")

    # Artifacts
    lines.append("## Detailed artifacts")
    lines.append("")
    lines.append("- `qa/` — source QA artifacts")
    lines.append("- `pair_reports/` — pairwise merge reports")
    lines.append("- `inventory/` — inventory summary")
    lines.append("- `neighborhoods/` — neighborhood grouping")
    lines.append("")

    if previous_run_id:
        lines.append("## Previous-run comparison")
        lines.append("")
        lines.append(f"See `compare_to_previous.md` for changes from run `{previous_run_id}`.")
        lines.append("")

    return "\n".join(lines)


def build_action_plan_md(action_plan: Any) -> str:
    """Build standalone action plan markdown."""
    lines: list[str] = []
    lines.append("# Inventory Action Plan")
    lines.append("")

    detail_map = {d[0]: (d[1], d[2]) for d in getattr(action_plan, "retained_pair_detail", ())}

    # Exclude / deprioritize
    lines.append("## Exclude / deprioritize")
    lines.append("")
    if action_plan.exclude:
        for item in action_plan.exclude:
            lines.append(f"- {item}")
    else:
        lines.append("- none")
    lines.append("")

    # Same-task safe zone (with per-pair risk/strategy)
    lines.append("## Same-task safe zone")
    lines.append("")
    if action_plan.same_task_priority:
        for pair in action_plan.same_task_priority:
            if pair in detail_map:
                risk, strategy = detail_map[pair]
                lines.append(f"- {pair}  ({risk} risk, {strategy})")
            else:
                lines.append(f"- {pair}")
    else:
        lines.append("- none")
    lines.append("")

    # Near-miss candidates
    near_miss = getattr(action_plan, "near_miss_candidates", ())
    if near_miss:
        nm_detail_map = {d[0]: (d[1], d[2], d[3]) for d in getattr(action_plan, "near_miss_detail", ())}
        lines.append("## Near-miss candidates")
        lines.append("")
        lines.append("Structurally plausible, evidence-constrained. Optional if evaluation budget allows.")
        lines.append("")
        for pair in near_miss:
            if pair in nm_detail_map:
                risk, strategy, reason = nm_detail_map[pair]
                lines.append(f"- {pair}  ({risk} risk, {strategy} — {reason})")
            else:
                lines.append(f"- {pair}")
        lines.append("")

    # Cross-task caution
    lines.append("## Cross-task caution")
    lines.append("")
    if action_plan.cross_task_caution:
        for item in action_plan.cross_task_caution:
            lines.append(f"- {item}")
    else:
        lines.append("- none")
    lines.append("")

    # Evaluate first
    lines.append("## Evaluate first")
    lines.append("")
    if action_plan.evaluate_first:
        for pair in action_plan.evaluate_first:
            if pair in detail_map:
                risk, strategy = detail_map[pair]
                lines.append(f"- {pair}  ({risk} risk, {strategy})")
            else:
                lines.append(f"- {pair}")
    else:
        lines.append("- none")
    lines.append("")

    # Provenance
    bev = getattr(action_plan, "behavioral_evidence_count", 0)
    tsc = getattr(action_plan, "total_source_count", 0)
    if tsc > 0:
        lines.append("## Provenance")
        lines.append("")
        lines.append(f"Sources with behavioral evidence: {bev}/{tsc}")
        lines.append("")

    # Summary
    lines.append("## Summary")
    lines.append("")
    lines.append(action_plan.summary_line)
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Inventory drift summary — derived from preflight_summary.json pairs
# ---------------------------------------------------------------------------

# Frozen implication wording table.
_IMPLICATION_TABLE: dict[str, str] = {
    "current_run_materially_narrower": "Current run is materially narrower than the previous run.",
    "current_run_materially_broader": "Current run has a broader candidate set than the previous run.",
    "source_composition_changed": "Source composition changed; candidate subset is effectively similar.",
    "evidence_improved_posture_unchanged": "Evidence quality improved, but exploration posture is unchanged.",
    "evidence_degraded": "Evidence quality degraded; review source QA before proceeding.",
    "no_substantial_change": "No substantial preflight change.",
}


def _zone_change(prev: int, curr: int) -> str:
    """Return directional label for a zone/count comparison."""
    if curr > prev:
        return "expanded"
    elif curr < prev:
        return "shrunk"
    return "unchanged"


def _evidence_profile_change(prev_bev: int, prev_tsc: int, curr_bev: int, curr_tsc: int) -> str:
    """Return directional label for evidence profile shift."""
    if prev_tsc == 0 or curr_tsc == 0:
        return "unchanged"
    prev_ratio = prev_bev / prev_tsc
    curr_ratio = curr_bev / curr_tsc
    if curr_ratio > prev_ratio:
        return "improved"
    elif curr_ratio < prev_ratio:
        return "degraded"
    return "unchanged"


def derive_drift_summary(
    current_summary: dict[str, Any],
    previous_summary: dict[str, Any],
) -> dict[str, Any]:
    """Derive a compact drift summary from two preflight_summary.json dicts.

    Pure derivation — no new scoring, no free-text.  All labels come
    from a controlled vocabulary.

    Parameters
    ----------
    current_summary
        The current run's preflight_summary.json (parsed dict).
    previous_summary
        The previous run's preflight_summary.json (parsed dict).

    Returns
    -------
    dict matching the ``inventory_drift_summary`` schema.
    """
    prev_adapters = previous_summary.get("adapter_count", 0)
    curr_adapters = current_summary.get("adapter_count", 0)
    prev_pairs = previous_summary.get("pair_count", 0)
    curr_pairs = current_summary.get("pair_count", 0)
    prev_retained = previous_summary.get("retained_candidate_count", 0)
    curr_retained = current_summary.get("retained_candidate_count", 0)
    prev_advisory = previous_summary.get("advisory_pair_count", 0)
    curr_advisory = current_summary.get("advisory_pair_count", 0)
    prev_bev = previous_summary.get("behavioral_evidence_count", 0)
    curr_bev = current_summary.get("behavioral_evidence_count", 0)
    prev_tsc = previous_summary.get("total_source_count", 0)
    curr_tsc = current_summary.get("total_source_count", 0)

    # --- Status ---
    if curr_retained < prev_retained:
        status = "narrowing"
    elif curr_retained > prev_retained:
        status = "broadening"
    else:
        status = "stable"

    # --- Zone changes ---
    safe_zone_change = _zone_change(prev_retained, curr_retained)
    caution_zone_change = _zone_change(prev_advisory, curr_advisory)

    # --- Evidence profile ---
    evidence_change = _evidence_profile_change(prev_bev, prev_tsc, curr_bev, curr_tsc)

    # --- Policy change ---
    prev_policy = previous_summary.get("inventory_policy_summary")
    curr_policy = current_summary.get("inventory_policy_summary")
    policy_change: dict[str, bool] | None = None
    if prev_policy and curr_policy:
        policy_change = {
            "inventory_type_changed": prev_policy.get("inventory_type") != curr_policy.get("inventory_type"),
            "dominant_driver_changed": prev_policy.get("dominant_driver") != curr_policy.get("dominant_driver"),
            "exploration_posture_changed": (
                prev_policy.get("exploration_posture") != curr_policy.get("exploration_posture")
            ),
            "constraint_changed": prev_policy.get("constraint") != curr_policy.get("constraint"),
        }

    # --- Implication (priority order) ---
    prev_excluded = set(previous_summary.get("excluded_sources", []))
    curr_excluded = set(current_summary.get("excluded_sources", []))

    if curr_retained < prev_retained:
        implication = "current_run_materially_narrower"
    elif curr_retained > prev_retained:
        implication = "current_run_materially_broader"
    elif curr_excluded != prev_excluded:
        implication = "source_composition_changed"
    elif evidence_change == "improved":
        implication = "evidence_improved_posture_unchanged"
    elif evidence_change == "degraded":
        implication = "evidence_degraded"
    else:
        implication = "no_substantial_change"

    result: dict[str, Any] = {
        "status": status,
        "adapter_count_delta": curr_adapters - prev_adapters,
        "pair_count_delta": curr_pairs - prev_pairs,
        "retained_candidate_delta": curr_retained - prev_retained,
        "same_task_safe_zone_change": safe_zone_change,
        "cross_task_caution_zone_change": caution_zone_change,
        "evidence_profile_change": evidence_change,
        "implication": implication,
    }
    if policy_change is not None:
        result["policy_change"] = policy_change

    return result


def format_drift_summary_md(
    drift: dict[str, Any],
    current_summary: dict[str, Any],
    previous_summary: dict[str, Any],
) -> str:
    """Format a drift summary as a markdown section for compare_to_previous.md.

    Parameters
    ----------
    drift
        The drift summary dict from :func:`derive_drift_summary`.
    current_summary, previous_summary
        The preflight_summary.json dicts (for raw values in key-changes).
    """
    lines: list[str] = []
    lines.append("## History / Drift Summary")
    lines.append("")

    # Status
    lines.append(f"**Status:** {drift['status']}")
    lines.append("")

    # Key changes
    lines.append("**Key changes:**")
    prev_adapters = previous_summary.get("adapter_count", 0)
    curr_adapters = current_summary.get("adapter_count", 0)
    delta_a = drift["adapter_count_delta"]
    sign_a = f"+{delta_a}" if delta_a > 0 else str(delta_a)
    lines.append(f"- Adapters: {prev_adapters} → {curr_adapters} ({sign_a})")

    prev_pairs = previous_summary.get("pair_count", 0)
    curr_pairs = current_summary.get("pair_count", 0)
    delta_p = drift["pair_count_delta"]
    sign_p = f"+{delta_p}" if delta_p > 0 else str(delta_p)
    lines.append(f"- Pairs: {prev_pairs} → {curr_pairs} ({sign_p})")

    prev_retained = previous_summary.get("retained_candidate_count", 0)
    curr_retained = current_summary.get("retained_candidate_count", 0)
    delta_r = drift["retained_candidate_delta"]
    sign_r = f"+{delta_r}" if delta_r > 0 else str(delta_r)
    lines.append(f"- Reduced candidate set: {prev_retained} → {curr_retained} ({sign_r})")

    lines.append(f"- Same-task safe zone: {drift['same_task_safe_zone_change']}")
    lines.append(f"- Cross-task caution zone: {drift['cross_task_caution_zone_change']}")
    lines.append(f"- Evidence profile: {drift['evidence_profile_change']}")
    lines.append("")

    # Policy change
    policy_change = drift.get("policy_change")
    if policy_change is not None:
        any_changed = any(policy_change.values())
        if any_changed:
            lines.append("**Policy change:**")
            prev_policy = previous_summary.get("inventory_policy_summary", {})
            curr_policy = current_summary.get("inventory_policy_summary", {})
            if policy_change.get("dominant_driver_changed"):
                lines.append(
                    f"- Dominant driver changed: "
                    f"{prev_policy.get('dominant_driver', '?')} → {curr_policy.get('dominant_driver', '?')}"
                )
            if policy_change.get("inventory_type_changed"):
                lines.append(
                    f"- Inventory type changed: "
                    f"{prev_policy.get('inventory_type', '?')} → {curr_policy.get('inventory_type', '?')}"
                )
            if policy_change.get("exploration_posture_changed"):
                lines.append(
                    f"- Exploration posture changed: "
                    f"{prev_policy.get('exploration_posture', '?')} → "
                    f"{curr_policy.get('exploration_posture', '?')}"
                )
            if policy_change.get("constraint_changed"):
                lines.append("- Constraint wording changed")
        else:
            lines.append("**Policy change:** none")
        lines.append("")

    # Implication
    implication_text = _IMPLICATION_TABLE.get(drift["implication"], drift["implication"])
    lines.append(f"**Implication:** {implication_text}")
    lines.append("")

    return "\n".join(lines)


def build_comparison_md(
    *,
    current_summary: dict[str, Any],
    previous_summary: dict[str, Any],
) -> str:
    """Build practical comparison between current and previous run."""
    lines: list[str] = []

    prev_id = previous_summary.get("run_id", "unknown")
    curr_id = current_summary.get("run_id", "unknown")

    lines.append("# Preflight Comparison")
    lines.append("")
    lines.append(f"**Previous run:** {prev_id}")
    lines.append(f"**Current run:** {curr_id}")
    lines.append("")

    # Source changes
    lines.append("## Source changes")
    lines.append("")
    prev_adapters = previous_summary.get("adapter_count", 0)
    curr_adapters = current_summary.get("adapter_count", 0)
    if curr_adapters != prev_adapters:
        lines.append(f"- Adapter count: {prev_adapters} → {curr_adapters}")
    else:
        lines.append(f"- Adapter count: {curr_adapters} (unchanged)")

    prev_excluded = set(previous_summary.get("excluded_sources", []))
    curr_excluded = set(current_summary.get("excluded_sources", []))
    new_excluded = curr_excluded - prev_excluded
    removed_excluded = prev_excluded - curr_excluded
    if new_excluded:
        lines.append(f"- Newly excluded: {', '.join(sorted(new_excluded))}")
    if removed_excluded:
        lines.append(f"- No longer excluded: {', '.join(sorted(removed_excluded))}")
    lines.append("")

    # Candidate-set changes
    lines.append("## Candidate-set changes")
    lines.append("")
    prev_pairs = previous_summary.get("pair_count", 0)
    curr_pairs = current_summary.get("pair_count", 0)
    lines.append(f"- Total pairs: {prev_pairs} → {curr_pairs}")

    prev_retained = previous_summary.get("retained_candidate_count", 0)
    curr_retained = current_summary.get("retained_candidate_count", 0)
    lines.append(f"- Retained candidates: {prev_retained} → {curr_retained}")

    prev_advisory = previous_summary.get("advisory_pair_count", 0)
    curr_advisory = current_summary.get("advisory_pair_count", 0)
    lines.append(f"- Advisory-bearing pairs: {prev_advisory} → {curr_advisory}")
    lines.append("")

    # Action-plan changes
    lines.append("## Action-plan changes")
    lines.append("")
    prev_eval = set(previous_summary.get("reduced_candidate_subset", []))
    curr_eval = set(current_summary.get("reduced_candidate_subset", []))
    if prev_eval == curr_eval:
        lines.append("- Evaluate-first subset: unchanged")
    else:
        added = curr_eval - prev_eval
        removed = prev_eval - curr_eval
        if added:
            lines.append(f"- Added to evaluate-first: {', '.join(sorted(added))}")
        if removed:
            lines.append(f"- Removed from evaluate-first: {', '.join(sorted(removed))}")
    lines.append("")

    # Provenance changes
    prev_bev = previous_summary.get("behavioral_evidence_count", 0)
    curr_bev = current_summary.get("behavioral_evidence_count", 0)
    prev_tsc = previous_summary.get("total_source_count", 0)
    curr_tsc = current_summary.get("total_source_count", 0)
    if prev_tsc > 0 or curr_tsc > 0:
        lines.append("## Provenance changes")
        lines.append("")
        if prev_bev == curr_bev and prev_tsc == curr_tsc:
            lines.append(f"- Behavioral evidence: {curr_bev}/{curr_tsc} (unchanged)")
        else:
            lines.append(f"- Behavioral evidence: {prev_bev}/{prev_tsc} → {curr_bev}/{curr_tsc}")
        lines.append("")

    # Top-line interpretation (legacy section — kept for backward compat)
    lines.append("## Interpretation")
    lines.append("")
    if curr_retained < prev_retained:
        lines.append("Current run is materially narrower than the previous run.")
    elif curr_retained > prev_retained:
        lines.append("Current run has a broader candidate set than the previous run.")
    elif curr_excluded != prev_excluded:
        lines.append("Source composition changed; candidate subset is effectively similar.")
    else:
        lines.append("No substantial preflight change.")
    lines.append("")

    # History / Drift Summary (structured)
    drift = derive_drift_summary(current_summary, previous_summary)
    lines.append(format_drift_summary_md(drift, current_summary, previous_summary))

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Review packet — consolidated handoff artifact
# ---------------------------------------------------------------------------

_REVIEW_PACKET_SCHEMA = "gradience.review_packet/v1"


def build_review_packet_md(
    *,
    inventory_id: str,
    run_id: str,
    summary_json: dict[str, Any],
    action_plan: Any,
    policy_summary: dict[str, str] | None = None,
    drift_summary: dict[str, Any] | None = None,
    previous_summary: dict[str, Any] | None = None,
    region_summary: dict[str, Any] | None = None,
) -> str:
    """Build a consolidated review packet as a single markdown document.

    Assembles existing stable outputs into seven sections.  No new
    analysis logic — assembly and formatting only.
    """
    lines: list[str] = []

    # ---------------------------------------------------------------
    # 1. Header / run metadata
    # ---------------------------------------------------------------
    lines.append(f"# Review Packet — {inventory_id}")
    lines.append("")
    lines.append(f"**Run:** {run_id}")
    lines.append(f"**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    lines.append(f"**Adapters:** {summary_json.get('adapter_count', 0)}")
    lines.append(f"**Pairs:** {summary_json.get('pair_count', 0)}")
    retained = summary_json.get("retained_candidate_count", 0)
    lines.append(f"**Retained candidates:** {retained}")
    lines.append("")

    # ---------------------------------------------------------------
    # 2. Inventory Policy Summary
    # ---------------------------------------------------------------
    lines.append("## Inventory Policy Summary")
    lines.append("")
    if policy_summary:
        lines.append(f"- **Type:** {policy_summary['inventory_type']}")
        lines.append(f"- **Driver:** {policy_summary['dominant_driver']}")
        lines.append(f"- **Posture:** {policy_summary['exploration_posture']}")
        lines.append(f"- **Constraint:** {policy_summary['constraint']}")
    else:
        lines.append("*Not available for this run.*")
    lines.append("")

    # ---------------------------------------------------------------
    # 2b. Large Inventory Region Summary (if threshold met)
    # ---------------------------------------------------------------
    if region_summary and region_summary.get("enabled"):
        lines.append("## Large Inventory Region Summary")
        lines.append("")
        rc = region_summary["region_counts"]
        lines.append(f"- Same-task safe regions: {rc['same_task_safe_regions']}")
        lines.append(f"- Cross-task caution regions: {rc['cross_task_caution_regions']}")
        lines.append(f"- Weak-evidence regions: {rc['weak_evidence_regions']}")
        lines.append(f"- Priority evaluation regions: {rc['priority_evaluation_regions']}")
        lines.append("")
        csmap = region_summary.get("candidate_space_map", [])
        if csmap:
            lines.append("### Candidate-Space Map")
            lines.append("")
            lines.append("| Region | Type | Pairs | Status |")
            lines.append("|---|---|---:|---|")
            _sd = {
                "evaluate_first": "evaluate first",
                "same_task_safe": "same-task safe",
                "do_not_explore_casually": "do not explore casually",
                "excluded": "excluded",
            }
            _td = {
                "same_task_safe_region": "same-task safe",
                "cross_task_caution_region": "cross-task caution",
                "weak_evidence_region": "weak-evidence",
            }
            for entry in csmap:
                r = entry["region"]
                t = _td.get(entry["type"], entry["type"])
                p = entry["pair_count"]
                s = _sd.get(entry["status"], entry["status"])
                lines.append(f"| {r} | {t} | {p} | {s} |")
            lines.append("")

    # ---------------------------------------------------------------
    # 3. Source QA / Trust Snapshot
    # ---------------------------------------------------------------
    lines.append("## Source QA / Trust Snapshot")
    lines.append("")
    qa_summary = summary_json.get("qa_summary", {})
    if qa_summary:
        for status, count in sorted(qa_summary.items()):
            lines.append(f"- {status}: {count}")
    else:
        lines.append("- No source QA data available.")
    lines.append("")

    bev = summary_json.get("behavioral_evidence_count", 0)
    tsc = summary_json.get("total_source_count", 0)
    if tsc > 0:
        lines.append(f"**Evidence:** {bev}/{tsc} sources with behavioral evidence")
        lines.append("")
        lines.append("*Behavioral scores are user-reported; Gradience does not independently")
        lines.append("verify claimed evaluation results.*")
        lines.append("")

    excluded = summary_json.get("excluded_sources", [])
    if excluded:
        lines.append("**Excluded sources:**")
        for ex in excluded:
            lines.append(f"- {ex}")
        lines.append("")

    # ---------------------------------------------------------------
    # 4. Action Plan
    # ---------------------------------------------------------------
    lines.append("## Action Plan")
    lines.append("")
    lines.append(f"**Starting pairs:** {action_plan.total_pairs}")
    lines.append(f"**Retained candidates:** {action_plan.retained_count}")
    if action_plan.total_pairs > 0:
        pct = round(100 * (1 - action_plan.retained_count / action_plan.total_pairs))
        lines.append(f"**Reduction:** {pct}%")
    if action_plan.cross_task_count > 0:
        lines.append(f"**Cross-task excluded:** {action_plan.cross_task_count}")
    lines.append("")

    # Evaluate first
    lines.append("**Evaluate first:**")
    if action_plan.evaluate_first:
        detail_map = {d[0]: (d[1], d[2]) for d in getattr(action_plan, "retained_pair_detail", ())}
        for pair in action_plan.evaluate_first:
            if pair in detail_map:
                risk, strategy = detail_map[pair]
                lines.append(f"- {pair}  ({risk} risk, {strategy})")
            else:
                lines.append(f"- {pair}")
    else:
        lines.append("- No clear priority candidates identified.")
    lines.append("")

    # Same-task safe zone
    if action_plan.same_task_priority:
        lines.append("**Same-task safe zone:**")
        detail_map = {d[0]: (d[1], d[2]) for d in getattr(action_plan, "retained_pair_detail", ())}
        for pair in action_plan.same_task_priority:
            if pair in detail_map:
                risk, strategy = detail_map[pair]
                lines.append(f"- {pair}  ({risk} risk, {strategy})")
            else:
                lines.append(f"- {pair}")
        lines.append("")

    # Cross-task caution zone
    if action_plan.cross_task_caution:
        lines.append("**Cross-task caution zone:**")
        for entry in action_plan.cross_task_caution:
            lines.append(f"- {entry}")
        lines.append("")

    lines.append(f"**Summary:** {action_plan.summary_line}")
    lines.append("")

    # ---------------------------------------------------------------
    # 5. Drift Summary (if previous run exists)
    # ---------------------------------------------------------------
    if drift_summary:
        lines.append("## Drift Summary")
        lines.append("")
        lines.append(f"**Status:** {drift_summary['status']}")
        lines.append("")

        # Key deltas
        delta_r = drift_summary.get("retained_candidate_delta", 0)
        sign_r = f"+{delta_r}" if delta_r > 0 else str(delta_r)
        lines.append("**Key changes:**")
        delta_a = drift_summary.get("adapter_count_delta", 0)
        sign_a = f"+{delta_a}" if delta_a > 0 else str(delta_a)
        lines.append(f"- Adapters: {sign_a}")
        delta_p = drift_summary.get("pair_count_delta", 0)
        sign_p = f"+{delta_p}" if delta_p > 0 else str(delta_p)
        lines.append(f"- Pairs: {sign_p}")
        lines.append(f"- Retained candidates: {sign_r}")
        lines.append(f"- Same-task safe zone: {drift_summary.get('same_task_safe_zone_change', 'unchanged')}")
        lines.append(f"- Cross-task caution zone: {drift_summary.get('cross_task_caution_zone_change', 'unchanged')}")
        lines.append(f"- Evidence profile: {drift_summary.get('evidence_profile_change', 'unchanged')}")
        lines.append("")

        # Policy change
        policy_change = drift_summary.get("policy_change")
        if policy_change and any(policy_change.values()):
            lines.append("**Policy change:**")
            prev_policy = (previous_summary or {}).get("inventory_policy_summary", {})
            curr_policy = (summary_json or {}).get("inventory_policy_summary", {})
            if policy_change.get("dominant_driver_changed"):
                lines.append(
                    f"- Dominant driver: "
                    f"{prev_policy.get('dominant_driver', '?')} → {curr_policy.get('dominant_driver', '?')}"
                )
            if policy_change.get("exploration_posture_changed"):
                lines.append(
                    f"- Exploration posture: "
                    f"{prev_policy.get('exploration_posture', '?')} → "
                    f"{curr_policy.get('exploration_posture', '?')}"
                )
            lines.append("")

        # Implication
        impl_key = drift_summary.get("implication", "no_substantial_change")
        impl_text = _IMPLICATION_TABLE.get(impl_key, impl_key)
        lines.append(f"**Implication:** {impl_text}")
        lines.append("")

    # ---------------------------------------------------------------
    # 6. Compare to Previous (highlights, if applicable)
    # ---------------------------------------------------------------
    if previous_summary:
        prev_id = previous_summary.get("run_id", "unknown")
        lines.append("## Compare to Previous")
        lines.append("")
        lines.append(f"**Previous run:** {prev_id}")
        lines.append("")

        prev_retained = previous_summary.get("retained_candidate_count", 0)
        curr_retained_val = summary_json.get("retained_candidate_count", 0)
        lines.append(f"- Retained candidates: {prev_retained} → {curr_retained_val}")

        prev_advisory = previous_summary.get("advisory_pair_count", 0)
        curr_advisory = summary_json.get("advisory_pair_count", 0)
        if prev_advisory != curr_advisory:
            lines.append(f"- Advisory-bearing pairs: {prev_advisory} → {curr_advisory}")

        prev_bev = previous_summary.get("behavioral_evidence_count", 0)
        prev_tsc_val = previous_summary.get("total_source_count", 0)
        if prev_tsc_val > 0 or tsc > 0:
            lines.append(f"- Behavioral evidence: {prev_bev}/{prev_tsc_val} → {bev}/{tsc}")

        prev_excluded_set = set(previous_summary.get("excluded_sources", []))
        curr_excluded_set = set(excluded)
        new_excl = curr_excluded_set - prev_excluded_set
        restored = prev_excluded_set - curr_excluded_set
        if new_excl:
            lines.append(f"- Newly excluded: {', '.join(sorted(new_excl))}")
        if restored:
            lines.append(f"- Restored: {', '.join(sorted(restored))}")

        lines.append("")
        lines.append("See `compare_to_previous.md` for the full comparison.")
        lines.append("")

    # ---------------------------------------------------------------
    # 7. Artifact Links
    # ---------------------------------------------------------------
    lines.append("## Artifacts")
    lines.append("")
    lines.append("- `preflight_summary.json` — machine-readable preflight summary")
    lines.append("- `preflight_summary.md` — human-readable preflight summary")
    lines.append("- `inventory_action_plan.md` — structured action plan")
    lines.append("- `run_manifest.json` — run metadata")
    if previous_summary:
        lines.append("- `compare_to_previous.md` — run-to-run comparison")
    lines.append("- `qa/` — source QA artifacts")
    lines.append("- `pair_reports/` — pairwise merge reports")
    lines.append("- `inventory/` — inventory summary")
    lines.append("")

    return "\n".join(lines)


def build_review_packet_json(
    *,
    inventory_id: str,
    run_id: str,
    summary_json: dict[str, Any],
    action_plan: Any,
    policy_summary: dict[str, str] | None = None,
    drift_summary: dict[str, Any] | None = None,
    previous_run_id: str | None = None,
    region_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a compact JSON companion for the review packet.

    Assembly only — no new derivation.
    """
    sections: list[str] = ["header", "trust_snapshot", "action_plan"]
    if policy_summary:
        sections.insert(1, "policy_summary")
    if region_summary and region_summary.get("enabled"):
        # Insert after policy_summary if present, else after header
        idx = sections.index("policy_summary") + 1 if "policy_summary" in sections else 1
        sections.insert(idx, "region_summary")
    if drift_summary:
        sections.append("drift_summary")
    if previous_run_id:
        sections.append("compare_to_previous")
    sections.append("artifact_links")

    bev = summary_json.get("behavioral_evidence_count", 0)
    tsc = summary_json.get("total_source_count", 0)

    result: dict[str, Any] = {
        "schema": _REVIEW_PACKET_SCHEMA,
        "inventory_id": inventory_id,
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "sections_present": sections,
        "adapter_count": summary_json.get("adapter_count", 0),
        "pair_count": summary_json.get("pair_count", 0),
        "retained_candidate_count": summary_json.get("retained_candidate_count", 0),
        "trust_snapshot": {
            "qa_summary": summary_json.get("qa_summary", {}),
            "behavioral_evidence_count": bev,
            "total_source_count": tsc,
            "excluded_source_count": len(summary_json.get("excluded_sources", [])),
        },
        "action_plan_summary": {
            "total_pairs": action_plan.total_pairs,
            "retained_count": action_plan.retained_count,
            "cross_task_count": action_plan.cross_task_count,
            "evaluate_first_count": len(action_plan.evaluate_first),
            "summary_line": action_plan.summary_line,
        },
    }

    if policy_summary:
        result["policy_summary"] = policy_summary
    else:
        result["policy_summary"] = None

    if drift_summary:
        result["drift_summary"] = drift_summary
    else:
        result["drift_summary"] = None

    if region_summary and region_summary.get("enabled"):
        result["large_inventory_region_summary"] = region_summary
    else:
        result["large_inventory_region_summary"] = None

    result["previous_run_id"] = previous_run_id

    return result


def emit_run_bundle(
    *,
    inventory_id: str,
    run_id: str,
    run_dir: Path,
    qa_artifacts: list[Any],
    merge_reports: list[Any],
    action_plan: Any,
    base_model: str | None = None,
    previous_run_dir: Path | None = None,
    policy_summary: dict[str, str] | None = None,
    region_summary: dict[str, Any] | None = None,
) -> Path:
    """Emit a complete preflight run bundle into run_dir.

    Generates all standard top-level files. Does not move or copy
    the detailed artifacts (qa/, pair_reports/, etc.) which should
    already exist in run_dir.

    Returns the run_dir path.
    """
    run_dir.mkdir(parents=True, exist_ok=True)

    advisory_count = sum(1 for r in merge_reports if r.task_relationship_advisory is not None)

    # Previous run handling
    previous_run_id: str | None = None
    previous_summary: dict[str, Any] | None = None
    if previous_run_dir and (previous_run_dir / "preflight_summary.json").exists():
        with open(previous_run_dir / "preflight_summary.json") as f:
            previous_summary = json.load(f)
        previous_run_id = previous_summary.get("run_id")

    # 1. preflight_summary.json
    summary_json = build_preflight_summary_json(
        inventory_id=inventory_id,
        run_id=run_id,
        qa_artifacts=qa_artifacts,
        merge_reports=merge_reports,
        action_plan=action_plan,
        policy_summary=policy_summary,
        region_summary=region_summary,
    )

    # Compute drift summary if previous run exists
    if previous_summary:
        drift = derive_drift_summary(summary_json, previous_summary)
        summary_json["inventory_drift_summary"] = drift

    with open(run_dir / "preflight_summary.json", "w") as f:
        json.dump(summary_json, f, indent=2)

    # 2. run_manifest.json
    manifest = build_run_manifest(
        inventory_id=inventory_id,
        run_id=run_id,
        run_dir=run_dir,
        adapter_count=len(qa_artifacts),
        pair_count=len(merge_reports),
        advisory_pair_count=advisory_count,
        previous_run_id=previous_run_id,
        base_model=base_model,
    )
    with open(run_dir / "run_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    # 3. preflight_summary.md
    summary_md = build_preflight_summary_md(
        inventory_id=inventory_id,
        run_id=run_id,
        qa_artifacts=qa_artifacts,
        merge_reports=merge_reports,
        action_plan=action_plan,
        previous_run_id=previous_run_id,
        policy_summary=policy_summary,
        region_summary=region_summary,
    )
    (run_dir / "preflight_summary.md").write_text(summary_md)

    # 4. inventory_action_plan.md
    action_md = build_action_plan_md(action_plan)
    (run_dir / "inventory_action_plan.md").write_text(action_md)

    # 5. compare_to_previous.md (if applicable)
    if previous_summary:
        comparison_md = build_comparison_md(
            current_summary=summary_json,
            previous_summary=previous_summary,
        )
        (run_dir / "compare_to_previous.md").write_text(comparison_md)

    # 6. review_packet.md
    drift = summary_json.get("inventory_drift_summary")  # type: ignore[assignment]
    review_md = build_review_packet_md(
        inventory_id=inventory_id,
        run_id=run_id,
        summary_json=summary_json,
        action_plan=action_plan,
        policy_summary=policy_summary,
        drift_summary=drift,
        previous_summary=previous_summary,
        region_summary=region_summary,
    )
    (run_dir / "review_packet.md").write_text(review_md)

    # 7. review_packet.json
    review_json = build_review_packet_json(
        inventory_id=inventory_id,
        run_id=run_id,
        summary_json=summary_json,
        action_plan=action_plan,
        policy_summary=policy_summary,
        drift_summary=drift,
        previous_run_id=previous_run_id,
        region_summary=region_summary,
    )
    with open(run_dir / "review_packet.json", "w") as f:
        json.dump(review_json, f, indent=2)

    return run_dir


def update_latest_pointer(inventory_root: Path, run_dir: Path) -> None:
    """Update the latest/ symlink at the inventory root to point to the current run.

    Uses atomic replacement via a temporary symlink + ``os.replace()``
    to avoid a TOCTOU race between unlinking the old symlink and
    creating the new one.
    """
    import os
    import tempfile

    latest = inventory_root / "latest"
    target = run_dir.resolve()

    # Create a temporary symlink next to the target, then atomically
    # move it into place.  os.replace() is atomic on POSIX when source
    # and destination are on the same filesystem.
    fd, tmp_path = tempfile.mkstemp(dir=str(inventory_root), prefix=".latest_")
    os.close(fd)
    os.unlink(tmp_path)  # mkstemp creates a regular file; we need a symlink
    os.symlink(str(target), tmp_path)
    try:
        os.replace(tmp_path, str(latest))
    except OSError:
        # Clean up the temp symlink on failure
        with contextlib.suppress(OSError):
            os.unlink(tmp_path)
        raise
