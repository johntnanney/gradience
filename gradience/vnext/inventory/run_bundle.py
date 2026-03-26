"""Preflight run bundle — packaging layer for inventory preflight results.

Assembles existing stable outputs (QA, pair reports, inventory summary,
neighborhoods, action plan) into a standard bundle with:
- preflight_summary.md
- inventory_action_plan.md
- preflight_summary.json
- run_manifest.json
- compare_to_previous.md (if prior run exists)

No new analysis logic. Presentation and packaging only.
"""

from __future__ import annotations

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

    return {
        "inventory_id": inventory_id,
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "adapter_count": len(qa_artifacts),
        "pair_count": len(merge_reports),
        "excluded_sources": excluded,
        "same_task_priority_pairs": same_task,
        "cross_task_caution_regions": cross_task,
        "reduced_candidate_subset": evaluate_first,
        "retained_candidate_count": action_plan.retained_count,
        "reduction_ratio": round(1 - action_plan.retained_count / len(merge_reports), 3) if merge_reports else 0.0,
        "advisory_pair_count": advisory_count,
        "qa_summary": qa_summary,
        "summary_line": action_plan.summary_line,
    }


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
        for pair in action_plan.evaluate_first:
            lines.append(f"- {pair}")
    else:
        lines.append("- no clear priority candidates identified")
    lines.append("")
    lines.append(f"**{action_plan.summary_line}**")
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
    from gradience.vnext.inventory.summary import format_action_plan

    # Convert terminal format to clean markdown
    lines: list[str] = []
    lines.append("# Inventory Action Plan")
    lines.append("")

    sections = [
        ("Exclude / deprioritize", action_plan.exclude),
        ("Prioritize same-task region", action_plan.same_task_priority),
        ("Cross-task caution", action_plan.cross_task_caution),
        ("Evaluate first", action_plan.evaluate_first),
    ]

    for heading, items in sections:
        lines.append(f"## {heading}")
        lines.append("")
        if items:
            for item in items:
                lines.append(f"- {item}")
        else:
            lines.append("- none")
        lines.append("")

    lines.append("## Summary")
    lines.append("")
    lines.append(action_plan.summary_line)
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

    lines.append(f"# Preflight Comparison")
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

    # Top-line interpretation
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

    return "\n".join(lines)


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
    )
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

    return run_dir


def update_latest_pointer(inventory_root: Path, run_dir: Path) -> None:
    """Update the latest/ symlink at the inventory root to point to the current run."""
    latest = inventory_root / "latest"
    if latest.is_symlink() or latest.exists():
        latest.unlink()
    latest.symlink_to(run_dir.resolve(), target_is_directory=True)
