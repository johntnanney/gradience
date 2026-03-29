"""Batch preflight ergonomics — cross-run summary and discovery.

Reads existing preflight run bundles and aggregates them into a
single comparison view.  No new analysis logic; presentation only.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from gradience.vnext.inventory.run_bundle import derive_drift_summary

# ---------------------------------------------------------------------------
# Run discovery
# ---------------------------------------------------------------------------


def discover_previous_run(bundle_parent: Path) -> Path | None:
    """Find the most recent completed run in *bundle_parent*.

    Looks for a ``latest`` symlink first.  If absent, falls back to
    scanning for sibling directories that contain ``preflight_summary.json``.

    Returns ``None`` when no previous run is found (first run).
    """
    latest = bundle_parent / "latest"
    if latest.is_symlink() and (latest / "preflight_summary.json").exists():
        return latest.resolve()

    # Fallback: find most recent sibling with a preflight_summary.json
    candidates: list[tuple[float, Path]] = []
    if bundle_parent.is_dir():
        for d in bundle_parent.iterdir():
            pf = d / "preflight_summary.json"
            if d.is_dir() and pf.exists() and not d.name.startswith("."):
                candidates.append((pf.stat().st_mtime, d))
    if not candidates:
        return None
    candidates.sort(key=lambda t: t[0])
    return candidates[-1][1]


# ---------------------------------------------------------------------------
# Run summary collection
# ---------------------------------------------------------------------------


def collect_run_summaries(root_dir: Path) -> list[dict[str, Any]]:
    """Collect all ``preflight_summary.json`` files under *root_dir*.

    Returns a list of parsed dicts sorted by timestamp (oldest first).
    Skips directories that don't contain valid summary files.
    """
    summaries: list[dict[str, Any]] = []
    if not root_dir.is_dir():
        return summaries
    for d in root_dir.iterdir():
        pf = d / "preflight_summary.json"
        if d.is_dir() and pf.exists():
            try:
                with open(pf) as f:
                    data = json.load(f)
                if isinstance(data, dict) and "run_id" in data:
                    data["_source_dir"] = str(d)
                    summaries.append(data)
            except (json.JSONDecodeError, OSError):
                continue
    # Sort by timestamp if available, else by run_id
    summaries.sort(key=lambda s: s.get("timestamp", s.get("run_id", "")))
    return summaries


# ---------------------------------------------------------------------------
# Batch summary builder
# ---------------------------------------------------------------------------


_BATCH_COLUMNS = (
    "run_id",
    "adapter_count",
    "pair_count",
    "retained_candidate_count",
    "advisory_pair_count",
    "behavioral_evidence_count",
    "total_source_count",
    "reduction_ratio",
)


def build_batch_summary(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate multiple run summaries into a cross-run comparison.

    Returns a dict with:
    - ``runs``: list of per-run metric rows (with per-run drift)
    - ``run_count``: number of runs
    - ``trend``: high-level trend assessment
    - ``evidence_trend``: evidence quality trajectory
    - ``policy_evolution``: whether policy summary changed across runs
    """
    rows: list[dict[str, Any]] = []
    for i, s in enumerate(summaries):
        row: dict[str, Any] = {}
        for col in _BATCH_COLUMNS:
            row[col] = s.get(col)
        # Compute evidence ratio string
        bev = s.get("behavioral_evidence_count", 0)
        tsc = s.get("total_source_count", 0)
        row["evidence_ratio"] = f"{bev}/{tsc}" if tsc > 0 else "n/a"

        # Per-run drift (vs previous run in sequence)
        if i > 0:
            drift = derive_drift_summary(s, summaries[i - 1])
            row["drift_status"] = drift["status"]
            row["drift"] = drift
        else:
            row["drift_status"] = None
            row["drift"] = None

        rows.append(row)

    # Trend assessment (first vs last retained)
    if len(rows) < 2:
        trend = "insufficient data"
    else:
        first_retained = rows[0].get("retained_candidate_count", 0) or 0
        last_retained = rows[-1].get("retained_candidate_count", 0) or 0
        if last_retained < first_retained:
            trend = "narrowing"
        elif last_retained > first_retained:
            trend = "broadening"
        else:
            trend = "stable"

    # Evidence trend (first vs last evidence ratio)
    evidence_trend = "unchanged"
    if len(summaries) >= 2:
        first_bev = summaries[0].get("behavioral_evidence_count", 0)
        first_tsc = summaries[0].get("total_source_count", 0)
        last_bev = summaries[-1].get("behavioral_evidence_count", 0)
        last_tsc = summaries[-1].get("total_source_count", 0)
        if first_tsc > 0 and last_tsc > 0:
            first_ratio = first_bev / first_tsc
            last_ratio = last_bev / last_tsc
            if last_ratio > first_ratio:
                evidence_trend = "improved"
            elif last_ratio < first_ratio:
                evidence_trend = "degraded"

    # Policy evolution (did policy summary change across any consecutive pair?)
    policy_ever_changed = False
    if len(summaries) >= 2:
        for i in range(1, len(summaries)):
            prev_p = summaries[i - 1].get("inventory_policy_summary")
            curr_p = summaries[i].get("inventory_policy_summary")
            if prev_p and curr_p and prev_p != curr_p:
                policy_ever_changed = True
                break

    return {
        "schema": "gradience.batch_summary/v1",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "run_count": len(rows),
        "runs": rows,
        "trend": trend,
        "evidence_trend": evidence_trend,
        "policy_changed": policy_ever_changed,
    }


# ---------------------------------------------------------------------------
# Terminal formatter
# ---------------------------------------------------------------------------


def format_batch_summary(batch: dict[str, Any]) -> str:
    """Format a batch summary as a terminal-readable table."""
    runs = batch.get("runs", [])
    if not runs:
        return "  No runs found.\n"

    lines: list[str] = []
    lines.append("")
    lines.append("  BATCH PREFLIGHT SUMMARY")
    lines.append("  " + "=" * 60)
    trend_line = f"  Runs: {batch.get('run_count', 0)}    Trend: {batch.get('trend', 'unknown')}"
    evidence_trend = batch.get("evidence_trend")
    if evidence_trend and evidence_trend != "unchanged":
        trend_line += f"    Evidence: {evidence_trend}"
    policy_changed = batch.get("policy_changed", False)
    if policy_changed:
        trend_line += "    Policy: changed"
    lines.append(trend_line)
    lines.append("")

    # Table header
    header = (
        f"  {'Run':<20s} {'Adapters':>8s} {'Pairs':>6s} "
        f"{'Retained':>8s} {'Reduction':>9s} {'Evidence':>10s} {'Drift':>10s}"
    )
    lines.append(header)
    lines.append("  " + "-" * 75)

    for row in runs:
        run_id = str(row.get("run_id", "?"))[:18]
        adapters = row.get("adapter_count", "?")
        pairs = row.get("pair_count", "?")
        retained = row.get("retained_candidate_count", "?")
        ratio = row.get("reduction_ratio")
        reduction = f"{ratio:.0%}" if isinstance(ratio, (int, float)) else "?"
        evidence = row.get("evidence_ratio", "n/a")
        drift_status = row.get("drift_status") or "—"
        lines.append(
            f"  {run_id:<20s} {str(adapters):>8s} {str(pairs):>6s} "
            f"{str(retained):>8s} {reduction:>9s} {evidence:>10s} {drift_status:>10s}"
        )

    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Markdown + JSON emitter
# ---------------------------------------------------------------------------


def emit_batch_summary(
    *,
    batch: dict[str, Any],
    output_dir: Path,
) -> Path:
    """Write batch summary to markdown and JSON files in *output_dir*.

    Creates:
    - ``batch_summary.json`` — machine-readable
    - ``batch_summary.md`` — human-readable table

    Returns the output_dir path.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # JSON
    with open(output_dir / "batch_summary.json", "w") as f:
        json.dump(batch, f, indent=2)

    # Markdown
    runs = batch.get("runs", [])
    lines: list[str] = []
    lines.append("# Batch Preflight Summary")
    lines.append("")
    lines.append(f"**Runs:** {batch.get('run_count', 0)}")
    lines.append(f"**Trend:** {batch.get('trend', 'unknown')}")
    evidence_trend = batch.get("evidence_trend")
    if evidence_trend and evidence_trend != "unchanged":
        lines.append(f"**Evidence trend:** {evidence_trend}")
    if batch.get("policy_changed", False):
        lines.append("**Policy:** changed across runs")
    lines.append(f"**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    lines.append("")

    if runs:
        lines.append("| Run | Adapters | Pairs | Retained | Reduction | Evidence | Drift |")
        lines.append("|-----|----------|-------|----------|-----------|----------|-------|")
        for row in runs:
            run_id = str(row.get("run_id", "?"))[:20]
            adapters = row.get("adapter_count", "?")
            pairs = row.get("pair_count", "?")
            retained = row.get("retained_candidate_count", "?")
            ratio = row.get("reduction_ratio")
            reduction = f"{ratio:.0%}" if isinstance(ratio, (int, float)) else "?"
            evidence = row.get("evidence_ratio", "n/a")
            drift_status = row.get("drift_status") or "—"
            lines.append(
                f"| {run_id} | {adapters} | {pairs} | {retained} | {reduction} | {evidence} | {drift_status} |"
            )
        lines.append("")
    else:
        lines.append("No runs found.")
        lines.append("")

    (output_dir / "batch_summary.md").write_text("\n".join(lines))
    return output_dir
