"""Static HTML preflight report — single-file renderer.

Consumes a preflight run bundle (the JSON artifacts produced by
``--emit-bundle``) and generates a self-contained static HTML page.
No JavaScript frameworks, no server, no external assets.  The only
dependency is Python's standard library.

Usage from CLI::

    gradience preflight-report --bundle-dir run-abc123/ -o report.html

Usage from Python::

    from gradience.vnext.inventory.html_report import render_html_report
    html = render_html_report(bundle_dir)
    Path("report.html").write_text(html)
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# CSS — clean technical style, muted palette, good typography
# ---------------------------------------------------------------------------

_CSS = """\
:root {
  --bg: #fafafa;
  --surface: #ffffff;
  --border: #e0e0e0;
  --text: #1a1a1a;
  --text-muted: #666666;
  --accent: #2563eb;
  --green: #16a34a;
  --amber: #d97706;
  --red: #dc2626;
  --mono: 'SF Mono', 'Cascadia Code', 'Fira Code', Menlo, monospace;
  --sans: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
}

* { margin: 0; padding: 0; box-sizing: border-box; }

body {
  font-family: var(--sans);
  font-size: 15px;
  line-height: 1.6;
  color: var(--text);
  background: var(--bg);
  max-width: 960px;
  margin: 0 auto;
  padding: 2rem 1.5rem 4rem;
}

h1 { font-size: 1.5rem; font-weight: 700; margin-bottom: 0.25rem; }
h2 {
  font-size: 1.1rem; font-weight: 600; margin: 2rem 0 0.75rem;
  padding-bottom: 0.4rem; border-bottom: 1px solid var(--border);
}
h3 { font-size: 0.95rem; font-weight: 600; margin: 1.25rem 0 0.5rem; }

.meta { color: var(--text-muted); font-size: 0.85rem; margin-bottom: 1.5rem; }
.meta span { margin-right: 1.5rem; }

/* Sticky jump-nav */
.jump-nav {
  position: sticky;
  top: 0;
  z-index: 10;
  background: var(--bg);
  border-bottom: 1px solid var(--border);
  padding: 0.45rem 0;
  margin: 0 0 1rem;
  display: flex;
  flex-wrap: wrap;
  gap: 0.25rem 1rem;
  font-size: 0.8rem;
}
.jump-nav a {
  color: var(--accent);
  text-decoration: none;
  white-space: nowrap;
}
.jump-nav a:hover { text-decoration: underline; }

/* Cards */
.card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 1rem 1.25rem;
  margin-bottom: 0.75rem;
}

/* Executive interpretation line */
.exec-line {
  font-size: 0.92rem;
  color: var(--text);
  margin: 0.5rem 0 0.5rem;
  padding: 0.6rem 1rem;
  background: #f0f4ff;
  border-left: 3px solid var(--accent);
  border-radius: 0 4px 4px 0;
}

/* Collapsible sections */
details { margin-bottom: 0.5rem; }
details summary {
  cursor: pointer;
  font-weight: 600;
  font-size: 1.05rem;
  padding: 0.75rem 1rem;
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 6px;
  list-style: none;
  user-select: none;
}
details summary::-webkit-details-marker { display: none; }
details summary::before {
  content: '\\25B6';
  display: inline-block;
  margin-right: 0.5rem;
  font-size: 0.7rem;
  transition: transform 0.15s;
  color: var(--text-muted);
}
details[open] summary::before { transform: rotate(90deg); }
details[open] summary { border-radius: 6px 6px 0 0; margin-bottom: 0; }
details .section-body {
  border: 1px solid var(--border);
  border-top: none;
  border-radius: 0 0 6px 6px;
  padding: 1rem 1.25rem;
  background: var(--surface);
}

/* Tables */
table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.88rem;
  margin: 0.5rem 0;
}
th, td {
  text-align: left;
  padding: 0.45rem 0.65rem;
  border-bottom: 1px solid var(--border);
}
th {
  font-weight: 600;
  font-size: 0.8rem;
  text-transform: uppercase;
  letter-spacing: 0.03em;
  color: var(--text-muted);
  background: #f5f5f5;
}
tr:last-child td { border-bottom: none; }

/* Group separator row */
.group-header td {
  font-weight: 600;
  font-size: 0.82rem;
  background: #f8f9fa;
  color: var(--text-muted);
  text-transform: uppercase;
  letter-spacing: 0.03em;
  padding: 0.55rem 0.65rem;
  border-bottom: 2px solid var(--border);
}

/* Badges */
.badge {
  display: inline-block;
  font-size: 0.75rem;
  font-weight: 600;
  padding: 0.15rem 0.55rem;
  border-radius: 10px;
  text-transform: uppercase;
  letter-spacing: 0.03em;
}
.badge-green { background: #dcfce7; color: #166534; }
.badge-amber { background: #fef3c7; color: #92400e; }
.badge-red { background: #fee2e2; color: #991b1b; }
.badge-gray { background: #f3f4f6; color: #374151; }
.badge-blue { background: #dbeafe; color: #1e40af; }

/* Stat grid */
.stat-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
  gap: 0.75rem;
  margin: 0.75rem 0;
}
.stat-box {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 0.75rem 1rem;
  text-align: center;
}
.stat-box .stat-value {
  font-size: 1.6rem;
  font-weight: 700;
  font-family: var(--mono);
  line-height: 1.2;
}
.stat-box .stat-label {
  font-size: 0.75rem;
  color: var(--text-muted);
  text-transform: uppercase;
  letter-spacing: 0.04em;
  margin-top: 0.2rem;
}
.stat-box .stat-subtitle {
  font-size: 0.72rem;
  color: var(--text-muted);
  margin-top: 0.1rem;
}
.stat-box.highlight { border-color: var(--accent); background: #f0f4ff; }
.stat-box.highlight .stat-value { color: var(--accent); }

/* Count bars */
.count-bar {
  display: flex;
  align-items: center;
  margin: 0.3rem 0;
  font-size: 0.85rem;
}
.count-bar .bar-label { width: 180px; color: var(--text-muted); }
.count-bar .bar-track {
  flex: 1;
  height: 18px;
  background: #f3f4f6;
  border-radius: 3px;
  overflow: hidden;
  margin: 0 0.5rem;
}
.count-bar .bar-fill {
  height: 100%;
  border-radius: 3px;
  min-width: 2px;
}
.count-bar .bar-value {
  font-family: var(--mono);
  font-weight: 600;
  width: 36px;
  text-align: right;
}

/* Action plan sections */
.action-section {
  margin: 0.75rem 0;
  padding: 0.75rem 1rem;
  border-left: 3px solid var(--border);
  background: #fafafa;
  border-radius: 0 4px 4px 0;
}
.action-section.exclude { border-left-color: var(--red); }
.action-section.safe { border-left-color: var(--green); }
.action-section.near-miss { border-left-color: #b08d57; }
.action-section.caution { border-left-color: var(--amber); }
.action-section.evaluate { border-left-color: var(--accent); }
.action-note { font-size: 0.88em; color: var(--text-muted); margin-bottom: 0.5em; font-style: italic; }

.action-section h3 { margin-top: 0; }

.action-item {
  font-family: var(--mono);
  font-size: 0.85rem;
  padding: 0.2rem 0;
}

/* Summary line */
.summary-line {
  font-style: italic;
  color: var(--text-muted);
  margin: 1rem 0;
  padding: 0.75rem 1rem;
  background: #f8fafc;
  border-radius: 4px;
  border: 1px solid var(--border);
}

/* Pair detail table */
.pair-table td:nth-child(2),
.pair-table td:nth-child(3) { font-family: var(--mono); font-size: 0.82rem; }

/* Footer */
.footer {
  margin-top: 3rem;
  padding-top: 1rem;
  border-top: 1px solid var(--border);
  font-size: 0.78rem;
  color: var(--text-muted);
}
.footer-links {
  margin-top: 0.5rem;
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem 1.5rem;
}
.footer-links a {
  color: var(--text-muted);
  text-decoration: none;
}
.footer-links a:hover { text-decoration: underline; color: var(--accent); }
"""


# ---------------------------------------------------------------------------
# Human-facing label maps
# ---------------------------------------------------------------------------

_INVENTORY_TYPE_LABELS: dict[str, str] = {
    "mixed_task": "Mixed-task inventory",
    "single_task": "Single-task inventory",
    "same_task": "Same-task inventory",
    "cross_task": "Cross-task inventory",
    "unknown": "Unknown inventory type",
}

_DRIVER_LABELS: dict[str, str] = {
    "task_boundary": "Task-boundary risk",
    "evidence_gap": "Evidence gap",
    "structural_flags": "Structural flags",
    "none": "No dominant driver",
}

_POSTURE_LABELS: dict[str, str] = {
    "conservative": "Conservative — behavioral evidence required",
    "moderate": "Moderate — mixed evidence accepted",
    "exploratory": "Exploratory — willing to proceed with limited evidence",
}

_DOMINANT_ISSUE_LABELS: dict[str, str] = {
    "norm_imbalance": "Norm imbalance",
    "subspace_conflict": "Subspace conflict",
    "high_redundancy": "High redundancy",
    "partial_redundancy": "Partial redundancy",
    "none": "No dominant issue",
    "unknown": "Unknown",
}

_DRIFT_IMPLICATION_TABLE: dict[str, str] = {
    "current_run_materially_narrower": (
        "Candidate space narrowed — fewer plausible merges remain, "
        "so evaluation effort can be concentrated on the surviving subset."
    ),
    "current_run_materially_broader": (
        "Candidate space broadened — more pairs entered the evaluation window, "
        "increasing the search budget needed before merging."
    ),
    "source_composition_changed": (
        "Source composition changed but candidate subset size is similar — "
        "review which adapters entered or left the pool."
    ),
    "evidence_improved_posture_unchanged": (
        "Evidence quality improved, but the exploration posture is unchanged — "
        "better confidence in existing judgments without changing scope."
    ),
    "evidence_degraded": (
        "Evidence quality degraded — review source QA before proceeding, "
        "as some adapters may have lost behavioral backing."
    ),
    "no_substantial_change": "No substantial preflight change between runs.",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _e(text: Any) -> str:
    """HTML-escape."""
    return escape(str(text))


def _badge(text: str, color: str = "gray") -> str:
    return f'<span class="badge badge-{color}">{_e(text)}</span>'


def _risk_badge(risk: str) -> str:
    colors = {"low": "green", "medium": "amber", "high": "red"}
    return _badge(risk, colors.get(risk, "gray"))


def _status_badge(status: str) -> str:
    colors = {
        "eligible": "green",
        "uncertain": "amber",
        "flagged_weak": "red",
        "unknown_no_behavioral_eval": "gray",
    }
    _labels = {
        "eligible": "eligible",
        "uncertain": "uncertain",
        "flagged_weak": "flagged weak",
        "unknown_no_behavioral_eval": "no behavioral eval",
    }
    return _badge(_labels.get(status, status.replace("_", " ")), colors.get(status, "gray"))


def _strategy_badge(strategy: str) -> str:
    colors = {"linear": "green", "norm_equalized": "amber", "audit_aware": "blue"}
    _labels = {
        "linear": "linear",
        "norm_equalized": "norm-equalized",
        "audit_aware": "audit-aware",
    }
    return _badge(_labels.get(strategy, strategy.replace("_", " ")), colors.get(strategy, "gray"))


def _issue_label(issue: str) -> str:
    """Human-facing label for dominant_issue."""
    return _DOMINANT_ISSUE_LABELS.get(issue, issue.replace("_", " "))


def _count_bar(label: str, value: int, max_val: int, color: str = "#2563eb") -> str:
    pct = (value / max_val * 100) if max_val > 0 else 0
    return (
        f'<div class="count-bar">'
        f'<span class="bar-label">{_e(label)}</span>'
        f'<span class="bar-track"><span class="bar-fill" '
        f'style="width:{pct:.1f}%;background:{color}"></span></span>'
        f'<span class="bar-value">{value}</span>'
        f"</div>"
    )


def _stat_box(value: Any, label: str, *, subtitle: str = "", highlight: bool = False) -> str:
    cls = "stat-box highlight" if highlight else "stat-box"
    sub = f'<div class="stat-subtitle">{_e(subtitle)}</div>' if subtitle else ""
    return (
        f'<div class="{cls}">'
        f'<div class="stat-value">{_e(value)}</div>'
        f'<div class="stat-label">{_e(label)}</div>'
        f"{sub}"
        f"</div>"
    )


def _executive_interpretation(summary: dict[str, Any]) -> str:
    """Build a one-line executive interpretation from summary data."""
    adapters = summary.get("adapter_count", 0)
    retained = summary.get("retained_candidate_count", 0)
    pairs = summary.get("pair_count", 0)
    advisory_count = summary.get("advisory_pair_count", 0)
    excluded = summary.get("excluded_sources", [])

    if adapters == 0:
        return ""

    parts: list[str] = []

    # Candidate reduction framing
    if pairs > 0 and retained > 0:
        parts.append(f"{retained} of {pairs} candidate pairs survived preflight")
    elif pairs > 0:
        parts.append(f"{pairs} candidate pairs assessed")

    # Exclusions
    if excluded:
        parts.append(f"{len(excluded)} source{'s' if len(excluded) != 1 else ''} excluded")

    # Cross-task advisory
    if advisory_count > 0:
        parts.append(f"{advisory_count} cross-task advisor{'ies' if advisory_count != 1 else 'y'}")

    if not parts:
        return ""

    return f'<div class="exec-line">{"; ".join(parts)}.</div>'


# ---------------------------------------------------------------------------
# Section renderers
# ---------------------------------------------------------------------------


def _render_header(summary: dict[str, Any], manifest: dict[str, Any] | None) -> str:
    inv_id = summary.get("inventory_id", "unknown")
    run_id = summary.get("run_id", "unknown")
    ts = summary.get("timestamp", "")
    adapters = summary.get("adapter_count", 0)
    pairs = summary.get("pair_count", 0)
    retained = summary.get("retained_candidate_count", 0)
    reduction = summary.get("reduction_ratio", 0)

    # Subtitle for the retained box: "X of Y retained"
    retained_subtitle = f"{retained} of {pairs} retained" if pairs > 0 else ""

    return (
        f"<h1>Preflight Report &mdash; {_e(inv_id)}</h1>"
        f'<div class="meta">'
        f"<span>Run: <code>{_e(run_id)}</code></span>"
        f"<span>{_e(ts[:19] if ts else '')}</span>"
        f"</div>"
        f'<div class="stat-grid">'
        + _stat_box(adapters, "Adapters")
        + _stat_box(pairs, "Pairs")
        + _stat_box(retained, "Retained", subtitle=retained_subtitle, highlight=True)
        + _stat_box(f"{reduction:.0%}", "Reduction")
        + "</div>"
        + _executive_interpretation(summary)
    )


def _render_jump_nav(sections: list[tuple[str, str]]) -> str:
    """Render a sticky jump-nav bar from (anchor, label) pairs."""
    if not sections:
        return ""
    links = " ".join(f'<a href="#{_e(anchor)}">{_e(label)}</a>' for anchor, label in sections)
    return f'<nav class="jump-nav">{links}</nav>'


def _render_trust_snapshot(summary: dict[str, Any]) -> str:
    qa = summary.get("qa_summary", {})
    bev = summary.get("behavioral_evidence_count", 0)
    tsc = summary.get("total_source_count", 0)

    # Status counts subsection
    status_rows = ""
    for status in ["eligible", "uncertain", "flagged_weak", "unknown_no_behavioral_eval"]:
        count = qa.get(status, 0)
        if count > 0:
            status_rows += f"<tr><td>{_status_badge(status)}</td><td>{count}</td></tr>"

    status_html = (
        "<h3>Eligibility Status</h3>"
        "<table><thead><tr><th>Status</th><th>Count</th></tr></thead>"
        f"<tbody>{status_rows}</tbody></table>"
    )

    # Evidence profile subsection
    evidence_html = ""
    if tsc > 0:
        evidence_pct = f"{bev / tsc:.0%}" if tsc > 0 else "—"
        evidence_html = (
            '<h3 style="margin-top:1rem">Evidence Profile</h3>'
            f'<p style="font-size:0.88rem">'
            f"<strong>{bev}</strong> of <strong>{tsc}</strong> sources ({evidence_pct}) have behavioral evidence. "
            f"<em>Scores are user-reported; Gradience does not independently verify.</em></p>"
        )

    return (
        '<details open id="sec-trust"><summary>Source QA / Trust Snapshot</summary>'
        f'<div class="section-body">{status_html}{evidence_html}</div>'
        "</details>"
    )


def _render_policy_summary(summary: dict[str, Any]) -> str:
    policy = summary.get("inventory_policy_summary")
    if not policy:
        return ""

    inv_type = _INVENTORY_TYPE_LABELS.get(policy.get("inventory_type", ""), policy.get("inventory_type", "—"))
    driver = _DRIVER_LABELS.get(policy.get("dominant_driver", ""), policy.get("dominant_driver", "—"))
    posture = _POSTURE_LABELS.get(policy.get("exploration_posture", ""), policy.get("exploration_posture", "—"))
    constraint = policy.get("constraint", "—")

    return (
        '<details open id="sec-policy"><summary>Inventory Policy Summary</summary>'
        '<div class="section-body">'
        "<table>"
        f'<tr><td style="width:120px;color:var(--text-muted)">Type</td><td>{_e(inv_type)}</td></tr>'
        f'<tr><td style="color:var(--text-muted)">Driver</td><td>{_e(driver)}</td></tr>'
        f'<tr><td style="color:var(--text-muted)">Posture</td><td>{_e(posture)}</td></tr>'
        f'<tr><td style="color:var(--text-muted)">Constraint</td><td>{_e(constraint)}</td></tr>'
        "</table>"
        "</div></details>"
    )


def _render_action_plan(summary: dict[str, Any], action_plan_md: str | None) -> str:
    excluded = summary.get("excluded_sources", [])
    same_task = summary.get("same_task_priority_pairs", [])
    cross_task = summary.get("cross_task_caution_regions", [])
    evaluate = summary.get("reduced_candidate_subset", [])
    near_miss = summary.get("near_miss_candidates", [])
    summary_line = summary.get("summary_line", "")

    def _items(lst: list[str], cls: str) -> str:
        if not lst:
            return '<div class="action-item" style="color:var(--text-muted)">none</div>'
        return "".join(f'<div class="action-item">{_e(item)}</div>' for item in lst)

    near_miss_html = ""
    if near_miss:
        near_miss_html = (
            '<div class="action-section near-miss">'
            "<h3>Near-miss candidates</h3>"
            '<div class="action-note">Structurally plausible, evidence-constrained. '
            "Optional if evaluation budget allows.</div>"
            f"{_items(near_miss, 'near-miss')}</div>"
        )

    return (
        '<details open id="sec-action"><summary>Action Plan</summary>'
        '<div class="section-body">'
        f'<div class="action-section exclude"><h3>Exclude / deprioritize</h3>{_items(excluded, "exclude")}</div>'
        f'<div class="action-section safe"><h3>Same-task safe zone</h3>{_items(same_task, "safe")}</div>'
        f"{near_miss_html}"
        f'<div class="action-section caution"><h3>Cross-task caution</h3>{_items(cross_task, "caution")}</div>'
        f'<div class="action-section evaluate"><h3>Evaluate first</h3>{_items(evaluate, "evaluate")}</div>'
        f'<div class="summary-line">{_e(summary_line)}</div>'
        "</div></details>"
    )


def _render_region_summary(summary: dict[str, Any]) -> str:
    region = summary.get("large_inventory_region_summary")
    if not region or not region.get("enabled"):
        return ""

    rc = region.get("region_counts", {})
    csmap = region.get("candidate_space_map", [])

    counts_html = (
        '<div class="stat-grid">'
        + _stat_box(rc.get("same_task_safe_regions", 0), "Safe regions")
        + _stat_box(rc.get("cross_task_caution_regions", 0), "Caution regions")
        + _stat_box(rc.get("weak_evidence_regions", 0), "Weak evidence")
        + _stat_box(rc.get("priority_evaluation_regions", 0), "Priority eval")
        + "</div>"
    )

    map_html = ""
    if csmap:
        _status_display = {
            "evaluate_first": "evaluate first",
            "same_task_safe": "same-task safe",
            "do_not_explore_casually": "do not explore casually",
            "excluded": "excluded",
        }
        _type_display = {
            "same_task_safe_region": "same-task safe",
            "cross_task_caution_region": "cross-task caution",
            "weak_evidence_region": "weak-evidence",
        }
        rows = ""
        for entry in csmap:
            r = entry.get("region", "")
            t = _type_display.get(entry.get("type", ""), entry.get("type", ""))
            p = entry.get("pair_count", 0)
            s = _status_display.get(entry.get("status", ""), entry.get("status", ""))
            rows += f"<tr><td>{_e(r)}</td><td>{_e(t)}</td><td style='text-align:right'>{p}</td><td>{_e(s)}</td></tr>"
        map_html = (
            "<h3>Candidate-Space Map</h3>"
            '<table><thead><tr><th>Region</th><th>Type</th><th style="text-align:right">Pairs</th><th>Status</th></tr></thead>'
            f"<tbody>{rows}</tbody></table>"
        )

    return (
        '<details open id="sec-regions"><summary>Region Summary</summary>'
        f'<div class="section-body">{counts_html}{map_html}</div>'
        "</details>"
    )


def _classify_pair_group(report: dict[str, Any]) -> str:
    """Classify a pair report into a display group."""
    advisory = report.get("task_relationship_advisory")
    risk = report.get("pair_risk", "low")
    if risk == "high":
        return "high-risk"
    if advisory:
        return "cross-task"
    return "same-task / low-risk"


_GROUP_ORDER = ["high-risk", "cross-task", "same-task / low-risk"]
_GROUP_LABELS = {
    "high-risk": "High-risk pairs",
    "cross-task": "Cross-task pairs",
    "same-task / low-risk": "Same-task / low-risk pairs",
}


def _render_pair_reports(reports: list[dict[str, Any]]) -> str:
    if not reports:
        return ""

    # Group reports
    groups: dict[str, list[dict[str, Any]]] = {}
    for r in reports:
        g = _classify_pair_group(r)
        groups.setdefault(g, []).append(r)

    # Determine if grouping adds value (multiple groups with content)
    active_groups = [g for g in _GROUP_ORDER if groups.get(g)]
    use_groups = len(active_groups) > 1

    rows = ""
    for group_key in _GROUP_ORDER:
        group_reports = groups.get(group_key)
        if not group_reports:
            continue
        if use_groups:
            rows += f'<tr class="group-header"><td colspan="5">{_e(_GROUP_LABELS.get(group_key, group_key))}</td></tr>'
        for r in group_reports:
            a_name = Path(r.get("adapter_a", {}).get("path", "?")).name
            b_name = Path(r.get("adapter_b", {}).get("path", "?")).name
            risk = r.get("pair_risk", "?")
            issue = _issue_label(r.get("dominant_issue", "?"))
            strategy = r.get("recommended_strategy", "?")
            confidence = r.get("confidence", "?")
            advisory = r.get("task_relationship_advisory")
            advisory_marker = (
                ' <span style="color:var(--amber);font-size:0.75rem" title="cross-task advisory">&#9888;</span>'
                if advisory
                else ""
            )

            rows += (
                f"<tr>"
                f"<td><code>{_e(a_name)}</code> &times; <code>{_e(b_name)}</code>{advisory_marker}</td>"
                f"<td>{_risk_badge(risk)}</td>"
                f"<td>{_e(issue)}</td>"
                f"<td>{_strategy_badge(strategy)}</td>"
                f"<td>{_badge(confidence)}</td>"
                f"</tr>"
            )

    return (
        '<details id="sec-pairs"><summary>Pair Summary</summary>'
        '<div class="section-body">'
        '<table class="pair-table"><thead>'
        "<tr><th>Pair</th><th>Risk</th><th>Issue</th><th>Strategy</th><th>Confidence</th></tr>"
        "</thead><tbody>"
        f"{rows}"
        "</tbody></table>"
        "</div></details>"
    )


def _render_adapter_details(artifacts: list[dict[str, Any]]) -> str:
    if not artifacts:
        return ""

    rows = ""
    for a in artifacts:
        name = Path(a.get("adapter_path", a.get("adapter_name", "?"))).name
        status = a.get("status", "?")
        rank = a.get("rank_nominal", "?")
        util = a.get("utilization_mean", 0)
        flags = a.get("structural_flags", [])
        confidence = a.get("confidence", "?")
        eval_ds = a.get("eval_dataset", "—") or "—"

        flag_html = (
            " ".join(_badge(f.replace("_", " "), "amber") for f in flags)
            if flags
            else '<span style="color:var(--text-muted)">none</span>'
        )

        rows += (
            f"<tr>"
            f"<td><code>{_e(name)}</code></td>"
            f"<td>{_status_badge(status)}</td>"
            f'<td style="text-align:right">{rank}</td>'
            f'<td style="text-align:right">{util:.2f}</td>'
            f"<td>{flag_html}</td>"
            f"<td>{_e(eval_ds)}</td>"
            f"<td>{_badge(confidence)}</td>"
            f"</tr>"
        )

    return (
        '<details id="sec-adapters"><summary>Source Adapter Details</summary>'
        '<div class="section-body">'
        "<table><thead>"
        '<tr><th>Adapter</th><th>Status</th><th style="text-align:right">Rank</th>'
        '<th style="text-align:right">Util</th><th>Flags</th><th>Eval</th><th>Confidence</th></tr>'
        "</thead><tbody>"
        f"{rows}"
        "</tbody></table>"
        "</div></details>"
    )


def _render_drift_summary(drift: dict[str, Any] | None) -> str:
    if not drift:
        return ""

    status = drift.get("status", "stable")
    status_colors = {"narrowing": "green", "broadening": "amber", "stable": "gray"}

    impl_key = drift.get("implication", "no_substantial_change")
    impl_text = _DRIFT_IMPLICATION_TABLE.get(impl_key, impl_key)

    delta_html = ""
    for key, label in [
        ("adapter_count_delta", "Adapters"),
        ("pair_count_delta", "Pairs"),
        ("retained_candidate_delta", "Retained"),
    ]:
        d = drift.get(key, 0)
        sign = f"+{d}" if d > 0 else str(d)
        delta_html += f"<tr><td>{label}</td><td style='font-family:var(--mono)'>{sign}</td></tr>"

    for key, label in [
        ("same_task_safe_zone_change", "Safe zone"),
        ("cross_task_caution_zone_change", "Caution zone"),
        ("evidence_profile_change", "Evidence"),
    ]:
        delta_html += f"<tr><td>{label}</td><td>{_e(drift.get(key, 'unchanged'))}</td></tr>"

    return (
        '<details id="sec-drift"><summary>History / Drift Summary</summary>'
        '<div class="section-body">'
        f"<p><strong>Status:</strong> {_badge(status, status_colors.get(status, 'gray'))}</p>"
        f'<table style="margin:0.75rem 0">{delta_html}</table>'
        f'<div class="summary-line">{_e(impl_text)}</div>'
        "</div></details>"
    )


def _render_footer(bundle_dir: Path) -> str:
    """Render footer with generation timestamp and artifact links."""
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # Discover available artifacts
    artifact_links: list[tuple[str, str]] = []
    candidates = [
        ("preflight_summary.json", "Preflight JSON"),
        ("review_packet.json", "Review Packet"),
        ("compare_to_previous.md", "Run Comparison"),
        ("inventory_action_plan.md", "Action Plan (Markdown)"),
        ("preflight_summary.md", "Preflight Summary (Markdown)"),
    ]
    for filename, label in candidates:
        if (bundle_dir / filename).exists():
            artifact_links.append((filename, label))

    links_html = ""
    if artifact_links:
        items = " ".join(f'<a href="{_e(fn)}">{_e(label)}</a>' for fn, label in artifact_links)
        links_html = f'<div class="footer-links">Bundle artifacts: {items}</div>'

    return f'<div class="footer">Generated by Gradience preflight-report &middot; {generated}{links_html}</div>'


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def render_html_report(
    bundle_dir: str | Path,
    *,
    title: str | None = None,
) -> str:
    """Render a complete preflight HTML report from a bundle directory.

    Parameters
    ----------
    bundle_dir
        Path to the run bundle (contains preflight_summary.json,
        run_manifest.json, etc.).
    title
        Optional override for the page title.

    Returns
    -------
    str
        A self-contained HTML document.
    """
    bd = Path(bundle_dir)

    # Load core JSON
    summary: dict[str, Any] = {}
    summary_path = bd / "preflight_summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)

    manifest: dict[str, Any] | None = None
    manifest_path = bd / "run_manifest.json"
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)

    # Load QA artifacts
    qa_artifacts: list[dict[str, Any]] = []
    qa_dir = bd / "qa"
    if qa_dir.is_dir():
        for p in sorted(qa_dir.glob("*.json")):
            with open(p) as f:
                qa_artifacts.append(json.load(f))

    # Load pair reports
    pair_reports: list[dict[str, Any]] = []
    reports_dir = bd / "pair_reports"
    if reports_dir.is_dir():
        for p in sorted(reports_dir.glob("*.json")):
            with open(p) as f:
                pair_reports.append(json.load(f))

    # Load action plan markdown (optional)
    action_plan_md: str | None = None
    ap_path = bd / "inventory_action_plan.md"
    if ap_path.exists():
        action_plan_md = ap_path.read_text()

    # Load drift summary (from review_packet.json if available)
    drift: dict[str, Any] | None = None
    review_path = bd / "review_packet.json"
    if review_path.exists():
        with open(review_path) as f:
            review = json.load(f)
            drift = review.get("drift_summary")

    # Build page title
    inv_id = summary.get("inventory_id", "Preflight Report")
    page_title = title or f"Gradience Preflight — {inv_id}"

    # Determine section ordering — region summary goes higher for large inventories
    has_regions = bool(summary.get("large_inventory_region_summary", {}).get("enabled"))

    # Collect sections and their nav entries
    header_html = _render_header(summary, manifest)
    trust_html = _render_trust_snapshot(summary)
    policy_html = _render_policy_summary(summary)
    action_html = _render_action_plan(summary, action_plan_md)
    region_html = _render_region_summary(summary)
    pairs_html = _render_pair_reports(pair_reports)
    adapters_html = _render_adapter_details(qa_artifacts)
    drift_html = _render_drift_summary(drift)

    # Build jump-nav entries based on what sections are actually present
    nav_entries: list[tuple[str, str]] = []
    nav_entries.append(("sec-trust", "Trust Snapshot"))
    if policy_html:
        nav_entries.append(("sec-policy", "Policy"))
    nav_entries.append(("sec-action", "Action Plan"))
    if region_html:
        nav_entries.append(("sec-regions", "Regions"))
    if pairs_html:
        nav_entries.append(("sec-pairs", "Pairs"))
    if adapters_html:
        nav_entries.append(("sec-adapters", "Adapters"))
    if drift_html:
        nav_entries.append(("sec-drift", "Drift"))

    jump_nav = _render_jump_nav(nav_entries)

    # Assemble sections — region summary higher when enabled
    if has_regions:
        sections = [
            header_html,
            jump_nav,
            trust_html,
            policy_html,
            region_html,
            action_html,
            pairs_html,
            adapters_html,
            drift_html,
        ]
    else:
        sections = [
            header_html,
            jump_nav,
            trust_html,
            policy_html,
            action_html,
            region_html,
            pairs_html,
            adapters_html,
            drift_html,
        ]

    body = "\n".join(s for s in sections if s)
    footer = _render_footer(bd)

    return (
        "<!DOCTYPE html>\n"
        f'<html lang="en">\n<head>\n'
        f'<meta charset="utf-8">\n'
        f'<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>{_e(page_title)}</title>\n"
        f"<style>\n{_CSS}\n</style>\n"
        f"</head>\n<body>\n"
        f"{body}\n{footer}\n"
        f"</body>\n</html>\n"
    )
