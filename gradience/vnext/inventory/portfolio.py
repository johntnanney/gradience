"""Portfolio view: landing-page index across multiple inventories.

Scans a directory of existing ``gradience.inventory_summary/v1`` JSON
artifacts (stateless — no new index is maintained) and produces a table
with one row per inventory.

Usage::

    from gradience.vnext.inventory.portfolio import scan_portfolio, format_portfolio_table

    view = scan_portfolio("/path/to/my/inventories")
    print(format_portfolio_table(view))
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_INVENTORY_SCHEMA_ID = "gradience.inventory_summary/v1"
_NONE_ISSUES = frozenset({"none", "unknown"})

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PortfolioRow:
    """One row in the portfolio view — represents a single discovered inventory."""

    name: str
    """Inventory name (subdirectory name or file stem)."""

    timestamp: str
    """File modification time, ISO 8601 UTC (e.g. ``2024-01-15 10:30 UTC``)."""

    adapter_count: int
    """Total QA artifacts scanned (``sources.qa_artifact_count``)."""

    pair_count: int
    """Total merge reports scanned (``sources.merge_report_count``)."""

    retained_candidates: int
    """Adapters with ``eligible`` status."""

    reduction_pct: float | None
    """Percent of adapters removed by QA filtering; ``None`` if no adapters."""

    dominant_driver: str
    """Top non-trivial dominant issue, or ``"none"`` / ``"—"``."""

    exploration_posture: str
    """Label derived from recommended-strategy distribution."""

    evidence_profile: str
    """Label derived from behavioral-eval coverage."""

    drift_status: str
    """``"—"`` unless a ``*drift*.json`` file is detected in the inventory dir."""

    html_report_link: str | None
    """Path to an HTML report if one is found adjacent to the artifact."""

    artifact_link: str
    """Path to the inventory summary JSON file."""


@dataclass(frozen=True)
class PortfolioView:
    """Cross-inventory portfolio view (presentation layer, not policy)."""

    rows: tuple[PortfolioRow, ...]
    scanned_dir: str
    generated_at: str
    """ISO 8601 UTC timestamp for when this view was generated."""


# ---------------------------------------------------------------------------
# Column-derivation helpers
# ---------------------------------------------------------------------------


def _derive_dominant_driver(dominant_issue_counts: dict[str, int]) -> str:
    """Return the most-frequent non-trivial dominant issue label."""
    signal = {k: v for k, v in dominant_issue_counts.items() if k not in _NONE_ISSUES and v > 0}
    if signal:
        return max(signal, key=lambda k: signal[k])
    if dominant_issue_counts.get("none", 0) > 0:
        return "none"
    return "—"


_POSTURE_MAP: dict[str, str] = {
    "linear": "standard",
    "norm_equalized": "balanced",
    "audit_aware": "high-scrutiny",
}


def _derive_exploration_posture(recommended_strategy_counts: dict[str, int]) -> str:
    """Derive a posture label from the dominant recommended-strategy bucket."""
    total = sum(recommended_strategy_counts.values())
    if total == 0:
        return "—"
    dominant = max(recommended_strategy_counts, key=lambda k: recommended_strategy_counts[k])
    return _POSTURE_MAP.get(dominant, dominant)


def _derive_evidence_profile(adapter_status_counts: dict[str, int]) -> str:
    """Derive a coverage label from the behavioral-eval status distribution."""
    total = sum(adapter_status_counts.values())
    if total == 0:
        return "—"
    unknown = adapter_status_counts.get("unknown_no_behavioral_eval", 0)
    eligible = adapter_status_counts.get("eligible", 0)
    weak = adapter_status_counts.get("flagged_weak", 0)
    if unknown == 0 and weak == 0:
        return "full"
    if unknown == total:
        return "blind"
    if eligible > 0:
        return "partial"
    return "weak"


# ---------------------------------------------------------------------------
# Scanner
# ---------------------------------------------------------------------------


def _try_load_summary(fp: Path) -> Any | None:
    """Return a parsed InventorySummary or None (without raising)."""
    try:
        from gradience.vnext.inventory.summary import InventorySummary

        with open(fp) as f:
            data: dict[str, Any] = json.load(f)
        if data.get("schema") != _INVENTORY_SCHEMA_ID:
            return None
        return InventorySummary.from_dict(data)
    except Exception:
        return None


def scan_portfolio(
    directory: str | Path,
    strict_input: bool = False,
) -> PortfolioView:
    """Scan *directory* for inventory summary artifacts and return a portfolio view.

    The scanner looks for ``gradience.inventory_summary/v1`` JSON files:

    * Directly inside *directory*
    * One level deep in named subdirectories

    Each unique inventory name (subdirectory name, or file stem for root-level
    files) yields one :class:`PortfolioRow`.  When multiple files map to the
    same name the most recently modified one wins.

    Parameters
    ----------
    directory
        Root directory to scan.
    strict_input
        When ``True``, re-raises on malformed artifacts instead of warning.

    Returns
    -------
    PortfolioView
    """
    from gradience.vnext.inventory.summary import InventorySummary  # noqa: F401 (side-effect import)

    root = Path(directory).resolve()
    if not root.exists():
        raise ValueError(f"Directory does not exist: {root}")
    if not root.is_dir():
        raise ValueError(f"Path is not a directory: {root}")

    # Collect (inventory_name, path) candidates.
    # Root-level files use their stem as the name; subdir files use the
    # subdirectory name (so multiple JSONs in the same subdir collapse to one).
    candidates: list[tuple[str, Path]] = []

    for fp in sorted(root.glob("*.json")):
        candidates.append((fp.stem, fp))

    for subdir in sorted(p for p in root.iterdir() if p.is_dir()):
        for fp in sorted(subdir.glob("*.json")):
            candidates.append((subdir.name, fp))

    # Quick schema-prefix filter + mtime-based deduplication.
    best: dict[str, tuple[Path, float]] = {}  # name -> (path, mtime)

    for name, fp in candidates:
        try:
            with open(fp) as f:
                raw: dict[str, Any] = json.load(f)
        except Exception:
            if strict_input:
                raise
            continue
        if raw.get("schema") != _INVENTORY_SCHEMA_ID:
            continue
        mtime = fp.stat().st_mtime
        if name not in best or mtime > best[name][1]:
            best[name] = (fp, mtime)

    # Build rows from the winners.
    rows: list[PortfolioRow] = []

    for name, (fp, mtime) in sorted(best.items()):
        try:
            with open(fp) as f:
                raw = json.load(f)
            summary = InventorySummary.from_dict(raw)
        except Exception as exc:
            if strict_input:
                raise
            print(f"WARNING: skipping malformed inventory summary: {fp} ({exc})", file=sys.stderr)
            continue

        # --- Counts ---
        adapter_count = summary.sources.get("qa_artifact_count", 0)
        pair_count = summary.sources.get("merge_report_count", 0)

        # --- Retained candidates & reduction ---
        status_counts = summary.adapter_status_counts
        retained = status_counts.get("eligible", 0)
        total_adapters = sum(status_counts.values())
        reduction_pct: float | None = None
        if total_adapters > 0:
            reduction_pct = round(100.0 * (total_adapters - retained) / total_adapters, 1)

        # --- Derived signal labels ---
        dominant_driver = _derive_dominant_driver(summary.dominant_issue_counts)
        posture = _derive_exploration_posture(summary.recommended_strategy_counts)
        evidence = _derive_evidence_profile(status_counts)

        # --- Drift: look for *drift*.json neighbours ---
        drift_files = list(fp.parent.glob("*drift*.json"))
        drift_status = drift_files[0].name if drift_files else "—"

        # --- HTML report: search adjacent files ---
        html_candidates = list(fp.parent.glob("*.html")) + list(fp.parent.parent.glob("*.html"))
        html_candidates_sorted = sorted(html_candidates, key=lambda p: p.stat().st_mtime, reverse=True)
        html_link: str | None = str(html_candidates_sorted[0]) if html_candidates_sorted else None

        # --- Timestamp ---
        ts = datetime.fromtimestamp(mtime, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

        rows.append(
            PortfolioRow(
                name=name,
                timestamp=ts,
                adapter_count=adapter_count,
                pair_count=pair_count,
                retained_candidates=retained,
                reduction_pct=reduction_pct,
                dominant_driver=dominant_driver,
                exploration_posture=posture,
                evidence_profile=evidence,
                drift_status=drift_status,
                html_report_link=html_link,
                artifact_link=str(fp),
            )
        )

    generated_at = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return PortfolioView(
        rows=tuple(rows),
        scanned_dir=str(root),
        generated_at=generated_at,
    )


# ---------------------------------------------------------------------------
# Terminal formatter
# ---------------------------------------------------------------------------

_HEADERS = (
    "Inventory",
    "Last Run",
    "Adapters",
    "Pairs",
    "Retained",
    "Reduction",
    "Dominant Driver",
    "Posture",
    "Evidence",
    "Drift",
    "HTML",
    "JSON",
)

_MIN_NAME_WIDTH = 12


def format_portfolio_table(view: PortfolioView) -> str:
    """Format a :class:`PortfolioView` as a fixed-width terminal table."""
    if not view.rows:
        return (
            f"\n  PORTFOLIO — {view.scanned_dir}\n"
            f"  No inventory summaries found.\n"
            f"  Generated: {view.generated_at}\n"
        )

    # --- Compute column widths ---
    name_w = max(_MIN_NAME_WIDTH, max(len(r.name) for r in view.rows), len(_HEADERS[0]))
    ts_w = max(len(_HEADERS[1]), max(len(r.timestamp) for r in view.rows))
    driver_w = max(len(_HEADERS[6]), max(len(r.dominant_driver) for r in view.rows))
    posture_w = max(len(_HEADERS[7]), max(len(r.exploration_posture) for r in view.rows))
    evidence_w = max(len(_HEADERS[8]), max(len(r.evidence_profile) for r in view.rows))
    drift_w = max(len(_HEADERS[9]), max(len(r.drift_status) for r in view.rows))

    fixed = {
        "adapters": max(len(_HEADERS[2]), 8),
        "pairs": max(len(_HEADERS[3]), 5),
        "retained": max(len(_HEADERS[4]), 8),
        "reduction": max(len(_HEADERS[5]), 9),
        "html": max(len(_HEADERS[10]), 4),
        "json": max(len(_HEADERS[11]), 4),
    }

    def row_str(
        name: str,
        ts: str,
        adapters: str,
        pairs: str,
        retained: str,
        reduction: str,
        driver: str,
        posture: str,
        evidence: str,
        drift: str,
        html: str,
        artifact: str,
    ) -> str:
        return (
            f"  {name:<{name_w}}  {ts:<{ts_w}}  "
            f"{adapters:>{fixed['adapters']}}  {pairs:>{fixed['pairs']}}  "
            f"{retained:>{fixed['retained']}}  {reduction:>{fixed['reduction']}}  "
            f"{driver:<{driver_w}}  {posture:<{posture_w}}  "
            f"{evidence:<{evidence_w}}  {drift:<{drift_w}}  "
            f"{html:<{fixed['html']}}  {artifact}"
        )

    header_line = row_str(
        _HEADERS[0], _HEADERS[1], _HEADERS[2], _HEADERS[3],
        _HEADERS[4], _HEADERS[5], _HEADERS[6], _HEADERS[7],
        _HEADERS[8], _HEADERS[9], _HEADERS[10], _HEADERS[11],
    )
    # Separator spans all fixed columns; JSON column is open-ended
    sep_line = "  " + "-" * (len(header_line) - 2)

    lines: list[str] = []
    lines.append("")
    lines.append(f"  PORTFOLIO — {view.scanned_dir}")
    lines.append(f"  {len(view.rows)} inventor{'y' if len(view.rows) == 1 else 'ies'} found")
    lines.append("")
    lines.append(header_line)
    lines.append(sep_line)

    for r in view.rows:
        reduction_str = f"{r.reduction_pct:.0f}%" if r.reduction_pct is not None else "—"
        html_str = "yes" if r.html_report_link else "—"
        lines.append(
            row_str(
                r.name,
                r.timestamp,
                str(r.adapter_count),
                str(r.pair_count),
                str(r.retained_candidates),
                reduction_str,
                r.dominant_driver,
                r.exploration_posture,
                r.evidence_profile,
                r.drift_status,
                html_str,
                r.artifact_link,
            )
        )

    lines.append("")
    lines.append(f"  Generated: {view.generated_at}")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# HTML formatter
# ---------------------------------------------------------------------------

_HTML_CSS = """\
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      font-size: 14px;
      background: #f8f9fa;
      color: #212529;
      margin: 0;
      padding: 24px;
    }
    h1 { font-size: 20px; font-weight: 600; margin-bottom: 4px; }
    .meta { color: #6c757d; font-size: 12px; margin-bottom: 24px; }
    table {
      width: 100%;
      border-collapse: collapse;
      background: #fff;
      border-radius: 6px;
      overflow: hidden;
      box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    thead tr { background: #343a40; color: #fff; }
    thead th {
      padding: 10px 12px;
      text-align: left;
      font-size: 12px;
      font-weight: 600;
      letter-spacing: 0.04em;
      white-space: nowrap;
    }
    thead th.num { text-align: right; }
    tbody tr:nth-child(even) { background: #f8f9fa; }
    tbody tr:hover { background: #e9ecef; }
    tbody td { padding: 8px 12px; font-size: 13px; border-bottom: 1px solid #e9ecef; }
    tbody td.num { text-align: right; font-variant-numeric: tabular-nums; }
    .badge {
      display: inline-block;
      padding: 2px 7px;
      border-radius: 3px;
      font-size: 11px;
      font-weight: 600;
    }
    .badge-full    { background: #d1e7dd; color: #0f5132; }
    .badge-partial { background: #fff3cd; color: #664d03; }
    .badge-blind   { background: #f8d7da; color: #842029; }
    .badge-weak    { background: #f8d7da; color: #842029; }
    .badge-standard     { background: #cfe2ff; color: #084298; }
    .badge-balanced     { background: #e2d9f3; color: #432874; }
    .badge-high-scrutiny{ background: #fff3cd; color: #664d03; }
    .badge-none    { background: #d1e7dd; color: #0f5132; }
    .badge-driver  { background: #fde8d8; color: #7d3a0a; }
    a { color: #0d6efd; text-decoration: none; }
    a:hover { text-decoration: underline; }
    .empty { color: #6c757d; }
"""

_EVIDENCE_BADGE: dict[str, str] = {
    "full": "badge-full",
    "partial": "badge-partial",
    "blind": "badge-blind",
    "weak": "badge-weak",
}

_POSTURE_BADGE: dict[str, str] = {
    "standard": "badge-standard",
    "balanced": "badge-balanced",
    "high-scrutiny": "badge-high-scrutiny",
}


def _badge(text: str, cls: str) -> str:
    return f'<span class="badge {cls}">{_esc(text)}</span>'


def _esc(s: str) -> str:
    from html import escape

    return escape(str(s), quote=True)


def format_portfolio_html(view: PortfolioView, title: str = "Gradience Portfolio") -> str:
    """Render a :class:`PortfolioView` as a standalone HTML page."""
    rows_html = _build_html_rows(view.rows)
    row_count = len(view.rows)
    inv_label = "inventor" + ("y" if row_count == 1 else "ies")

    return f"""\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{_esc(title)}</title>
  <style>
{_HTML_CSS}
  </style>
</head>
<body>
  <h1>{_esc(title)}</h1>
  <div class="meta">
    {row_count} {inv_label} &bull;
    Scanned: <code>{_esc(view.scanned_dir)}</code> &bull;
    Generated: {_esc(view.generated_at)}
  </div>
  <table>
    <thead>
      <tr>
        <th>Inventory</th>
        <th>Last Run</th>
        <th class="num">Adapters</th>
        <th class="num">Pairs</th>
        <th class="num">Retained</th>
        <th class="num">Reduction</th>
        <th>Dominant Driver</th>
        <th>Posture</th>
        <th>Evidence</th>
        <th>Drift</th>
        <th>Reports</th>
      </tr>
    </thead>
    <tbody>
{rows_html}\
    </tbody>
  </table>
</body>
</html>
"""


def _build_html_rows(rows: tuple[PortfolioRow, ...]) -> str:
    if not rows:
        return '      <tr><td colspan="11" class="empty" style="text-align:center">No inventories found</td></tr>\n'

    parts: list[str] = []
    for r in rows:
        reduction_str = f"{r.reduction_pct:.0f}%" if r.reduction_pct is not None else "—"

        # Evidence badge
        ev_cls = _EVIDENCE_BADGE.get(r.evidence_profile, "")
        evidence_cell = _badge(r.evidence_profile, ev_cls) if ev_cls else _esc(r.evidence_profile)

        # Posture badge
        po_cls = _POSTURE_BADGE.get(r.exploration_posture, "")
        posture_cell = _badge(r.exploration_posture, po_cls) if po_cls else _esc(r.exploration_posture)

        # Dominant driver
        driver_cell = (
            _badge(r.dominant_driver, "badge-driver")
            if r.dominant_driver not in ("none", "—")
            else (_badge("none", "badge-none") if r.dominant_driver == "none" else "—")
        )

        # Reports links
        links: list[str] = []
        if r.html_report_link:
            links.append(f'<a href="{_esc(r.html_report_link)}">HTML</a>')
        links.append(f'<a href="{_esc(r.artifact_link)}">JSON</a>')
        reports_cell = " &bull; ".join(links)

        parts.append(
            f"      <tr>\n"
            f"        <td><strong>{_esc(r.name)}</strong></td>\n"
            f"        <td>{_esc(r.timestamp)}</td>\n"
            f"        <td class='num'>{r.adapter_count}</td>\n"
            f"        <td class='num'>{r.pair_count}</td>\n"
            f"        <td class='num'>{r.retained_candidates}</td>\n"
            f"        <td class='num'>{_esc(reduction_str)}</td>\n"
            f"        <td>{driver_cell}</td>\n"
            f"        <td>{posture_cell}</td>\n"
            f"        <td>{evidence_cell}</td>\n"
            f"        <td>{_esc(r.drift_status)}</td>\n"
            f"        <td>{reports_cell}</td>\n"
            f"      </tr>\n"
        )

    return "".join(parts)
