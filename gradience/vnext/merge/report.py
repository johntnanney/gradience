"""
Report generation for merge audit results.

Produces:
  merge_audit.json  -- Machine-readable, structured for downstream tools
  merge_audit.md    -- Human-readable, suitable for code review / team sharing

Follows the same artifact philosophy as bench.json / bench.md.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from gradience.vnext.merge.containers import (
    AdapterMetadata,
    AggregateResult,
    MatchingSummary,
)
from gradience.vnext.merge.verdicts import (
    CompatibilityVerdict,
    LayerVerdict,
    VerdictThresholds,
)

if TYPE_CHECKING:
    from gradience.vnext.merge.io import AdapterInfo


# ---------------------------------------------------------------------------
# Report data structure
# ---------------------------------------------------------------------------


@dataclass
class MergeAuditReport:
    """Top-level merge audit report.  Mutable during construction.

    Fields use typed containers (``AdapterMetadata``, ``MatchingSummary``,
    ``AggregateResult``) instead of raw dicts.  The containers support
    both attribute access (``report.adapter_a.rank``) and dict-style
    access (``report.adapter_a["rank"]``) for backward compatibility.
    """

    adapter_a: AdapterMetadata
    adapter_b: AdapterMetadata
    matching: MatchingSummary
    layer_verdicts: list[dict[str, Any]]  # LayerVerdict.to_dict() for each layer
    aggregate: AggregateResult
    recommendations: list[str]
    issues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    thresholds: dict[str, float] = field(default_factory=dict)
    timestamp: str = ""
    gradience_version: str = ""
    source_qa: dict[str, Any] | None = None  # eligibility screening results


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------


def build_report(
    adapter_a_info: AdapterInfo,
    adapter_b_info: AdapterInfo,
    shared: list[str],
    only_a: list[str],
    only_b: list[str],
    layer_verdicts: list[LayerVerdict],
    overall_verdict: CompatibilityVerdict,
    score: float,
    recommendations: list[str],
    thresholds: VerdictThresholds,
    source_qa_a: Any | None = None,
    source_qa_b: Any | None = None,
) -> MergeAuditReport:
    """Assemble full report from analysis results."""

    def safe_mean(xs: list[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    try:
        from importlib.metadata import version as pkg_version

        grad_version = pkg_version("gradience")
    except Exception:  # broad-except: optional metadata lookup
        grad_version = "dev"

    overlaps = [lv.metrics.mean_overlap for lv in layer_verdicts]
    agreements = [lv.metrics.directional_agreement for lv in layer_verdicts]

    # Symmetric scale metrics (getattr for safety during parallel dev)
    scale_bounded_ratios = [getattr(lv.metrics, 'scale_bounded_ratio', 0.0) for lv in layer_verdicts]
    scale_log_ratios = [getattr(lv.metrics, 'scale_log_ratio', 0.0) for lv in layer_verdicts]
    frob_bounded_ratios = [getattr(lv.metrics, 'frob_bounded_ratio', 0.0) for lv in layer_verdicts]
    frob_log_ratios = [getattr(lv.metrics, 'frob_log_ratio', 0.0) for lv in layer_verdicts]
    mag_ratios = [getattr(lv.metrics, 'magnitude_ratio', 0.0) for lv in layer_verdicts]

    # Build source QA section if provided
    source_qa: dict[str, Any] | None = None
    if source_qa_a is not None or source_qa_b is not None:
        source_qa = {}
        if source_qa_a is not None:
            source_qa["adapter_a"] = source_qa_a.to_dict()
        if source_qa_b is not None:
            source_qa["adapter_b"] = source_qa_b.to_dict()

    # Eligibility warnings
    from gradience.vnext.merge.eligibility import screen_adapters

    eligibility_warnings = screen_adapters(source_qa_a, source_qa_b)

    warnings: list[str] = []
    warnings.extend(eligibility_warnings)
    if only_a:
        warnings.append(
            f"{len(only_a)} layer(s) only in adapter A: {', '.join(only_a[:5])}"
        )
    if only_b:
        warnings.append(
            f"{len(only_b)} layer(s) only in adapter B: {', '.join(only_b[:5])}"
        )

    base_a = getattr(adapter_a_info.config, "raw", {}).get("base_model_name_or_path", "unknown")
    base_b = getattr(adapter_b_info.config, "raw", {}).get("base_model_name_or_path", "unknown")

    return MergeAuditReport(
        adapter_a=AdapterMetadata(
            path=str(adapter_a_info.path),
            rank=adapter_a_info.rank,
            alpha=adapter_a_info.alpha,
            base_model=base_a,
            n_layers=len(adapter_a_info.lora_pairs),
        ),
        adapter_b=AdapterMetadata(
            path=str(adapter_b_info.path),
            rank=adapter_b_info.rank,
            alpha=adapter_b_info.alpha,
            base_model=base_b,
            n_layers=len(adapter_b_info.lora_pairs),
        ),
        matching=MatchingSummary(
            n_shared=len(shared),
            n_only_a=len(only_a),
            n_only_b=len(only_b),
            shared_layers=tuple(shared),
            only_a_layers=tuple(only_a),
            only_b_layers=tuple(only_b),
        ),
        layer_verdicts=[lv.to_dict() for lv in layer_verdicts],
        aggregate=AggregateResult(
            overall_verdict=overall_verdict.value,
            compatibility_score=round(score, 4),
            mean_overlap=round(sum(overlaps) / len(overlaps), 4) if overlaps else 0.0,
            median_overlap=round(
                statistics.median(overlaps), 4
            ) if overlaps else 0.0,
            max_overlap=round(max(overlaps), 4) if overlaps else 0.0,
            mean_agreement=round(
                sum(agreements) / len(agreements), 4
            ) if agreements else 0.0,
            n_safe=sum(
                1 for lv in layer_verdicts if lv.verdict == CompatibilityVerdict.SAFE
            ),
            n_redundant=sum(
                1 for lv in layer_verdicts if lv.verdict == CompatibilityVerdict.REDUNDANT
            ),
            n_conflicting=sum(
                1 for lv in layer_verdicts if lv.verdict == CompatibilityVerdict.CONFLICTING
            ),
            n_imbalanced=sum(
                1 for lv in layer_verdicts if lv.verdict == CompatibilityVerdict.IMBALANCED
            ),
            mean_scale_bounded_ratio=round(safe_mean(scale_bounded_ratios), 4),
            mean_scale_log_ratio=round(safe_mean(scale_log_ratios), 4),
            mean_frob_bounded_ratio=round(safe_mean(frob_bounded_ratios), 4),
            mean_frob_log_ratio=round(safe_mean(frob_log_ratios), 4),
            mean_magnitude_ratio=round(safe_mean(mag_ratios), 4),
        ),
        recommendations=recommendations,
        issues=[],
        warnings=warnings,
        thresholds=thresholds.to_dict(),
        timestamp=datetime.now(timezone.utc).isoformat(),
        gradience_version=grad_version,
        source_qa=source_qa,
    )


# ---------------------------------------------------------------------------
# JSON
# ---------------------------------------------------------------------------


def to_json(report: MergeAuditReport) -> dict[str, Any]:
    """Convert report to JSON-serializable dict."""
    # Use .to_dict() on typed containers; fall back to raw value for dicts
    def _serialize(val: Any) -> Any:
        if hasattr(val, 'to_dict'):
            return val.to_dict()
        return val

    d: dict[str, Any] = {
        "schema_version": "gradience.merge_audit/v2",
        "timestamp": report.timestamp,
        "gradience_version": report.gradience_version,
        "adapter_a": _serialize(report.adapter_a),
        "adapter_b": _serialize(report.adapter_b),
        "matching": _serialize(report.matching),
        "aggregate": _serialize(report.aggregate),
        "per_layer": report.layer_verdicts,
        "recommendations": report.recommendations,
        "issues": report.issues,
        "warnings": report.warnings,
        "thresholds": report.thresholds,
    }
    if report.source_qa is not None:
        d["source_qa"] = report.source_qa
    return d


def write_json_report(report: MergeAuditReport, output_path: Path) -> None:
    """Write merge_audit.json."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = to_json(report)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------


def _shorten_layer_name(name: str) -> str:
    """Shorten a fully-qualified layer name while keeping it unique.

    Extracts the block index and module suffix, e.g.:
        "base_model.model.model.layers.10.self_attn.k_proj"  -> "L10.k_proj"
        "base_model.model.lm_head"                           -> "lm_head"
    """
    parts = name.split(".")

    # Find the last numeric component (block index)
    block_idx = None
    suffix_start = len(parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i].isdigit():
            block_idx = parts[i]
            suffix_start = i + 1
            break

    suffix = ".".join(parts[suffix_start:]) if suffix_start < len(parts) else parts[-1]

    if block_idx is not None:
        return f"L{block_idx}.{suffix}"
    return suffix


_VERDICT_EMOJI = {
    "safe": "\u2705",         # green check
    "redundant": "\u26a0\ufe0f",  # warning
    "conflicting": "\u274c",  # red X
    "imbalanced": "\u2696\ufe0f",  # balance scale
}


def to_markdown(report: MergeAuditReport) -> str:
    """Render human-readable markdown report."""
    lines: list[str] = []
    agg = report.aggregate

    # --- Title ---
    lines.append("# Merge Compatibility Audit\n")

    # --- Adapters ---
    lines.append("## Adapters\n")
    lines.append("| | Adapter A | Adapter B |")
    lines.append("|---|---|---|")
    lines.append(
        f"| **Path** | `{report.adapter_a.path}` | `{report.adapter_b.path}` |"
    )
    lines.append(
        f"| **Rank** | {report.adapter_a.rank} | {report.adapter_b.rank} |"
    )
    lines.append(
        f"| **Alpha** | {report.adapter_a.alpha} | {report.adapter_b.alpha} |"
    )
    lines.append(
        f"| **Base model** | {report.adapter_a.base_model} | {report.adapter_b.base_model} |"
    )
    lines.append(
        f"| **Layers** | {report.adapter_a.n_layers} | {report.adapter_b.n_layers} |"
    )
    lines.append("")

    # --- Compatibility summary ---
    verdict_str = agg.overall_verdict
    emoji = _VERDICT_EMOJI.get(verdict_str, "")
    lines.append("## Compatibility Summary\n")
    lines.append(f"**Overall verdict:** {emoji} **{verdict_str.upper()}**\n")
    lines.append(f"- Compatibility score: {agg.compatibility_score:.3f}")
    lines.append(f"- Mean overlap: {agg.mean_overlap:.3f}")
    lines.append(f"- Max overlap: {agg.max_overlap:.3f}")
    lines.append(f"- Mean agreement: {agg.mean_agreement:.3f}")
    lines.append(
        f"- Shared layers: {report.matching.n_shared} | "
        f"Only A: {report.matching.n_only_a} | "
        f"Only B: {report.matching.n_only_b}"
    )
    lines.append(
        f"- Verdicts: {agg.n_safe} safe, "
        f"{agg.n_redundant} redundant, "
        f"{agg.n_conflicting} conflicting, "
        f"{agg.n_imbalanced} imbalanced"
    )
    lines.append("")

    # --- Per-layer analysis ---
    if report.layer_verdicts:
        lines.append("## Per-Layer Analysis\n")
        lines.append(
            "| Layer | r_a | r_b | Overlap | Agreement | Scale | Mag ratio | Verdict | Strategy |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|---:|---|---|")

        for lv in report.layer_verdicts:
            m = lv["metrics"]
            v_emoji = _VERDICT_EMOJI.get(lv["verdict"], "")
            short_name = _shorten_layer_name(lv["layer_name"])
            lines.append(
                f"| {short_name} "
                f"| {m['nominal_rank_a']} "
                f"| {m['nominal_rank_b']} "
                f"| {m['mean_overlap']:.3f} "
                f"| {m['directional_agreement']:.3f} "
                f"| {m.get('scale_bounded_ratio', 0.0):.3f} "
                f"| {m['magnitude_ratio']:.1f}x "
                f"| {v_emoji} {lv['verdict']} "
                f"| {lv['suggested_strategy']} |"
            )
        lines.append("")

    # --- Conflict details ---
    conflicts = [lv for lv in report.layer_verdicts if lv["verdict"] == "conflicting"]
    if conflicts:
        lines.append("## Conflict Details\n")
        for lv in conflicts:
            lines.append(f"### {lv['layer_name']}\n")
            lines.append(f"- {lv['recommendation']}")
            lines.append(f"- Conflicting dimensions: {lv['conflict_dimensions']}")
            lines.append("")

    # --- Source eligibility ---
    if report.source_qa:
        lines.append("## Source Adapter Eligibility\n")
        lines.append(
            "> **Note:** Merge recommendations are structural. "
            "Source eligibility screening checks whether each adapter "
            "is worth preserving based on behavioral evaluation.\n"
        )
        for label, key in [("Adapter A", "adapter_a"), ("Adapter B", "adapter_b")]:
            qa = report.source_qa.get(key)
            if qa:
                status = qa.get("status", "unknown").upper()
                status_indicator = {
                    "ELIGIBLE": "PASS",
                    "UNCERTAIN": "UNCERTAIN",
                    "FLAGGED_WEAK": "WEAK",
                    "UNKNOWN": "NO DATA",
                }.get(status, status)
                lines.append(f"- **{label}:** {status_indicator}")
                if qa.get("notes"):
                    lines.append(f"  - {qa['notes']}")
                if qa.get("adapter_metric") is not None and qa.get("base_metric") is not None:
                    lines.append(
                        f"  - {qa.get('metric_name', 'metric')}: "
                        f"adapter={qa['adapter_metric']:.4f}, "
                        f"base={qa['base_metric']:.4f}"
                    )
        lines.append("")

    # --- Recommendations ---
    if report.recommendations:
        lines.append("## Recommendations\n")
        for rec in report.recommendations:
            lines.append(f"- {rec}")
        lines.append("")

    # --- Warnings ---
    if report.warnings:
        lines.append("## Warnings\n")
        for warn in report.warnings:
            lines.append(f"- {warn}")
        lines.append("")

    # --- Footer ---
    lines.append(f"*Generated by Gradience {report.gradience_version} on {report.timestamp}*\n")

    return "\n".join(lines)


def write_markdown_report(report: MergeAuditReport, output_path: Path) -> None:
    """Write merge_audit.md."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    md = to_markdown(report)
    with open(output_path, "w") as f:
        f.write(md)


def write_reports(report: MergeAuditReport, output_dir: str | Path) -> None:
    """Write both merge_audit.json and merge_audit.md."""
    output_dir = Path(output_dir)
    write_json_report(report, output_dir / "merge_audit.json")
    write_markdown_report(report, output_dir / "merge_audit.md")
