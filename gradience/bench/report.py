"""
Bench report generation (v0.1).

Writes:
- bench.json: machine-readable summary
- bench.md: human-readable summary

This module is intentionally conservative: it does not assume optional keys exist.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple


def _write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def render_markdown(report: Dict[str, Any]) -> str:
    """
    Render a minimal, stable markdown summary.
    Expected (v0.1): report contains keys similar to the planned bench schema,
    but we degrade gracefully if fields are missing.
    """
    bench_version = report.get("bench_version", "unknown")
    model = report.get("model", report.get("model_name", "unknown"))
    task = report.get("task", report.get("dataset", "unknown"))

    probe = report.get("probe", {}) or {}
    probe_rank = probe.get("rank", "n/a")
    probe_params = probe.get("params", "n/a")
    probe_acc = probe.get("accuracy", "n/a")

    compressed = report.get("compressed", {}) or {}

    lines = []
    lines.append(f"# Gradience Bench v{bench_version}")
    lines.append("")
    lines.append(f"- **Model:** {model}")
    lines.append(f"- **Task:** {task}")
    lines.append("")
    lines.append("## Probe")
    lines.append("")
    lines.append(f"- **Rank:** {probe_rank}")
    lines.append(f"- **LoRA params:** {probe_params}")
    lines.append(f"- **Accuracy:** {probe_acc}")
    lines.append("")

    # Results table
    lines.append("## Compression results")
    lines.append("")
    lines.append("| Variant | Params | Accuracy | Δ vs probe | Param reduction | Verdict |")
    lines.append("|---|---:|---:|---:|---:|---|")

    # Render known variants in stable order (extras allowed)
    ordered = ["uniform_median", "uniform_p90", "per_layer"]
    keys = ordered + [k for k in compressed.keys() if k not in ordered]

    for k in keys:
        r = compressed.get(k, {}) or {}
        params = r.get("params", "n/a")
        acc = r.get("accuracy", "n/a")
        delta = r.get("delta_vs_probe", "n/a")
        red = r.get("param_reduction", "n/a")
        verdict = r.get("verdict", "n/a")
        lines.append(f"| `{k}` | {params} | {acc} | {delta} | {red} | {verdict} |")

    lines.append("")
    summary = report.get("summary", {}) or {}
    if summary:
        lines.append("## Summary")
        lines.append("")
        for k, v in summary.items():
            lines.append(f"- **{k}:** {v}")
        lines.append("")

    return "\n".join(lines)


def write_report(output_dir: str | Path, report: Dict[str, Any]) -> Tuple[Path, Path]:
    """
    Write bench.json and bench.md into output_dir.
    Returns (json_path, md_path).
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    json_path = out / "bench.json"
    md_path = out / "bench.md"

    _write_json(json_path, report)
    md_path.write_text(render_markdown(report) + "\n", encoding="utf-8")

    return json_path, md_path