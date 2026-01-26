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
    
    # Magnitude diagnostics section
    # Try instrumentation first (for v0.1 schema), then fallback to top-level
    instrumentation = report.get("instrumentation", {})
    composition = instrumentation.get("composition", {}) or report.get("composition", {})
    gain_summary = report.get("summary", {}).get("gain", {})
    global_gain = report.get("global", {}).get("gain", {})
    
    # Check if composition analysis was enabled
    has_composition = bool(composition)
    
    if gain_summary or global_gain:
        lines.append("## Magnitude diagnostics (LoRA ΔW)")
        lines.append("")
        
        # Overall magnitude metrics
        delta_fro_mean = gain_summary.get("delta_fro_mean")
        delta_op_mean = gain_summary.get("delta_op_mean")
        if delta_fro_mean is not None or delta_op_mean is not None:
            lines.append("### Update magnitude")
            lines.append("")
            if delta_fro_mean is not None:
                lines.append(f"- **Mean ||ΔW||_F:** {delta_fro_mean:.6f}")
            if delta_op_mean is not None:
                lines.append(f"- **Mean ||ΔW||_2:** {delta_op_mean:.6f}")
            lines.append("")
        
        # Top 5 layers by energy concentration (if composition analysis enabled)
        if has_composition and composition.get("top_k", {}).get("layers"):
            lines.append("### Top 5 layers by Δ energy")
            lines.append("")
            top_layers = composition["top_k"]["layers"][:5]  # Ensure max 5
            total_energy = composition.get("energy_total_fro2", 0)
            
            for i, layer_info in enumerate(top_layers, 1):
                layer_num = layer_info["layer"]
                share = layer_info["share"]
                energy = layer_info["energy_fro2"]
                lines.append(f"{i}. **Layer {layer_num}:** {share:.1%} ({energy:.6f})")
            lines.append("")
        
        # Top 5 modules by Frobenius norm
        top_modules = global_gain.get("top_modules_by_delta_fro", [])
        if top_modules:
            lines.append("### Top 5 modules by ||ΔW||_F")
            lines.append("")
            for i, module_info in enumerate(top_modules[:5], 1):  # Ensure max 5
                module_name = module_info["module"]
                delta_fro = module_info["delta_fro"]
                layer_num = module_info.get("layer", "?")
                # Shorten long module names for readability
                short_name = module_name.split(".")[-2:] if "." in module_name else [module_name]
                short_name = ".".join(short_name)
                lines.append(f"{i}. **{short_name}** (L{layer_num}): {delta_fro:.6f}")
            lines.append("")
        
        # Energy concentration summary (if composition analysis enabled)
        if has_composition:
            top_10pct_share = composition.get("top_10pct", {}).get("share")
            concentration_index = composition.get("concentration_index")
            if top_10pct_share is not None or concentration_index is not None:
                lines.append("### Energy concentration")
                lines.append("")
                if top_10pct_share is not None:
                    n_layers = composition.get("top_10pct", {}).get("n", 0)
                    lines.append(f"- **Top-{n_layers} layers (10%):** {top_10pct_share:.1%} of energy")
                if concentration_index is not None:
                    lines.append(f"- **Concentration index (HHI):** {concentration_index:.3f}")
                    # Simple interpretation
                    if concentration_index > 0.4:
                        lines.append("- 🚨 **Highly concentrated** adaptation")
                    elif concentration_index > 0.25:
                        lines.append("- ⚠️ **Moderately concentrated** adaptation")
                    else:
                        lines.append("- ✅ **Well distributed** adaptation")
                lines.append("")
        elif gain_summary or global_gain:
            # Show note that composition analysis was disabled
            lines.append("### Energy concentration")
            lines.append("")
            lines.append("- *Composition analysis disabled in config (audit.enable_composition_analysis: false)*")
            lines.append("")

    summary = report.get("summary", {}) or {}
    if summary:
        lines.append("## Summary")
        lines.append("")
        for k, v in summary.items():
            if k != "gain":  # Skip gain summary as it's already shown above
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