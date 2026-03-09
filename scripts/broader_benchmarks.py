#!/usr/bin/env python3
"""
Study 14: Broader Benchmarks — Spectral Audits of Real-World LoRA Adapters
===========================================================================

Downloads pre-trained LoRA adapters from HuggingFace Hub and runs Gradience
spectral audit on each, building a dataset of utilization patterns, module-type
breakdowns, and compression potential across the ecosystem.

Usage:
    python scripts/broader_benchmarks.py \
        --output-dir ./results/study14_broader_benchmarks \
        --cache-dir /tmp/gradience_adapter_cache \
        --max-adapters 30

The script is idempotent: re-running skips already-audited adapters.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Ensure gradience is importable
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from gradience.vnext.audit.lora_audit import LoRAAuditResult, audit_lora_peft_dir

# ============================================================================
# Data structures
# ============================================================================


@dataclass
class AdapterRecord:
    """Metadata for one adapter to audit."""

    repo_id: str
    base_model: str
    task: str
    # These get filled after download / audit:
    rank: int | None = None
    alpha: float | None = None
    n_layers: int | None = None
    total_params: int | None = None
    download_size_mb: float | None = None
    status: str = "pending"  # pending | downloaded | audited | failed
    issues: list[str] = field(default_factory=list)
    audit_time_s: float | None = None


# ============================================================================
# Discovery — find real PEFT LoRA adapters on HuggingFace Hub
# ============================================================================

# Base models we want diversity across
BASE_MODELS = [
    "meta-llama/Llama-2-7b-hf",
    "meta-llama/Llama-2-7b-chat-hf",
    "mistralai/Mistral-7B-v0.1",
    "mistralai/Mistral-7B-Instruct-v0.1",
    "microsoft/phi-2",
    "google/gemma-2b",
    "google/gemma-7b",
    "meta-llama/Meta-Llama-3-8B",
    "meta-llama/Meta-Llama-3.1-8B",
    "Qwen/Qwen2-7B",
]


def discover_adapters(
    max_per_base: int = 5,
    max_total: int = 30,
    min_downloads: int = 10,
) -> list[AdapterRecord]:
    """
    Search HuggingFace Hub for PEFT LoRA adapters across diverse base models.

    Returns a curated list of AdapterRecords ready for download + audit.
    """
    from huggingface_hub import HfApi

    api = HfApi()
    records: list[AdapterRecord] = []
    seen_repos = set()

    for base_model in BASE_MODELS:
        if len(records) >= max_total:
            break

        print(f"\n🔍 Searching for LoRA adapters based on: {base_model}")
        try:
            # Search for PEFT models with this base
            models = api.list_models(
                filter="peft",
                search=base_model.split("/")[-1],
                sort="downloads",
                direction=-1,
                limit=50,
                full=True,
            )

            count_for_base = 0
            for model in models:
                if count_for_base >= max_per_base or len(records) >= max_total:
                    break

                repo_id = model.id
                if repo_id in seen_repos:
                    continue

                # Skip base models themselves
                if repo_id in BASE_MODELS:
                    continue

                # Check download count
                downloads = getattr(model, "downloads", 0) or 0
                if downloads < min_downloads:
                    continue

                # Check for required PEFT files
                try:
                    siblings = model.siblings or []
                    filenames = {s.rfilename for s in siblings}
                except Exception:
                    filenames = set()

                has_config = any(f.startswith("adapter_config") for f in filenames)
                has_weights = any(f.startswith("adapter_model") or f == "pytorch_model.bin" for f in filenames)

                if not (has_config and has_weights):
                    continue

                # Infer task from tags or model card
                tags = set(getattr(model, "tags", []) or [])
                task = _infer_task(tags, repo_id)

                seen_repos.add(repo_id)
                records.append(
                    AdapterRecord(
                        repo_id=repo_id,
                        base_model=base_model,
                        task=task,
                    )
                )
                count_for_base += 1
                print(f"   ✅ {repo_id} ({downloads} downloads, task={task})")

        except Exception as e:
            print(f"   ⚠️  Error searching for {base_model}: {e}")
            continue

    print(f"\n📋 Discovered {len(records)} adapters across {len(set(r.base_model for r in records))} base models")
    return records


def _infer_task(tags: set, repo_id: str) -> str:
    """Heuristic task inference from model tags and repo name."""
    name_lower = repo_id.lower()

    # Check tags first
    tag_map = {
        "text-generation": "text-generation",
        "text-classification": "classification",
        "question-answering": "qa",
        "summarization": "summarization",
        "translation": "translation",
        "conversational": "chat",
        "text2text-generation": "text2text",
    }
    for tag, task in tag_map.items():
        if tag in tags:
            return task

    # Fall back to repo name heuristics
    if any(k in name_lower for k in ("code", "coder", "starcoder")):
        return "code"
    if any(k in name_lower for k in ("math", "gsm", "arithmetic")):
        return "math"
    if any(k in name_lower for k in ("chat", "instruct", "sft", "rlhf", "dpo")):
        return "chat"
    if any(k in name_lower for k in ("medical", "bio", "clinical")):
        return "medical"
    if any(k in name_lower for k in ("legal", "law", "contract")):
        return "legal"
    if any(k in name_lower for k in ("sql", "data", "table")):
        return "data"
    if any(k in name_lower for k in ("ner", "classify", "sentiment")):
        return "classification"

    return "general"


# ============================================================================
# Download
# ============================================================================


def download_adapter(
    repo_id: str,
    cache_dir: Path,
    max_size_mb: float = 500.0,
) -> tuple[Path, float]:
    """
    Download a PEFT adapter from HuggingFace Hub.

    Returns (local_path, size_mb).
    Only downloads adapter_config* and adapter_model* files.
    """
    from huggingface_hub import snapshot_download

    local_dir = cache_dir / repo_id.replace("/", "--")

    # Check if already downloaded
    config_candidates = list(local_dir.glob("adapter_config*"))
    weight_candidates = list(local_dir.glob("adapter_model*"))
    if config_candidates and weight_candidates:
        size_mb = sum(f.stat().st_size for f in local_dir.rglob("*") if f.is_file()) / (1024 * 1024)
        return local_dir, size_mb

    # Download only adapter files
    local_path = snapshot_download(
        repo_id,
        local_dir=str(local_dir),
        allow_patterns=[
            "adapter_config*",
            "adapter_model*",
            "pytorch_model.bin",
        ],
        ignore_patterns=[
            "*.md",
            "*.txt",
            ".gitattributes",
            "tokenizer*",
            "special_tokens*",
            "training_args*",
        ],
    )

    size_mb = sum(f.stat().st_size for f in Path(local_path).rglob("*") if f.is_file()) / (1024 * 1024)

    if size_mb > max_size_mb:
        raise ValueError(f"Adapter {repo_id} is {size_mb:.1f} MB (limit: {max_size_mb} MB)")

    return Path(local_path), size_mb


# ============================================================================
# Audit
# ============================================================================


def audit_adapter(
    repo_id: str,
    adapter_path: Path,
    record: AdapterRecord,
) -> tuple[dict[str, Any] | None, AdapterRecord]:
    """
    Run Gradience spectral audit on one adapter.

    Returns (audit_dict, updated_record).
    audit_dict is None if audit failed.
    """
    t0 = time.time()
    try:
        result: LoRAAuditResult = audit_lora_peft_dir(
            adapter_path,
            compute_udr=False,  # No base model needed
            include_top_singular_values=4,
            rank_policies=["energy@0.90", "knee", "erank"],
        )

        elapsed = time.time() - t0
        record.audit_time_s = elapsed
        record.n_layers = result.n_layers
        record.total_params = result.total_lora_params

        if result.issues:
            record.issues.extend(result.issues)

        if result.n_layers == 0:
            record.status = "failed"
            record.issues.append("No LoRA layers found in adapter")
            return None, record

        # Extract rank from first layer
        if result.layers:
            record.rank = result.layers[0].r
            record.alpha = result.layers[0].alpha

        audit_dict = result.to_summary_dict(include_layers=True)
        record.status = "audited"
        return audit_dict, record

    except Exception as e:
        elapsed = time.time() - t0
        record.audit_time_s = elapsed
        record.status = "failed"
        record.issues.append(f"Audit error: {e}")
        traceback.print_exc()
        return None, record


# ============================================================================
# Analysis
# ============================================================================


def analyze_results(
    records: list[AdapterRecord],
    audit_data: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """
    Compute aggregate statistics across all audited adapters.
    """
    analysis: dict[str, Any] = {}

    # Filter to successful audits
    audited = [r for r in records if r.status == "audited" and r.repo_id in audit_data]
    if not audited:
        return {"error": "No successful audits to analyze"}

    # ---- Global utilization distribution ----
    utils_all = [audit_data[r.repo_id]["utilization_mean"] for r in audited]
    stable_ranks_all = [audit_data[r.repo_id]["stable_rank_mean"] for r in audited]
    eff_ranks_all = [audit_data[r.repo_id]["effective_rank_mean"] for r in audited]
    energy_90_p50s = [audit_data[r.repo_id]["energy_rank_90_p50"] for r in audited]

    analysis["global_utilization"] = _distribution_stats(utils_all)
    analysis["global_stable_rank"] = _distribution_stats(stable_ranks_all)
    analysis["global_effective_rank"] = _distribution_stats(eff_ranks_all)
    analysis["global_energy_rank_90_p50"] = _distribution_stats(energy_90_p50s)

    # ---- By module type (attn vs mlp) ----
    module_utils = {"attn": [], "mlp": [], "other": []}
    module_stable = {"attn": [], "mlp": [], "other": []}
    for r in audited:
        data = audit_data[r.repo_id]
        layer_data = data.get("layer_data", {})
        layer_rows = layer_data.get("layer_rows", [])
        for layer in layer_rows:
            mt = layer.get("module_type", "other")
            if mt not in module_utils:
                mt = "other"
            u = layer.get("utilization")
            sr = layer.get("stable_rank")
            if u is not None:
                module_utils[mt].append(u)
            if sr is not None:
                module_stable[mt].append(sr)

    analysis["by_module_type"] = {}
    for mt in ("attn", "mlp", "other"):
        if module_utils[mt]:
            analysis["by_module_type"][mt] = {
                "utilization": _distribution_stats(module_utils[mt]),
                "stable_rank": _distribution_stats(module_stable[mt]),
                "n_layers": len(module_utils[mt]),
            }

    # ---- By nominal rank ----
    by_rank: dict[int, list[float]] = {}
    for r in audited:
        if r.rank is not None:
            by_rank.setdefault(r.rank, []).append(audit_data[r.repo_id]["utilization_mean"])
    analysis["by_nominal_rank"] = {str(rank): _distribution_stats(vals) for rank, vals in sorted(by_rank.items())}

    # ---- By base model ----
    by_base: dict[str, list[float]] = {}
    for r in audited:
        short_base = r.base_model.split("/")[-1]
        by_base.setdefault(short_base, []).append(audit_data[r.repo_id]["utilization_mean"])
    analysis["by_base_model"] = {base: _distribution_stats(vals) for base, vals in sorted(by_base.items())}

    # ---- By task ----
    by_task: dict[str, list[float]] = {}
    for r in audited:
        by_task.setdefault(r.task, []).append(audit_data[r.repo_id]["utilization_mean"])
    analysis["by_task"] = {task: _distribution_stats(vals) for task, vals in sorted(by_task.items())}

    # ---- Energy rank analysis (compression potential) ----
    # How much of the nominal rank is actually needed at 90% energy?
    compression_ratios = []
    for r in audited:
        data = audit_data[r.repo_id]
        if r.rank and r.rank > 0:
            e90_p50 = data.get("energy_rank_90_p50", 0)
            if e90_p50 > 0:
                compression_ratios.append(e90_p50 / r.rank)
    if compression_ratios:
        analysis["compression_potential"] = {
            "energy_90_fraction_of_nominal": _distribution_stats(compression_ratios),
            "interpretation": (
                f"Median adapter needs only {_percentile(compression_ratios, 0.5):.0%} "
                f"of its nominal rank to capture 90% energy"
            ),
        }

    # ---- Wasteful layers (lowest utilization across all adapters) ----
    all_layers = []
    for r in audited:
        data = audit_data[r.repo_id]
        layer_rows = data.get("layer_data", {}).get("layer_rows", [])
        for layer in layer_rows:
            u = layer.get("utilization")
            if u is not None:
                all_layers.append(
                    {
                        "adapter": r.repo_id,
                        "layer": layer.get("name", "?"),
                        "module_type": layer.get("module_type", "?"),
                        "utilization": u,
                        "rank": layer.get("r", "?"),
                        "stable_rank": layer.get("stable_rank"),
                    }
                )
    all_layers.sort(key=lambda x: x["utilization"])
    analysis["most_wasteful_layers"] = all_layers[:20]

    # ---- Rank–utilization correlation ----
    ranks_for_corr = []
    utils_for_corr = []
    for r in audited:
        if r.rank is not None and r.rank > 0:
            ranks_for_corr.append(r.rank)
            utils_for_corr.append(audit_data[r.repo_id]["utilization_mean"])
    if len(ranks_for_corr) >= 5:
        corr = _pearson(ranks_for_corr, utils_for_corr)
        analysis["rank_utilization_correlation"] = {
            "pearson_r": corr,
            "n": len(ranks_for_corr),
            "interpretation": ("Weak negative" if corr < -0.1 else "Weak positive" if corr > 0.1 else "Near zero")
            + f" correlation ({corr:.3f}): "
            + (
                "higher-rank adapters tend to underutilize"
                if corr < -0.1
                else "higher-rank adapters tend to use rank more fully"
                if corr > 0.1
                else "no clear relationship between rank and utilization"
            ),
        }

    return analysis


def _distribution_stats(vals: list[float]) -> dict[str, Any]:
    """Compute distribution statistics for a list of values."""
    if not vals:
        return {"n": 0}
    vals_sorted = sorted(vals)
    n = len(vals_sorted)
    mean = sum(vals_sorted) / n
    variance = sum((v - mean) ** 2 for v in vals_sorted) / max(n - 1, 1)
    std = math.sqrt(variance)
    return {
        "n": n,
        "mean": round(mean, 4),
        "median": round(_percentile(vals_sorted, 0.5), 4),
        "std": round(std, 4),
        "min": round(vals_sorted[0], 4),
        "max": round(vals_sorted[-1], 4),
        "p25": round(_percentile(vals_sorted, 0.25), 4),
        "p75": round(_percentile(vals_sorted, 0.75), 4),
        "p90": round(_percentile(vals_sorted, 0.90), 4),
    }


def _percentile(sorted_vals: list[float], q: float) -> float:
    """Simple percentile on pre-sorted list."""
    if not sorted_vals:
        return 0.0
    idx = (len(sorted_vals) - 1) * q
    lo = int(math.floor(idx))
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = idx - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def _pearson(x: list[float], y: list[float]) -> float:
    """Simple Pearson correlation coefficient."""
    n = len(x)
    if n < 2:
        return 0.0
    mx = sum(x) / n
    my = sum(y) / n
    num = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    dx = math.sqrt(sum((xi - mx) ** 2 for xi in x))
    dy = math.sqrt(sum((yi - my) ** 2 for yi in y))
    if dx * dy == 0:
        return 0.0
    return num / (dx * dy)


# ============================================================================
# Report generation
# ============================================================================


def generate_markdown_report(
    records: list[AdapterRecord],
    audit_data: dict[str, dict[str, Any]],
    analysis: dict[str, Any],
) -> str:
    """Generate a human-readable markdown report for the blog post."""
    audited = [r for r in records if r.status == "audited"]
    failed = [r for r in records if r.status == "failed"]

    lines = [
        "# Study 14: Spectral Audits of Real-World LoRA Adapters",
        "",
        "## Overview",
        "",
        f"Gradience audited **{len(audited)}** publicly available LoRA adapters "
        f"from HuggingFace Hub across **{len(set(r.base_model for r in audited))}** "
        f"base models and **{len(set(r.task for r in audited))}** task categories.",
        "",
        f"- Adapters discovered: {len(records)}",
        f"- Successfully audited: {len(audited)}",
        f"- Failed: {len(failed)}",
        "- All computation: CPU-only (no base models downloaded)",
        "",
    ]

    # Key findings
    gu = analysis.get("global_utilization", {})
    lines += [
        "## Key Findings",
        "",
    ]

    if gu.get("n", 0) > 0:
        lines.append(
            f"**1. Utilization is widely variable.** "
            f"Mean: {gu['mean']:.3f}, median: {gu['median']:.3f}, "
            f"std: {gu['std']:.3f}. Range: [{gu['min']:.3f}, {gu['max']:.3f}]."
        )
        lines.append("")

    bmt = analysis.get("by_module_type", {})
    attn_u = bmt.get("attn", {}).get("utilization", {}).get("mean")
    mlp_u = bmt.get("mlp", {}).get("utilization", {}).get("mean")
    if attn_u is not None and mlp_u is not None:
        lines.append(
            f"**2. Attention vs MLP.** "
            f"Attention layers: {attn_u:.3f} mean utilization. "
            f"MLP layers: {mlp_u:.3f}. "
            f"{'Attention uses rank more fully.' if attn_u > mlp_u else 'MLP uses rank more fully.'}"
        )
        lines.append("")

    cp = analysis.get("compression_potential", {})
    cp_frac = cp.get("energy_90_fraction_of_nominal", {})
    if cp_frac.get("n", 0) > 0:
        lines.append(
            f"**3. Compression potential.** "
            f"Median adapter needs only {cp_frac['median']:.0%} of its nominal rank "
            f"to capture 90% spectral energy (p90: {cp_frac['p90']:.0%})."
        )
        lines.append("")

    ruc = analysis.get("rank_utilization_correlation", {})
    if ruc:
        lines.append(f"**4. Rank–utilization correlation.** {ruc.get('interpretation', 'N/A')}")
        lines.append("")

    # Adapter summary table
    lines += [
        "## Adapter Summary",
        "",
        "| Adapter | Base Model | Task | Rank | Layers | Util (mean) | Stable Rank | Energy r@90% p50 |",
        "|---------|-----------|------|------|--------|-------------|-------------|-----------------|",
    ]
    for r in sorted(audited, key=lambda x: x.repo_id):
        data = audit_data.get(r.repo_id, {})
        short_base = r.base_model.split("/")[-1] if r.base_model else "?"
        lines.append(
            f"| `{r.repo_id}` | {short_base} | {r.task} "
            f"| {r.rank or '?'} | {r.n_layers or '?'} "
            f"| {data.get('utilization_mean', 0):.3f} "
            f"| {data.get('stable_rank_mean', 0):.2f} "
            f"| {data.get('energy_rank_90_p50', 0):.1f} |"
        )
    lines.append("")

    # By nominal rank
    by_rank = analysis.get("by_nominal_rank", {})
    if by_rank:
        lines += [
            "## Utilization by Nominal Rank",
            "",
            "| Rank | N | Util (mean) | Util (median) | Util (std) |",
            "|------|---|-------------|---------------|------------|",
        ]
        for rank_str, stats in by_rank.items():
            lines.append(
                f"| {rank_str} | {stats['n']} | {stats['mean']:.3f} | {stats['median']:.3f} | {stats['std']:.3f} |"
            )
        lines.append("")

    # By base model
    by_base = analysis.get("by_base_model", {})
    if by_base:
        lines += [
            "## Utilization by Base Model",
            "",
            "| Base Model | N | Util (mean) | Util (median) |",
            "|-----------|---|-------------|---------------|",
        ]
        for base, stats in by_base.items():
            lines.append(f"| {base} | {stats['n']} | {stats['mean']:.3f} | {stats['median']:.3f} |")
        lines.append("")

    # By task
    by_task = analysis.get("by_task", {})
    if by_task:
        lines += [
            "## Utilization by Task",
            "",
            "| Task | N | Util (mean) | Util (median) |",
            "|------|---|-------------|---------------|",
        ]
        for task, stats in by_task.items():
            lines.append(f"| {task} | {stats['n']} | {stats['mean']:.3f} | {stats['median']:.3f} |")
        lines.append("")

    # Module type breakdown
    if bmt:
        lines += [
            "## Module Type Breakdown",
            "",
            "| Type | N Layers | Util (mean) | Util (median) | Stable Rank (mean) |",
            "|------|----------|-------------|---------------|-------------------|",
        ]
        for mt in ("attn", "mlp", "other"):
            info = bmt.get(mt, {})
            u_stats = info.get("utilization", {})
            sr_stats = info.get("stable_rank", {})
            if u_stats.get("n", 0) > 0:
                lines.append(
                    f"| {mt} | {info.get('n_layers', 0)} "
                    f"| {u_stats.get('mean', 0):.3f} | {u_stats.get('median', 0):.3f} "
                    f"| {sr_stats.get('mean', 0):.2f} |"
                )
        lines.append("")

    # Most wasteful layers
    wasteful = analysis.get("most_wasteful_layers", [])
    if wasteful:
        lines += [
            "## Most Wasteful Layers (Top 10)",
            "",
            "| Adapter | Layer | Type | Rank | Utilization | Stable Rank |",
            "|---------|-------|------|------|-------------|-------------|",
        ]
        for w in wasteful[:10]:
            sr = w.get("stable_rank")
            sr_str = f"{sr:.2f}" if isinstance(sr, (int, float)) else "?"
            lines.append(
                f"| `{w['adapter']}` | {w['layer']} | {w['module_type']} "
                f"| {w['rank']} | {w['utilization']:.3f} "
                f"| {sr_str} |"
            )
        lines.append("")

    # Failed adapters
    if failed:
        lines += [
            "## Failed Audits",
            "",
        ]
        for r in failed:
            lines.append(f"- `{r.repo_id}`: {'; '.join(r.issues)}")
        lines.append("")

    # Methodology
    lines += [
        "## Methodology",
        "",
        "- **Discovery**: HuggingFace Hub API, filtered for PEFT library tag, "
        "sorted by downloads, verified adapter_config.json + weights present.",
        "- **Audit**: Gradience spectral audit (CPU-only, float64). "
        "Computes stable rank, effective rank, utilization, energy ranks "
        "via efficient r×r eigendecomposition.",
        "- **No UDR**: Base model weights not downloaded. Study focuses on adapter-intrinsic spectral structure.",
        f"- **Date**: {datetime.now(timezone.utc).strftime('%Y-%m-%d')}",
        "- **Gradience version**: 0.11.0",
        "",
    ]

    return "\n".join(lines)


# ============================================================================
# Main orchestrator
# ============================================================================


def scan_local_adapters(local_dirs: list[Path]) -> list[tuple[AdapterRecord, Path]]:
    """
    Scan local directories for PEFT adapter directories.

    Each entry in local_dirs can be either:
    - A direct adapter dir (containing adapter_config.json)
    - A parent dir to scan recursively for adapter dirs

    Returns list of (AdapterRecord, adapter_path) tuples.
    """
    found: list[tuple[AdapterRecord, Path]] = []

    for d in local_dirs:
        d = Path(d)
        if not d.exists():
            print(f"⚠️  Path does not exist: {d}")
            continue

        # Check if this directory itself is an adapter
        if (d / "adapter_config.json").exists():
            record = _make_local_record(d)
            found.append((record, d))
            continue

        # Scan recursively
        for config_path in d.rglob("adapter_config.json"):
            adapter_dir = config_path.parent
            record = _make_local_record(adapter_dir)
            found.append((record, adapter_dir))

    print(f"\n📋 Found {len(found)} local adapter directories")
    return found


def _make_local_record(adapter_dir: Path) -> AdapterRecord:
    """Create an AdapterRecord from a local adapter directory."""
    # Try to read adapter_config.json for metadata
    config_path = adapter_dir / "adapter_config.json"
    base_model = "unknown"
    task = "local"

    if config_path.exists():
        try:
            with open(config_path) as f:
                cfg = json.load(f)
            base_model = cfg.get("base_model_name_or_path", "unknown") or "unknown"
        except Exception:
            pass

    # Use directory path as a readable identifier
    repo_id = str(adapter_dir.name)
    # Try to make a more informative name from parent dirs
    parts = adapter_dir.parts
    if len(parts) >= 2:
        repo_id = f"{parts[-2]}/{parts[-1]}"

    return AdapterRecord(
        repo_id=repo_id,
        base_model=base_model,
        task=task,
    )


def run_study(
    output_dir: Path,
    cache_dir: Path,
    max_adapters: int = 30,
    max_per_base: int = 5,
    min_downloads: int = 10,
    local_dirs: list[Path] | None = None,
) -> None:
    """Run the full broader benchmarks study."""
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Check for existing results (idempotent resume)
    manifest_path = output_dir / "manifest.json"
    results_path = output_dir / "study14_broader_benchmarks.json"
    existing_audit_data: dict[str, dict[str, Any]] = {}

    if results_path.exists():
        print(f"\n📂 Found existing results at {results_path}, loading for resume...")
        with open(results_path) as f:
            existing = json.load(f)
        existing_audit_data = existing.get("adapter_audits", {})
        print(f"   {len(existing_audit_data)} adapters already audited")

    # ---------- Mode: Local directories ----------
    if local_dirs:
        print("\n" + "=" * 72)
        print("MODE: LOCAL ADAPTER SCAN")
        print("=" * 72)

        local_found = scan_local_adapters(local_dirs)
        records = [rec for rec, _ in local_found]
        adapter_paths = {rec.repo_id: path for rec, path in local_found}

        # Mark all as "downloaded" (they're already local)
        for r in records:
            if r.repo_id not in existing_audit_data:
                r.status = "downloaded"
            else:
                r.status = "audited"

    else:
        # ---------- Mode: HuggingFace discovery ----------
        # Step 1: Discover adapters
        print("\n" + "=" * 72)
        print("STEP 1: DISCOVER ADAPTERS")
        print("=" * 72)

        records = discover_adapters(
            max_per_base=max_per_base,
            max_total=max_adapters,
            min_downloads=min_downloads,
        )

        if not records:
            print("❌ No adapters found. Check network connectivity and HuggingFace Hub access.")
            return

        # Step 2: Download adapters
        print("\n" + "=" * 72)
        print("STEP 2: DOWNLOAD ADAPTERS")
        print("=" * 72)

        adapter_paths: dict[str, Path] = {}
        for i, record in enumerate(records):
            if record.repo_id in existing_audit_data:
                print(f"\n[{i + 1}/{len(records)}] ⏭️  {record.repo_id} — already audited, skipping download")
                record.status = "audited"
                continue

            print(f"\n[{i + 1}/{len(records)}] 📥 Downloading: {record.repo_id}")
            try:
                adapter_path, size_mb = download_adapter(record.repo_id, cache_dir)
                record.download_size_mb = size_mb
                record.status = "downloaded"
                adapter_paths[record.repo_id] = adapter_path
                print(f"   ✅ Downloaded ({size_mb:.1f} MB)")
            except Exception as e:
                record.status = "failed"
                record.issues.append(f"Download error: {e}")
                print(f"   ❌ Failed: {e}")

        downloaded = [r for r in records if r.status == "downloaded"]
        already_done = [r for r in records if r.repo_id in existing_audit_data]
        print(
            f"\n📊 Downloaded: {len(downloaded)}, Already audited: {len(already_done)}, "
            f"Failed: {len([r for r in records if r.status == 'failed'])}"
        )

    # Step 3: Audit each adapter
    print("\n" + "=" * 72)
    print("STEP 3: AUDIT ADAPTERS")
    print("=" * 72)

    audit_data: dict[str, dict[str, Any]] = dict(existing_audit_data)
    total_audit_time = 0.0

    for i, record in enumerate(records):
        if record.repo_id in existing_audit_data:
            continue

        if record.status != "downloaded":
            continue

        # Get adapter path
        if record.repo_id in adapter_paths:
            ap = adapter_paths[record.repo_id]
        else:
            ap = cache_dir / record.repo_id.replace("/", "--")

        print(f"\n[{i + 1}/{len(records)}] 🔬 Auditing: {record.repo_id}")

        result_dict, record = audit_adapter(record.repo_id, ap, record)

        if result_dict is not None:
            audit_data[record.repo_id] = result_dict
            total_audit_time += record.audit_time_s or 0
            print(
                f"   ✅ {record.n_layers} layers, rank={record.rank}, "
                f"util={result_dict.get('utilization_mean', 0):.3f}, "
                f"time={record.audit_time_s:.1f}s"
            )
        else:
            print(f"   ❌ Failed: {'; '.join(record.issues)}")

    audited = [r for r in records if r.status == "audited" or r.repo_id in audit_data]
    failed = [r for r in records if r.status == "failed"]
    print(f"\n📊 Audited: {len(audited)}, Failed: {len(failed)}, Total audit time: {total_audit_time:.1f}s")

    if not audit_data:
        print("❌ No successful audits. Cannot generate analysis.")
        return

    # For records that came from existing data, fill in metadata
    for r in records:
        if r.repo_id in existing_audit_data and r.status != "failed":
            data = existing_audit_data[r.repo_id]
            r.status = "audited"
            r.n_layers = data.get("n_layers")
            r.total_params = data.get("total_lora_params")
            layer_rows = data.get("layer_data", {}).get("layer_rows", [])
            if layer_rows:
                r.rank = layer_rows[0].get("r")
                r.alpha = layer_rows[0].get("alpha")

    # Step 4: Analysis
    print("\n" + "=" * 72)
    print("STEP 4: ANALYSIS")
    print("=" * 72)

    analysis = analyze_results(records, audit_data)

    # Print key findings to console
    gu = analysis.get("global_utilization", {})
    if gu.get("n", 0) > 0:
        print(f"\n🎯 Global utilization: mean={gu['mean']:.3f}, median={gu['median']:.3f}, std={gu['std']:.3f}")

    bmt = analysis.get("by_module_type", {})
    for mt in ("attn", "mlp"):
        info = bmt.get(mt, {}).get("utilization", {})
        if info.get("n", 0) > 0:
            print(f"   {mt:>4}: util={info['mean']:.3f} ({info['n']} layers)")

    cp = analysis.get("compression_potential", {})
    if cp:
        print(f"\n📉 {cp.get('interpretation', '')}")

    ruc = analysis.get("rank_utilization_correlation", {})
    if ruc:
        print(f"📊 {ruc.get('interpretation', '')}")

    # Step 5: Save outputs
    print("\n" + "=" * 72)
    print("STEP 5: SAVE RESULTS")
    print("=" * 72)

    # Full JSON dataset
    study_output = {
        "study_metadata": {
            "study_id": "study14_broader_benchmarks",
            "title": "Spectral Audits of Real-World LoRA Adapters",
            "date": datetime.now(timezone.utc).isoformat(),
            "n_adapters_discovered": len(records),
            "n_adapters_audited": len([r for r in records if r.repo_id in audit_data]),
            "n_adapters_failed": len(failed),
            "total_audit_time_s": total_audit_time,
            "gradience_version": "0.11.0",
        },
        "adapter_records": [
            {
                "repo_id": r.repo_id,
                "base_model": r.base_model,
                "task": r.task,
                "rank": r.rank,
                "alpha": r.alpha,
                "n_layers": r.n_layers,
                "total_params": r.total_params,
                "download_size_mb": r.download_size_mb,
                "status": r.status,
                "audit_time_s": r.audit_time_s,
                "issues": r.issues,
            }
            for r in records
        ],
        "adapter_audits": audit_data,
        "analysis": analysis,
    }

    with open(results_path, "w") as f:
        json.dump(study_output, f, indent=2, default=str)
    print(f"✅ Full results: {results_path}")

    # Manifest (lightweight tracking)
    with open(manifest_path, "w") as f:
        json.dump(
            {
                "adapters": [
                    {
                        "repo_id": r.repo_id,
                        "base_model": r.base_model,
                        "task": r.task,
                        "rank": r.rank,
                        "status": r.status,
                    }
                    for r in records
                ]
            },
            f,
            indent=2,
        )
    print(f"✅ Manifest: {manifest_path}")

    # Markdown report
    report_path = output_dir / "study14_summary.md"
    report_md = generate_markdown_report(records, audit_data, analysis)
    with open(report_path, "w") as f:
        f.write(report_md)
    print(f"✅ Report: {report_path}")

    print("\n" + "=" * 72)
    print("STUDY 14 COMPLETE")
    print("=" * 72)


# ============================================================================
# CLI
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Study 14: Broader Benchmarks — Spectral Audits of Real-World LoRA Adapters"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to save study results",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("/tmp/gradience_adapter_cache"),
        help="Directory to cache downloaded adapters (default: /tmp/gradience_adapter_cache)",
    )
    parser.add_argument(
        "--max-adapters",
        type=int,
        default=30,
        help="Maximum number of adapters to audit (default: 30)",
    )
    parser.add_argument(
        "--max-per-base",
        type=int,
        default=5,
        help="Maximum adapters per base model (default: 5)",
    )
    parser.add_argument(
        "--min-downloads",
        type=int,
        default=10,
        help="Minimum downloads to consider an adapter (default: 10)",
    )
    parser.add_argument(
        "--local-dirs",
        type=Path,
        nargs="+",
        default=None,
        help="Scan local directories for adapter dirs instead of downloading from HF Hub. "
        "Each path can be a direct adapter dir or a parent dir to scan recursively.",
    )
    args = parser.parse_args()

    run_study(
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        max_adapters=args.max_adapters,
        max_per_base=args.max_per_base,
        min_downloads=args.min_downloads,
        local_dirs=args.local_dirs,
    )


if __name__ == "__main__":
    main()
