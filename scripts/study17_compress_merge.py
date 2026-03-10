#!/usr/bin/env python3
"""
Study 17: Compress-Then-Merge for LoRA Adapters — Phase 1 (CPU).

Tests whether spectral compression before merging improves LoRA merge
outcomes relative to full-rank baselines.  Compares 5 merge conditions
(A–E) across 4 primary pairs + 1 boundary pair of public Llama-2-7B
adapters chosen to represent balanced, moderate-imbalance, and
severe-imbalance regimes.

Merge conditions:
    A. Full-rank naive          — 0.5/0.5 uniform_linear
    B. Full-rank norm-equalized — global Frobenius-aware reweighting
    C. Full-rank recommended    — audit_aware (6-branch layerwise engine)
    D. Compress + norm-equalized
    E. Compress + recommended

Phase 1 (this script) runs on CPU and produces:
    - Merge audit for each pair (original + compressed)
    - SVD compression reports (95% energy retention)
    - Merged adapter artifacts for all 5 conditions
    - Spectral outcome metrics (Q_min, D, cosine retention)
    - Merged adapter audit summaries
    - Per-pair comparison JSON
    - Study-level summary JSON

Phase 2 (study17_perplexity.py) loads merged adapters on GPU for
downstream behavioral evaluation.

Usage:
    python scripts/study17_compress_merge.py \\
        --output-dir results/study17 \\
        --cache-dir ~/.cache/gradience/adapters \\
        --verbose

    # Single pair dry-run:
    python scripts/study17_compress_merge.py \\
        --output-dir results/study17 \\
        --pairs pair_02 \\
        --verbose
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

logger = logging.getLogger("study17")


# ═══════════════════════════════════════════════════════════════════════════
# Pair definitions
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class AdapterPairDef:
    """Definition of an adapter pair for the study."""

    pair_id: str
    repo_a: str
    repo_b: str
    label_a: str
    label_b: str
    task_a: str
    task_b: str
    expected_verdict: str
    is_boundary: bool = False  # Pair 06: analyzed separately


PAIRS: list[AdapterPairDef] = [
    AdapterPairDef(
        pair_id="pair_01",
        repo_a="LoRA-TMLR-2024/metamath-lora-rank-16-alpha-32",
        repo_b="LoRA-TMLR-2024/openwebmath-lora-rank-16-20B-tokens",
        label_a="metamath-r16",
        label_b="openwebmath-r16",
        task_a="math",
        task_b="math",
        expected_verdict="redundant",
    ),
    AdapterPairDef(
        pair_id="pair_02",
        repo_a="LoRA-TMLR-2024/metamath-lora-rank-16-alpha-32",
        repo_b="LoRA-TMLR-2024/magicoder-lora-rank-16-alpha-32",
        label_a="metamath-r16",
        label_b="magicoder-r16",
        task_a="math",
        task_b="code",
        expected_verdict="moderate",
    ),
    AdapterPairDef(
        pair_id="pair_03",
        repo_a="LoRA-TMLR-2024/magicoder-lora-rank-16-alpha-32",
        repo_b="AIRLab-POLIMI/llama-2-7b-chat-hf-btgenbot-adapter",
        label_a="magicoder-r16",
        label_b="btgenbot-r8",
        task_a="code",
        task_b="chat",
        expected_verdict="conflicting",
    ),
    AdapterPairDef(
        pair_id="pair_04",
        repo_a="LoRA-TMLR-2024/openwebmath-lora-rank-64-20B-tokens",
        repo_b="AIRLab-POLIMI/llama-2-7b-chat-hf-btgenbot-adapter",
        label_a="openwebmath-r64",
        label_b="btgenbot-r8",
        task_a="math",
        task_b="chat",
        expected_verdict="imbalanced",
    ),
    AdapterPairDef(
        pair_id="pair_06",
        repo_a="shivanikerai/Llama-2-7b-chat-hf-adapter-cat-subcat-mapping-v2.0",
        repo_b="AIRLab-POLIMI/llama-2-7b-chat-hf-btgenbot-adapter",
        label_a="catsubcat-r16",
        label_b="btgenbot-r8",
        task_a="chat",
        task_b="chat",
        expected_verdict="mixed",
        is_boundary=True,
    ),
]

# All pairs get all 5 conditions.
ENERGY_THRESHOLD = 0.95


# ═══════════════════════════════════════════════════════════════════════════
# Download
# ═══════════════════════════════════════════════════════════════════════════


def download_adapter(repo_id: str, cache_dir: Path) -> Path:
    """Download a PEFT adapter from HuggingFace Hub.  Returns local path."""
    from huggingface_hub import snapshot_download

    local_dir = cache_dir / repo_id.replace("/", "--")

    config_candidates = list(local_dir.glob("adapter_config*"))
    weight_candidates = list(local_dir.glob("adapter_model*"))
    if config_candidates and weight_candidates:
        logger.info("  Already cached: %s", local_dir)
        return local_dir

    logger.info("  Downloading %s ...", repo_id)
    snapshot_download(
        repo_id,
        local_dir=str(local_dir),
        allow_patterns=["adapter_config*", "adapter_model*", "pytorch_model.bin"],
        ignore_patterns=["*.md", "*.txt", ".gitattributes", "tokenizer*", "special_tokens*", "training_args*"],
    )
    return local_dir


# ═══════════════════════════════════════════════════════════════════════════
# Compression: energy-threshold SVD truncation
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class CompressionReport:
    """Per-adapter compression summary."""

    adapter_label: str
    nominal_rank: int
    target_rank: int  # rank chosen by energy threshold
    energy_retained: float
    per_layer_ranks: list[dict[str, Any]]  # [{layer, nominal, effective_at_threshold, energy}]
    mean_effective_rank: float
    mean_reduction: float  # nominal - effective
    n_layers_heavily_truncated: int  # layers where effective < nominal/2
    source_dir: str
    compressed_dir: str


def compute_effective_rank_at_threshold(peft_dir: Path, threshold: float = 0.95) -> tuple[int, list[dict]]:
    """Compute the smallest rank k such that >=threshold energy is retained.

    Examines each LoRA layer's singular values and finds k_i per layer,
    then returns max(k_i) as the global target rank (so no layer loses
    more than (1-threshold) energy).
    """
    from gradience.vnext.audit.lora_audit import find_peft_files, load_adapter_state_dict, load_peft_adapter_config
    from gradience.vnext.svd_truncate import _parse_and_pair_lora_matrices

    config_path, weights_path, issues = find_peft_files(peft_dir)
    if config_path is None or weights_path is None:
        raise FileNotFoundError(f"Could not find PEFT files in {peft_dir}. Issues: {issues}")

    load_peft_adapter_config(config_path)  # validate config exists
    state_dict = load_adapter_state_dict(weights_path)
    pairs = _parse_and_pair_lora_matrices(state_dict)

    per_layer = []
    max_effective = 1

    for base_key, pair_data in pairs.items():
        _key_a, lora_A = pair_data["A"]
        _key_b, lora_B = pair_data["B"]

        # Compute singular values of dW = B @ A via small matrix
        A_f = lora_A.cpu().to(torch.float32)
        B_f = lora_B.cpu().to(torch.float32)

        Qb, Rb = torch.linalg.qr(B_f)
        Qa, Ra = torch.linalg.qr(A_f.T)
        M = Rb @ Ra.T
        S = torch.linalg.svdvals(M)

        total_energy = (S**2).sum().item()
        if total_energy < 1e-12:
            per_layer.append({"layer": base_key, "nominal": int(S.shape[0]), "effective": 1, "energy": 0.0})
            continue

        cumulative = torch.cumsum(S**2, dim=0) / total_energy
        # k = smallest index where cumulative >= threshold
        above = (cumulative >= threshold).nonzero(as_tuple=True)[0]
        if len(above) > 0:
            k = int(above[0].item()) + 1
        else:
            k = int(S.shape[0])

        per_layer.append({
            "layer": base_key,
            "nominal": int(S.shape[0]),
            "effective": k,
            "energy": float(cumulative[k - 1].item()),
        })
        max_effective = max(max_effective, k)

    return max_effective, per_layer


def compress_adapter(
    source_dir: Path,
    out_dir: Path,
    adapter_label: str,
    threshold: float = ENERGY_THRESHOLD,
    verbose: bool = False,
) -> CompressionReport:
    """Compress an adapter using energy-threshold SVD truncation.

    1. Compute per-layer effective rank at the given energy threshold.
    2. Use max(effective_ranks) as global target_rank.
    3. Run svd_truncate_peft_dir with that target_rank.
    """
    from gradience.vnext.audit.lora_audit import find_peft_files, load_peft_adapter_config
    from gradience.vnext.svd_truncate import svd_truncate_peft_dir

    # Step 1: determine nominal rank
    config_path, _, _ = find_peft_files(source_dir)
    config = load_peft_adapter_config(config_path)
    nominal_rank = config.r

    # Step 2: compute effective rank at threshold
    target_rank, per_layer_info = compute_effective_rank_at_threshold(source_dir, threshold)

    if verbose:
        print(f"    {adapter_label}: nominal={nominal_rank}, effective@{threshold:.0%}={target_rank}")
        for layer in per_layer_info:
            if layer["effective"] < layer["nominal"]:
                print(f"      {layer['layer'].split('.')[-2]}: {layer['nominal']} → {layer['effective']}")

    # Step 3: truncate (only if target < nominal)
    if target_rank >= nominal_rank:
        # No compression needed — copy as-is
        import shutil

        if out_dir.exists():
            shutil.rmtree(out_dir)
        shutil.copytree(source_dir, out_dir)
        if verbose:
            print(f"    No compression needed (effective rank {target_rank} >= nominal {nominal_rank})")

        return CompressionReport(
            adapter_label=adapter_label,
            nominal_rank=nominal_rank,
            target_rank=nominal_rank,
            energy_retained=1.0,
            per_layer_ranks=per_layer_info,
            mean_effective_rank=float(np.mean([l["effective"] for l in per_layer_info])),
            mean_reduction=0.0,
            n_layers_heavily_truncated=0,
            source_dir=str(source_dir),
            compressed_dir=str(out_dir),
        )

    report = svd_truncate_peft_dir(source_dir, out_dir, target_rank)

    mean_eff = float(np.mean([l["effective"] for l in per_layer_info]))
    mean_reduction = nominal_rank - mean_eff
    n_heavy = sum(1 for l in per_layer_info if l["effective"] < nominal_rank / 2)

    return CompressionReport(
        adapter_label=adapter_label,
        nominal_rank=nominal_rank,
        target_rank=target_rank,
        energy_retained=report.energy_retained,
        per_layer_ranks=per_layer_info,
        mean_effective_rank=round(mean_eff, 2),
        mean_reduction=round(mean_reduction, 2),
        n_layers_heavily_truncated=n_heavy,
        source_dir=str(source_dir),
        compressed_dir=str(out_dir),
    )


# ═══════════════════════════════════════════════════════════════════════════
# Spectral evaluation (reused from study16)
# ═══════════════════════════════════════════════════════════════════════════


def audit_merged_adapter(merged_dir: Path) -> dict[str, Any]:
    """Run spectral audit on a merged adapter and return summary metrics."""
    from gradience.vnext.audit.lora_audit import audit_lora_peft_dir

    result = audit_lora_peft_dir(merged_dir, compute_dtype=torch.float64, compute_udr=False)

    layers = result.layers if hasattr(result, "layers") else []
    if not layers:
        return {"n_layers": 0, "mean_stable_rank": 0.0, "mean_utilisation": 0.0}

    stable_ranks = [l.stable_rank for l in layers if hasattr(l, "stable_rank")]
    utilisations = [l.utilization for l in layers if hasattr(l, "utilization")]
    energy_ranks = [l.energy_rank_90 for l in layers if hasattr(l, "energy_rank_90")]

    nominal_rank = getattr(result, "nominal_rank", 0)
    if not nominal_rank and hasattr(result, "adapter_config") and result.adapter_config:
        nominal_rank = getattr(result.adapter_config, "r", 0)
    if not nominal_rank and layers:
        nominal_rank = layers[0].r

    return {
        "n_layers": len(layers),
        "nominal_rank": nominal_rank,
        "mean_stable_rank": round(sum(stable_ranks) / len(stable_ranks), 4) if stable_ranks else 0.0,
        "mean_utilisation": round(sum(utilisations) / len(utilisations), 4) if utilisations else 0.0,
        "mean_energy_rank_90": round(sum(energy_ranks) / len(energy_ranks), 2) if energy_ranks else 0.0,
        "median_stable_rank": round(sorted(stable_ranks)[len(stable_ranks) // 2], 4) if stable_ranks else 0.0,
        "median_utilisation": round(sorted(utilisations)[len(utilisations) // 2], 4) if utilisations else 0.0,
    }


def compute_spectral_outcomes(
    adapter_a_dir: Path,
    adapter_b_dir: Path,
    merged_dir: Path,
) -> dict[str, Any]:
    """Compute spectral retention and dominance metrics (Q_min, D, cosine)."""
    from gradience.vnext.merge.io import extract_factors, load_adapter, match_layers

    info_a = load_adapter(adapter_a_dir)
    info_b = load_adapter(adapter_b_dir)
    info_m = load_adapter(merged_dir)

    # Frobenius norms
    frob_sq_a = frob_sq_b = frob_sq_m = 0.0

    for prefix in info_a.module_prefixes:
        try:
            A, B, r = extract_factors(info_a, prefix)
            dW = (info_a.alpha / r) * B @ A
            frob_sq_a += torch.norm(dW, p="fro").item() ** 2
        except (KeyError, RuntimeError):
            continue

    for prefix in info_b.module_prefixes:
        try:
            A, B, r = extract_factors(info_b, prefix)
            dW = (info_b.alpha / r) * B @ A
            frob_sq_b += torch.norm(dW, p="fro").item() ** 2
        except (KeyError, RuntimeError):
            continue

    for prefix in info_m.module_prefixes:
        try:
            A, B, r = extract_factors(info_m, prefix)
            dW = (info_m.alpha / r) * B @ A
            frob_sq_m += torch.norm(dW, p="fro").item() ** 2
        except (KeyError, RuntimeError):
            continue

    frob_a = frob_sq_a**0.5
    frob_b = frob_sq_b**0.5
    frob_m = frob_sq_m**0.5

    # Cosine similarity: merged vs each source
    shared_a, _, _ = match_layers(info_a, info_m)
    shared_b, _, _ = match_layers(info_b, info_m)

    cos_sims_a = []
    cos_sims_b = []

    for prefix in shared_a:
        try:
            A_a, B_a, r_a = extract_factors(info_a, prefix)
            A_m, B_m, r_m = extract_factors(info_m, prefix)
            dW_a = (info_a.alpha / r_a) * B_a @ A_a
            dW_m = (info_m.alpha / r_m) * B_m @ A_m
            cos = torch.nn.functional.cosine_similarity(dW_a.flatten().unsqueeze(0), dW_m.flatten().unsqueeze(0)).item()
            cos_sims_a.append(cos)
        except (KeyError, RuntimeError):
            continue

    for prefix in shared_b:
        try:
            A_b, B_b, r_b = extract_factors(info_b, prefix)
            A_m, B_m, r_m = extract_factors(info_m, prefix)
            dW_b = (info_b.alpha / r_b) * B_b @ A_b
            dW_m = (info_m.alpha / r_m) * B_m @ A_m
            cos = torch.nn.functional.cosine_similarity(dW_b.flatten().unsqueeze(0), dW_m.flatten().unsqueeze(0)).item()
            cos_sims_b.append(cos)
        except (KeyError, RuntimeError):
            continue

    mean_cos_a = sum(cos_sims_a) / len(cos_sims_a) if cos_sims_a else 0.0
    mean_cos_b = sum(cos_sims_b) / len(cos_sims_b) if cos_sims_b else 0.0

    eps = 1e-12
    D = abs(mean_cos_a - mean_cos_b) / (abs(mean_cos_a) + abs(mean_cos_b) + eps)
    Q_min = min(mean_cos_a, mean_cos_b)

    return {
        "frob_a": round(frob_a, 6),
        "frob_b": round(frob_b, 6),
        "frob_merged": round(frob_m, 6),
        "frob_retention_a": round(frob_m / max(frob_a, eps), 6),
        "frob_retention_b": round(frob_m / max(frob_b, eps), 6),
        "mean_cosine_sim_a": round(mean_cos_a, 6),
        "mean_cosine_sim_b": round(mean_cos_b, 6),
        "Q_min_cosine": round(Q_min, 6),
        "D_cosine": round(D, 6),
        "is_dominated": D > 0.2,
        "n_layers_compared_a": len(cos_sims_a),
        "n_layers_compared_b": len(cos_sims_b),
    }


def compute_global_frob_coefficients(dir_a: Path, dir_b: Path) -> tuple[float, float]:
    """Compute global Frobenius-proportional merge coefficients.

    Inverse-proportional: weaker adapter gets more weight.
    """
    from gradience.vnext.merge.io import extract_factors, load_adapter

    info_a = load_adapter(dir_a)
    info_b = load_adapter(dir_b)

    frob_sq_a = frob_sq_b = 0.0

    for prefix in info_a.module_prefixes:
        try:
            A, B, r = extract_factors(info_a, prefix)
            dW = (info_a.alpha / r) * B @ A
            frob_sq_a += torch.norm(dW, p="fro").item() ** 2
        except (KeyError, RuntimeError):
            continue

    for prefix in info_b.module_prefixes:
        try:
            A, B, r = extract_factors(info_b, prefix)
            dW = (info_b.alpha / r) * B @ A
            frob_sq_b += torch.norm(dW, p="fro").item() ** 2
        except (KeyError, RuntimeError):
            continue

    frob_a = frob_sq_a**0.5
    frob_b = frob_sq_b**0.5
    total = frob_a + frob_b

    if total < 1e-12:
        return (0.5, 0.5)

    return (frob_b / total, frob_a / total)


# ═══════════════════════════════════════════════════════════════════════════
# Per-pair pipeline
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class ConditionResult:
    """Metrics from one merge condition."""

    condition_name: str
    strategy: str
    is_compressed: bool
    merge_time_s: float = 0.0
    mean_reconstruction_error: float = 0.0
    max_reconstruction_error: float = 0.0
    merged_audit: dict[str, Any] = field(default_factory=dict)
    spectral_outcomes: dict[str, Any] = field(default_factory=dict)
    recon_error_p50: float = 0.0
    recon_error_p90: float = 0.0
    error: str | None = None


@dataclass
class PairResult:
    """Aggregated results for one adapter pair across all conditions."""

    pair_id: str
    label_a: str
    label_b: str
    task_a: str
    task_b: str
    expected_verdict: str
    is_boundary: bool
    actual_overall_verdict: str = ""
    actual_compatibility_score: float = 0.0
    audit_verdict_counts: dict[str, int] = field(default_factory=dict)
    recommendation_summary: dict[str, Any] = field(default_factory=dict)
    compression_a: dict[str, Any] = field(default_factory=dict)
    compression_b: dict[str, Any] = field(default_factory=dict)
    conditions: dict[str, dict[str, Any]] = field(default_factory=dict)
    error: str | None = None


def run_pair(
    pair_def: AdapterPairDef,
    cache_dir: Path,
    output_dir: Path,
    verbose: bool = False,
) -> PairResult:
    """Run all 5 merge conditions for one adapter pair."""
    from gradience.vnext.merge import (
        execute_merge,
        merge_audit,
        plan_from_audit,
        recommend_merge,
    )

    result = PairResult(
        pair_id=pair_def.pair_id,
        label_a=pair_def.label_a,
        label_b=pair_def.label_b,
        task_a=pair_def.task_a,
        task_b=pair_def.task_b,
        expected_verdict=pair_def.expected_verdict,
        is_boundary=pair_def.is_boundary,
    )

    pair_dir = output_dir / pair_def.pair_id
    pair_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"\n{'=' * 70}")
        print(f"  Pair: {pair_def.label_a}  ×  {pair_def.label_b}")
        print(f"  Expected: {pair_def.expected_verdict}  {'[BOUNDARY]' if pair_def.is_boundary else ''}")
        print(f"{'=' * 70}")

    # --- Download adapters ---
    try:
        dir_a = download_adapter(pair_def.repo_a, cache_dir)
        dir_b = download_adapter(pair_def.repo_b, cache_dir)
    except Exception as e:
        result.error = f"Download failed: {e}"
        logger.error("Download failed for %s: %s", pair_def.pair_id, e)
        return result

    # --- Audit original pair ---
    if verbose:
        print("\n  Running merge audit (original)...")

    try:
        audit_dir = pair_dir / "audit_original"
        report = merge_audit(str(dir_a), str(dir_b), output_dir=str(audit_dir), verbose=verbose)
    except Exception as e:
        result.error = f"Merge audit failed: {e}"
        logger.error("Audit failed for %s: %s", pair_def.pair_id, e)
        return result

    result.actual_overall_verdict = report.aggregate.get("overall_verdict", "unknown")
    result.actual_compatibility_score = report.aggregate.get("compatibility_score", 0.0)

    verdict_counts: dict[str, int] = {}
    for lv in report.layer_verdicts:
        v = lv.get("verdict", "unknown")
        verdict_counts[v] = verdict_counts.get(v, 0) + 1
    result.audit_verdict_counts = verdict_counts

    rec = recommend_merge(report)
    result.recommendation_summary = {
        "overall_strategy": rec.overall_strategy,
        "overall_risk": rec.overall_risk,
        "compression_needed": rec.compression_needed,
        "n_layers_needing_compression": rec.n_layers_needing_compression,
    }

    rank_a = report.adapter_a.get("rank", 16)
    rank_b = report.adapter_b.get("rank", 16)
    output_rank = max(rank_a, rank_b)
    output_alpha = report.adapter_a.get("alpha", 32.0)

    if verbose:
        print(f"  Overall verdict: {result.actual_overall_verdict}")
        print(f"  Compatibility score: {result.actual_compatibility_score:.3f}")
        print(f"  Verdict counts: {verdict_counts}")
        print(f"  Output rank: {output_rank}, alpha: {output_alpha}")

    # --- Compress adapters ---
    if verbose:
        print(f"\n  Compressing adapters (energy threshold: {ENERGY_THRESHOLD:.0%})...")

    comp_dir = pair_dir / "compressed"
    comp_dir.mkdir(parents=True, exist_ok=True)

    try:
        comp_report_a = compress_adapter(dir_a, comp_dir / "a", pair_def.label_a, ENERGY_THRESHOLD, verbose)
        comp_report_b = compress_adapter(dir_b, comp_dir / "b", pair_def.label_b, ENERGY_THRESHOLD, verbose)
        result.compression_a = asdict(comp_report_a)
        result.compression_b = asdict(comp_report_b)
    except Exception as e:
        result.error = f"Compression failed: {e}"
        logger.error("Compression failed for %s: %s", pair_def.pair_id, e, exc_info=True)
        return result

    compressed_a = comp_dir / "a"
    compressed_b = comp_dir / "b"

    # --- Audit compressed pair ---
    if verbose:
        print("\n  Running merge audit (compressed)...")

    try:
        audit_comp_dir = pair_dir / "audit_compressed"
        compressed_report = merge_audit(
            str(compressed_a), str(compressed_b), output_dir=str(audit_comp_dir), verbose=verbose
        )
    except Exception as e:
        logger.warning("Compressed audit failed for %s: %s", pair_def.pair_id, e)
        compressed_report = None

    # --- Compute norm-equalized coefficients for both original and compressed ---
    try:
        coeff_a_orig, coeff_b_orig = compute_global_frob_coefficients(dir_a, dir_b)
        if verbose:
            print(f"  Norm-eq coeff (original):   A={coeff_a_orig:.4f}  B={coeff_b_orig:.4f}")
    except Exception as e:
        logger.warning("Norm-eq coeff (original) failed: %s", e)
        coeff_a_orig, coeff_b_orig = 0.5, 0.5

    try:
        coeff_a_comp, coeff_b_comp = compute_global_frob_coefficients(compressed_a, compressed_b)
        if verbose:
            print(f"  Norm-eq coeff (compressed): A={coeff_a_comp:.4f}  B={coeff_b_comp:.4f}")
    except Exception as e:
        logger.warning("Norm-eq coeff (compressed) failed: %s", e)
        coeff_a_comp, coeff_b_comp = 0.5, 0.5

    # Determine compressed output rank
    comp_output_rank = max(comp_report_a.target_rank, comp_report_b.target_rank)
    comp_output_alpha = output_alpha  # Keep original alpha

    # --- Define conditions ---
    conditions = [
        # (name, strategy, adapter_a_dir, adapter_b_dir, report, extra_plan_kwargs, is_compressed)
        ("A_fullrank_naive", "uniform_linear", dir_a, dir_b, report, {"output_rank": output_rank, "output_alpha": output_alpha}, False),
        ("B_fullrank_normeq", "uniform_linear", dir_a, dir_b, report, {"output_rank": output_rank, "output_alpha": output_alpha, "coefficients": (coeff_a_orig, coeff_b_orig)}, False),
        ("C_fullrank_recommended", "audit_aware", dir_a, dir_b, report, {"output_rank": output_rank, "output_alpha": output_alpha}, False),
        ("D_compress_normeq", "uniform_linear", compressed_a, compressed_b, compressed_report or report, {"output_rank": comp_output_rank, "output_alpha": comp_output_alpha, "coefficients": (coeff_a_comp, coeff_b_comp)}, True),
        ("E_compress_recommended", "audit_aware", compressed_a, compressed_b, compressed_report or report, {"output_rank": comp_output_rank, "output_alpha": comp_output_alpha}, True),
    ]

    # --- Run conditions ---
    for cond_name, strategy, src_a, src_b, audit_report, plan_kwargs, is_compressed in conditions:
        if verbose:
            print(f"\n  --- Condition {cond_name} ({strategy}) ---")

        cond_dir = pair_dir / cond_name
        cond_dir.mkdir(parents=True, exist_ok=True)

        try:
            t0 = time.time()

            plan = plan_from_audit(strategy, audit_report, str(src_a), str(src_b), **plan_kwargs)
            merge_result = execute_merge(plan, str(cond_dir), verbose=verbose)

            merge_time = time.time() - t0

            if verbose:
                print(f"    Merge: {merge_time:.1f}s  recon={merge_result.mean_reconstruction_error:.6f}")

            # Save merge result JSON
            merge_result.to_json(cond_dir / "merge_result.json")

            # Audit merged adapter
            if verbose:
                print("    Auditing merged adapter...")
            merged_audit = audit_merged_adapter(cond_dir)

            # Spectral outcomes: always compare merged against ORIGINAL sources
            if verbose:
                print("    Computing spectral outcomes...")
            spectral = compute_spectral_outcomes(dir_a, dir_b, cond_dir)

            recon_errors = [lr.reconstruction_error for lr in merge_result.layer_results]

            cond_result = ConditionResult(
                condition_name=cond_name,
                strategy=strategy,
                is_compressed=is_compressed,
                merge_time_s=round(merge_time, 2),
                mean_reconstruction_error=round(merge_result.mean_reconstruction_error, 6),
                max_reconstruction_error=round(merge_result.max_reconstruction_error, 6),
                merged_audit=merged_audit,
                spectral_outcomes=spectral,
                recon_error_p50=round(sorted(recon_errors)[len(recon_errors) // 2], 6) if recon_errors else 0.0,
                recon_error_p90=round(sorted(recon_errors)[int(len(recon_errors) * 0.9)], 6) if recon_errors else 0.0,
            )

            result.conditions[cond_name] = asdict(cond_result)

            if verbose:
                so = spectral
                print(f"    Q_min={so['Q_min_cosine']:.4f}  D={so['D_cosine']:.4f}")
                print(f"    cos_A={so['mean_cosine_sim_a']:.4f}  cos_B={so['mean_cosine_sim_b']:.4f}")
                print(f"    utilisation={merged_audit.get('mean_utilisation', 0):.4f}")

            # Clean up large safetensors to save disk
            # (keep adapter_config.json and merge_result.json for Phase 2 re-merge)
            # NOTE: Phase 2 will need to re-run merges. Delete weights to save ~GB of disk.
            for sf in cond_dir.glob("*.safetensors"):
                sf.unlink()
                if verbose:
                    print(f"    Cleaned up {sf.name}")

        except Exception as e:
            result.conditions[cond_name] = {"error": str(e), "condition_name": cond_name}
            logger.error("Condition %s failed for %s: %s", cond_name, pair_def.pair_id, e, exc_info=True)

    # Clean up compressed adapter weights (not needed after merge)
    for sf in comp_dir.rglob("*.safetensors"):
        sf.unlink()
    for bf in comp_dir.rglob("*.bin"):
        bf.unlink()

    return result


# ═══════════════════════════════════════════════════════════════════════════
# Comparison tables
# ═══════════════════════════════════════════════════════════════════════════


def compute_comparison_table(results: list[PairResult]) -> list[dict[str, Any]]:
    """Build comparison table: full-rank vs compressed for each pair."""
    rows = []

    for r in results:
        if r.error:
            rows.append({"pair_id": r.pair_id, "label": f"{r.label_a} × {r.label_b}", "error": r.error})
            continue

        def _get(cond_name: str, metric_path: str, default: float = 0.0, _r: PairResult = r) -> float:
            cond = _r.conditions.get(cond_name, {})
            if cond.get("error"):
                return default
            parts = metric_path.split(".")
            val = cond
            for p in parts:
                if isinstance(val, dict):
                    val = val.get(p, default)
                else:
                    return default
            return float(val) if val is not None else default

        row = {
            "pair_id": r.pair_id,
            "label": f"{r.label_a} × {r.label_b}",
            "is_boundary": r.is_boundary,
            "expected_verdict": r.expected_verdict,
            "actual_verdict": r.actual_overall_verdict,
            "compatibility_score": r.actual_compatibility_score,
            # Full-rank norm-equalized (B)
            "B_Q_min": _get("B_fullrank_normeq", "spectral_outcomes.Q_min_cosine"),
            "B_D": _get("B_fullrank_normeq", "spectral_outcomes.D_cosine"),
            "B_util": _get("B_fullrank_normeq", "merged_audit.mean_utilisation"),
            "B_recon": _get("B_fullrank_normeq", "mean_reconstruction_error"),
            # Compress + norm-equalized (D)
            "D_Q_min": _get("D_compress_normeq", "spectral_outcomes.Q_min_cosine"),
            "D_D": _get("D_compress_normeq", "spectral_outcomes.D_cosine"),
            "D_util": _get("D_compress_normeq", "merged_audit.mean_utilisation"),
            "D_recon": _get("D_compress_normeq", "mean_reconstruction_error"),
            # Full-rank recommended (C)
            "C_Q_min": _get("C_fullrank_recommended", "spectral_outcomes.Q_min_cosine"),
            "C_D": _get("C_fullrank_recommended", "spectral_outcomes.D_cosine"),
            "C_util": _get("C_fullrank_recommended", "merged_audit.mean_utilisation"),
            "C_recon": _get("C_fullrank_recommended", "mean_reconstruction_error"),
            # Compress + recommended (E)
            "E_Q_min": _get("E_compress_recommended", "spectral_outcomes.Q_min_cosine"),
            "E_D": _get("E_compress_recommended", "spectral_outcomes.D_cosine"),
            "E_util": _get("E_compress_recommended", "merged_audit.mean_utilisation"),
            "E_recon": _get("E_compress_recommended", "mean_reconstruction_error"),
            # Deltas: compress - fullrank (positive = compression better for Q_min/util, negative better for D)
            "delta_normeq_Q_min": round(
                _get("D_compress_normeq", "spectral_outcomes.Q_min_cosine")
                - _get("B_fullrank_normeq", "spectral_outcomes.Q_min_cosine"),
                6,
            ),
            "delta_normeq_D": round(
                _get("B_fullrank_normeq", "spectral_outcomes.D_cosine")
                - _get("D_compress_normeq", "spectral_outcomes.D_cosine"),
                6,
            ),
            "delta_rec_Q_min": round(
                _get("E_compress_recommended", "spectral_outcomes.Q_min_cosine")
                - _get("C_fullrank_recommended", "spectral_outcomes.Q_min_cosine"),
                6,
            ),
            "delta_rec_D": round(
                _get("C_fullrank_recommended", "spectral_outcomes.D_cosine")
                - _get("E_compress_recommended", "spectral_outcomes.D_cosine"),
                6,
            ),
            # Compression summary
            "comp_a_nominal": r.compression_a.get("nominal_rank", 0),
            "comp_a_target": r.compression_a.get("target_rank", 0),
            "comp_a_energy": r.compression_a.get("energy_retained", 0.0),
            "comp_b_nominal": r.compression_b.get("nominal_rank", 0),
            "comp_b_target": r.compression_b.get("target_rank", 0),
            "comp_b_energy": r.compression_b.get("energy_retained", 0.0),
        }
        rows.append(row)

    return rows


def print_comparison_table(rows: list[dict[str, Any]]) -> None:
    """Pretty-print the primary comparison table (B vs D, C vs E)."""
    print("\n" + "=" * 120)
    print("  STUDY 17 — COMPRESS-THEN-MERGE COMPARISON TABLE")
    print("=" * 120)

    # Table 1: Structural comparison
    print("\n  Table 1: NormEq Full-Rank (B) vs Compress+NormEq (D)")
    print(f"  {'Pair':<38s}  {'Verdict':<12s}  "
          f"{'B Q_min':>8s}  {'D Q_min':>8s}  {'dQ':>7s}  "
          f"{'B D':>6s}  {'D D':>6s}  {'dD':>6s}")
    print("  " + "-" * 110)

    for row in rows:
        if "error" in row:
            print(f"  {row['label']:<38s}  ERROR: {row['error']}")
            continue
        flag = " *" if row.get("is_boundary") else ""
        print(f"  {row['label']:<38s}  {row['actual_verdict']:<12s}  "
              f"{row['B_Q_min']:>8.4f}  {row['D_Q_min']:>8.4f}  {row['delta_normeq_Q_min']:>+7.4f}  "
              f"{row['B_D']:>6.4f}  {row['D_D']:>6.4f}  {row['delta_normeq_D']:>+6.4f}{flag}")

    print("\n  Table 2: Recommended Full-Rank (C) vs Compress+Recommended (E)")
    print(f"  {'Pair':<38s}  {'Verdict':<12s}  "
          f"{'C Q_min':>8s}  {'E Q_min':>8s}  {'dQ':>7s}  "
          f"{'C D':>6s}  {'E D':>6s}  {'dD':>6s}")
    print("  " + "-" * 110)

    for row in rows:
        if "error" in row:
            continue
        flag = " *" if row.get("is_boundary") else ""
        print(f"  {row['label']:<38s}  {row['actual_verdict']:<12s}  "
              f"{row['C_Q_min']:>8.4f}  {row['E_Q_min']:>8.4f}  {row['delta_rec_Q_min']:>+7.4f}  "
              f"{row['C_D']:>6.4f}  {row['E_D']:>6.4f}  {row['delta_rec_D']:>+6.4f}{flag}")

    # Table 3: Compression summary
    print("\n  Table 3: Compression Summary")
    print(f"  {'Pair':<38s}  {'A nom':>5s}  {'A eff':>5s}  {'A energy':>8s}  "
          f"{'B nom':>5s}  {'B eff':>5s}  {'B energy':>8s}")
    print("  " + "-" * 80)

    for row in rows:
        if "error" in row:
            continue
        print(f"  {row['label']:<38s}  "
              f"{row['comp_a_nominal']:>5d}  {row['comp_a_target']:>5d}  {row['comp_a_energy']:>8.4f}  "
              f"{row['comp_b_nominal']:>5d}  {row['comp_b_target']:>5d}  {row['comp_b_energy']:>8.4f}")

    print("=" * 120)

    # Summary stats (primary pairs only)
    primary = [r for r in rows if "error" not in r and not r.get("is_boundary")]
    if primary:
        mean_dQ_ne = sum(r["delta_normeq_Q_min"] for r in primary) / len(primary)
        mean_dD_ne = sum(r["delta_normeq_D"] for r in primary) / len(primary)
        wins_Q_ne = sum(1 for r in primary if r["delta_normeq_Q_min"] > 0)
        wins_D_ne = sum(1 for r in primary if r["delta_normeq_D"] > 0)

        mean_dQ_r = sum(r["delta_rec_Q_min"] for r in primary) / len(primary)
        mean_dD_r = sum(r["delta_rec_D"] for r in primary) / len(primary)
        wins_Q_r = sum(1 for r in primary if r["delta_rec_Q_min"] > 0)
        wins_D_r = sum(1 for r in primary if r["delta_rec_D"] > 0)

        n = len(primary)
        print(f"\n  PRIMARY PAIRS ONLY ({n} pairs):")
        print(f"  NormEq:      mean dQ_min={mean_dQ_ne:+.4f} ({wins_Q_ne}/{n} improved)  mean dD={mean_dD_ne:+.4f} ({wins_D_ne}/{n} improved)")
        print(f"  Recommended: mean dQ_min={mean_dQ_r:+.4f} ({wins_Q_r}/{n} improved)  mean dD={mean_dD_r:+.4f} ({wins_D_r}/{n} improved)")
        print()


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════


def main() -> None:
    parser = argparse.ArgumentParser(description="Study 17: Compress-Then-Merge (Phase 1 — CPU)")
    parser.add_argument("--output-dir", type=Path, default=Path("results/study17"))
    parser.add_argument("--cache-dir", type=Path, default=Path.home() / ".cache" / "gradience" / "adapters")
    parser.add_argument("--pairs", type=str, nargs="*", default=None, help="Specific pair IDs (default: all)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    pairs_to_run = PAIRS
    if args.pairs:
        pairs_to_run = [p for p in PAIRS if p.pair_id in args.pairs]
        if not pairs_to_run:
            print(f"No matching pairs. Available: {[p.pair_id for p in PAIRS]}")
            sys.exit(1)

    print("Study 17: Compress-Then-Merge — Phase 1 (CPU)")
    print(f"  Pairs: {len(pairs_to_run)} ({', '.join(p.pair_id for p in pairs_to_run)})")
    print("  Conditions: A (naive), B (norm-eq), C (recommended), D (compress+norm-eq), E (compress+rec)")
    print(f"  Energy threshold: {ENERGY_THRESHOLD:.0%}")
    print(f"  Output: {args.output_dir}")
    print(f"  Cache:  {args.cache_dir}")

    t_start = time.time()
    all_results: list[PairResult] = []

    for i, pair_def in enumerate(pairs_to_run):
        print(f"\n[{i + 1}/{len(pairs_to_run)}] Processing {pair_def.pair_id}...")
        pair_result = run_pair(pair_def, args.cache_dir, args.output_dir, args.verbose)
        all_results.append(pair_result)

        # Save incremental pair result
        pair_path = args.output_dir / pair_def.pair_id / "pair_result.json"
        with open(pair_path, "w") as f:
            json.dump(asdict(pair_result), f, indent=2, default=str)

    total_time = time.time() - t_start

    # --- Comparison table ---
    comparison = compute_comparison_table(all_results)
    print_comparison_table(comparison)

    # --- Save summary ---
    study_output = {
        "study_metadata": {
            "study_id": "study17_compress_merge",
            "title": "Compress-Then-Merge for LoRA Adapters",
            "date": datetime.now(timezone.utc).isoformat(),
            "n_pairs": len(pairs_to_run),
            "n_primary_pairs": sum(1 for p in pairs_to_run if not p.is_boundary),
            "n_boundary_pairs": sum(1 for p in pairs_to_run if p.is_boundary),
            "conditions": ["A_fullrank_naive", "B_fullrank_normeq", "C_fullrank_recommended", "D_compress_normeq", "E_compress_recommended"],
            "energy_threshold": ENERGY_THRESHOLD,
            "total_time_s": round(total_time, 2),
            "base_model": "meta-llama/Llama-2-7b-hf",
            "phase": "phase_1_spectral",
        },
        "pair_results": [asdict(r) for r in all_results],
        "comparison_table": comparison,
    }

    output_path = args.output_dir / "study17_compress_merge.json"
    with open(output_path, "w") as f:
        json.dump(study_output, f, indent=2, default=str)

    print(f"\nResults saved to {output_path}")
    print(f"Total time: {total_time:.1f}s")

    # Quick summary
    primary = [r for r in all_results if not r.is_boundary and not r.error]
    if primary:
        print(f"\nPrimary pairs processed: {len(primary)}")
        for r in primary:
            cB = r.conditions.get("B_fullrank_normeq", {})
            cD = r.conditions.get("D_compress_normeq", {})
            if "error" not in cB and "error" not in cD:
                Bq = cB.get("spectral_outcomes", {}).get("Q_min_cosine", 0)
                Dq = cD.get("spectral_outcomes", {}).get("Q_min_cosine", 0)
                delta = Dq - Bq
                emoji = "+" if delta > 0.001 else ("−" if delta < -0.001 else "=")
                print(f"  {r.label_a} × {r.label_b}: NormEq Q_min {Bq:.4f} → Compress+NormEq {Dq:.4f}  ({emoji}{abs(delta):.4f})")


if __name__ == "__main__":
    main()
