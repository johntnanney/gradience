#!/usr/bin/env python3
"""
Study 17B: Threshold Sensitivity for Compress-Then-Merge — Phase 1 (CPU).

Follow-up to Study 17A (95% energy threshold = no-op).  Tests whether more
aggressive compression (90%, 80% energy retention) improves LoRA merge
outcomes relative to full-rank norm-equalized merging.

Focused pair set:
    pair_03  magicoder-r16 × btgenbot-r8    (imbalanced, rank mismatch)
    pair_04  openwebmath-r64 × btgenbot-r8  (imbalanced, extreme rank mismatch)
    pair_02  metamath-r16 × magicoder-r16   (optional do-no-harm control)

Primary conditions:
    full_normeq       — full-rank norm-equalized baseline
    comp90_normeq     — compress @90% + norm-equalized
    comp80_normeq     — compress @80% + norm-equalized

Secondary conditions (--secondary):
    full_recommended   — full-rank audit_aware baseline
    comp90_recommended — compress @90% + audit_aware
    comp80_recommended — compress @80% + audit_aware

Usage:
    # Primary pairs only (pair_03, pair_04):
    python scripts/study17b_threshold_sensitivity.py \\
        --output-dir results/study17b \\
        --verbose

    # Include control pair:
    python scripts/study17b_threshold_sensitivity.py \\
        --output-dir results/study17b \\
        --pairs pair_02 pair_03 pair_04 \\
        --verbose

    # Add recommended strategy family:
    python scripts/study17b_threshold_sensitivity.py \\
        --output-dir results/study17b \\
        --secondary \\
        --verbose

    # Smoke test — single pair, single threshold:
    python scripts/study17b_threshold_sensitivity.py \\
        --output-dir results/study17b_smoke \\
        --pairs pair_03 \\
        --thresholds 0.90 \\
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

logger = logging.getLogger("study17b")


# ═══════════════════════════════════════════════════════════════════════════
# Pair definitions (subset of study17 pairs)
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
    role: str = "primary"  # "primary" or "control"


PAIRS: list[AdapterPairDef] = [
    AdapterPairDef(
        pair_id="pair_02",
        repo_a="LoRA-TMLR-2024/metamath-lora-rank-16-alpha-32",
        repo_b="LoRA-TMLR-2024/magicoder-lora-rank-16-alpha-32",
        label_a="metamath-r16",
        label_b="magicoder-r16",
        task_a="math",
        task_b="code",
        expected_verdict="moderate",
        role="control",
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
        role="primary",
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
        role="primary",
    ),
]

DEFAULT_PRIMARY_PAIRS = ["pair_03", "pair_04"]
THRESHOLDS = [0.90, 0.80]


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
    threshold: float
    nominal_rank: int
    target_rank: int
    energy_retained: float
    per_layer_ranks: list[dict[str, Any]]
    mean_effective_rank: float
    mean_reduction: float
    pct_rank_reduction: float  # (nominal - target) / nominal * 100
    n_layers_heavily_truncated: int
    reconstruction_error: float  # layerwise mean Frobenius reconstruction error
    source_dir: str
    compressed_dir: str


def compute_effective_rank_at_threshold(peft_dir: Path, threshold: float) -> tuple[int, list[dict]]:
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
        above = (cumulative >= threshold).nonzero(as_tuple=True)[0]
        if len(above) > 0:
            k = int(above[0].item()) + 1
        else:
            k = int(S.shape[0])

        per_layer.append(
            {
                "layer": base_key,
                "nominal": int(S.shape[0]),
                "effective": k,
                "energy": float(cumulative[k - 1].item()),
            }
        )
        max_effective = max(max_effective, k)

    return max_effective, per_layer


def compute_compression_distortion(original_dir: Path, compressed_dir: Path) -> float:
    """Compute mean per-layer Frobenius reconstruction error between original and compressed.

    Returns mean ||dW_orig - dW_comp||_F / ||dW_orig||_F across layers.
    """
    from gradience.vnext.merge.io import extract_factors, load_adapter

    info_orig = load_adapter(original_dir)
    info_comp = load_adapter(compressed_dir)

    rel_errors = []
    for prefix in info_orig.module_prefixes:
        try:
            A_o, B_o, r_o = extract_factors(info_orig, prefix)
            A_c, B_c, r_c = extract_factors(info_comp, prefix)
            dW_o = (info_orig.alpha / r_o) * B_o @ A_o
            dW_c = (info_comp.alpha / r_c) * B_c @ A_c
            orig_norm = torch.norm(dW_o, p="fro").item()
            if orig_norm < 1e-12:
                continue
            err = torch.norm(dW_o - dW_c, p="fro").item() / orig_norm
            rel_errors.append(err)
        except (KeyError, RuntimeError):
            continue

    return float(np.mean(rel_errors)) if rel_errors else 0.0


def compress_adapter(
    source_dir: Path,
    out_dir: Path,
    adapter_label: str,
    threshold: float,
    verbose: bool = False,
) -> CompressionReport:
    """Compress an adapter using energy-threshold SVD truncation."""
    from gradience.vnext.audit.lora_audit import find_peft_files, load_peft_adapter_config
    from gradience.vnext.svd_truncate import svd_truncate_peft_dir

    config_path, _, _ = find_peft_files(source_dir)
    config = load_peft_adapter_config(config_path)
    nominal_rank = config.r

    target_rank, per_layer_info = compute_effective_rank_at_threshold(source_dir, threshold)

    if verbose:
        print(f"    {adapter_label}: nominal={nominal_rank}, effective@{threshold:.0%}={target_rank}")

    if target_rank >= nominal_rank:
        import shutil

        if out_dir.exists():
            shutil.rmtree(out_dir)
        shutil.copytree(source_dir, out_dir)
        if verbose:
            print(f"    No compression needed (effective rank {target_rank} >= nominal {nominal_rank})")

        mean_eff = float(np.mean([l["effective"] for l in per_layer_info]))
        return CompressionReport(
            adapter_label=adapter_label,
            threshold=threshold,
            nominal_rank=nominal_rank,
            target_rank=nominal_rank,
            energy_retained=1.0,
            per_layer_ranks=per_layer_info,
            mean_effective_rank=round(mean_eff, 2),
            mean_reduction=0.0,
            pct_rank_reduction=0.0,
            n_layers_heavily_truncated=0,
            reconstruction_error=0.0,
            source_dir=str(source_dir),
            compressed_dir=str(out_dir),
        )

    report = svd_truncate_peft_dir(source_dir, out_dir, target_rank)

    # Compute distortion
    distortion = compute_compression_distortion(source_dir, out_dir)

    mean_eff = float(np.mean([l["effective"] for l in per_layer_info]))
    mean_reduction = nominal_rank - mean_eff
    n_heavy = sum(1 for l in per_layer_info if l["effective"] < nominal_rank / 2)
    pct_reduction = (nominal_rank - target_rank) / nominal_rank * 100

    if verbose:
        print(f"    Compression: {nominal_rank} → {target_rank} ({pct_reduction:.1f}% reduction)")
        print(f"    Distortion (mean relative Frobenius error): {distortion:.6f}")

    return CompressionReport(
        adapter_label=adapter_label,
        threshold=threshold,
        nominal_rank=nominal_rank,
        target_rank=target_rank,
        energy_retained=report.energy_retained,
        per_layer_ranks=per_layer_info,
        mean_effective_rank=round(mean_eff, 2),
        mean_reduction=round(mean_reduction, 2),
        pct_rank_reduction=round(pct_reduction, 2),
        n_layers_heavily_truncated=n_heavy,
        reconstruction_error=round(distortion, 6),
        source_dir=str(source_dir),
        compressed_dir=str(out_dir),
    )


# ═══════════════════════════════════════════════════════════════════════════
# Spectral evaluation
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
    """Compute global Frobenius-proportional merge coefficients (inverse-proportional)."""
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
    threshold: float | None  # None = full-rank
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
    role: str
    actual_overall_verdict: str = ""
    actual_compatibility_score: float = 0.0
    audit_verdict_counts: dict[str, int] = field(default_factory=dict)
    recommendation_summary: dict[str, Any] = field(default_factory=dict)
    # Compression reports keyed by threshold string ("0.90", "0.80")
    compressions: dict[str, dict[str, Any]] = field(default_factory=dict)
    conditions: dict[str, dict[str, Any]] = field(default_factory=dict)
    error: str | None = None


def run_pair(
    pair_def: AdapterPairDef,
    thresholds: list[float],
    cache_dir: Path,
    output_dir: Path,
    run_secondary: bool = False,
    verbose: bool = False,
) -> PairResult:
    """Run all conditions for one adapter pair across all thresholds."""
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
        role=pair_def.role,
    )

    pair_dir = output_dir / pair_def.pair_id
    pair_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"\n{'=' * 70}")
        print(f"  Pair: {pair_def.label_a}  ×  {pair_def.label_b}")
        print(f"  Expected: {pair_def.expected_verdict}  Role: {pair_def.role}")
        print(f"  Thresholds: {[f'{t:.0%}' for t in thresholds]}")
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

    # --- Compute full-rank norm-equalized coefficients ---
    try:
        coeff_a_full, coeff_b_full = compute_global_frob_coefficients(dir_a, dir_b)
        if verbose:
            print(f"  Norm-eq coeff (full): A={coeff_a_full:.4f}  B={coeff_b_full:.4f}")
    except Exception as e:
        logger.warning("Norm-eq coeff failed: %s", e)
        coeff_a_full, coeff_b_full = 0.5, 0.5

    # --- Compress at each threshold ---
    compressed_dirs: dict[float, tuple[Path, Path]] = {}
    compressed_coeffs: dict[float, tuple[float, float]] = {}
    compressed_output_ranks: dict[float, int] = {}

    for threshold in thresholds:
        thr_label = f"{threshold:.2f}"
        if verbose:
            print(f"\n  Compressing adapters @ {threshold:.0%} energy...")

        comp_dir = pair_dir / f"compressed_{thr_label}"
        comp_dir.mkdir(parents=True, exist_ok=True)

        try:
            comp_a = compress_adapter(dir_a, comp_dir / "a", pair_def.label_a, threshold, verbose)
            comp_b = compress_adapter(dir_b, comp_dir / "b", pair_def.label_b, threshold, verbose)

            result.compressions[thr_label] = {
                "a": asdict(comp_a),
                "b": asdict(comp_b),
            }

            compressed_dirs[threshold] = (comp_dir / "a", comp_dir / "b")
            compressed_output_ranks[threshold] = max(comp_a.target_rank, comp_b.target_rank)

            # Compute norm-eq coefficients for compressed adapters
            ca, cb = compute_global_frob_coefficients(comp_dir / "a", comp_dir / "b")
            compressed_coeffs[threshold] = (ca, cb)
            if verbose:
                print(f"  Norm-eq coeff (comp@{threshold:.0%}): A={ca:.4f}  B={cb:.4f}")

        except Exception as e:
            result.compressions[thr_label] = {"error": str(e)}
            logger.error("Compression @%s failed for %s: %s", thr_label, pair_def.pair_id, e, exc_info=True)

    # --- Audit compressed pairs ---
    compressed_reports: dict[float, Any] = {}
    for threshold in thresholds:
        if threshold not in compressed_dirs:
            continue
        thr_label = f"{threshold:.2f}"
        try:
            audit_comp_dir = pair_dir / f"audit_compressed_{thr_label}"
            ca_dir, cb_dir = compressed_dirs[threshold]
            compressed_reports[threshold] = merge_audit(
                str(ca_dir), str(cb_dir), output_dir=str(audit_comp_dir), verbose=verbose
            )
        except Exception as e:
            logger.warning("Compressed audit @%s failed: %s", thr_label, e)

    # --- Build condition list ---
    # (name, strategy, dir_a, dir_b, audit_report, plan_kwargs, is_compressed, threshold)
    conditions: list[tuple[str, str, Path, Path, Any, dict, bool, float | None]] = []

    # Full-rank NormEq baseline
    conditions.append(
        (
            "full_normeq",
            "uniform_linear",
            dir_a,
            dir_b,
            report,
            {"output_rank": output_rank, "output_alpha": output_alpha, "coefficients": (coeff_a_full, coeff_b_full)},
            False,
            None,
        )
    )

    # Compressed NormEq at each threshold
    for threshold in thresholds:
        if threshold not in compressed_dirs:
            continue
        thr_pct = int(threshold * 100)
        ca_dir, cb_dir = compressed_dirs[threshold]
        comp_report = compressed_reports.get(threshold, report)
        ca, cb = compressed_coeffs.get(threshold, (0.5, 0.5))
        comp_rank = compressed_output_ranks.get(threshold, output_rank)

        conditions.append(
            (
                f"comp{thr_pct}_normeq",
                "uniform_linear",
                ca_dir,
                cb_dir,
                comp_report,
                {"output_rank": comp_rank, "output_alpha": output_alpha, "coefficients": (ca, cb)},
                True,
                threshold,
            )
        )

    # Secondary: recommended family
    if run_secondary:
        conditions.append(
            (
                "full_recommended",
                "audit_aware",
                dir_a,
                dir_b,
                report,
                {"output_rank": output_rank, "output_alpha": output_alpha},
                False,
                None,
            )
        )

        for threshold in thresholds:
            if threshold not in compressed_dirs:
                continue
            thr_pct = int(threshold * 100)
            ca_dir, cb_dir = compressed_dirs[threshold]
            comp_report = compressed_reports.get(threshold, report)
            comp_rank = compressed_output_ranks.get(threshold, output_rank)

            conditions.append(
                (
                    f"comp{thr_pct}_recommended",
                    "audit_aware",
                    ca_dir,
                    cb_dir,
                    comp_report,
                    {"output_rank": comp_rank, "output_alpha": output_alpha},
                    True,
                    threshold,
                )
            )

    # --- Run conditions ---
    for cond_name, strategy, src_a, src_b, audit_report, plan_kwargs, is_compressed, threshold in conditions:
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

            merge_result.to_json(cond_dir / "merge_result.json")

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
                threshold=threshold,
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
            for sf in cond_dir.glob("*.safetensors"):
                sf.unlink()
                if verbose:
                    print(f"    Cleaned up {sf.name}")

        except Exception as e:
            result.conditions[cond_name] = {"error": str(e), "condition_name": cond_name}
            logger.error("Condition %s failed for %s: %s", cond_name, pair_def.pair_id, e, exc_info=True)

    # Clean up compressed adapter weights
    for threshold in thresholds:
        thr_label = f"{threshold:.2f}"
        comp_dir = pair_dir / f"compressed_{thr_label}"
        for sf in comp_dir.rglob("*.safetensors"):
            sf.unlink()
        for bf in comp_dir.rglob("*.bin"):
            bf.unlink()

    return result


# ═══════════════════════════════════════════════════════════════════════════
# Result extraction and comparison tables
# ═══════════════════════════════════════════════════════════════════════════


def _get(conditions: dict, cond_name: str, metric_path: str, default: float = 0.0) -> float:
    """Safely extract a nested metric from a condition dict."""
    cond = conditions.get(cond_name, {})
    if cond.get("error"):
        return default
    parts = metric_path.split(".")
    val: Any = cond
    for p in parts:
        if isinstance(val, dict):
            val = val.get(p, default)
        else:
            return default
    return float(val) if val is not None else default


def print_threshold_comparison(results: list[PairResult], thresholds: list[float]) -> None:
    """Print Table 1: Threshold comparison (NormEq family)."""
    print("\n" + "=" * 120)
    print("  STUDY 17B — THRESHOLD SENSITIVITY COMPARISON")
    print("=" * 120)

    # --- Table 1: Threshold comparison (Q_min) ---
    thr_labels = ["full"] + [f"comp{int(t * 100)}" for t in thresholds]
    print("\n  Table 1: NormEq Q_min Across Thresholds")
    hdr = f"  {'Pair':<38s}  {'Role':<8s}  {'Verdict':<12s}"
    for tl in thr_labels:
        hdr += f"  {tl + ' Q':>9s}"
    hdr += f"  {'Winner':>8s}"
    print(hdr)
    print("  " + "-" * 115)

    for r in results:
        if r.error:
            print(f"  {r.label_a} x {r.label_b:<38s}  ERROR: {r.error}")
            continue

        label = f"{r.label_a} x {r.label_b}"
        conds = r.conditions

        vals = {}
        vals["full"] = _get(conds, "full_normeq", "spectral_outcomes.Q_min_cosine")
        for t in thresholds:
            thr_pct = int(t * 100)
            vals[f"comp{thr_pct}"] = _get(conds, f"comp{thr_pct}_normeq", "spectral_outcomes.Q_min_cosine")

        best_label = max(vals, key=lambda k: vals[k])

        line = f"  {label:<38s}  {r.role:<8s}  {r.actual_overall_verdict:<12s}"
        for tl in thr_labels:
            line += f"  {vals[tl]:>9.4f}"
        line += f"  {best_label:>8s}"
        print(line)

    # --- Table 2: Structural tradeoff ---
    print("\n  Table 2: Structural Tradeoff (deltas vs full NormEq)")
    print(f"  {'Pair':<38s}  {'Threshold':>9s}  {'Q_min':>8s}  {'D':>8s}  {'dQ_min':>8s}  {'dD':>8s}  {'Distort':>8s}")
    print("  " + "-" * 100)

    for r in results:
        if r.error:
            continue

        label = f"{r.label_a} x {r.label_b}"
        conds = r.conditions

        full_q = _get(conds, "full_normeq", "spectral_outcomes.Q_min_cosine")
        full_d = _get(conds, "full_normeq", "spectral_outcomes.D_cosine")

        # Full baseline row
        print(f"  {label:<38s}  {'full':>9s}  {full_q:>8.4f}  {full_d:>8.4f}  {'---':>8s}  {'---':>8s}  {'---':>8s}")

        for t in thresholds:
            thr_pct = int(t * 100)
            cond_name = f"comp{thr_pct}_normeq"
            q = _get(conds, cond_name, "spectral_outcomes.Q_min_cosine")
            d = _get(conds, cond_name, "spectral_outcomes.D_cosine")
            dq = q - full_q
            dd = full_d - d  # positive = less dominance = better

            # Get distortion from compression reports
            thr_label = f"{t:.2f}"
            comp_data = r.compressions.get(thr_label, {})
            dist_a = comp_data.get("a", {}).get("reconstruction_error", 0.0)
            dist_b = comp_data.get("b", {}).get("reconstruction_error", 0.0)
            mean_dist = (dist_a + dist_b) / 2

            print(f"  {'':38s}  {f'{thr_pct}%':>9s}  {q:>8.4f}  {d:>8.4f}  {dq:>+8.4f}  {dd:>+8.4f}  {mean_dist:>8.4f}")

    # --- Table 3: Compression intensity ---
    print("\n  Table 3: Compression Intensity")
    print(
        f"  {'Pair':<38s}  {'Thr':>4s}  {'A nom>eff':>10s}  {'B nom>eff':>10s}  {'A %red':>7s}  {'B %red':>7s}  {'A dist':>7s}  {'B dist':>7s}"
    )
    print("  " + "-" * 100)

    for r in results:
        if r.error:
            continue

        label = f"{r.label_a} x {r.label_b}"

        for t in thresholds:
            thr_label = f"{t:.2f}"
            thr_pct = int(t * 100)
            comp_data = r.compressions.get(thr_label, {})
            if comp_data.get("error"):
                print(f"  {label:<38s}  {thr_pct:>3d}%  ERROR")
                continue

            ca = comp_data.get("a", {})
            cb = comp_data.get("b", {})

            a_nom = ca.get("nominal_rank", 0)
            a_eff = ca.get("target_rank", 0)
            b_nom = cb.get("nominal_rank", 0)
            b_eff = cb.get("target_rank", 0)
            a_pct = ca.get("pct_rank_reduction", 0.0)
            b_pct = cb.get("pct_rank_reduction", 0.0)
            a_dist = ca.get("reconstruction_error", 0.0)
            b_dist = cb.get("reconstruction_error", 0.0)

            print(
                f"  {label:<38s}  {thr_pct:>3d}%  {a_nom:>4d}>{a_eff:<4d}  {b_nom:>4d}>{b_eff:<4d}  "
                f"{a_pct:>6.1f}%  {b_pct:>6.1f}%  {a_dist:>7.4f}  {b_dist:>7.4f}"
            )

    print("=" * 120)

    # --- Summary stats (primary pairs only) ---
    primary = [r for r in results if not r.error and r.role == "primary"]
    if primary:
        n = len(primary)
        print(f"\n  PRIMARY PAIRS ({n}):")
        for t in thresholds:
            thr_pct = int(t * 100)
            cond_name = f"comp{thr_pct}_normeq"
            deltas_q = []
            deltas_d = []
            for r in primary:
                full_q = _get(r.conditions, "full_normeq", "spectral_outcomes.Q_min_cosine")
                comp_q = _get(r.conditions, cond_name, "spectral_outcomes.Q_min_cosine")
                full_d = _get(r.conditions, "full_normeq", "spectral_outcomes.D_cosine")
                comp_d = _get(r.conditions, cond_name, "spectral_outcomes.D_cosine")
                deltas_q.append(comp_q - full_q)
                deltas_d.append(full_d - comp_d)

            mean_dq = sum(deltas_q) / n
            mean_dd = sum(deltas_d) / n
            wins_q = sum(1 for d in deltas_q if d > 0)
            wins_d = sum(1 for d in deltas_d if d > 0)
            print(
                f"  comp@{thr_pct}% NormEq: mean dQ_min={mean_dq:+.4f} ({wins_q}/{n} improved)  "
                f"mean dD={mean_dd:+.4f} ({wins_d}/{n} improved)"
            )

        print()


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════


def main() -> None:
    parser = argparse.ArgumentParser(description="Study 17B: Threshold Sensitivity (Phase 1 — CPU)")
    parser.add_argument("--output-dir", type=Path, default=Path("results/study17b"))
    parser.add_argument("--cache-dir", type=Path, default=Path.home() / ".cache" / "gradience" / "adapters")
    parser.add_argument(
        "--pairs", type=str, nargs="*", default=None, help="Specific pair IDs (default: pair_03, pair_04)"
    )
    parser.add_argument(
        "--thresholds", type=float, nargs="*", default=None, help="Energy thresholds to test (default: 0.90 0.80)"
    )
    parser.add_argument("--secondary", action="store_true", help="Also run recommended strategy variants")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    # Resolve pairs
    if args.pairs:
        pairs_to_run = [p for p in PAIRS if p.pair_id in args.pairs]
        if not pairs_to_run:
            print(f"No matching pairs. Available: {[p.pair_id for p in PAIRS]}")
            sys.exit(1)
    else:
        pairs_to_run = [p for p in PAIRS if p.pair_id in DEFAULT_PRIMARY_PAIRS]

    # Resolve thresholds
    thresholds = args.thresholds if args.thresholds else THRESHOLDS

    print("Study 17B: Threshold Sensitivity — Phase 1 (CPU)")
    print(f"  Pairs: {len(pairs_to_run)} ({', '.join(p.pair_id for p in pairs_to_run)})")
    print(f"  Thresholds: {[f'{t:.0%}' for t in thresholds]}")
    print("  Conditions: full_normeq" + "".join(f", comp{int(t * 100)}_normeq" for t in thresholds))
    if args.secondary:
        print("  Secondary:  full_recommended" + "".join(f", comp{int(t * 100)}_recommended" for t in thresholds))
    print(f"  Output: {args.output_dir}")
    print(f"  Cache:  {args.cache_dir}")

    t_start = time.time()
    all_results: list[PairResult] = []

    for i, pair_def in enumerate(pairs_to_run):
        print(f"\n[{i + 1}/{len(pairs_to_run)}] Processing {pair_def.pair_id}...")
        pair_result = run_pair(pair_def, thresholds, args.cache_dir, args.output_dir, args.secondary, args.verbose)
        all_results.append(pair_result)

        # Save incremental pair result
        pair_path = args.output_dir / pair_def.pair_id / "pair_result.json"
        with open(pair_path, "w") as f:
            json.dump(asdict(pair_result), f, indent=2, default=str)

    total_time = time.time() - t_start

    # --- Comparison tables ---
    print_threshold_comparison(all_results, thresholds)

    # --- Save summary ---
    study_output = {
        "study_metadata": {
            "study_id": "study17b_threshold_sensitivity",
            "title": "Threshold Sensitivity for Compress-Then-Merge",
            "date": datetime.now(timezone.utc).isoformat(),
            "n_pairs": len(pairs_to_run),
            "n_primary_pairs": sum(1 for p in pairs_to_run if p.role == "primary"),
            "n_control_pairs": sum(1 for p in pairs_to_run if p.role == "control"),
            "thresholds": thresholds,
            "conditions_run": _list_conditions(thresholds, args.secondary),
            "total_time_s": round(total_time, 2),
            "base_model": "meta-llama/Llama-2-7b-hf",
            "phase": "phase_1_spectral",
            "secondary_included": args.secondary,
        },
        "pair_results": [asdict(r) for r in all_results],
    }

    output_path = args.output_dir / "study17b_results.json"
    with open(output_path, "w") as f:
        json.dump(study_output, f, indent=2, default=str)

    print(f"\nResults saved to {output_path}")
    print(f"Total time: {total_time:.1f}s")


def _list_conditions(thresholds: list[float], secondary: bool) -> list[str]:
    """Build list of condition names that were run."""
    conds = ["full_normeq"]
    for t in thresholds:
        conds.append(f"comp{int(t * 100)}_normeq")
    if secondary:
        conds.append("full_recommended")
        for t in thresholds:
            conds.append(f"comp{int(t * 100)}_recommended")
    return conds


if __name__ == "__main__":
    main()
