#!/usr/bin/env python3
"""
Study 16: End-to-End Merge Ablation.

Compares naive (uniform_linear) merges against Gradience's audit-aware
merge recommendations across 6 adapter pairs spanning the compatibility
spectrum (SAFE, REDUNDANT, CONFLICTING, IMBALANCED).

Phase 1 (CPU): Spectral evaluation metrics (Q_min, dominance, retention,
reconstruction error, merged adapter audit).

Phase 2 (GPU, deferred): Downstream task evaluation via lm-eval.

Usage:
    python scripts/study16_merge_ablation.py \
        --output-dir results/study16_merge_ablation \
        --cache-dir ~/.cache/gradience/adapters \
        --verbose
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

logger = logging.getLogger("study16")


# ---------------------------------------------------------------------------
# Pair definitions
# ---------------------------------------------------------------------------

@dataclass
class AdapterPairDef:
    """Definition of an adapter pair for the ablation."""
    pair_id: str
    repo_a: str
    repo_b: str
    label_a: str
    label_b: str
    task_a: str
    task_b: str
    expected_verdict: str  # Hypothesis about dominant verdict


PAIRS: List[AdapterPairDef] = [
    AdapterPairDef(
        pair_id="pair_01_metamath_x_openwebmath16",
        repo_a="LoRA-TMLR-2024/metamath-lora-rank-16-alpha-32",
        repo_b="LoRA-TMLR-2024/openwebmath-lora-rank-16-20B-tokens",
        label_a="metamath-r16",
        label_b="openwebmath-r16",
        task_a="math",
        task_b="math",
        expected_verdict="redundant",
    ),
    AdapterPairDef(
        pair_id="pair_02_metamath_x_magicoder",
        repo_a="LoRA-TMLR-2024/metamath-lora-rank-16-alpha-32",
        repo_b="LoRA-TMLR-2024/magicoder-lora-rank-16-alpha-32",
        label_a="metamath-r16",
        label_b="magicoder-r16",
        task_a="math",
        task_b="code",
        expected_verdict="moderate",
    ),
    AdapterPairDef(
        pair_id="pair_03_magicoder_x_btgenbot",
        repo_a="LoRA-TMLR-2024/magicoder-lora-rank-16-alpha-32",
        repo_b="AIRLab-POLIMI/llama-2-7b-chat-hf-btgenbot-adapter",
        label_a="magicoder-r16",
        label_b="btgenbot-r8",
        task_a="code",
        task_b="chat",
        expected_verdict="conflicting",
    ),
    AdapterPairDef(
        pair_id="pair_04_openwebmath64_x_btgenbot",
        repo_a="LoRA-TMLR-2024/openwebmath-lora-rank-64-20B-tokens",
        repo_b="AIRLab-POLIMI/llama-2-7b-chat-hf-btgenbot-adapter",
        label_a="openwebmath-r64",
        label_b="btgenbot-r8",
        task_a="math",
        task_b="chat",
        expected_verdict="imbalanced",
    ),
    AdapterPairDef(
        pair_id="pair_05_metamath_x_llm2vec",
        repo_a="LoRA-TMLR-2024/metamath-lora-rank-16-alpha-32",
        repo_b="McGill-NLP/LLM2Vec-Llama-2-7b-chat-hf-mntp-supervised",
        label_a="metamath-r16",
        label_b="llm2vec-r16",
        task_a="math",
        task_b="classification",
        expected_verdict="conflicting",
    ),
    AdapterPairDef(
        pair_id="pair_06_catsubcat_x_btgenbot",
        repo_a="shivanikerai/Llama-2-7b-chat-hf-adapter-cat-subcat-mapping-v2.0",
        repo_b="AIRLab-POLIMI/llama-2-7b-chat-hf-btgenbot-adapter",
        label_a="catsubcat-r16",
        label_b="btgenbot-r8",
        task_a="chat",
        task_b="chat",
        expected_verdict="mixed",
    ),
]


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def download_adapter(repo_id: str, cache_dir: Path) -> Path:
    """Download a PEFT adapter from HuggingFace Hub. Returns local path."""
    from huggingface_hub import snapshot_download

    local_dir = cache_dir / repo_id.replace("/", "--")

    # Check if already downloaded
    config_candidates = list(local_dir.glob("adapter_config*"))
    weight_candidates = list(local_dir.glob("adapter_model*"))
    if config_candidates and weight_candidates:
        logger.info("  Already cached: %s", local_dir)
        return local_dir

    logger.info("  Downloading %s ...", repo_id)
    snapshot_download(
        repo_id,
        local_dir=str(local_dir),
        allow_patterns=[
            "adapter_config*",
            "adapter_model*",
            "pytorch_model.bin",
        ],
        ignore_patterns=[
            "*.md", "*.txt", ".gitattributes",
            "tokenizer*", "special_tokens*", "training_args*",
        ],
    )
    return local_dir


# ---------------------------------------------------------------------------
# Spectral evaluation of a merged adapter
# ---------------------------------------------------------------------------

def audit_merged_adapter(merged_dir: Path) -> Dict[str, Any]:
    """Run spectral audit on a merged adapter and return summary metrics."""
    from gradience.vnext.audit.lora_audit import audit_lora_peft_dir

    result = audit_lora_peft_dir(
        merged_dir,
        compute_dtype=torch.float64,
        compute_udr=False,  # No base model available
    )

    # Extract summary metrics
    layers = result.layers if hasattr(result, "layers") else []
    if not layers:
        return {
            "n_layers": 0,
            "mean_stable_rank": 0.0,
            "mean_utilisation": 0.0,
            "mean_energy_rank_90": 0.0,
            "nominal_rank": getattr(result, "nominal_rank", 0),
        }

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
        "mean_stable_rank": sum(stable_ranks) / len(stable_ranks) if stable_ranks else 0.0,
        "mean_utilisation": sum(utilisations) / len(utilisations) if utilisations else 0.0,
        "mean_energy_rank_90": sum(energy_ranks) / len(energy_ranks) if energy_ranks else 0.0,
        "median_stable_rank": sorted(stable_ranks)[len(stable_ranks) // 2] if stable_ranks else 0.0,
        "median_utilisation": sorted(utilisations)[len(utilisations) // 2] if utilisations else 0.0,
    }


# ---------------------------------------------------------------------------
# Spectral-only merge outcome metrics
# ---------------------------------------------------------------------------

def compute_spectral_outcomes(
    adapter_a_dir: Path,
    adapter_b_dir: Path,
    merged_dir: Path,
) -> Dict[str, Any]:
    """Compute spectral retention and dominance metrics.

    Uses Frobenius norms as proxy scores: the 'score' of each adapter
    is its total Frobenius norm (sum across layers), and the 'score' of
    the merged adapter relative to each source is how much of each
    source's spectral energy is preserved.
    """
    from gradience.vnext.merge.io import load_adapter, match_layers, extract_factors

    info_a = load_adapter(adapter_a_dir)
    info_b = load_adapter(adapter_b_dir)
    info_m = load_adapter(merged_dir)

    # Compute per-source Frobenius norms
    frob_a_total = 0.0
    frob_b_total = 0.0
    frob_m_total = 0.0

    # For retention: project merged onto each source's subspace
    # Simpler approach: compare Frobenius norms as a scale proxy
    for prefix in info_a.module_prefixes:
        try:
            A_a, B_a, r_a = extract_factors(info_a, prefix)
            dW_a = (info_a.alpha / r_a) * B_a @ A_a
            frob_a_total += torch.norm(dW_a, p="fro").item() ** 2
        except (KeyError, RuntimeError):
            continue

    for prefix in info_b.module_prefixes:
        try:
            A_b, B_b, r_b = extract_factors(info_b, prefix)
            dW_b = (info_b.alpha / r_b) * B_b @ A_b
            frob_b_total += torch.norm(dW_b, p="fro").item() ** 2
        except (KeyError, RuntimeError):
            continue

    for prefix in info_m.module_prefixes:
        try:
            A_m, B_m, r_m = extract_factors(info_m, prefix)
            dW_m = (info_m.alpha / r_m) * B_m @ A_m
            frob_m_total += torch.norm(dW_m, p="fro").item() ** 2
        except (KeyError, RuntimeError):
            continue

    frob_a = frob_a_total ** 0.5
    frob_b = frob_b_total ** 0.5
    frob_m = frob_m_total ** 0.5

    # Compute per-layer cosine similarity between merged and each source
    # to measure directional retention
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
            cos = torch.nn.functional.cosine_similarity(
                dW_a.flatten().unsqueeze(0),
                dW_m.flatten().unsqueeze(0),
            ).item()
            cos_sims_a.append(cos)
        except (KeyError, RuntimeError):
            continue

    for prefix in shared_b:
        try:
            A_b, B_b, r_b = extract_factors(info_b, prefix)
            A_m, B_m, r_m = extract_factors(info_m, prefix)
            dW_b = (info_b.alpha / r_b) * B_b @ A_b
            dW_m = (info_m.alpha / r_m) * B_m @ A_m
            cos = torch.nn.functional.cosine_similarity(
                dW_b.flatten().unsqueeze(0),
                dW_m.flatten().unsqueeze(0),
            ).item()
            cos_sims_b.append(cos)
        except (KeyError, RuntimeError):
            continue

    mean_cos_a = sum(cos_sims_a) / len(cos_sims_a) if cos_sims_a else 0.0
    mean_cos_b = sum(cos_sims_b) / len(cos_sims_b) if cos_sims_b else 0.0

    # Retention: fraction of source Frobenius energy preserved in merge
    retention_a = frob_m / max(frob_a, 1e-12)
    retention_b = frob_m / max(frob_b, 1e-12)

    # Dominance: how asymmetric is the directional retention
    eps = 1e-12
    D = abs(mean_cos_a - mean_cos_b) / (abs(mean_cos_a) + abs(mean_cos_b) + eps)

    Q_min = min(mean_cos_a, mean_cos_b)

    return {
        "frob_a": round(frob_a, 6),
        "frob_b": round(frob_b, 6),
        "frob_merged": round(frob_m, 6),
        "frob_retention_a": round(retention_a, 6),
        "frob_retention_b": round(retention_b, 6),
        "mean_cosine_sim_a": round(mean_cos_a, 6),
        "mean_cosine_sim_b": round(mean_cos_b, 6),
        "Q_min_cosine": round(Q_min, 6),
        "D_cosine": round(D, 6),
        "is_dominated": D > 0.2,
        "n_layers_compared_a": len(cos_sims_a),
        "n_layers_compared_b": len(cos_sims_b),
    }


# ---------------------------------------------------------------------------
# Single pair pipeline
# ---------------------------------------------------------------------------

@dataclass
class PairResult:
    """Result for one adapter pair across all merge conditions."""
    pair_id: str
    label_a: str
    label_b: str
    task_a: str
    task_b: str
    expected_verdict: str
    actual_overall_verdict: str = ""
    actual_compatibility_score: float = 0.0
    audit_verdict_counts: Dict[str, int] = field(default_factory=dict)
    recommendation_summary: Dict[str, Any] = field(default_factory=dict)
    conditions: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    error: Optional[str] = None


def run_pair(
    pair_def: AdapterPairDef,
    cache_dir: Path,
    output_dir: Path,
    verbose: bool = False,
) -> PairResult:
    """Run the full ablation pipeline for one adapter pair."""
    from gradience.vnext.merge import (
        merge_audit,
        plan_from_audit,
        execute_merge,
        recommend_merge,
    )

    result = PairResult(
        pair_id=pair_def.pair_id,
        label_a=pair_def.label_a,
        label_b=pair_def.label_b,
        task_a=pair_def.task_a,
        task_b=pair_def.task_b,
        expected_verdict=pair_def.expected_verdict,
    )

    pair_dir = output_dir / pair_def.pair_id
    pair_dir.mkdir(parents=True, exist_ok=True)

    # --- Step 1: Download adapters ---
    if verbose:
        print(f"\n{'='*60}")
        print(f"  Pair: {pair_def.label_a}  x  {pair_def.label_b}")
        print(f"  Expected: {pair_def.expected_verdict}")
        print(f"{'='*60}")

    try:
        dir_a = download_adapter(pair_def.repo_a, cache_dir)
        dir_b = download_adapter(pair_def.repo_b, cache_dir)
    except Exception as e:
        result.error = f"Download failed: {e}"
        logger.error("Download failed for %s: %s", pair_def.pair_id, e)
        return result

    # --- Step 2: Merge audit ---
    if verbose:
        print("\n  Running merge audit...")

    try:
        audit_dir = pair_dir / "audit"
        report = merge_audit(
            str(dir_a), str(dir_b),
            output_dir=str(audit_dir),
            verbose=verbose,
        )
    except Exception as e:
        result.error = f"Merge audit failed: {e}"
        logger.error("Audit failed for %s: %s", pair_def.pair_id, e)
        return result

    result.actual_overall_verdict = report.aggregate.get("overall_verdict", "unknown")
    result.actual_compatibility_score = report.aggregate.get("compatibility_score", 0.0)

    # Count per-layer verdicts
    verdict_counts: Dict[str, int] = {}
    for lv in report.layer_verdicts:
        v = lv.get("verdict", "unknown")
        verdict_counts[v] = verdict_counts.get(v, 0) + 1
    result.audit_verdict_counts = verdict_counts

    # Get recommendation
    rec = recommend_merge(report)
    result.recommendation_summary = {
        "overall_strategy": rec.overall_strategy,
        "overall_risk": rec.overall_risk,
        "compression_needed": rec.compression_needed,
        "n_layers_needing_compression": rec.n_layers_needing_compression,
    }

    # Determine output rank and alpha for plans
    # Use the max of the two adapters' ranks for output to avoid information loss
    rank_a = report.adapter_a.get("rank", 16)
    rank_b = report.adapter_b.get("rank", 16)
    output_rank = max(rank_a, rank_b)
    alpha_a = report.adapter_a.get("alpha", 32.0)
    output_alpha = alpha_a  # Use adapter A's alpha as default

    if verbose:
        print(f"  Overall verdict: {result.actual_overall_verdict}")
        print(f"  Compatibility score: {result.actual_compatibility_score:.3f}")
        print(f"  Verdict counts: {verdict_counts}")
        print(f"  Output rank: {output_rank}, alpha: {output_alpha}")

    # --- Step 3: Plan and execute merges ---

    conditions_to_run = [
        ("naive", "uniform_linear"),
        ("recommended", "audit_aware"),
    ]

    for cond_name, strategy_name in conditions_to_run:
        if verbose:
            print(f"\n  --- Condition: {cond_name} ({strategy_name}) ---")

        cond_dir = pair_dir / cond_name
        cond_dir.mkdir(parents=True, exist_ok=True)

        try:
            t0 = time.time()

            plan = plan_from_audit(
                strategy_name, report,
                str(dir_a), str(dir_b),
                output_rank=output_rank,
                output_alpha=output_alpha,
            )

            merge_result = execute_merge(
                plan, str(cond_dir),
                verbose=verbose,
            )

            merge_time = time.time() - t0

            if verbose:
                print(f"  Merge completed in {merge_time:.1f}s")
                print(f"  Mean recon error: {merge_result.mean_reconstruction_error:.6f}")
                print(f"  Max recon error: {merge_result.max_reconstruction_error:.6f}")

            # Save merge result
            merge_result.to_json(cond_dir / "merge_result.json")

            # --- Step 4: Evaluate merged adapter ---
            if verbose:
                print(f"  Auditing merged adapter...")

            merged_audit = audit_merged_adapter(cond_dir)

            if verbose:
                print(f"  Computing spectral outcomes...")

            spectral_outcomes = compute_spectral_outcomes(dir_a, dir_b, cond_dir)

            # Collect per-layer reconstruction errors
            recon_errors = [lr.reconstruction_error for lr in merge_result.layer_results]

            result.conditions[cond_name] = {
                "strategy": strategy_name,
                "merge_time_s": round(merge_time, 2),
                "mean_reconstruction_error": round(merge_result.mean_reconstruction_error, 6),
                "max_reconstruction_error": round(merge_result.max_reconstruction_error, 6),
                "merged_audit": merged_audit,
                "spectral_outcomes": spectral_outcomes,
                "recon_error_p50": round(sorted(recon_errors)[len(recon_errors) // 2], 6) if recon_errors else 0.0,
                "recon_error_p90": round(sorted(recon_errors)[int(len(recon_errors) * 0.9)], 6) if recon_errors else 0.0,
            }

            if verbose:
                so = spectral_outcomes
                print(f"  Q_min (cosine): {so['Q_min_cosine']:.4f}")
                print(f"  D (cosine):     {so['D_cosine']:.4f}")
                print(f"  cos_sim A:      {so['mean_cosine_sim_a']:.4f}")
                print(f"  cos_sim B:      {so['mean_cosine_sim_b']:.4f}")
                print(f"  Utilisation:    {merged_audit['mean_utilisation']:.4f}")

            # Clean up large safetensors to free disk space (metrics are saved in JSON)
            for sf in cond_dir.glob("*.safetensors"):
                sf.unlink()
                if verbose:
                    print(f"  Cleaned up {sf.name} to free disk space")

        except Exception as e:
            result.conditions[cond_name] = {"error": str(e)}
            logger.error("Condition %s failed for %s: %s", cond_name, pair_def.pair_id, e, exc_info=True)

    return result


# ---------------------------------------------------------------------------
# Summary and comparison
# ---------------------------------------------------------------------------

def compute_comparison_table(results: List[PairResult]) -> List[Dict[str, Any]]:
    """Build a comparison table: naive vs. recommended for each pair."""
    rows = []

    for r in results:
        if r.error:
            rows.append({
                "pair_id": r.pair_id,
                "label": f"{r.label_a} x {r.label_b}",
                "error": r.error,
            })
            continue

        naive = r.conditions.get("naive", {})
        rec = r.conditions.get("recommended", {})

        if "error" in naive or "error" in rec:
            rows.append({
                "pair_id": r.pair_id,
                "label": f"{r.label_a} x {r.label_b}",
                "error": naive.get("error") or rec.get("error"),
            })
            continue

        naive_so = naive.get("spectral_outcomes", {})
        rec_so = rec.get("spectral_outcomes", {})
        naive_audit = naive.get("merged_audit", {})
        rec_audit = rec.get("merged_audit", {})

        row = {
            "pair_id": r.pair_id,
            "label": f"{r.label_a} x {r.label_b}",
            "expected_verdict": r.expected_verdict,
            "actual_verdict": r.actual_overall_verdict,
            "compatibility_score": r.actual_compatibility_score,
            # Naive metrics
            "naive_Q_min": naive_so.get("Q_min_cosine", 0.0),
            "naive_D": naive_so.get("D_cosine", 0.0),
            "naive_cos_a": naive_so.get("mean_cosine_sim_a", 0.0),
            "naive_cos_b": naive_so.get("mean_cosine_sim_b", 0.0),
            "naive_utilisation": naive_audit.get("mean_utilisation", 0.0),
            "naive_recon_error": naive.get("mean_reconstruction_error", 0.0),
            # Recommended metrics
            "rec_Q_min": rec_so.get("Q_min_cosine", 0.0),
            "rec_D": rec_so.get("D_cosine", 0.0),
            "rec_cos_a": rec_so.get("mean_cosine_sim_a", 0.0),
            "rec_cos_b": rec_so.get("mean_cosine_sim_b", 0.0),
            "rec_utilisation": rec_audit.get("mean_utilisation", 0.0),
            "rec_recon_error": rec.get("mean_reconstruction_error", 0.0),
            # Deltas (positive = recommended is better)
            "delta_Q_min": round(
                rec_so.get("Q_min_cosine", 0.0) - naive_so.get("Q_min_cosine", 0.0), 6
            ),
            "delta_D": round(
                naive_so.get("D_cosine", 0.0) - rec_so.get("D_cosine", 0.0), 6
            ),
            "delta_utilisation": round(
                rec_audit.get("mean_utilisation", 0.0) - naive_audit.get("mean_utilisation", 0.0), 6
            ),
        }
        rows.append(row)

    return rows


def print_comparison_table(rows: List[Dict[str, Any]]) -> None:
    """Pretty-print the comparison table."""
    print("\n" + "=" * 100)
    print("  STUDY 16 — MERGE ABLATION COMPARISON TABLE")
    print("=" * 100)
    print(
        f"  {'Pair':<40s}  {'Verdict':<12s}  "
        f"{'Naive Q_min':>11s}  {'Rec Q_min':>9s}  {'dQ_min':>7s}  "
        f"{'Naive D':>7s}  {'Rec D':>6s}  {'dD':>6s}"
    )
    print("  " + "-" * 96)

    for row in rows:
        if "error" in row:
            print(f"  {row['label']:<40s}  ERROR: {row['error']}")
            continue
        print(
            f"  {row['label']:<40s}  {row['actual_verdict']:<12s}  "
            f"  {row['naive_Q_min']:>9.4f}  {row['rec_Q_min']:>9.4f}  {row['delta_Q_min']:>+7.4f}  "
            f"{row['naive_D']:>7.4f}  {row['rec_D']:>6.4f}  {row['delta_D']:>+6.4f}"
        )

    print("=" * 100)

    # Summary stats
    valid = [r for r in rows if "error" not in r]
    if valid:
        mean_dQ = sum(r["delta_Q_min"] for r in valid) / len(valid)
        mean_dD = sum(r["delta_D"] for r in valid) / len(valid)
        wins_Q = sum(1 for r in valid if r["delta_Q_min"] > 0)
        wins_D = sum(1 for r in valid if r["delta_D"] > 0)
        print(f"\n  Mean delta Q_min: {mean_dQ:+.4f}  (recommended better in {wins_Q}/{len(valid)} pairs)")
        print(f"  Mean delta D:     {mean_dD:+.4f}  (recommended better in {wins_D}/{len(valid)} pairs)")
        print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Study 16: Merge Ablation")
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("results/study16_merge_ablation"),
        help="Directory for all output files",
    )
    parser.add_argument(
        "--cache-dir", type=Path,
        default=Path.home() / ".cache" / "gradience" / "adapters",
        help="Cache directory for downloaded adapters",
    )
    parser.add_argument(
        "--pairs", type=str, nargs="*", default=None,
        help="Specific pair IDs to run (default: all)",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    # Filter pairs if specified
    pairs_to_run = PAIRS
    if args.pairs:
        pairs_to_run = [p for p in PAIRS if p.pair_id in args.pairs]
        if not pairs_to_run:
            print(f"No matching pairs found. Available: {[p.pair_id for p in PAIRS]}")
            sys.exit(1)

    print(f"Study 16: Merge Ablation — {len(pairs_to_run)} pairs")
    print(f"Output: {args.output_dir}")
    print(f"Cache:  {args.cache_dir}")

    t_start = time.time()
    all_results: List[PairResult] = []

    for i, pair_def in enumerate(pairs_to_run):
        print(f"\n[{i+1}/{len(pairs_to_run)}] Processing {pair_def.pair_id}...")
        pair_result = run_pair(pair_def, args.cache_dir, args.output_dir, args.verbose)
        all_results.append(pair_result)

    total_time = time.time() - t_start

    # --- Build comparison table ---
    comparison = compute_comparison_table(all_results)
    print_comparison_table(comparison)

    # --- Save full results ---
    study_output = {
        "study_metadata": {
            "study_id": "study16_merge_ablation",
            "title": "End-to-End Merge Ablation: Naive vs. Audit-Aware",
            "date": datetime.now(timezone.utc).isoformat(),
            "n_pairs": len(pairs_to_run),
            "total_time_s": round(total_time, 2),
            "base_model": "meta-llama/Llama-2-7b-hf",
            "phase": "phase_1_spectral",
        },
        "pair_results": [asdict(r) for r in all_results],
        "comparison_table": comparison,
    }

    output_path = args.output_dir / "study16_merge_ablation.json"
    with open(output_path, "w") as f:
        json.dump(study_output, f, indent=2, default=str)

    print(f"\nResults saved to {output_path}")
    print(f"Total time: {total_time:.1f}s")


if __name__ == "__main__":
    main()
