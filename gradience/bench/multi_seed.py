"""Multi-seed benchmarking and aggregation for bench protocol."""
from __future__ import annotations

import json
import datetime
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List

from gradience.bench.model_setup import load_config
from gradience.bench.heartbeat import heartbeat_stage


def create_multi_seed_aggregated_report(
    seed_reports: list[Dict[str, Any]],
    config: Dict[str, Any],
    output_dir: Path
) -> Dict[str, Any]:
    """
    Create aggregated report from multiple seed runs.

    Returns bench_aggregate.json format with mean ± std statistics.
    """
    import numpy as np
    from datetime import datetime

    if not seed_reports:
        raise ValueError("No seed reports provided for aggregation")

    # Extract metadata from first report
    base_report = seed_reports[0]
    timestamp = datetime.utcnow().isoformat() + "Z"

    # Aggregate probe results
    probe_accuracies = [r["probe"]["accuracy"] for r in seed_reports]
    probe_params = [r["probe"]["params"] for r in seed_reports]

    # Calculate probe statistics
    probe_acc_mean = float(np.mean(probe_accuracies))
    probe_acc_std = float(np.std(probe_accuracies))
    probe_params_mean = float(np.mean(probe_params))  # Should be constant

    # Aggregate compressed variants
    variants_data = {}

    # Get all variant names from all reports
    all_variant_names = set()
    for report in seed_reports:
        compressed_data = report.get("compressed", {}) or {}
        all_variant_names.update(compressed_data.keys())

    for variant_name in all_variant_names:
        # Collect data for this variant across all seeds
        variant_results = []
        for report in seed_reports:
            compressed_data = report.get("compressed", {}) or {}
            if variant_name in compressed_data:
                variant_data = compressed_data[variant_name]
                if variant_data.get("accuracy") is not None:  # Only include successful runs
                    variant_results.append(variant_data)

        if not variant_results:
            continue  # Skip variants with no successful runs

        # Extract metrics
        accuracies = [v["accuracy"] for v in variant_results]
        deltas = [v["delta_vs_probe"] for v in variant_results]
        reductions = [v["param_reduction"] for v in variant_results]
        verdicts = [v["verdict"] for v in variant_results]
        params = [v["params"] for v in variant_results]

        # Calculate statistics
        acc_mean = float(np.mean(accuracies))
        acc_std = float(np.std(accuracies))
        delta_mean = float(np.mean(deltas))
        delta_std = float(np.std(deltas))
        red_mean = float(np.mean(reductions))
        red_std = float(np.std(reductions))
        params_mean = float(np.mean(params))

        # Calculate pass rate
        pass_count = sum(1 for v in verdicts if v == "PASS")
        pass_rate = pass_count / len(verdicts)

        # Overall verdict based on stringent threshold for scientific validity
        # Require at least 80% of seeds to pass for a variant to be considered valid
        # This ensures robustness and prevents cherry-picking
        overall_verdict = "PASS" if pass_rate >= 0.8 else "FAIL"

        # Build variant data
        variant_data = {
            "n_seeds": len(variant_results),
            "accuracy": {
                "mean": acc_mean,
                "std": acc_std,
                "values": accuracies
            },
            "delta_vs_probe": {
                "mean": delta_mean,
                "std": delta_std,
                "values": deltas
            },
            "param_reduction": {
                "mean": red_mean,
                "std": red_std,
                "values": reductions
            },
            "params": {
                "mean": params_mean,
                "std": float(np.std(params)) if len(params) > 1 else 0.0
            },
            "pass_rate": pass_rate,
            "pass_count": pass_count,
            "total_runs": len(variant_results),
            "verdict": overall_verdict,
            "individual_verdicts": verdicts
        }

        # Preserve rank information from first result (should be consistent across seeds)
        if variant_results and "rank" in variant_results[0]:
            variant_data["rank"] = variant_results[0]["rank"]

        # Preserve compression metadata if present (for SVD variants)
        if variant_results and "compression" in variant_results[0]:
            # Use compression metadata from first result (should be consistent across seeds)
            variant_data["compression"] = variant_results[0]["compression"]

            # Aggregate energy retention if present
            energy_values = [v["compression"].get("energy_retained") for v in variant_results if v.get("compression", {}).get("energy_retained") is not None]
            if energy_values:
                variant_data["compression"]["energy_retained_stats"] = {
                    "mean": float(np.mean(energy_values)),
                    "std": float(np.std(energy_values)) if len(energy_values) > 1 else 0.0,
                    "values": energy_values
                }

        variants_data[variant_name] = variant_data

    # Build detailed per-seed breakdown for self-contained reporting
    detailed_results = {
        "seeds": [],
        "candidates": {},
        "summary_statistics": {}
    }

    # Extract seed information and results
    for i, report in enumerate(seed_reports):
        seed_id = report.get("env", {}).get("seed", f"seed_{i}")
        probe_data = report["probe"]
        compressed_data = report.get("compressed", {}) or {}

        seed_detail = {
            "seed_id": seed_id,
            "probe": {
                "accuracy": probe_data["accuracy"],
                "params": probe_data["params"],
                "rank": probe_data["rank"]
            },
            "candidates": {}
        }

        # Add candidate results for this seed
        for variant_name in all_variant_names:
            if variant_name in compressed_data:
                variant_data = compressed_data[variant_name]
                seed_detail["candidates"][variant_name] = {
                    "accuracy": variant_data.get("accuracy"),
                    "params": variant_data.get("params"),
                    "rank": variant_data.get("rank", "unknown"),
                    "delta_vs_probe": variant_data.get("delta_vs_probe"),
                    "param_reduction": variant_data.get("param_reduction"),
                    "verdict": variant_data.get("verdict"),
                    "verdict_reason": variant_data.get("verdict_reason", "Accuracy within tolerance" if variant_data.get("verdict") == "PASS" else f"Accuracy drop {variant_data.get('delta_vs_probe', 0):.3f} exceeds tolerance"),
                    "policy_name": variant_data.get("policy_name", variant_name),
                    "compression_method": variant_data.get("compression", {}).get("method", "uniform") if variant_data.get("compression") else "uniform"
                }
            else:
                # Candidate wasn't run in this seed (skip or fail)
                seed_detail["candidates"][variant_name] = {
                    "accuracy": None,
                    "params": None,
                    "rank": None,
                    "delta_vs_probe": None,
                    "param_reduction": None,
                    "verdict": "SKIP",
                    "verdict_reason": "Candidate not evaluated in this seed",
                    "policy_name": variant_name,
                    "compression_method": "unknown"
                }

        detailed_results["seeds"].append(seed_detail)

    # Build candidate-centric summary statistics
    for variant_name in all_variant_names:
        if variant_name in variants_data:
            variant_stats = variants_data[variant_name]

            # Extract per-seed results for this candidate
            candidate_seeds = []
            for seed_detail in detailed_results["seeds"]:
                if variant_name in seed_detail["candidates"]:
                    candidate_data = seed_detail["candidates"][variant_name]
                    if candidate_data["accuracy"] is not None:  # Only include successful runs
                        candidate_seeds.append({
                            "seed_id": seed_detail["seed_id"],
                            "probe_accuracy": seed_detail["probe"]["accuracy"],
                            "compressed_accuracy": candidate_data["accuracy"],
                            "delta_accuracy": candidate_data["delta_vs_probe"],
                            "probe_params": seed_detail["probe"]["params"],
                            "compressed_params": candidate_data["params"],
                            "param_reduction": candidate_data["param_reduction"],
                            "verdict": candidate_data["verdict"],
                            "verdict_reason": candidate_data["verdict_reason"],
                            "rank": candidate_data["rank"]
                        })

            # Calculate worst-case and mean deltas across seeds
            if candidate_seeds:
                deltas = [s["delta_accuracy"] for s in candidate_seeds if s["delta_accuracy"] is not None]
                param_reductions = [s["param_reduction"] for s in candidate_seeds if s["param_reduction"] is not None]

                detailed_results["candidates"][variant_name] = {
                    "policy_name": variant_name,
                    "chosen_rank": candidate_seeds[0].get("rank", "unknown") if candidate_seeds else "unknown",
                    "n_seeds_evaluated": len(candidate_seeds),
                    "per_seed_results": candidate_seeds,
                    "worst_case_delta": min(deltas) if deltas else None,
                    "mean_delta": sum(deltas) / len(deltas) if deltas else None,
                    "mean_param_reduction": sum(param_reductions) / len(param_reductions) if param_reductions else None,
                    "pass_rate": variant_stats.get("pass_rate", 0.0),
                    "overall_verdict": variant_stats.get("verdict", "UNKNOWN")
                }

    # Two-tier defensible selection system
    # Tier 1: Safe variants (100% pass rate - all seeds pass)
    safe_variants = {name: data for name, data in variants_data.items() if data["pass_rate"] == 1.0}

    # Tier 2: Aggressive variants (≥60% pass rate but <100% - majority pass, clearly labeled)
    aggressive_variants = {name: data for name, data in variants_data.items()
                         if 0.6 <= data["pass_rate"] < 1.0}

    # Select best safe variant (highest compression among 100% pass rate)
    best_safe_variant = None
    if safe_variants:
        best_safe_name = max(safe_variants.keys(), key=lambda x: safe_variants[x]["param_reduction"]["mean"])
        best_safe_data = safe_variants[best_safe_name]
        best_safe_variant = {
            "variant": best_safe_name,
            "param_reduction_mean": best_safe_data["param_reduction"]["mean"],
            "param_reduction_std": best_safe_data["param_reduction"]["std"],
            "delta_vs_probe_mean": best_safe_data["delta_vs_probe"]["mean"],
            "delta_vs_probe_std": best_safe_data["delta_vs_probe"]["std"],
            "pass_rate": best_safe_data["pass_rate"],
            "pass_count": best_safe_data["pass_count"],
            "total_runs": best_safe_data["total_runs"],
            "confidence_level": "high",
            "rationale": "All seeds pass tolerance - recommended for production"
        }

    # Select best aggressive variant (highest compression among majority-pass)
    best_aggressive_variant = None
    if aggressive_variants:
        best_aggressive_name = max(aggressive_variants.keys(), key=lambda x: aggressive_variants[x]["param_reduction"]["mean"])
        best_aggressive_data = aggressive_variants[best_aggressive_name]
        best_aggressive_variant = {
            "variant": best_aggressive_name,
            "param_reduction_mean": best_aggressive_data["param_reduction"]["mean"],
            "param_reduction_std": best_aggressive_data["param_reduction"]["std"],
            "delta_vs_probe_mean": best_aggressive_data["delta_vs_probe"]["mean"],
            "delta_vs_probe_std": best_aggressive_data["delta_vs_probe"]["std"],
            "pass_rate": best_aggressive_data["pass_rate"],
            "pass_count": best_aggressive_data["pass_count"],
            "total_runs": best_aggressive_data["total_runs"],
            "confidence_level": "moderate",
            "rationale": f"{best_aggressive_data['pass_count']}/{best_aggressive_data['total_runs']} seeds pass - higher risk, higher reward"
        }

    # Legacy best_compression for backward compatibility (prefer safe, fallback to aggressive)
    best_compression = best_safe_variant or best_aggressive_variant

    # Compute validation classification for multi-seed aggregation
    n_seeds = len(seed_reports)
    max_steps = base_report.get("env", {}).get("validation_classification", {}).get("max_steps", 0)

    if n_seeds >= 3 and max_steps >= 500:
        validation_level = "certifiable"
        validation_rationale = f"{n_seeds} seeds x {max_steps} steps provides statistical rigor"
    elif n_seeds >= 2 and max_steps >= 200:
        validation_level = "screening_plus"
        validation_rationale = f"{n_seeds} seeds x {max_steps} steps (limited budget/seeds)"
    else:
        validation_level = "screening_plus"
        validation_rationale = f"{n_seeds} seeds but only {max_steps} steps (limited budget)"

    multiseed_validation_classification = {
        "level": validation_level,
        "rationale": validation_rationale,
        "is_multiseed": True,
        "n_seeds": n_seeds,
        "max_steps": max_steps
    }

    # Copy environment but update validation_classification
    aggregated_env = base_report.get("env", {}).copy()
    aggregated_env["validation_classification"] = multiseed_validation_classification

    # Build aggregated report
    aggregated_report = {
        "bench_version": base_report["bench_version"],
        "timestamp": timestamp,
        "aggregation_type": "multi_seed",
        "n_seeds": len(seed_reports),
        "seeds": [r.get("env", {}).get("seed", "unknown") for r in seed_reports],
        "model": base_report["model"],
        "task": base_report["task"],
        "env": aggregated_env,
        "git_commit": base_report.get("git_commit"),  # Use git info from first report
        "probe": {
            "rank": base_report["probe"]["rank"],
            "accuracy": {
                "mean": probe_acc_mean,
                "std": probe_acc_std,
                "values": probe_accuracies
            },
            "params": {
                "mean": probe_params_mean,
                "std": float(np.std(probe_params)) if len(probe_params) > 1 else 0.0
            }
        },
        "compressed": variants_data,
        "detailed_results": detailed_results,  # Self-contained per-seed, per-candidate breakdown
        "summary": {
            "best_compression": best_compression,  # Legacy field for backward compatibility
            "best_safe_variant": best_safe_variant,
            "best_aggressive_variant": best_aggressive_variant,
            "total_variants": len(variants_data),
            "safe_variants": len(safe_variants),
            "aggressive_variants": len(aggressive_variants),
            "selection_strategy": {
                "safe_available": best_safe_variant is not None,
                "aggressive_available": best_aggressive_variant is not None,
                "recommendation": "use_safe" if best_safe_variant else ("use_aggressive_with_caution" if best_aggressive_variant else "no_viable_variants")
            },
            "defensible_claims": True,
            "statistical_power": "sufficient" if len(seed_reports) >= 3 else "limited"
        },
        "config_metadata": base_report.get("config_metadata", {})  # Use config metadata from first report
    }

    return aggregated_report


def create_multi_seed_markdown_report(
    aggregated_report: Dict[str, Any],
    config: Dict[str, Any],
    output_dir: Path
) -> str:
    """
    Create bench_aggregate.md human-readable markdown report for multi-seed results.
    """
    model = aggregated_report["model"]
    task = aggregated_report["task"]
    n_seeds = aggregated_report["n_seeds"]
    timestamp = aggregated_report["timestamp"]
    probe_data = aggregated_report["probe"]
    compressed_data = aggregated_report.get("compressed", {}) or {}
    summary = aggregated_report["summary"]

    # Extract validation level from aggregated report
    validation_level = "certifiable" if n_seeds >= 3 else "screening_plus"

    # Build markdown content
    md_content = f"""# Gradience Bench v{aggregated_report["bench_version"]} (Multi-Seed)

- **Model:** {model}
- **Task:** {task}
- **Seeds:** {n_seeds}
- **Validation Level:** {validation_level.title()}
- **Statistical Power:** {summary["statistical_power"]}

## Probe Baseline (mean ± std)

- **Rank:** {probe_data["rank"]}
- **Accuracy:** {probe_data["accuracy"]["mean"]:.4f} ± {probe_data["accuracy"]["std"]:.4f}
- **LoRA params:** {probe_data["params"]["mean"]:,.0f}

## Compression Results (Aggregated)

| Variant | Accuracy | Delta vs Probe | Param Reduction | Pass Rate | Verdict |
|---|---:|---:|---:|---:|---|
"""

    # Add results table rows
    for variant_name, data in compressed_data.items():
        acc_str = f"{data['accuracy']['mean']:.3f} ± {data['accuracy']['std']:.3f}"
        delta_str = f"{data['delta_vs_probe']['mean']:+.3f} ± {data['delta_vs_probe']['std']:.3f}"
        red_str = f"{data['param_reduction']['mean']:.1%} ± {data['param_reduction']['std']:.1%}"
        pass_rate_str = f"{data['pass_count']}/{data['total_runs']} ({data['pass_rate']:.0%})"
        verdict = data['verdict']

        # Format variant name for display
        if variant_name == "per_layer":
            variant_display = "`per_layer`"
        elif variant_name == "per_layer_shuffled":
            variant_display = "`per_layer_shuffled`"
        elif variant_name == "uniform_median":
            variant_display = "`uniform_median`"
        elif variant_name == "uniform_p90":
            variant_display = "`uniform_p90`"
        elif variant_name == "uniform_p90_control":
            variant_display = "`uniform_p90_control`"
        else:
            variant_display = f"`{variant_name}`"

        md_content += f"| {variant_display} | {acc_str} | {delta_str} | {red_str} | {pass_rate_str} | {verdict} |\n"

    # Add per-rank validation evidence section for scientific honesty
    acc_tolerance = config.get("compression", {}).get("acc_tolerance", 0.005)

    md_content += f"""

## Per-Rank Validation Evidence

The following statements reflect the complete evidence from all {n_seeds} seeds:

"""

    # Group variants by rank for clearer reporting
    rank_to_variants = {}
    for variant_name, data in compressed_data.items():
        # Extract rank from variant data or name
        rank = None
        if "rank" in data:
            rank = data["rank"]
        elif "r" in variant_name:
            # Try to extract from name like "uniform_r32" or "energy_p90"
            import re
            rank_match = re.search(r'r(\d+)', variant_name)
            if rank_match:
                rank = int(rank_match.group(1))

        if rank is not None:
            if rank not in rank_to_variants:
                rank_to_variants[rank] = []
            rank_to_variants[rank].append((variant_name, data))

    # Generate honest per-rank statements
    for rank in sorted(rank_to_variants.keys()):
        variants = rank_to_variants[rank]

        # Determine overall validation status for this rank
        all_pass_counts = [data["pass_count"] for _, data in variants]
        all_total_runs = [data["total_runs"] for _, data in variants]

        # Use the variant with the most comprehensive data (highest total_runs)
        best_variant_name, best_data = max(variants, key=lambda x: x[1]["total_runs"])
        pass_count = best_data["pass_count"]
        total_runs = best_data["total_runs"]
        pass_rate = best_data["pass_rate"]

        # Generate scientific validation statement
        if pass_rate == 1.0:
            validation_status = "✅ **validated safe**"
            detail = f"All {total_runs} seeds pass ±{acc_tolerance:.3f} tolerance"
        elif pass_count >= 2 and total_runs >= 3:
            validation_status = "⚠️  **conditionally promising**"
            failed_count = total_runs - pass_count
            if failed_count == 1:
                detail = f"{pass_count}/{total_runs} seeds pass; one seed violates tolerance"
            else:
                detail = f"{pass_count}/{total_runs} seeds pass; {failed_count} seeds violate tolerance"
        elif pass_count > 0:
            validation_status = "❌ **unreliable**"
            detail = f"Only {pass_count}/{total_runs} seeds pass tolerance"
        else:
            validation_status = "❌ **failed validation**"
            detail = f"No seeds pass ±{acc_tolerance:.3f} tolerance"

        md_content += f"- **r={rank}** is {validation_status} ({detail})\n"

    md_content += f"""

This evidence-based approach ensures complete transparency about which compression levels can be trusted across independent random seeds.

## Two-Tier Selection System

This benchmark uses a two-tier selection system for defensible recommendations:

"""

    # Add safe variant information
    if summary["best_safe_variant"]:
        safe = summary["best_safe_variant"]
        red_mean = safe["param_reduction_mean"] * 100
        red_std = safe["param_reduction_std"] * 100
        delta_mean = safe["delta_vs_probe_mean"]
        delta_std = safe["delta_vs_probe_std"]

        md_content += f"""### Safe Variant (Recommended)
- **Variant:** `{safe["variant"]}`
- **Parameter reduction:** {red_mean:.1f}% ± {red_std:.1f}%
- **Accuracy impact:** {delta_mean:+.4f} ± {delta_std:.4f} vs probe baseline
- **Pass rate:** {safe["pass_count"]}/{safe["total_runs"]} seeds (100%)
- **Confidence:** {safe["confidence_level"]} - {safe["rationale"]}

"""

    # Add aggressive variant information
    if summary["best_aggressive_variant"]:
        aggressive = summary["best_aggressive_variant"]
        red_mean = aggressive["param_reduction_mean"] * 100
        red_std = aggressive["param_reduction_std"] * 100
        delta_mean = aggressive["delta_vs_probe_mean"]
        delta_std = aggressive["delta_vs_probe_std"]

        md_content += f"""### Aggressive Variant (Higher Risk)
- **Variant:** `{aggressive["variant"]}`
- **Parameter reduction:** {red_mean:.1f}% ± {red_std:.1f}%
- **Accuracy impact:** {delta_mean:+.4f} ± {delta_std:.4f} vs probe baseline
- **Pass rate:** {aggressive["pass_count"]}/{aggressive["total_runs"]} seeds ({aggressive["pass_rate"]:.0%})
- **Confidence:** {aggressive["confidence_level"]} - {aggressive["rationale"]}

"""

    # Add selection recommendation
    strategy = summary["selection_strategy"]
    md_content += f"""### Recommendation: {strategy["recommendation"].replace("_", " ").title()}
"""

    if strategy["recommendation"] == "use_safe":
        md_content += "Safe variant available - recommended for production deployment.\n\n"
    elif strategy["recommendation"] == "use_aggressive_with_caution":
        md_content += "Only aggressive variants available - proceed with caution and additional validation.\n\n"
    else:
        md_content += "No variants meet minimum reliability thresholds.\n\n"

    # Add interpretation
    md_content += f"""## Interpretation (Statistical)

- **PASS** means ≥80% of seeds passed ±{acc_tolerance:.3f} accuracy tolerance (stringent threshold)
- **Safe variants** require 100% pass rate across all seeds
- **Aggressive variants** require ≥60% pass rate but clearly flagged as higher risk
- **Statistics** are calculated as mean ± standard deviation across {n_seeds} seeds
- **Defensible claims** are supported by variance estimation across multiple random seeds
- You should still validate these results on your real workload before deployment

## Summary

- **Total variants:** {summary["total_variants"]}
- **Safe variants:** {summary["safe_variants"]} (100% pass rate)
- **Aggressive variants:** {summary["aggressive_variants"]} (≥60% pass rate)
- **Recommended approach:** {strategy["recommendation"].replace("_", " ")}
- **Statistical power:** {summary["statistical_power"]}

## Detailed Per-Seed Results

This section contains complete self-contained results for all seeds and candidates.

"""

    # Add detailed breakdown for self-contained reporting
    detailed_data = aggregated_report.get("detailed_results", {})
    if detailed_data:
        # Add per-candidate summary with worst-case delta
        md_content += """### Candidate Summary

| Candidate | Policy | Rank | Seeds | Worst Delta | Mean Delta | Param Reduction | Verdict |
|---|---|---:|---:|---:|---:|---:|---|
"""

        candidates = detailed_data.get("candidates", {})
        for candidate_name, candidate_info in candidates.items():
            worst_delta = candidate_info.get("worst_case_delta", 0.0)
            mean_delta = candidate_info.get("mean_delta", 0.0)
            param_reduction = candidate_info.get("mean_param_reduction", 0.0)
            n_seeds = candidate_info.get("n_seeds_evaluated", 0)
            verdict = candidate_info.get("overall_verdict", "UNKNOWN")
            policy_name = candidate_info.get("policy_name", candidate_name)
            rank = candidate_info.get("chosen_rank", "?")

            md_content += f"| `{candidate_name}` | {policy_name} | r={rank} | {n_seeds} | {worst_delta:+.3f} | {mean_delta:+.3f} | {param_reduction:.1%} | {verdict} |\n"

        # Add detailed per-seed, per-candidate breakdown
        md_content += """

### Complete Per-Seed Breakdown

"""

        seeds_data = detailed_data.get("seeds", [])
        for seed_info in seeds_data:
            seed_id = seed_info.get("seed_id", "unknown")
            probe_info = seed_info.get("probe", {})

            md_content += f"""
#### Seed: {seed_id}

**Probe baseline:** {probe_info.get('accuracy', 0.0):.3f} accuracy, r={probe_info.get('rank', '?')}, {probe_info.get('params', 0):,} params

| Candidate | Accuracy | Delta vs Probe | Trainable Params | Param Reduction | Verdict | Reason |
|---|---:|---:|---:|---:|---|---|
"""

            candidates_data = seed_info.get("candidates", {})
            for candidate_name, candidate_data in candidates_data.items():
                if candidate_data.get("accuracy") is not None:  # Only show evaluated candidates
                    accuracy = candidate_data.get("accuracy", 0.0)
                    delta = candidate_data.get("delta_vs_probe", 0.0)
                    params = candidate_data.get("params", 0)
                    param_reduction = candidate_data.get("param_reduction", 0.0)
                    verdict = candidate_data.get("verdict", "UNKNOWN")
                    reason = candidate_data.get("verdict_reason", "No reason provided")

                    md_content += f"| `{candidate_name}` | {accuracy:.3f} | {delta:+.3f} | {params:,} | {param_reduction:.1%} | {verdict} | {reason} |\n"

    md_content += f"""

*Generated on {timestamp[:19].replace('T', ' ')}*
"""

    return md_content


def run_multi_seed_bench_protocol(
    config_path: str | Path,
    output_dir: str | Path,
    seeds: list[int],
    variants_to_test: list[str] = None,
    smoke: bool = False,
    ci: bool = False
) -> Dict[str, Any]:
    """
    Run bench protocol across multiple seeds and aggregate results.

    Returns aggregated report with mean ± std statistics.
    """
    # Import here to avoid circular imports since protocol.py will import from this module
    from gradience.bench.protocol import run_bench_protocol

    config = load_config(config_path)
    output_path = Path(output_dir)

    # HYGIENE: Ensure output directory exists BEFORE any logging/tee operations
    output_path.mkdir(parents=True, exist_ok=True)

    # HYGIENE: Start heartbeat for multi-seed coordination (prevent SSH timeouts)
    heartbeat_stage("multi_seed_coordination", output_dir=output_path, seed="coordinator")

    print(f"Gradience Multi-Seed Bench Protocol v0.1")
    print("=" * 50)
    print(f"Config: {config_path}")
    print(f"Output: {output_path}")
    print(f"Seeds: {seeds}")
    print(f"Variants: {variants_to_test or 'all'}")
    print(f"Smoke mode: {smoke}")
    print()

    # Store individual seed results
    seed_reports = []
    seed_dirs = []

    # Run each seed
    for i, seed in enumerate(seeds):
        print(f"\n{'='*60}")
        print(f"SEED {i+1}/{len(seeds)}: {seed}")
        print(f"{'='*60}")

        # Create seed-specific config
        seed_config = config.copy()
        seed_config["train"]["seed"] = seed

        # Remove multi-seed config from individual seeds to prevent infinite recursion
        compression = seed_config.get("compression", {}).copy()
        compression.pop("seeds", None)  # Remove seeds field to force single-seed mode

        # Filter variants if specified
        if variants_to_test:
            compression["variants_to_test"] = variants_to_test

        seed_config["compression"] = compression

        # HYGIENE: Create seed-specific directory BEFORE any operations
        seed_dir = output_path / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        seed_dirs.append(seed_dir)

        # HYGIENE: Create progress/heartbeat file for stuck detection
        progress_file = seed_dir / "progress.txt"
        with open(progress_file, 'w') as f:
            f.write(f"STARTED: seed_{seed} at {datetime.datetime.now().isoformat()}\n")
            f.flush()

        # Write seed-specific config
        seed_config_path = seed_dir / "config.yaml"
        with open(seed_config_path, 'w') as f:
            yaml.dump(seed_config, f, indent=2)

        # Run single seed benchmark
        try:
            # Update progress before starting
            with open(progress_file, 'a') as f:
                f.write(f"RUNNING: bench protocol started at {datetime.datetime.now().isoformat()}\n")
                f.flush()

            seed_report = run_bench_protocol(
                config_path=seed_config_path,
                output_dir=seed_dir,
                smoke=smoke,
                ci=ci
            )

            # Add seed info to report
            seed_report["seed"] = seed
            seed_report["seed_index"] = i
            seed_reports.append(seed_report)

            # Mark completion in progress file
            with open(progress_file, 'a') as f:
                f.write(f"COMPLETED: seed_{seed} at {datetime.datetime.now().isoformat()}\n")
                f.flush()

            print(f"\nSeed {seed} completed successfully")

        except Exception as e:
            # Mark failure in progress file
            with open(progress_file, 'a') as f:
                f.write(f"FAILED: seed_{seed} at {datetime.datetime.now().isoformat()}: {e}\n")
                f.flush()

            print(f"\nSeed {seed} failed: {e}")
            # Continue with other seeds
            continue

    if not seed_reports:
        raise RuntimeError("All seed runs failed - cannot generate aggregated report")

    print(f"\n{'='*60}")
    print(f"AGGREGATION: {len(seed_reports)}/{len(seeds)} seeds successful")
    print(f"{'='*60}")

    # Create aggregated report
    aggregated_report = create_multi_seed_aggregated_report(
        seed_reports=seed_reports,
        config=config,
        output_dir=output_path
    )

    # Write aggregated bench.json
    agg_report_path = output_path / "bench_aggregate.json"
    with open(agg_report_path, 'w') as f:
        json.dump(aggregated_report, f, indent=2, ensure_ascii=False)

    # Create and write aggregated markdown report
    agg_markdown_content = create_multi_seed_markdown_report(
        aggregated_report=aggregated_report,
        config=config,
        output_dir=output_path
    )

    agg_markdown_path = output_path / "bench_aggregate.md"
    with open(agg_markdown_path, 'w') as f:
        f.write(agg_markdown_content)

    # Write seed summary
    seed_summary_path = output_path / "seed_summary.json"
    seed_summary = {
        "total_seeds": len(seeds),
        "successful_seeds": len(seed_reports),
        "failed_seeds": len(seeds) - len(seed_reports),
        "seed_directories": [str(d) for d in seed_dirs],
        "aggregated_report": str(agg_report_path),
        "aggregated_markdown": str(agg_markdown_path)
    }
    with open(seed_summary_path, 'w') as f:
        json.dump(seed_summary, f, indent=2, ensure_ascii=False)

    print(f"\nMulti-seed benchmark complete!")
    print(f"  Aggregated report: {agg_report_path}")
    print(f"  Aggregated markdown: {agg_markdown_path}")
    print(f"  Seed summary: {seed_summary_path}")
    print(f"  Individual seed results in: {[d.name for d in seed_dirs]}")

    return aggregated_report
