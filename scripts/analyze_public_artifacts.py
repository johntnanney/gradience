#!/usr/bin/env python3
"""
Public Artifact Analysis for Mistral + GSM8K Benchmark

Extracts key insights from the benchmark results for public demonstration:
- Second rung logic decision traces
- Compression effectiveness analysis
- Statistical robustness across seeds
- Mathematical reasoning preservation

Usage:
    python scripts/analyze_public_artifacts.py /workspace/public_artifacts/mistral_gsm8k_demo
"""

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any


def load_json_safe(path: Path) -> dict[str, Any]:
    """Safely load JSON file with error handling."""
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        print(f"⚠️  Could not load {path}: {e}")
        return {}


def analyze_decision_trace(seed_dir: Path) -> dict[str, Any]:
    """Analyze decision trace artifacts from a seed directory."""
    compression_configs = load_json_safe(seed_dir / "compression_configs.json")

    analysis = {
        "second_rung_triggered": False,
        "tier_a_candidates": [],
        "tier_b_candidates": [],
        "policy_candidates": [],
        "total_candidates": 0,
        "decision_factors": {},
    }

    if not compression_configs:
        return analysis

    analysis["total_candidates"] = len(compression_configs)

    for variant_name, config in compression_configs.items():
        policy_type = config.get("policy_type", "unknown")

        if policy_type == "second_rung_tier_a":
            analysis["second_rung_triggered"] = True
            analysis["tier_a_candidates"].append(
                {
                    "variant": variant_name,
                    "rank": config.get("actual_r", 0),
                    "suggested_r": config.get("suggested_r", 0),
                }
            )
        elif policy_type == "second_rung_tier_b":
            analysis["second_rung_triggered"] = True
            analysis["tier_b_candidates"].append(
                {
                    "variant": variant_name,
                    "rank": config.get("actual_r", 0),
                    "suggested_r": config.get("suggested_r", 0),
                }
            )
        else:
            analysis["policy_candidates"].append(
                {
                    "variant": variant_name,
                    "policy": policy_type,
                    "rank": config.get("actual_r", 0),
                    "suggested_r": config.get("suggested_r", 0),
                }
            )

    return analysis


def analyze_compression_effectiveness(seed_dir: Path) -> dict[str, Any]:
    """Analyze compression effectiveness from bench results."""
    bench_data = load_json_safe(seed_dir / "bench.json")

    if not bench_data:
        return {}

    probe = bench_data.get("probe", {})
    compressed = bench_data.get("compressed", {})

    analysis = {
        "probe_quality": {
            "accuracy": probe.get("accuracy", 0),
            "params": probe.get("params", 0),
            "rank": probe.get("rank", 0),
        },
        "compression_candidates": {},
        "best_compression": None,
        "effectiveness_metrics": {},
    }

    # Analyze each compression candidate
    for variant_name, variant_data in compressed.items():
        if isinstance(variant_data, dict) and "accuracy" in variant_data:
            params = variant_data.get("params", 0)
            accuracy = variant_data.get("accuracy", 0)
            rank = variant_data.get("rank", 0)

            # Calculate compression metrics
            param_reduction = (probe.get("params", 1) - params) / probe.get("params", 1)
            accuracy_retention = accuracy / max(probe.get("accuracy", 0.001), 0.001)
            efficiency = accuracy_retention / max((params / probe.get("params", 1)), 0.001)

            analysis["compression_candidates"][variant_name] = {
                "accuracy": accuracy,
                "params": params,
                "rank": rank,
                "param_reduction": param_reduction,
                "accuracy_retention": accuracy_retention,
                "efficiency_score": efficiency,
            }

    # Identify best compression
    best_variant = None
    best_score = 0
    for variant_name, metrics in analysis["compression_candidates"].items():
        score = metrics["efficiency_score"]
        if score > best_score:
            best_score = score
            best_variant = variant_name

    analysis["best_compression"] = best_variant

    return analysis


def analyze_aggregate_results(output_dir: Path) -> dict[str, Any]:
    """Analyze multi-seed aggregate results."""
    aggregate_data = load_json_safe(output_dir / "bench_aggregate.json")

    if not aggregate_data:
        return {}

    analysis = {"statistical_summary": {}, "cross_seed_consistency": {}, "variance_analysis": {}}

    # Extract probe statistics
    probe_data = aggregate_data.get("probe", {})
    if "accuracy" in probe_data:
        acc_stats = probe_data["accuracy"]
        analysis["statistical_summary"]["probe_accuracy"] = {
            "mean": acc_stats.get("mean", 0),
            "std": acc_stats.get("std", 0),
            "values": acc_stats.get("values", []),
        }

    # Extract compression statistics from detailed results
    detailed = aggregate_data.get("detailed_results", {})
    seeds_data = detailed.get("seeds", [])

    if seeds_data:
        # Analyze cross-seed candidate consistency
        all_candidates = set()
        seed_candidates = []

        for seed_data in seeds_data:
            candidates = set(seed_data.get("candidates", {}).keys())
            all_candidates.update(candidates)
            seed_candidates.append(candidates)

        # Calculate candidate consistency
        consistent_candidates = all_candidates.copy()
        for candidates in seed_candidates:
            consistent_candidates &= candidates

        analysis["cross_seed_consistency"] = {
            "total_unique_candidates": len(all_candidates),
            "consistent_candidates": list(consistent_candidates),
            "consistency_rate": len(consistent_candidates) / max(len(all_candidates), 1),
        }

    return analysis


def generate_public_summary(output_dir: Path) -> str:
    """Generate a public-facing summary of the benchmark results."""

    # Analyze individual seeds
    seed_analyses = []
    seed_dirs = [d for d in output_dir.iterdir() if d.is_dir() and d.name.startswith("seed_")]

    for seed_dir in sorted(seed_dirs):
        decision_analysis = analyze_decision_trace(seed_dir)
        compression_analysis = analyze_compression_effectiveness(seed_dir)

        seed_analyses.append(
            {"seed_dir": seed_dir.name, "decision_trace": decision_analysis, "compression": compression_analysis}
        )

    # Analyze aggregate results
    aggregate_analysis = analyze_aggregate_results(output_dir)

    # Generate summary
    summary = []
    summary.append("# Gradience Benchmark: Mistral-7B Mathematical Reasoning Compression")
    summary.append("=" * 70)
    summary.append("")

    # Executive summary
    total_seeds = len(seed_analyses)
    second_rung_seeds = sum(1 for s in seed_analyses if s["decision_trace"]["second_rung_triggered"])

    summary.append("## Executive Summary")
    summary.append("")
    summary.append("**Model**: Mistral-7B-v0.1 fine-tuned on GSM8K mathematical reasoning")
    summary.append(f"**Seeds Analyzed**: {total_seeds}")
    summary.append(f"**Second Rung Logic Triggered**: {second_rung_seeds}/{total_seeds} seeds")
    summary.append("")

    # Statistical robustness
    if aggregate_analysis and "statistical_summary" in aggregate_analysis:
        probe_stats = aggregate_analysis["statistical_summary"].get("probe_accuracy", {})
        if probe_stats:
            mean_acc = probe_stats.get("mean", 0)
            std_acc = probe_stats.get("std", 0)
            summary.append(f"**Probe Quality**: {mean_acc:.3f} ± {std_acc:.3f} exact match accuracy")

    # Second rung analysis
    summary.append("")
    summary.append("## Audit-Driven Compression Analysis")
    summary.append("")

    if second_rung_seeds > 0:
        summary.append("✅ **Second Rung Logic Successfully Triggered**")
        summary.append("")

        # Aggregate second rung candidates
        all_tier_a = []
        all_tier_b = []

        for analysis in seed_analyses:
            dt = analysis["decision_trace"]
            all_tier_a.extend(dt["tier_a_candidates"])
            all_tier_b.extend(dt["tier_b_candidates"])

        if all_tier_a:
            summary.append(f"**Tier A (Moderate) Candidates**: {len(all_tier_a)} generated")
            avg_rank_a = statistics.mean([c["rank"] for c in all_tier_a])
            summary.append(f"  - Average rank: {avg_rank_a:.1f}")

        if all_tier_b:
            summary.append(f"**Tier B (Aggressive) Candidates**: {len(all_tier_b)} generated")
            avg_rank_b = statistics.mean([c["rank"] for c in all_tier_b])
            summary.append(f"  - Average rank: {avg_rank_b:.1f}")

    else:
        summary.append("⚪ **Second Rung Logic Not Triggered**")
        summary.append("  - Audit metrics did not meet thresholds for additional compression candidates")
        summary.append("  - Policy-based compression candidates generated successfully")

    # Compression effectiveness
    summary.append("")
    summary.append("## Compression Effectiveness")
    summary.append("")

    # Find best compressions across seeds
    best_compressions = []
    for analysis in seed_analyses:
        comp = analysis["compression"]
        if comp.get("best_compression"):
            best_compressions.append(comp)

    if best_compressions:
        summary.append("**Best Compression Candidates**:")
        for i, comp in enumerate(best_compressions):
            best_name = comp["best_compression"]
            if best_name in comp["compression_candidates"]:
                metrics = comp["compression_candidates"][best_name]
                param_red = metrics["param_reduction"] * 100
                acc_ret = metrics["accuracy_retention"] * 100
                summary.append(
                    f"  - Seed {i + 1}: {best_name} ({param_red:.1f}% size reduction, {acc_ret:.1f}% accuracy retention)"
                )

    # Cross-seed consistency
    consistency = aggregate_analysis.get("cross_seed_consistency", {})
    if consistency:
        rate = consistency.get("consistency_rate", 0) * 100
        summary.append("")
        summary.append(f"**Cross-Seed Consistency**: {rate:.1f}% of candidates appeared across all seeds")

    # Artifacts generated
    summary.append("")
    summary.append("## Generated Artifacts")
    summary.append("")
    summary.append("- `bench_aggregate.json`: Complete multi-seed statistical analysis")
    summary.append("- `bench_aggregate.md`: Human-readable benchmark report")
    summary.append("- `seed_*/compression_configs.json`: Decision trace for each seed")
    summary.append("- `seed_*/bench.json`: Detailed per-seed results")
    summary.append("- `seed_*/progress.txt` & `seed_*/heartbeat.log`: Engineering monitoring")

    return "\n".join(summary)


def main():
    """Main analysis function."""
    parser = argparse.ArgumentParser(description="Analyze public benchmark artifacts")
    parser.add_argument("output_dir", help="Path to benchmark output directory")
    parser.add_argument("--format", choices=["summary", "detailed"], default="summary", help="Output format")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        print(f"❌ Output directory does not exist: {output_dir}")
        sys.exit(1)

    print(f"📊 Analyzing benchmark results: {output_dir}")
    print()

    if args.format == "summary":
        summary = generate_public_summary(output_dir)
        print(summary)
    else:
        # Detailed analysis (for development)
        seed_dirs = [d for d in output_dir.iterdir() if d.is_dir() and d.name.startswith("seed_")]
        for seed_dir in sorted(seed_dirs):
            print(f"🔍 {seed_dir.name}:")
            decision_trace = analyze_decision_trace(seed_dir)
            compression = analyze_compression_effectiveness(seed_dir)

            print(f"  Second rung triggered: {decision_trace['second_rung_triggered']}")
            print(f"  Total candidates: {decision_trace['total_candidates']}")
            print(f"  Best compression: {compression.get('best_compression', 'None')}")
            print()


if __name__ == "__main__":
    main()
