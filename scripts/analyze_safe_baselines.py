#!/usr/bin/env python3
"""
Analyze safety baseline candidates to identify genuinely safe uniform approaches.

Usage:
    python scripts/analyze_safe_baselines.py \
        --uniform-r16 bench_runs/uniform_r16_seed* \
        --uniform-r16-extended bench_runs/*uniform_r16_extended_seed* \
        --uniform-r20 bench_runs/*uniform_r20_seed* \
        --uniform-r24 bench_runs/*uniform_r24_seed*
"""
import json
import argparse
from pathlib import Path
import numpy as np
from collections import defaultdict

def load_bench_result(bench_dir):
    """Load results from a bench.json file."""
    bench_path = Path(bench_dir) / "bench.json"
    if not bench_path.exists():
        print(f"Warning: No bench.json in {bench_dir}")
        return None
    
    with open(bench_path) as f:
        return json.load(f)

def extract_safety_metrics(result, baseline_name):
    """Extract safety-focused metrics for uniform baselines."""
    if not result:
        return None
        
    probe = result.get("probe", {})
    compressed = result.get("compressed", {})
    
    # Find the uniform variant
    variant_data = None
    if "uniform_median" in compressed:
        variant_data = compressed["uniform_median"]
    elif "uniform_p90" in compressed:
        variant_data = compressed["uniform_p90"]
    elif len(compressed) == 1:  # Single variant
        variant_data = list(compressed.values())[0]
    
    if not variant_data:
        print(f"Warning: No uniform data found in {result.get('config_path', 'unknown')}")
        return None
    
    return {
        "baseline": baseline_name,
        "seed": result.get("config_path", "unknown"),
        "probe_accuracy": probe.get("accuracy"),
        "probe_params": probe.get("params"),
        "compressed_accuracy": variant_data.get("accuracy"), 
        "compressed_params": variant_data.get("params"),
        "param_reduction": variant_data.get("param_reduction"),
        "delta_vs_probe": variant_data.get("delta_vs_probe"),
        "verdict": variant_data.get("verdict"),
        "rank_used": variant_data.get("rank", "unknown"),
    }

def calculate_safety_stats(values, label):
    """Calculate safety-focused statistics."""
    if not values:
        return {"mean": "N/A", "std": "N/A", "min": "N/A", "max": "N/A"}
    
    arr = np.array(values)
    return {
        "mean": arr.mean(),
        "std": arr.std(),
        "min": arr.min(),
        "max": arr.max(),
        "count": len(values)
    }

def analyze_safety_criteria(baselines, safety_policy=None):
    """Analyze which baselines meet safety criteria based on explicit policy."""
    if safety_policy is None:
        safety_policy = {
            "min_pass_rate": 2/3,  # ≥ 2/3 seeds must pass
            "min_worst_delta": -0.025,  # Worst seed Δ ≥ -2.5%
            "name": "Default Safe Uniform Policy"
        }
    
    safety_analysis = {}
    
    for baseline_name, metrics_list in baselines.items():
        if not metrics_list:
            continue
            
        # Extract key safety metrics
        accuracies = [m["compressed_accuracy"] for m in metrics_list if m["compressed_accuracy"] is not None]
        deltas = [m["delta_vs_probe"] for m in metrics_list if m["delta_vs_probe"] is not None]
        reductions = [m["param_reduction"]*100 for m in metrics_list if m["param_reduction"] is not None]
        verdicts = [m["verdict"] for m in metrics_list if m["verdict"] is not None]
        
        # Safety criteria analysis
        pass_count = sum(1 for v in verdicts if v == "PASS")
        total_count = len(verdicts)
        pass_rate = pass_count / total_count if total_count > 0 else 0
        
        # Stability analysis (lower std = more stable)
        delta_stats = calculate_safety_stats(deltas, "delta")
        acc_stats = calculate_safety_stats(accuracies, "accuracy")
        
        # Conservative thresholds for "safe"
        worst_delta = max(deltas) if deltas else float('inf')  # Worst accuracy drop
        best_delta = min(deltas) if deltas else float('-inf')  # Best accuracy retention
        
        # Apply explicit safety policy
        meets_pass_rate = pass_rate >= safety_policy["min_pass_rate"]
        meets_worst_delta = worst_delta >= safety_policy["min_worst_delta"]
        is_safe = meets_pass_rate and meets_worst_delta
        
        safety_analysis[baseline_name] = {
            "pass_rate": pass_rate,
            "pass_fraction": f"{pass_count}/{total_count}",
            "delta_stability": delta_stats["std"],
            "worst_case_delta": worst_delta,
            "best_case_delta": best_delta,
            "mean_reduction": calculate_safety_stats(reductions, "reduction")["mean"],
            "total_seeds": total_count,
            "meets_pass_rate": meets_pass_rate,
            "meets_worst_delta": meets_worst_delta,
            "verdict": "SAFE" if is_safe else "RISKY",
            "safety_policy": safety_policy
        }
    
    return safety_analysis

def main():
    parser = argparse.ArgumentParser(description="Analyze uniform baseline safety")
    parser.add_argument("--uniform-r16", nargs="*", help="Uniform r=16 benchmark directories")
    parser.add_argument("--uniform-r16-extended", nargs="*", help="Uniform r=16 extended training directories") 
    parser.add_argument("--uniform-r20", nargs="*", help="Uniform r=20 benchmark directories")
    parser.add_argument("--uniform-r24", nargs="*", help="Uniform r=24 benchmark directories")
    args = parser.parse_args()
    
    baselines = {}
    
    # Load all baseline results
    for baseline_name, bench_dirs in [
        ("Uniform r=16", args.uniform_r16 or []),
        ("Uniform r=16 Extended", args.uniform_r16_extended or []),
        ("Uniform r=20", args.uniform_r20 or []),
        ("Uniform r=24", args.uniform_r24 or []),
    ]:
        if not bench_dirs:
            continue
            
        baseline_metrics = []
        for bench_dir in bench_dirs:
            result = load_bench_result(bench_dir)
            metrics = extract_safety_metrics(result, baseline_name)
            if metrics:
                baseline_metrics.append(metrics)
        
        if baseline_metrics:
            baselines[baseline_name] = baseline_metrics
    
    if not baselines:
        print("❌ No baseline data found. Please run benchmarks first.")
        return
    
    print("=== SAFE UNIFORM BASELINE ANALYSIS ===")
    print()
    
    # Explicit Safety Policy Statement
    safety_results = analyze_safety_criteria(baselines)
    first_result = next(iter(safety_results.values()), {})
    policy = first_result.get("safety_policy", {})
    
    print("📋 **SAFETY POLICY DEFINITION:**")
    print(f"   • Safe uniform baseline = ≥ {policy.get('min_pass_rate', 2/3):.0%} seeds PASS AND worst seed Δ ≥ {policy.get('min_worst_delta', -0.025):+.1%}")
    print(f"   • Preferred baseline = highest compression among safe candidates")  
    print(f"   • If no candidate is safe → recommend per-layer adaptive as default")
    print()
    
    # Safety overview table
    print(f"{'Baseline':<20} {'Pass Rate':<10} {'Worst Δ':<10} {'Best Δ':<10} {'Reduction':<12} {'Safety':<8}")
    print("-" * 80)
    
    for baseline_name in baselines.keys():
        if baseline_name not in safety_results:
            continue
            
        safety = safety_results[baseline_name]
        pass_rate = f"{safety['pass_fraction']}"
        worst_delta = f"{safety['worst_case_delta']:+.3f}" if safety['worst_case_delta'] != float('inf') else "N/A"
        best_delta = f"{safety['best_case_delta']:+.3f}" if safety['best_case_delta'] != float('-inf') else "N/A"
        reduction = f"{safety['mean_reduction']:.1f}%" if safety['mean_reduction'] != "N/A" else "N/A"
        verdict = safety['verdict']
        
        print(f"{baseline_name:<20} {pass_rate:<10} {worst_delta:<10} {best_delta:<10} {reduction:<12} {verdict:<8}")
    
    print()
    print("=== PRODUCT DECISION ===")
    
    # Find safe baselines based on explicit policy
    safe_baselines = {name: safety for name, safety in safety_results.items() 
                      if safety['verdict'] == 'SAFE'}
    
    if not safe_baselines:
        print("🚫 **DECISION: No safe uniform baseline found**")
        print()
        print("**Policy Violation Summary:**")
        for baseline_name, safety in safety_results.items():
            meets_pass = "✅" if safety['meets_pass_rate'] else "❌"
            meets_delta = "✅" if safety['meets_worst_delta'] else "❌"
            print(f"   {baseline_name}: {meets_pass} Pass rate | {meets_delta} Worst delta")
        print()
        print("**RECOMMENDED ACTION:**")
        print("→ **Default to per-layer adaptive compression** as primary strategy")
        print("→ Consider uniform r=24+ for conservative safety margin")
        print("→ Re-evaluate policy thresholds if business requirements allow")
        
    else:
        # Rank safe candidates by compression efficiency (highest compression wins)
        safe_ranked = sorted(safe_baselines.items(), 
                           key=lambda x: x[1]['mean_reduction'], reverse=True)
        
        preferred_baseline, preferred_safety = safe_ranked[0]
        
        print(f"✅ **DECISION: {preferred_baseline} selected as safe uniform baseline**")
        print()
        print(f"**Rationale:** Highest compression ({preferred_safety['mean_reduction']:.1f}% reduction) among policy-compliant candidates")
        print()
        
        print("**Policy Compliance:**")
        for baseline, safety in safe_ranked:
            pass_icon = "✅" if safety['meets_pass_rate'] else "❌"
            delta_icon = "✅" if safety['meets_worst_delta'] else "❌"
            print(f"   {baseline}: {pass_icon} {safety['pass_fraction']} pass rate | {delta_icon} {safety['worst_case_delta']:+.3f} worst delta")
        
        print()
        print("**IMPLEMENTATION GUIDANCE:**")
        print(f"→ **Primary:** Use {preferred_baseline} for uniform compression needs")
        if len(safe_ranked) > 1:
            fallback_baseline, fallback_safety = safe_ranked[1]
            print(f"→ **Fallback:** Use {fallback_baseline} for ultra-conservative scenarios")
        print(f"→ **Avoid:** All non-compliant uniform approaches")
        print(f"→ **Monitor:** Validate safety policy holds on production workloads")
    
    print("=== DETAILED BREAKDOWN ===")
    for baseline_name, metrics_list in baselines.items():
        print(f"\n{baseline_name}:")
        print(f"{'Seed':<15} {'Rank':<6} {'Compressed Acc':<15} {'Δ vs Probe':<12} {'Reduction':<12} {'Verdict'}")
        print("-" * 80)
        
        for m in metrics_list:
            seed_name = Path(m["seed"]).stem if m["seed"] != "unknown" else "unknown"
            rank = str(m["rank_used"])
            comp_acc = f"{m['compressed_accuracy']:.4f}" if m["compressed_accuracy"] else "N/A"
            delta = f"{m['delta_vs_probe']:+.4f}" if m["delta_vs_probe"] else "N/A"
            reduction = f"{m['param_reduction']*100:.1f}%" if m["param_reduction"] else "N/A"
            verdict = m["verdict"] or "N/A"
            
            print(f"{seed_name:<15} {rank:<6} {comp_acc:<15} {delta:<12} {reduction:<12} {verdict}")

if __name__ == "__main__":
    main()