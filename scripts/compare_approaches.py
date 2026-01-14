#!/usr/bin/env python3
"""
Compare different compression approaches across multiple seeds.

Usage:
    python scripts/compare_approaches.py \
        --per-layer bench_runs/multiseed_seed* \
        --uniform-r8 bench_runs/uniform_r8_seed*
"""
import json
import argparse
from pathlib import Path
import numpy as np

def load_bench_result(bench_dir):
    """Load results from a bench.json file."""
    bench_path = Path(bench_dir) / "bench.json"
    if not bench_path.exists():
        print(f"Warning: No bench.json in {bench_dir}")
        return None
    
    with open(bench_path) as f:
        return json.load(f)

def extract_compression_metrics(result, variant_name):
    """Extract compression metrics for a specific variant."""
    if not result:
        return None
        
    probe = result.get("probe", {})
    compressed = result.get("compressed", {})
    
    # Find the specific variant
    variant_data = None
    if variant_name in compressed:
        variant_data = compressed[variant_name]
    elif "per_layer" in compressed:  # fallback for per_layer
        variant_data = compressed["per_layer"]
    elif "uniform_median" in compressed:  # fallback for uniform
        variant_data = compressed["uniform_median"]
    
    if not variant_data:
        print(f"Warning: No {variant_name} data found in {result.get('config_path', 'unknown')}")
        return None
    
    return {
        "seed": result.get("config_path", "unknown"),
        "probe_accuracy": probe.get("accuracy"),
        "probe_params": probe.get("params"),
        "compressed_accuracy": variant_data.get("accuracy"), 
        "compressed_params": variant_data.get("params"),
        "param_reduction": variant_data.get("param_reduction"),
        "delta_vs_probe": variant_data.get("delta_vs_probe"),
        "verdict": variant_data.get("verdict"),
    }

def calculate_stats(values):
    """Calculate mean ± std."""
    if not values:
        return "N/A", "N/A"
    arr = np.array(values)
    return arr.mean(), arr.std()

def main():
    parser = argparse.ArgumentParser(description="Compare compression approaches")
    parser.add_argument("--per-layer", nargs="+", help="Per-layer benchmark directories")
    parser.add_argument("--uniform-r8", nargs="+", help="Uniform r=8 benchmark directories") 
    parser.add_argument("--uniform-r16", nargs="+", help="Uniform r=16 benchmark directories")
    args = parser.parse_args()
    
    approaches = {}
    
    # Load per-layer results
    if args.per_layer:
        per_layer_metrics = []
        for bench_dir in args.per_layer:
            result = load_bench_result(bench_dir)
            metrics = extract_compression_metrics(result, "per_layer")
            if metrics:
                per_layer_metrics.append(metrics)
        approaches["Per-Layer Adaptive"] = per_layer_metrics
    
    # Load uniform r=8 results  
    if args.uniform_r8:
        uniform_metrics = []
        for bench_dir in args.uniform_r8:
            result = load_bench_result(bench_dir)
            metrics = extract_compression_metrics(result, "uniform_median")
            if metrics:
                uniform_metrics.append(metrics)
        approaches["Uniform r=8"] = uniform_metrics
    
    # Load uniform r=16 results
    if args.uniform_r16:
        uniform_r16_metrics = []
        for bench_dir in args.uniform_r16:
            result = load_bench_result(bench_dir)
            metrics = extract_compression_metrics(result, "uniform_median")
            if metrics:
                uniform_r16_metrics.append(metrics)
        approaches["Uniform r=16 (Safe)"] = uniform_r16_metrics
    
    print("=== COMPRESSION APPROACH COMPARISON ===")
    print()
    
    # Summary table
    print(f"{'Approach':<20} {'Accuracy':<12} {'Δ vs Probe':<12} {'Reduction %':<12} {'Success Rate':<12}")
    print("-" * 80)
    
    for approach_name, metrics_list in approaches.items():
        if not metrics_list:
            continue
            
        accuracies = [m["compressed_accuracy"] for m in metrics_list if m["compressed_accuracy"] is not None]
        deltas = [m["delta_vs_probe"] for m in metrics_list if m["delta_vs_probe"] is not None]
        reductions = [m["param_reduction"]*100 for m in metrics_list if m["param_reduction"] is not None]
        verdicts = [m["verdict"] for m in metrics_list if m["verdict"] is not None]
        
        acc_mean, acc_std = calculate_stats(accuracies)
        delta_mean, delta_std = calculate_stats(deltas)
        red_mean, red_std = calculate_stats(reductions)
        
        passing = sum(1 for v in verdicts if v == "PASS")
        success_rate = f"{passing}/{len(verdicts)}"
        
        acc_str = f"{acc_mean:.3f}±{acc_std:.3f}" if acc_mean != "N/A" else "N/A"
        delta_str = f"{delta_mean:+.3f}±{delta_std:.3f}" if delta_mean != "N/A" else "N/A"
        red_str = f"{red_mean:.1f}±{red_std:.1f}%" if red_mean != "N/A" else "N/A"
        
        print(f"{approach_name:<20} {acc_str:<12} {delta_str:<12} {red_str:<12} {success_rate:<12}")
    
    print()
    print("=== PRACTICAL INSIGHTS ===")
    
    if len(approaches) >= 2:
        approach_names = list(approaches.keys())
        per_layer_data = approaches.get("Per-Layer Adaptive", [])
        uniform_r8_data = approaches.get("Uniform r=8", [])
        uniform_r16_data = approaches.get("Uniform r=16 (Safe)", [])
        
        # Compare per-layer vs uniform r=16 (the key comparison)
        if per_layer_data and uniform_r16_data:
            pl_reductions = [m["param_reduction"]*100 for m in per_layer_data if m["param_reduction"]]
            r16_reductions = [m["param_reduction"]*100 for m in uniform_r16_data if m["param_reduction"]]
            
            pl_deltas = [m["delta_vs_probe"] for m in per_layer_data if m["delta_vs_probe"]]
            r16_deltas = [m["delta_vs_probe"] for m in uniform_r16_data if m["delta_vs_probe"]]
            
            if pl_reductions and r16_reductions:
                pl_red_mean = np.mean(pl_reductions)
                r16_red_mean = np.mean(r16_reductions)
                red_advantage = pl_red_mean - r16_red_mean
                
                pl_delta_mean = np.mean(pl_deltas) if pl_deltas else 0
                r16_delta_mean = np.mean(r16_deltas) if r16_deltas else 0
                acc_advantage = r16_delta_mean - pl_delta_mean  # positive = uniform r=16 better
                
                print(f"• **Key Question**: How much does per-layer buy beyond safe uniform r=16?")
                print(f"• Per-layer achieves {red_advantage:.1f}% more compression than uniform r=16")
                
                if acc_advantage > 0.005:  # 0.5%+ better
                    print(f"• Uniform r=16 maintains {acc_advantage:+.3f} better accuracy vs per-layer")
                    print("• **Recommendation**: Start with uniform r=16 for safety")
                elif red_advantage > 5:  # 5%+ more compression
                    print(f"• Per-layer's complexity justified by {red_advantage:.1f}% extra compression")
                    print("• **Recommendation**: Use per-layer when maximizing compression matters")
                else:
                    print("• Compression difference is marginal between approaches")
                    print("• **Recommendation**: Choose based on implementation complexity preference")
        
        # Also compare against uniform r=8 if available
        elif per_layer_data and uniform_r8_data:
            pl_reductions = [m["param_reduction"]*100 for m in per_layer_data if m["param_reduction"]]
            r8_reductions = [m["param_reduction"]*100 for m in uniform_r8_data if m["param_reduction"]]
            
            pl_deltas = [m["delta_vs_probe"] for m in per_layer_data if m["delta_vs_probe"]]
            r8_deltas = [m["delta_vs_probe"] for m in uniform_r8_data if m["delta_vs_probe"]]
            
            if pl_reductions and r8_reductions:
                pl_red_mean = np.mean(pl_reductions)
                r8_red_mean = np.mean(r8_reductions)
                red_advantage = pl_red_mean - r8_red_mean
                
                pl_delta_mean = np.mean(pl_deltas) if pl_deltas else 0
                r8_delta_mean = np.mean(r8_deltas) if r8_deltas else 0
                acc_advantage = r8_delta_mean - pl_delta_mean  # positive = uniform better
                
                print(f"• Per-layer achieves {red_advantage:.1f}% more parameter reduction than uniform r=8")
                
                if acc_advantage > 0.005:  # 0.5%+ better
                    print(f"• Uniform r=8 maintains {acc_advantage:+.3f} better accuracy vs per-layer")
                    print("• **Recommendation**: Start with uniform r=8 for simplicity")
                elif red_advantage > 3:  # 3%+ more compression
                    print(f"• Per-layer's complexity justified by {red_advantage:.1f}% extra compression")
                    print("• **Recommendation**: Use per-layer when maximizing compression matters")
                else:
                    print("• Performance difference is marginal")
                    print("• **Recommendation**: Choose based on implementation complexity preference")
    
    print()
    print("=== DETAILED BREAKDOWN ===")
    for approach_name, metrics_list in approaches.items():
        print(f"\n{approach_name}:")
        print(f"{'Seed':<10} {'Probe Acc':<10} {'Compressed':<10} {'Δ vs Probe':<12} {'Reduction %':<12} {'Verdict'}")
        print("-" * 70)
        
        for m in metrics_list:
            seed_name = Path(m["seed"]).stem if m["seed"] != "unknown" else "unknown"
            probe_acc = f"{m['probe_accuracy']:.4f}" if m["probe_accuracy"] else "N/A"
            comp_acc = f"{m['compressed_accuracy']:.4f}" if m["compressed_accuracy"] else "N/A"
            delta = f"{m['delta_vs_probe']:+.4f}" if m["delta_vs_probe"] else "N/A"
            reduction = f"{m['param_reduction']*100:.1f}" if m["param_reduction"] else "N/A"
            verdict = m["verdict"] or "N/A"
            
            print(f"{seed_name:<10} {probe_acc:<10} {comp_acc:<10} {delta:<12} {reduction:<12} {verdict}")

if __name__ == "__main__":
    main()