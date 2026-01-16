#!/usr/bin/env python3
"""
Aggregate multi-seed benchmark results for defensible statistics.

Usage:
    python scripts/aggregate_multiseed.py bench_runs/multiseed_seed*
"""
import json
import sys
from pathlib import Path
import numpy as np

def load_bench_result(bench_dir, skip_smoke=True):
    """Load results from a bench.json file."""
    bench_path = Path(bench_dir) / "bench.json"
    if not bench_path.exists():
        print(f"Warning: No bench.json in {bench_dir}")
        return None
    
    with open(bench_path) as f:
        result = json.load(f)
    
    # Check if this is a smoke run
    if skip_smoke and result.get("status") == "UNDERTRAINED_SMOKE":
        print(f"Info: Skipping smoke run {Path(bench_dir).name}")
        return None
        
    return result

def extract_key_metrics(result):
    """Extract key metrics for aggregation."""
    if not result:
        return None
        
    probe = result.get("probe", {})
    compressed = result.get("compressed", {})
    per_layer = compressed.get("per_layer", {})
    
    return {
        "seed": result.get("config_path", "unknown"),
        "probe_accuracy": probe.get("accuracy"),
        "probe_params": probe.get("params"),
        "per_layer_accuracy": per_layer.get("accuracy"), 
        "per_layer_params": per_layer.get("params"),
        "per_layer_reduction": per_layer.get("param_reduction"),
        "per_layer_delta": per_layer.get("delta_vs_probe"),
        "per_layer_verdict": per_layer.get("verdict"),
        "probe_quality": result.get("summary", {}).get("probe_quality")
    }

def main():
    if len(sys.argv) < 2:
        print("Usage: python aggregate_multiseed.py bench_runs/multiseed_seed*")
        sys.exit(1)
    
    bench_dirs = sys.argv[1:]
    
    print("=== MULTI-SEED AGGREGATION ===")
    print(f"Processing {len(bench_dirs)} benchmark results...")
    print()
    
    # Load all results
    all_metrics = []
    for bench_dir in bench_dirs:
        result = load_bench_result(bench_dir)
        metrics = extract_key_metrics(result)
        if metrics:
            all_metrics.append(metrics)
            print(f"✅ Loaded: {Path(bench_dir).name}")
        else:
            print(f"❌ Failed: {Path(bench_dir).name}")
    
    if not all_metrics:
        print("No valid results found!")
        sys.exit(1)
        
    print(f"\nSuccessfully loaded {len(all_metrics)} results")
    print()
    
    # Aggregate probe statistics
    probe_accuracies = [m["probe_accuracy"] for m in all_metrics if m["probe_accuracy"] is not None]
    probe_params = [m["probe_params"] for m in all_metrics if m["probe_params"] is not None]
    
    # Aggregate per_layer statistics
    per_layer_accuracies = [m["per_layer_accuracy"] for m in all_metrics if m["per_layer_accuracy"] is not None]
    per_layer_reductions = [m["per_layer_reduction"] for m in all_metrics if m["per_layer_reduction"] is not None]
    per_layer_deltas = [m["per_layer_delta"] for m in all_metrics if m["per_layer_delta"] is not None]
    
    def stats(values):
        """Calculate mean ± std."""
        if not values:
            return "N/A"
        arr = np.array(values)
        return f"{arr.mean():.4f} ± {arr.std():.4f}"
    
    # Print results
    print("=== PROBE BASELINE (r=32) ===")
    print(f"Accuracy: {stats(probe_accuracies)}")
    print(f"Parameters: {probe_params[0]:,} (constant)")
    print()
    
    print("=== PER-LAYER COMPRESSION ===")
    print(f"Accuracy: {stats(per_layer_accuracies)}")
    print(f"Δ vs Probe: {stats(per_layer_deltas)}")
    print(f"Param Reduction: {stats([r*100 for r in per_layer_reductions])}%")
    print()
    
    # Individual results table
    print("=== INDIVIDUAL RESULTS ===")
    print(f"{'Seed':<10} {'Probe Acc':<10} {'Per-Layer Acc':<12} {'Δ vs Probe':<12} {'Reduction %':<12} {'Verdict'}")
    print("-" * 75)
    
    for m in all_metrics:
        seed_name = Path(m["seed"]).stem if m["seed"] != "unknown" else "unknown"
        probe_acc = f"{m['probe_accuracy']:.4f}" if m["probe_accuracy"] else "N/A"
        per_acc = f"{m['per_layer_accuracy']:.4f}" if m["per_layer_accuracy"] else "N/A"
        delta = f"{m['per_layer_delta']:+.4f}" if m["per_layer_delta"] else "N/A"
        reduction = f"{m['per_layer_reduction']*100:.1f}" if m["per_layer_reduction"] else "N/A"
        verdict = m["per_layer_verdict"] or "N/A"
        
        print(f"{seed_name:<10} {probe_acc:<10} {per_acc:<12} {delta:<12} {reduction:<12} {verdict}")
    
    # Summary statistics
    print()
    print("=== DEFENSIBLE CLAIMS ===")
    if len(per_layer_reductions) >= 3:
        mean_reduction = np.mean(per_layer_reductions) * 100
        std_reduction = np.std(per_layer_reductions) * 100
        mean_delta = np.mean(per_layer_deltas) 
        std_delta = np.std(per_layer_deltas)
        
        print(f"• Per-layer compression achieves {mean_reduction:.1f}% ± {std_reduction:.1f}% parameter reduction")
        print(f"• Accuracy impact: {mean_delta:+.4f} ± {std_delta:.4f} vs probe baseline") 
        print(f"• Based on n={len(per_layer_reductions)} independent seeds")
        
        # Check if consistently passing
        verdicts = [m["per_layer_verdict"] for m in all_metrics]
        passing = sum(1 for v in verdicts if v == "PASS")
        print(f"• Success rate: {passing}/{len(verdicts)} seeds pass ±2.5% tolerance")
    else:
        print("Insufficient data for statistical claims (need ≥3 seeds)")

if __name__ == "__main__":
    main()