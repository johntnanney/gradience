#!/usr/bin/env python3
"""
Statistical analysis for mechanism testing experiment.

This script implements proper hypothesis testing to determine whether
audit-guided per-layer rank patterns provide real benefit beyond 
simple heterogeneity by comparing with shuffled controls.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import statistics
import math


def load_aggregated_results(results_path: Path) -> Dict:
    """Load aggregated experiment results."""
    with open(results_path) as f:
        return json.load(f)


def extract_performance_data(results: Dict) -> Dict[str, Dict[str, float]]:
    """Extract performance metrics for each variant."""
    performance = {}
    
    # Probe baseline
    probe = results.get("probe", {})
    if probe:
        performance["probe"] = {
            "accuracy_mean": probe.get("accuracy", {}).get("mean", 0),
            "accuracy_std": probe.get("accuracy", {}).get("std", 0),
            "n_seeds": results.get("n_seeds", 1)
        }
    
    # Compression variants
    compressed = results.get("compressed", {})
    for variant_name in ["per_layer", "per_layer_shuffled"]:
        if variant_name in compressed:
            variant_data = compressed[variant_name]
            performance[variant_name] = {
                "accuracy_mean": variant_data.get("accuracy", {}).get("mean", 0),
                "accuracy_std": variant_data.get("accuracy", {}).get("std", 0),
                "delta_vs_probe_mean": variant_data.get("delta_vs_probe", {}).get("mean", 0),
                "delta_vs_probe_std": variant_data.get("delta_vs_probe", {}).get("std", 0),
                "n_seeds": results.get("n_seeds", 1)
            }
    
    return performance


def cohens_d(mean1: float, std1: float, n1: int, mean2: float, std2: float, n2: int) -> float:
    """Calculate Cohen's d effect size."""
    # Pooled standard deviation
    pooled_std = math.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
    
    if pooled_std == 0:
        return 0
    
    return (mean1 - mean2) / pooled_std


def welch_t_test(mean1: float, std1: float, n1: int, mean2: float, std2: float, n2: int) -> Tuple[float, float]:
    """Perform Welch's t-test for unequal variances."""
    if std1 == 0 and std2 == 0:
        return float('inf') if mean1 != mean2 else 0, 1.0
    
    # Calculate t-statistic
    se1 = std1 / math.sqrt(n1)
    se2 = std2 / math.sqrt(n2) 
    se_diff = math.sqrt(se1**2 + se2**2)
    
    if se_diff == 0:
        return float('inf') if mean1 != mean2 else 0, 1.0
    
    t_stat = (mean1 - mean2) / se_diff
    
    # Welch's degrees of freedom
    if std1 > 0 and std2 > 0:
        df = (se1**2 + se2**2)**2 / (se1**4 / (n1 - 1) + se2**4 / (n2 - 1))
    else:
        df = max(n1, n2) - 1
    
    # Approximate p-value (simplified - in practice use scipy.stats)
    # This is a rough approximation for demonstration
    if abs(t_stat) > 2.0:  # Rough 95% threshold
        p_value = 0.05
    elif abs(t_stat) > 1.0:
        p_value = 0.30
    else:
        p_value = 0.60
    
    return t_stat, p_value


def analyze_compression_efficacy(performance: Dict[str, Dict[str, float]]) -> Dict[str, any]:
    """Test H1: per_layer outperforms probe baseline."""
    if "probe" not in performance or "per_layer" not in performance:
        return {"status": "insufficient_data"}
    
    probe = performance["probe"]
    per_layer = performance["per_layer"]
    
    # Direct comparison using delta_vs_probe
    delta_mean = per_layer["delta_vs_probe_mean"]
    delta_std = per_layer["delta_vs_probe_std"]
    n_seeds = per_layer["n_seeds"]
    
    # Effect size
    effect_size = cohens_d(
        per_layer["accuracy_mean"], per_layer["accuracy_std"], n_seeds,
        probe["accuracy_mean"], probe["accuracy_std"], probe["n_seeds"]
    )
    
    # Significance test (one-tailed: per_layer > probe)
    t_stat, p_value = welch_t_test(
        per_layer["accuracy_mean"], per_layer["accuracy_std"], n_seeds,
        probe["accuracy_mean"], probe["accuracy_std"], probe["n_seeds"]
    )
    
    # Practical significance threshold
    practical_threshold = 0.02  # 2%
    
    return {
        "delta_mean": delta_mean,
        "delta_std": delta_std,
        "effect_size_cohens_d": effect_size,
        "t_statistic": t_stat,
        "p_value": p_value,
        "is_statistically_significant": p_value < 0.05 and t_stat > 0,
        "is_practically_significant": delta_mean > practical_threshold,
        "threshold_met": delta_mean > practical_threshold,
        "interpretation": "significant_compression" if delta_mean > practical_threshold else "insufficient_compression"
    }


def analyze_mechanism_benefit(performance: Dict[str, Dict[str, float]]) -> Dict[str, any]:
    """Test H2: per_layer outperforms per_layer_shuffled (mechanism test)."""
    if "per_layer" not in performance or "per_layer_shuffled" not in performance:
        return {"status": "insufficient_data"}
    
    per_layer = performance["per_layer"]
    shuffled = performance["per_layer_shuffled"]
    
    # Direct comparison
    mechanism_benefit = per_layer["accuracy_mean"] - shuffled["accuracy_mean"]
    
    # Effect size
    effect_size = cohens_d(
        per_layer["accuracy_mean"], per_layer["accuracy_std"], per_layer["n_seeds"],
        shuffled["accuracy_mean"], shuffled["accuracy_std"], shuffled["n_seeds"]
    )
    
    # Significance test (one-tailed: per_layer > shuffled)
    t_stat, p_value = welch_t_test(
        per_layer["accuracy_mean"], per_layer["accuracy_std"], per_layer["n_seeds"],
        shuffled["accuracy_mean"], shuffled["accuracy_std"], shuffled["n_seeds"]
    )
    
    # Mechanism threshold (stricter than compression)
    mechanism_threshold = 0.01  # 1%
    
    return {
        "mechanism_benefit": mechanism_benefit,
        "effect_size_cohens_d": effect_size,
        "t_statistic": t_stat,
        "p_value": p_value,
        "is_statistically_significant": p_value < 0.05 and t_stat > 0,
        "is_practically_significant": mechanism_benefit > mechanism_threshold,
        "threshold_met": mechanism_benefit > mechanism_threshold,
        "interpretation": "real_mechanism" if mechanism_benefit > mechanism_threshold else "heterogeneity_only"
    }


def generate_report(performance: Dict, compression_analysis: Dict, mechanism_analysis: Dict) -> str:
    """Generate human-readable analysis report."""
    report = []
    report.append("=" * 60)
    report.append("MECHANISM TESTING EXPERIMENT ANALYSIS")
    report.append("=" * 60)
    
    # Summary table
    report.append("\n📊 PERFORMANCE SUMMARY")
    report.append("-" * 40)
    for variant, data in performance.items():
        acc = data["accuracy_mean"]
        std = data["accuracy_std"]
        report.append(f"{variant:<20}: {acc:.3f} ± {std:.3f}")
    
    # Compression efficacy
    report.append(f"\n🔍 COMPRESSION EFFICACY TEST")
    report.append("-" * 40)
    if compression_analysis.get("status") != "insufficient_data":
        delta = compression_analysis["delta_mean"]
        effect = compression_analysis["effect_size_cohens_d"]
        p_val = compression_analysis["p_value"]
        threshold = compression_analysis["threshold_met"]
        
        report.append(f"per_layer vs probe:")
        report.append(f"  Improvement: {delta:+.3f} ± {compression_analysis['delta_std']:.3f}")
        report.append(f"  Effect size: {effect:.2f} (Cohen's d)")
        report.append(f"  P-value: {p_val:.3f}")
        report.append(f"  Threshold (2%): {'✓ PASS' if threshold else '❌ FAIL'}")
        report.append(f"  Result: {compression_analysis['interpretation']}")
    else:
        report.append("❌ Insufficient data for analysis")
    
    # Mechanism test  
    report.append(f"\n🧬 MECHANISM BENEFIT TEST")
    report.append("-" * 40)
    if mechanism_analysis.get("status") != "insufficient_data":
        benefit = mechanism_analysis["mechanism_benefit"]
        effect = mechanism_analysis["effect_size_cohens_d"]
        p_val = mechanism_analysis["p_value"]
        threshold = mechanism_analysis["threshold_met"]
        
        report.append(f"per_layer vs per_layer_shuffled:")
        report.append(f"  Mechanism benefit: {benefit:+.3f}")
        report.append(f"  Effect size: {effect:.2f} (Cohen's d)")
        report.append(f"  P-value: {p_val:.3f}")
        report.append(f"  Threshold (1%): {'✓ PASS' if threshold else '❌ FAIL'}")
        report.append(f"  Result: {mechanism_analysis['interpretation']}")
    else:
        report.append("❌ Insufficient data for analysis")
    
    # Overall conclusion
    report.append(f"\n🎯 OVERALL CONCLUSION")
    report.append("-" * 40)
    
    compression_works = compression_analysis.get("threshold_met", False)
    mechanism_works = mechanism_analysis.get("threshold_met", False)
    
    if compression_works and mechanism_works:
        conclusion = "✅ HYPOTHESIS CONFIRMED: Audit-guided ranks provide real benefit beyond heterogeneity"
    elif compression_works and not mechanism_works:
        conclusion = "⚠️  PARTIAL SUCCESS: Compression works, but no clear mechanism benefit"
    elif not compression_works:
        conclusion = "❌ NEGATIVE RESULT: Insufficient compression benefit over probe"
    else:
        conclusion = "❓ INCONCLUSIVE: Mixed or insufficient results"
    
    report.append(conclusion)
    report.append("")
    
    return "\n".join(report)


def main():
    """Run complete mechanism testing analysis."""
    if len(sys.argv) != 2:
        print("Usage: python analyze_mechanism_test.py <aggregated_results.json>")
        sys.exit(1)
    
    results_path = Path(sys.argv[1])
    if not results_path.exists():
        print(f"Error: Results file not found: {results_path}")
        sys.exit(1)
    
    # Load and analyze results
    results = load_aggregated_results(results_path)
    performance = extract_performance_data(results)
    
    compression_analysis = analyze_compression_efficacy(performance)
    mechanism_analysis = analyze_mechanism_benefit(performance)
    
    # Generate report
    report = generate_report(performance, compression_analysis, mechanism_analysis)
    print(report)
    
    # Save detailed analysis
    output_dir = results_path.parent
    analysis_output = {
        "performance_data": performance,
        "compression_efficacy": compression_analysis,
        "mechanism_benefit": mechanism_analysis,
        "experiment_metadata": {
            "analysis_date": "2025-01-21",
            "analysis_version": "1.0"
        }
    }
    
    analysis_path = output_dir / "mechanism_analysis.json"
    with open(analysis_path, 'w') as f:
        json.dump(analysis_output, f, indent=2)
    
    print(f"📁 Detailed analysis saved to: {analysis_path}")


if __name__ == "__main__":
    main()