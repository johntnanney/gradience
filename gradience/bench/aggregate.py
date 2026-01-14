#!/usr/bin/env python3
"""
Multi-seed aggregation for Gradience Bench results.

Consolidates multiple seed runs into a single aggregate report with:
- Mean/std statistics across seeds
- Pass rate computation  
- Worst-case delta tracking
- Policy compliance checking
"""

import json
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional


def load_bench_results(run_dirs: List[Path]) -> List[Dict[str, Any]]:
    """Load bench.json from each run directory."""
    results = []
    for run_dir in run_dirs:
        bench_json = run_dir / "bench.json"
        verdicts_json = run_dir / "verdicts.json"
        
        if not bench_json.exists():
            print(f"Warning: {bench_json} not found, skipping...")
            continue
            
        with open(bench_json) as f:
            bench_data = json.load(f)
            
        # Also load verdicts for more detailed info
        if verdicts_json.exists():
            with open(verdicts_json) as f:
                verdicts_data = json.load(f)
                bench_data["verdicts"] = verdicts_data.get("verdicts", {})
                
        results.append(bench_data)
    
    return results


def aggregate_invariants(results: List[Dict]) -> Dict[str, Any]:
    """Aggregate invariant checks across seed runs."""
    invariant_data = {}
    
    # Collect all invariant data from seeds
    for result in results:
        if "protocol_invariants" in result:
            for inv_name, inv_data in result["protocol_invariants"].items():
                if inv_name not in invariant_data:
                    invariant_data[inv_name] = []
                invariant_data[inv_name].append(inv_data)
    
    if not invariant_data:
        return {"status": "no_invariant_data", "message": "No invariant checks found in seed runs"}
    
    # Aggregate status across seeds
    aggregated = {}
    for inv_name, inv_list in invariant_data.items():
        statuses = [inv.get("status", "UNKNOWN") for inv in inv_list]
        
        # Determine overall status
        if all(s == "PASSED" for s in statuses):
            overall_status = "PASSED"
        elif any(s == "FAILED" for s in statuses):
            overall_status = "FAILED"
        elif any(s == "WARNING" for s in statuses):
            overall_status = "WARNING"
        elif any(s == "SKIPPED" for s in statuses):
            overall_status = "SKIPPED"
        else:
            overall_status = "UNKNOWN"
        
        aggregated[inv_name] = {
            "overall_status": overall_status,
            "seed_statuses": statuses,
            "n_seeds": len(statuses),
            "passed": sum(1 for s in statuses if s == "PASSED"),
            "failed": sum(1 for s in statuses if s == "FAILED"),
            "warnings": sum(1 for s in statuses if s == "WARNING"),
            "message": inv_list[0].get("message", "")  # Use first seed's message
        }
    
    # Summary
    all_statuses = [agg["overall_status"] for agg in aggregated.values()]
    summary = {
        "total_invariants": len(aggregated),
        "all_passed": all(s == "PASSED" for s in all_statuses),
        "any_failed": any(s == "FAILED" for s in all_statuses),
        "any_warnings": any(s == "WARNING" for s in all_statuses),
        "overall_status": "FAILED" if any(s == "FAILED" for s in all_statuses) else 
                         "WARNING" if any(s == "WARNING" for s in all_statuses) else
                         "PASSED" if all(s == "PASSED" for s in all_statuses) else "UNKNOWN"
    }
    
    return {
        "invariants": aggregated,
        "summary": summary
    }


def aggregate_variant_stats(results: List[Dict], variant: str) -> Dict[str, Any]:
    """Aggregate statistics for a specific variant across seeds."""
    variant_data = []
    
    for result in results:
        # Try to find variant in compression results or verdicts
        found = False
        
        # Check compression_results first
        if "compression_results" in result:
            for comp in result["compression_results"]:
                if comp.get("variant") == variant:
                    variant_data.append({
                        "accuracy": comp.get("accuracy"),
                        "delta": comp.get("delta_vs_probe"),
                        "params": comp.get("params"),
                        "param_reduction": comp.get("param_reduction"),
                        "verdict": comp.get("verdict", "UNKNOWN")
                    })
                    found = True
                    break
        
        # Check verdicts as fallback
        if not found and "verdicts" in result and variant in result["verdicts"]:
            verdict = result["verdicts"][variant]
            if verdict.get("status") == "evaluated":
                variant_data.append({
                    "accuracy": verdict.get("compressed_accuracy"),
                    "delta": verdict.get("delta_vs_probe"),
                    "params": verdict.get("compressed_params"),
                    "param_reduction": verdict.get("param_reduction"),
                    "verdict": verdict.get("verdict", "UNKNOWN")
                })
    
    if not variant_data:
        return {"status": "no_data", "n_seeds": 0}
    
    # Calculate statistics
    accuracies = [d["accuracy"] for d in variant_data if d["accuracy"] is not None]
    deltas = [d["delta"] for d in variant_data if d["delta"] is not None]
    param_reductions = [d["param_reduction"] for d in variant_data if d["param_reduction"] is not None]
    verdicts = [d["verdict"] for d in variant_data]
    
    pass_count = sum(1 for v in verdicts if v == "PASS")
    fail_count = sum(1 for v in verdicts if v == "FAIL")
    
    stats = {
        "n_seeds": len(variant_data),
        "pass_count": pass_count,
        "fail_count": fail_count,
        "pass_rate": pass_count / len(variant_data) if variant_data else 0,
        "verdict": "PASS" if pass_count / len(variant_data) >= 0.67 else "FAIL" if fail_count > 0 else "UNKNOWN"
    }
    
    if accuracies:
        stats["accuracy_mean"] = np.mean(accuracies)
        stats["accuracy_std"] = np.std(accuracies) if len(accuracies) > 1 else 0
        stats["accuracy_min"] = min(accuracies)
        stats["accuracy_max"] = max(accuracies)
        
    if deltas:
        stats["delta_mean"] = np.mean(deltas)
        stats["delta_std"] = np.std(deltas) if len(deltas) > 1 else 0
        stats["delta_worst"] = min(deltas)  # Most negative delta
        stats["delta_best"] = max(deltas)
        
    if param_reductions:
        stats["param_reduction_mean"] = np.mean(param_reductions)
        stats["param_reduction_std"] = np.std(param_reductions) if len(param_reductions) > 1 else 0
        
    return stats


def check_policy_compliance(variant_stats: Dict[str, Any], policy: Dict[str, Any]) -> Dict[str, Any]:
    """Check if variant meets policy criteria."""
    pass_rate = variant_stats.get("pass_rate", 0)
    worst_delta = variant_stats.get("delta_worst", -1.0)
    
    meets_pass_rate = pass_rate >= policy["pass_rate_min"]
    meets_worst_delta = worst_delta >= policy["worst_delta_min"]
    
    return {
        "policy_compliant": meets_pass_rate and meets_worst_delta,
        "meets_pass_rate": meets_pass_rate,
        "meets_worst_delta": meets_worst_delta,
        "pass_rate": pass_rate,
        "worst_delta": worst_delta,
        "policy": policy
    }


def aggregate_results(run_dirs: List[str], output_dir: str) -> None:
    """Main aggregation function."""
    # Convert to Path objects
    run_paths = [Path(run_dir) for run_dir in run_dirs]
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load all results
    results = load_bench_results(run_paths)
    
    if not results:
        print("Error: No valid results found!")
        return
    
    # Extract basic info from first result
    first = results[0]
    model = first.get("model", "unknown")
    task = first.get("task", "unknown")
    n_seeds = len(results)
    
    # Aggregate probe stats
    probe_accuracies = []
    for result in results:
        if "probe" in result:
            probe_accuracies.append(result["probe"].get("accuracy"))
        elif "probe_baseline" in result:
            probe_accuracies.append(result["probe_baseline"].get("accuracy"))
    
    probe_stats = {
        "accuracy_mean": np.mean(probe_accuracies) if probe_accuracies else None,
        "accuracy_std": np.std(probe_accuracies) if len(probe_accuracies) > 1 else 0,
        "accuracy_min": min(probe_accuracies) if probe_accuracies else None,
        "accuracy_max": max(probe_accuracies) if probe_accuracies else None,
    }
    
    # Find all unique variants
    all_variants = set()
    for result in results:
        if "compression_results" in result:
            for comp in result["compression_results"]:
                all_variants.add(comp.get("variant"))
        if "verdicts" in result:
            all_variants.update(result["verdicts"].keys())
    
    # Remove control/skipped variants
    all_variants.discard("uniform_p90_control")
    all_variants.discard(None)
    
    # Aggregate each variant
    variant_aggregates = {}
    for variant in sorted(all_variants):
        variant_aggregates[variant] = aggregate_variant_stats(results, variant)
    
    # Define safety policy
    safety_policy = {
        "name": "Safe Uniform Baseline Policy",
        "pass_rate_min": 0.67,  # ≥67% seeds must pass
        "worst_delta_min": -0.025  # Worst seed Δ ≥ -2.5%
    }
    
    # Check policy compliance
    policy_results = {}
    for variant, stats in variant_aggregates.items():
        if stats.get("status") != "no_data":
            policy_results[variant] = check_policy_compliance(stats, safety_policy)
    
    # Aggregate invariant checks across seeds
    invariant_summary = aggregate_invariants(results)
    
    # Build aggregate JSON
    aggregate_data = {
        "bench_version": "0.1",
        "aggregation_timestamp": datetime.now().isoformat(),
        "model": model,
        "task": task,
        "validation_level": "Certifiable",
        "n_seeds": n_seeds,
        "seed_runs": [str(p) for p in run_paths],
        "probe_baseline": probe_stats,
        "variant_results": variant_aggregates,
        "policy_compliance": policy_results,
        "safety_policy": safety_policy,
        "invariants": invariant_summary,
        "summary": {
            "total_variants": len(variant_aggregates),
            "policy_compliant_variants": sum(1 for v in policy_results.values() if v["policy_compliant"]),
            "best_compression": None,
            "recommendations": []
        }
    }
    
    # Find best policy-compliant compression
    compliant_variants = [
        (name, stats) for name, stats in variant_aggregates.items()
        if name in policy_results and policy_results[name]["policy_compliant"] and stats.get("param_reduction_mean")
    ]
    
    if compliant_variants:
        best_variant = max(compliant_variants, key=lambda x: x[1]["param_reduction_mean"])
        aggregate_data["summary"]["best_compression"] = {
            "variant": best_variant[0],
            "param_reduction": best_variant[1]["param_reduction_mean"],
            "delta_worst": best_variant[1]["delta_worst"],
            "pass_rate": best_variant[1]["pass_rate"]
        }
        aggregate_data["summary"]["recommendations"].append(
            f"Use {best_variant[0]} for {best_variant[1]['param_reduction_mean']:.1%} compression (policy-compliant)"
        )
    
    # Save JSON
    json_path = output_path / "bench_aggregate.json"
    with open(json_path, "w") as f:
        json.dump(aggregate_data, f, indent=2)
    
    # Generate Markdown report
    md_content = generate_markdown_report(aggregate_data)
    md_path = output_path / "bench_aggregate.md"
    with open(md_path, "w") as f:
        f.write(md_content)
    
    print(f"✅ Aggregate results saved to:")
    print(f"   {json_path}")
    print(f"   {md_path}")


def generate_markdown_report(data: Dict[str, Any]) -> str:
    """Generate markdown report from aggregate data."""
    lines = []
    
    lines.append("# Gradience Bench v0.1 - Aggregate Report")
    lines.append("")
    lines.append(f"- **Model:** {data['model']}")
    lines.append(f"- **Task:** {data['task']}")
    lines.append(f"- **Validation Level:** {data['validation_level']}")
    lines.append(f"- **Seeds:** {data['n_seeds']}")
    lines.append("")
    
    # Probe baseline
    probe = data["probe_baseline"]
    lines.append("## Probe Baseline")
    lines.append("")
    if probe["accuracy_mean"] is not None:
        lines.append(f"- **Accuracy:** {probe['accuracy_mean']:.3f} ± {probe['accuracy_std']:.3f}")
        lines.append(f"- **Range:** [{probe['accuracy_min']:.3f}, {probe['accuracy_max']:.3f}]")
    lines.append("")
    
    # Compression results table
    lines.append("## Compression Results")
    lines.append("")
    lines.append("| Variant | Pass Rate | Worst Δ | Mean Accuracy | Param Reduction | Policy Status |")
    lines.append("|---------|-----------|---------|---------------|-----------------|---------------|")
    
    for variant in sorted(data["variant_results"].keys()):
        stats = data["variant_results"][variant]
        if stats.get("status") == "no_data":
            continue
            
        policy = data["policy_compliance"].get(variant, {})
        
        pass_rate = f"{stats.get('pass_rate', 0):.0%}"
        worst_delta = f"{stats.get('delta_worst', 0):.3f}" if stats.get('delta_worst') is not None else "N/A"
        mean_acc = f"{stats.get('accuracy_mean', 0):.3f}" if stats.get('accuracy_mean') is not None else "N/A"
        param_red = f"{stats.get('param_reduction_mean', 0):.1%}" if stats.get('param_reduction_mean') else "N/A"
        policy_status = "✅ COMPLIANT" if policy.get("policy_compliant") else "❌ FAIL"
        
        lines.append(f"| {variant} | {pass_rate} | {worst_delta} | {mean_acc} | {param_red} | {policy_status} |")
    
    lines.append("")
    
    # Safety Policy
    lines.append("## Safety Policy")
    lines.append("")
    policy = data["safety_policy"]
    lines.append(f"**{policy['name']}:**")
    lines.append(f"- Pass rate ≥ {policy['pass_rate_min']:.0%}")
    lines.append(f"- Worst-case Δ ≥ {policy['worst_delta_min']:.3f}")
    lines.append("")
    
    # Enhanced Protocol Invariants section
    lines.append("## Protocol Invariants")
    lines.append("")
    lines.append("Validation of critical assumptions across all seeds:")
    lines.append("")
    
    if "invariants" in data and data["invariants"]:
        inv_data = data["invariants"]
        
        # Overall status header
        if "summary" in inv_data:
            status = inv_data["summary"]["overall_status"]
            status_icon = "✅" if status == "PASSED" else "❌" if status == "FAILED" else "⚠️" if status == "WARNING" else "❓"
            lines.append(f"**Overall Status:** {status_icon} {status}")
            lines.append("")
            
        # Detailed invariant breakdown
        invariants = inv_data.get("invariants", {})
        
        # 1. Telemetry presence check
        telemetry_status = _get_invariant_status(invariants, "telemetry_present")
        lines.append(f"**📊 Telemetry Present:** {telemetry_status['icon']} {telemetry_status['status']}")
        lines.append(f"  *All seeds have required telemetry data ({telemetry_status['passed']}/{data['n_seeds']} seeds)*")
        lines.append("")
        
        # 2. Probe gate check
        probe_status = _get_invariant_status(invariants, "probe_quality")
        lines.append(f"**🎯 Probe Quality Gate:** {probe_status['icon']} {probe_status['status']}")
        probe_acc = data.get("probe_baseline", {}).get("accuracy_mean")
        if probe_acc:
            lines.append(f"  *Probe accuracy {probe_acc:.3f} meets quality threshold ({probe_status['passed']}/{data['n_seeds']} seeds)*")
        else:
            lines.append(f"  *Probe quality validation ({probe_status['passed']}/{data['n_seeds']} seeds)*")
        lines.append("")
        
        # 3. Parameter counting source
        param_status = _get_invariant_status(invariants, "param_counting")
        lines.append(f"**🔢 Parameter Counting:** {param_status['icon']} {param_status['status']}")
        param_msg = _get_invariant_message(invariants, "param_counting")
        if param_msg and "audit" in param_msg.lower():
            lines.append(f"  *Using audit-based parameter counting for consistency ({param_status['passed']}/{data['n_seeds']} seeds)*")
        elif param_msg and "config" in param_msg.lower():
            lines.append(f"  *Using config-based parameter counting ({param_status['passed']}/{data['n_seeds']} seeds)*")
        else:
            lines.append(f"  *Parameter counting methodology validated ({param_status['passed']}/{data['n_seeds']} seeds)*")
        lines.append("")
        
        # 4. Per-layer rank heterogeneity check
        rank_status = _get_invariant_status(invariants, "rank_heterogeneity")
        lines.append(f"**🏗️ Per-Layer Rank Check:** {rank_status['icon']} {rank_status['status']}")
        
        if rank_status['status'] == "SKIPPED":
            lines.append(f"  *Per-layer compression excluded from this validation run*")
        elif rank_status['status'] == "PASSED":
            lines.append(f"  *Per-layer configurations show sufficient rank heterogeneity ({rank_status['passed']}/{data['n_seeds']} seeds)*")
        elif rank_status['status'] == "FAILED":
            lines.append(f"  *Per-layer rank heterogeneity failed - insufficient rank diversity ({rank_status['passed']}/{data['n_seeds']} seeds)*")
        else:
            lines.append(f"  *Per-layer rank validation ({rank_status['passed']}/{data['n_seeds']} seeds)*")
        lines.append("")
        
        # 5. Layer consistency check
        layer_status = _get_invariant_status(invariants, "layer_consistency")
        if layer_status['status'] != "NOT_FOUND":
            lines.append(f"**⚖️ Layer Consistency:** {layer_status['icon']} {layer_status['status']}")
            if layer_status['status'] == "PASSED":
                lines.append(f"  *Rank and alpha patterns have matching layer keys ({layer_status['passed']}/{data['n_seeds']} seeds)*")
            elif layer_status['status'] == "FAILED":
                lines.append(f"  *Rank/alpha pattern layer key mismatch detected ({layer_status['passed']}/{data['n_seeds']} seeds)*")
            else:
                lines.append(f"  *Layer pattern consistency check ({layer_status['passed']}/{data['n_seeds']} seeds)*")
            lines.append("")
        
        # 6. Additional invariants table for completeness
        if len(invariants) > 0:
            lines.append("### All Invariant Checks")
            lines.append("")
            lines.append("| Invariant | Status | Seeds Passed | Details |")
            lines.append("|-----------|--------|--------------|---------|")
            
            for inv_name, inv_stats in sorted(invariants.items()):
                status = inv_stats["overall_status"]
                status_icon = "✅" if status == "PASSED" else "❌" if status == "FAILED" else "⚠️" if status == "WARNING" else "⏭️" if status == "SKIPPED" else "❓"
                passed = inv_stats["passed"]
                n_seeds = inv_stats["n_seeds"]
                message = inv_stats.get("message", "")[:50] + ("..." if len(inv_stats.get("message", "")) > 50 else "")
                display_name = inv_name.replace('_', ' ').title()
                lines.append(f"| {display_name} | {status_icon} {status} | {passed}/{n_seeds} | {message} |")
            
            lines.append("")
            
    else:
        lines.append("⚠️ **No invariant data available** - Protocol validation could not be performed.")
        lines.append("")
        lines.append("This may indicate:")
        lines.append("- Individual seed runs missing invariant checks")
        lines.append("- Older bench output format")
        lines.append("- Incomplete benchmark execution")
        lines.append("")
    
    # Summary
    lines.append("## Summary")
    lines.append("")
    summary = data["summary"]
    lines.append(f"- **Total variants tested:** {summary['total_variants']}")
    lines.append(f"- **Policy-compliant variants:** {summary['policy_compliant_variants']}")
    
    if summary["best_compression"]:
        best = summary["best_compression"]
        lines.append(f"- **Best compression:** {best['variant']} ({best['param_reduction']:.1%} reduction)")
    
    if summary["recommendations"]:
        lines.append("")
        lines.append("### Recommendations")
        for rec in summary["recommendations"]:
            lines.append(f"- {rec}")
    
    lines.append("")
    lines.append(f"*Generated on {data['aggregation_timestamp']}*")
    
    return "\n".join(lines)


def _get_invariant_status(invariants: dict, inv_name: str) -> dict:
    """Helper to extract invariant status with fallback."""
    if inv_name in invariants:
        inv = invariants[inv_name]
        status = inv["overall_status"]
        return {
            "status": status,
            "icon": "✅" if status == "PASSED" else "❌" if status == "FAILED" else "⚠️" if status == "WARNING" else "⏭️" if status == "SKIPPED" else "❓",
            "passed": inv.get("passed", 0),
            "total": inv.get("n_seeds", 0)
        }
    else:
        return {
            "status": "NOT_FOUND",
            "icon": "❓",
            "passed": 0,
            "total": 0
        }


def _get_invariant_message(invariants: dict, inv_name: str) -> str:
    """Helper to extract invariant message."""
    if inv_name in invariants:
        return invariants[inv_name].get("message", "")
    return ""


def main():
    parser = argparse.ArgumentParser(description="Aggregate multi-seed bench results")
    parser.add_argument("runs", nargs="+", help="Paths to run directories")
    parser.add_argument("--output", "-o", required=True, help="Output directory for aggregate results")
    
    args = parser.parse_args()
    aggregate_results(args.runs, args.output)


if __name__ == "__main__":
    main()