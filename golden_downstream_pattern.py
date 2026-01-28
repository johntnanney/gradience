#!/usr/bin/env python3
"""
Golden Downstream Script Pattern

Demonstrates the recommended approach for consuming Gradience audit JSON
with intelligent fallback strategies and robust error handling.

Usage:
    python golden_downstream_pattern.py audit_results.json [--top-n 3]
"""

import json
import argparse
import sys
from typing import List, Dict, Any, Optional

def load_audit_data(audit_path: str) -> Optional[Dict[str, Any]]:
    """Load and validate audit JSON data."""
    try:
        with open(audit_path, 'r') as f:
            audit = json.load(f)
        
        # Basic validation
        if "policy_disagreement_analysis" not in audit:
            print(f"⚠️  Warning: No policy_disagreement_analysis found in {audit_path}")
            return None
            
        return audit
    except FileNotFoundError:
        print(f"❌ Error: Audit file not found: {audit_path}")
        return None
    except json.JSONDecodeError as e:
        print(f"❌ Error: Invalid JSON in {audit_path}: {e}")
        return None
    except Exception as e:
        print(f"❌ Error loading audit data: {e}")
        return None

def get_priority_layers(pd_analysis: Dict[str, Any], top_n: int = 3) -> List[Dict[str, Any]]:
    """
    Get layers for validation with intelligent fallback strategy.
    
    Strategy:
    1. Use flagged layers if available (high-impact layers)
    2. Fall back to top-N layers by priority_score if no flagged layers
    3. Fall back to layers with disagreement if no priority scores
    """
    
    # Strategy 1: Use flagged layers (preferred)
    flagged_layers = pd_analysis.get("flagged_layers", [])
    if flagged_layers:
        print(f"🎯 Using {len(flagged_layers)} flagged high-impact layers")
        return flagged_layers
    
    # Strategy 2: Fall back to top-N by priority_score
    all_layers = pd_analysis.get("all_layers_with_disagreement", [])
    if all_layers:
        # Sort by priority_score (higher = more important)
        layers_with_scores = []
        for layer in all_layers:
            rationale = layer.get("flagging_rationale", {})
            priority_score = rationale.get("priority_score", 0)
            if priority_score > 0:  # Only include layers with valid priority scores
                layers_with_scores.append(layer)
        
        if layers_with_scores:
            sorted_layers = sorted(
                layers_with_scores, 
                key=lambda x: x["flagging_rationale"]["priority_score"], 
                reverse=True
            )
            focus_layers = sorted_layers[:top_n]
            print(f"📊 No flagged layers found. Using top-{len(focus_layers)} layers by priority_score")
            return focus_layers
    
    # Strategy 3: Ultimate fallback - layers with disagreement (no priority filtering)
    if all_layers:
        focus_layers = all_layers[:top_n]
        print(f"🔄 Using top-{len(focus_layers)} layers with disagreement (fallback mode)")
        return focus_layers
    
    # No layers found
    print("⚠️  No layers with disagreement found in audit")
    return []

def analyze_layer_validation_priority(layer: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze a layer and provide validation recommendations."""
    
    layer_name = layer.get("layer_name", "unknown")
    rationale = layer.get("flagging_rationale", {})
    
    # Extract key metrics
    spread = rationale.get("spread", 0)
    spread_threshold = rationale.get("spread_threshold")
    uniform_mult = rationale.get("uniform_mult", 0)
    uniform_mult_threshold = rationale.get("uniform_mult_threshold")
    importance_share = rationale.get("importance_share", 0)
    priority_score = rationale.get("priority_score", 0)
    flagged = rationale.get("flagged_as_high_impact", False)
    failed_reasons = rationale.get("failed_reasons", [])
    
    # Determine validation recommendation
    if flagged:
        recommendation = "HIGH PRIORITY"
        reason = "Flagged as high-impact by all criteria"
        validation_type = "per_layer"
    elif priority_score >= 2.0:
        recommendation = "MEDIUM PRIORITY" 
        reason = f"Strong priority score ({priority_score:.2f})"
        validation_type = "per_layer"
    elif priority_score >= 1.0:
        recommendation = "LOW PRIORITY"
        reason = f"Moderate priority score ({priority_score:.2f})"
        validation_type = "focused_uniform"
    else:
        recommendation = "SKIP"
        reason = f"Low priority score ({priority_score:.2f})"
        validation_type = "uniform_only"
    
    return {
        "layer_name": layer_name,
        "recommendation": recommendation,
        "reason": reason,
        "validation_type": validation_type,
        "metrics": {
            "spread": spread,
            "spread_threshold": spread_threshold,
            "uniform_mult": uniform_mult,
            "uniform_mult_threshold": uniform_mult_threshold,
            "importance_share": importance_share,
            "priority_score": priority_score,
            "flagged": flagged,
            "failed_reasons": failed_reasons
        }
    }

def format_threshold_status(value: float, threshold: Optional[float], name: str) -> str:
    """Format threshold comparison with pass/fail status."""
    if threshold is None:
        return f"{name}={value:.3f} (no threshold)"
    
    if value >= threshold:
        return f"{name}={value:.3f} (≥{threshold:.3f}) ✅"
    else:
        return f"{name}={value:.3f} (<{threshold:.3f}) ❌"

def print_validation_plan(layers: List[Dict[str, Any]], audit_data: Dict[str, Any]):
    """Print comprehensive validation plan for downstream consumption."""
    
    print("\n" + "="*80)
    print("🎯 VALIDATION PLAN")
    print("="*80)
    
    # Show focus strategy if available
    pd_analysis = audit_data.get("policy_disagreement_analysis", {})
    focus_set = pd_analysis.get("disagreement_focus_set", {})
    if focus_set:
        strategy = focus_set.get("focus_strategy", "unknown")
        message = focus_set.get("message", "No message")
        print(f"📋 Focus Strategy: {strategy}")
        print(f"💡 Recommendation: {message}")
        print()
    
    if not layers:
        print("⚠️  No layers recommended for validation")
        return
    
    high_priority = []
    medium_priority = []
    low_priority = []
    
    print(f"📊 LAYER ANALYSIS ({len(layers)} layers):")
    print("-" * 80)
    
    for i, layer in enumerate(layers, 1):
        analysis = analyze_layer_validation_priority(layer)
        metrics = analysis["metrics"]
        
        # Categorize by recommendation
        if analysis["recommendation"] == "HIGH PRIORITY":
            high_priority.append(analysis)
        elif analysis["recommendation"] == "MEDIUM PRIORITY":
            medium_priority.append(analysis)
        elif analysis["recommendation"] == "LOW PRIORITY":
            low_priority.append(analysis)
        
        # Format output
        print(f"{i}. {analysis['layer_name']}")
        print(f"   Status: {analysis['recommendation']} ({analysis['reason']})")
        
        # Threshold analysis
        spread_status = format_threshold_status(
            metrics["spread"], metrics["spread_threshold"], "spread"
        )
        uniform_mult_status = format_threshold_status(
            metrics["uniform_mult"], metrics["uniform_mult_threshold"], "uniform_mult"
        )
        
        print(f"   Metrics: {spread_status}, {uniform_mult_status}")
        print(f"   Share: {metrics['importance_share']:.3f}, Priority: {metrics['priority_score']:.2f}")
        
        if metrics["failed_reasons"]:
            print(f"   Failures: {', '.join(metrics['failed_reasons'])}")
        
        print()
    
    # Summary and recommendations
    print("🚀 VALIDATION RECOMMENDATIONS:")
    print("-" * 50)
    
    if high_priority:
        print(f"🔥 HIGH PRIORITY ({len(high_priority)} layers):")
        for analysis in high_priority:
            print(f"   • {analysis['layer_name']}: Use per-layer optimization")
    
    if medium_priority:
        print(f"⚡ MEDIUM PRIORITY ({len(medium_priority)} layers):")
        for analysis in medium_priority:
            print(f"   • {analysis['layer_name']}: Test per-layer if resources allow")
    
    if low_priority:
        print(f"📊 LOW PRIORITY ({len(low_priority)} layers):")
        for analysis in low_priority:
            print(f"   • {analysis['layer_name']}: Consider uniform rank instead")
    
    print(f"\n💡 Recommended focus: Validate {len(high_priority)} high-priority layers first")
    if high_priority:
        print(f"   Start with: {high_priority[0]['layer_name']}")

def generate_bench_config(layers: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate Bench configuration from validation plan."""
    
    per_layer_pattern = {}
    uniform_fallback_rank = 8  # Conservative default
    
    for layer in layers:
        analysis = analyze_layer_validation_priority(layer)
        layer_name = analysis["layer_name"]
        
        if analysis["validation_type"] == "per_layer":
            # Use highest suggested rank for conservative approach
            rationale = layer.get("flagging_rationale", {})
            k_values = rationale.get("k_values", [])
            if k_values:
                suggested_rank = max(k_values)  # Conservative choice
                per_layer_pattern[layer_name] = suggested_rank
    
    config = {
        "compression_strategies": []
    }
    
    # Add uniform strategy
    config["compression_strategies"].append({
        "variant": "uniform",
        "rank": uniform_fallback_rank,
        "description": "Conservative uniform baseline"
    })
    
    # Add per-layer strategy if we have layers to optimize
    if per_layer_pattern:
        config["compression_strategies"].append({
            "variant": "per_layer",
            "rank_pattern": per_layer_pattern,
            "description": f"Per-layer optimization for {len(per_layer_pattern)} critical layers"
        })
    
    return config

def main():
    """Main execution function implementing the golden pattern."""
    
    parser = argparse.ArgumentParser(description="Golden downstream script pattern for Gradience audit consumption")
    parser.add_argument("audit_file", help="Path to audit JSON file")
    parser.add_argument("--top-n", type=int, default=3, help="Fallback to top-N layers if no flagged layers (default: 3)")
    parser.add_argument("--generate-config", action="store_true", help="Generate Bench configuration JSON")
    
    args = parser.parse_args()
    
    print("🏆 GOLDEN DOWNSTREAM PATTERN")
    print("="*50)
    print(f"Audit file: {args.audit_file}")
    print(f"Fallback top-N: {args.top_n}")
    print()
    
    # Load audit data
    audit_data = load_audit_data(args.audit_file)
    if not audit_data:
        sys.exit(1)
    
    # Extract policy disagreement analysis
    pd_analysis = audit_data.get("policy_disagreement_analysis", {})
    if not pd_analysis.get("analysis_performed", False):
        print("⚠️  Warning: Policy disagreement analysis not performed")
        sys.exit(1)
    
    # Get priority layers using intelligent fallback
    priority_layers = get_priority_layers(pd_analysis, args.top_n)
    
    # Print validation plan
    print_validation_plan(priority_layers, audit_data)
    
    # Generate Bench configuration if requested
    if args.generate_config:
        print("\n" + "="*80)
        print("⚙️  BENCH CONFIGURATION")
        print("="*80)
        
        config = generate_bench_config(priority_layers)
        print(json.dumps(config, indent=2))
    
    print("\n✅ Golden pattern execution complete!")
    print("\n💡 Key Benefits:")
    print("  • Intelligent fallback strategy (flagged → top-N → disagreement)")
    print("  • Clear validation priorities and recommendations")
    print("  • Robust error handling and informative output")
    print("  • Direct Bench configuration generation")
    print("  • Works with any Gradience audit JSON format")

if __name__ == "__main__":
    main()