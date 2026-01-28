#!/usr/bin/env python3
"""
Simple Downstream Pattern

The concise "golden" pattern suggested by the user - handles both
flagged layers and top-N by priority_score fallback.

Usage:
    python simple_downstream_pattern.py audit_results.json
"""

import json
import sys

def main():
    if len(sys.argv) != 2:
        print("Usage: python simple_downstream_pattern.py audit_results.json")
        sys.exit(1)
    
    audit_file = sys.argv[1]
    
    # Load audit data
    try:
        audit = json.load(open(audit_file))
    except Exception as e:
        print(f"Error loading {audit_file}: {e}")
        sys.exit(1)
    
    pd = audit["policy_disagreement_analysis"]
    
    # Get focus layers using the suggested pattern
    focus = pd.get("flagged_layers", [])
    
    # Optional: fall back to top-N by priority_score if nothing flagged
    if not focus:
        all_layers = pd.get("all_layers_with_disagreement", [])
        if all_layers:
            focus = sorted(
                all_layers, 
                key=lambda x: x["flagging_rationale"].get("priority_score", 0), 
                reverse=True
            )[:3]  # Top 3 by priority
            print(f"📊 No flagged layers found. Using top-{len(focus)} by priority_score")
        else:
            print("⚠️  No layers found for validation")
            return
    else:
        print(f"🎯 Using {len(focus)} flagged layers")
    
    # Validate each focus layer
    print("\n🔍 LAYER VALIDATION PRIORITIES:")
    print("="*70)
    
    for i, layer in enumerate(focus, 1):
        r = layer["flagging_rationale"]
        layer_name = layer["layer_name"]
        
        # Extract key metrics (with safe defaults)
        spread = r.get("spread", "?")
        spread_threshold = r.get("spread_threshold", "?")
        uniform_mult = r.get("uniform_mult", 0)
        uniform_mult_threshold = r.get("uniform_mult_threshold", "?")
        importance_share = r.get("importance_share", 0)
        priority_score = r.get("priority_score", 0)
        
        # Format output as suggested
        print(
            f"{i}. Validate {layer_name}: "
            f"spread={spread} (≥{spread_threshold}), "
            f"uniform_mult={uniform_mult:.2f} (≥{uniform_mult_threshold}), "
            f"share={importance_share:.3f}, "
            f"priority={priority_score:.2f}"
        )
        
        # Show k_values if available for rank selection
        k_values = r.get("k_values", [])
        policies = r.get("policies", [])
        if k_values and policies:
            policy_ranks = ", ".join([f"{pol}:{k}" for pol, k in zip(policies, k_values)])
            print(f"   Policy suggestions: {policy_ranks}")
            suggested_rank = max(k_values)  # Conservative choice
            print(f"   Recommended rank: {suggested_rank} (conservative)")
        
        print()

if __name__ == "__main__":
    main()