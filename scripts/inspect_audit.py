#!/usr/bin/env python3
"""
Quick inspection script for LoRA gain audit results.

Usage:
    python scripts/inspect_audit.py /path/to/probe_r16/audit.json
    python scripts/inspect_audit.py /tmp/gradience_demo_gain_cpu/probe_r16/audit.json

Power user one-liner for extracting key gain metrics without jq dependency.
"""

import json
import sys
from pathlib import Path


def find_key(d, key):
    """Recursively search for key in nested dict/list structure."""
    if isinstance(d, dict):
        if key in d:
            return d[key]
        for v in d.values():
            r = find_key(v, key)
            if r is not None:
                return r
    elif isinstance(d, list):
        for v in d:
            r = find_key(v, key)
            if r is not None:
                return r
    return None


def inspect_audit(audit_path):
    """Extract and display key gain metrics from audit.json."""
    try:
        audit_data = json.loads(Path(audit_path).read_text())
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"❌ Error reading {audit_path}: {e}")
        return False
    
    # Extract key metrics with fallback search
    gain_summary = find_key(audit_data, "gain") or {}
    composition = find_key(audit_data, "composition") or {}
    
    # Magnitude metrics
    mean_fro = gain_summary.get("delta_fro_mean") or find_key(audit_data, "delta_fro_mean")
    mean_op = gain_summary.get("delta_op_mean") or find_key(audit_data, "delta_op_mean")
    
    # Concentration metrics  
    hhi = composition.get("concentration_index") or find_key(audit_data, "concentration_index")
    
    # Top layers
    top_k = composition.get("top_k", {})
    top_layers = top_k.get("layers", [])
    
    # Display results
    print("🔍 LoRA Gain Audit Inspection")
    print("=" * 30)
    
    if mean_fro is not None:
        print(f"Mean ||ΔW||_F: {mean_fro:.6f}")
    else:
        print("Mean ||ΔW||_F: Not found")
        
    if mean_op is not None:
        print(f"Mean ||ΔW||_2: {mean_op:.6f}")
    else:
        print("Mean ||ΔW||_2: Not found")
        
    if hhi is not None:
        print(f"HHI concentration: {hhi:.3f}")
        # Interpretation
        if hhi > 0.4:
            print("  → 🚨 Highly concentrated")
        elif hhi > 0.25:
            print("  → ⚠️  Moderately concentrated") 
        else:
            print("  → ✅ Well distributed")
    else:
        print("HHI concentration: Not found")
    
    if top_layers:
        print("Top layers by energy:")
        for i, layer_info in enumerate(top_layers[:5], 1):
            layer = layer_info.get("layer", "?")
            share = layer_info.get("share", 0)
            print(f"  {i}. Layer {layer}: {share:.1%}")
    else:
        print("Top layers: Not found")
    
    return True


def main():
    """Main entry point."""
    if len(sys.argv) != 2:
        print("Usage: python scripts/inspect_audit.py <audit.json>")
        print("")
        print("Example:")
        print("  python scripts/inspect_audit.py /tmp/gradience_demo_gain_cpu/probe_r16/audit.json")
        sys.exit(1)
    
    audit_path = sys.argv[1]
    
    # Support glob-style patterns
    if "*" in audit_path:
        from glob import glob
        matches = glob(audit_path)
        if not matches:
            print(f"❌ No files found matching: {audit_path}")
            sys.exit(1)
        audit_path = matches[0]  # Use first match
        print(f"📁 Using: {audit_path}")
        print()
    
    success = inspect_audit(audit_path)
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()