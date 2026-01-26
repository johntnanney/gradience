#!/usr/bin/env python3
"""
Power user one-liner for LoRA gain audit inspection.

Copy-paste friendly version for quick audit.json analysis.
Usage: python3 - <<'PY' ... PY (see below)
"""

# START COPY-PASTE ONE-LINER
python3 - <<'PY'
import json
from pathlib import Path

# EDIT THIS PATH - point to your audit.json
audit_path = Path("/tmp/gradience_gain_audit_demo/probe_r16/audit.json")

a = json.loads(audit_path.read_text())

# Forgiving key search - handles any nesting structure
def find_key(d, key):
    if isinstance(d, dict):
        if key in d: return d[key]
        for v in d.values():
            r = find_key(v, key)
            if r is not None: return r
    elif isinstance(d, list):
        for v in d:
            r = find_key(v, key)
            if r is not None: return r
    return None

# Extract key metrics
gain = find_key(a, "gain") or {}
comp = find_key(a, "composition") or {}

mean_fro = gain.get("delta_fro_mean")
mean_op = gain.get("delta_op_mean") 
hhi = comp.get("concentration_index")
top_layers = comp.get("top_k", {}).get("layers", [])

# Quick display
print("LoRA Gain Summary:")
print(f"  ||ΔW||_F: {mean_fro:.6f}" if mean_fro else "  ||ΔW||_F: not found")
print(f"  ||ΔW||_2: {mean_op:.6f}" if mean_op else "  ||ΔW||_2: not found")
print(f"  HHI: {hhi:.3f}" if hhi else "  HHI: not found")

if top_layers:
    print("  Top layers:")
    for i, layer in enumerate(top_layers[:3], 1):
        print(f"    {i}. Layer {layer.get('layer', '?')}: {layer.get('share', 0):.1%}")
PY
# END COPY-PASTE ONE-LINER