#!/usr/bin/env python3
"""
Create test scenario for golden pattern fallback behavior.

This demonstrates the fallback strategy when no layers are flagged as high-impact.
"""

import json
import sys
sys.path.insert(0, '.')

from gradience.cli import _analyze_policy_disagreements

def create_fallback_scenario():
    """Create audit data with no flagged layers but multiple disagreement layers."""
    
    class MockLayer:
        def __init__(self, name, rank_suggestions, frob_sq=1.0, params=1000, utilization=0.5):
            self.name = name
            self.rank_suggestions = rank_suggestions
            self.frob_sq = frob_sq
            self.params = params
            self.utilization = utilization
    
    # Create layers with disagreement but none flagged (low energy shares)
    layers = [
        # Medium disagreement, low energy share
        MockLayer("layer.0.attn", {
            'energy_threshold': {'k': 6, 'confidence': 0.85},
            'knee_elbow': {'k': 3, 'confidence': 0.80},
            'entropy_effective': {'k': 5, 'confidence': 0.85}
        }, frob_sq=8.0, params=10000, utilization=0.4),
        
        # Lower disagreement, low energy share  
        MockLayer("layer.1.mlp", {
            'energy_threshold': {'k': 5, 'confidence': 0.85},
            'knee_elbow': {'k': 3, 'confidence': 0.80},
            'entropy_effective': {'k': 4, 'confidence': 0.85}
        }, frob_sq=6.0, params=8000, utilization=0.3),
        
        # Minimal disagreement, very low energy share
        MockLayer("layer.2.norm", {
            'energy_threshold': {'k': 4, 'confidence': 0.80},
            'knee_elbow': {'k': 2, 'confidence': 0.75},
            'entropy_effective': {'k': 3, 'confidence': 0.80}
        }, frob_sq=3.0, params=1000, utilization=0.2),
        
        # Small disagreement, low energy share
        MockLayer("layer.3.dropout", {
            'energy_threshold': {'k': 3, 'confidence': 0.75},
            'knee_elbow': {'k': 1, 'confidence': 0.70},
            'entropy_effective': {'k': 2, 'confidence': 0.75}
        }, frob_sq=1.5, params=500, utilization=0.1),
    ]
    
    name_mapping = {
        'energy_threshold': 'energy@0.85',
        'knee_elbow': 'knee', 
        'entropy_effective': 'erank'
    }
    
    # Use conservative config so no layers get flagged
    conservative_config = {
        'quantile_threshold': 0.90,  # Very high threshold
        'uniform_mult_gate': 3.0,    # Very strict gate
        'metric': 'energy_share'
    }
    
    analysis = _analyze_policy_disagreements(layers, name_mapping, conservative_config, "full")
    
    # Create audit structure
    audit_data = {
        "schema_version": 1,
        "computed_at": "2026-01-28T17:30:00Z",
        "computed_with": {
            "gradience_version": "0.5.0",
            "git_sha": "2277c7b7fe3f"
        },
        "policy_disagreement_analysis": analysis,
        "adapter_info": {
            "total_layers": len(layers),
            "model_name": "fallback_test_model"
        }
    }
    
    return audit_data

if __name__ == '__main__':
    print("📊 CREATING FALLBACK SCENARIO")
    print("="*60)
    print("Creating audit data with no flagged layers but multiple disagreement layers")
    print("to demonstrate fallback behavior of golden pattern script.")
    print("="*60)
    
    audit_data = create_fallback_scenario()
    
    # Save to file
    with open('fallback_scenario_audit.json', 'w') as f:
        json.dump(audit_data, f, indent=2)
    
    # Show summary
    analysis = audit_data["policy_disagreement_analysis"]
    flagged = len(analysis.get("flagged_layers", []))
    total_disagreement = len(analysis.get("all_layers_with_disagreement", []))
    
    print(f"✅ Created fallback_scenario_audit.json")
    print(f"   Flagged layers: {flagged}")
    print(f"   Total disagreement layers: {total_disagreement}")
    
    if flagged == 0 and total_disagreement > 0:
        print(f"   Perfect for testing fallback → top-N by priority_score")
    
    print(f"\n🧪 Test the golden pattern fallback:")
    print(f"   python3 golden_downstream_pattern.py fallback_scenario_audit.json")
    print(f"   python3 golden_downstream_pattern.py fallback_scenario_audit.json --top-n 2")