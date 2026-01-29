#!/usr/bin/env python3
"""Create sample audit with flat distribution for explain command testing."""

import json
import sys
sys.path.insert(0, '.')

from gradience.cli import _analyze_policy_disagreements

def create_flat_audit_sample():
    """Create audit JSON with flat distribution for testing."""
    
    class MockLayer:
        def __init__(self, name, rank_suggestions, frob_sq=1.0, params=1000, utilization=0.5):
            self.name = name
            self.rank_suggestions = rank_suggestions
            self.frob_sq = frob_sq
            self.params = params
            self.utilization = utilization
    
    # Create layers with nearly equal energy shares → flat distribution
    layers = [
        MockLayer("layer.0.attn", {
            'energy_threshold': {'k': 6, 'confidence': 0.85},
            'knee_elbow': {'k': 4, 'confidence': 0.80},
            'entropy_effective': {'k': 5, 'confidence': 0.85}
        }, frob_sq=5.2, params=10000, utilization=0.4),
        
        MockLayer("layer.1.mlp", {
            'energy_threshold': {'k': 5, 'confidence': 0.85},
            'knee_elbow': {'k': 3, 'confidence': 0.80},
            'entropy_effective': {'k': 4, 'confidence': 0.85}
        }, frob_sq=5.0, params=9500, utilization=0.4),
        
        MockLayer("layer.2.norm", {
            'energy_threshold': {'k': 4, 'confidence': 0.80},
            'knee_elbow': {'k': 2, 'confidence': 0.75},
            'entropy_effective': {'k': 3, 'confidence': 0.80}
        }, frob_sq=4.8, params=9000, utilization=0.35),
    ]
    
    name_mapping = {
        'energy_threshold': 'energy@0.85',
        'knee_elbow': 'knee',
        'entropy_effective': 'erank'
    }
    
    # Config that will produce flat distribution
    config = {
        'quantile_threshold': 0.75,
        'uniform_mult_gate': 1.5,  # No layer will exceed this in flat scenario
        'metric': 'energy_share'
    }
    
    analysis = _analyze_policy_disagreements(layers, name_mapping, config, "full")
    
    # Create complete audit structure
    audit_data = {
        "schema_version": 1,
        "computed_at": "2026-01-28T17:40:00Z",
        "computed_with": {
            "gradience_version": "0.5.0",
            "git_sha": "2277c7b7fe3f"
        },
        "policy_disagreement_analysis": analysis,
        "adapter_info": {
            "total_layers": len(layers),
            "model_name": "flat_distribution_example"
        }
    }
    
    return audit_data

if __name__ == '__main__':
    audit_data = create_flat_audit_sample()
    
    # Save to file
    with open('flat_distribution_audit.json', 'w') as f:
        json.dump(audit_data, f, indent=2)
    
    # Show summary
    analysis = audit_data["policy_disagreement_analysis"]
    distribution = analysis["distribution"]
    witness = distribution["flatness_witness"]
    
    print("📊 CREATED FLAT DISTRIBUTION AUDIT SAMPLE")
    print("="*60)
    print(f"File: flat_distribution_audit.json")
    print(f"Distribution: {'FLAT' if distribution['is_flat'] else 'HIERARCHICAL'}")
    print(f"Witness: {witness['mathematical_proof']}")
    print(f"Flagged layers: {len(analysis.get('flagged_layers', []))}")
    print(f"Total disagreement layers: {len(analysis.get('all_layers_with_disagreement', []))}")
    print()
    print("🧪 Test the explain command with flatness witness:")
    print("python3 -m gradience.cli explain --audit-json flat_distribution_audit.json --layer 'layer.0.attn' --verbose")