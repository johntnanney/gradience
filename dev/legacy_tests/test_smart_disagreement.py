#!/usr/bin/env python3
"""
Test smart disagreement detection weighted by layer importance.

Demonstrates how the new system prioritizes layers that are both 
ambiguous AND important, filtering out noisy disagreements.
"""

import sys
sys.path.insert(0, '.')

from gradience.cli import _print_policy_disagreement_summary

class MockLayer:
    """Mock layer with importance attributes."""
    
    def __init__(self, name, rank_suggestions, frob_sq=1.0, params=1000, utilization=0.5):
        self.name = name
        self.rank_suggestions = rank_suggestions
        self.frob_sq = frob_sq  # ||ΔW||_F²
        self.params = params
        self.utilization = utilization

def test_smart_disagreement_detection():
    """Test the importance-weighted disagreement detection."""
    
    print("🧠 Testing Smart Disagreement Detection")
    print("=" * 60)
    
    # Name mapping for policies
    name_mapping = {
        'energy_threshold': 'energy@0.90',
        'knee_elbow': 'knee', 
        'entropy_effective': 'erank',
        'optimal_hard_threshold': 'oht'
    }
    
    # Create mix of layers: some important with disagreement, some not
    layers = [
        # HIGH IMPORTANCE + HIGH DISAGREEMENT → Should be flagged as critical
        MockLayer(
            "model.layers.0.self_attn.q_proj",  # Important attention layer
            {
                'energy_threshold': {'k': 8, 'confidence': 0.90},
                'knee_elbow': {'k': 2, 'confidence': 0.85},  
                'entropy_effective': {'k': 6, 'confidence': 0.90},
                'optimal_hard_threshold': {'k': 1, 'confidence': 0.95}
            },
            frob_sq=25.0,  # High magnitude update ||ΔW||_F = 5.0
            params=50000,   # Large layer
            utilization=0.8 # High utilization
        ),
        
        MockLayer(
            "model.layers.5.mlp.up_proj",       # Important MLP layer  
            {
                'energy_threshold': {'k': 7, 'confidence': 0.90},
                'knee_elbow': {'k': 3, 'confidence': 0.80},
                'entropy_effective': {'k': 5, 'confidence': 0.85},
                'optimal_hard_threshold': {'k': 2, 'confidence': 0.90}
            },
            frob_sq=16.0,  # ||ΔW||_F = 4.0
            params=40000,
            utilization=0.7
        ),
        
        # LOW IMPORTANCE + HIGH DISAGREEMENT → Should be deprioritized
        MockLayer(
            "model.layers.15.norm1",            # Low-impact norm layer
            {
                'energy_threshold': {'k': 6, 'confidence': 0.90},
                'knee_elbow': {'k': 1, 'confidence': 0.80},
                'entropy_effective': {'k': 4, 'confidence': 0.85}
            },
            frob_sq=0.25,  # Low magnitude ||ΔW||_F = 0.5  
            params=1000,   # Small layer
            utilization=0.2 # Low utilization
        ),
        
        MockLayer(
            "model.layers.20.dropout",          # Very low-impact layer
            {
                'energy_threshold': {'k': 4, 'confidence': 0.90},
                'knee_elbow': {'k': 1, 'confidence': 0.85},
                'entropy_effective': {'k': 3, 'confidence': 0.90}
            },
            frob_sq=0.01,  # Very low magnitude ||ΔW||_F = 0.1
            params=500,
            utilization=0.1
        ),
        
        # MEDIUM IMPORTANCE + LOW DISAGREEMENT → Should not be flagged  
        MockLayer(
            "model.layers.8.self_attn.v_proj",   # Medium attention layer
            {
                'energy_threshold': {'k': 4, 'confidence': 0.90},
                'knee_elbow': {'k': 4, 'confidence': 0.85},
                'entropy_effective': {'k': 5, 'confidence': 0.90}
            },
            frob_sq=4.0,   # Medium magnitude ||ΔW||_F = 2.0
            params=20000,
            utilization=0.6
        ),
        
        # HIGH IMPORTANCE + LOW DISAGREEMENT → Should not be flagged
        MockLayer(
            "model.layers.2.mlp.down_proj",     # Important but consensus
            {
                'energy_threshold': {'k': 3, 'confidence': 0.90},
                'knee_elbow': {'k': 3, 'confidence': 0.85},
                'entropy_effective': {'k': 4, 'confidence': 0.90}
            },
            frob_sq=20.25, # High magnitude ||ΔW||_F = 4.5
            params=45000,
            utilization=0.9
        )
    ]
    
    print("Test scenario:")
    print("  • 2 layers: HIGH importance + HIGH disagreement (should be critical)")
    print("  • 2 layers: LOW importance + HIGH disagreement (should be deprioritized)")  
    print("  • 2 layers: Various importance + LOW disagreement (should be ignored)")
    print()
    
    # Run the smart disagreement detection
    _print_policy_disagreement_summary(layers, name_mapping)
    
    print("\n" + "=" * 60)
    print("✅ Smart disagreement detection demonstrated!")
    print()
    print("🎯 Key improvements:")
    print("  • Importance weighting: ||ΔW||_F (60%) + params (30%) + utilization (10%)")
    print("  • Smart filtering: spread >= max(3, 0.5*max_k) AND importance >= p75") 
    print("  • Priority focus: Show critical layers first, mention others")
    print("  • Actionable guidance: Clear next steps for Bench validation")

def test_edge_cases():
    """Test edge cases for smart disagreement detection."""
    
    print("\n" + "=" * 60)
    print("🧪 Testing Edge Cases")
    print("=" * 60)
    
    name_mapping = {
        'energy_threshold': 'energy@0.90',
        'knee_elbow': 'knee'
    }
    
    print("\nCase 1: No disagreements (all consensus)")
    consensus_layers = [
        MockLayer("consensus.layer", {
            'energy_threshold': {'k': 4, 'confidence': 0.90},
            'knee_elbow': {'k': 4, 'confidence': 0.85}
        })
    ]
    _print_policy_disagreement_summary(consensus_layers, name_mapping)
    
    print("\nCase 2: All disagreements are low importance")
    low_importance_layers = [
        MockLayer("unimportant.layer1", {
            'energy_threshold': {'k': 8, 'confidence': 0.90},
            'knee_elbow': {'k': 2, 'confidence': 0.85}
        }, frob_sq=0.01, params=100),
        MockLayer("unimportant.layer2", {
            'energy_threshold': {'k': 7, 'confidence': 0.90}, 
            'knee_elbow': {'k': 1, 'confidence': 0.80}
        }, frob_sq=0.01, params=100)
    ]
    _print_policy_disagreement_summary(low_importance_layers, name_mapping)
    
    print("\nCase 3: Flat distribution edge case (all values ~5.0)")
    # This should NOT mark any layer as "high-impact" because distribution is flat
    flat_distribution_layers = [
        MockLayer("layer1", {
            'energy_threshold': {'k': 8, 'confidence': 0.90},
            'knee_elbow': {'k': 2, 'confidence': 0.85}
        }, frob_sq=4.9*4.9, params=1000, utilization=0.5),  # importance ≈ 4.9
        MockLayer("layer2", {
            'energy_threshold': {'k': 6, 'confidence': 0.90},
            'knee_elbow': {'k': 3, 'confidence': 0.80}
        }, frob_sq=5.0*5.0, params=1000, utilization=0.5),  # importance ≈ 5.0
        MockLayer("layer3", {
            'energy_threshold': {'k': 7, 'confidence': 0.90},
            'knee_elbow': {'k': 1, 'confidence': 0.85}
        }, frob_sq=5.1*5.1, params=1000, utilization=0.5),  # importance ≈ 5.1
        MockLayer("layer4", {
            'energy_threshold': {'k': 4, 'confidence': 0.90},
            'knee_elbow': {'k': 3, 'confidence': 0.80}
        }, frob_sq=5.0*5.0, params=1000, utilization=0.5),  # importance ≈ 5.0
    ]
    print("Expected: NO high-impact layers (distribution is flat)")
    print("Fixed: New implementation should detect flat distribution and show no high-impact layers")
    _print_policy_disagreement_summary(flat_distribution_layers, name_mapping)
    
    print("\nCase 4: Single dominant layer (should be high-impact)")
    single_dominant_layers = [
        MockLayer("dominant_layer", {
            'energy_threshold': {'k': 8, 'confidence': 0.90},
            'knee_elbow': {'k': 2, 'confidence': 0.85}
        }, frob_sq=10.0*10.0, params=5000, utilization=0.8),  # Much higher energy
        MockLayer("normal_layer1", {
            'energy_threshold': {'k': 6, 'confidence': 0.90},
            'knee_elbow': {'k': 3, 'confidence': 0.80}
        }, frob_sq=2.0*2.0, params=1000, utilization=0.3),
        MockLayer("normal_layer2", {
            'energy_threshold': {'k': 7, 'confidence': 0.90},
            'knee_elbow': {'k': 1, 'confidence': 0.75}
        }, frob_sq=1.5*1.5, params=1000, utilization=0.4),
    ]
    print("Expected: dominant_layer marked as high-impact (uniform mult >> 1.5×)")
    _print_policy_disagreement_summary(single_dominant_layers, name_mapping)
    
    print("\nCase 5: Two dominant layers (should both be high-impact)")
    two_dominant_layers = [
        MockLayer("dominant_layer1", {
            'energy_threshold': {'k': 8, 'confidence': 0.90},
            'knee_elbow': {'k': 2, 'confidence': 0.85}
        }, frob_sq=8.0*8.0, params=4000, utilization=0.8),
        MockLayer("dominant_layer2", {
            'energy_threshold': {'k': 6, 'confidence': 0.90},
            'knee_elbow': {'k': 3, 'confidence': 0.80}
        }, frob_sq=7.0*7.0, params=3500, utilization=0.7),
        MockLayer("normal_layer1", {
            'energy_threshold': {'k': 7, 'confidence': 0.90},
            'knee_elbow': {'k': 1, 'confidence': 0.75}
        }, frob_sq=1.0*1.0, params=1000, utilization=0.2),
        MockLayer("normal_layer2", {
            'energy_threshold': {'k': 5, 'confidence': 0.90},
            'knee_elbow': {'k': 2, 'confidence': 0.75}
        }, frob_sq=1.2*1.2, params=1000, utilization=0.3),
    ]
    print("Expected: Both dominant layers marked as high-impact")
    _print_policy_disagreement_summary(two_dominant_layers, name_mapping)
    
    print("\nCase 6: Very small N=2 (edge case)")
    tiny_n_layers = [
        MockLayer("layer1", {
            'energy_threshold': {'k': 8, 'confidence': 0.90},
            'knee_elbow': {'k': 2, 'confidence': 0.85}
        }, frob_sq=5.0*5.0, params=2000, utilization=0.6),
        MockLayer("layer2", {
            'energy_threshold': {'k': 6, 'confidence': 0.90},
            'knee_elbow': {'k': 3, 'confidence': 0.80}
        }, frob_sq=3.0*3.0, params=1000, utilization=0.4),
    ]
    print("Expected: Handles N=2 gracefully (uniform_share=0.5, need 1.5× threshold)")
    _print_policy_disagreement_summary(tiny_n_layers, name_mapping)

if __name__ == '__main__':
    test_smart_disagreement_detection()
    test_edge_cases()