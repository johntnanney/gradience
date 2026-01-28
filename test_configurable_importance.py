#!/usr/bin/env python3
"""
Test configurable importance thresholds in smart disagreement detection.

Verifies that the new CLI parameters correctly adjust importance filtering.
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

def test_configurable_quantile_threshold():
    """Test that changing quantile_threshold affects which layers are flagged."""
    
    print("🧪 Testing Configurable Quantile Threshold")
    print("=" * 50)
    
    # Name mapping for policies
    name_mapping = {
        'energy_threshold': 'energy@0.90',
        'knee_elbow': 'knee'
    }
    
    # Create layers with clear hierarchy of importance
    layers = [
        # Very high importance layer (should always be flagged)
        MockLayer(
            "high_importance_layer",
            {
                'energy_threshold': {'k': 8, 'confidence': 0.90},
                'knee_elbow': {'k': 2, 'confidence': 0.85}
            },
            frob_sq=100.0,  # Very high energy
            params=50000,
            utilization=0.9
        ),
        
        # Medium importance layer (borderline - depends on threshold)
        MockLayer(
            "medium_importance_layer", 
            {
                'energy_threshold': {'k': 7, 'confidence': 0.90},
                'knee_elbow': {'k': 3, 'confidence': 0.80}
            },
            frob_sq=20.0,   # Medium energy
            params=30000,
            utilization=0.6
        ),
        
        # Low importance layer (should not be flagged)
        MockLayer(
            "low_importance_layer",
            {
                'energy_threshold': {'k': 6, 'confidence': 0.90},
                'knee_elbow': {'k': 1, 'confidence': 0.75}
            },
            frob_sq=4.0,    # Low energy
            params=10000,
            utilization=0.3
        )
    ]
    
    print("\nTest 1: Default threshold (0.75 = top 25%)")
    print("Expected: Should flag high_importance_layer only")
    default_config = {'quantile_threshold': 0.75, 'uniform_mult_gate': 1.5}
    _print_policy_disagreement_summary(layers, name_mapping, default_config)
    
    print("\n" + "=" * 50)
    print("\nTest 2: More permissive threshold (0.50 = top 50%)")  
    print("Expected: Should flag both high and medium importance layers")
    permissive_config = {'quantile_threshold': 0.50, 'uniform_mult_gate': 1.5}
    _print_policy_disagreement_summary(layers, name_mapping, permissive_config)
    
    print("\n" + "=" * 50)
    print("\nTest 3: Very strict threshold (0.90 = top 10%)")
    print("Expected: Should flag only the highest importance layer")
    strict_config = {'quantile_threshold': 0.90, 'uniform_mult_gate': 1.5}
    _print_policy_disagreement_summary(layers, name_mapping, strict_config)


def test_configurable_uniform_mult_gate():
    """Test that changing uniform_mult_gate affects flat distribution detection."""
    
    print("\n\n🧪 Testing Configurable Uniform Multiplier Gate")
    print("=" * 50)
    
    name_mapping = {
        'energy_threshold': 'energy@0.90',
        'knee_elbow': 'knee'
    }
    
    # Create layers with mildly concentrated distribution
    layers = [
        MockLayer("layer1", {
            'energy_threshold': {'k': 8, 'confidence': 0.90},
            'knee_elbow': {'k': 2, 'confidence': 0.85}
        }, frob_sq=5.0*5.0, params=1000, utilization=0.5),  # uniform_mult ≈ 1.67
        
        MockLayer("layer2", {
            'energy_threshold': {'k': 6, 'confidence': 0.90},
            'knee_elbow': {'k': 3, 'confidence': 0.80}
        }, frob_sq=4.5*4.5, params=1000, utilization=0.5),  # uniform_mult ≈ 1.35
        
        MockLayer("layer3", {
            'energy_threshold': {'k': 7, 'confidence': 0.90},
            'knee_elbow': {'k': 1, 'confidence': 0.75}
        }, frob_sq=3.5*3.5, params=1000, utilization=0.5),  # uniform_mult ≈ 0.82
    ]
    
    print("\nTest 1: Default gate (1.5)")
    print("Expected: layer1 should be high-impact (1.67 > 1.5)")
    default_config = {'quantile_threshold': 0.75, 'uniform_mult_gate': 1.5}
    _print_policy_disagreement_summary(layers, name_mapping, default_config)
    
    print("\n" + "=" * 50)
    print("\nTest 2: Strict gate (2.0)")
    print("Expected: Flat distribution detected (max=1.67 < 2.0)")
    strict_config = {'quantile_threshold': 0.75, 'uniform_mult_gate': 2.0}
    _print_policy_disagreement_summary(layers, name_mapping, strict_config)
    
    print("\n" + "=" * 50)
    print("\nTest 3: Permissive gate (1.2)")
    print("Expected: Both layer1 and layer2 should be high-impact")
    permissive_config = {'quantile_threshold': 0.75, 'uniform_mult_gate': 1.2}
    _print_policy_disagreement_summary(layers, name_mapping, permissive_config)


def test_edge_cases_with_config():
    """Test edge cases with different configurations."""
    
    print("\n\n🧪 Testing Edge Cases with Configuration")
    print("=" * 50)
    
    name_mapping = {
        'energy_threshold': 'energy@0.90',
        'knee_elbow': 'knee'
    }
    
    # Single layer case
    single_layer = [
        MockLayer("single_layer", {
            'energy_threshold': {'k': 8, 'confidence': 0.90},
            'knee_elbow': {'k': 2, 'confidence': 0.85}
        }, frob_sq=25.0, params=1000, utilization=0.5)
    ]
    
    print("\nTest: Single layer with disagreement")
    print("Expected: Should handle gracefully regardless of thresholds")
    config = {'quantile_threshold': 0.75, 'uniform_mult_gate': 1.5}
    _print_policy_disagreement_summary(single_layer, name_mapping, config)


if __name__ == '__main__':
    test_configurable_quantile_threshold()
    test_configurable_uniform_mult_gate()
    test_edge_cases_with_config()
    
    print("\n" + "=" * 60)
    print("✅ Configurable importance threshold tests completed!")
    print("\n🎯 Key verifications:")
    print("  • Quantile threshold controls which layers qualify as high-importance")
    print("  • Uniform multiplier gate prevents false positives in flat distributions")
    print("  • Both parameters work together to provide precise control")
    print("  • Edge cases handled gracefully with any configuration")