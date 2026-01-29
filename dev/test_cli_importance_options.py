#!/usr/bin/env python3
"""
Test CLI importance options in a realistic scenario.

Shows how different CLI parameter combinations affect importance filtering.
"""

import sys
sys.path.insert(0, '.')

from gradience.cli import _print_policy_disagreement_summary

def test_cli_parameter_combinations():
    """Test realistic CLI parameter combinations."""
    
    print("🎛️ Testing CLI Importance Parameter Combinations")
    print("=" * 60)
    
    # Create a realistic scenario with transformer layers
    class MockLayer:
        def __init__(self, name, rank_suggestions, frob_sq=1.0, params=1000, utilization=0.5):
            self.name = name
            self.rank_suggestions = rank_suggestions
            self.frob_sq = frob_sq
            self.params = params
            self.utilization = utilization
    
    name_mapping = {
        'energy_threshold': 'energy@0.90',
        'knee_elbow': 'knee',
        'entropy_effective': 'erank'
    }
    
    # Transformer model layers with realistic importance distribution
    layers = [
        # High-impact attention layers
        MockLayer("transformer.h.0.attn.q_proj", {
            'energy_threshold': {'k': 8, 'confidence': 0.90},
            'knee_elbow': {'k': 4, 'confidence': 0.85},
            'entropy_effective': {'k': 6, 'confidence': 0.90}
        }, frob_sq=45.0, params=50000, utilization=0.8),  # High energy
        
        MockLayer("transformer.h.1.attn.k_proj", {
            'energy_threshold': {'k': 7, 'confidence': 0.90},
            'knee_elbow': {'k': 3, 'confidence': 0.80},
            'entropy_effective': {'k': 5, 'confidence': 0.85}
        }, frob_sq=38.0, params=50000, utilization=0.7),  # High energy
        
        # Medium-impact MLP layers
        MockLayer("transformer.h.2.mlp.c_fc", {
            'energy_threshold': {'k': 6, 'confidence': 0.85},
            'knee_elbow': {'k': 4, 'confidence': 0.80},
            'entropy_effective': {'k': 5, 'confidence': 0.85}
        }, frob_sq=20.0, params=30000, utilization=0.6),  # Medium energy
        
        MockLayer("transformer.h.3.mlp.c_proj", {
            'energy_threshold': {'k': 5, 'confidence': 0.85},
            'knee_elbow': {'k': 3, 'confidence': 0.75},
            'entropy_effective': {'k': 4, 'confidence': 0.80}
        }, frob_sq=18.0, params=30000, utilization=0.5),  # Medium energy
        
        # Low-impact norm layers
        MockLayer("transformer.h.4.ln_1", {
            'energy_threshold': {'k': 4, 'confidence': 0.80},
            'knee_elbow': {'k': 2, 'confidence': 0.75},
            'entropy_effective': {'k': 3, 'confidence': 0.80}
        }, frob_sq=2.0, params=1000, utilization=0.3),   # Low energy
        
        MockLayer("transformer.h.5.ln_2", {
            'energy_threshold': {'k': 3, 'confidence': 0.80},
            'knee_elbow': {'k': 1, 'confidence': 0.70},
            'entropy_effective': {'k': 2, 'confidence': 0.75}
        }, frob_sq=1.5, params=1000, utilization=0.2),   # Low energy
    ]
    
    print("Scenario: 6 transformer layers with realistic importance hierarchy")
    print("  • 2 attention layers (high energy)")
    print("  • 2 MLP layers (medium energy)")  
    print("  • 2 norm layers (low energy)")
    print()
    
    # Test different CLI configurations
    configurations = [
        {
            'name': 'Conservative (default)',
            'description': 'Standard configuration - top quartile, 1.5× gate',
            'config': {
                'quantile_threshold': 0.75,
                'uniform_mult_gate': 1.5,
                'metric': 'energy_share'
            },
            'cli_args': '--importance-quantile 0.75 --importance-uniform-mult-gate 1.5'
        },
        {
            'name': 'Aggressive',
            'description': 'More layers flagged - median threshold, relaxed gate', 
            'config': {
                'quantile_threshold': 0.50,
                'uniform_mult_gate': 1.2,
                'metric': 'energy_share'
            },
            'cli_args': '--importance-quantile 0.50 --importance-uniform-mult-gate 1.2'
        },
        {
            'name': 'Very Conservative',
            'description': 'Only top 10%, strict gate for high confidence',
            'config': {
                'quantile_threshold': 0.90,
                'uniform_mult_gate': 2.0,
                'metric': 'energy_share'
            },
            'cli_args': '--importance-quantile 0.90 --importance-uniform-mult-gate 2.0'
        },
        {
            'name': 'Research Mode',
            'description': 'Balanced for exploration - 60th percentile, standard gate',
            'config': {
                'quantile_threshold': 0.60,
                'uniform_mult_gate': 1.5,
                'metric': 'energy_share'
            },
            'cli_args': '--importance-quantile 0.60 --importance-uniform-mult-gate 1.5'
        }
    ]
    
    for i, config_info in enumerate(configurations, 1):
        print(f"Configuration {i}: {config_info['name']}")
        print(f"Description: {config_info['description']}")
        print(f"CLI usage: gradience audit --peft-dir ./adapter {config_info['cli_args']}")
        print()
        
        _print_policy_disagreement_summary(layers, name_mapping, config_info['config'])
        
        print("\n" + "=" * 60)
        print()
    
    print("🎯 Key Insights from Parameter Combinations:")
    print()
    print("1. **Conservative (default)**: Good balance for production use")
    print("   - Flags only clearly important + disagreeing layers")
    print("   - Minimizes false positives from noise")
    print()
    print("2. **Aggressive**: Useful for comprehensive analysis")
    print("   - Catches more potential issues")
    print("   - May include some lower-impact layers")
    print()
    print("3. **Very Conservative**: For high-stakes environments")
    print("   - Flags only the most critical disagreements")
    print("   - Reduces validation overhead")
    print()
    print("4. **Research Mode**: Balanced for exploration")
    print("   - Good compromise for iterative development")
    print("   - Adapts well to different model architectures")
    print()
    print("💡 **Recommendation**: Start with default, adjust based on your priorities:")
    print("   - More false positives OK → Lower quantile + gate")
    print("   - Fewer false positives → Higher quantile + gate")
    print("   - Different model types → Tune gate threshold")


if __name__ == '__main__':
    test_cli_parameter_combinations()
    
    print("\n✅ CLI importance options testing completed!")
    print("\n🔧 Configuration is now fully customizable:")
    print("  • --importance-quantile: Controls importance threshold (0.0-1.0)")
    print("  • --importance-uniform-mult-gate: Controls flat distribution detection")
    print("  • --importance-metric: Future-proofs for different importance calculations")
    print("\n🎯 Users can now tune policy disagreement detection for their specific needs!")