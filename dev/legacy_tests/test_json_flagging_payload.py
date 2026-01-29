#!/usr/bin/env python3
"""
Test JSON flagging payload for policy disagreement analysis.

Verifies that the machine-readable output includes detailed rationale
for why each layer was flagged or not flagged.
"""

import sys
import json
sys.path.insert(0, '.')

from gradience.cli import _analyze_policy_disagreements

def test_json_flagging_rationale():
    """Test the detailed JSON flagging rationale structure."""
    
    print("🤖 Testing JSON Flagging Payload")
    print("=" * 50)
    
    # Create test layers with clear disagreement patterns
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
    
    # Scenario: Mixed importance with clear flagging reasons
    layers = [
        # High-impact layer - should be flagged
        MockLayer(
            "transformer.h.0.attn.q_proj",
            {
                'energy_threshold': {'k': 8, 'confidence': 0.90},
                'knee_elbow': {'k': 2, 'confidence': 0.85},
                'entropy_effective': {'k': 6, 'confidence': 0.90}
            },
            frob_sq=50.0,   # High energy → high uniform_mult
            params=50000,
            utilization=0.8
        ),
        
        # Medium-impact layer - borderline case
        MockLayer(
            "transformer.h.1.mlp.c_fc", 
            {
                'energy_threshold': {'k': 6, 'confidence': 0.85},
                'knee_elbow': {'k': 3, 'confidence': 0.80},
                'entropy_effective': {'k': 5, 'confidence': 0.85}
            },
            frob_sq=15.0,   # Medium energy
            params=30000,
            utilization=0.6
        ),
        
        # Low-impact layer - should not be flagged
        MockLayer(
            "transformer.h.2.ln_1",
            {
                'energy_threshold': {'k': 4, 'confidence': 0.80},
                'knee_elbow': {'k': 1, 'confidence': 0.75},
                'entropy_effective': {'k': 3, 'confidence': 0.80}
            },
            frob_sq=2.0,    # Low energy → low uniform_mult
            params=1000,
            utilization=0.3
        ),
        
        # No disagreement layer - should not appear in analysis
        MockLayer(
            "transformer.h.3.ln_2",
            {
                'energy_threshold': {'k': 4, 'confidence': 0.80},
                'knee_elbow': {'k': 4, 'confidence': 0.80},  # Same k = no disagreement
                'entropy_effective': {'k': 4, 'confidence': 0.80}
            },
            frob_sq=1.0,
            params=1000,
            utilization=0.2
        )
    ]
    
    # Test default configuration
    print("Testing with default configuration...")
    config = {'quantile_threshold': 0.75, 'uniform_mult_gate': 1.5, 'metric': 'energy_share'}
    
    analysis = _analyze_policy_disagreements(layers, name_mapping, config)
    
    print(f"\n📊 Analysis Summary:")
    print(f"  Analysis performed: {analysis['analysis_performed']}")
    print(f"  Layers with disagreement: {analysis['summary']['layers_with_disagreement']}")
    print(f"  Layers flagged as high-impact: {analysis['summary']['layers_flagged_as_high_impact']}")
    print(f"  Distribution is flat: {analysis['distribution']['is_flat']}")
    
    # Show detailed flagging rationale for each layer
    print(f"\n🔍 Detailed Flagging Rationale:")
    print("=" * 60)
    
    for layer_data in analysis['all_layers_with_disagreement']:
        layer_name = layer_data['layer_name']
        rationale = layer_data['flagging_rationale']
        
        print(f"\nLayer: {layer_name}")
        
        # Handle both full and condensed rationale formats
        spread_threshold = rationale.get('spread_threshold', 'N/A')
        threshold_str = f"{spread_threshold:.1f}" if isinstance(spread_threshold, (int, float)) else str(spread_threshold)
        print(f"  Spread: {rationale['spread']} (threshold: {threshold_str})")
        
        if 'meets_spread_threshold' in rationale:
            print(f"  Meets spread threshold: {rationale['meets_spread_threshold']}")
        
        print(f"  Importance share: {rationale['importance_share']:.3f}")
        
        uniform_mult_threshold = rationale.get('uniform_mult_threshold', 'N/A')
        threshold_str = f"{uniform_mult_threshold:.1f}×" if isinstance(uniform_mult_threshold, (int, float)) else str(uniform_mult_threshold)
        print(f"  Uniform multiplier: {rationale['uniform_mult']:.2f}× (threshold: {threshold_str})")
        
        if 'meets_uniform_mult_threshold' in rationale:
            print(f"  Meets uniform mult threshold: {rationale['meets_uniform_mult_threshold']}")
        
        # Handle quantile threshold (may be in condensed or full format)
        meets_quantile = rationale.get('meets_quantile_threshold')
        if meets_quantile is not None:
            print(f"  Meets quantile threshold: {meets_quantile}")
            energy_threshold = rationale.get('energy_quantile_threshold', 0.0)
            print(f"  Energy quantile threshold: {energy_threshold:.3f}")
        else:
            print(f"  Quantile check: N/A (flat distribution or condensed format)")
        
        # These fields may not be in condensed format
        if 'passed_gate' in rationale:
            print(f"  Passed gate: {rationale['passed_gate']}")
        if 'flagged_as_high_impact' in rationale:
            print(f"  Flagged as high-impact: {rationale['flagged_as_high_impact']}")
        if 'k_values' in rationale and 'policies' in rationale:
            print(f"  K values: {rationale['k_values']} (policies: {', '.join(rationale['policies'])})")
        
        # Show condensed-specific info if present
        if 'failed_reasons' in rationale:
            print(f"  Failed reasons: {rationale['failed_reasons']}")
    
    # Show just the flagged layers
    print(f"\n🎯 High-Impact Flagged Layers:")
    print("=" * 60)
    
    for layer_data in analysis['flagged_layers']:
        layer_name = layer_data['layer_name']
        rationale = layer_data['flagging_rationale']
        
        print(f"\n✅ {layer_name}")
        print(f"   Why flagged:")
        print(f"   • Spread {rationale['spread']} ≥ {rationale['spread_threshold']:.1f} ✓")
        print(f"   • Uniform mult {rationale['uniform_mult']:.2f}× ≥ {rationale['uniform_mult_threshold']:.1f}× ✓")
        if rationale['meets_quantile_threshold']:
            print(f"   • Energy share ≥ {rationale['quantile_threshold']:.0%} quantile ✓")
        print(f"   • Distribution not flat ✓")
    
    # Test different configurations to show how rationale changes
    print(f"\n🧪 Testing Different Configurations:")
    print("=" * 60)
    
    configs = [
        {'name': 'Aggressive', 'config': {'quantile_threshold': 0.50, 'uniform_mult_gate': 1.2}},
        {'name': 'Very Conservative', 'config': {'quantile_threshold': 0.90, 'uniform_mult_gate': 2.0}}
    ]
    
    for test_config in configs:
        print(f"\n{test_config['name']} Configuration:")
        analysis = _analyze_policy_disagreements(layers, name_mapping, test_config['config'])
        
        flagged_count = analysis['summary']['layers_flagged_as_high_impact']
        total_disagreement = analysis['summary']['layers_with_disagreement']
        
        print(f"  Flagged: {flagged_count}/{total_disagreement} layers")
        
        for layer_data in analysis['flagged_layers']:
            layer_name = layer_data['layer_name']
            rationale = layer_data['flagging_rationale']
            print(f"  • {layer_name}: spread={rationale['spread']}, uniform_mult={rationale['uniform_mult']:.2f}×")


def test_json_serialization():
    """Test that the analysis results are properly JSON serializable."""
    
    print("\n\n📝 Testing JSON Serialization")
    print("=" * 50)
    
    # Simple test case
    class MockLayer:
        def __init__(self, name, rank_suggestions, frob_sq=1.0):
            self.name = name
            self.rank_suggestions = rank_suggestions
            self.frob_sq = frob_sq
            self.params = 1000
            self.utilization = 0.5
    
    layers = [
        MockLayer("test_layer", {
            'energy_threshold': {'k': 8, 'confidence': 0.90},
            'knee_elbow': {'k': 2, 'confidence': 0.85}
        }, frob_sq=25.0)
    ]
    
    name_mapping = {'energy_threshold': 'energy@0.90', 'knee_elbow': 'knee'}
    analysis = _analyze_policy_disagreements(layers, name_mapping)
    
    # Test JSON serialization
    try:
        json_str = json.dumps(analysis, indent=2)
        print("✅ JSON serialization successful")
        
        # Test deserialization  
        parsed = json.loads(json_str)
        print("✅ JSON deserialization successful")
        
        # Show sample of the JSON structure
        print("\n📄 Sample JSON structure:")
        print(json_str[:500] + "..." if len(json_str) > 500 else json_str)
        
    except Exception as e:
        print(f"❌ JSON serialization failed: {e}")


def test_edge_cases():
    """Test edge cases for JSON flagging rationale."""
    
    print("\n\n🧪 Testing Edge Cases")
    print("=" * 50)
    
    class MockLayer:
        def __init__(self, name, rank_suggestions, frob_sq=1.0):
            self.name = name
            self.rank_suggestions = rank_suggestions
            self.frob_sq = frob_sq
            self.params = 1000
            self.utilization = 0.5
    
    name_mapping = {'energy_threshold': 'energy@0.90', 'knee_elbow': 'knee'}
    
    # Test 1: No layers
    print("\nTest 1: No layers")
    analysis = _analyze_policy_disagreements([], name_mapping)
    print(f"  Analysis performed: {analysis['analysis_performed']}")
    print(f"  Reason: {analysis.get('reason', 'N/A')}")
    
    # Test 2: No disagreements (all consensus)
    print("\nTest 2: No disagreements")
    consensus_layers = [
        MockLayer("consensus_layer", {
            'energy_threshold': {'k': 4, 'confidence': 0.90},
            'knee_elbow': {'k': 4, 'confidence': 0.85}  # Same k
        })
    ]
    analysis = _analyze_policy_disagreements(consensus_layers, name_mapping)
    print(f"  Analysis performed: {analysis['analysis_performed']}")
    print(f"  Reason: {analysis.get('reason', 'N/A')}")
    
    # Test 3: Flat distribution
    print("\nTest 3: Flat distribution") 
    flat_layers = [
        MockLayer("layer1", {
            'energy_threshold': {'k': 8, 'confidence': 0.90},
            'knee_elbow': {'k': 2, 'confidence': 0.85}
        }, frob_sq=5.0),  # All similar energy
        MockLayer("layer2", {
            'energy_threshold': {'k': 6, 'confidence': 0.90},
            'knee_elbow': {'k': 3, 'confidence': 0.80}
        }, frob_sq=5.1),
        MockLayer("layer3", {
            'energy_threshold': {'k': 7, 'confidence': 0.90},
            'knee_elbow': {'k': 1, 'confidence': 0.75}
        }, frob_sq=4.9)
    ]
    analysis = _analyze_policy_disagreements(flat_layers, name_mapping)
    print(f"  Distribution is flat: {analysis['distribution']['is_flat']}")
    print(f"  Max uniform mult: {analysis['distribution']['max_uniform_mult']:.2f}")
    print(f"  Layers flagged: {analysis['summary']['layers_flagged_as_high_impact']}")


if __name__ == '__main__':
    test_json_flagging_rationale()
    test_json_serialization()
    test_edge_cases()
    
    print("\n" + "=" * 60)
    print("✅ JSON flagging payload tests completed!")
    print("\n🎯 Key benefits of machine-readable rationale:")
    print("  • Debugging: 'Why was this layer flagged?' → Clear metrics")
    print("  • Automation: Scripts can filter by specific criteria")
    print("  • Benchmarking: Systematic evaluation of policy disagreements")
    print("  • Reproducibility: Exact flagging logic captured in JSON")
    print("\n🤖 Perfect for Bench integration and downstream analysis!")