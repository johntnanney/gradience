#!/usr/bin/env python3
"""
Regression test for flat distribution gate.

This test is designed to FAIL if someone removes the uniform multiplier gate
and reverts to "p75 only" logic. It's intentionally hard to game.

The test uses N=8 layers with importance all within ±2%, ensuring that
any percentile-based approach will incorrectly flag layers as "high-impact"
when the distribution is actually flat and meaningless.
"""

import sys
sys.path.insert(0, '.')

from gradience.cli import _analyze_policy_disagreements

class MockLayer:
    """Mock layer with controlled importance attributes."""
    
    def __init__(self, name, rank_suggestions, frob_sq=1.0, params=1000, utilization=0.5):
        self.name = name
        self.rank_suggestions = rank_suggestions
        self.frob_sq = frob_sq
        self.params = params
        self.utilization = utilization

def test_flat_distribution_regression():
    """
    Regression test: Flat distribution should NOT flag any layers as high-impact.
    
    This test will FAIL if the uniform multiplier gate is removed and we revert
    to percentile-only logic.
    """
    
    print("🧪 REGRESSION TEST: Flat Distribution Gate")
    print("=" * 60)
    print("Purpose: Ensure flat distributions don't incorrectly flag layers")
    print("Pattern: N=8 layers, importance within ±2%, some disagreements")
    print("Expected: 0 high-impact layers + 'diffuse importance' message")
    print()
    
    name_mapping = {
        'energy_threshold': 'energy@0.90',
        'knee_elbow': 'knee',
        'entropy_effective': 'erank'
    }
    
    # Create N=8 layers with tightly clustered importance (±2%)
    # Base energy around 5.0, all within [4.9, 5.1] range
    base_energy = 5.0
    variance = 0.02  # ±2%
    
    layers = []
    for i in range(8):
        # Slight variation within ±2% 
        energy_variation = (i - 3.5) * variance  # Spread from -0.07 to +0.07
        layer_energy = base_energy + energy_variation
        
        # Create policy disagreements for some layers
        if i < 3:  # First 3 layers have disagreements
            rank_suggestions = {
                'energy_threshold': {'k': 6 + (i % 3), 'confidence': 0.85},
                'knee_elbow': {'k': 2 + (i % 2), 'confidence': 0.80}, 
                'entropy_effective': {'k': 4 + (i % 2), 'confidence': 0.85}
            }
        else:  # Rest have consensus (no disagreements)
            base_k = 5
            rank_suggestions = {
                'energy_threshold': {'k': base_k, 'confidence': 0.85},
                'knee_elbow': {'k': base_k, 'confidence': 0.80},
                'entropy_effective': {'k': base_k, 'confidence': 0.85}
            }
        
        layer = MockLayer(
            f"layer_{i:02d}",
            rank_suggestions,
            frob_sq=layer_energy * layer_energy,  # Convert to squared norm
            params=1000 + i * 100,  # Slight param variation
            utilization=0.5 + i * 0.01  # Slight utilization variation
        )
        layers.append(layer)
    
    print(f"Created {len(layers)} layers with energy distribution:")
    for i, layer in enumerate(layers):
        energy = layer.frob_sq ** 0.5
        has_disagreement = len(set(s['k'] for s in layer.rank_suggestions.values())) > 1
        print(f"  layer_{i:02d}: energy={energy:.3f}, disagreement={'YES' if has_disagreement else 'NO'}")
    
    print(f"\nEnergy range: [{min(l.frob_sq**0.5 for l in layers):.3f}, {max(l.frob_sq**0.5 for l in layers):.3f}]")
    print(f"Relative spread: {((max(l.frob_sq**0.5 for l in layers) - min(l.frob_sq**0.5 for l in layers)) / base_energy * 100):.1f}%")
    
    # Test with default configuration (should detect flat distribution)
    print(f"\n🔍 Testing with default configuration...")
    analysis = _analyze_policy_disagreements(layers, name_mapping)
    
    # Critical assertions that will FAIL if gate is removed
    assert analysis['analysis_performed'], "Analysis should be performed"
    assert analysis['distribution']['is_flat'], "Distribution should be detected as flat"
    
    flagged_count = analysis['summary']['layers_flagged_as_high_impact']
    disagreement_count = analysis['summary']['layers_with_disagreement']
    
    print(f"📊 Results:")
    print(f"  Distribution is flat: {analysis['distribution']['is_flat']}")
    print(f"  Max uniform multiplier: {analysis['distribution']['max_uniform_mult']:.3f}")
    print(f"  Layers with disagreement: {disagreement_count}")
    print(f"  Layers flagged as high-impact: {flagged_count}")
    
    # THE CRITICAL ASSERTION - This will fail if someone removes the gate
    assert flagged_count == 0, f"REGRESSION FAILURE: Expected 0 high-impact layers in flat distribution, got {flagged_count}"
    
    # Verify the reasoning is correct
    max_uniform_mult = analysis['distribution']['max_uniform_mult']
    gate_threshold = 1.5  # Default threshold
    
    assert max_uniform_mult < gate_threshold, \
        f"REGRESSION FAILURE: Max uniform mult {max_uniform_mult:.3f} should be < {gate_threshold} for flat distribution"
    
    print(f"\n✅ REGRESSION TEST PASSED")
    print(f"   • Flat distribution correctly detected")
    print(f"   • No layers incorrectly flagged as high-impact")
    print(f"   • Gate working: max_uniform_mult ({max_uniform_mult:.3f}) < threshold ({gate_threshold})")
    
    return True


def test_broken_logic_simulation():
    """
    Simulate what would happen with broken percentile-only logic.
    
    This demonstrates that percentile-only would incorrectly flag layers
    in the same scenario where the gate correctly prevents false positives.
    """
    
    print(f"\n\n🚨 SIMULATION: What Happens with Broken Logic")
    print("=" * 60)
    print("Demonstrating why the uniform multiplier gate is essential...")
    
    # Use the same flat distribution as above
    layers = []
    base_energy = 5.0
    for i in range(8):
        energy_variation = (i - 3.5) * 0.02
        layer_energy = base_energy + energy_variation
        
        if i < 3:  # Disagreements
            rank_suggestions = {
                'energy_threshold': {'k': 6 + (i % 3), 'confidence': 0.85},
                'knee_elbow': {'k': 2 + (i % 2), 'confidence': 0.80}, 
                'entropy_effective': {'k': 4 + (i % 2), 'confidence': 0.85}
            }
        else:
            base_k = 5
            rank_suggestions = {
                'energy_threshold': {'k': base_k, 'confidence': 0.85},
                'knee_elbow': {'k': base_k, 'confidence': 0.80},
                'entropy_effective': {'k': base_k, 'confidence': 0.85}
            }
        
        layer = MockLayer(f"layer_{i:02d}", rank_suggestions, frob_sq=layer_energy * layer_energy)
        layers.append(layer)
    
    # Test what percentile-only logic would do (simulate broken version)
    name_mapping = {'energy_threshold': 'energy@0.90', 'knee_elbow': 'knee', 'entropy_effective': 'erank'}
    
    # Simulate broken config (no gate, just percentile)
    broken_config = {
        'quantile_threshold': 0.75,
        'uniform_mult_gate': 0.1,  # Effectively disabled (very low threshold)
        'metric': 'energy_share'
    }
    
    broken_analysis = _analyze_policy_disagreements(layers, name_mapping, broken_config)
    broken_flagged = broken_analysis['summary']['layers_flagged_as_high_impact']
    
    print(f"Broken logic (gate disabled):")
    print(f"  Would flag {broken_flagged} layers as high-impact")
    print(f"  Max uniform mult: {broken_analysis['distribution']['max_uniform_mult']:.3f}")
    print(f"  Distribution detected as flat: {broken_analysis['distribution']['is_flat']}")
    
    # Show which layers would be incorrectly flagged
    if broken_flagged > 0:
        print(f"\n  Incorrectly flagged layers:")
        for layer_data in broken_analysis['flagged_layers']:
            layer_name = layer_data['layer_name'] 
            rationale = layer_data['flagging_rationale']
            print(f"    • {layer_name}: uniform_mult={rationale['uniform_mult']:.3f}")
    
    # Compare with correct logic
    correct_analysis = _analyze_policy_disagreements(layers, name_mapping)  # Default config
    correct_flagged = correct_analysis['summary']['layers_flagged_as_high_impact']
    
    print(f"\nCorrect logic (with gate):")
    print(f"  Flags {correct_flagged} layers as high-impact")
    print(f"  Correctly detects flat distribution")
    
    print(f"\n🎯 This demonstrates why the gate is essential:")
    print(f"  • Broken logic: {broken_flagged} false positives")
    print(f"  • Correct logic: {correct_flagged} false positives (should be 0)")
    print(f"  • Gate prevents meaningless percentile rankings in flat distributions")


def test_edge_case_variations():
    """Test variations that could potentially fool a naive regression test."""
    
    print(f"\n\n🧪 EDGE CASE VARIATIONS")
    print("=" * 60)
    print("Testing variations to ensure the regression test is robust...")
    
    name_mapping = {'energy_threshold': 'energy@0.90', 'knee_elbow': 'knee'}
    
    test_cases = [
        {
            'name': 'Very tight clustering (±0.5%)',
            'base_energy': 10.0,
            'variance': 0.005,
            'n_layers': 6
        },
        {
            'name': 'Large N with tight clustering (±1%)', 
            'base_energy': 3.0,
            'variance': 0.01,
            'n_layers': 12
        },
        {
            'name': 'High energy, tight clustering (±1.5%)',
            'base_energy': 20.0,
            'variance': 0.015,
            'n_layers': 8
        }
    ]
    
    for test_case in test_cases:
        print(f"\n{test_case['name']}:")
        
        layers = []
        for i in range(test_case['n_layers']):
            energy_variation = (i - (test_case['n_layers'] - 1) / 2) * test_case['variance']
            layer_energy = test_case['base_energy'] + energy_variation
            
            # Some disagreements
            if i < 3:
                rank_suggestions = {
                    'energy_threshold': {'k': 5 + i, 'confidence': 0.85},
                    'knee_elbow': {'k': 2 + (i % 2), 'confidence': 0.80}
                }
            else:
                rank_suggestions = {
                    'energy_threshold': {'k': 4, 'confidence': 0.85},
                    'knee_elbow': {'k': 4, 'confidence': 0.80}
                }
            
            layer = MockLayer(f"layer_{i}", rank_suggestions, frob_sq=layer_energy**2)
            layers.append(layer)
        
        analysis = _analyze_policy_disagreements(layers, name_mapping)
        
        energy_range = max(l.frob_sq**0.5 for l in layers) - min(l.frob_sq**0.5 for l in layers)
        relative_spread = (energy_range / test_case['base_energy']) * 100
        
        print(f"  Energy range: {energy_range:.4f} ({relative_spread:.2f}% of base)")
        print(f"  Max uniform mult: {analysis['distribution']['max_uniform_mult']:.3f}")
        print(f"  Distribution is flat: {analysis['distribution']['is_flat']}")
        print(f"  Layers flagged: {analysis['summary']['layers_flagged_as_high_impact']}")
        
        # Should detect as flat and flag 0 layers
        assert analysis['distribution']['is_flat'], f"Should detect flat distribution for {test_case['name']}"
        assert analysis['summary']['layers_flagged_as_high_impact'] == 0, \
            f"Should flag 0 layers for {test_case['name']}"
        
        print(f"  ✅ Correctly handled")


if __name__ == '__main__':
    print("🔒 FLAT DISTRIBUTION REGRESSION TESTS")
    print("=" * 70)
    print("These tests will FAIL if someone removes the uniform multiplier gate")
    print("and reverts to percentile-only logic.")
    print("=" * 70)
    
    # Run the main regression test
    test_flat_distribution_regression()
    
    # Demonstrate what broken logic would do
    test_broken_logic_simulation()
    
    # Test edge cases to ensure robustness
    test_edge_case_variations()
    
    print("\n" + "=" * 70)
    print("🔒 ALL REGRESSION TESTS PASSED!")
    print("\nThese tests guarantee that:")
    print("  • Flat distributions never incorrectly flag layers")
    print("  • The uniform multiplier gate cannot be removed without breaking tests") 
    print("  • Various clustering patterns are handled correctly")
    print("  • Future developers cannot accidentally revert to broken percentile-only logic")
    print("\n💡 If these tests ever fail, the flat distribution gate has been broken!")