#!/usr/bin/env python3
"""
Extreme regression test that WILL fail if the gate is removed.

This creates a scenario where percentile logic would definitely flag layers,
but the gate correctly prevents it.
"""

import sys
sys.path.insert(0, '.')

from gradience.cli import _analyze_policy_disagreements

def test_extreme_flat_case():
    """Create a case that percentile logic would definitely flag incorrectly."""
    
    print("🔥 EXTREME REGRESSION TEST")
    print("=" * 50)
    print("This test WILL fail if someone removes the uniform multiplier gate")
    print()
    
    class MockLayer:
        def __init__(self, name, rank_suggestions, frob_sq=1.0):
            self.name = name
            self.rank_suggestions = rank_suggestions
            self.frob_sq = frob_sq
            self.params = 1000
            self.utilization = 0.5
    
    name_mapping = {'energy_threshold': 'energy@0.90', 'knee_elbow': 'knee'}
    
    # Create an extreme case: 10 layers, all energy ~1.0 except one slightly higher
    # This ensures percentile logic would flag the "highest" layer even though
    # the difference is meaningless
    layers = []
    
    # 9 layers with energy = 1.0 (exactly the same)
    for i in range(9):
        layers.append(MockLayer(
            f"identical_layer_{i}",
            {
                'energy_threshold': {'k': 6, 'confidence': 0.85},
                'knee_elbow': {'k': 3, 'confidence': 0.80}  # Disagreement
            },
            frob_sq=1.0  # Energy = 1.0
        ))
    
    # 1 layer with energy = 1.001 (0.1% higher - meaningless difference)
    layers.append(MockLayer(
        "slightly_higher_layer",
        {
            'energy_threshold': {'k': 5, 'confidence': 0.85}, 
            'knee_elbow': {'k': 2, 'confidence': 0.80}  # Disagreement
        },
        frob_sq=1.001001  # Energy = 1.001
    ))
    
    print(f"Created extreme test case:")
    for layer in layers:
        energy = layer.frob_sq ** 0.5
        print(f"  {layer.name}: energy={energy:.6f}")
    
    total_energy = sum(l.frob_sq for l in layers)
    uniform_share = 1.0 / len(layers)  # 0.1 for 10 layers
    
    print(f"\nDistribution analysis:")
    print(f"  Total energy: {total_energy:.6f}")
    print(f"  Uniform share: {uniform_share:.6f}")
    
    # Calculate what uniform multipliers would be
    for layer in layers:
        energy_share = layer.frob_sq / total_energy
        uniform_mult = energy_share / uniform_share
        is_highest = layer.name == "slightly_higher_layer"
        print(f"  {layer.name}: energy_share={energy_share:.6f}, uniform_mult={uniform_mult:.6f} {'← highest' if is_highest else ''}")
    
    # Test with current logic (should correctly flag 0 layers)
    print(f"\n🔍 Testing with CORRECT logic (gate enabled):")
    analysis = _analyze_policy_disagreements(layers, name_mapping, None, "full")
    
    flagged_count = analysis['summary']['layers_flagged_as_high_impact']
    is_flat = analysis['distribution']['is_flat']
    max_uniform_mult = analysis['distribution']['max_uniform_mult']
    
    print(f"  Flagged layers: {flagged_count}")
    print(f"  Distribution flat: {is_flat}")
    print(f"  Max uniform mult: {max_uniform_mult:.6f}")
    
    # The critical assertion
    assert flagged_count == 0, f"REGRESSION FAILURE: Should flag 0 layers, got {flagged_count}"
    assert is_flat, f"REGRESSION FAILURE: Should detect flat distribution"
    assert max_uniform_mult < 1.5, f"REGRESSION FAILURE: Max uniform mult should be < 1.5, got {max_uniform_mult}"
    
    print(f"  ✅ Correctly flagged 0 layers (no false positives)")
    
    # Now demonstrate what broken percentile-only logic would do
    print(f"\n🚨 Simulating BROKEN logic (percentile-only, no gate):")
    
    # Manually calculate what percentile logic would do
    energy_shares = [l.frob_sq / total_energy for l in layers]
    p75 = sorted(energy_shares)[int(0.75 * len(energy_shares))]
    
    print(f"  75th percentile energy share: {p75:.6f}")
    
    would_be_flagged = []
    for i, layer in enumerate(layers):
        energy_share = energy_shares[i]
        spread = max([s['k'] for s in layer.rank_suggestions.values()]) - min([s['k'] for s in layer.rank_suggestions.values()])
        
        if energy_share >= p75 and spread >= 3:  # Simulate old broken logic
            would_be_flagged.append(layer.name)
            print(f"    ❌ Would incorrectly flag: {layer.name} (energy_share={energy_share:.6f} >= {p75:.6f})")
    
    print(f"  Broken logic would flag: {len(would_be_flagged)} layers")
    print(f"  Current logic flags: {flagged_count} layers")
    print(f"  Difference: {len(would_be_flagged) - flagged_count} false positives prevented!")
    
    return len(would_be_flagged) > 0  # Return True if broken logic would create false positives


def test_with_different_distributions():
    """Test various flat distribution patterns to ensure robustness."""
    
    print(f"\n\n🧪 Testing Various Flat Distribution Patterns")
    print("=" * 50)
    
    class MockLayer:
        def __init__(self, name, rank_suggestions, frob_sq=1.0):
            self.name = name
            self.rank_suggestions = rank_suggestions
            self.frob_sq = frob_sq
            self.params = 1000
            self.utilization = 0.5
    
    name_mapping = {'energy_threshold': 'energy@0.90', 'knee_elbow': 'knee'}
    
    test_cases = [
        {
            'name': 'All identical energies',
            'energies': [1.0] * 8,
            'expected_max_mult': 1.0
        },
        {
            'name': 'Tiny variation (±0.001)',
            'energies': [1.0 + i * 0.0001 for i in range(-4, 4)],
            'expected_max_mult': 1.001
        },
        {
            'name': 'Single outlier at +0.1%',
            'energies': [1.0] * 9 + [1.001],
            'expected_max_mult': 1.001
        }
    ]
    
    for test_case in test_cases:
        print(f"\n{test_case['name']}:")
        
        layers = []
        for i, energy in enumerate(test_case['energies']):
            # Some layers have disagreements
            if i % 3 == 0:
                rank_suggestions = {
                    'energy_threshold': {'k': 6, 'confidence': 0.85},
                    'knee_elbow': {'k': 2, 'confidence': 0.80}
                }
            else:
                rank_suggestions = {
                    'energy_threshold': {'k': 4, 'confidence': 0.85},
                    'knee_elbow': {'k': 4, 'confidence': 0.80}
                }
            
            layers.append(MockLayer(f"layer_{i}", rank_suggestions, frob_sq=energy**2))
        
        analysis = _analyze_policy_disagreements(layers, name_mapping, None, "full")
        
        flagged_count = analysis['summary']['layers_flagged_as_high_impact']
        max_uniform_mult = analysis['distribution']['max_uniform_mult']
        is_flat = analysis['distribution']['is_flat']
        
        print(f"  Max uniform mult: {max_uniform_mult:.6f} (expected ≈{test_case['expected_max_mult']:.3f})")
        print(f"  Distribution flat: {is_flat}")
        print(f"  Flagged layers: {flagged_count}")
        
        # All these should result in 0 flagged layers
        assert flagged_count == 0, f"Should flag 0 layers for {test_case['name']}, got {flagged_count}"
        assert is_flat, f"Should detect flat distribution for {test_case['name']}"
        
        print(f"  ✅ Correctly handled")


if __name__ == '__main__':
    print("🔥 EXTREME FLAT DISTRIBUTION REGRESSION TEST")
    print("=" * 70)
    print("This test creates scenarios where percentile logic WOULD flag layers,")
    print("but the uniform multiplier gate correctly prevents false positives.")
    print("=" * 70)
    
    # Run extreme test
    would_break = test_extreme_flat_case()
    
    # Test various patterns
    test_with_different_distributions()
    
    print("\n" + "=" * 70)
    print("🔒 EXTREME REGRESSION TESTS PASSED!")
    print(f"\n✅ Verification complete:")
    print(f"  • Broken percentile logic would create false positives: {'YES' if would_break else 'NO'}")
    print(f"  • Current gate logic correctly prevents false positives: YES")
    print(f"  • Various flat distribution patterns handled correctly: YES")
    print(f"\n🛡️ This test provides ironclad protection against flat distribution bugs!")
    print(f"   If this test ever fails, someone has broken the uniform multiplier gate.")