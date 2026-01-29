#!/usr/bin/env python3
"""
Test Bench focus integration with disagreement_focus_set.

Verifies that audit emits machine-readable focus information that Bench
can consume directly to restrict per-layer validation to critical layers.
"""

import sys
import json
sys.path.insert(0, '.')

from gradience.cli import _analyze_policy_disagreements

def test_focus_set_structure():
    """Test that disagreement_focus_set has the correct structure for Bench."""
    
    print("🎯 Testing Focus Set Structure for Bench Integration")
    print("=" * 70)
    
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
    
    # Test scenario: Clear hierarchy with 2 high-impact + 3 low-impact layers
    layers = [
        # High-impact layers (should be flagged)
        MockLayer("model.layers.0.self_attn.q_proj", {
            'energy_threshold': {'k': 8, 'confidence': 0.90},
            'knee_elbow': {'k': 2, 'confidence': 0.85},
            'entropy_effective': {'k': 6, 'confidence': 0.90}
        }, frob_sq=50.0, params=50000, utilization=0.8),
        
        MockLayer("model.layers.1.mlp.up_proj", {
            'energy_threshold': {'k': 7, 'confidence': 0.90},
            'knee_elbow': {'k': 3, 'confidence': 0.85},
            'entropy_effective': {'k': 5, 'confidence': 0.90}
        }, frob_sq=40.0, params=40000, utilization=0.7),
        
        # Low-impact layers (disagreement but not flagged)
        MockLayer("model.layers.2.norm1", {
            'energy_threshold': {'k': 4, 'confidence': 0.90},
            'knee_elbow': {'k': 1, 'confidence': 0.85},
            'entropy_effective': {'k': 3, 'confidence': 0.90}
        }, frob_sq=5.0, params=2000, utilization=0.3),
        
        MockLayer("model.layers.3.norm2", {
            'energy_threshold': {'k': 5, 'confidence': 0.90},
            'knee_elbow': {'k': 2, 'confidence': 0.85},
            'entropy_effective': {'k': 4, 'confidence': 0.90}
        }, frob_sq=3.0, params=1500, utilization=0.2),
        
        MockLayer("model.layers.4.dropout", {
            'energy_threshold': {'k': 3, 'confidence': 0.90},
            'knee_elbow': {'k': 1, 'confidence': 0.85},
            'entropy_effective': {'k': 2, 'confidence': 0.90}
        }, frob_sq=1.0, params=1000, utilization=0.1)
    ]
    
    print(f"Test scenario: {len(layers)} layers (expecting 2 high-impact)")
    
    # Run analysis
    analysis = _analyze_policy_disagreements(layers, name_mapping, None, "full")
    
    # Verify focus set exists and has correct structure
    focus_set = analysis.get('disagreement_focus_set')
    assert focus_set is not None, "disagreement_focus_set missing from analysis"
    
    # Check required fields
    required_fields = [
        'high_impact_layers',
        'recommended_focus_n',
        'focus_strategy',
        'message',
        'distribution_type'
    ]
    
    for field in required_fields:
        assert field in focus_set, f"Missing required field: {field}"
        print(f"  ✓ {field}: present")
    
    # Verify field types and values
    high_impact_layers = focus_set['high_impact_layers']
    recommended_focus_n = focus_set['recommended_focus_n']
    focus_strategy = focus_set['focus_strategy']
    message = focus_set['message']
    distribution_type = focus_set['distribution_type']
    
    assert isinstance(high_impact_layers, list), "high_impact_layers should be list"
    assert isinstance(recommended_focus_n, int), "recommended_focus_n should be int"
    assert isinstance(focus_strategy, str), "focus_strategy should be string"
    assert isinstance(message, str), "message should be string"
    assert isinstance(distribution_type, str), "distribution_type should be string"
    
    print(f"\\nFocus Set Contents:")
    print(f"  high_impact_layers: {high_impact_layers}")
    print(f"  recommended_focus_n: {recommended_focus_n}")
    print(f"  focus_strategy: {focus_strategy}")
    print(f"  distribution_type: {distribution_type}")
    print(f"  message: {message}")
    
    # Verify logical consistency
    assert len(high_impact_layers) == recommended_focus_n, f"Inconsistent counts: {len(high_impact_layers)} != {recommended_focus_n}"
    assert distribution_type in ['flat', 'hierarchical'], f"Invalid distribution_type: {distribution_type}"
    print(f"  ✓ Logical consistency verified")
    
    # Verify layer names are strings
    for layer_name in high_impact_layers:
        assert isinstance(layer_name, str), f"Layer name should be string: {layer_name}"
        assert len(layer_name) > 0, f"Layer name should not be empty: {layer_name}"
    
    print(f"  ✓ All layer names are valid strings")
    
    # Check that high-impact layers are sorted by priority_score
    flagged_layers = analysis['flagged_layers']
    expected_order = [layer['layer_name'] for layer in flagged_layers]
    
    assert high_impact_layers == expected_order, f"Focus set order doesn't match flagged layer order: {high_impact_layers} vs {expected_order}"
    print(f"  ✓ High-impact layers correctly ordered by priority_score")
    
    return focus_set


def test_flat_distribution_focus():
    """Test focus set for flat distribution scenario."""
    
    print(f"\\n\\n🏔️ Testing Flat Distribution Focus Set")
    print("=" * 70)
    
    class MockLayer:
        def __init__(self, name, rank_suggestions, frob_sq=1.0):
            self.name = name
            self.rank_suggestions = rank_suggestions
            self.frob_sq = frob_sq
            self.params = 1000
            self.utilization = 0.5
    
    name_mapping = {'energy_threshold': 'energy@0.90', 'knee_elbow': 'knee'}
    
    # Create flat distribution: all layers ~same energy 
    layers = [
        MockLayer("flat_layer_1", {
            'energy_threshold': {'k': 6, 'confidence': 0.90},
            'knee_elbow': {'k': 3, 'confidence': 0.85}  # spread=3
        }, frob_sq=5.0),
        
        MockLayer("flat_layer_2", {
            'energy_threshold': {'k': 5, 'confidence': 0.90},
            'knee_elbow': {'k': 2, 'confidence': 0.85}  # spread=3  
        }, frob_sq=4.8),
        
        MockLayer("flat_layer_3", {
            'energy_threshold': {'k': 4, 'confidence': 0.90},
            'knee_elbow': {'k': 1, 'confidence': 0.85}  # spread=3
        }, frob_sq=5.2),
        
        MockLayer("flat_layer_4", {
            'energy_threshold': {'k': 7, 'confidence': 0.90},
            'knee_elbow': {'k': 4, 'confidence': 0.85}  # spread=3
        }, frob_sq=4.9)
    ]
    
    print(f"Flat distribution test: {len(layers)} layers with similar energy")
    
    analysis = _analyze_policy_disagreements(layers, name_mapping, None, "full")
    focus_set = analysis['disagreement_focus_set']
    
    # Verify flat distribution handling
    assert focus_set['distribution_type'] == 'flat', f"Expected flat distribution, got {focus_set['distribution_type']}"
    assert focus_set['focus_strategy'] == 'top_disagreement_priority', f"Expected top_disagreement_priority strategy"
    assert focus_set['recommended_focus_n'] <= 3, f"Flat distribution should recommend ≤3 layers, got {focus_set['recommended_focus_n']}"
    
    print(f"  ✓ Flat distribution correctly identified")
    print(f"  ✓ Focus strategy: {focus_set['focus_strategy']}")
    print(f"  ✓ Recommended focus: {focus_set['recommended_focus_n']} layers")
    print(f"  ✓ Top layers: {focus_set['high_impact_layers']}")
    
    # Verify message mentions flat distribution
    message = focus_set['message'].lower()
    assert 'flat' in message, f"Message should mention flat distribution: {focus_set['message']}"
    print(f"  ✓ Message correctly explains flat distribution")


def test_no_disagreement_focus():
    """Test focus set when no high-impact layers are found."""
    
    print(f"\\n\\n❌ Testing No High-Impact Focus Set")
    print("=" * 70)
    
    class MockLayer:
        def __init__(self, name, rank_suggestions, frob_sq=0.1):  # Very low energy
            self.name = name
            self.rank_suggestions = rank_suggestions
            self.frob_sq = frob_sq
            self.params = 100
            self.utilization = 0.1
    
    name_mapping = {'energy_threshold': 'energy@0.90', 'knee_elbow': 'knee'}
    
    # Create layers with disagreement but very low importance
    layers = [
        MockLayer("unimportant_layer_1", {
            'energy_threshold': {'k': 6, 'confidence': 0.90},
            'knee_elbow': {'k': 3, 'confidence': 0.85}
        }, frob_sq=0.1),
        
        MockLayer("unimportant_layer_2", {
            'energy_threshold': {'k': 5, 'confidence': 0.90},
            'knee_elbow': {'k': 2, 'confidence': 0.85}
        }, frob_sq=0.05)
    ]
    
    print(f"No high-impact test: {len(layers)} layers with very low importance")
    
    analysis = _analyze_policy_disagreements(layers, name_mapping, None, "full")
    focus_set = analysis['disagreement_focus_set']
    
    # Should have no flagged layers but still provide focus guidance
    flagged_count = analysis['summary']['layers_flagged_as_high_impact']
    assert flagged_count == 0, f"Expected 0 flagged layers, got {flagged_count}"
    
    # Focus set should indicate no high-impact layers
    if focus_set['distribution_type'] == 'flat':
        # Flat distribution case - should recommend top disagreement layers
        assert focus_set['recommended_focus_n'] > 0, "Flat distribution should still recommend focus layers"
        assert len(focus_set['high_impact_layers']) > 0, "Should recommend top disagreement layers"
        print(f"  ✓ Flat distribution recommends {focus_set['recommended_focus_n']} top disagreement layers")
    else:
        # Hierarchical but no high-impact layers
        assert focus_set['recommended_focus_n'] == 0, f"Expected 0 focus layers, got {focus_set['recommended_focus_n']}"
        assert focus_set['focus_strategy'] == 'none', f"Expected 'none' strategy, got {focus_set['focus_strategy']}"
        print(f"  ✓ No high-impact layers correctly identified")
    
    print(f"  ✓ Focus strategy: {focus_set['focus_strategy']}")
    print(f"  ✓ Message: {focus_set['message']}")


def test_bench_integration_scenarios():
    """Test various scenarios that Bench would encounter."""
    
    print(f"\\n\\n⚡ Testing Bench Integration Scenarios")
    print("=" * 70)
    
    scenarios = [
        {
            'name': 'Single Critical Layer',
            'description': 'One layer dominates, clear focus',
            'expected_focus_n': 1,
            'expected_strategy': 'single_layer'
        },
        {
            'name': 'Multiple Critical Layers',
            'description': 'Several layers need attention',
            'expected_focus_n': 3,
            'expected_strategy': 'multiple_layers'
        },
        {
            'name': 'Edge Case: All Consensus',
            'description': 'No disagreements found',
            'expected_focus_n': 0,
            'expected_strategy': None  # May vary based on implementation
        }
    ]
    
    class MockLayer:
        def __init__(self, name, rank_suggestions, frob_sq=1.0):
            self.name = name
            self.rank_suggestions = rank_suggestions
            self.frob_sq = frob_sq
            self.params = 1000
            self.utilization = 0.5
    
    name_mapping = {'energy_threshold': 'energy@0.90', 'knee_elbow': 'knee'}
    
    # Scenario 1: Single critical layer
    print(f"\\nScenario 1: {scenarios[0]['name']}")
    single_layer = [
        MockLayer("critical_layer", {
            'energy_threshold': {'k': 8, 'confidence': 0.90},
            'knee_elbow': {'k': 2, 'confidence': 0.85}
        }, frob_sq=100.0),
        MockLayer("normal_layer", {
            'energy_threshold': {'k': 4, 'confidence': 0.90},
            'knee_elbow': {'k': 3, 'confidence': 0.85}
        }, frob_sq=5.0)
    ]
    
    analysis = _analyze_policy_disagreements(single_layer, name_mapping, None, "full")
    focus_set = analysis['disagreement_focus_set']
    
    print(f"  Focus strategy: {focus_set['focus_strategy']}")
    print(f"  Recommended focus: {focus_set['recommended_focus_n']} layers")
    print(f"  High-impact layers: {focus_set['high_impact_layers']}")
    print(f"  Bench usage: Restrict per-layer validation to {focus_set['high_impact_layers']}")
    
    # Scenario 2: Multiple critical layers
    print(f"\\nScenario 2: {scenarios[1]['name']}")
    multiple_layers = [
        MockLayer("critical_layer_1", {
            'energy_threshold': {'k': 8, 'confidence': 0.90},
            'knee_elbow': {'k': 2, 'confidence': 0.85}
        }, frob_sq=50.0),
        MockLayer("critical_layer_2", {
            'energy_threshold': {'k': 7, 'confidence': 0.90},
            'knee_elbow': {'k': 3, 'confidence': 0.85}
        }, frob_sq=45.0),
        MockLayer("critical_layer_3", {
            'energy_threshold': {'k': 6, 'confidence': 0.90},
            'knee_elbow': {'k': 2, 'confidence': 0.85}
        }, frob_sq=40.0),
        MockLayer("normal_layer", {
            'energy_threshold': {'k': 4, 'confidence': 0.90},
            'knee_elbow': {'k': 3, 'confidence': 0.85}
        }, frob_sq=5.0)
    ]
    
    analysis = _analyze_policy_disagreements(multiple_layers, name_mapping, None, "full")
    focus_set = analysis['disagreement_focus_set']
    
    print(f"  Focus strategy: {focus_set['focus_strategy']}")
    print(f"  Recommended focus: {focus_set['recommended_focus_n']} layers")
    print(f"  High-impact layers: {focus_set['high_impact_layers']}")
    print(f"  Bench usage: Restrict per-layer validation to these {focus_set['recommended_focus_n']} layers")
    
    # Show practical Bench integration
    print(f"\\n🔧 Bench Integration Example:")
    print(f"```python")
    print(f"# In Bench variant configuration:")
    print(f"if variant.type == 'per_layer' or variant.type == 'rank_pattern':")
    print(f"    focus_layers = audit_result['disagreement_focus_set']['high_impact_layers']")
    print(f"    if focus_layers:")
    print(f"        # Restrict to high-impact layers only")
    print(f"        variant.target_layers = focus_layers")
    print(f"    else:")
    print(f"        # Fall back to uniform suggestions")
    print(f"        variant.use_uniform_suggestions()")
    print(f"```")


if __name__ == '__main__':
    print("🎯 BENCH FOCUS INTEGRATION TEST")
    print("=" * 80)
    print("Testing disagreement_focus_set for direct Bench consumption")
    print("Enables Bench to restrict per-layer validation to critical layers only")
    print("=" * 80)
    
    # Test focus set structure
    focus_set = test_focus_set_structure()
    
    # Test flat distribution handling
    test_flat_distribution_focus()
    
    # Test no high-impact scenarios
    test_no_disagreement_focus()
    
    # Test Bench integration scenarios
    test_bench_integration_scenarios()
    
    print("\\n" + "=" * 80)
    print("✅ BENCH FOCUS INTEGRATION TESTS PASSED!")
    print()
    print("🎯 Benefits:")
    print("  • Machine-readable focus set for direct Bench consumption")
    print("  • Eliminates redundant disagreement analysis in Bench")
    print("  • Restricts per-layer validation to only critical layers")
    print("  • Provides fallback strategies for edge cases")
    print()
    print("🚀 Bench Integration:")
    print("  • Check disagreement_focus_set.high_impact_layers")
    print("  • If non-empty: restrict per-layer variants to these layers")
    print("  • If empty: fall back to uniform suggestions")
    print("  • Saves compute: only pay per-layer complexity when justified")
    print()
    print("📊 Example Focus Set:")
    print(f"  high_impact_layers: {focus_set['high_impact_layers']}")
    print(f"  recommended_focus_n: {focus_set['recommended_focus_n']}")
    print(f"  focus_strategy: {focus_set['focus_strategy']}")
    print(f"  message: '{focus_set['message']}'")
    print()
    print("This transforms audit → Bench handoff from manual interpretation")
    print("to direct machine consumption. Big practical win! 🎯")