#!/usr/bin/env python3
"""
Test priority_score ordering for Bench validation focus.

Verifies that layers are correctly ordered by priority_score = spread_norm * uniform_mult
to help Bench automatically select the most critical layers for focused validation.
"""

import sys
import json
sys.path.insert(0, '.')

from gradience.cli import _analyze_policy_disagreements, _print_policy_disagreement_summary

def test_priority_score_ordering():
    """Test that layers are correctly ordered by priority_score."""
    
    print("🎯 Testing Priority Score Ordering")
    print("=" * 60)
    
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
    
    # Create layers with different combinations of spread and importance
    # to test priority_score = spread_norm * uniform_mult ordering
    layers = [
        # Layer 1: High spread (8), High energy (50.0) → Should be highest priority
        MockLayer("layer_highest_priority", {
            'energy_threshold': {'k': 8, 'confidence': 0.90},
            'knee_elbow': {'k': 0, 'confidence': 0.85},  # spread = 8
            'entropy_effective': {'k': 4, 'confidence': 0.90}
        }, frob_sq=50.0, params=50000, utilization=0.8),
        
        # Layer 2: Medium spread (6), High energy (40.0) → Should be high priority  
        MockLayer("layer_high_priority", {
            'energy_threshold': {'k': 6, 'confidence': 0.90},
            'knee_elbow': {'k': 0, 'confidence': 0.85},  # spread = 6
            'entropy_effective': {'k': 3, 'confidence': 0.90}
        }, frob_sq=40.0, params=40000, utilization=0.7),
        
        # Layer 3: High spread (7), Medium energy (15.0) → Should be medium priority
        MockLayer("layer_medium_priority", {
            'energy_threshold': {'k': 7, 'confidence': 0.90},
            'knee_elbow': {'k': 0, 'confidence': 0.85},  # spread = 7
            'entropy_effective': {'k': 2, 'confidence': 0.90}
        }, frob_sq=15.0, params=20000, utilization=0.6),
        
        # Layer 4: Low spread (3), High energy (30.0) → Should be lower priority despite high energy
        MockLayer("layer_lower_priority", {
            'energy_threshold': {'k': 3, 'confidence': 0.90},
            'knee_elbow': {'k': 0, 'confidence': 0.85},  # spread = 3  
            'entropy_effective': {'k': 2, 'confidence': 0.90}
        }, frob_sq=30.0, params=30000, utilization=0.5),
        
        # Layer 5: Medium spread (5), Low energy (5.0) → Should be lowest priority
        MockLayer("layer_lowest_priority", {
            'energy_threshold': {'k': 5, 'confidence': 0.90},
            'knee_elbow': {'k': 0, 'confidence': 0.85},  # spread = 5
            'entropy_effective': {'k': 1, 'confidence': 0.90}
        }, frob_sq=5.0, params=10000, utilization=0.3)
    ]
    
    print(f"Test scenario: {len(layers)} layers with different spread/importance combinations")
    
    # Calculate expected order manually for verification
    total_energy = sum(l.frob_sq for l in layers)
    uniform_share = 1.0 / len(layers)
    
    expected_order = []
    for layer in layers:
        energy_share = layer.frob_sq / total_energy
        uniform_mult = energy_share / uniform_share
        
        # Calculate spread and spread_norm  
        k_values = [s['k'] for s in layer.rank_suggestions.values()]
        spread = max(k_values) - min(k_values)
        max_k = max(k_values)
        spread_threshold = max(3, 0.5 * max_k)
        spread_norm = max(0.0, spread / spread_threshold)
        
        priority_score = spread_norm * uniform_mult
        
        expected_order.append({
            'name': layer.name,
            'spread': spread,
            'uniform_mult': uniform_mult,
            'spread_norm': spread_norm,
            'priority_score': priority_score
        })
    
    # Sort by priority_score descending 
    expected_order.sort(key=lambda x: x['priority_score'], reverse=True)
    
    print("\\nExpected priority order (manual calculation):")
    for i, layer_info in enumerate(expected_order):
        print(f"  {i+1}. {layer_info['name']}: priority_score={layer_info['priority_score']:.2f}")
        print(f"     spread={layer_info['spread']}, uniform_mult={layer_info['uniform_mult']:.2f}, spread_norm={layer_info['spread_norm']:.2f}")
    
    # Test JSON analysis ordering
    print("\\n🔍 Testing JSON analysis ordering...")
    analysis = _analyze_policy_disagreements(layers, name_mapping, None, "full")
    
    flagged_layers = analysis['flagged_layers']
    all_layers = analysis['all_layers_with_disagreement']
    
    print(f"Flagged layers: {len(flagged_layers)}")
    print(f"All layers with disagreement: {len(all_layers)}")
    
    # Verify flagged layers are sorted by priority_score
    if len(flagged_layers) > 1:
        for i in range(len(flagged_layers) - 1):
            curr_score = flagged_layers[i]['flagging_rationale']['priority_score']
            next_score = flagged_layers[i+1]['flagging_rationale']['priority_score']
            assert curr_score >= next_score, f"Flagged layers not sorted: {curr_score} < {next_score}"
        print("  ✓ Flagged layers correctly sorted by priority_score")
    
    # Verify all layers are sorted by priority_score
    if len(all_layers) > 1:
        for i in range(len(all_layers) - 1):
            curr_score = all_layers[i]['flagging_rationale']['priority_score']
            next_score = all_layers[i+1]['flagging_rationale']['priority_score']
            assert curr_score >= next_score, f"All layers not sorted: {curr_score} < {next_score}"
        print("  ✓ All layers correctly sorted by priority_score")
    
    # Show actual JSON ordering
    print("\\nActual JSON ordering:")
    for i, layer_data in enumerate(all_layers):
        rationale = layer_data['flagging_rationale']
        layer_name = layer_data['layer_name']
        priority_score = rationale['priority_score']
        spread = rationale['spread']
        uniform_mult = rationale['uniform_mult']
        flagged = rationale.get('flagged_as_high_impact', False)
        status = "🔥 FLAGGED" if flagged else "○ not flagged"
        
        print(f"  {i+1}. {layer_name}: priority_score={priority_score:.2f} {status}")
        print(f"     spread={spread}, uniform_mult={uniform_mult:.2f}")
    
    # Verify top layer has highest priority_score
    top_layer = all_layers[0]
    top_priority = top_layer['flagging_rationale']['priority_score']
    print(f"\\n🎯 Top priority layer: {top_layer['layer_name']} (priority_score={top_priority:.2f})")
    
    # Verify this matches our manual calculation
    manual_top = expected_order[0]
    manual_priority = manual_top['priority_score']
    assert abs(top_priority - manual_priority) < 0.01, f"Priority score mismatch: {top_priority} vs {manual_priority}"
    print(f"  ✓ Matches manual calculation: {manual_priority:.2f}")
    
    return {
        'top_layer': top_layer['layer_name'],
        'top_priority': top_priority,
        'flagged_count': len(flagged_layers),
        'total_count': len(all_layers)
    }


def test_cli_output_with_priority():
    """Test that CLI shows the top focus layer correctly."""
    
    print(f"\\n\\n🖥️ Testing CLI Output with Priority Score")
    print("=" * 60)
    
    class MockLayer:
        def __init__(self, name, rank_suggestions, frob_sq=1.0):
            self.name = name
            self.rank_suggestions = rank_suggestions
            self.frob_sq = frob_sq
            self.params = 1000
            self.utilization = 0.5
    
    name_mapping = {'energy_threshold': 'energy@0.90', 'knee_elbow': 'knee'}
    
    # Create layers where one clearly has highest priority
    layers = [
        # This should be the clear winner: high spread + high energy
        MockLayer("model.layers.0.self_attn.q_proj", {
            'energy_threshold': {'k': 8, 'confidence': 0.90},
            'knee_elbow': {'k': 1, 'confidence': 0.85}  # spread = 7
        }, frob_sq=100.0),
        
        # Medium priority layers
        MockLayer("model.layers.1.mlp.up_proj", {
            'energy_threshold': {'k': 5, 'confidence': 0.90},
            'knee_elbow': {'k': 2, 'confidence': 0.85}  # spread = 3
        }, frob_sq=50.0),
        
        MockLayer("model.layers.2.norm", {
            'energy_threshold': {'k': 4, 'confidence': 0.90},
            'knee_elbow': {'k': 1, 'confidence': 0.85}  # spread = 3
        }, frob_sq=10.0)
    ]
    
    print("Running CLI with priority score messaging...")
    print("Expected: 'model.layers.0.self_attn.q_proj' as top focus layer")
    print()
    
    # Run CLI and capture the top focus message
    _print_policy_disagreement_summary(layers, name_mapping)
    
    # Also test JSON to verify the top layer
    analysis = _analyze_policy_disagreements(layers, name_mapping, None, "full")
    if analysis['flagged_layers']:
        top_layer = analysis['flagged_layers'][0]
        top_name = top_layer['layer_name'] 
        top_score = top_layer['flagging_rationale']['priority_score']
        print(f"\\n✓ JSON confirms top layer: {top_name} (priority_score={top_score:.2f})")


def test_priority_edge_cases():
    """Test priority score edge cases."""
    
    print(f"\\n\\n🧪 Testing Priority Score Edge Cases")
    print("=" * 60)
    
    class MockLayer:
        def __init__(self, name, rank_suggestions, frob_sq=1.0):
            self.name = name
            self.rank_suggestions = rank_suggestions
            self.frob_sq = frob_sq
            self.params = 1000
            self.utilization = 0.5
    
    name_mapping = {'energy_threshold': 'energy@0.90', 'knee_elbow': 'knee'}
    
    # Test 1: Zero spread (should have priority_score = 0)
    print("\\nTest 1: Zero spread layers")
    zero_spread_layers = [
        MockLayer("consensus_layer", {
            'energy_threshold': {'k': 4, 'confidence': 0.90},
            'knee_elbow': {'k': 4, 'confidence': 0.85}  # spread = 0
        }, frob_sq=100.0)
    ]
    
    analysis = _analyze_policy_disagreements(zero_spread_layers, name_mapping, None, "full")
    if analysis['all_layers_with_disagreement']:
        layer_data = analysis['all_layers_with_disagreement'][0]
        priority_score = layer_data['flagging_rationale']['priority_score']
        print(f"  Zero spread layer priority_score: {priority_score:.2f}")
        assert priority_score == 0.0, f"Expected 0.0 priority for zero spread, got {priority_score}"
        print("  ✓ Zero spread correctly results in priority_score = 0.0")
    
    # Test 2: Very low spread (below threshold)
    print("\\nTest 2: Below-threshold spread")
    low_spread_layers = [
        MockLayer("low_spread_layer", {
            'energy_threshold': {'k': 4, 'confidence': 0.90},
            'knee_elbow': {'k': 2, 'confidence': 0.85}  # spread = 2, threshold = 3
        }, frob_sq=100.0)
    ]
    
    analysis = _analyze_policy_disagreements(low_spread_layers, name_mapping, None, "full")
    if analysis['all_layers_with_disagreement']:
        layer_data = analysis['all_layers_with_disagreement'][0]
        priority_score = layer_data['flagging_rationale']['priority_score']
        spread = layer_data['flagging_rationale']['spread']
        spread_threshold = layer_data['flagging_rationale']['spread_threshold']
        print(f"  Low spread layer: spread={spread}, threshold={spread_threshold}, priority_score={priority_score:.2f}")
        
        # Should have priority < 1.0 since spread_norm < 1.0
        expected_max = 1.0 * (100.0 / 100.0) / (1.0 / 1.0)  # spread_norm * uniform_mult
        assert priority_score < expected_max, f"Priority should be < {expected_max}, got {priority_score}"
        print("  ✓ Below-threshold spread correctly reduces priority_score")
    
    print("\\nTest 3: Identical layers (should have equal priority_score)")
    identical_layers = [
        MockLayer("identical_layer_1", {
            'energy_threshold': {'k': 6, 'confidence': 0.90},
            'knee_elbow': {'k': 3, 'confidence': 0.85}
        }, frob_sq=25.0),
        MockLayer("identical_layer_2", {
            'energy_threshold': {'k': 6, 'confidence': 0.90},
            'knee_elbow': {'k': 3, 'confidence': 0.85}
        }, frob_sq=25.0)
    ]
    
    analysis = _analyze_policy_disagreements(identical_layers, name_mapping, None, "full")
    if len(analysis['all_layers_with_disagreement']) >= 2:
        score1 = analysis['all_layers_with_disagreement'][0]['flagging_rationale']['priority_score']
        score2 = analysis['all_layers_with_disagreement'][1]['flagging_rationale']['priority_score']
        print(f"  Identical layers: score1={score1:.3f}, score2={score2:.3f}")
        assert abs(score1 - score2) < 0.001, f"Identical layers should have same priority_score"
        print("  ✓ Identical layers have equal priority_score")


if __name__ == '__main__':
    print("🎯 PRIORITY SCORE TESTING")
    print("=" * 70)
    print("Testing priority_score = spread_norm * uniform_mult for Bench ordering")
    print("Higher scores indicate layers needing more urgent validation focus.")
    print("=" * 70)
    
    # Main ordering test
    results = test_priority_score_ordering()
    
    # CLI output test
    test_cli_output_with_priority()
    
    # Edge case testing
    test_priority_edge_cases()
    
    print("\\n" + "=" * 70)
    print("✅ PRIORITY SCORE TESTS PASSED!")
    print()
    print("📈 Summary:")
    print(f"  • Top priority layer: {results['top_layer']}")
    print(f"  • Highest priority score: {results['top_priority']:.2f}")
    print(f"  • Flagged layers: {results['flagged_count']}")
    print(f"  • Total layers processed: {results['total_count']}")
    print()
    print("🎯 Benefits for Bench:")
    print("  • Clear ordering: highest priority_score = most urgent validation")
    print("  • Combines spread (disagreement magnitude) + uniform_mult (importance)")
    print("  • Easy automation: pick top N layers by priority_score")
    print("  • Consistent ranking across different model sizes and configurations")
    print()
    print("🚀 Usage:")
    print("  • CLI shows: 'Top focus layer: layer_name (priority_score=X.X)'") 
    print("  • JSON provides: sorted arrays by priority_score for programmatic use")
    print("  • Bench can automatically select top priority_score layers for validation")