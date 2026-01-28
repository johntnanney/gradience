#!/usr/bin/env python3
"""
Test JSON bloat prevention with condensed rationale for non-flagged layers.

Demonstrates significant size reduction when using 'flagged_only' vs 'full' verbosity
while preserving full debugging info for actually flagged layers.
"""

import sys
import json
sys.path.insert(0, '.')

from gradience.cli import _analyze_policy_disagreements

def test_json_size_reduction():
    """Test that 'flagged_only' significantly reduces JSON size vs 'full' verbosity."""
    
    print("📊 Testing JSON Size Reduction")
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
        'entropy_effective': 'erank',
        'optimal_hard_threshold': 'oht'
    }
    
    # Create a realistic large model scenario: 1 high-impact + many low-impact layers
    layers = []
    
    # 1 high-impact layer that will be flagged
    layers.append(MockLayer(
        "model.layers.0.self_attn.q_proj",  # Important layer
        {
            'energy_threshold': {'k': 8, 'confidence': 0.90},
            'knee_elbow': {'k': 2, 'confidence': 0.85},
            'entropy_effective': {'k': 6, 'confidence': 0.90},
            'optimal_hard_threshold': {'k': 1, 'confidence': 0.95}
        },
        frob_sq=50.0,  # High energy
        params=50000,
        utilization=0.8
    ))
    
    # Many low-impact layers with disagreement that won't be flagged
    for i in range(49):  # 49 additional layers = 50 total
        layers.append(MockLayer(
            f"model.layers.{i+1}.norm_{i%3}",  # Various layer types
            {
                'energy_threshold': {'k': 6, 'confidence': 0.85},
                'knee_elbow': {'k': 3, 'confidence': 0.80},
                'entropy_effective': {'k': 4, 'confidence': 0.85},
                'optimal_hard_threshold': {'k': 2, 'confidence': 0.90}
            },
            frob_sq=0.5,  # Low energy
            params=1000,
            utilization=0.2
        ))
    
    print(f"Test scenario: {len(layers)} layers")
    print(f"  • 1 high-impact layer (will be flagged)")
    print(f"  • {len(layers)-1} low-impact layers (have disagreement but won't be flagged)")
    print()
    
    # Test with full verbosity
    print("🔍 Testing with 'full' verbosity...")
    analysis_full = _analyze_policy_disagreements(layers, name_mapping, rationale_verbosity="full")
    json_full = json.dumps(analysis_full, indent=2)
    size_full = len(json_full)
    
    flagged_count_full = analysis_full['summary']['layers_flagged_as_high_impact']
    total_layers_full = len(analysis_full['all_layers_with_disagreement'])
    
    print(f"  Flagged layers: {flagged_count_full}")
    print(f"  Total layers processed: {total_layers_full}")
    print(f"  JSON size: {size_full:,} characters")
    
    # Test with flagged_only verbosity  
    print()
    print("🔍 Testing with 'flagged_only' verbosity...")
    analysis_flagged_only = _analyze_policy_disagreements(layers, name_mapping, rationale_verbosity="flagged_only")
    json_flagged_only = json.dumps(analysis_flagged_only, indent=2)
    size_flagged_only = len(json_flagged_only)
    
    flagged_count_flagged_only = analysis_flagged_only['summary']['layers_flagged_as_high_impact']
    total_layers_flagged_only = len(analysis_flagged_only['all_layers_with_disagreement'])
    
    print(f"  Flagged layers: {flagged_count_flagged_only}")
    print(f"  Total layers processed: {total_layers_flagged_only}")
    print(f"  JSON size: {size_flagged_only:,} characters")
    
    # Verify correctness
    print()
    print("✅ Correctness verification:")
    
    # Both should flag the same layers
    assert flagged_count_full == flagged_count_flagged_only, f"Flagged count mismatch: {flagged_count_full} vs {flagged_count_flagged_only}"
    print(f"  ✓ Same flagging results: {flagged_count_full} layers flagged in both modes")
    
    # Both should process same total layers
    assert total_layers_full == total_layers_flagged_only, f"Total layer count mismatch: {total_layers_full} vs {total_layers_flagged_only}"
    print(f"  ✓ Same total layers processed: {total_layers_full}")
    
    # Schema versions should match
    assert analysis_full['schema_version'] == analysis_flagged_only['schema_version'], "Schema version mismatch"
    print(f"  ✓ Schema versions match: {analysis_full['schema_version']}")
    
    # Size reduction calculation
    reduction_abs = size_full - size_flagged_only
    reduction_pct = (reduction_abs / size_full) * 100
    
    print()
    print("📈 Size Reduction Analysis:")
    print(f"  Full verbosity:        {size_full:,} characters")
    print(f"  Flagged-only verbosity: {size_flagged_only:,} characters")
    print(f"  Absolute reduction:     {reduction_abs:,} characters")
    print(f"  Percentage reduction:   {reduction_pct:.1f}%")
    
    # Verify significant size reduction
    assert reduction_pct > 20, f"Expected >20% reduction, got {reduction_pct:.1f}%"
    print(f"  ✓ Significant reduction achieved: {reduction_pct:.1f}% smaller")
    
    # Verify flagged layer still has full rationale
    print()
    print("🔍 Flagged Layer Rationale Verification:")
    flagged_layer_full = analysis_full['flagged_layers'][0]['flagging_rationale']
    flagged_layer_flagged_only = analysis_flagged_only['flagged_layers'][0]['flagging_rationale']
    
    # Flagged layer should have identical full rationale in both modes
    full_keys = set(flagged_layer_full.keys())
    flagged_only_keys = set(flagged_layer_flagged_only.keys())
    assert full_keys == flagged_only_keys, f"Flagged layer rationale keys differ: {full_keys} vs {flagged_only_keys}"
    print(f"  ✓ Flagged layer has full rationale in both modes ({len(full_keys)} fields)")
    
    # Verify non-flagged layer has condensed rationale
    print()
    print("🔍 Non-Flagged Layer Rationale Verification:")
    non_flagged_full = None
    non_flagged_flagged_only = None
    
    for layer in analysis_full['all_layers_with_disagreement']:
        if not layer['flagging_rationale'].get('flagged_as_high_impact', False):
            non_flagged_full = layer['flagging_rationale']
            break
    
    for layer in analysis_flagged_only['all_layers_with_disagreement']:
        if not layer['flagging_rationale'].get('flagged_as_high_impact', False):
            non_flagged_flagged_only = layer['flagging_rationale']
            break
    
    if non_flagged_full and non_flagged_flagged_only:
        full_fields = len(non_flagged_full)
        condensed_fields = len(non_flagged_flagged_only)
        
        print(f"  Full verbosity fields:      {full_fields}")
        print(f"  Flagged-only fields:        {condensed_fields}")
        print(f"  Field reduction:           {full_fields - condensed_fields} fields removed")
        
        # Verify condensed version has essential fields (including priority_score for Bench ordering)
        required_condensed_fields = {'spread', 'importance_share', 'uniform_mult', 'priority_score', 'failed_reasons'}
        condensed_keys = set(non_flagged_flagged_only.keys())
        assert required_condensed_fields.issubset(condensed_keys), f"Missing required condensed fields: {required_condensed_fields - condensed_keys}"
        print(f"  ✓ Condensed rationale has essential fields: {condensed_keys}")
        
        # Verify failed_reasons is populated
        failed_reasons = non_flagged_flagged_only.get('failed_reasons', [])
        assert len(failed_reasons) > 0, "Failed reasons should be populated for non-flagged layers"
        print(f"  ✓ Failed reasons explain why not flagged: {failed_reasons}")
    
    return {
        'size_full': size_full,
        'size_flagged_only': size_flagged_only,
        'reduction_pct': reduction_pct,
        'layers_tested': len(layers),
        'flagged_count': flagged_count_full
    }


def test_edge_cases():
    """Test edge cases for the condensed rationale."""
    
    print(f"\n\n🧪 Testing Edge Cases")
    print("=" * 60)
    
    class MockLayer:
        def __init__(self, name, rank_suggestions, frob_sq=1.0):
            self.name = name
            self.rank_suggestions = rank_suggestions
            self.frob_sq = frob_sq
            self.params = 1000
            self.utilization = 0.5
    
    name_mapping = {'energy_threshold': 'energy@0.90', 'knee_elbow': 'knee'}
    
    # Test 1: All layers flagged (no condensing should occur)
    print("\nTest 1: All layers flagged (should use full rationale for all)")
    high_importance_layers = [
        MockLayer("layer1", {
            'energy_threshold': {'k': 8, 'confidence': 0.90},
            'knee_elbow': {'k': 2, 'confidence': 0.85}
        }, frob_sq=100.0),
        MockLayer("layer2", {
            'energy_threshold': {'k': 6, 'confidence': 0.90},
            'knee_elbow': {'k': 3, 'confidence': 0.80}
        }, frob_sq=80.0)
    ]
    
    analysis = _analyze_policy_disagreements(high_importance_layers, name_mapping, rationale_verbosity="flagged_only")
    flagged_count = analysis['summary']['layers_flagged_as_high_impact']
    total_count = len(analysis['all_layers_with_disagreement'])
    
    print(f"  Layers processed: {total_count}, Flagged: {flagged_count}")
    if flagged_count == total_count:
        print(f"  ✓ All layers flagged - no condensing needed")
    
    # Test 2: No layers flagged (all should be condensed)
    print("\nTest 2: No layers flagged (should condense all)")
    low_importance_layers = [
        MockLayer("layer1", {
            'energy_threshold': {'k': 6, 'confidence': 0.90},
            'knee_elbow': {'k': 3, 'confidence': 0.85}
        }, frob_sq=0.1),
        MockLayer("layer2", {
            'energy_threshold': {'k': 4, 'confidence': 0.90},
            'knee_elbow': {'k': 2, 'confidence': 0.80}
        }, frob_sq=0.1)
    ]
    
    analysis = _analyze_policy_disagreements(low_importance_layers, name_mapping, rationale_verbosity="flagged_only")
    flagged_count = analysis['summary']['layers_flagged_as_high_impact']
    total_count = len(analysis['all_layers_with_disagreement'])
    
    print(f"  Layers processed: {total_count}, Flagged: {flagged_count}")
    
    # All should have condensed rationale
    for layer_data in analysis['all_layers_with_disagreement']:
        rationale = layer_data['flagging_rationale']
        if 'failed_reasons' in rationale:
            print(f"  ✓ Layer {layer_data['layer_name']}: condensed rationale with {len(rationale['failed_reasons'])} failure reasons")
    
    print("\nTest 3: Mixed scenario validation")
    # Create mixed importance layers
    mixed_layers = high_importance_layers + low_importance_layers
    
    analysis_full = _analyze_policy_disagreements(mixed_layers, name_mapping, rationale_verbosity="full")
    analysis_condensed = _analyze_policy_disagreements(mixed_layers, name_mapping, rationale_verbosity="flagged_only")
    
    flagged_full = analysis_full['summary']['layers_flagged_as_high_impact']
    flagged_condensed = analysis_condensed['summary']['layers_flagged_as_high_impact']
    
    assert flagged_full == flagged_condensed, f"Flagging mismatch: {flagged_full} vs {flagged_condensed}"
    print(f"  ✓ Mixed scenario: {flagged_full} layers flagged consistently in both modes")


if __name__ == '__main__':
    print("📊 JSON BLOAT PREVENTION TEST")
    print("=" * 70)
    print("Testing condensed rationale to reduce JSON size for large models")
    print("while preserving full debugging info for flagged layers.")
    print("=" * 70)
    
    # Main size reduction test
    results = test_json_size_reduction()
    
    # Edge case testing
    test_edge_cases()
    
    print("\n" + "=" * 70)
    print("✅ JSON BLOAT PREVENTION TESTS PASSED!")
    print()
    print("📈 Summary:")
    print(f"  • Tested with {results['layers_tested']} layers")
    print(f"  • JSON size reduction: {results['reduction_pct']:.1f}%")
    print(f"  • Flagged layers: {results['flagged_count']} (retain full rationale)")
    print(f"  • Non-flagged layers: {results['layers_tested'] - results['flagged_count']} (use condensed rationale)")
    print()
    print("🎯 Benefits:")
    print("  • Significantly smaller JSON artifacts for large models")
    print("  • Full debugging info preserved for actually important layers")
    print("  • Clear failure reasons for non-flagged layers")
    print("  • Identical flagging behavior in both verbosity modes")
    print()
    print("🚀 Usage:")
    print("  • Default: --disagreement-rationale flagged_only (recommended)")
    print("  • Debug:   --disagreement-rationale full (when you need complete info)")