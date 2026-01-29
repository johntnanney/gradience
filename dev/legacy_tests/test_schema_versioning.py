#!/usr/bin/env python3
"""
Test schema versioning and configuration capture in policy disagreement analysis.

Verifies that the JSON output includes complete versioning, configuration,
and timestamp information for reproducibility and debugging.
"""

import sys
import json
import time
sys.path.insert(0, '.')

from gradience.cli import _analyze_policy_disagreements

def test_schema_versioning():
    """Test that schema versioning captures all necessary information."""
    
    print("🔒 Testing Schema Versioning and Configuration Capture")
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
    
    # Create test layers
    layers = [
        MockLayer(
            "transformer.h.0.attn.q_proj",
            {
                'energy_threshold': {'k': 8, 'confidence': 0.90},
                'knee_elbow': {'k': 2, 'confidence': 0.85},
                'entropy_effective': {'k': 6, 'confidence': 0.90}
            },
            frob_sq=50.0,
            params=50000,
            utilization=0.8
        ),
        MockLayer(
            "transformer.h.1.mlp.c_fc", 
            {
                'energy_threshold': {'k': 6, 'confidence': 0.85},
                'knee_elbow': {'k': 3, 'confidence': 0.80},
                'entropy_effective': {'k': 5, 'confidence': 0.85}
            },
            frob_sq=15.0,
            params=30000,
            utilization=0.6
        )
    ]
    
    # Test with custom configuration
    custom_config = {
        'quantile_threshold': 0.60,
        'uniform_mult_gate': 1.8,
        'metric': 'energy_share'
    }
    
    print("Testing with custom configuration...")
    analysis = _analyze_policy_disagreements(layers, name_mapping, custom_config, "full")
    
    # Verify schema structure
    print("\n📋 Schema Verification:")
    required_top_level_fields = [
        "schema_version",
        "computed_at",
        "computed_with",
        "disagreement_config",
        "analysis_performed",
        "distribution",
        "summary",
        "flagged_layers",
        "all_layers_with_disagreement"
    ]
    
    for field in required_top_level_fields:
        assert field in analysis, f"Missing required field: {field}"
        print(f"  ✓ {field}: present")
    
    # Verify schema version
    assert analysis["schema_version"] == 1, f"Expected schema_version=1, got {analysis['schema_version']}"
    print(f"  ✓ Schema version: {analysis['schema_version']}")
    
    # Verify timestamp format (ISO 8601)
    computed_at = analysis["computed_at"]
    assert "T" in computed_at and computed_at.endswith("Z"), f"Invalid timestamp format: {computed_at}"
    print(f"  ✓ Timestamp: {computed_at}")
    
    # Verify version information
    computed_with = analysis["computed_with"]
    required_version_fields = ["gradience_version"]
    for field in required_version_fields:
        assert field in computed_with, f"Missing version field: {field}"
        print(f"  ✓ {field}: {computed_with[field]}")
    
    if computed_with.get("git_sha"):
        print(f"  ✓ git_sha: {computed_with['git_sha']}")
    else:
        print(f"  ✓ git_sha: not available (expected in non-git environments)")
    
    # Verify configuration capture
    print(f"\n⚙️ Configuration Capture:")
    disagreement_config = analysis["disagreement_config"]
    
    # Verify all configuration parameters are captured
    required_config_fields = [
        "quantile_threshold",
        "uniform_mult_gate", 
        "importance_metric",
        "quantile_pct",
        "min_uniform_mult_threshold",
        "spread_base_threshold",
        "spread_dynamic_factor",
        "spread_calculation_formula",
        "flat_detection_enabled",
        "spread_filter_enabled",
        "quantile_filter_enabled",
        "config_capture_version",
        "algorithm_name"
    ]
    
    for field in required_config_fields:
        assert field in disagreement_config, f"Missing config field: {field}"
        print(f"  ✓ {field}: {disagreement_config[field]}")
    
    # Verify configuration values match input
    assert disagreement_config["quantile_threshold"] == 0.60, "Quantile threshold not captured correctly"
    assert disagreement_config["uniform_mult_gate"] == 1.8, "Uniform mult gate not captured correctly"
    assert disagreement_config["importance_metric"] == "energy_share", "Importance metric not captured correctly"
    
    print(f"  ✓ Custom configuration correctly captured")
    
    # Test JSON serialization
    print(f"\n📄 JSON Serialization:")
    try:
        json_str = json.dumps(analysis, indent=2)
        print(f"  ✓ JSON serialization successful ({len(json_str):,} characters)")
        
        # Test round-trip
        parsed = json.loads(json_str)
        assert parsed["schema_version"] == 1, "Schema version lost in round-trip"
        assert parsed["disagreement_config"]["algorithm_name"] == "energy_share_uniform_multiplier_gate", "Algorithm name lost"
        print(f"  ✓ Round-trip serialization successful")
        
    except Exception as e:
        print(f"  ❌ JSON serialization failed: {e}")
        raise


def test_configuration_combinations():
    """Test different configuration combinations to ensure all are captured."""
    
    print(f"\n\n🧪 Testing Configuration Combinations")
    print("=" * 70)
    
    class MockLayer:
        def __init__(self, name, frob_sq=1.0):
            self.name = name
            self.rank_suggestions = {
                'energy_threshold': {'k': 6, 'confidence': 0.85},
                'knee_elbow': {'k': 3, 'confidence': 0.80}
            }
            self.frob_sq = frob_sq
            self.params = 1000
            self.utilization = 0.5
    
    name_mapping = {'energy_threshold': 'energy@0.90', 'knee_elbow': 'knee'}
    layers = [MockLayer("layer1", 10.0), MockLayer("layer2", 5.0)]
    
    test_configs = [
        {
            'name': 'Default configuration',
            'config': None
        },
        {
            'name': 'Conservative configuration',
            'config': {'quantile_threshold': 0.90, 'uniform_mult_gate': 2.0, 'metric': 'energy_share'}
        },
        {
            'name': 'Aggressive configuration', 
            'config': {'quantile_threshold': 0.50, 'uniform_mult_gate': 1.2, 'metric': 'frobenius_norm'}
        }
    ]
    
    for test_case in test_configs:
        print(f"\n{test_case['name']}:")
        analysis = _analyze_policy_disagreements(layers, name_mapping, test_case['config'], "full")
        
        config = analysis["disagreement_config"]
        
        if test_case['config'] is None:
            # Default values
            expected_quantile = 0.75
            expected_gate = 1.5
            expected_metric = 'energy_share'
        else:
            expected_quantile = test_case['config']['quantile_threshold']
            expected_gate = test_case['config']['uniform_mult_gate']
            expected_metric = test_case['config']['metric']
        
        print(f"  Expected quantile: {expected_quantile}, Got: {config['quantile_threshold']}")
        print(f"  Expected gate: {expected_gate}, Got: {config['uniform_mult_gate']}")
        print(f"  Expected metric: {expected_metric}, Got: {config['importance_metric']}")
        
        assert config['quantile_threshold'] == expected_quantile, "Quantile threshold mismatch"
        assert config['uniform_mult_gate'] == expected_gate, "Gate threshold mismatch"
        assert config['importance_metric'] == expected_metric, "Metric mismatch"
        
        print(f"  ✓ Configuration captured correctly")


def test_edge_case_schemas():
    """Test schema versioning for edge cases."""
    
    print(f"\n\n🧪 Testing Edge Case Schemas")
    print("=" * 70)
    
    name_mapping = {'energy_threshold': 'energy@0.90', 'knee_elbow': 'knee'}
    
    # Test 1: No layers
    print(f"\nTest 1: No layers")
    analysis = _analyze_policy_disagreements([], name_mapping, None, "full")
    
    assert analysis["schema_version"] == 1, "Schema version missing for no layers case"
    assert analysis["analysis_performed"] == False, "Analysis performed should be False"
    assert analysis["reason"] == "no_layers", "Reason should be no_layers"
    assert "computed_with" in analysis, "Version info missing for no layers case"
    print(f"  ✓ Schema correctly versioned for no layers case")
    
    # Test 2: No disagreements
    print(f"\nTest 2: No disagreements")
    class MockLayer:
        def __init__(self, name):
            self.name = name
            self.rank_suggestions = {
                'energy_threshold': {'k': 4, 'confidence': 0.85},
                'knee_elbow': {'k': 4, 'confidence': 0.80}  # Same k = no disagreement
            }
            self.frob_sq = 1.0
            self.params = 1000
            self.utilization = 0.5
    
    consensus_layers = [MockLayer("consensus_layer")]
    analysis = _analyze_policy_disagreements(consensus_layers, name_mapping, None, "full")
    
    assert analysis["schema_version"] == 1, "Schema version missing for no disagreements case"
    # For consensus layers, analysis is performed but no layers are flagged
    assert analysis["analysis_performed"] == True, "Analysis should be performed"
    # The layer is processed but has spread=0, so it doesn't meet the spread threshold
    assert len(analysis["all_layers_with_disagreement"]) == 1, "Should process the layer"
    assert analysis["summary"]["layers_flagged_as_high_impact"] == 0, "Should flag 0 layers as high-impact"
    
    # Check that the layer didn't meet spread threshold
    layer_rationale = analysis["all_layers_with_disagreement"][0]["flagging_rationale"]
    assert layer_rationale["spread"] == 0, "Consensus layer should have spread=0"
    assert layer_rationale["meets_spread_threshold"] == False, "Should not meet spread threshold"
    print(f"  ✓ Schema correctly versioned for no disagreements case")


def show_example_schema():
    """Show a complete example of the versioned schema."""
    
    print(f"\n\n📄 Complete Schema Example")
    print("=" * 70)
    
    class MockLayer:
        def __init__(self, name, rank_suggestions, frob_sq=1.0):
            self.name = name
            self.rank_suggestions = rank_suggestions
            self.frob_sq = frob_sq
            self.params = 1000
            self.utilization = 0.5
    
    layers = [
        MockLayer("layer1", {
            'energy_threshold': {'k': 8, 'confidence': 0.90},
            'knee_elbow': {'k': 2, 'confidence': 0.85}
        }, frob_sq=25.0)
    ]
    
    name_mapping = {'energy_threshold': 'energy@0.90', 'knee_elbow': 'knee'}
    analysis = _analyze_policy_disagreements(layers, name_mapping, None, "full")
    
    # Show key parts of the schema
    print("Key schema sections:")
    print("\n1. Schema metadata:")
    print(f"   schema_version: {analysis['schema_version']}")
    print(f"   computed_at: {analysis['computed_at']}")
    
    print("\n2. Version information:")
    computed_with = analysis['computed_with']
    for key, value in computed_with.items():
        print(f"   {key}: {value}")
    
    print("\n3. Algorithm configuration:")
    config = analysis['disagreement_config']
    print(f"   algorithm_name: {config['algorithm_name']}")
    print(f"   config_capture_version: {config['config_capture_version']}")
    print(f"   quantile_threshold: {config['quantile_threshold']}")
    print(f"   uniform_mult_gate: {config['uniform_mult_gate']}")
    print(f"   spread_calculation_formula: {config['spread_calculation_formula']}")
    
    print("\n4. Analysis results:")
    print(f"   layers_with_disagreement: {analysis['summary']['layers_with_disagreement']}")
    print(f"   layers_flagged_as_high_impact: {analysis['summary']['layers_flagged_as_high_impact']}")
    print(f"   distribution_is_flat: {analysis['distribution']['is_flat']}")


if __name__ == '__main__':
    test_schema_versioning()
    test_configuration_combinations() 
    test_edge_case_schemas()
    show_example_schema()
    
    print("\n" + "=" * 70)
    print("✅ SCHEMA VERSIONING TESTS PASSED!")
    print("\n🔒 Complete configuration capture verified:")
    print("  • Schema version and timestamps for reproducibility")
    print("  • Gradience version and git SHA for traceability")  
    print("  • All algorithm parameters and thresholds captured")
    print("  • Configuration combinations handled correctly")
    print("  • Edge cases properly versioned")
    print("\n💡 Benefits:")
    print("  • Future algorithm changes won't break comparisons")
    print("  • Complete audit trail for debugging disagreement flagging")
    print("  • Reproducible analysis with exact parameter capture")
    print("  • Version-aware downstream tools can adapt to changes")