#!/usr/bin/env python3
"""
Complete example of the versioned schema output.

Shows the full JSON schema with all versioning, configuration capture,
and detailed flagging rationale for documentation purposes.
"""

import sys
import json
import time
sys.path.insert(0, '.')

from gradience.cli import _analyze_policy_disagreements

def demonstrate_complete_schema():
    """Show a complete example of the versioned schema."""
    
    print("📄 COMPLETE SCHEMA EXAMPLE")
    print("=" * 80)
    print("This demonstrates the full JSON output with schema versioning,")
    print("configuration capture, and detailed flagging rationale.")
    print("=" * 80)
    
    class MockLayer:
        def __init__(self, name, rank_suggestions, frob_sq=1.0, params=1000, utilization=0.5):
            self.name = name
            self.rank_suggestions = rank_suggestions
            self.frob_sq = frob_sq
            self.params = params
            self.utilization = utilization
    
    # Create a realistic scenario with different layer types
    layers = [
        # High-impact attention layer with strong disagreement
        MockLayer(
            "transformer.h.0.attn.q_proj",
            {
                'energy_threshold': {'k': 8, 'confidence': 0.90},
                'knee_elbow': {'k': 2, 'confidence': 0.85},
                'entropy_effective': {'k': 6, 'confidence': 0.90}
            },
            frob_sq=45.0,    # High energy
            params=50000,
            utilization=0.8
        ),
        
        # Medium-impact MLP layer with moderate disagreement
        MockLayer(
            "transformer.h.1.mlp.c_fc",
            {
                'energy_threshold': {'k': 6, 'confidence': 0.85},
                'knee_elbow': {'k': 3, 'confidence': 0.80},
                'entropy_effective': {'k': 5, 'confidence': 0.85}
            },
            frob_sq=20.0,    # Medium energy  
            params=30000,
            utilization=0.6
        ),
        
        # Low-impact norm layer with disagreement
        MockLayer(
            "transformer.h.2.ln_1",
            {
                'energy_threshold': {'k': 4, 'confidence': 0.80},
                'knee_elbow': {'k': 1, 'confidence': 0.75},
                'entropy_effective': {'k': 3, 'confidence': 0.80}
            },
            frob_sq=2.5,     # Low energy
            params=1000,
            utilization=0.3
        ),
        
        # Consensus layer (no disagreement)
        MockLayer(
            "transformer.h.3.ln_2",
            {
                'energy_threshold': {'k': 4, 'confidence': 0.80},
                'knee_elbow': {'k': 4, 'confidence': 0.80},  # Same k = consensus
                'entropy_effective': {'k': 4, 'confidence': 0.80}
            },
            frob_sq=1.0,     # Very low energy
            params=500,
            utilization=0.2
        )
    ]
    
    name_mapping = {
        'energy_threshold': 'energy@0.90',
        'knee_elbow': 'knee',
        'entropy_effective': 'erank'
    }
    
    # Use custom configuration for demonstration
    custom_config = {
        'quantile_threshold': 0.60,  # More aggressive than default
        'uniform_mult_gate': 1.8,    # Stricter than default
        'metric': 'energy_share'
    }
    
    print("Configuration used:")
    print(f"  • quantile_threshold: {custom_config['quantile_threshold']} (60th percentile)")
    print(f"  • uniform_mult_gate: {custom_config['uniform_mult_gate']} (1.8× uniform threshold)")
    print(f"  • importance_metric: {custom_config['metric']}")
    print()
    
    # Generate the analysis
    print("Generating analysis...")
    start_time = time.time()
    analysis = _analyze_policy_disagreements(layers, name_mapping, custom_config, "full")
    analysis_time = time.time() - start_time
    
    print(f"Analysis completed in {analysis_time:.3f}s")
    print()
    
    # Show the complete JSON structure
    print("COMPLETE JSON OUTPUT:")
    print("=" * 50)
    
    # Pretty print the JSON
    json_output = json.dumps(analysis, indent=2)
    print(json_output)
    
    print("\n" + "=" * 80)
    
    # Highlight key benefits of the schema
    print("🔒 SCHEMA BENEFITS:")
    print()
    print("1. **Reproducibility**: Every configuration parameter captured")
    print(f"   • Algorithm: {analysis['disagreement_config']['algorithm_name']}")
    print(f"   • Schema version: {analysis['schema_version']}")
    print(f"   • Computed at: {analysis['computed_at']}")
    print()
    
    print("2. **Traceability**: Version information for debugging")
    computed_with = analysis['computed_with']
    print(f"   • Gradience version: {computed_with['gradience_version']}")
    if computed_with.get('git_sha'):
        print(f"   • Git SHA: {computed_with['git_sha']}")
    print()
    
    print("3. **Complete Configuration Capture**: No hidden parameters")
    config = analysis['disagreement_config']
    print(f"   • Core thresholds: quantile={config['quantile_threshold']}, gate={config['uniform_mult_gate']}")
    print(f"   • Algorithm flags: flat_detection={config['flat_detection_enabled']}")
    print(f"   • Spread formula: {config['spread_calculation_formula']}")
    print()
    
    print("4. **Detailed Rationale**: Every layer explains its flagging decision")
    flagged_count = analysis['summary']['layers_flagged_as_high_impact']
    total_count = analysis['summary']['layers_with_disagreement']
    print(f"   • {total_count} layers processed, {flagged_count} flagged as high-impact")
    
    if analysis['flagged_layers']:
        example_layer = analysis['flagged_layers'][0]
        rationale = example_layer['flagging_rationale']
        print(f"   • Example: {example_layer['layer_name']}")
        print(f"     - Spread: {rationale['spread']} ≥ {rationale['spread_threshold']} ✓")
        print(f"     - Uniform mult: {rationale['uniform_mult']:.2f} ≥ {rationale['uniform_mult_threshold']} ✓")
        print(f"     - Energy share: meaningful fraction ✓")
        print(f"     - Result: flagged_as_high_impact = {rationale['flagged_as_high_impact']}")
    
    print()
    print("5. **Future-Proof**: Schema evolution handled gracefully")
    print(f"   • Schema version {analysis['schema_version']} enables compatibility checks")
    print(f"   • Config version {config['config_capture_version']} tracks parameter evolution")
    print(f"   • Algorithm name tracks implementation changes")
    
    return analysis


def show_comparison_scenario():
    """Show how schema versioning prevents comparison bugs."""
    
    print("\n\n🚨 PREVENTING COMPARISON BUGS")
    print("=" * 80)
    print("Example: Two teams analyze the same model with different configurations")
    print("Without schema versioning → mysterious differences, debugging nightmare")
    print("With schema versioning → clear understanding of why results differ")
    print("=" * 80)
    
    # Same layers, different configs
    class MockLayer:
        def __init__(self, name, frob_sq=10.0):
            self.name = name
            self.rank_suggestions = {
                'energy_threshold': {'k': 7, 'confidence': 0.85},
                'knee_elbow': {'k': 3, 'confidence': 0.80}
            }
            self.frob_sq = frob_sq
            self.params = 1000
            self.utilization = 0.5
    
    layers = [MockLayer("test_layer", 10.0), MockLayer("test_layer2", 8.0)]
    name_mapping = {'energy_threshold': 'energy@0.90', 'knee_elbow': 'knee'}
    
    # Team A: Conservative configuration
    config_a = {'quantile_threshold': 0.90, 'uniform_mult_gate': 2.0, 'metric': 'energy_share'}
    analysis_a = _analyze_policy_disagreements(layers, name_mapping, config_a, "full")
    
    # Team B: Aggressive configuration  
    config_b = {'quantile_threshold': 0.50, 'uniform_mult_gate': 1.2, 'metric': 'energy_share'}
    analysis_b = _analyze_policy_disagreements(layers, name_mapping, config_b, "full")
    
    print("TEAM A RESULTS:")
    print(f"  Configuration: {analysis_a['disagreement_config']['quantile_threshold']:.1f} quantile, {analysis_a['disagreement_config']['uniform_mult_gate']:.1f}× gate")
    print(f"  Flagged layers: {analysis_a['summary']['layers_flagged_as_high_impact']}")
    print(f"  Computed at: {analysis_a['computed_at']}")
    
    print("\nTEAM B RESULTS:")
    print(f"  Configuration: {analysis_b['disagreement_config']['quantile_threshold']:.1f} quantile, {analysis_b['disagreement_config']['uniform_mult_gate']:.1f}× gate") 
    print(f"  Flagged layers: {analysis_b['summary']['layers_flagged_as_high_impact']}")
    print(f"  Computed at: {analysis_b['computed_at']}")
    
    print(f"\n💡 INSIGHT: Different results due to different configurations!")
    print(f"   Without schema versioning: 'Why do we get different answers?'")
    print(f"   With schema versioning: 'Team A used conservative thresholds, Team B used aggressive'")
    print(f"   → Clear understanding, no debugging mystery")


if __name__ == '__main__':
    analysis = demonstrate_complete_schema()
    show_comparison_scenario()
    
    print("\n" + "=" * 80)
    print("✅ SCHEMA VERSIONING IMPLEMENTATION COMPLETE!")
    print("\nThis transforms disagreement detection from 'black box algorithm'")
    print("to 'fully traceable decision system' with:")
    print()
    print("🔒 **Complete Reproducibility**")
    print("  • Every parameter, threshold, and flag captured")
    print("  • Exact algorithm version and implementation tracked")
    print("  • Timestamp and environment information included")
    print()
    print("🚨 **Bug Prevention**") 
    print("  • No more 'why do two audits give different results?' mysteries")
    print("  • Configuration differences immediately visible")
    print("  • Algorithm changes tracked and versioned")
    print()
    print("🤖 **Machine-Readable Rationale**")
    print("  • Every flagging decision explained with metrics")
    print("  • Downstream tools can filter and analyze systematically") 
    print("  • Perfect for Bench integration and automation")
    print()
    print("🔮 **Future-Proof**")
    print("  • Schema versioning enables backward compatibility")
    print("  • Algorithm evolution won't break existing tools")
    print("  • Configuration parameter additions handled gracefully")
    print()
    print("This is the difference between 'smart' and 'debuggable smart' - ")
    print("exactly what makes Gradience feel trustworthy! 🎯")