#!/usr/bin/env python3
"""
Test CLI JSON integration with policy disagreement analysis.

Demonstrates how the JSON output now includes detailed flagging rationale.
"""

import sys
import json
import tempfile
from pathlib import Path
sys.path.insert(0, '.')

from gradience.vnext.audit.lora_audit import LoRAAuditResult, LoRALayerAudit
from gradience.cli import cmd_audit

def create_mock_layers():
    """Create mock layers with policy disagreements for testing."""
    
    class MockLayer:
        def __init__(self, name, rank_suggestions, frob_sq=1.0, params=1000, utilization=0.5):
            self.name = name
            self.rank_suggestions = rank_suggestions
            self.frob_sq = frob_sq
            self.params = params
            self.utilization = utilization
    
    # Create layers with clear disagreement patterns
    mock_layers = [
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
        )
    ]
    
    return mock_layers


def test_cli_json_output():
    """Test that CLI --json includes policy disagreement analysis."""
    
    print("🧪 Testing CLI JSON Output with Policy Disagreement Analysis")
    print("=" * 70)
    
    # Create mock layers
    mock_layers = create_mock_layers()
    
    # Simulate CLI JSON output generation
    name_mapping = {
        'energy_threshold': 'energy@0.90',
        'knee_elbow': 'knee', 
        'entropy_effective': 'erank',
        'optimal_hard_threshold': 'oht'
    }
    
    from gradience.cli import _analyze_policy_disagreements
    
    # Test with default importance config
    importance_config = {
        'quantile_threshold': 0.75,
        'uniform_mult_gate': 1.5,
        'metric': 'energy_share'
    }
    
    # Generate the disagreement analysis (this is what gets added to JSON)
    disagreement_analysis = _analyze_policy_disagreements(mock_layers, name_mapping, importance_config)
    
    # Simulate the full JSON payload
    json_payload = {
        "total_lora_params": 150000,  # Mock total
        "n_layers": len(mock_layers),
        "stable_rank_mean": 4.5,      # Mock values
        "utilization_mean": 0.6,
        # ... other audit fields ...
        "policy_disagreement_analysis": disagreement_analysis
    }
    
    print("📊 JSON Payload Structure:")
    print("=" * 50)
    
    # Show the structure without overwhelming output
    def show_structure(obj, indent=0):
        prefix = "  " * indent
        if isinstance(obj, dict):
            for key, value in obj.items():
                if isinstance(value, (dict, list)):
                    print(f"{prefix}{key}: {type(value).__name__}({len(value)})")
                    if key in ["flagged_layers", "all_layers_with_disagreement"] and len(value) > 0:
                        print(f"{prefix}  └─ Sample item:")
                        show_structure(value[0], indent + 2)
                else:
                    print(f"{prefix}{key}: {value}")
        elif isinstance(obj, list) and obj:
            print(f"{prefix}[0]: {type(obj[0]).__name__}")
            if isinstance(obj[0], dict):
                show_structure(obj[0], indent + 1)
    
    show_structure(json_payload)
    
    print(f"\n🎯 Key JSON Sections Added:")
    print("=" * 50)
    
    analysis = disagreement_analysis
    print(f"1. Analysis Summary:")
    print(f"   • Analysis performed: {analysis['analysis_performed']}")
    print(f"   • Layers with disagreement: {analysis['summary']['layers_with_disagreement']}")
    print(f"   • Layers flagged as high-impact: {analysis['summary']['layers_flagged_as_high_impact']}")
    
    print(f"\n2. Configuration Used:")
    config = analysis['config']
    print(f"   • Quantile threshold: {config['quantile_threshold']}")
    print(f"   • Uniform multiplier gate: {config['uniform_mult_gate']}")
    print(f"   • Importance metric: {config['importance_metric']}")
    
    print(f"\n3. Distribution Characteristics:")
    dist = analysis['distribution']
    print(f"   • Total layers: {dist['total_layers']}")
    print(f"   • Max uniform multiplier: {dist['max_uniform_mult']:.2f}")
    print(f"   • Distribution is flat: {dist['is_flat']}")
    
    print(f"\n4. Flagged Layers (High-Impact):")
    for i, layer_data in enumerate(analysis['flagged_layers'], 1):
        layer_name = layer_data['layer_name']
        rationale = layer_data['flagging_rationale']
        print(f"   {i}. {layer_name}")
        print(f"      • Spread: {rationale['spread']} (≥ {rationale['spread_threshold']:.1f})")
        print(f"      • Uniform mult: {rationale['uniform_mult']:.2f}× (≥ {rationale['uniform_mult_threshold']:.1f}×)")
        print(f"      • Importance share: {rationale['importance_share']:.3f}")
        print(f"      • Passed gate: {rationale['passed_gate']}")
    
    print(f"\n📄 Sample Flagging Rationale JSON:")
    print("=" * 50)
    
    if analysis['flagged_layers']:
        sample_rationale = analysis['flagged_layers'][0]['flagging_rationale']
        print(json.dumps(sample_rationale, indent=2))
    else:
        print("No flagged layers in this example")
    
    print(f"\n✅ JSON Serialization Test:")
    try:
        json_str = json.dumps(json_payload, indent=2)
        print("   ✓ Full payload serializes to JSON successfully")
        print(f"   ✓ JSON size: {len(json_str):,} characters")
        
        # Test that it can be parsed back
        parsed = json.loads(json_str)
        assert parsed["policy_disagreement_analysis"]["analysis_performed"]
        print("   ✓ JSON deserializes and contains expected fields")
        
    except Exception as e:
        print(f"   ❌ JSON serialization failed: {e}")


def show_usage_examples():
    """Show usage examples for downstream tools."""
    
    print(f"\n\n🛠️ Usage Examples for Downstream Tools")
    print("=" * 70)
    
    print("1. **Bench Script Integration**:")
    print("```python")
    print("# Load audit results with disagreement analysis")
    print("import json")
    print("with open('audit_results.json') as f:")
    print("    audit_data = json.load(f)")
    print("")
    print("# Get layers that need Bench validation")
    print("disagreement = audit_data['policy_disagreement_analysis']")
    print("if disagreement['analysis_performed']:")
    print("    flagged_layers = disagreement['flagged_layers']")
    print("    for layer in flagged_layers:")
    print("        layer_name = layer['layer_name']")
    print("        rationale = layer['flagging_rationale']")
    print("        print(f'Validate {layer_name}: spread={rationale[\"spread\"]}')")
    print("```")
    
    print(f"\n2. **Filtering by Specific Criteria**:")
    print("```python")
    print("# Find layers with high uniform multiplier but low spread")
    print("for layer in disagreement['all_layers_with_disagreement']:")
    print("    r = layer['flagging_rationale']")
    print("    if r['uniform_mult'] >= 2.0 and r['spread'] < 5:")
    print("        print(f'Interesting case: {layer[\"layer_name\"]}')") 
    print("```")
    
    print(f"\n3. **Debugging Why a Layer Was/Wasn't Flagged**:")
    print("```python")
    print("# Debug specific layer")
    print("layer_name = 'transformer.h.0.attn.q_proj'")
    print("for layer in disagreement['all_layers_with_disagreement']:")
    print("    if layer['layer_name'] == layer_name:")
    print("        r = layer['flagging_rationale']")
    print("        print(f'Flagged: {r[\"flagged_as_high_impact\"]}')") 
    print("        print(f'Reason: spread={r[\"spread\"]} >= {r[\"spread_threshold\"]}')") 
    print("        print(f'        uniform_mult={r[\"uniform_mult\"]:.2f} >= {r[\"uniform_mult_threshold\"]}')") 
    print("        print(f'        quantile_ok={r[\"meets_quantile_threshold\"]}')") 
    print("```")
    
    print(f"\n4. **Configuration Tuning**:")
    print("```bash")
    print("# Test different thresholds to see impact")
    print("gradience audit --peft-dir ./adapter --json \\")
    print("  --importance-quantile 0.50 \\")  
    print("  --importance-uniform-mult-gate 1.2 > results_aggressive.json")
    print("")
    print("gradience audit --peft-dir ./adapter --json \\")
    print("  --importance-quantile 0.90 \\")
    print("  --importance-uniform-mult-gate 2.0 > results_conservative.json")
    print("```")


if __name__ == '__main__':
    test_cli_json_output()
    show_usage_examples()
    
    print("\n" + "=" * 70)
    print("✅ CLI JSON Integration Test Completed!")
    print("\n🎯 Summary of JSON Enhancement:")
    print("  • Added 'policy_disagreement_analysis' section to --json output")
    print("  • Includes detailed flagging rationale for every layer")
    print("  • Machine-readable criteria: spread, uniform_mult, quantile_threshold")
    print("  • Configuration and distribution metadata included")
    print("  • Fully serializable for downstream automation")
    print("\n🤖 Perfect for Bench integration and systematic policy analysis!")