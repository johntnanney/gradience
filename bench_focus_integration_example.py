#!/usr/bin/env python3
"""
Example: Bench Focus Integration

Shows how Bench can consume disagreement_focus_set directly from audit
to restrict per-layer validation to only critical layers.

This is the practical win: you don't pay per-layer complexity unless justified.
"""

import json

def demonstrate_focus_consumption():
    """Show how Bench would consume the focus set from audit."""
    
    print("⚡ BENCH FOCUS INTEGRATION EXAMPLE")
    print("=" * 70)
    
    # Simulate audit result with focus set (from our previous implementation)
    audit_result = {
        "schema_version": 1,
        "computed_at": "2026-01-28T17:15:00Z",
        "disagreement_focus_set": {
            "high_impact_layers": [
                "model.layers.0.self_attn.q_proj",
                "model.layers.1.mlp.up_proj"
            ],
            "recommended_focus_n": 2,
            "focus_strategy": "multiple_layers",
            "message": "Focus Bench validation on 2 energy-significant layers (highest priority: model.layers.0.self_attn.q_proj)",
            "distribution_type": "hierarchical"
        },
        "per_layer_suggestions": {
            "rank_pattern": {
                "model.layers.0.self_attn.q_proj": 8,
                "model.layers.1.mlp.up_proj": 6,
                "model.layers.2.norm1": 3,
                "model.layers.3.norm2": 2,
                "model.layers.4.dropout": 1,
                "model.layers.5.final_layer": 1
            }
        }
    }
    
    print("🔍 Original audit suggests ranks for ALL 6 layers:")
    original_ranks = audit_result["per_layer_suggestions"]["rank_pattern"]
    for layer, rank in original_ranks.items():
        print(f"  {layer}: rank={rank}")
    
    print(f"\\n🎯 Focus set recommends limiting to {audit_result['disagreement_focus_set']['recommended_focus_n']} critical layers:")
    for layer in audit_result['disagreement_focus_set']['high_impact_layers']:
        print(f"  {layer}: rank={original_ranks[layer]}")
    
    # Demonstrate the integration logic
    print(f"\\n🚀 BENCH INTEGRATION LOGIC:")
    print("=" * 50)
    
    # Original approach (before focus integration)
    print("📊 BEFORE (traditional approach):")
    print("```python")
    print("# Traditional per_layer variant generation")
    print("if per_layer_suggestions:")
    print("    rank_pattern = per_layer_suggestions['rank_pattern']")
    print("    # Use ALL suggested layers")
    print("    compression_configs['per_layer'] = {")
    print("        'variant': 'per_layer',")
    print("        'rank_pattern': rank_pattern  # All 6 layers")
    print("    }")
    print("```")
    print(f"Result: Bench tests per-layer optimization on ALL {len(original_ranks)} layers")
    print("Cost: High complexity, many variants to validate")
    
    # New focus-based approach
    print(f"\\n⚡ AFTER (focus-based approach):")
    print("```python")
    print("# Focus-based per_layer variant generation")
    print("def create_focused_per_layer_variant(audit_data):")
    print("    focus_set = audit_data.get('disagreement_focus_set', {})")
    print("    per_layer_suggestions = audit_data.get('per_layer_suggestions', {})")
    print("    ")
    print("    if not focus_set or not per_layer_suggestions:")
    print("        # Fall back to traditional approach")
    print("        return create_traditional_per_layer_variant(audit_data)")
    print("    ")
    print("    high_impact_layers = focus_set.get('high_impact_layers', [])")
    print("    full_rank_pattern = per_layer_suggestions.get('rank_pattern', {})")
    print("    ")
    print("    if not high_impact_layers:")
    print("        # No high-impact layers - fall back to uniform")
    print("        return None  # Skip per_layer variant")
    print("    ")
    print("    # Restrict rank_pattern to only high-impact layers")
    print("    focused_rank_pattern = {")
    print("        layer: full_rank_pattern[layer]")
    print("        for layer in high_impact_layers")
    print("        if layer in full_rank_pattern")
    print("    }")
    print("    ")
    print("    return {")
    print("        'variant': 'per_layer',")
    print("        'rank_pattern': focused_rank_pattern,")
    print("        'focus_strategy': focus_set.get('focus_strategy'),")
    print("        'focus_justification': focus_set.get('message')")
    print("    }")
    print("```")
    
    # Show the practical result
    focus_set = audit_result['disagreement_focus_set']
    high_impact_layers = focus_set['high_impact_layers']
    full_rank_pattern = audit_result['per_layer_suggestions']['rank_pattern']
    
    # Apply the focus logic
    focused_rank_pattern = {
        layer: full_rank_pattern[layer]
        for layer in high_impact_layers
        if layer in full_rank_pattern
    }
    
    print(f"\\nResult: Bench tests per-layer optimization on only {len(focused_rank_pattern)} critical layers")
    print("Cost: Reduced complexity, focused validation")
    print(f"Focus justification: '{focus_set['message']}'")
    
    print(f"\\n📈 IMPACT COMPARISON:")
    print("=" * 50)
    print(f"Traditional approach:")
    print(f"  • Layers tested: {len(original_ranks)}")
    print(f"  • Validation complexity: HIGH (test every layer)")
    print(f"  • Risk: Waste time on unimportant layers")
    
    print(f"\\nFocus-based approach:")
    print(f"  • Layers tested: {len(focused_rank_pattern)}")
    print(f"  • Validation complexity: LOW (test only critical layers)")
    print(f"  • Benefit: Focus effort where it matters most")
    print(f"  • Reduction: {((len(original_ranks) - len(focused_rank_pattern)) / len(original_ranks)) * 100:.0f}% fewer layers to validate")
    
    # Show the focused configuration
    print(f"\\n🎯 FOCUSED PER-LAYER CONFIGURATION:")
    print(f"```json")
    focused_config = {
        "variant": "per_layer",
        "rank_pattern": focused_rank_pattern,
        "focus_strategy": focus_set['focus_strategy'],
        "focus_justification": focus_set['message'],
        "layers_skipped": len(original_ranks) - len(focused_rank_pattern)
    }
    print(json.dumps(focused_config, indent=2))
    print(f"```")
    
    return {
        'original_layer_count': len(original_ranks),
        'focused_layer_count': len(focused_rank_pattern),
        'reduction_pct': ((len(original_ranks) - len(focused_rank_pattern)) / len(original_ranks)) * 100,
        'focus_strategy': focus_set['focus_strategy']
    }


def show_edge_cases():
    """Show how focus integration handles edge cases."""
    
    print(f"\\n\\n🧪 EDGE CASE HANDLING")
    print("=" * 70)
    
    edge_cases = [
        {
            'name': 'Flat Distribution',
            'focus_set': {
                'high_impact_layers': ['layer_1', 'layer_2', 'layer_3'],
                'recommended_focus_n': 3,
                'focus_strategy': 'top_disagreement_priority',
                'distribution_type': 'flat',
                'message': 'Energy distribution is flat. Consider top 3 disagreement layers.'
            },
            'expected_behavior': 'Restrict to top 3 disagreement layers by priority'
        },
        {
            'name': 'No High-Impact Layers',
            'focus_set': {
                'high_impact_layers': [],
                'recommended_focus_n': 0,
                'focus_strategy': 'none',
                'distribution_type': 'hierarchical',
                'message': 'No layers meet high-impact criteria.'
            },
            'expected_behavior': 'Skip per_layer variant, fall back to uniform'
        },
        {
            'name': 'Single Critical Layer',
            'focus_set': {
                'high_impact_layers': ['critical_layer'],
                'recommended_focus_n': 1,
                'focus_strategy': 'single_layer',
                'distribution_type': 'hierarchical',
                'message': 'Focus on 1 energy-significant layer: critical_layer'
            },
            'expected_behavior': 'Create minimal per_layer variant with 1 layer'
        }
    ]
    
    for case in edge_cases:
        print(f"\\n{case['name']}:")
        focus_set = case['focus_set']
        
        print(f"  Focus strategy: {focus_set['focus_strategy']}")
        print(f"  High-impact layers: {focus_set['high_impact_layers']}")
        print(f"  Expected behavior: {case['expected_behavior']}")
        
        # Show integration logic for this case
        if focus_set['focus_strategy'] == 'none':
            print("  🔄 Integration: Skip per_layer variant entirely")
            print("     → Fall back to uniform suggestions only")
        elif focus_set['focus_strategy'] == 'single_layer':
            print("  🎯 Integration: Minimal per_layer with 1 layer")
            print("     → Lowest complexity while still testing per-layer concept")
        elif focus_set['focus_strategy'] == 'top_disagreement_priority':
            print("  📊 Integration: Focus on top disagreement layers")
            print("     → Even in flat distributions, test most promising layers")
        else:
            print("  ⚡ Integration: Standard focused per_layer variant")
            print("     → Restrict to high-impact layers only")


def show_practical_integration():
    """Show the exact code changes needed in Bench protocol.py."""
    
    print(f"\\n\\n🔧 PRACTICAL INTEGRATION: EXACT CODE CHANGES")
    print("=" * 80)
    
    print("📍 Location: gradience/bench/protocol.py, around line 1202")
    print()
    print("🔴 CURRENT CODE (traditional approach):")
    print("```python")
    print("per_layer_suggestions = audit_data.get('per_layer_suggestions')")
    print("if per_layer_suggestions and (not fast_mode or len(filtered_candidates) < max_candidates):")
    print("    rank_pattern = per_layer_suggestions['rank_pattern']")
    print("    ")
    print("    # Clamp ranks to allowed values")
    print("    clamped_rank_pattern = {}")
    print("    for module_name, suggested_r in rank_pattern.items():")
    print("        # ... clamping logic ...")
    print("    ")
    print("    compression_configs['per_layer'] = {")
    print("        'variant': 'per_layer',")
    print("        'rank_pattern': clamped_rank_pattern")
    print("    }")
    print("```")
    
    print("\\n🟢 NEW CODE (focus-based approach):")
    print("```python")
    print("per_layer_suggestions = audit_data.get('per_layer_suggestions')")
    print("focus_set = audit_data.get('disagreement_focus_set', {})")
    print("")
    print("if per_layer_suggestions and (not fast_mode or len(filtered_candidates) < max_candidates):")
    print("    full_rank_pattern = per_layer_suggestions['rank_pattern']")
    print("    high_impact_layers = focus_set.get('high_impact_layers', [])")
    print("    ")
    print("    # Apply focus filtering if available")
    print("    if high_impact_layers and focus_set.get('focus_strategy') != 'none':")
    print("        # Restrict to high-impact layers only")
    print("        rank_pattern = {")
    print("            layer: full_rank_pattern[layer]")
    print("            for layer in high_impact_layers")
    print("            if layer in full_rank_pattern")
    print("        }")
    print("        focus_justification = focus_set.get('message', 'No justification')")
    print("    else:")
    print("        # Fall back to traditional approach")
    print("        rank_pattern = full_rank_pattern")
    print("        focus_justification = 'Traditional per-layer (no focus available)'")
    print("    ")
    print("    if not rank_pattern:")
    print("        # No valid layers to test - skip per_layer variant")
    print("        continue")
    print("    ")
    print("    # Clamp ranks to allowed values (same as before)")
    print("    clamped_rank_pattern = {}")
    print("    for module_name, suggested_r in rank_pattern.items():")
    print("        # ... existing clamping logic ...")
    print("    ")
    print("    compression_configs['per_layer'] = {")
    print("        'variant': 'per_layer',")
    print("        'rank_pattern': clamped_rank_pattern,")
    print("        'focus_strategy': focus_set.get('focus_strategy', 'traditional'),")
    print("        'focus_justification': focus_justification,")
    print("        'layers_tested': len(clamped_rank_pattern),")
    print("        'layers_skipped': len(full_rank_pattern) - len(rank_pattern)")
    print("    }")
    print("```")
    
    print("\\n🎯 KEY BENEFITS:")
    print("  • Backward compatible: Works with or without focus_set")
    print("  • Significant speedup: Only test layers that matter")
    print("  • Clear justification: Each run explains its focus strategy")
    print("  • Fallback safety: Traditional approach if focus unavailable")
    print("  • Audit trail: Track how many layers were skipped and why")
    
    print("\\n⚡ PRACTICAL IMPACT:")
    print("  • Large models: Focus on 2-3 critical layers instead of 50+")
    print("  • Faster Bench runs: Less per-layer complexity means faster validation")
    print("  • Better focus: Spend validation effort where disagreement analysis says it matters")
    print("  • Cost efficiency: Don't pay per-layer cost unless justified by importance")


if __name__ == '__main__':
    print("⚡ BENCH FOCUS INTEGRATION")
    print("=" * 80)
    print("Demonstrating how Bench consumes disagreement_focus_set directly")
    print("to restrict per-layer validation to only critical layers.")
    print("=" * 80)
    
    # Main demonstration
    results = demonstrate_focus_consumption()
    
    # Edge case handling
    show_edge_cases()
    
    # Practical integration guide
    show_practical_integration()
    
    print("\\n" + "=" * 80)
    print("✅ BENCH FOCUS INTEGRATION COMPLETE!")
    print()
    print("🎯 Summary:")
    print(f"  • Reduction in validation complexity: {results['reduction_pct']:.0f}%")
    print(f"  • Focus strategy: {results['focus_strategy']}")
    print(f"  • Layers tested: {results['focused_layer_count']} vs {results['original_layer_count']} (traditional)")
    print()
    print("⚡ Big Practical Win:")
    print("  • Audit does the hard work of finding critical layers")
    print("  • Bench consumes focus set directly (no re-derivation)")
    print("  • Per-layer complexity only applied where justified")
    print("  • Faster benchmarks, better resource utilization")
    print()
    print("🚀 Next Steps for Integration:")
    print("  1. Add focus consumption logic to protocol.py (line ~1202)")
    print("  2. Update per_layer variant generation to use high_impact_layers")
    print("  3. Add focus metrics to bench reports (layers_tested vs layers_available)")
    print("  4. Test with real audit data to validate focus effectiveness")
    print()
    print("This transforms audit → Bench from 'analyze everything' to")
    print("'analyze what matters' - exactly the efficiency win we need! 🎯")