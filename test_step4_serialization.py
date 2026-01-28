#!/usr/bin/env python3
"""
Test serialization of rank_suggestions in audit output.
"""

import torch
import json
from gradience.vnext.audit.lora_audit import audit_lora_state_dict

def test_rank_suggestions_serialization():
    """Test that rank_suggestions appear correctly in serialized output."""
    
    print("🧪 Testing Rank Suggestions Serialization")
    print("=" * 50)
    
    # Create simple test case
    state_dict = {
        'test.lora_A.weight': torch.randn(4, 256),
        'test.lora_B.weight': torch.randn(512, 4)
    }
    
    # Run audit with all policies
    result = audit_lora_state_dict(
        state_dict, 
        rank_policies=["optimal_hard_threshold", "entropy_effective", "knee_elbow"]
    )
    
    print(f"Audited {result.n_layers} layers")
    
    # Test layer-level serialization
    if result.layers:
        layer = result.layers[0]
        layer_dict = layer.to_dict()
        
        print("\n📊 Layer Serialization Test:")
        print(f"Layer name: {layer_dict['name']}")
        print(f"Has rank_suggestions: {'rank_suggestions' in layer_dict}")
        
        if 'rank_suggestions' in layer_dict:
            rank_suggestions = layer_dict['rank_suggestions']
            print("\nRank Suggestions:")
            for policy, data in rank_suggestions.items():
                k = data.get('k', 'N/A')
                conf = data.get('confidence', 'N/A')
                print(f"  {policy}: k={k}, confidence={conf}")
        
        # Test summary-level serialization  
        print("\n📈 Summary Serialization Test:")
        summary = result.to_summary_dict(include_layers=True)
        
        print(f"Has layer_data: {'layer_data' in summary}")
        if 'layer_data' in summary and 'layer_rows' in summary['layer_data']:
            layer_rows = summary['layer_data']['layer_rows']
            if layer_rows:
                first_layer = layer_rows[0]
                print(f"First layer has rank_suggestions: {'rank_suggestions' in first_layer}")
                
                if 'rank_suggestions' in first_layer:
                    rs = first_layer['rank_suggestions']
                    print(f"Policies in rank_suggestions: {list(rs.keys())}")
        
        # Pretty print the full layer serialization
        print("\n🔍 Full Layer Serialization (rank_suggestions only):")
        if 'rank_suggestions' in layer_dict:
            print(json.dumps(layer_dict['rank_suggestions'], indent=2))
    
    print("\n✅ Serialization test complete!")

if __name__ == "__main__":
    test_rank_suggestions_serialization()