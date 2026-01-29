#!/usr/bin/env python3
"""
Test script for Step 4 - Wire policies into audit pipeline.

Creates a synthetic LoRA adapter and tests that the new rank selection 
policies are correctly integrated into the per-layer audit loop.
"""

import torch
import numpy as np
import json
from gradience.vnext.audit.lora_audit import audit_lora_state_dict

def create_test_lora_adapter():
    """Create a synthetic LoRA adapter for testing."""
    
    # Create test LoRA weights with different characteristics
    state_dict = {}
    
    # Layer 1: Clear low-rank structure (should suggest low rank)
    r1 = 8
    A1 = torch.randn(r1, 512) * 0.1
    # Make it clearly rank-3 by zeroing out most singular values
    U, s, Vt = torch.linalg.svd(A1, full_matrices=False)
    s[3:] = s[3:] * 0.01  # Make higher modes very small
    A1 = (U * s.unsqueeze(0)) @ Vt
    
    B1 = torch.randn(768, r1) * 0.1
    
    state_dict['model.layers.0.self_attn.q_proj.lora_A.weight'] = A1
    state_dict['model.layers.0.self_attn.q_proj.lora_B.weight'] = B1
    
    # Layer 2: Gradual decay structure
    r2 = 8
    A2 = torch.randn(r2, 512) 
    U, s, Vt = torch.linalg.svd(A2, full_matrices=False)
    # Exponential decay
    s = torch.tensor([16.0, 8.0, 4.0, 2.0, 1.0, 0.5, 0.25, 0.125])
    A2 = (U * s.unsqueeze(0)) @ Vt
    
    B2 = torch.randn(768, r2) * 0.1
    
    state_dict['model.layers.1.mlp.up_proj.lora_A.weight'] = A2
    state_dict['model.layers.1.mlp.up_proj.lora_B.weight'] = B2
    
    # Layer 3: High noise (should suggest conservative rank)
    r3 = 8
    A3 = torch.randn(r3, 512)
    U, s, Vt = torch.linalg.svd(A3, full_matrices=False)
    # Two dominant, rest is noise
    s = torch.tensor([15.0, 8.0, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05])
    A3 = (U * s.unsqueeze(0)) @ Vt
    
    B3 = torch.randn(768, r3) * 0.1
    
    state_dict['model.layers.2.self_attn.v_proj.lora_A.weight'] = A3
    state_dict['model.layers.2.self_attn.v_proj.lora_B.weight'] = B3
    
    return state_dict

def test_policy_integration():
    """Test that policies are correctly integrated into audit pipeline."""
    
    print("🧪 Step 4 - Testing Policy Integration in Audit Pipeline")
    print("=" * 60)
    
    # Create test adapter
    print("Creating synthetic LoRA adapter...")
    state_dict = create_test_lora_adapter()
    print(f"Created adapter with {len(state_dict)//2} LoRA layers")
    print()
    
    # Test with different policy combinations
    test_cases = [
        ("No policies", None),
        ("OHT only", ["optimal_hard_threshold"]),
        ("Entropy only", ["entropy_effective"]),
        ("Knee only", ["knee_elbow"]),
        ("All policies", ["optimal_hard_threshold", "entropy_effective", "knee_elbow"])
    ]
    
    for name, policies in test_cases:
        print(f"📊 Test Case: {name}")
        print("-" * 40)
        
        # Run audit
        try:
            result = audit_lora_state_dict(
                state_dict,
                rank_policies=policies,
                include_top_singular_values=5
            )
            
            print(f"✅ Audit completed successfully")
            print(f"   Layers audited: {result.n_layers}")
            print(f"   Total parameters: {result.total_lora_params:,}")
            print()
            
            # Check rank suggestions for first layer
            if result.layers:
                layer = result.layers[0] 
                print(f"📈 Layer: {layer.name}")
                print(f"   Shape: {layer.a_shape} × {layer.b_shape}")
                print(f"   Rank: {layer.r}")
                print(f"   Energy@90: {layer.energy_rank_90}")
                
                if layer.rank_suggestions:
                    print(f"   Rank Suggestions:")
                    for policy, data in layer.rank_suggestions.items():
                        if policy == 'error':
                            print(f"     ❌ Error: {data}")
                        else:
                            k = data.get('k', 'N/A')
                            conf = data.get('confidence', 0.0)
                            print(f"     {policy}: k={k}, conf={conf:.2f}")
                            
                            # Show policy-specific details
                            if policy == 'oht' and 'tau' in data:
                                print(f"        tau={data['tau']:.4f}, omega={data['omega']:.2f}, beta={data['beta']:.3f}")
                            elif policy == 'erank' and 'erank' in data:
                                print(f"        erank={data['erank']:.2f}, entropy={data['entropy']:.2f}")
                            elif policy == 'knee' and 'score' in data:
                                print(f"        score={data['score']:.3f}")
                else:
                    print(f"   ❌ No rank suggestions found")
                print()
                
        except Exception as e:
            print(f"❌ Audit failed: {e}")
            import traceback
            traceback.print_exc()
            print()
    
    # Test schema compliance
    print("🔍 Schema Compliance Test")
    print("-" * 40)
    
    result = audit_lora_state_dict(
        state_dict, 
        rank_policies=["optimal_hard_threshold", "entropy_effective", "knee_elbow"]
    )
    
    # Check that rank_suggestions matches expected schema
    if result.layers:
        layer = result.layers[0]
        rank_suggestions = layer.rank_suggestions
        
        print("Expected schema example:")
        expected = {
            "energy_90": {"k": 12},
            "oht": {"k": 8, "tau": 0.013, "omega": 2.43, "beta": 0.70},
            "erank": {"k": 9, "erank": 8.4, "entropy": 2.13},
            "knee": {"k": 10, "score": 0.22}
        }
        print(json.dumps(expected, indent=2))
        print()
        
        print("Actual rank_suggestions:")
        if rank_suggestions:
            print(json.dumps(rank_suggestions, indent=2))
            
            # Check required keys
            required_policies = ['energy_90', 'oht', 'erank', 'knee'] 
            for policy in required_policies:
                if policy in rank_suggestions:
                    data = rank_suggestions[policy]
                    if 'k' in data:
                        print(f"✅ {policy}: has required 'k' field")
                    else:
                        print(f"❌ {policy}: missing 'k' field")
                else:
                    print(f"❌ Missing policy: {policy}")
        else:
            print("❌ No rank_suggestions found")
    
    print()
    print("🎯 Step 4 Integration Test Complete!")

if __name__ == "__main__":
    test_policy_integration()