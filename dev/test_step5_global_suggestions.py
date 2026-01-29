#!/usr/bin/env python3
"""
Test script for Step 5 - Global policy suggestions computation.

Verifies that median/p90/max global suggestions are computed correctly
per policy across all layers.
"""

import torch
import numpy as np
import json
from gradience.vnext.audit.lora_audit import audit_lora_state_dict

def create_test_adapter_with_varied_ranks():
    """Create test adapter with layers having different rank characteristics."""
    
    state_dict = {}
    
    # Create layers with different rank patterns for testing
    test_layers = [
        # Layer, Singular values pattern, Expected ranks
        ("layer_0.q_proj", [10.0, 5.0, 2.0, 1.0, 0.5, 0.2, 0.1, 0.05], "low_rank"),
        ("layer_0.k_proj", [8.0, 6.0, 4.0, 3.0, 2.0, 1.0, 0.5, 0.2], "gradual_decay"),
        ("layer_1.v_proj", [15.0, 8.0, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05], "high_noise"),
        ("layer_1.o_proj", [6.0, 4.0, 2.0, 1.0, 0.5, 0.25, 0.125, 0.06], "exponential"),
        ("layer_2.gate", [12.0, 10.0, 8.0, 6.0, 4.0, 2.0, 1.0, 0.5], "rich_structure"),
        ("layer_2.up", [20.0, 2.0, 1.0, 0.8, 0.5, 0.3, 0.1, 0.05], "one_dominant"),
        ("layer_3.down", [5.0, 4.5, 4.0, 3.5, 3.0, 2.5, 2.0, 1.5], "almost_uniform"),
    ]
    
    for module_name, singular_values, description in test_layers:
        r = len(singular_values)
        
        # Create A and B matrices with controlled singular values
        A = torch.randn(r, 512)
        U, _, Vt = torch.linalg.svd(A, full_matrices=False) 
        s = torch.tensor(singular_values, dtype=torch.float32)
        A = (U * s.unsqueeze(0)) @ Vt
        
        B = torch.randn(768, r) * 0.1
        
        state_dict[f"{module_name}.lora_A.weight"] = A
        state_dict[f"{module_name}.lora_B.weight"] = B
    
    return state_dict, test_layers

def test_global_policy_suggestions():
    """Test that global policy suggestions are computed correctly."""
    
    print("🧪 Step 5 - Global Policy Suggestions Test")
    print("=" * 60)
    
    # Create test adapter
    state_dict, test_layers = create_test_adapter_with_varied_ranks()
    print(f"Created test adapter with {len(test_layers)} layers")
    print()
    
    # Run audit with all policies
    result = audit_lora_state_dict(
        state_dict,
        rank_policies=["optimal_hard_threshold", "entropy_effective", "knee_elbow"]
    )
    
    print(f"✅ Audit completed: {result.n_layers} layers processed")
    print()
    
    # Check that global suggestions were computed
    if result.policy_global_suggestions is None:
        print("❌ policy_global_suggestions is None")
        return
    
    print("📊 Policy Global Suggestions:")
    print("-" * 40)
    
    for policy_name, stats in result.policy_global_suggestions.items():
        print(f"{policy_name}:")
        print(f"  uniform_median = {stats['uniform_median']}")
        print(f"  uniform_p90    = {stats['uniform_p90']}")
        print(f"  uniform_max    = {stats['uniform_max']}")
        print(f"  n_layers       = {stats['n_layers']}")
        print()
    
    # Test individual policy collection
    print("🔍 Per-Layer Policy Values:")
    print("-" * 40)
    
    for policy in ['energy_90', 'oht', 'erank', 'knee']:
        if policy in result.policy_global_suggestions:
            values = []
            for layer in result.layers:
                if (layer.rank_suggestions and 
                    policy in layer.rank_suggestions and
                    'k' in layer.rank_suggestions[policy]):
                    k = layer.rank_suggestions[policy]['k']
                    values.append(k)
            
            print(f"{policy}: {values}")
            
            # Verify manual computation matches global computation
            if values:
                manual_median = float(np.percentile(values, 50))
                manual_p90 = float(np.percentile(values, 90))
                manual_max = float(max(values))
                
                computed = result.policy_global_suggestions[policy]
                
                print(f"  Manual:   median={manual_median}, p90={manual_p90}, max={manual_max}")
                print(f"  Computed: median={computed['uniform_median']}, p90={computed['uniform_p90']}, max={computed['uniform_max']}")
                
                # Check if they match (allow for small floating point differences)
                median_match = abs(manual_median - computed['uniform_median']) < 0.1
                p90_match = abs(manual_p90 - computed['uniform_p90']) < 0.1
                max_match = abs(manual_max - computed['uniform_max']) < 0.1
                
                if median_match and p90_match and max_match:
                    print("  ✅ Manual and computed values match")
                else:
                    print("  ❌ Mismatch detected!")
            print()
    
    # Test JSON serialization
    print("📄 JSON Serialization Test:")
    print("-" * 40)
    
    summary = result.to_summary_dict()
    if 'policy_global_suggestions' in summary:
        print("✅ policy_global_suggestions included in summary")
        print("Sample policy_global_suggestions JSON:")
        print(json.dumps(summary['policy_global_suggestions'], indent=2))
    else:
        print("❌ policy_global_suggestions missing from summary")
    
    print()
    print("🎯 Step 5 Global Suggestions Test Complete!")

def test_candidate_compression_targets():
    """Show how the global suggestions provide multiple compression targets."""
    
    print("\n🎯 Candidate Compression Targets Demo")
    print("=" * 60)
    
    state_dict, _ = create_test_adapter_with_varied_ranks()
    result = audit_lora_state_dict(
        state_dict,
        rank_policies=["optimal_hard_threshold", "entropy_effective", "knee_elbow"]
    )
    
    if not result.policy_global_suggestions:
        print("❌ No global suggestions computed")
        return
    
    print("Multiple candidate compression targets for Bench validation:")
    print()
    
    # Format as compression targets
    targets = []
    for policy, stats in result.policy_global_suggestions.items():
        targets.append({
            'policy': policy,
            'median': int(stats['uniform_median']),
            'conservative_p90': int(stats['uniform_p90']),
            'ultra_conservative_max': int(stats['uniform_max'])
        })
    
    for target in targets:
        policy = target['policy']
        print(f"Policy: {policy}")
        print(f"  Aggressive:        r = {target['median']}")
        print(f"  Conservative:      r = {target['conservative_p90']}")
        print(f"  Ultra-conservative: r = {target['ultra_conservative_max']}")
        print()
    
    # Show traditional vs policy-based suggestions
    print("Comparison with traditional approach:")
    print(f"  Traditional energy@90 median: {result.energy_rank_90_p50}")
    print(f"  Traditional energy@90 p90:    {result.energy_rank_90_p90}")
    
    print()
    print("✅ Multiple defensible compression targets available for benchmarking!")

if __name__ == "__main__":
    test_global_policy_suggestions()
    test_candidate_compression_targets()