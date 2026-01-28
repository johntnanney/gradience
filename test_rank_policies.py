#!/usr/bin/env python3
"""
Test script for new rank selection policies.

Demonstrates how to use the enhanced audit system with multiple
rank selection heuristics (OHT, entropy effective rank, knee/elbow, etc).
"""

import torch
from gradience.vnext.rank_policies import (
    get_available_policies,
    apply_policy, 
    apply_all_policies,
    get_policy_summary
)

def test_rank_policies():
    """Test rank policies on synthetic singular value data."""
    
    # Create test singular values with clear rank structure
    # Simulate a rank-4 matrix with noise
    signal_values = torch.tensor([10.0, 5.0, 2.0, 1.0])  # Clear signal
    noise_values = torch.tensor([0.1, 0.08, 0.05, 0.03, 0.01, 0.005]) # Noise floor
    singular_values = torch.cat([signal_values, noise_values])
    
    print("🔍 Testing Rank Selection Policies")
    print("=" * 50)
    print(f"Input singular values: {singular_values.tolist()}")
    print(f"Expected rank (visual inspection): ~4")
    print()
    
    # Test individual policies
    policies = get_available_policies()
    print(f"📋 Available policies: {list(policies.keys())}")
    print()
    
    # Apply all policies and compare
    results = apply_all_policies(singular_values)
    
    print("📊 Policy Comparison:")
    print("-" * 70)
    print(f"{'Policy':<20} {'Rank':<6} {'Confidence':<12} {'Key Metadata'}")
    print("-" * 70)
    
    for policy_name, result in results.items():
        if 'error' not in result.metadata:
            key_info = ""
            if policy_name.startswith("energy"):
                key_info = f"energy={result.metadata.get('actual_energy_captured', 0):.2f}"
            elif policy_name == "entropy_effective":
                key_info = f"eff_rank={result.metadata.get('effective_rank_raw', 0):.1f}"
            elif policy_name == "oht":
                key_info = f"snr={result.metadata.get('signal_to_noise_ratio', 0):.1f}"
            elif policy_name == "knee_elbow":
                key_info = f"elbow_idx={result.metadata.get('elbow_index', -1)}"
            elif policy_name == "stable_rank_ceil":
                key_info = f"stable={result.metadata.get('stable_rank_raw', 0):.1f}"
            
            print(f"{policy_name:<20} {result.suggested_rank:<6} {result.confidence:<12.2f} {key_info}")
        else:
            print(f"{policy_name:<20} {'ERROR':<6} {'0.00':<12} {result.metadata['error'][:30]}...")
    
    print()
    
    # Test policy summary
    summary = get_policy_summary(singular_values, ["energy_90", "entropy_effective", "oht", "knee_elbow"])
    
    print("🎯 Policy Summary:")
    consensus = summary["rank_consensus"]
    print(f"  Median suggestion: {consensus['median']}")
    print(f"  Range: {consensus['range']}")
    print(f"  High confidence policies: {consensus['high_confidence']}")
    
    sv_stats = summary["singular_values_stats"]
    print(f"  Condition number: {sv_stats['ratio_max_min']:.1f}")
    
    print()
    print("✅ All policies tested successfully!")
    
    return results

def test_audit_integration():
    """Test integration with the audit system."""
    import tempfile
    import json
    from pathlib import Path
    from gradience.vnext.audit.lora_audit import audit_lora_state_dict
    
    print("\n🔗 Testing Audit Integration")
    print("=" * 50)
    
    # Create mock LoRA state dict
    state_dict = {
        "base_model.model.bert.encoder.layer.0.attention.self.query.lora_A.default.weight": 
            torch.randn(8, 768, dtype=torch.float16) * 0.1,
        "base_model.model.bert.encoder.layer.0.attention.self.query.lora_B.default.weight":
            torch.randn(768, 8, dtype=torch.float16) * 0.1,
    }
    
    # Test without policies (backward compatibility)
    result_basic = audit_lora_state_dict(state_dict)
    print(f"✅ Basic audit (no policies): {len(result_basic.layers)} layers")
    
    # Test with policies enabled
    rank_policies = ["energy_90", "entropy_effective", "oht"]
    result_with_policies = audit_lora_state_dict(state_dict, rank_policies=rank_policies)
    
    print(f"✅ Enhanced audit (with policies): {len(result_with_policies.layers)} layers")
    
    # Check that policy results are included
    layer = result_with_policies.layers[0]
    if layer.policy_rank_suggestions:
        print(f"📊 Policy results for layer '{layer.name}':")
        for policy, data in layer.policy_rank_suggestions.items():
            rank = data.get('suggested_rank', 0)
            conf = data.get('confidence', 0)
            print(f"  {policy}: rank={rank}, confidence={conf:.2f}")
    else:
        print("⚠️  No policy results found")
    
    print("✅ Integration test completed!")


if __name__ == "__main__":
    # Test the policy system
    results = test_rank_policies()
    
    # Test integration with audit system
    test_audit_integration()