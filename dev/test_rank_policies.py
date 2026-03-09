#!/usr/bin/env python3
"""
Test script for new rank selection policies.

Demonstrates how to use the enhanced audit system with multiple
rank selection heuristics (OHT, entropy effective rank, knee/elbow, etc).
"""

import numpy as np
import torch

from gradience.vnext.audit.rank_policies import (
    RankPolicySpec,
    RankSuggestion,
    analyze_policy_consensus,
    apply_rank_policy,
    get_standard_policies,
)


def test_rank_policies():
    """Test rank policies on synthetic singular value data."""

    # Create test singular values with clear rank structure
    # Simulate a rank-4 matrix with noise
    signal_values = torch.tensor([10.0, 5.0, 2.0, 1.0])  # Clear signal
    noise_values = torch.tensor([0.1, 0.08, 0.05, 0.03, 0.01, 0.005])  # Noise floor
    singular_values = torch.cat([signal_values, noise_values])
    s_np = singular_values.numpy()

    print("🔍 Testing Rank Selection Policies")
    print("=" * 50)
    print(f"Input singular values: {singular_values.tolist()}")
    print("Expected rank (visual inspection): ~4")
    print()

    # Test individual policies
    policies = get_standard_policies()
    print(f"📋 Available policies: {[p.name for p in policies]}")
    print()

    # Apply all policies and compare
    shape = (768, 768)
    r_alloc = len(s_np)

    results = {}
    for policy_spec in policies:
        try:
            result = apply_rank_policy(policy_spec, s_np, shape, r_alloc)
            results[policy_spec.name] = result
        except Exception as e:
            print(f"⚠️  Policy {policy_spec.name} failed: {e}")
            results[policy_spec.name] = None

    print("📊 Policy Comparison:")
    print("-" * 70)
    print(f"{'Policy':<25} {'Rank':<6} {'Confidence':<12} {'Key Metadata'}")
    print("-" * 70)

    for policy_name, result in results.items():
        if result is None:
            print(f"{policy_name:<25} {'ERROR':<6}")
            continue
        key_info = ""
        if "actual_energy_captured" in result.details:
            key_info = f"energy={result.details.get('actual_energy_captured', 0):.2f}"
        elif "erank_float" in result.details:
            key_info = f"eff_rank={result.details.get('erank_float', 0):.1f}"
        elif "tau" in result.details:
            key_info = f"tau={result.details.get('tau', 0):.2f}"
        elif "knee_index" in result.details:
            key_info = f"elbow_idx={result.details.get('knee_index', -1)}"
        elif "stable_rank" in result.details:
            key_info = f"stable={result.details.get('stable_rank', 0):.1f}"

        print(f"{policy_name:<25} {result.k:<6} {result.confidence:<12.2f} {key_info}")

    print()

    # Test consensus analysis
    consensus = analyze_policy_consensus(policies, s_np, shape, r_alloc)

    print("🎯 Policy Consensus:")
    print(f"  Median suggestion: {consensus.median_k}")
    print(f"  Range: {consensus.k_range}")
    print(f"  High confidence policies: {consensus.high_confidence_policies}")
    print(f"  Disagreement score: {consensus.disagreement_score:.2f}")

    print()
    print("✅ All policies tested successfully!")

    return results


def test_audit_integration():
    """Test integration with the audit system."""
    import json
    import tempfile
    from pathlib import Path

    from gradience.vnext.audit.lora_audit import audit_lora_state_dict

    print("\n🔗 Testing Audit Integration")
    print("=" * 50)

    # Create mock LoRA state dict
    state_dict = {
        "base_model.model.bert.encoder.layer.0.attention.self.query.lora_A.default.weight": torch.randn(
            8, 768, dtype=torch.float16
        )
        * 0.1,
        "base_model.model.bert.encoder.layer.0.attention.self.query.lora_B.default.weight": torch.randn(
            768, 8, dtype=torch.float16
        )
        * 0.1,
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
            rank = data.get("suggested_rank", 0)
            conf = data.get("confidence", 0)
            print(f"  {policy}: rank={rank}, confidence={conf:.2f}")
    else:
        print("⚠️  No policy results found")

    print("✅ Integration test completed!")


if __name__ == "__main__":
    # Test the policy system
    results = test_rank_policies()

    # Test integration with audit system
    test_audit_integration()
