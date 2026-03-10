#!/usr/bin/env python3
"""
Demo: Enhanced Rank Selection Policies in Gradience

Shows the new pluggable rank selection system in action,
comparing different heuristics (OHT, entropy effective rank, knee/elbow)
with the traditional energy@90% approach.
"""

import torch
import numpy as np
from gradience.vnext.audit.rank_policies import (
    apply_rank_policy,
    analyze_policy_consensus,
    get_standard_policies,
    create_energy_policy,
    create_entropy_policy,
    create_oht_policy,
    create_knee_policy,
    create_stable_ceil_policy,
    RankPolicySpec,
)

def demo_policy_comparison():
    """Demonstrate policy comparison on synthetic data."""

    print("🎯 Gradience Enhanced Rank Selection Demo")
    print("=" * 60)

    # Create test cases with different rank structures
    test_cases = [
        {
            "name": "Clear Low-Rank (rank~3)",
            "singular_values": torch.tensor([10.0, 5.0, 2.0, 0.1, 0.05, 0.02, 0.01]),
            "expected_rank": 3,
        },
        {
            "name": "Gradual Decay (rank~5)",
            "singular_values": torch.tensor([8.0, 6.0, 4.0, 3.0, 2.0, 1.0, 0.5, 0.1]),
            "expected_rank": 5,
        },
        {
            "name": "High Noise (rank~2)",
            "singular_values": torch.tensor([15.0, 8.0, 0.3, 0.25, 0.2, 0.15, 0.1, 0.08, 0.05]),
            "expected_rank": 2,
        }
    ]

    policies = [
        create_energy_policy(0.90),
        create_entropy_policy(),
        create_oht_policy(),
        create_knee_policy(),
        create_stable_ceil_policy(),
    ]

    shape = (768, 768)

    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📊 Test Case {i}: {test_case['name']}")
        print(f"   Singular values: {test_case['singular_values'].tolist()}")
        print(f"   Expected rank: {test_case['expected_rank']}")
        print("-" * 60)

        s_np = test_case['singular_values'].numpy()
        r_alloc = len(s_np)

        # Get consensus analysis
        consensus = analyze_policy_consensus(policies, s_np, shape, r_alloc)

        # Display results table
        print(f"{'Policy':<25} {'Rank':<6} {'Confidence':<12} {'Agreement':<12}")
        print("-" * 60)

        for policy_spec, result in zip(policies, consensus.suggestions):
            rank = result.k
            confidence = result.confidence

            # Check agreement with expected
            diff = abs(rank - test_case['expected_rank'])
            if diff == 0:
                agreement = "✅ Perfect"
            elif diff == 1:
                agreement = "📊 Close"
            else:
                agreement = f"❌ Off by {diff}"

            print(f"{policy_spec.name:<25} {rank:<6} {confidence:<12.2f} {agreement}")

        print()
        print(f"📈 Consensus: median={consensus.median_k}, range={consensus.k_range}")
        print(f"🎯 High confidence: {consensus.high_confidence_policies}")

        # Condition number
        if len(s_np) > 0 and s_np[-1] > 0:
            cond = s_np[0] / s_np[-1]
            print(f"📉 Condition number: {cond:.1f}")

def demo_cli_integration():
    """Show how the new policies integrate with CLI."""

    print("\n\n🖥️  CLI Integration Example")
    print("=" * 60)

    example_commands = [
        "# Compare multiple policies on an adapter",
        "gradience audit --peft-dir my_adapter --rank-policies oht entropy_effective knee_elbow --json",
        "",
        "# Include per-layer analysis with policies",
        "gradience audit --peft-dir my_adapter --rank-policies energy_90 energy_95 oht --layers",
        "",
        "# Use in bench protocol for compression variants",
        "gradience bench configs/my_config.yaml  # (with policy: variants in compression config)",
    ]

    for cmd in example_commands:
        if cmd.startswith("#"):
            print(f"\033[92m{cmd}\033[0m")  # Green comments
        elif cmd == "":
            print()
        else:
            print(f"\033[94m{cmd}\033[0m")  # Blue commands

def demo_programmatic_usage():
    """Show programmatic usage of the policy system."""

    print("\n\n🐍 Programmatic Usage Example")
    print("=" * 60)

    # Example singular values from real adapter
    singular_values = torch.tensor([12.5, 8.2, 4.1, 2.8, 1.6, 0.9, 0.4, 0.2, 0.1, 0.05])
    s_np = singular_values.numpy()
    shape = (768, 768)
    r_alloc = len(s_np)

    print("from gradience.vnext.audit.rank_policies import apply_rank_policy, create_oht_policy")
    print()
    print("# Apply individual policy")
    print(f"s = np.array({singular_values.tolist()})")

    # Actually run it
    oht_policy = create_oht_policy()
    result = apply_rank_policy(oht_policy, s_np, shape, r_alloc)

    print(f"result = apply_rank_policy(create_oht_policy(), s, shape=(768, 768), r_alloc={r_alloc})")
    print(f"# Result: rank={result.k}, confidence={result.confidence:.2f}")
    print(f"# Details: {dict(result.details)}")
    print()

    # Show available policies
    policies = get_standard_policies()
    print("# Standard policies:")
    print(f"policies = {[p.name for p in policies]}")

def main():
    """Run the complete demo."""

    # Test policy comparison
    demo_policy_comparison()

    # Show CLI integration
    demo_cli_integration()

    # Show programmatic usage
    demo_programmatic_usage()

    print("\n\n✨ Summary")
    print("=" * 60)
    print("The enhanced rank selection system provides:")
    print("• 🔬 Multiple scientifically-grounded heuristics")
    print("• 📊 Confidence scores for each suggestion")
    print("• 🔌 Seamless integration with existing Gradience workflows")
    print("• 🎯 Consensus-based decision making")
    print("• 📈 CLI and programmatic interfaces")
    print()
    print("No more 'energy@90% or bust' - now you have a defensible menu!")

if __name__ == "__main__":
    main()
