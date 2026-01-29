#!/usr/bin/env python3
"""
Demo: Enhanced Rank Selection Policies in Gradience

Shows the new pluggable rank selection system in action,
comparing different heuristics (OHT, entropy effective rank, knee/elbow)
with the traditional energy@90% approach.
"""

import torch
import json
from gradience.vnext.rank_policies import get_available_policies, get_policy_summary

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
    
    policies_to_test = ["energy_90", "entropy_effective", "oht", "knee_elbow", "stable_rank_ceil"]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📊 Test Case {i}: {test_case['name']}")
        print(f"   Singular values: {test_case['singular_values'].tolist()}")
        print(f"   Expected rank: {test_case['expected_rank']}")
        print("-" * 60)
        
        # Get policy summary
        summary = get_policy_summary(test_case['singular_values'], policies_to_test)
        
        # Display results table
        print(f"{'Policy':<18} {'Rank':<6} {'Confidence':<12} {'Agreement':<12}")
        print("-" * 60)
        
        for policy_name in policies_to_test:
            if policy_name in summary['policy_results']:
                result = summary['policy_results'][policy_name]
                rank = result.suggested_rank
                confidence = result.confidence
                
                # Check agreement with expected
                diff = abs(rank - test_case['expected_rank'])
                if diff == 0:
                    agreement = "✅ Perfect"
                elif diff == 1:
                    agreement = "📊 Close"
                else:
                    agreement = f"❌ Off by {diff}"
                
                print(f"{policy_name:<18} {rank:<6} {confidence:<12.2f} {agreement}")
        
        print()
        consensus = summary['rank_consensus']
        print(f"📈 Consensus: median={consensus['median']}, range={consensus['range']}")
        print(f"🎯 High confidence: {consensus['high_confidence']}")
        
        # Show singular value stats
        sv_stats = summary['singular_values_stats'] 
        print(f"📉 Condition number: {sv_stats['ratio_max_min']:.1f}")

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
    
    print("from gradience.vnext.rank_policies import apply_policy, get_available_policies")
    print()
    print("# Apply individual policy")
    print(f"s = torch.tensor({singular_values.tolist()})")
    print("result = apply_policy('oht', s)")
    
    # Actually run it
    from gradience.vnext.rank_policies import apply_policy
    result = apply_policy('oht', singular_values)
    
    print(f"# Result: rank={result.suggested_rank}, confidence={result.confidence:.2f}")
    print(f"# Metadata: {result.metadata}")
    print()
    
    # Show available policies
    policies = get_available_policies()
    print("# Available policies:")
    print(f"policies = {list(policies.keys())}")

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