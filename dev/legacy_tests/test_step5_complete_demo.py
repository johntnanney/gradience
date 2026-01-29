#!/usr/bin/env python3
"""
Complete demo of Step 5 - Global policy suggestions.

Shows how multiple policies provide different compression targets
that the Bench can validate.
"""

import torch
from gradience.vnext.audit.lora_audit import audit_lora_state_dict

def create_realistic_adapter():
    """Create adapter mimicking real-world LoRA characteristics."""
    state_dict = {}
    
    # Typical transformer layers with different adaptation patterns
    layers = [
        # Attention layers tend to have lower effective rank
        ("model.layers.0.self_attn.q_proj", [8.2, 4.1, 2.3, 1.1, 0.5, 0.3, 0.1, 0.05]),
        ("model.layers.0.self_attn.k_proj", [7.5, 3.8, 1.9, 0.9, 0.4, 0.2, 0.08, 0.03]),
        ("model.layers.0.self_attn.v_proj", [9.1, 5.2, 2.8, 1.4, 0.7, 0.35, 0.15, 0.08]),
        ("model.layers.0.self_attn.o_proj", [6.8, 3.2, 1.6, 0.8, 0.4, 0.2, 0.1, 0.05]),
        
        # MLP layers tend to have higher effective rank
        ("model.layers.1.mlp.gate_proj", [12.5, 9.2, 6.8, 4.5, 3.1, 2.2, 1.5, 1.0]),
        ("model.layers.1.mlp.up_proj", [11.8, 8.6, 6.2, 4.1, 2.8, 1.9, 1.2, 0.8]),
        ("model.layers.1.mlp.down_proj", [10.3, 7.4, 5.1, 3.4, 2.3, 1.6, 1.0, 0.6]),
        
        # Later layers often have different characteristics
        ("model.layers.15.self_attn.q_proj", [15.2, 3.8, 1.9, 1.2, 0.6, 0.3, 0.15, 0.08]),
        ("model.layers.15.self_attn.v_proj", [14.8, 4.2, 2.1, 1.0, 0.5, 0.25, 0.12, 0.06]),
        ("model.layers.15.mlp.gate_proj", [18.5, 12.3, 8.7, 5.9, 4.2, 2.8, 1.9, 1.3]),
    ]
    
    for name, sing_vals in layers:
        r = len(sing_vals)
        A = torch.randn(r, 4096)
        U, _, Vt = torch.linalg.svd(A, full_matrices=False)
        s = torch.tensor(sing_vals, dtype=torch.float32)
        A = (U * s.unsqueeze(0)) @ Vt
        B = torch.randn(4096, r) * 0.1
        
        state_dict[f"{name}.lora_A.weight"] = A
        state_dict[f"{name}.lora_B.weight"] = B
    
    return state_dict

def main():
    """Complete Step 5 demonstration."""
    
    print("🎯 Step 5 — Global Policy Suggestions — Complete Demo")
    print("=" * 70)
    
    # Create realistic adapter
    state_dict = create_realistic_adapter()
    print(f"Created realistic adapter with {len(state_dict)//2} layers")
    print()
    
    # Audit with all policies
    result = audit_lora_state_dict(
        state_dict,
        rank_policies=["optimal_hard_threshold", "entropy_effective", "knee_elbow"]
    )
    
    print("✅ Audit completed with global policy suggestions")
    print()
    
    # Show traditional vs new approach
    print("📊 Traditional vs Policy-Based Global Suggestions")
    print("-" * 55)
    print("Traditional approach (energy@90 only):")
    print(f"  Uniform median: {result.energy_rank_90_p50}")
    print(f"  Uniform p90:    {result.energy_rank_90_p90}")
    print()
    
    print("New policy-based approach:")
    if result.policy_global_suggestions:
        for policy, stats in result.policy_global_suggestions.items():
            if policy == 'energy_90':
                continue  # Skip showing traditional one again
            print(f"  {policy}:")
            print(f"    median = {stats['uniform_median']}")
            print(f"    p90    = {stats['uniform_p90']}")
            print(f"    max    = {stats['uniform_max']}")
    print()
    
    # Show candidate compression targets
    print("🔧 Multiple Candidate Compression Targets for Bench")
    print("-" * 55)
    
    targets = []
    if result.policy_global_suggestions:
        for policy, stats in result.policy_global_suggestions.items():
            targets.extend([
                f"{policy}_median={int(stats['uniform_median'])}",
                f"{policy}_p90={int(stats['uniform_p90'])}", 
                f"{policy}_max={int(stats['uniform_max'])}"
            ])
    
    # Group by rank value for easier comparison
    rank_groups = {}
    for target in targets:
        rank = int(target.split('=')[1])
        if rank not in rank_groups:
            rank_groups[rank] = []
        rank_groups[rank].append(target.split('=')[0])
    
    print("Compression targets grouped by rank:")
    for rank in sorted(rank_groups.keys()):
        sources = rank_groups[rank]
        print(f"  r = {rank:2}: {', '.join(sources)}")
    print()
    
    # Show per-layer vs global
    print("🔍 Per-Layer vs Global Analysis Example")
    print("-" * 55)
    print("OHT policy across layers:")
    oht_values = []
    for layer in result.layers:
        if (layer.rank_suggestions and 
            'oht' in layer.rank_suggestions):
            k = layer.rank_suggestions['oht']['k']
            oht_values.append(k)
            print(f"  {layer.name[:30]:<30}: k={k}")
    
    if result.policy_global_suggestions and 'oht' in result.policy_global_suggestions:
        global_oht = result.policy_global_suggestions['oht']
        print(f"\n  Individual values: {oht_values}")
        print(f"  Global median:     {global_oht['uniform_median']}")
        print(f"  Global p90:        {global_oht['uniform_p90']}")
        print(f"  Global max:        {global_oht['uniform_max']}")
    print()
    
    # Bench integration preview
    print("🏗️  Bench Integration Preview")
    print("-" * 55)
    print("Example bench config variants:")
    print("compression:")
    print("  policy_variants:")
    
    if result.policy_global_suggestions:
        for policy, stats in result.policy_global_suggestions.items():
            median_r = int(stats['uniform_median'])
            p90_r = int(stats['uniform_p90'])
            max_r = int(stats['uniform_max'])
            
            print(f"    - name: {policy}_aggressive")
            print(f"      uniform_rank: {median_r}")
            print(f"    - name: {policy}_conservative") 
            print(f"      uniform_rank: {p90_r}")
            if max_r != p90_r:
                print(f"    - name: {policy}_ultra_conservative")
                print(f"      uniform_rank: {max_r}")
    
    print()
    print("✅ Step 5 Complete: Multiple defensible compression targets available!")
    print("   Bench can now validate different policy-based suggestions")

if __name__ == "__main__":
    main()