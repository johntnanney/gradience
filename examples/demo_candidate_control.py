#!/usr/bin/env python3
"""
Demo: Bench Candidate Control - Preventing Training Job Factory Explosion

Shows the 3 key features implemented:
1. De-duplicate ranks: if knee_p90 and energy_p90 both suggest r=32, run it once
2. Cap candidates: limit to top 4 by conservatism or diversity 
3. Fast mode: only energy_p90, knee_p90, erank_p90 (practitioner-friendly default)
"""

def demonstrate_candidate_control():
    print("🚀 Bench Candidate Control Demo")
    print("=" * 50)
    
    # Simulate what the old system would generate
    print("❌ OLD SYSTEM (without control):")
    print("   Could generate up to 12+ variants per seed:")
    old_variants = [
        "uniform_median (r=6)",
        "uniform_p90 (r=8)",  
        "uniform_knee_p90 (r=4)",
        "uniform_erank_p90 (r=8)",  # DUPLICATE rank with uniform_p90!
        "uniform_oht_p90 (r=3)", 
        "uniform_energy90_median (r=6)",  # DUPLICATE rank with uniform_median!
        "uniform_energy90_p90 (r=8)",  # DUPLICATE rank again!
        "per_layer (r=6.0)",
        "per_layer_shuffled (r=6.0)",  # Another duplicate!
        "svd_trunc_r4 (r=4)",  # Duplicate with knee!
        "svd_trunc_r8 (r=8)",  # More duplicates!
        # + any other policy combinations...
    ]
    
    for variant in old_variants:
        print(f"     • {variant}")
    
    print(f"\n   Total: {len(old_variants)} variants x 3 seeds = 36 training jobs! 😱")
    
    print("\n" + "=" * 50)
    print("✅ NEW SYSTEM (with candidate control):")
    
    # Fast mode (default)
    print("\n🏃 FAST MODE (default):")
    fast_variants = [
        "energy_p90 (r=8, conservatism=3.0)",
        "knee_p90 (r=4, conservatism=2.0)", 
        # erank_p90 deduplicated with energy_p90 since both suggest r=8
        "per_layer (r=6.0, conservatism=2.0)"  # only if different from uniform
    ]
    
    for variant in fast_variants:
        print(f"     • {variant}")
    
    print(f"   Total: {len(fast_variants)} variants x 3 seeds = 9 training jobs ✨")
    
    # Full mode (research)
    print("\n🔬 FULL MODE (--full-mode):")
    full_variants = [
        "energy_p90 (r=8, conservatism=3.0)",  # Best policy for r=8
        "knee_p90 (r=4, conservatism=2.0)",
        "oht_p90 (r=3, conservatism=2.5)",
        "uniform_median (r=6, conservatism=4.0)"  # Capped at 4 total
    ]
    
    for variant in full_variants:
        print(f"     • {variant}")
        
    print(f"   Total: {len(full_variants)} variants x 3 seeds = 12 training jobs 👍")
    
    print("\n" + "=" * 50)
    print("🎯 KEY IMPROVEMENTS:")
    print("   1. DE-DUPLICATION: Same ranks (r=8) run once, not 3x")
    print("   2. CAPPING: Limited to 4 candidates max") 
    print("   3. FAST MODE: Practitioner-friendly default (3 policies)")
    print("   4. SMART SELECTION: Pick best policy per rank by conservatism")
    
    print("\n📊 TRAINING JOB REDUCTION:")
    print("   • Fast mode: 36 → 9 jobs (75% reduction)")
    print("   • Full mode: 36 → 12 jobs (67% reduction)")
    print("   • No more training job factory explosion!")
    
    print("\n💡 CLI USAGE:")
    print("   # Default (fast mode)")
    print("   python -m gradience.bench.run_bench --config config.yaml --output results/")
    print()
    print("   # Research mode (more policies)")
    print("   python -m gradience.bench.run_bench --config config.yaml --output results/ --full-mode")
    print()
    print("   # Custom candidate limit")
    print("   python -m gradience.bench.run_bench --config config.yaml --output results/ --full-mode --max-candidates 6")

if __name__ == "__main__":
    demonstrate_candidate_control()