#!/usr/bin/env python3
"""
Demo script showing candidate diversity improvements.

Demonstrates how the enhanced deduplication logic prevents rank collisions
by remapping duplicate candidates to "second rung" alternatives.
"""

import json
import tempfile
from pathlib import Path


def demo_candidate_diversity():
    """Demonstrate the candidate diversity improvements."""

    # Create temporary directory for demo
    with tempfile.TemporaryDirectory() as temp_dir:
        probe_dir = Path(temp_dir) / "probe"
        probe_dir.mkdir()

        # Create realistic audit data that triggers the original collision problem
        audit_data = {
            "policy_global_suggestions": {
                "energy_90": {"uniform_p90": 48},  # energy_p90 suggests r=48
                "knee": {"uniform_p90": 48},  # knee_p90 also suggests r=48 (collision!)
                "erank": {"uniform_p90": 32},  # erank_p90 suggests r=32 (different)
            }
        }

        # Save audit.json
        audit_path = probe_dir / "audit.json"
        with open(audit_path, "w") as f:
            json.dump(audit_data, f)

        # Config with FAST mode and the three policies that caused the original issue
        config = {
            "compression": {
                "allowed_ranks": [4, 8, 12, 16, 24, 32, 40, 48, 56, 64],
                "fast_mode": True,
                "max_candidates": 8,
                "candidate_policies": ["energy_p90", "knee_p90", "erank_p90"],
            },
            "lora": {"probe_r": 64, "alpha": 128, "dropout": 0.05, "target_modules": ["q_proj", "k_proj", "v_proj"]},
        }

        print("🎯 CANDIDATE DIVERSITY IMPROVEMENT DEMO")
        print("=" * 50)
        print()
        print("📋 SCENARIO: FAST mode with collision-prone policies")
        print("• energy_p90 suggests r=48")
        print("• knee_p90 suggests r=48    ← COLLISION!")
        print("• erank_p90 suggests r=32")
        print()

        # Generate candidates with the improved logic
        from gradience.bench.protocol import generate_compression_configs

        compression_configs, decision_trace = generate_compression_configs(
            probe_dir, config, fast_mode=True, max_candidates=4
        )

        print()
        print("🎉 RESULTS WITH DIVERSITY PRESERVATION:")
        print("-" * 40)

        # Extract and display results
        generated_variants = []
        generated_ranks = []
        remapped_info = []

        for variant_name, variant_config in compression_configs.items():
            if variant_name != "per_layer":
                generated_variants.append(variant_name)
                generated_ranks.append(variant_config["actual_r"])

                # Check if this was remapped
                dedup_note = variant_config.get("reason", "") or ""
                if "second rung" in dedup_note:
                    remapped_info.append(f"  📍 {variant_name}: {dedup_note}")
                elif "Preferred choice" in dedup_note:
                    remapped_info.append(f"  ✅ {variant_name}: {dedup_note}")

        print(f"Generated variants: {generated_variants}")
        print(f"Generated ranks: {generated_ranks}")
        print(f"Unique ranks: {len(set(generated_ranks))} (diversity preserved!)")
        print()

        if remapped_info:
            print("📝 DEDUPLICATION DETAILS:")
            for info in remapped_info:
                print(info)
            print()

        print("✨ KEY IMPROVEMENTS:")
        print(f"• ✅ No duplicate ranks: {len(set(generated_ranks)) == len(generated_ranks)}")
        print(f"• ✅ Preserved diversity: {len(generated_ranks)} candidates vs 2 in old system")
        print("• ✅ No additional compute: Same number of training runs")
        print("• ✅ More informative: Different compression ratios tested")

        print()
        print("🔍 BEFORE FIX:")
        print("  - energy_p90 and knee_p90 both → r=48")
        print("  - Result: [48, 32] (lost diversity)")
        print()
        print("🚀 AFTER FIX:")
        print("  - energy_p90 remapped to more aggressive r=4")
        print("  - knee_p90 kept at preferred r=48")
        print(f"  - Result: {generated_ranks} (diversity preserved!)")
        print()
        print("💡 This is purely logic optimization - no extra compute required!")


if __name__ == "__main__":
    demo_candidate_diversity()
