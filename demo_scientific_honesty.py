#!/usr/bin/env python3
"""
Demo script showing Priority 3: Scientific honesty in markdown reports.

Demonstrates how per-rank validation evidence statements make reports more
scientific and trustworthy by explicitly stating the evidence behind claims.
"""

import tempfile
from pathlib import Path


def demo_scientific_honesty():
    """Demonstrate the scientific honesty improvements in markdown reports."""

    print("🔬 PRIORITY 3: SCIENTIFIC HONESTY DEMONSTRATION")
    print("=" * 60)
    print()
    print("📊 SCENARIO: Multi-seed validation with mixed results")
    print("• r=48 (energy_p90): All 3 seeds pass tolerance")
    print("• r=32 (uniform_r32): Only 2/3 seeds pass tolerance")
    print("• Question: How should we report this honestly?")
    print()

    # Create realistic test scenario that demonstrates the honesty
    with tempfile.TemporaryDirectory() as temp_dir:
        seed_reports = []

        # Define exact pass/fail patterns for demonstration
        pass_patterns = {
            "energy_p90": ["PASS", "PASS", "PASS"],  # r=48: perfect validation
            "uniform_r32": ["PASS", "PASS", "FAIL"],  # r=32: partial validation
        }

        print("🧪 CREATING EVIDENCE PACK:")
        for i, seed in enumerate([42, 123, 456]):
            energy_verdict = pass_patterns["energy_p90"][i]
            uniform_verdict = pass_patterns["uniform_r32"][i]

            print(f"   Seed {seed}: energy_p90={energy_verdict}, uniform_r32={uniform_verdict}")

            report = {
                "bench_version": "0.1",
                "timestamp": f"2026-02-07T12:0{i}:00.000000",
                "model": "mistralai/Mistral-7B-v0.1",
                "task": "gsm8k/main",
                "env": {"seed": seed, "validation_classification": {"max_steps": 1000}},
                "probe": {
                    "rank": 64,
                    "params": 167772160,
                    "accuracy": 0.336 + i * 0.001,  # Small variance
                },
                "compressed": {
                    "energy_p90": {
                        "rank": 48,
                        "params": 125829120,
                        "accuracy": 0.368 + i * 0.001,
                        "delta_vs_probe": 0.032 + i * 0.001,
                        "param_reduction": 0.25,
                        "verdict": energy_verdict,
                    },
                    "uniform_r32": {
                        "rank": 32,
                        "params": 83886080,
                        "accuracy": 0.348 + i * 0.001,
                        "delta_vs_probe": 0.012 + i * 0.001,
                        "param_reduction": 0.5,
                        "verdict": uniform_verdict,
                    },
                },
            }
            seed_reports.append(report)

        print()
        print("📝 GENERATING HONEST MARKDOWN REPORT:")
        print("-" * 40)

        from gradience.bench.protocol import create_multi_seed_aggregated_report, create_multi_seed_markdown_report

        config = {"bench_version": "0.1", "compression": {"acc_tolerance": 0.025}}

        # Generate the scientifically honest markdown
        agg_report = create_multi_seed_aggregated_report(seed_reports, config, Path(temp_dir))
        md_content = create_multi_seed_markdown_report(agg_report, config, Path(temp_dir))

        # Extract and display the per-rank validation evidence
        start = md_content.find("## Per-Rank Validation Evidence")
        end = md_content.find("## Two-Tier Selection System")

        if start != -1 and end != -1:
            evidence_section = md_content[start:end].strip()
            print(evidence_section)
        else:
            print("❌ Could not extract evidence section")
            return

        print()
        print("✨ SCIENTIFIC HONESTY IMPROVEMENTS:")
        print("=" * 50)
        print()

        print("🔍 BEFORE (typical industry approach):")
        print('   "Both r=48 and r=32 show good compression results"')
        print("   → Hides the fact that r=32 failed 1/3 seeds")
        print("   → Users might deploy unreliable configurations")
        print()

        print("🚀 AFTER (evidence-based honesty):")
        print('   "r=48 is validated safe (3/3)"')
        print('   "r=32 is conditionally promising (2/3; one seed violates tolerance)"')
        print("   → Complete transparency about evidence")
        print("   → Users can make informed decisions")
        print()

        print("💡 KEY SCIENTIFIC PRINCIPLES:")
        print("• ✅ Report all evidence, including negative results")
        print("• ✅ Use precise language that matches the data")
        print("• ✅ Distinguish between validated vs. promising approaches")
        print("• ✅ Enable informed risk assessment")
        print("• ✅ Maintain reproducibility and audit trails")
        print()

        print("🎯 IMPACT ON TRUSTWORTHINESS:")
        print("• Researchers can trust the complete picture")
        print("• Practitioners understand the reliability of each option")
        print("• Peer reviewers see honest evidence-based claims")
        print("• Results are more likely to replicate in other settings")
        print()

        print("📈 EXAMPLE STATEMENTS FROM GENERATED REPORT:")

        # Extract specific validation statements
        if "r=48** is ✅ **validated safe**" in md_content:
            print('   ✅ "r=48 is validated safe (All 3 seeds pass ±0.025 tolerance)"')

        if "r=32** is ⚠️  **conditionally promising**" in md_content:
            print('   ⚠️  "r=32 is conditionally promising (2/3 seeds pass; one seed violates tolerance)"')

        print()
        print("🏆 RESULT: Evidence pack that researchers can trust and cite!")


if __name__ == "__main__":
    demo_scientific_honesty()
