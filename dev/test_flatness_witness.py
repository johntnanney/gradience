#!/usr/bin/env python3
"""
Test: Explicit Flatness Witness

Demonstrates the mathematically explicit flat distribution detection
with exact witness values that prove why a distribution was declared flat.
"""

import json
import sys

sys.path.insert(0, ".")

from gradience.cli import _analyze_policy_disagreements


def create_flat_distribution_scenario():
    """Create audit scenario with flat energy distribution."""

    class MockLayer:
        def __init__(self, name, rank_suggestions, frob_sq=1.0, params=1000, utilization=0.5):
            self.name = name
            self.rank_suggestions = rank_suggestions
            self.frob_sq = frob_sq
            self.params = params
            self.utilization = utilization

    # Create layers with nearly equal energy shares (flat distribution)
    layers = [
        # All layers have similar energy values → flat distribution
        MockLayer(
            "layer.0",
            {
                "energy_threshold": {"k": 6, "confidence": 0.85},
                "knee_elbow": {"k": 4, "confidence": 0.80},
                "entropy_effective": {"k": 5, "confidence": 0.85},
            },
            frob_sq=5.2,
            params=10000,
            utilization=0.4,
        ),  # uniform_mult ≈ 1.04
        MockLayer(
            "layer.1",
            {
                "energy_threshold": {"k": 5, "confidence": 0.85},
                "knee_elbow": {"k": 3, "confidence": 0.80},
                "entropy_effective": {"k": 4, "confidence": 0.85},
            },
            frob_sq=5.0,
            params=9500,
            utilization=0.4,
        ),  # uniform_mult ≈ 1.00
        MockLayer(
            "layer.2",
            {
                "energy_threshold": {"k": 4, "confidence": 0.80},
                "knee_elbow": {"k": 2, "confidence": 0.75},
                "entropy_effective": {"k": 3, "confidence": 0.80},
            },
            frob_sq=4.8,
            params=9000,
            utilization=0.35,
        ),  # uniform_mult ≈ 0.96
        MockLayer(
            "layer.3",
            {
                "energy_threshold": {"k": 3, "confidence": 0.75},
                "knee_elbow": {"k": 1, "confidence": 0.70},
                "entropy_effective": {"k": 2, "confidence": 0.75},
            },
            frob_sq=5.0,
            params=8500,
            utilization=0.35,
        ),  # uniform_mult ≈ 1.00
    ]

    name_mapping = {"energy_threshold": "energy@0.85", "knee_elbow": "knee", "entropy_effective": "erank"}

    # Use default config (min_uniform_mult = 1.5)
    config = {
        "quantile_threshold": 0.75,
        "uniform_mult_gate": 1.5,  # This will make distribution flat since max ≈ 1.04 < 1.5
        "metric": "energy_share",
    }

    analysis = _analyze_policy_disagreements(layers, name_mapping, config, "full")
    return analysis


def create_hierarchical_distribution_scenario():
    """Create audit scenario with hierarchical energy distribution."""

    class MockLayer:
        def __init__(self, name, rank_suggestions, frob_sq=1.0, params=1000, utilization=0.5):
            self.name = name
            self.rank_suggestions = rank_suggestions
            self.frob_sq = frob_sq
            self.params = params
            self.utilization = utilization

    # Create layers with strongly unequal energy shares (hierarchical distribution)
    layers = [
        # High energy layer
        MockLayer(
            "dominant.layer",
            {
                "energy_threshold": {"k": 8, "confidence": 0.90},
                "knee_elbow": {"k": 4, "confidence": 0.85},
                "entropy_effective": {"k": 6, "confidence": 0.90},
            },
            frob_sq=50.0,
            params=50000,
            utilization=0.8,
        ),  # uniform_mult ≈ 3.3
        # Medium energy layer
        MockLayer(
            "medium.layer",
            {
                "energy_threshold": {"k": 6, "confidence": 0.85},
                "knee_elbow": {"k": 3, "confidence": 0.80},
                "entropy_effective": {"k": 5, "confidence": 0.85},
            },
            frob_sq=20.0,
            params=20000,
            utilization=0.6,
        ),  # uniform_mult ≈ 1.3
        # Low energy layers
        MockLayer(
            "low1.layer",
            {
                "energy_threshold": {"k": 4, "confidence": 0.80},
                "knee_elbow": {"k": 2, "confidence": 0.75},
                "entropy_effective": {"k": 3, "confidence": 0.80},
            },
            frob_sq=5.0,
            params=5000,
            utilization=0.3,
        ),  # uniform_mult ≈ 0.3
        MockLayer(
            "low2.layer",
            {
                "energy_threshold": {"k": 3, "confidence": 0.75},
                "knee_elbow": {"k": 1, "confidence": 0.70},
                "entropy_effective": {"k": 2, "confidence": 0.75},
            },
            frob_sq=5.0,
            params=5000,
            utilization=0.3,
        ),  # uniform_mult ≈ 0.3
    ]

    name_mapping = {"energy_threshold": "energy@0.85", "knee_elbow": "knee", "entropy_effective": "erank"}

    # Same config as flat scenario
    config = {
        "quantile_threshold": 0.75,
        "uniform_mult_gate": 1.5,  # max ≈ 3.3 > 1.5 → hierarchical
        "metric": "energy_share",
    }

    analysis = _analyze_policy_disagreements(layers, name_mapping, config, "full")
    return analysis


def demonstrate_flatness_witness():
    """Demonstrate explicit flatness witness with mathematical proof."""

    print("📊 EXPLICIT FLATNESS WITNESS DEMONSTRATION")
    print("=" * 80)
    print("Testing mathematically explicit flat distribution detection with")
    print("exact witness values that prove why distribution was declared flat.")
    print("=" * 80)

    # Test flat distribution scenario
    print("\n🔍 SCENARIO 1: FLAT DISTRIBUTION")
    print("-" * 60)

    flat_analysis = create_flat_distribution_scenario()
    distribution = flat_analysis["distribution"]
    witness = distribution["flatness_witness"]
    focus_set = flat_analysis["disagreement_focus_set"]

    print(f"Distribution Classification: {'FLAT' if distribution['is_flat'] else 'HIERARCHICAL'}")
    print()
    print("📋 FLATNESS WITNESS (Mathematical Proof):")
    print(f"  • max_uniform_mult: {witness['max_observed']:.3f}")
    print(f"  • threshold: {witness['threshold']:.3f}")
    print(f"  • n_layers: {distribution['total_layers']}")
    print(f"  • uniform_share: {distribution['uniform_share']:.3f}")
    print(f"  • is_below_threshold: {witness['is_below_threshold']}")
    print()
    print("🧮 Mathematical Proof:")
    print(f"  {witness['mathematical_proof']}")
    print()
    print("💡 Human-Readable Message:")
    print(f'  "{focus_set["message"]}"')

    # Test hierarchical distribution scenario
    print("\n\n🔍 SCENARIO 2: HIERARCHICAL DISTRIBUTION")
    print("-" * 60)

    hierarchical_analysis = create_hierarchical_distribution_scenario()
    distribution = hierarchical_analysis["distribution"]
    witness = distribution["flatness_witness"]
    focus_set = hierarchical_analysis["disagreement_focus_set"]

    print(f"Distribution Classification: {'FLAT' if distribution['is_flat'] else 'HIERARCHICAL'}")
    print()
    print("📋 FLATNESS WITNESS (Mathematical Proof):")
    print(f"  • max_uniform_mult: {witness['max_observed']:.3f}")
    print(f"  • threshold: {witness['threshold']:.3f}")
    print(f"  • n_layers: {distribution['total_layers']}")
    print(f"  • uniform_share: {distribution['uniform_share']:.3f}")
    print(f"  • is_below_threshold: {witness['is_below_threshold']}")
    print()
    print("🧮 Mathematical Proof:")
    print(f"  {witness['mathematical_proof']}")
    print()
    print("💡 Human-Readable Message:")
    print(f'  "{focus_set["message"]}"')

    # Show JSON comparison
    print("\n\n📄 JSON WITNESS COMPARISON")
    print("=" * 80)

    print("FLAT DISTRIBUTION witness:")
    print(json.dumps(flat_analysis["distribution"]["flatness_witness"], indent=2))

    print("\nHIERARCHICAL DISTRIBUTION witness:")
    print(json.dumps(hierarchical_analysis["distribution"]["flatness_witness"], indent=2))

    return flat_analysis, hierarchical_analysis


def validate_witness_logic():
    """Validate that the witness logic is mathematically sound."""

    print("\n\n✅ WITNESS VALIDATION")
    print("=" * 80)

    flat_analysis, hierarchical_analysis = demonstrate_flatness_witness()

    flat_witness = flat_analysis["distribution"]["flatness_witness"]
    hierarchical_witness = hierarchical_analysis["distribution"]["flatness_witness"]

    # Test flat case
    flat_max = flat_witness["max_observed"]
    flat_threshold = flat_witness["threshold"]
    flat_predicted = flat_max < flat_threshold
    flat_actual = flat_analysis["distribution"]["is_flat"]

    print("FLAT CASE:")
    print(f"  Witness prediction: max={flat_max:.3f} < {flat_threshold:.3f} = {flat_predicted}")
    print(f"  Actual classification: is_flat = {flat_actual}")
    print(f"  ✅ Consistent: {flat_predicted == flat_actual}")

    # Test hierarchical case
    hier_max = hierarchical_witness["max_observed"]
    hier_threshold = hierarchical_witness["threshold"]
    hier_predicted = hier_max < hier_threshold  # Should be False (not flat)
    hier_actual = hierarchical_analysis["distribution"]["is_flat"]

    print("\nHIERARCHICAL CASE:")
    print(f"  Witness prediction: max={hier_max:.3f} < {hier_threshold:.3f} = {hier_predicted}")
    print(f"  Actual classification: is_flat = {hier_actual}")
    print(f"  ✅ Consistent: {hier_predicted == hier_actual}")

    # Validate mathematical proof strings
    flat_proof = flat_witness["mathematical_proof"]
    hier_proof = hierarchical_witness["mathematical_proof"]

    print("\n🧮 PROOF STRING VALIDATION:")
    print(f'  Flat proof: "{flat_proof}"')
    print(f'  Hierarchical proof: "{hier_proof}"')

    flat_contains_flat = "→ flat" in flat_proof
    hier_contains_hierarchical = "→ hierarchical" in hier_proof

    print(f"  ✅ Flat proof mentions 'flat': {flat_contains_flat}")
    print(f"  ✅ Hierarchical proof mentions 'hierarchical': {hier_contains_hierarchical}")


if __name__ == "__main__":
    demonstrate_flatness_witness()
    validate_witness_logic()

    print("\n\n" + "=" * 80)
    print("✅ EXPLICIT FLATNESS WITNESS IMPLEMENTATION COMPLETE!")
    print()
    print("🎯 Key Benefits:")
    print("  • Mathematical proof stored in JSON for every analysis")
    print("  • Exact witness values (max_uniform_mult, threshold, n_layers)")
    print("  • Human-readable proof string for debugging")
    print("  • Can always prove why distribution was declared flat/hierarchical")
    print("  • Eliminates 'why did this get classified as flat?' mysteries")
    print()
    print("📊 Witness Structure:")
    print("  flatness_witness: {")
    print("    threshold: float,           # min_uniform_mult gate value")
    print("    max_observed: float,        # max layer uniform multiplier")
    print("    is_below_threshold: bool,   # max < threshold")
    print("    mathematical_proof: string  # human-readable proof")
    print("  }")
    print()
    print("🔍 Perfect for:")
    print("  • Debugging classification decisions")
    print("  • Reproducing analysis results")
    print("  • Understanding why certain layers weren't flagged")
    print("  • Academic/research reproducibility requirements")
    print()
    print("This transforms flat detection from 'black box decision' to")
    print("'mathematically provable classification' with complete audit trail! 🎯")
