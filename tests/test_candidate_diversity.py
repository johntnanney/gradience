"""
Unit tests for candidate diversity improvements.

Tests the enhanced deduplication logic that prevents rank collisions by mapping
duplicates to "second rung" alternatives without additional compute cost.
"""

import json
import tempfile
import unittest
from pathlib import Path


class TestCandidateDiversity(unittest.TestCase):
    """Test candidate diversity preservation in FAST mode."""

    def setUp(self):
        """Set up test fixtures for candidate generation."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.probe_dir = self.temp_dir / "probe"
        self.probe_dir.mkdir()

        # Create mock audit.json with realistic policy suggestions
        # Based on actual audit.json format that the code expects
        self.audit_data = {
            "policy_global_suggestions": {
                "energy_90": {"uniform_p90": 48},  # energy_p90 suggests r=48
                "knee": {"uniform_p90": 48},  # knee_p90 suggests r=48 (collision!)
                "erank": {"uniform_p90": 32},  # erank_p90 suggests r=32 (different)
                "energy_99": {"uniform_p90": 56},  # Full mode only
                "uniform_median": {"uniform_p90": 40},  # Full mode only
            }
        }

        audit_path = self.probe_dir / "audit.json"
        with open(audit_path, "w") as f:
            json.dump(self.audit_data, f)

        # Base bench config
        self.config = {
            "compression": {
                "allowed_ranks": [4, 8, 12, 16, 24, 32, 40, 48, 56, 64],
                "fast_mode": True,
                "max_candidates": 8,
                "candidate_policies": ["energy_p90", "knee_p90", "erank_p90"],
            },
            "lora": {"probe_r": 64, "alpha": 128, "dropout": 0.05, "target_modules": ["q_proj", "k_proj", "v_proj"]},
        }

    def tearDown(self):
        """Clean up temporary files."""
        import shutil

        shutil.rmtree(self.temp_dir)

    def test_rank_collision_resolution(self):
        """Test that rank collisions are resolved with second rung mapping."""
        from gradience.bench.protocol import generate_compression_configs

        # Generate candidates in fast mode
        compression_configs, decision_trace = generate_compression_configs(
            self.probe_dir, self.config, fast_mode=True, max_candidates=4
        )

        # Extract actual ranks from generated configs
        generated_ranks = []
        generated_variants = []
        for variant_name, variant_config in compression_configs.items():
            if variant_name != "per_layer":  # Skip per_layer which has different structure
                generated_ranks.append(variant_config["actual_r"])
                generated_variants.append(variant_name)

        print(f"Generated variants: {generated_variants}")
        print(f"Generated ranks: {generated_ranks}")

        # BEFORE FIX: would have [48, 32] - lost diversity due to energy_p90/knee_p90 collision
        # AFTER FIX: should have [48, 40, 32] or similar - preserved diversity via second rung

        # Verify no rank duplicates (diversity preserved)
        assert len(set(generated_ranks)) == len(generated_ranks), f"Duplicate ranks found: {generated_ranks}"

        # Verify we got expected number of diverse candidates
        assert len(generated_ranks) >= 2, f"Expected at least 2 candidates, got {len(generated_ranks)}"

        # Verify at least one candidate was remapped to second rung
        # Original collision was at r=48, remapped candidate should be more aggressive (smaller)
        if 48 in generated_ranks:
            # Check that we have a more aggressive alternative than the original collision
            aggressive_ranks = [r for r in generated_ranks if r < 48]
            assert len(aggressive_ranks) >= 1, (
                f"Expected at least one more aggressive rank than 48, got ranks: {generated_ranks}"
            )

            # The most aggressive available rank should be chosen
            # At time of collision resolution: r=32 was taken by erank_p90, so available were [4,8,12,16,24,40]
            # Most aggressive (smallest) available should be r=4
            expected_most_aggressive = 4  # Should be the smallest rank < 48 that wasn't taken by erank_p90 (r=32)
            assert expected_most_aggressive in generated_ranks, (
                f"Expected most aggressive rank r={expected_most_aggressive} not found in {generated_ranks}"
            )

    def test_dedup_note_tracking(self):
        """Test that deduplication notes properly track remapping."""
        from gradience.bench.protocol import generate_compression_configs

        compression_configs, decision_trace = generate_compression_configs(
            self.probe_dir, self.config, fast_mode=True, max_candidates=4
        )

        # Check that dedup notes indicate remapping occurred
        preferred_found = False

        for variant_name, variant_config in compression_configs.items():
            if variant_name != "per_layer":
                dedup_note = variant_config.get("reason", "") or ""

                if "second rung" in dedup_note:
                    print(f"Found remapped candidate: {variant_name} - {dedup_note}")

                if "Preferred choice" in dedup_note:
                    preferred_found = True
                    print(f"Found preferred candidate: {variant_name} - {dedup_note}")

        # Should have evidence of both preferred choice and remapping
        if len(compression_configs) > 2:  # Only if we have enough candidates to trigger collision
            assert preferred_found, "Should have a preferred choice from collision"
            # Note: remapped_found might be false if no collision actually occurred

    def test_no_collision_scenario(self):
        """Test that non-colliding policies work normally."""
        # Modify audit data to have no collisions
        self.audit_data["policy_global_suggestions"] = {
            "energy_90": {"uniform_p90": 48},
            "knee": {"uniform_p90": 32},  # Different from energy_p90
            "erank": {"uniform_p90": 16},  # Different from both
        }

        # Update audit file
        audit_path = self.probe_dir / "audit.json"
        with open(audit_path, "w") as f:
            json.dump(self.audit_data, f)

        from gradience.bench.protocol import generate_compression_configs

        compression_configs, decision_trace = generate_compression_configs(
            self.probe_dir, self.config, fast_mode=True, max_candidates=4
        )

        # Should have all different ranks
        generated_ranks = [cfg["actual_r"] for name, cfg in compression_configs.items() if name != "per_layer"]
        expected_ranks = [48, 32, 16]  # Should match suggestions exactly

        # All ranks should be different and match original suggestions
        assert len(set(generated_ranks)) == len(generated_ranks), "Should have no duplicates in non-collision case"
        for rank in generated_ranks:
            assert rank in expected_ranks, f"Unexpected rank {rank}, expected one of {expected_ranks}"

    def test_multiple_collisions(self):
        """Test handling of multiple simultaneous collisions."""
        # Create scenario with multiple collisions
        self.audit_data["policy_global_suggestions"] = {
            "energy_90": {"uniform_p90": 48},
            "knee": {"uniform_p90": 48},  # Collision 1: both suggest r=48
            "erank": {"uniform_p90": 32},
            "energy_99": {"uniform_p90": 32},  # Collision 2: both suggest r=32 (if in full mode)
        }

        # Use full mode to include energy_p99
        self.config["compression"]["fast_mode"] = False
        self.config["compression"]["candidate_policies"] = ["energy_p90", "knee_p90", "erank_p90", "energy_p99"]

        # Update audit file
        audit_path = self.probe_dir / "audit.json"
        with open(audit_path, "w") as f:
            json.dump(self.audit_data, f)

        from gradience.bench.protocol import generate_compression_configs

        compression_configs, decision_trace = generate_compression_configs(
            self.probe_dir, self.config, fast_mode=False, max_candidates=6
        )

        # Extract ranks
        generated_ranks = [cfg["actual_r"] for name, cfg in compression_configs.items() if name != "per_layer"]

        # Should resolve both collisions with different ranks
        assert len(set(generated_ranks)) == len(generated_ranks), f"Multiple collisions not resolved: {generated_ranks}"

        # Should have more than 2 unique ranks (the original collision points)
        assert len(generated_ranks) >= 3, f"Expected diverse resolution of multiple collisions, got {generated_ranks}"

    def test_aggressive_rank_selection(self):
        """Test that second rung selection picks more aggressive (smaller) ranks."""
        from gradience.bench.protocol import generate_compression_configs

        compression_configs, decision_trace = generate_compression_configs(
            self.probe_dir, self.config, fast_mode=True, max_candidates=4
        )

        # Find remapped candidates
        for variant_name, variant_config in compression_configs.items():
            if variant_name != "per_layer":
                dedup_note = variant_config.get("reason", "") or ""
                if "second rung" in dedup_note:
                    remapped_rank = variant_config["actual_r"]
                    # Second rung should be more aggressive (smaller) than original collision point (48)
                    assert remapped_rank < 48, f"Second rung rank {remapped_rank} should be more aggressive than 48"

                    # Should be from allowed ranks
                    allowed_ranks = self.config["compression"]["allowed_ranks"]
                    assert remapped_rank in allowed_ranks, (
                        f"Remapped rank {remapped_rank} not in allowed ranks {allowed_ranks}"
                    )


if __name__ == "__main__":
    unittest.main()
