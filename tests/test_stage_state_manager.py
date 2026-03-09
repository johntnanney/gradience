"""
Unit tests for the stage state manager used in resume functionality.

Tests the state tracking system that enables expensive stage skipping
in bench protocols, especially valuable for 90-minute probe training phases.
"""

import json
import tempfile
import unittest
from pathlib import Path

from gradience.bench.stage_state import StageStateManager, create_stage_manager


class TestStageStateManager(unittest.TestCase):
    """Test stage state management functionality."""

    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.output_dir = Path(self.temp_dir)
        self.manager = StageStateManager(self.output_dir)

    def tearDown(self):
        """Clean up test files."""
        import shutil

        shutil.rmtree(self.temp_dir)

    def test_stage_state_lifecycle(self):
        """Test basic stage completion lifecycle."""
        stage_name = "probe_r16_trained"

        # Initially not completed
        self.assertFalse(self.manager.is_stage_completed(stage_name))

        # Mark as completed
        metadata = {"probe_rank": 16, "max_steps": 1000}
        self.manager.mark_stage_completed(stage_name, metadata)

        # Should now be completed
        self.assertTrue(self.manager.is_stage_completed(stage_name))

        # Should be in completed stages set
        completed_stages = self.manager.get_completed_stages()
        self.assertIn(stage_name, completed_stages)

    def test_variant_state_lifecycle(self):
        """Test variant completion lifecycle."""
        variant_name = "energy_p90"

        # Initially not completed
        self.assertFalse(self.manager.is_variant_completed(variant_name))

        # Mark as completed
        metadata = {"actual_r": 8, "max_steps": 500}
        self.manager.mark_variant_completed(variant_name, metadata)

        # Should now be completed
        self.assertTrue(self.manager.is_variant_completed(variant_name))

        # Should be in completed variants set
        completed_variants = self.manager.get_completed_variants()
        self.assertIn(variant_name, completed_variants)

    def test_state_persistence(self):
        """Test that state persists across manager instances."""
        stage_name = "probe_r16_evaluated"
        variant_name = "knee_p90"

        # Mark stages as completed
        self.manager.mark_stage_completed(stage_name)
        self.manager.mark_variant_completed(variant_name)

        # Create new manager instance
        new_manager = StageStateManager(self.output_dir)

        # Should load existing state
        self.assertTrue(new_manager.is_stage_completed(stage_name))
        self.assertTrue(new_manager.is_variant_completed(variant_name))

    def test_state_file_format(self):
        """Test the state file format and structure."""
        stage_name = "probe_r16_audited"
        self.manager.mark_stage_completed(stage_name, {"test": "metadata"})

        # Check state file exists and has correct structure
        state_file = self.output_dir / "stage_state.json"
        self.assertTrue(state_file.exists())

        with open(state_file) as f:
            state_data = json.load(f)

        # Check structure
        self.assertIn("created_at", state_data)
        self.assertIn("last_updated", state_data)
        self.assertIn("stages", state_data)
        self.assertIn("variants", state_data)

        # Check stage data
        self.assertIn(stage_name, state_data["stages"])
        stage_info = state_data["stages"][stage_name]
        self.assertEqual(stage_info["completed"], True)
        self.assertIn("completed_at", stage_info)
        self.assertEqual(stage_info["metadata"]["test"], "metadata")

    def test_should_skip_probe_training(self):
        """Test probe training skip logic."""
        probe_rank = 16

        # Create fake probe artifacts
        probe_dir = self.output_dir / f"probe_r{probe_rank}"
        probe_dir.mkdir(parents=True)
        (probe_dir / "run.jsonl").touch()
        (probe_dir / "adapter_config.json").touch()

        # Mark as completed
        self.manager.mark_stage_completed(f"probe_r{probe_rank}_trained")

        # Should skip training
        self.assertTrue(self.manager.should_skip_probe_training(probe_rank))

    def test_should_skip_probe_evaluation(self):
        """Test probe evaluation skip logic."""
        probe_rank = 16

        # Create fake eval artifact
        probe_dir = self.output_dir / f"probe_r{probe_rank}"
        probe_dir.mkdir(parents=True)
        (probe_dir / "eval.json").touch()

        # Mark as completed
        self.manager.mark_stage_completed(f"probe_r{probe_rank}_evaluated")

        # Should skip evaluation
        self.assertTrue(self.manager.should_skip_probe_evaluation(probe_rank))

    def test_should_skip_probe_audit(self):
        """Test probe audit skip logic."""
        probe_rank = 16

        # Create fake audit artifact
        probe_dir = self.output_dir / f"probe_r{probe_rank}"
        probe_dir.mkdir(parents=True)
        (probe_dir / "audit.json").touch()

        # Mark as completed
        self.manager.mark_stage_completed(f"probe_r{probe_rank}_audited")

        # Should skip audit
        self.assertTrue(self.manager.should_skip_probe_audit(probe_rank))

    def test_should_skip_config_generation(self):
        """Test config generation skip logic."""
        # Create fake config artifact
        (self.output_dir / "compression_configs.json").touch()

        # Mark as completed
        self.manager.mark_stage_completed("compression_configs_generated")

        # Should skip config generation
        self.assertTrue(self.manager.should_skip_config_generation())

    def test_should_skip_variant_training(self):
        """Test variant training skip logic."""
        variant_name = "energy_p90"

        # Create fake variant artifacts
        variant_dir = self.output_dir / variant_name
        variant_dir.mkdir(parents=True)
        (variant_dir / "eval.json").touch()
        (variant_dir / "adapter_config.json").touch()

        # Mark as completed
        self.manager.mark_variant_completed(variant_name)

        # Should skip variant training
        self.assertTrue(self.manager.should_skip_variant_training(variant_name))

    def test_artifact_validation(self):
        """Test that skip logic validates artifact existence."""
        probe_rank = 16

        # Mark as completed but don't create artifacts
        self.manager.mark_stage_completed(f"probe_r{probe_rank}_trained")

        # Should not skip because artifacts are missing
        self.assertFalse(self.manager.should_skip_probe_training(probe_rank))

    def test_clean_invalid_state(self):
        """Test cleanup of state entries with missing artifacts."""
        probe_rank = 16
        variant_name = "energy_p90"

        # Mark stages as completed
        self.manager.mark_stage_completed(f"probe_r{probe_rank}_trained")
        self.manager.mark_variant_completed(variant_name)

        # Verify they're marked as completed
        self.assertTrue(self.manager.is_stage_completed(f"probe_r{probe_rank}_trained"))
        self.assertTrue(self.manager.is_variant_completed(variant_name))

        # Run cleanup (should remove entries since artifacts don't exist)
        self.manager.clean_invalid_state()

        # Should no longer be marked as completed
        self.assertFalse(self.manager.is_stage_completed(f"probe_r{probe_rank}_trained"))
        self.assertFalse(self.manager.is_variant_completed(variant_name))

    def test_resume_summary(self):
        """Test resume summary generation."""
        # Mark some stages and variants as completed
        self.manager.mark_stage_completed("probe_r16_trained")
        self.manager.mark_stage_completed("probe_r16_evaluated")
        self.manager.mark_variant_completed("energy_p90")
        self.manager.mark_variant_completed("knee_p90")

        summary = self.manager.get_resume_summary()

        # Check summary content
        self.assertEqual(summary["total_completed_stages"], 2)
        self.assertEqual(summary["total_completed_variants"], 2)
        self.assertIn("probe_r16_trained", summary["completed_stages"])
        self.assertIn("probe_r16_evaluated", summary["completed_stages"])
        self.assertIn("energy_p90", summary["completed_variants"])
        self.assertIn("knee_p90", summary["completed_variants"])

        # Check metadata
        self.assertIn("last_updated", summary)
        self.assertIn("created_at", summary)
        self.assertEqual(summary["state_file"], str(self.manager.state_file))

    def test_reset_state(self):
        """Test state reset functionality."""
        # Mark some stages as completed
        self.manager.mark_stage_completed("probe_r16_trained")
        self.manager.mark_variant_completed("energy_p90")

        # Verify they're completed
        self.assertTrue(self.manager.is_stage_completed("probe_r16_trained"))
        self.assertTrue(self.manager.is_variant_completed("energy_p90"))

        # Reset state
        self.manager.reset_state()

        # Should no longer be completed
        self.assertFalse(self.manager.is_stage_completed("probe_r16_trained"))
        self.assertFalse(self.manager.is_variant_completed("energy_p90"))

        # Summary should show empty state
        summary = self.manager.get_resume_summary()
        self.assertEqual(summary["total_completed_stages"], 0)
        self.assertEqual(summary["total_completed_variants"], 0)

    def test_factory_function(self):
        """Test the create_stage_manager factory function."""
        manager = create_stage_manager(self.output_dir)
        self.assertIsInstance(manager, StageStateManager)
        self.assertEqual(manager.output_dir, self.output_dir)

    def test_corrupted_state_file_recovery(self):
        """Test recovery from corrupted state files."""
        # Create corrupted state file
        with open(self.manager.state_file, "w") as f:
            f.write("invalid json content")

        # Should recover gracefully
        new_manager = StageStateManager(self.output_dir)
        self.assertIsInstance(new_manager._state, dict)
        self.assertIn("stages", new_manager._state)
        self.assertIn("variants", new_manager._state)


if __name__ == "__main__":
    unittest.main()
