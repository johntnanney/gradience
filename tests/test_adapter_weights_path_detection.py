"""
Unit tests for adapter weights path detection functionality.

Tests the find_adapter_weights_path function's search order and error handling:
1. <variant_dir>/peft/adapter_model.safetensors
2. newest <variant_dir>/checkpoint-*/adapter_model.safetensors
3. newest <variant_dir>/**/adapter_model.safetensors
"""

import os
import tempfile
import time
import unittest
from pathlib import Path

from gradience.peft_utils import check_heterogeneous_ranks, find_adapter_weights_path


class TestAdapterWeightsPathDetection(unittest.TestCase):
    """Test adapter weights path detection with various directory structures."""

    def setUp(self):
        """Create a temporary directory for test files."""
        self.test_dir = tempfile.mkdtemp()
        self.variant_dir = Path(self.test_dir)

    def tearDown(self):
        """Clean up temporary test files."""
        import shutil

        shutil.rmtree(self.test_dir)

    def _create_fake_adapter_file(self, path: Path, mtime: float = None) -> None:
        """Create a fake adapter file with explicit mtime for deterministic ordering."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
        if mtime is not None:
            os.utime(path, (mtime, mtime))
            os.utime(path.parent, (mtime, mtime))

    def test_strategy_1_peft_directory_priority(self):
        """Test that peft/adapter_model.safetensors takes highest priority."""
        # Create files in multiple locations
        self._create_fake_adapter_file(self.variant_dir / "peft" / "adapter_model.safetensors")
        self._create_fake_adapter_file(self.variant_dir / "checkpoint-50" / "adapter_model.safetensors")
        self._create_fake_adapter_file(self.variant_dir / "other" / "adapter_model.safetensors")

        result = find_adapter_weights_path(self.variant_dir)

        expected = str(self.variant_dir / "peft" / "adapter_model.safetensors")
        self.assertEqual(result, expected)

    def test_strategy_2_newest_checkpoint_directory(self):
        """Test checkpoint-* directory selection by newest modification time."""
        base = time.time()
        # Create multiple checkpoint directories with different timestamps
        self._create_fake_adapter_file(self.variant_dir / "checkpoint-100" / "adapter_model.safetensors", mtime=base)
        self._create_fake_adapter_file(
            self.variant_dir / "checkpoint-200" / "adapter_model.safetensors", mtime=base + 1
        )
        newest_path = self.variant_dir / "checkpoint-300" / "adapter_model.safetensors"
        self._create_fake_adapter_file(newest_path, mtime=base + 2)

        result = find_adapter_weights_path(self.variant_dir)

        expected = str(newest_path)
        self.assertEqual(result, expected)

    def test_strategy_3_fallback_to_any_subdirectory(self):
        """Test fallback to newest adapter_model.safetensors in any subdirectory."""
        base = time.time()
        # Create files in various subdirectories (no peft or checkpoint-* dirs)
        self._create_fake_adapter_file(self.variant_dir / "old_dir" / "adapter_model.safetensors", mtime=base)
        self._create_fake_adapter_file(self.variant_dir / "some_dir" / "adapter_model.safetensors", mtime=base + 1)
        newest_path = self.variant_dir / "newest_dir" / "adapter_model.safetensors"
        self._create_fake_adapter_file(newest_path, mtime=base + 2)

        result = find_adapter_weights_path(self.variant_dir)

        expected = str(newest_path)
        self.assertEqual(result, expected)

    def test_checkpoint_directory_naming_variations(self):
        """Test that different checkpoint-* naming patterns are handled."""
        base = time.time()
        # Test various checkpoint directory naming patterns
        self._create_fake_adapter_file(self.variant_dir / "checkpoint-1" / "adapter_model.safetensors", mtime=base)
        self._create_fake_adapter_file(self.variant_dir / "checkpoint-50" / "adapter_model.safetensors", mtime=base + 1)
        newest_path = self.variant_dir / "checkpoint-1000" / "adapter_model.safetensors"
        self._create_fake_adapter_file(newest_path, mtime=base + 2)

        result = find_adapter_weights_path(self.variant_dir)

        expected = str(newest_path)
        self.assertEqual(result, expected)

    def test_file_not_found_error(self):
        """Test that FileNotFoundError is raised when no adapter files exist."""
        # Create empty variant directory with no adapter files

        with self.assertRaises(FileNotFoundError) as context:
            find_adapter_weights_path(self.variant_dir)

        error_message = str(context.exception)
        self.assertIn("No adapter_model.safetensors found", error_message)
        self.assertIn("1) peft/", error_message)
        self.assertIn("2) checkpoint-*/", error_message)
        self.assertIn("3) any subdirectory", error_message)

    def test_search_order_comprehensive(self):
        """Test complete search order: peft > checkpoint-* > any subdirectory."""
        base = time.time()
        # Strategy 3: Create file in random subdirectory (lowest priority)
        self._create_fake_adapter_file(self.variant_dir / "random" / "adapter_model.safetensors", mtime=base)

        # Strategy 2: Create newer file in checkpoint directory (medium priority)
        self._create_fake_adapter_file(
            self.variant_dir / "checkpoint-100" / "adapter_model.safetensors", mtime=base + 1
        )

        # Strategy 1: Create newest file in peft directory (highest priority)
        peft_path = self.variant_dir / "peft" / "adapter_model.safetensors"
        self._create_fake_adapter_file(peft_path, mtime=base + 2)

        result = find_adapter_weights_path(self.variant_dir)

        # Should return peft path despite being created last
        expected = str(peft_path)
        self.assertEqual(result, expected)

    def test_path_types_accepted(self):
        """Test that function accepts both string and Path objects."""
        peft_path = self.variant_dir / "peft" / "adapter_model.safetensors"
        self._create_fake_adapter_file(peft_path)

        # Test with Path object
        result_path = find_adapter_weights_path(self.variant_dir)
        self.assertEqual(result_path, str(peft_path))

        # Test with string path
        result_str = find_adapter_weights_path(str(self.variant_dir))
        self.assertEqual(result_str, str(peft_path))

    def test_nested_directory_structure(self):
        """Test detection in deeply nested directory structures."""
        # Create adapter file in nested structure
        nested_path = self.variant_dir / "deep" / "nested" / "structure" / "adapter_model.safetensors"
        self._create_fake_adapter_file(nested_path)

        result = find_adapter_weights_path(self.variant_dir)

        expected = str(nested_path)
        self.assertEqual(result, expected)

    def test_empty_checkpoint_directories_ignored(self):
        """Test that checkpoint directories without adapter files are ignored."""
        # Create checkpoint directories without adapter files
        (self.variant_dir / "checkpoint-50").mkdir()
        (self.variant_dir / "checkpoint-100").mkdir()

        # Create adapter file in fallback location
        fallback_path = self.variant_dir / "fallback" / "adapter_model.safetensors"
        self._create_fake_adapter_file(fallback_path)

        result = find_adapter_weights_path(self.variant_dir)

        # Should fall back to strategy 3 since checkpoint dirs are empty
        expected = str(fallback_path)
        self.assertEqual(result, expected)


class TestHeterogeneousRankCheck(unittest.TestCase):
    """Test heterogeneous rank checking and degrade-to-uniform logic."""

    def setUp(self):
        """Create a temporary directory for test files."""
        self.test_dir = tempfile.mkdtemp()
        self.adapter_path = Path(self.test_dir) / "adapter_model.safetensors"

    def tearDown(self):
        """Clean up temporary test files."""
        import shutil

        shutil.rmtree(self.test_dir)

    def _create_mock_adapter_weights(self, rank_sequence):
        """Create mock adapter weights with specified rank sequence.

        Args:
            rank_sequence: List of ranks for consecutive lora_A layers
        """
        try:
            import torch
            from safetensors.torch import save_file

            tensors = {}
            for i, rank in enumerate(rank_sequence):
                # Create mock lora_A weights with specified rank (rank x hidden_size)
                tensors[f"base_model.model.layer.{i}.attention.self.query.lora_A.weight"] = torch.randn(rank, 64)
                # Also create corresponding lora_B weights (hidden_size x rank)
                tensors[f"base_model.model.layer.{i}.attention.self.query.lora_B.weight"] = torch.randn(64, rank)

            save_file(tensors, self.adapter_path)
            return True
        except ImportError:
            # Skip test if safetensors/torch not available
            return False

    def test_heterogeneous_ranks_pass(self):
        """Test that heterogeneous ranks pass validation."""
        if not self._create_mock_adapter_weights([2, 4, 8, 4]):  # Mixed ranks
            self.skipTest("safetensors/torch not available")

        result = check_heterogeneous_ranks(str(self.adapter_path), [2, 4, 8])

        self.assertTrue(result["passed"])
        self.assertEqual(len(result["unique_ranks"]), 3)  # 2, 4, 8
        self.assertEqual(set(result["unique_ranks"]), {2, 4, 8})
        self.assertEqual(result["total_modules"], 4)
        self.assertIsNone(result["reason"])
        self.assertNotIn("degrade_to_uniform", result)

    def test_uniform_ranks_degrade_to_uniform(self):
        """Test that uniform ranks trigger degrade-to-uniform logic."""
        if not self._create_mock_adapter_weights([4, 4, 4, 4]):  # All rank 4
            self.skipTest("safetensors/torch not available")

        result = check_heterogeneous_ranks(str(self.adapter_path), [2, 4, 8])

        # Should pass but signal degeneration
        self.assertTrue(result["passed"])
        self.assertTrue(result["degrade_to_uniform"])
        self.assertEqual(len(result["unique_ranks"]), 1)
        self.assertEqual(result["unique_ranks"], [4])
        self.assertEqual(result["total_modules"], 4)
        self.assertIn("collapsed to uniform ranks", result["reason"])
        self.assertIn("acceptable degeneration", result["reason"])

    def test_unexpected_ranks_fail(self):
        """Test that unexpected ranks fail validation."""
        if not self._create_mock_adapter_weights([2, 4, 16]):  # 16 not in allowed
            self.skipTest("safetensors/torch not available")

        result = check_heterogeneous_ranks(str(self.adapter_path), [2, 4, 8])

        self.assertFalse(result["passed"])
        self.assertEqual(len(result["unique_ranks"]), 3)
        self.assertIn("Unexpected ranks found", result["reason"])
        self.assertIn("16", result["reason"])
        self.assertNotIn("degrade_to_uniform", result)

    def test_no_lora_weights_fail(self):
        """Test that adapters with no lora_A weights fail."""
        try:
            import torch
            from safetensors.torch import save_file

            # Create weights without lora_A
            tensors = {
                "base_model.model.layer.0.attention.self.query.weight": torch.randn(64, 64),
                "base_model.model.layer.0.attention.bias": torch.randn(64),
            }
            save_file(tensors, self.adapter_path)
        except ImportError:
            self.skipTest("safetensors/torch not available")

        result = check_heterogeneous_ranks(str(self.adapter_path), [2, 4, 8])

        self.assertFalse(result["passed"])
        self.assertEqual(result["total_modules"], 0)
        self.assertIn("No lora_A weights found", result["reason"])

    def test_missing_file_error_handling(self):
        """Test error handling for missing adapter files."""
        nonexistent_path = str(self.adapter_path.with_name("nonexistent.safetensors"))

        result = check_heterogeneous_ranks(nonexistent_path, [2, 4, 8])

        self.assertFalse(result["passed"])
        self.assertEqual(result["total_modules"], 0)
        self.assertIn("Failed to check ranks", result["reason"])


if __name__ == "__main__":
    unittest.main()
