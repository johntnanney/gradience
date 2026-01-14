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

from gradience.peft_utils import find_adapter_weights_path


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
    
    def _create_fake_adapter_file(self, path: Path) -> None:
        """Create a fake empty adapter_model.safetensors file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
        # Set a distinct modification time to help with newest file detection
        time.sleep(0.001)  # Ensure different mtimes
    
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
        # Create multiple checkpoint directories with different timestamps
        self._create_fake_adapter_file(self.variant_dir / "checkpoint-100" / "adapter_model.safetensors")
        time.sleep(0.01)  # Ensure distinct modification times
        self._create_fake_adapter_file(self.variant_dir / "checkpoint-200" / "adapter_model.safetensors")
        time.sleep(0.01)
        newest_path = self.variant_dir / "checkpoint-300" / "adapter_model.safetensors"
        self._create_fake_adapter_file(newest_path)
        
        result = find_adapter_weights_path(self.variant_dir)
        
        expected = str(newest_path)
        self.assertEqual(result, expected)
    
    def test_strategy_3_fallback_to_any_subdirectory(self):
        """Test fallback to newest adapter_model.safetensors in any subdirectory."""
        # Create files in various subdirectories (no peft or checkpoint-* dirs)
        self._create_fake_adapter_file(self.variant_dir / "old_dir" / "adapter_model.safetensors")
        time.sleep(0.01)
        self._create_fake_adapter_file(self.variant_dir / "some_dir" / "adapter_model.safetensors")
        time.sleep(0.01)
        newest_path = self.variant_dir / "newest_dir" / "adapter_model.safetensors"
        self._create_fake_adapter_file(newest_path)
        
        result = find_adapter_weights_path(self.variant_dir)
        
        expected = str(newest_path)
        self.assertEqual(result, expected)
    
    def test_checkpoint_directory_naming_variations(self):
        """Test that different checkpoint-* naming patterns are handled."""
        # Test various checkpoint directory naming patterns
        self._create_fake_adapter_file(self.variant_dir / "checkpoint-1" / "adapter_model.safetensors")
        time.sleep(0.01)
        self._create_fake_adapter_file(self.variant_dir / "checkpoint-50" / "adapter_model.safetensors") 
        time.sleep(0.01)
        newest_path = self.variant_dir / "checkpoint-1000" / "adapter_model.safetensors"
        self._create_fake_adapter_file(newest_path)
        
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
        # Strategy 3: Create file in random subdirectory (lowest priority)
        self._create_fake_adapter_file(self.variant_dir / "random" / "adapter_model.safetensors")
        
        # Strategy 2: Create newer file in checkpoint directory (medium priority)
        time.sleep(0.01)
        self._create_fake_adapter_file(self.variant_dir / "checkpoint-100" / "adapter_model.safetensors")
        
        # Strategy 1: Create newest file in peft directory (highest priority)
        time.sleep(0.01)
        peft_path = self.variant_dir / "peft" / "adapter_model.safetensors"
        self._create_fake_adapter_file(peft_path)
        
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


if __name__ == "__main__":
    unittest.main()