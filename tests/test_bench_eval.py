import json
import tempfile
import unittest
from pathlib import Path

from gradience.bench.protocol import write_probe_eval_json


class TestBenchEval(unittest.TestCase):
    def test_write_probe_eval_json_smoke(self):
        """Test eval.json writing with minimal data."""
        # Mock evaluation results
        eval_results = {
            "eval_accuracy": 0.85,
            "eval_loss": 0.42,
            "eval_runtime": 1.23,
            "eval_samples_per_second": 123.45,
            "eval_steps_per_second": 10.5
        }
        
        # Mock configuration
        config = {
            "train": {"seed": 42},
            "lora": {"probe_r": 16}
        }
        
        with tempfile.TemporaryDirectory() as td:
            probe_dir = Path(td)
            eval_dataset_size = 1000
            
            # Write eval.json
            eval_path = write_probe_eval_json(
                probe_dir=probe_dir,
                eval_results=eval_results,
                eval_dataset_size=eval_dataset_size,
                config=config
            )
            
            # Verify file was created
            self.assertTrue(eval_path.exists())
            self.assertEqual(eval_path.name, "eval.json")
            
            # Verify content
            with open(eval_path) as f:
                eval_data = json.load(f)
            
            # Check required fields
            self.assertEqual(eval_data["accuracy"], 0.85)
            self.assertEqual(eval_data["eval_loss"], 0.42)
            self.assertEqual(eval_data["eval_samples"], 1000)
            self.assertEqual(eval_data["seed"], 42)
            self.assertEqual(eval_data["rank"], 16)
            
            # Check optional performance fields
            self.assertEqual(eval_data["eval_runtime"], 1.23)
            self.assertEqual(eval_data["eval_samples_per_second"], 123.45)
            self.assertEqual(eval_data["eval_steps_per_second"], 10.5)

    def test_write_probe_eval_json_missing_optional_fields(self):
        """Test eval.json writing handles missing optional fields gracefully."""
        # Minimal evaluation results (only accuracy)
        eval_results = {
            "eval_accuracy": 0.75
            # Missing optional fields
        }
        
        config = {
            "train": {"seed": 123},
            "lora": {"probe_r": 8}
        }
        
        with tempfile.TemporaryDirectory() as td:
            probe_dir = Path(td)
            eval_dataset_size = 500
            
            # Should not crash with missing fields
            eval_path = write_probe_eval_json(
                probe_dir=probe_dir,
                eval_results=eval_results,
                eval_dataset_size=eval_dataset_size,
                config=config
            )
            
            # Verify content
            with open(eval_path) as f:
                eval_data = json.load(f)
            
            # Required fields should be present
            self.assertEqual(eval_data["accuracy"], 0.75)
            self.assertEqual(eval_data["eval_samples"], 500)
            self.assertEqual(eval_data["seed"], 123)
            self.assertEqual(eval_data["rank"], 8)
            
            # Optional fields should be None
            self.assertIsNone(eval_data["eval_loss"])
            self.assertIsNone(eval_data["eval_runtime"])


if __name__ == "__main__":
    unittest.main()