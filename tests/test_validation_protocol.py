"""
Test validation protocol functionality.

These tests ensure the validation protocol components work correctly
without requiring full training runs.
"""

import json
import tempfile
import unittest
from pathlib import Path
from scripts.validation_protocol import ValidationProtocol


class TestValidationProtocol(unittest.TestCase):
    """Test validation protocol components."""

    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.protocol = ValidationProtocol(
            base_dir=Path(self.temp_dir),
            model="tiny-distilbert",
            dataset="tiny", 
            probe_r=16,
            verbose=False
        )

    def tearDown(self):
        """Clean up test files."""
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_protocol_initialization(self):
        """Test protocol initialization."""
        self.assertEqual(self.protocol.model, "tiny-distilbert")
        self.assertEqual(self.protocol.dataset, "tiny")
        self.assertEqual(self.protocol.probe_r, 16)
        self.assertTrue(self.protocol.protocol_dir.exists())

    def test_model_path_mapping(self):
        """Test model path mapping."""
        path = self.protocol._get_model_path()
        self.assertIn("tiny", path.lower())
        self.assertIn("distilbert", path.lower())

    def test_dataset_config_mapping(self):
        """Test dataset configuration mapping."""
        dataset_config = self.protocol._get_dataset_config()
        task_name = self.protocol._get_task_name()
        
        self.assertIsInstance(dataset_config, str)
        self.assertIsInstance(task_name, str)

    def test_training_config_creation(self):
        """Test training configuration creation."""
        output_dir = Path(self.temp_dir) / "test_config"
        output_dir.mkdir()
        
        config_path = self.protocol.create_tiny_training_config(output_dir, rank=8)
        
        self.assertTrue(config_path.exists())
        
        with open(config_path) as f:
            config = json.load(f)
        
        # Check required fields
        self.assertEqual(config["lora_r"], 8)
        self.assertEqual(config["lora_alpha"], 16)  # 2x scaling
        self.assertIn("lora_target_modules", config)
        self.assertEqual(config["output_dir"], str(output_dir))

    def test_training_config_with_rank_pattern(self):
        """Test training config creation with rank pattern."""
        output_dir = Path(self.temp_dir) / "test_pattern"
        output_dir.mkdir()
        
        rank_pattern = {"layer.0.attention.q": 2, "layer.1.attention.k": 4}
        config_path = self.protocol.create_tiny_training_config(
            output_dir, rank=8, rank_pattern=rank_pattern
        )
        
        with open(config_path) as f:
            config = json.load(f)
        
        self.assertEqual(config["lora_rank_pattern"], rank_pattern)

    def test_parameter_estimation(self):
        """Test parameter estimation functions."""
        # Test uniform rank estimation
        params_8 = self.protocol._estimate_params(8)
        params_4 = self.protocol._estimate_params(4)
        
        self.assertGreater(params_8, params_4)
        self.assertEqual(params_4, params_8 // 2)  # Linear scaling
        
        # Test strategy-based estimation
        strategies = {
            "uniform": {"type": "uniform", "rank": 4},
            "module": {"type": "module", "base_rank": 6},
            "per_layer": {
                "type": "per_layer", 
                "default_rank": 2,
                "rank_pattern": {"layer.0.attention.q": 4}
            }
        }
        
        for name, strategy in strategies.items():
            params = self.protocol._estimate_params_with_strategy(strategy)
            self.assertGreater(params, 0, f"Strategy {name} should have positive params")

    def test_suggestion_extraction(self):
        """Test rank suggestion strategy extraction."""
        # Mock suggestion data
        suggestions = {
            "default_r": 4,
            "rank_pattern": {
                "layer.0.attention.q": 2,
                "layer.1.attention.k": 8
            },
            "by_module_type_p90": {
                "attn": 6,
                "mlp": 4
            }
        }
        
        strategies = self.protocol.extract_strategies(suggestions)
        
        # Check all strategies are present
        self.assertIn("uniform_p90", strategies)
        self.assertIn("module_p90", strategies) 
        self.assertIn("per_layer", strategies)
        
        # Check uniform strategy
        uniform = strategies["uniform_p90"]
        self.assertEqual(uniform["type"], "uniform")
        self.assertEqual(uniform["rank"], 4)
        
        # Check module strategy
        module = strategies["module_p90"]
        self.assertEqual(module["type"], "module")
        self.assertIn("module_ranks", module)
        
        # Check per-layer strategy
        per_layer = strategies["per_layer"]
        self.assertEqual(per_layer["type"], "per_layer")
        self.assertEqual(per_layer["default_rank"], 4)
        self.assertEqual(per_layer["rank_pattern"], suggestions["rank_pattern"])

    def test_empty_suggestions_handling(self):
        """Test handling of empty or minimal suggestions."""
        empty_suggestions = {}
        strategies = self.protocol.extract_strategies(empty_suggestions)
        
        # Should still generate strategies with fallbacks
        self.assertIn("uniform_p90", strategies)
        self.assertIn("module_p90", strategies)
        self.assertIn("per_layer", strategies)
        
        # Should use fallback ranks
        uniform = strategies["uniform_p90"]
        self.assertGreater(uniform["rank"], 0)

    def test_command_building(self):
        """Test training command building."""
        output_dir = Path(self.temp_dir) / "test_cmd"
        output_dir.mkdir()
        
        config_path = self.protocol.create_tiny_training_config(output_dir, rank=4)
        cmd = self.protocol._get_training_command(config_path, output_dir)
        
        self.assertIsInstance(cmd, list)
        self.assertGreater(len(cmd), 0)
        self.assertEqual(cmd[0], "python")

    def test_results_tracking(self):
        """Test results tracking functionality."""
        # Test initial state
        self.assertEqual(len(self.protocol.results), 0)
        
        # Add some mock results
        self.protocol.results["probe"] = {"status": "success", "rank": 16}
        self.protocol.results["retrain_uniform"] = {"status": "success", "param_reduction": 0.5}
        
        self.assertEqual(len(self.protocol.results), 2)
        self.assertEqual(self.protocol.results["probe"]["status"], "success")

    def test_evaluation_logic(self):
        """Test evaluation and comparison logic.""" 
        # Set up mock results
        self.protocol.results = {
            "probe": {"status": "success", "rank": 16},
            "suggestions": {"default_r": 4},
            "retrain_uniform_p90": {
                "status": "success",
                "param_reduction": 0.75,
                "strategy": {"description": "Uniform rank 4"}
            },
            "retrain_per_layer": {
                "status": "success", 
                "param_reduction": 0.85,
                "strategy": {"description": "Per-layer pattern"}
            }
        }
        
        evaluation = self.protocol.evaluate_results()
        
        # Check evaluation structure
        self.assertIn("protocol_summary", evaluation)
        self.assertIn("strategies_tested", evaluation)
        self.assertIn("parameter_reductions", evaluation)
        self.assertIn("recommendations", evaluation)
        
        # Check summary data
        summary = evaluation["protocol_summary"]
        self.assertEqual(summary["model"], "tiny-distilbert")
        self.assertEqual(summary["probe_rank"], 16)
        
        # Check strategies
        strategies = evaluation["strategies_tested"]
        self.assertEqual(len(strategies), 2)
        
        # Check reductions
        reductions = evaluation["parameter_reductions"]
        self.assertIn("uniform_p90", reductions)
        self.assertIn("per_layer", reductions)
        
        # Check recommendations
        recommendations = evaluation["recommendations"]
        self.assertGreater(len(recommendations), 0)
        self.assertTrue(any("per_layer" in rec for rec in recommendations))


if __name__ == "__main__":
    unittest.main()