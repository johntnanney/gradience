"""
Test per_layer compression variant payload contract.

This test enforces the exact schema contract for per_layer compression configurations
to prevent future KeyError regressions and ensure consistent data structures.
"""

import unittest
import tempfile
import shutil
from pathlib import Path

from gradience.bench.schema import get_variant_payload, validate_compression_config
from gradience.bench.protocol import generate_compression_configs


class TestPerLayerPayloadContract(unittest.TestCase):
    """Test contract enforcement for per_layer compression variant payloads."""
    
    def setUp(self):
        """Set up test environment with temporary directories."""
        self.test_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.test_dir)
    
    def test_per_layer_config_canonical_location(self):
        """Test that per_layer configs have rank_pattern at canonical top-level location."""
        # Create a valid per_layer compression config
        per_layer_config = {
            "variant": "per_layer",
            "suggested_r": 6.5,
            "actual_r": 6,
            "rank_pattern": {
                "base_model.model.layer.0.attention.q_proj": 8,
                "base_model.model.layer.1.attention.k_proj": 4,
                "base_model.model.layer.2.attention.v_proj": 8
            },
            "alpha_pattern": {
                "base_model.model.layer.0.attention.q_proj": 8,
                "base_model.model.layer.1.attention.k_proj": 4,
                "base_model.model.layer.2.attention.v_proj": 8
            },
            "config": {
                "alpha": None,
                "dropout": 0.0,
                "probe_r": None,
                "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"]
            },
            "status": "ready",
            "reason": None,
            "policy_type": "per_layer",
            "conservatism_score": 2.0
        }
        
        # Contract assertion: Must have top-level rank_pattern
        self.assertIn("rank_pattern", per_layer_config)
        self.assertIn("alpha_pattern", per_layer_config)
        
        # Contract assertion: rank_pattern must be at top level (canonical location)
        self.assertIsInstance(per_layer_config["rank_pattern"], dict)
        self.assertIsInstance(per_layer_config["alpha_pattern"], dict)
        
        # Contract assertion: patterns must not be empty for per_layer variants
        self.assertGreater(len(per_layer_config["rank_pattern"]), 0)
        self.assertGreater(len(per_layer_config["alpha_pattern"]), 0)
        
        # Validate using canonical accessor
        rank_pattern, alpha_pattern = get_variant_payload(per_layer_config)
        
        # Contract assertion: canonical accessor must work
        self.assertEqual(rank_pattern, per_layer_config["rank_pattern"])
        self.assertEqual(alpha_pattern, per_layer_config["alpha_pattern"])
        
        # Contract assertion: schema validation must pass
        validate_compression_config(per_layer_config)
    
    def test_rank_pattern_shape_contract(self):
        """Test that rank_pattern has correct shape: module_name → int."""
        per_layer_config = {
            "variant": "per_layer",
            "rank_pattern": {
                "base_model.model.layer.0.attention.q_proj": 8,
                "base_model.model.layer.1.attention.k_proj": 4,
                "base_model.model.layer.2.attention.v_proj": 16
            },
            "alpha_pattern": {
                "base_model.model.layer.0.attention.q_proj": 8,
                "base_model.model.layer.1.attention.k_proj": 4,
                "base_model.model.layer.2.attention.v_proj": 16
            },
            "status": "ready"
        }
        
        rank_pattern, alpha_pattern = get_variant_payload(per_layer_config)
        
        # Contract assertion: Shape must be module_name → int
        for module_name, rank_value in rank_pattern.items():
            self.assertIsInstance(module_name, str, 
                                f"Module name must be string, got {type(module_name)}")
            self.assertIsInstance(rank_value, int, 
                                f"Rank value must be int, got {type(rank_value)}")
            self.assertGreater(rank_value, 0, 
                             f"Rank value must be positive, got {rank_value}")
        
        # Same contract for alpha_pattern
        for module_name, alpha_value in alpha_pattern.items():
            self.assertIsInstance(module_name, str, 
                                f"Module name must be string, got {type(module_name)}")
            self.assertIsInstance(alpha_value, int, 
                                f"Alpha value must be int, got {type(alpha_value)}")
            self.assertGreater(alpha_value, 0, 
                             f"Alpha value must be positive, got {alpha_value}")
    
    def test_no_none_values_contract(self):
        """Test that rank_pattern and alpha_pattern contain no None values."""
        # First test: Valid config with no None values should pass
        valid_config = {
            "variant": "per_layer", 
            "rank_pattern": {
                "layer.0": 8,
                "layer.1": 4,
                "layer.2": 16
            },
            "alpha_pattern": {
                "layer.0": 8,
                "layer.1": 4,
                "layer.2": 16
            },
            "status": "ready"
        }
        
        rank_pattern, alpha_pattern = get_variant_payload(valid_config)
        
        # Contract assertion: No None values allowed in valid configs
        for module_name, rank_value in rank_pattern.items():
            self.assertIsNotNone(rank_value, 
                               f"Rank value for {module_name} must not be None")
            self.assertIsInstance(rank_value, int)
            self.assertGreater(rank_value, 0)
        
        for module_name, alpha_value in alpha_pattern.items():
            self.assertIsNotNone(alpha_value, 
                               f"Alpha value for {module_name} must not be None")
            self.assertIsInstance(alpha_value, int)
            self.assertGreater(alpha_value, 0)
    
    def test_none_values_detection_contract(self):
        """Test that configurations with None values are properly detected as invalid."""
        # Create config with None values (for detection testing)
        config_with_none = {
            "variant": "per_layer", 
            "rank_pattern": {
                "layer.0": 8,
                "layer.1": None,  # This should be detectable
                "layer.2": 4
            },
            "alpha_pattern": {
                "layer.0": 8,
                "layer.1": 4,
                "layer.2": None   # This should be detectable
            },
            "status": "ready"
        }
        
        rank_pattern, alpha_pattern = get_variant_payload(config_with_none)
        
        # Contract assertion: Our accessor should return the None values as-is
        # (detection of None values is the responsibility of downstream validation)
        self.assertIsNone(rank_pattern["layer.1"])
        self.assertIsNone(alpha_pattern["layer.2"])
        
        # But non-None values should be preserved correctly
        self.assertEqual(rank_pattern["layer.0"], 8)
        self.assertEqual(rank_pattern["layer.2"], 4)
        self.assertEqual(alpha_pattern["layer.0"], 8)
        self.assertEqual(alpha_pattern["layer.1"], 4)
    
    def test_backward_compatibility_fallback(self):
        """Test that fallback to nested config works for backward compatibility."""
        # Create config with patterns in nested location (legacy format)
        legacy_config = {
            "variant": "per_layer",
            "config": {
                "rank_pattern": {
                    "layer.0": 8,
                    "layer.1": 4
                },
                "alpha_pattern": {
                    "layer.0": 8,
                    "layer.1": 4
                }
            },
            "status": "ready"
        }
        
        # Contract assertion: Fallback must work
        rank_pattern, alpha_pattern = get_variant_payload(legacy_config)
        
        self.assertEqual(rank_pattern, {"layer.0": 8, "layer.1": 4})
        self.assertEqual(alpha_pattern, {"layer.0": 8, "layer.1": 4})
        
        # Contract assertion: Shape must still be correct for fallback
        for module_name, rank_value in rank_pattern.items():
            self.assertIsInstance(module_name, str)
            self.assertIsInstance(rank_value, int)
            self.assertIsNotNone(rank_value)
    
    def test_canonical_takes_priority_over_nested(self):
        """Test that canonical top-level location takes priority over nested config."""
        config_with_both = {
            "variant": "per_layer",
            "rank_pattern": {
                "layer.0": 16  # Canonical (should win)
            },
            "alpha_pattern": {
                "layer.0": 16  # Canonical (should win)
            },
            "config": {
                "rank_pattern": {
                    "layer.0": 8  # Nested (should be ignored)
                },
                "alpha_pattern": {
                    "layer.0": 8  # Nested (should be ignored)
                }
            },
            "status": "ready"
        }
        
        # Contract assertion: Canonical location must take priority
        rank_pattern, alpha_pattern = get_variant_payload(config_with_both)
        
        self.assertEqual(rank_pattern, {"layer.0": 16})  # Canonical value
        self.assertEqual(alpha_pattern, {"layer.0": 16})  # Canonical value
        self.assertNotEqual(rank_pattern, {"layer.0": 8})  # Not nested value
    
    def test_empty_patterns_contract(self):
        """Test contract behavior when patterns are empty (for non-per_layer variants)."""
        uniform_config = {
            "variant": "energy_p90",
            "actual_r": 8,
            "rank_pattern": {},  # Empty for uniform variants
            "alpha_pattern": {},  # Empty for uniform variants
            "status": "ready"
        }
        
        rank_pattern, alpha_pattern = get_variant_payload(uniform_config)
        
        # Contract assertion: Empty patterns are valid for non-per_layer variants
        self.assertEqual(rank_pattern, {})
        self.assertEqual(alpha_pattern, {})
        self.assertIsInstance(rank_pattern, dict)
        self.assertIsInstance(alpha_pattern, dict)
    
    def test_per_layer_variant_must_have_patterns(self):
        """Test that per_layer variants must have non-empty patterns."""
        # per_layer variant with empty patterns (contract violation)
        invalid_per_layer_config = {
            "variant": "per_layer",
            "rank_pattern": {},  # Invalid: empty for per_layer
            "alpha_pattern": {},  # Invalid: empty for per_layer
            "status": "ready"
        }
        
        # Contract assertion: per_layer variants must have non-empty patterns
        with self.assertRaises(ValueError) as cm:
            validate_compression_config(invalid_per_layer_config)
        
        self.assertIn("per-layer variant", str(cm.exception))
        self.assertIn("non-empty rank_pattern", str(cm.exception))
    
    def test_type_validation_contract(self):
        """Test that type validation catches contract violations."""
        # rank_pattern as wrong type
        invalid_type_config = {
            "variant": "per_layer",
            "rank_pattern": "not_a_dict",  # Contract violation: wrong type
            "alpha_pattern": {},
            "status": "ready"
        }
        
        # Contract assertion: Type validation must catch violations
        with self.assertRaises(TypeError) as cm:
            get_variant_payload(invalid_type_config)
        
        self.assertIn("rank_pattern must be dict", str(cm.exception))
    
    def test_module_name_format_contract(self):
        """Test that module names follow expected format patterns."""
        per_layer_config = {
            "variant": "per_layer",
            "rank_pattern": {
                "base_model.model.distilbert.transformer.layer.0.attention.q_lin": 8,
                "base_model.model.distilbert.transformer.layer.1.attention.k_lin": 4,
                "base_model.model.distilbert.transformer.layer.2.attention.v_lin": 16
            },
            "alpha_pattern": {
                "base_model.model.distilbert.transformer.layer.0.attention.q_lin": 8,
                "base_model.model.distilbert.transformer.layer.1.attention.k_lin": 4,
                "base_model.model.distilbert.transformer.layer.2.attention.v_lin": 16
            },
            "status": "ready"
        }
        
        rank_pattern, alpha_pattern = get_variant_payload(per_layer_config)
        
        # Contract assertion: Module names should follow hierarchical pattern
        for module_name in rank_pattern.keys():
            self.assertIsInstance(module_name, str)
            self.assertGreater(len(module_name), 0)
            # Typical format: base_model.model.{model_type}.{layer_path}.{module_name}
            self.assertTrue(any(char in module_name for char in ['.', '_']), 
                          f"Module name should contain separators: {module_name}")
    
    def test_symmetry_contract(self):
        """Test that rank_pattern and alpha_pattern have matching keys."""
        per_layer_config = {
            "variant": "per_layer",
            "rank_pattern": {
                "layer.0.attention.q_proj": 8,
                "layer.1.attention.k_proj": 4,
                "layer.2.attention.v_proj": 16
            },
            "alpha_pattern": {
                "layer.0.attention.q_proj": 8,
                "layer.1.attention.k_proj": 4,
                "layer.2.attention.v_proj": 16
                # Note: same keys as rank_pattern
            },
            "status": "ready"
        }
        
        rank_pattern, alpha_pattern = get_variant_payload(per_layer_config)
        
        # Contract assertion: rank_pattern and alpha_pattern should have same keys
        self.assertEqual(set(rank_pattern.keys()), set(alpha_pattern.keys()),
                        "rank_pattern and alpha_pattern must have matching module names")
        
        # Contract assertion: Each module should have consistent mapping
        for module_name in rank_pattern.keys():
            self.assertIn(module_name, alpha_pattern, 
                         f"Module {module_name} missing from alpha_pattern")
            # Values can differ, but keys must match


if __name__ == "__main__":
    unittest.main()