"""
Test bench schema accessor functions.
"""

import unittest
from gradience.bench.schema import (
    get_variant_payload, 
    validate_compression_config, 
    get_variant_rank
)


class TestBenchSchema(unittest.TestCase):
    
    def test_get_variant_payload_canonical_location(self):
        """Test that get_variant_payload correctly extracts from canonical top-level location."""
        config = {
            "variant": "per_layer",
            "rank_pattern": {"layer.0": 4, "layer.1": 8},
            "alpha_pattern": {"layer.0": 4, "layer.1": 8},
            "config": {},  # Empty nested config
            "status": "ready"
        }
        
        rank_pattern, alpha_pattern = get_variant_payload(config)
        
        self.assertEqual(rank_pattern, {"layer.0": 4, "layer.1": 8})
        self.assertEqual(alpha_pattern, {"layer.0": 4, "layer.1": 8})
    
    def test_get_variant_payload_fallback_location(self):
        """Test that get_variant_payload falls back to nested config location."""
        config = {
            "variant": "per_layer",
            "config": {
                "rank_pattern": {"layer.0": 2, "layer.1": 4},
                "alpha_pattern": {"layer.0": 2, "layer.1": 4},
            },
            "status": "ready"
        }
        
        rank_pattern, alpha_pattern = get_variant_payload(config)
        
        self.assertEqual(rank_pattern, {"layer.0": 2, "layer.1": 4})
        self.assertEqual(alpha_pattern, {"layer.0": 2, "layer.1": 4})
    
    def test_get_variant_payload_canonical_priority(self):
        """Test that canonical location takes priority over nested config."""
        config = {
            "variant": "per_layer",
            "rank_pattern": {"layer.0": 8},  # Canonical
            "alpha_pattern": {"layer.0": 8},  # Canonical
            "config": {
                "rank_pattern": {"layer.0": 4},  # Fallback (should be ignored)
                "alpha_pattern": {"layer.0": 4},  # Fallback (should be ignored)
            },
            "status": "ready"
        }
        
        rank_pattern, alpha_pattern = get_variant_payload(config)
        
        self.assertEqual(rank_pattern, {"layer.0": 8})  # Should use canonical
        self.assertEqual(alpha_pattern, {"layer.0": 8})  # Should use canonical
    
    def test_get_variant_payload_empty_defaults(self):
        """Test that get_variant_payload returns empty dicts when patterns don't exist."""
        config = {
            "variant": "uniform",
            "config": {},
            "status": "ready"
        }
        
        rank_pattern, alpha_pattern = get_variant_payload(config)
        
        self.assertEqual(rank_pattern, {})
        self.assertEqual(alpha_pattern, {})
    
    def test_get_variant_payload_type_validation(self):
        """Test that get_variant_payload validates pattern types."""
        config = {
            "variant": "per_layer",
            "rank_pattern": "not_a_dict",  # Invalid type
            "alpha_pattern": {},
            "status": "ready"
        }
        
        with self.assertRaises(TypeError) as cm:
            get_variant_payload(config)
        
        self.assertIn("rank_pattern must be dict", str(cm.exception))
    
    def test_validate_compression_config_valid(self):
        """Test that validate_compression_config accepts valid configs."""
        config = {
            "variant": "per_layer",
            "rank_pattern": {"layer.0": 4},
            "alpha_pattern": {"layer.0": 4},
            "status": "ready",
            "actual_r": 4
        }
        
        # Should not raise
        validate_compression_config(config)
    
    def test_validate_compression_config_missing_fields(self):
        """Test that validate_compression_config catches missing required fields."""
        config = {
            # Missing "variant" and "status"
            "rank_pattern": {}
        }
        
        with self.assertRaises(KeyError) as cm:
            validate_compression_config(config)
        
        self.assertIn("Missing required field: variant", str(cm.exception))
    
    def test_validate_compression_config_invalid_status(self):
        """Test that validate_compression_config catches invalid status values."""
        config = {
            "variant": "per_layer",
            "status": "invalid_status",  # Invalid
            "rank_pattern": {"layer.0": 4},
            "alpha_pattern": {"layer.0": 4}
        }
        
        with self.assertRaises(ValueError) as cm:
            validate_compression_config(config)
        
        self.assertIn("status must be one of", str(cm.exception))
    
    def test_get_variant_rank_uniform(self):
        """Test get_variant_rank for uniform variants."""
        config = {
            "variant": "energy_p90",
            "actual_r": 8,
            "status": "ready"
        }
        
        rank = get_variant_rank(config)
        self.assertEqual(rank, 8)
    
    def test_get_variant_rank_per_layer(self):
        """Test get_variant_rank for per-layer variants (returns most common rank)."""
        config = {
            "variant": "per_layer",
            "rank_pattern": {
                "layer.0": 4,
                "layer.1": 8,
                "layer.2": 8,  # 8 appears twice (most common)
                "layer.3": 2
            },
            "alpha_pattern": {},
            "status": "ready"
        }
        
        rank = get_variant_rank(config)
        self.assertEqual(rank, 8)  # Most common rank


if __name__ == "__main__":
    unittest.main()