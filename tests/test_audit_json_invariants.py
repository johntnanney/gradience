"""
Invariant tests for audit JSON structure.

These tests prevent future refactors from silently breaking the layer data structure
that downstream tools depend on.
"""

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict


class TestAuditJsonInvariants(unittest.TestCase):
    """Test invariant properties of audit JSON output structure."""

    def setUp(self):
        """Set up a minimal PEFT directory for testing."""
        self.temp_dir = tempfile.mkdtemp()
        self.peft_dir = Path(self.temp_dir) / "test_peft"
        self.peft_dir.mkdir()
        
        # Create minimal adapter_config.json
        config = {
            "r": 4,
            "lora_alpha": 16,
            "target_modules": ["q_proj", "v_proj"],
            "peft_type": "LORA"
        }
        with open(self.peft_dir / "adapter_config.json", "w") as f:
            json.dump(config, f)
        
        # Create minimal adapter weights (mock data)
        import torch
        weights = {
            "base_model.model.layer.0.self_attn.q_proj.lora_A.weight": torch.randn(4, 64),
            "base_model.model.layer.0.self_attn.q_proj.lora_B.weight": torch.randn(64, 4),
            "base_model.model.layer.0.self_attn.v_proj.lora_A.weight": torch.randn(4, 64), 
            "base_model.model.layer.0.self_attn.v_proj.lora_B.weight": torch.randn(64, 4),
        }
        torch.save(weights, self.peft_dir / "adapter_model.bin")

    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_audit_json_structure_invariants(self):
        """Test that audit JSON maintains required structure."""
        from gradience.vnext.audit import audit_lora_peft_dir
        
        # Run audit with layers
        result = audit_lora_peft_dir(str(self.peft_dir))
        audit_dict = result.to_summary_dict(include_layers=True)
        
        # Test top-level structure
        self.assertIsInstance(audit_dict, dict)
        self.assertIn("layer_data", audit_dict)
        
        # Test layer_data structure
        layer_data = audit_dict["layer_data"]
        self.assertIsInstance(layer_data, dict)
        self.assertIn("layer_rows_schema", layer_data)
        self.assertIn("layer_rows", layer_data)
        
        # Test schema version
        self.assertEqual(layer_data["layer_rows_schema"], "v1")
        
        # Test layer_rows structure
        layer_rows = layer_data["layer_rows"]
        self.assertIsInstance(layer_rows, list)
        self.assertGreater(len(layer_rows), 0, "Should have at least one layer")

    def test_layer_row_required_fields(self):
        """Test that every layer row has required fields with correct types."""
        from gradience.vnext.audit import audit_lora_peft_dir
        
        result = audit_lora_peft_dir(str(self.peft_dir))
        audit_dict = result.to_summary_dict(include_layers=True)
        layer_rows = audit_dict["layer_data"]["layer_rows"]
        
        for i, row in enumerate(layer_rows):
            with self.subTest(layer_index=i):
                # Test required fields exist
                self.assertIn("name", row)
                self.assertIn("r", row)
                self.assertIn("energy_rank_90", row)
                
                # Test name field
                name = row["name"]
                self.assertIsInstance(name, str)
                self.assertGreater(len(name), 0, "name should be non-empty")
                
                # Test r field
                r = row["r"]
                self.assertIsInstance(r, int)
                self.assertGreater(r, 0, "r should be positive integer")
                
                # Test energy_rank_90 field
                energy_rank_90 = row["energy_rank_90"]
                self.assertIsInstance(energy_rank_90, (int, float))
                self.assertGreaterEqual(energy_rank_90, 0, "energy_rank_90 should be >= 0")
                self.assertLessEqual(energy_rank_90, r, "energy_rank_90 should be <= r")

    def test_layer_row_optional_fields_types(self):
        """Test that optional fields have correct types when present."""
        from gradience.vnext.audit import audit_lora_peft_dir
        
        result = audit_lora_peft_dir(str(self.peft_dir))
        audit_dict = result.to_summary_dict(include_layers=True)
        layer_rows = audit_dict["layer_data"]["layer_rows"]
        
        for i, row in enumerate(layer_rows):
            with self.subTest(layer_index=i):
                # Test optional numeric fields
                for field in ["stable_rank", "utilization", "sigma_max", "effective_rank"]:
                    if field in row and row[field] is not None:
                        self.assertIsInstance(row[field], (int, float), f"{field} should be numeric")
                        self.assertGreaterEqual(row[field], 0, f"{field} should be non-negative")
                
                # Test optional string fields
                for field in ["module_type", "a_key", "b_key"]:
                    if field in row and row[field] is not None:
                        self.assertIsInstance(row[field], str, f"{field} should be string")

    def test_audit_backward_compatibility(self):
        """Test that audit works without --layers flag (backward compatibility)."""
        from gradience.vnext.audit import audit_lora_peft_dir
        
        result = audit_lora_peft_dir(str(self.peft_dir))
        
        # Without include_layers=True, should not have layer_data
        audit_dict = result.to_summary_dict(include_layers=False)
        self.assertNotIn("layer_data", audit_dict)
        self.assertNotIn("layers", audit_dict)  # Old format should also be absent
        
        # Should still have summary statistics
        self.assertIn("total_lora_params", audit_dict)
        self.assertIn("stable_rank_mean", audit_dict)
        self.assertIn("energy_rank_90_p50", audit_dict)

    def test_audit_json_serializable(self):
        """Test that audit JSON is fully serializable."""
        from gradience.vnext.audit import audit_lora_peft_dir
        
        result = audit_lora_peft_dir(str(self.peft_dir))
        audit_dict = result.to_summary_dict(include_layers=True)
        
        # Should be JSON serializable without errors
        try:
            json_str = json.dumps(audit_dict)
            reconstructed = json.loads(json_str)
            self.assertEqual(reconstructed["layer_data"]["layer_rows_schema"], "v1")
        except (TypeError, ValueError) as e:
            self.fail(f"Audit dict should be JSON serializable: {e}")


if __name__ == "__main__":
    unittest.main()