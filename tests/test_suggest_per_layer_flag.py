"""
Test for --suggest-per-layer flag in audit command.

Ensures the flag works correctly and doesn't affect default behavior.
"""

import json
import subprocess
import tempfile
import unittest
from pathlib import Path


class TestSuggestPerLayerFlag(unittest.TestCase):
    """Test --suggest-per-layer flag functionality."""

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
        
        # Create minimal adapter weights
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

    def _run_audit(self, *args):
        """Helper to run audit command and return result."""
        # Use python3 for better compatibility on macOS/Linux systems
        cmd = ["python3", "-m", "gradience", "audit", "--peft-dir", str(self.peft_dir)] + list(args)
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result

    def test_default_behavior_unchanged(self):
        """Test that default audit output doesn't include rank_suggestions."""
        result = self._run_audit("--json")
        self.assertEqual(result.returncode, 0)
        
        data = json.loads(result.stdout)
        self.assertNotIn("rank_suggestions", data)
        self.assertNotIn("layer_data", data)

    def test_layers_only_no_rank_suggestions(self):
        """Test that --layers alone doesn't add rank_suggestions."""
        result = self._run_audit("--layers", "--json")
        self.assertEqual(result.returncode, 0)
        
        data = json.loads(result.stdout)
        self.assertIn("layer_data", data)
        self.assertNotIn("rank_suggestions", data)

    def test_suggest_per_layer_requires_layers(self):
        """Test that --suggest-per-layer requires --layers flag."""
        result = self._run_audit("--suggest-per-layer", "--json")
        self.assertEqual(result.returncode, 1)
        self.assertIn("--suggest-per-layer requires --layers", result.stderr)

    def test_suggest_per_layer_with_layers(self):
        """Test that --suggest-per-layer with --layers includes rank_suggestions."""
        result = self._run_audit("--layers", "--suggest-per-layer", "--json")
        self.assertEqual(result.returncode, 0)
        
        data = json.loads(result.stdout)
        self.assertIn("layer_data", data)
        self.assertIn("rank_suggestions", data)
        
        # Check rank_suggestions structure
        rank_suggestions = data["rank_suggestions"]
        self.assertIn("default_r", rank_suggestions)
        self.assertIn("rank_pattern", rank_suggestions)
        self.assertIn("layers", rank_suggestions)
        self.assertIn("by_module_type_p90", rank_suggestions)
        self.assertIn("notes", rank_suggestions)

    def test_rank_suggestions_structure(self):
        """Test that rank_suggestions contains expected data structure."""
        result = self._run_audit("--layers", "--suggest-per-layer", "--json")
        self.assertEqual(result.returncode, 0)
        
        data = json.loads(result.stdout)
        rank_suggestions = data["rank_suggestions"]
        
        # Test basic structure
        self.assertIsInstance(rank_suggestions["default_r"], int)
        self.assertIsInstance(rank_suggestions["rank_pattern"], dict)
        self.assertIsInstance(rank_suggestions["layers"], list)
        self.assertIsInstance(rank_suggestions["by_module_type_p90"], dict)
        self.assertIsInstance(rank_suggestions["notes"], str)
        
        # Test layer suggestions structure
        if rank_suggestions["layers"]:
            layer = rank_suggestions["layers"][0]
            self.assertIn("name", layer)
            self.assertIn("current_r", layer)
            self.assertIn("suggested_r", layer)
            self.assertIn("reduction_ratio", layer)
            self.assertIn("energy_rank_90", layer)

    def test_flag_doesnt_affect_text_output(self):
        """Test that --suggest-per-layer doesn't change text output format."""
        result_normal = self._run_audit()
        result_with_flag = self._run_audit("--suggest-per-layer")
        
        # Both should succeed (flag is ignored for text output)
        self.assertEqual(result_normal.returncode, 0)
        self.assertEqual(result_with_flag.returncode, 0)
        
        # Text output should have identical content (ignoring policy ordering differences)
        def normalize_and_sort_policies(text):
            """Normalize policy names and sort policy lines for consistent comparison."""
            lines = text.split('\n')
            normalized_lines = []
            
            for line in lines:
                # Normalize policy names
                normalized_line = line
                normalized_line = normalized_line.replace('energy@0.90', 'energy_90')
                normalized_line = normalized_line.replace('erank', 'entropy_effective')
                normalized_line = normalized_line.replace('knee', 'knee_elbow')
                normalized_line = normalized_line.replace('oht', 'optimal_hard_threshold')
                normalized_lines.append(normalized_line)
            
            # Find the policy table section and sort those lines
            policy_start = -1
            policy_end = -1
            
            for i, line in enumerate(normalized_lines):
                if 'Policy            Median   P90   Max' in line:
                    policy_start = i + 2  # Skip header and separator
                elif policy_start != -1 and line.strip() == '':
                    policy_end = i
                    break
            
            if policy_start != -1 and policy_end != -1:
                # Sort policy lines for consistent comparison
                policy_lines = normalized_lines[policy_start:policy_end]
                policy_lines = [line for line in policy_lines if line.strip()]  # Remove empty lines
                policy_lines.sort()  # Sort alphabetically
                normalized_lines[policy_start:policy_end] = policy_lines
            
            return '\n'.join(normalized_lines)
        
        normalized_normal = normalize_and_sort_policies(result_normal.stdout)
        normalized_with_flag = normalize_and_sort_policies(result_with_flag.stdout)
        self.assertEqual(normalized_normal, normalized_with_flag)

    def test_invariants_preserved(self):
        """Test that rank suggestions maintain expected invariants."""
        result = self._run_audit("--layers", "--suggest-per-layer", "--json")
        self.assertEqual(result.returncode, 0)
        
        data = json.loads(result.stdout)
        rank_suggestions = data["rank_suggestions"]
        
        # Test invariants for each layer suggestion
        for layer in rank_suggestions["layers"]:
            # suggested_r should be <= current_r
            self.assertLessEqual(layer["suggested_r"], layer["current_r"])
            
            # suggested_r should be positive
            self.assertGreater(layer["suggested_r"], 0)
            
            # reduction_ratio should be between 0 and 1
            self.assertGreaterEqual(layer["reduction_ratio"], 0.0)
            self.assertLessEqual(layer["reduction_ratio"], 1.0)
            
            # reduction_ratio should match calculation
            expected_ratio = 1.0 - (layer["suggested_r"] / layer["current_r"])
            self.assertAlmostEqual(layer["reduction_ratio"], expected_ratio, places=6)


if __name__ == "__main__":
    unittest.main()