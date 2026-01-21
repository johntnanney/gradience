#!/usr/bin/env python3
"""
Test 7: Simplified integration test for UDR/SDI pipeline without external dependencies.

Tests the complete audit CLI pipeline with realistic synthetic adapters:
- JSON output schema validation
- CLI flag behavior
- Error handling with base model loading
- End-to-end audit completion

No external dependencies required - uses synthetic realistic data.
"""

import json
import tempfile
import torch
import subprocess
import sys
import os
from pathlib import Path
from typing import Dict, Any

# Add gradience to path
sys.path.insert(0, '/Users/john/code/gradience')


def create_realistic_test_adapter(out_dir: Path, model_type: str = "gpt2") -> Path:
    """Create realistic test adapter that mimics real PEFT structure."""
    adapter_dir = out_dir / "adapter"
    adapter_dir.mkdir()
    
    # Create realistic adapter config based on model type
    if model_type == "gpt2":
        config = {
            "peft_type": "LORA",
            "auto_mapping": None,
            "base_model_name_or_path": "gpt2",
            "revision": None,
            "task_type": "CAUSAL_LM",
            "inference_mode": False,
            "r": 8,
            "target_modules": ["c_attn", "c_proj"],
            "lora_alpha": 16,
            "lora_dropout": 0.1,
            "bias": "none",
        }
        
        # GPT-2 style weights (realistic dimensions)
        weights = {
            "base_model.transformer.h.0.attn.c_attn.lora_A.weight": torch.randn(8, 768) * 0.1,
            "base_model.transformer.h.0.attn.c_attn.lora_B.weight": torch.randn(2304, 8) * 0.1,  # 3*768 for q,k,v
            "base_model.transformer.h.0.attn.c_proj.lora_A.weight": torch.randn(8, 768) * 0.1,
            "base_model.transformer.h.0.attn.c_proj.lora_B.weight": torch.randn(768, 8) * 0.1,
            "base_model.transformer.h.1.attn.c_attn.lora_A.weight": torch.randn(8, 768) * 0.1,
            "base_model.transformer.h.1.attn.c_attn.lora_B.weight": torch.randn(2304, 8) * 0.1,
            "base_model.transformer.h.1.attn.c_proj.lora_A.weight": torch.randn(8, 768) * 0.1,
            "base_model.transformer.h.1.attn.c_proj.lora_B.weight": torch.randn(768, 8) * 0.1,
        }
        
    elif model_type == "bert":
        config = {
            "peft_type": "LORA",
            "base_model_name_or_path": "bert-base-uncased",
            "task_type": "SEQ_CLS",
            "r": 4,
            "target_modules": ["query", "value"],
            "lora_alpha": 8,
            "lora_dropout": 0.1,
            "bias": "none",
        }
        
        weights = {
            "base_model.encoder.layer.0.attention.self.query.lora_A.weight": torch.randn(4, 768) * 0.1,
            "base_model.encoder.layer.0.attention.self.query.lora_B.weight": torch.randn(768, 4) * 0.1,
            "base_model.encoder.layer.0.attention.self.value.lora_A.weight": torch.randn(4, 768) * 0.1,
            "base_model.encoder.layer.0.attention.self.value.lora_B.weight": torch.randn(768, 4) * 0.1,
        }
    
    # Save config and weights
    with (adapter_dir / "adapter_config.json").open('w') as f:
        json.dump(config, f, indent=2)
    
    torch.save(weights, adapter_dir / "adapter_model.bin")
    
    return adapter_dir


def create_base_norms_for_model(model_type: str) -> Dict[str, Dict[str, float]]:
    """Create realistic base norms for different model types."""
    if model_type == "gpt2":
        # GPT-2 style base norms (realistic for 768-dim model)
        return {
            "base_model.transformer.h.0.attn.c_attn": {"sigma_max": 18.5, "fro_norm": 145.2},
            "base_model.transformer.h.0.attn.c_proj": {"sigma_max": 16.8, "fro_norm": 128.6},
            "base_model.transformer.h.1.attn.c_attn": {"sigma_max": 17.9, "fro_norm": 142.1},
            "base_model.transformer.h.1.attn.c_proj": {"sigma_max": 16.4, "fro_norm": 125.8},
        }
    elif model_type == "bert":
        return {
            "base_model.encoder.layer.0.attention.self.query": {"sigma_max": 15.2, "fro_norm": 112.8},
            "base_model.encoder.layer.0.attention.self.value": {"sigma_max": 14.9, "fro_norm": 108.4},
        }
    return {}


class TestUDRIntegrationSimple:
    """Integration tests without external model dependencies."""
    
    def test_audit_json_schema_complete(self):
        """Validate complete audit.json output schema with UDR."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create realistic GPT-2 style adapter
            adapter_dir = create_realistic_test_adapter(temp_path, "gpt2")
            
            # Create base norms
            base_norms = create_base_norms_for_model("gpt2")
            cache_path = temp_path / "base_norms.json"
            with cache_path.open('w') as f:
                json.dump(base_norms, f)
            
            # Run audit with UDR
            cmd = [
                "gradience", "audit",
                "--peft-dir", str(adapter_dir),
                "--base-norms-cache", str(cache_path),
                "--json"
            ]
            
            result = subprocess.run(
                cmd, capture_output=True, text=True,
                env={**os.environ, "PYTHONPATH": "/Users/john/code/gradience"}
            )
            
            assert result.returncode == 0, f"Audit failed: {result.stderr}"
            
            audit_result = json.loads(result.stdout)
            
            # Validate complete schema
            required_top_level = [
                "n_layers", "total_lora_params", 
                "stable_rank_mean", "utilization_mean"
            ]
            
            for field in required_top_level:
                assert field in audit_result, f"Missing required field: {field}"
            
            # Validate UDR summary fields
            expected_udr_summary = [
                "udr_mean", "udr_median", "udr_p90", "udr_max",
                "sdi_mean", "sdi_median", "sdi_p90", 
                "fraction_udr_gt_0_1", "fraction_udr_gt_0_3",
                "n_layers_with_udr"
            ]
            
            for field in expected_udr_summary:
                assert field in audit_result, f"Missing UDR summary field: {field}"
                assert audit_result[field] is not None, f"UDR summary field {field} is None"
            
            # Validate layers have UDR fields
            if "layers" in audit_result:
                udr_layers = [l for l in audit_result["layers"] if l.get("udr") is not None]
                assert len(udr_layers) > 0, "No layers have UDR computed"
                
                # Check per-layer schema
                sample_layer = udr_layers[0]
                required_layer_fields = [
                    "delta_sigma_max", "delta_fro_norm", "scale",
                    "base_sigma_max", "udr", "sdi"
                ]
                
                for field in required_layer_fields:
                    assert field in sample_layer, f"Missing layer field: {field}"
            
            # Validate UDR values are reasonable
            assert audit_result["n_layers_with_udr"] > 0
            assert 0 < audit_result["udr_mean"] < 100  # Reasonable range
            assert audit_result["udr_median"] > 0
            
            print(f"✅ Schema validation passed")
            print(f"   - Layers with UDR: {audit_result['n_layers_with_udr']}/{audit_result['n_layers']}")
            print(f"   - UDR mean: {audit_result['udr_mean']:.4f}")
    
    def test_no_udr_flag_behavior(self):
        """Test --no-udr flag completely disables UDR computation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            adapter_dir = create_realistic_test_adapter(temp_path, "bert")
            base_norms = create_base_norms_for_model("bert")
            cache_path = temp_path / "base_norms.json"
            with cache_path.open('w') as f:
                json.dump(base_norms, f)
            
            # Test with --no-udr
            cmd = [
                "gradience", "audit",
                "--peft-dir", str(adapter_dir),
                "--base-norms-cache", str(cache_path),
                "--no-udr",
                "--json"
            ]
            
            result = subprocess.run(
                cmd, capture_output=True, text=True,
                env={**os.environ, "PYTHONPATH": "/Users/john/code/gradience"}
            )
            
            assert result.returncode == 0, f"Audit with --no-udr failed: {result.stderr}"
            
            audit_result = json.loads(result.stdout)
            
            # Should have basic audit but NO UDR fields
            assert audit_result["n_layers"] > 0, "Basic audit failed"
            
            # Check that UDR fields are absent
            udr_fields = [k for k in audit_result.keys() if 'udr' in k.lower() or 'sdi' in k.lower()]
            assert len(udr_fields) == 0, f"UDR fields present with --no-udr: {udr_fields}"
            
            # Check layers also have no UDR
            if "layers" in audit_result:
                for layer in audit_result["layers"]:
                    assert layer.get("udr") is None, "Layer has UDR despite --no-udr"
                    assert layer.get("sdi") is None, "Layer has SDI despite --no-udr"
            
            print("✅ --no-udr flag correctly disables UDR")
    
    def test_missing_base_norms_graceful_handling(self):
        """Test audit gracefully handles missing base norms."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            adapter_dir = create_realistic_test_adapter(temp_path, "gpt2")
            
            # Run audit WITHOUT base norms
            cmd = [
                "gradience", "audit",
                "--peft-dir", str(adapter_dir),
                "--json"
            ]
            
            result = subprocess.run(
                cmd, capture_output=True, text=True,
                env={**os.environ, "PYTHONPATH": "/Users/john/code/gradience"}
            )
            
            assert result.returncode == 0, f"Audit without base norms failed: {result.stderr}"
            
            audit_result = json.loads(result.stdout)
            
            # Should complete basic audit
            assert audit_result["n_layers"] > 0, "Basic audit failed"
            assert audit_result["total_lora_params"] > 0, "No LoRA params found"
            
            # Should have UDR summary fields but with appropriate values
            if "n_layers_with_udr" in audit_result:
                assert audit_result["n_layers_with_udr"] == 0, "Should have 0 layers with UDR without base norms"
            
            # Individual layers should not have UDR
            if "layers" in audit_result:
                for layer in audit_result["layers"]:
                    assert layer.get("udr") is None, "Layer has UDR without base norms"
                    # But should still have delta norms
                    assert layer.get("delta_sigma_max") is not None, "Missing delta norms"
            
            print("✅ Missing base norms handled gracefully")
    
    def test_corrupt_base_norms_cache_handling(self):
        """Test audit handles corrupt base norms cache gracefully."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            adapter_dir = create_realistic_test_adapter(temp_path, "gpt2")
            
            # Create corrupt cache file
            cache_path = temp_path / "corrupt_cache.json"
            with cache_path.open('w') as f:
                f.write("{ invalid json content here !@# }")
            
            cmd = [
                "gradience", "audit",
                "--peft-dir", str(adapter_dir),
                "--base-norms-cache", str(cache_path),
                "--json"
            ]
            
            result = subprocess.run(
                cmd, capture_output=True, text=True,
                env={**os.environ, "PYTHONPATH": "/Users/john/code/gradience"}
            )
            
            # Should not crash
            assert result.returncode == 0, f"Audit crashed on corrupt cache: {result.stderr}"
            
            audit_result = json.loads(result.stdout)
            
            # Should complete basic audit
            assert audit_result["n_layers"] > 0, "Basic audit failed"
            
            # Should record issues about cache loading
            if "issues" in audit_result and audit_result["issues"]:
                issues_text = " ".join(audit_result["issues"])
                assert any(word in issues_text.lower() for word in ["cache", "failed", "load"]), \
                    f"Expected cache loading issue in: {audit_result['issues']}"
            
            print("✅ Corrupt cache handled gracefully")
    
    def test_realistic_udr_values(self):
        """Test that UDR values are in realistic ranges for fine-tuning."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            adapter_dir = create_realistic_test_adapter(temp_path, "gpt2")
            base_norms = create_base_norms_for_model("gpt2")
            cache_path = temp_path / "base_norms.json"
            with cache_path.open('w') as f:
                json.dump(base_norms, f)
            
            cmd = [
                "gradience", "audit",
                "--peft-dir", str(adapter_dir),
                "--base-norms-cache", str(cache_path),
                "--json"
            ]
            
            result = subprocess.run(
                cmd, capture_output=True, text=True,
                env={**os.environ, "PYTHONPATH": "/Users/john/code/gradience"}
            )
            
            assert result.returncode == 0, f"Audit failed: {result.stderr}"
            
            audit_result = json.loads(result.stdout)
            
            # Check UDR values are in reasonable ranges for fine-tuning
            udr_mean = audit_result["udr_mean"]
            udr_median = audit_result["udr_median"]
            udr_max = audit_result["udr_max"]
            
            # For typical fine-tuning, UDR should be << 1 (small perturbations)
            # Our synthetic data might be larger, but should still be reasonable
            assert 0.001 < udr_mean < 10.0, f"UDR mean {udr_mean} outside reasonable range"
            assert 0.001 < udr_median < 10.0, f"UDR median {udr_median} outside reasonable range"
            assert udr_max > udr_mean, "UDR max should be >= UDR mean"
            
            # SDI should be finite and reasonable
            sdi_mean = audit_result["sdi_mean"]
            import math
            assert math.isfinite(sdi_mean), f"SDI mean {sdi_mean} is not finite"
            assert -5 < sdi_mean < 5, f"SDI mean {sdi_mean} outside reasonable log scale"
            
            # Fraction thresholds should be between 0 and 1
            frac_01 = audit_result["fraction_udr_gt_0_1"]
            frac_03 = audit_result["fraction_udr_gt_0_3"]
            assert 0 <= frac_01 <= 1, f"fraction_udr_gt_0_1 = {frac_01} outside [0,1]"
            assert 0 <= frac_03 <= 1, f"fraction_udr_gt_0_3 = {frac_03} outside [0,1]"
            assert frac_03 <= frac_01, "fraction_udr_gt_0_3 should be <= fraction_udr_gt_0_1"
            
            print(f"✅ UDR values in realistic ranges:")
            print(f"   - UDR mean/median/max: {udr_mean:.4f}/{udr_median:.4f}/{udr_max:.4f}")
            print(f"   - SDI mean: {sdi_mean:.4f}")
            print(f"   - Fractions > 0.1/0.3: {frac_01:.3f}/{frac_03:.3f}")


# Test runner
if __name__ == "__main__":
    test_instance = TestUDRIntegrationSimple()
    test_methods = [
        "test_audit_json_schema_complete",
        "test_no_udr_flag_behavior",
        "test_missing_base_norms_graceful_handling", 
        "test_corrupt_base_norms_cache_handling",
        "test_realistic_udr_values"
    ]
    
    total_tests = 0
    passed_tests = 0
    
    print("🧪 Running UDR Integration Tests (No External Dependencies)")
    print("=" * 60)
    
    for method_name in test_methods:
        total_tests += 1
        try:
            method = getattr(test_instance, method_name)
            method()
            print(f"✅ {method_name}")
            passed_tests += 1
        except Exception as e:
            print(f"❌ {method_name}: {e}")
    
    print(f"\n🎉 Integration Test Results: {passed_tests}/{total_tests} passed")
    
    if passed_tests == total_tests:
        print("✅ All UDR integration tests passed!")
        exit(0)
    else:
        print("❌ Some integration tests failed")
        exit(1)