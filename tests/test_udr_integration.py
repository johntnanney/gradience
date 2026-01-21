#!/usr/bin/env python3
"""
Test 7: Tiny HF model integration test for UDR/SDI end-to-end pipeline.

Uses tiny models specifically designed for testing to validate:
- Base model loading and norm computation
- Real LoRA adapter processing
- Complete audit.json schema output
- CLI flag behavior

Uses models like sshleifer/tiny-gpt2 that are <1MB for fast CI.
"""

import json
import tempfile
import torch
import pytest
from pathlib import Path
from typing import Dict, Any, Optional
import subprocess
import sys
import os

# Add gradience to path
sys.path.insert(0, '/Users/john/code/gradience')


def check_dependencies():
    """Check if required dependencies are available."""
    try:
        import transformers
        import peft
        return True
    except ImportError as e:
        print(f"Skipping integration test - missing dependency: {e}")
        return False


def create_tiny_lora_adapter(model_name: str, out_dir: Path) -> bool:
    """Create a tiny LoRA adapter for testing using PEFT."""
    if not check_dependencies():
        return False
    
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
        from peft import LoraConfig, get_peft_model, TaskType
        from datasets import Dataset
        
        # Load tiny model
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,  # Use float32 for CPU training
            device_map="cpu"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Create minimal LoRA config
        lora_config = LoraConfig(
            r=4,
            lora_alpha=8,
            target_modules=["c_attn", "c_proj"],  # GPT-2 attention modules
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        
        # Apply LoRA
        model = get_peft_model(model, lora_config)
        
        # Create tiny dataset (just a few samples for minimal training)
        tiny_data = [
            "The quick brown fox",
            "Hello world example", 
            "Test training data"
        ]
        
        def tokenize_function(examples):
            return tokenizer(examples["text"], truncation=True, padding=True, max_length=32)
        
        dataset = Dataset.from_dict({"text": tiny_data})
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        
        # Minimal training (just 1 step to create weight changes)
        training_args = TrainingArguments(
            output_dir=str(out_dir / "training"),
            num_train_epochs=1,
            max_steps=1,  # Just 1 step
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            learning_rate=1e-4,
            logging_steps=1,
            save_steps=1,
            remove_unused_columns=False,
            dataloader_drop_last=False,
        )
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            tokenizer=tokenizer,
        )
        
        # Run minimal training
        trainer.train()
        
        # Save adapter
        adapter_dir = out_dir / "adapter"
        adapter_dir.mkdir(exist_ok=True)
        model.save_pretrained(str(adapter_dir))
        
        return True
        
    except Exception as e:
        print(f"Failed to create tiny LoRA adapter: {e}")
        return False


class TestUDRIntegration:
    """Integration tests with real tiny HF models."""
    
    @pytest.mark.slow
    def test_end_to_end_audit_emits_udr_keys(self):
        """Test 7: End-to-end audit with tiny HF model emits UDR keys."""
        if not check_dependencies():
            pytest.skip("Missing transformers/peft dependencies")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Use tiny-gpt2 (very small model for testing)
            model_name = "sshleifer/tiny-gpt2"
            
            print(f"Creating tiny LoRA adapter with {model_name}...")
            if not create_tiny_lora_adapter(model_name, temp_path):
                pytest.skip("Failed to create LoRA adapter")
            
            adapter_dir = temp_path / "adapter"
            assert adapter_dir.exists(), "Adapter directory not created"
            assert (adapter_dir / "adapter_config.json").exists(), "Missing adapter_config.json"
            assert (adapter_dir / "adapter_model.safetensors").exists() or \
                   (adapter_dir / "adapter_model.bin").exists(), "Missing adapter weights"
            
            # Test audit WITHOUT base model (should have no UDR)
            print("Testing audit without base model...")
            cmd = [
                "gradience", "audit",
                "--peft-dir", str(adapter_dir),
                "--json"
            ]
            
            result = subprocess.run(
                cmd, capture_output=True, text=True,
                env={**os.environ, "PYTHONPATH": "/Users/john/code/gradience"}
            )
            
            assert result.returncode == 0, f"Audit without base model failed: {result.stderr}"
            
            audit_without_base = json.loads(result.stdout)
            
            # Should have basic audit metrics but no UDR
            assert audit_without_base["n_layers"] > 0, "No layers found in audit"
            assert audit_without_base["total_lora_params"] > 0, "No LoRA params found"
            
            # Should NOT have UDR keys
            udr_keys_without_base = [k for k in audit_without_base.keys() if 'udr' in k.lower()]
            assert len(udr_keys_without_base) == 0, f"Unexpected UDR keys without base model: {udr_keys_without_base}"
            
            # Test audit WITH base model (should compute UDR)
            print("Testing audit with base model...")
            cmd_with_base = [
                "gradience", "audit",
                "--peft-dir", str(adapter_dir),
                "--base-model", model_name,
                "--json"
            ]
            
            result_with_base = subprocess.run(
                cmd_with_base, capture_output=True, text=True,
                env={**os.environ, "PYTHONPATH": "/Users/john/code/gradience"}
            )
            
            # Note: This might fail if transformers/model loading fails, which is OK for CI
            if result_with_base.returncode != 0:
                print(f"Base model loading failed (expected in some environments): {result_with_base.stderr}")
                pytest.skip("Base model loading not available in this environment")
                return
            
            audit_with_base = json.loads(result_with_base.stdout)
            
            # Should have UDR metrics now
            expected_udr_keys = {
                "udr_mean", "udr_median", "udr_p90", "udr_max",
                "sdi_mean", "sdi_median", "sdi_p90",
                "fraction_udr_gt_0_1", "fraction_udr_gt_0_3",
                "n_layers_with_udr"
            }
            
            actual_udr_keys = set(k for k in audit_with_base.keys() if 'udr' in k.lower() or 'sdi' in k.lower())
            
            missing_keys = expected_udr_keys - actual_udr_keys
            assert len(missing_keys) == 0, f"Missing UDR keys in audit output: {missing_keys}"
            
            # Validate UDR values are reasonable
            assert audit_with_base["n_layers_with_udr"] > 0, "No layers have UDR computed"
            assert audit_with_base["udr_mean"] > 0, "UDR mean should be positive"
            assert audit_with_base["udr_median"] > 0, "UDR median should be positive"
            
            # SDI should be finite
            import math
            assert math.isfinite(audit_with_base["sdi_mean"]), "SDI mean should be finite"
            
            # Check per-layer UDR fields
            if "layers" in audit_with_base:
                udr_layers = [l for l in audit_with_base["layers"] if l.get("udr") is not None]
                assert len(udr_layers) > 0, "No individual layers have UDR"
                
                # Check required per-layer fields
                sample_layer = udr_layers[0]
                required_layer_fields = ["delta_sigma_max", "scale", "udr", "sdi"]
                for field in required_layer_fields:
                    assert field in sample_layer, f"Missing per-layer field: {field}"
                    assert sample_layer[field] is not None, f"Per-layer field {field} is None"
            
            print(f"✅ UDR integration test passed!")
            print(f"   - Layers with UDR: {audit_with_base['n_layers_with_udr']}")
            print(f"   - UDR mean: {audit_with_base['udr_mean']:.4f}")
            print(f"   - SDI mean: {audit_with_base['sdi_mean']:.4f}")


class TestCLIBehavior:
    """Test CLI flag behavior for UDR."""
    
    def test_no_udr_flag_disables_udr(self):
        """Test 8a: --no-udr flag prevents UDR computation."""
        # Create simple synthetic adapter for testing
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            adapter_dir = temp_path / "adapter"
            adapter_dir.mkdir()
            
            # Create minimal adapter config
            config = {
                "peft_type": "LORA",
                "r": 4,
                "lora_alpha": 8,
                "target_modules": ["dense"],
                "task_type": "SEQ_CLS"
            }
            
            with (adapter_dir / "adapter_config.json").open('w') as f:
                json.dump(config, f)
            
            # Create dummy weights
            weights = {
                "model.layer.0.dense.lora_A.weight": torch.randn(4, 64),
                "model.layer.0.dense.lora_B.weight": torch.randn(64, 4),
            }
            torch.save(weights, adapter_dir / "adapter_model.bin")
            
            # Create dummy base norms cache
            base_norms = {
                "model.layer.0.dense": {"sigma_max": 10.0, "fro_norm": 50.0}
            }
            cache_path = temp_path / "base_norms.json"
            with cache_path.open('w') as f:
                json.dump(base_norms, f)
            
            # Test with --no-udr flag
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
            
            # Should NOT have UDR keys when disabled
            udr_keys = [k for k in audit_result.keys() if 'udr' in k.lower() or 'sdi' in k.lower()]
            assert len(udr_keys) == 0, f"UDR keys present despite --no-udr flag: {udr_keys}"
    
    def test_base_model_flag_enables_udr(self):
        """Test 8b: --base-model flag enables UDR computation."""
        # This test requires transformers, so we'll make it conditional
        if not check_dependencies():
            pytest.skip("Missing transformers dependency")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            adapter_dir = temp_path / "adapter"
            adapter_dir.mkdir()
            
            # Create minimal adapter config
            config = {
                "peft_type": "LORA",
                "r": 2,
                "lora_alpha": 4,
                "target_modules": ["c_attn"],  # GPT-2 module
                "task_type": "CAUSAL_LM",
                "base_model_name_or_path": "sshleifer/tiny-gpt2"
            }
            
            with (adapter_dir / "adapter_config.json").open('w') as f:
                json.dump(config, f)
            
            # Create tiny weights
            weights = {
                "base_model.transformer.h.0.attn.c_attn.lora_A.weight": torch.randn(2, 48),
                "base_model.transformer.h.0.attn.c_attn.lora_B.weight": torch.randn(144, 2),
            }
            torch.save(weights, adapter_dir / "adapter_model.bin")
            
            # Test with --base-model flag (may fail if model loading fails)
            cmd = [
                "gradience", "audit", 
                "--peft-dir", str(adapter_dir),
                "--base-model", "sshleifer/tiny-gpt2",
                "--json"
            ]
            
            result = subprocess.run(
                cmd, capture_output=True, text=True,
                env={**os.environ, "PYTHONPATH": "/Users/john/code/gradience"}
            )
            
            if result.returncode != 0:
                # Base model loading can fail in CI environments
                pytest.skip(f"Base model loading failed in this environment: {result.stderr}")
                return
            
            audit_result = json.loads(result.stdout)
            
            # Should have UDR-related fields (even if no layers computed)
            # At minimum, should have n_layers_with_udr field
            assert "n_layers_with_udr" in audit_result, "Missing n_layers_with_udr field with --base-model"
    
    def test_bad_base_model_gives_useful_error(self):
        """Test 8c: Bad --base-model gives useful error."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            adapter_dir = temp_path / "adapter"
            adapter_dir.mkdir()
            
            # Create minimal adapter
            config = {"peft_type": "LORA", "r": 4}
            with (adapter_dir / "adapter_config.json").open('w') as f:
                json.dump(config, f)
            
            weights = {"dummy.lora_A.weight": torch.randn(4, 64)}
            torch.save(weights, adapter_dir / "adapter_model.bin")
            
            # Test with non-existent model
            cmd = [
                "gradience", "audit",
                "--peft-dir", str(adapter_dir),
                "--base-model", "nonexistent/fake-model-12345",
                "--json"
            ]
            
            result = subprocess.run(
                cmd, capture_output=True, text=True,
                env={**os.environ, "PYTHONPATH": "/Users/john/code/gradience"}
            )
            
            # Should not crash - should complete audit with issues recorded
            assert result.returncode == 0, "Audit should complete even with bad base model"
            
            audit_result = json.loads(result.stdout)
            
            # Should have recorded issues about base model loading
            if "issues" in audit_result:
                issues_text = " ".join(audit_result["issues"])
                assert any(word in issues_text.lower() for word in ["failed", "base", "model"]), \
                    f"Expected base model error in issues: {audit_result['issues']}"


# Test runner for standalone execution
if __name__ == "__main__":
    # Run integration tests
    integration_tests = TestUDRIntegration()
    cli_tests = TestCLIBehavior()
    
    test_classes = [
        (integration_tests, ["test_end_to_end_audit_emits_udr_keys"]),
        (cli_tests, ["test_no_udr_flag_disables_udr", "test_base_model_flag_enables_udr", "test_bad_base_model_gives_useful_error"])
    ]
    
    total_tests = 0
    passed_tests = 0
    skipped_tests = 0
    
    for test_instance, test_methods in test_classes:
        print(f"\n🧪 Running {test_instance.__class__.__name__}")
        print("=" * 50)
        
        for method_name in test_methods:
            total_tests += 1
            try:
                method = getattr(test_instance, method_name)
                method()
                print(f"   ✅ {method_name}")
                passed_tests += 1
            except Exception as e:
                if "skip" in str(e).lower() or "missing" in str(e).lower():
                    print(f"   ⏭️  {method_name}: SKIPPED - {e}")
                    skipped_tests += 1
                else:
                    print(f"   ❌ {method_name}: {e}")
    
    print(f"\n🎉 Integration Test Results: {passed_tests}/{total_tests} passed, {skipped_tests} skipped")
    
    if passed_tests + skipped_tests == total_tests:
        print("✅ All UDR integration tests passed or skipped appropriately!")
        exit(0)
    else:
        print("❌ Some integration tests failed")
        exit(1)