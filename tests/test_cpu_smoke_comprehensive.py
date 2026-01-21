"""
Comprehensive CPU-only smoke tests.

This test suite verifies that all core pipeline components work without GPUs.
Designed for contributors to run quickly before commits.

Coverage:
- Config parsing → TaskProfile routing
- Task-specific formatting and evaluation
- Audit with minimal adapter artifacts  
- Aggregation with backward compatibility
- Core validation protocols
"""

import unittest
import tempfile
import json
from pathlib import Path


class TestCpuSmokeComprehensive(unittest.TestCase):
    """Comprehensive smoke test covering all critical pipeline components."""
    
    def test_end_to_end_config_to_profile_pipeline(self):
        """Test complete config → profile → tokenization pipeline."""
        from gradience.bench.task_profiles import get_task_profile_from_config
        from unittest.mock import Mock
        
        # Test GSM8K pipeline
        config = {
            "model": {"type": "causal_lm"},
            "task": {"dataset": "gsm8k", "profile": "gsm8k_causal_lm"},
            "train": {"train_samples": 10}
        }
        
        profile = get_task_profile_from_config(config)
        assert profile.name == "gsm8k_causal_lm"
        assert profile.primary_metric == "exact_match"
        
        # Test basic answer extraction
        test_response = "The answer is 42. #### 42"
        extracted = profile._extract_answer(test_response)
        assert extracted == "42"
        
        # Test GLUE pipeline
        config = {
            "model": {"type": "seqcls"},
            "task": {"dataset": "glue", "subset": "sst2"},
            "train": {"train_samples": 10}
        }
        
        profile = get_task_profile_from_config(config)
        assert profile.name == "seqcls_glue"
        assert profile.primary_metric == "accuracy"
    
    def test_minimal_audit_to_suggestions_pipeline(self):
        """Test audit → rank suggestions pipeline with minimal data."""
        import torch
        
        # Create minimal adapter artifacts
        temp_dir = Path(tempfile.mkdtemp())
        peft_dir = temp_dir / "test_adapter"
        peft_dir.mkdir()
        
        # Minimal adapter config
        config = {
            "r": 8,
            "lora_alpha": 16,
            "target_modules": ["q_proj", "v_proj"],
            "peft_type": "LORA"
        }
        with open(peft_dir / "adapter_config.json", "w") as f:
            json.dump(config, f)
        
        # Minimal adapter weights
        weights = {
            "base_model.model.layer.0.self_attn.q_proj.lora_A.weight": torch.randn(8, 64),
            "base_model.model.layer.0.self_attn.q_proj.lora_B.weight": torch.randn(64, 8),
            "base_model.model.layer.0.self_attn.v_proj.lora_A.weight": torch.randn(8, 64),
            "base_model.model.layer.0.self_attn.v_proj.lora_B.weight": torch.randn(64, 8),
        }
        torch.save(weights, peft_dir / "adapter_model.bin")
        
        # Test audit pipeline
        from gradience.vnext.audit import audit_lora_peft_dir
        from gradience.vnext.rank_suggestion import suggest_global_ranks_from_audit
        
        # Run audit
        audit_result = audit_lora_peft_dir(str(peft_dir))
        audit_dict = audit_result.to_summary_dict(include_layers=True)
        
        # Verify audit structure
        assert "layer_data" in audit_dict
        assert "layer_rows" in audit_dict["layer_data"]
        assert len(audit_dict["layer_data"]["layer_rows"]) > 0
        
        # Test rank suggestions (function expects audit dict, not result object)
        audit_summary = audit_result.to_summary_dict(include_layers=False)
        suggestions = suggest_global_ranks_from_audit(audit_summary)
        
        # Verify suggestion object structure
        assert hasattr(suggestions, "suggested_r_median")
        assert hasattr(suggestions, "suggested_r_p90")
        assert isinstance(suggestions.suggested_r_median, int)
        assert suggestions.suggested_r_median > 0
        assert isinstance(suggestions.suggested_r_p90, int)
        assert suggestions.suggested_r_p90 > 0
        
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir)
    
    def test_aggregation_with_different_report_formats(self):
        """Test aggregation handles old/new report formats gracefully."""
        from gradience.bench.protocol import create_multi_seed_aggregated_report
        
        temp_dir = Path(tempfile.mkdtemp())
        
        # Create reports with mixed task formats
        reports = [
            {
                "bench_version": "0.1",
                "model": "test-model",
                "task": "glue/sst2",  # Old style: string
                "env": {"seed": 42},
                "probe": {"rank": 16, "accuracy": 0.85, "params": 1000},
                "compressed": {},
                "summary": {"best_compression": None}
            },
            {
                "bench_version": "0.1", 
                "model": "test-model",
                "task": {"dataset": "glue", "subset": "sst2"},  # New style: dict
                "env": {"seed": 43},
                "probe": {"rank": 16, "accuracy": 0.87, "params": 1000},
                "compressed": {},
                "summary": {"best_compression": None}
            }
        ]
        
        config = {"bench_version": "0.1"}
        result = create_multi_seed_aggregated_report(reports, config, temp_dir)
        
        # Should aggregate without errors
        assert result["n_seeds"] == 2
        assert "probe" in result
        assert "accuracy" in result["probe"]
        assert "mean" in result["probe"]["accuracy"]
        
        # Should use first report's task format
        assert result["task"] == "glue/sst2"
        
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir)
    
    def test_protocol_metadata_inclusion(self):
        """Test that new self-describing metadata is properly included."""
        from gradience.bench.protocol import (
            gather_environment_info, get_git_commit, get_git_tag,
            get_primary_metric_key, create_config_hash, 
            extract_model_dataset_info
        )
        
        # Test environment gathering
        env_info = gather_environment_info()
        required_keys = ["python_version", "torch_version", "cuda_available"]
        for key in required_keys:
            assert key in env_info, f"Missing required env key: {key}"
        
        # Test git info (might be None in some environments)
        git_commit = get_git_commit()
        git_tag = get_git_tag()
        # Just ensure they don't crash
        assert git_commit is None or isinstance(git_commit, str)
        assert git_tag is None or isinstance(git_tag, str)
        
        # Test config utilities
        config = {
            "task": {"dataset": "gsm8k"},
            "model": {"name": "test-model"}
        }
        
        metric_key = get_primary_metric_key(config)
        assert metric_key == "eval_exact_match"  # GSM8K uses exact_match
        
        config_hash = create_config_hash(config)
        assert isinstance(config_hash, str)
        assert len(config_hash) == 16  # Should be 16 char hash
        
        # Test model/dataset extraction
        config_with_metadata = {
            "model_id": "microsoft/DialoGPT-small",
            "dataset": {"name": "glue", "split": "train"}
        }
        metadata = extract_model_dataset_info(config_with_metadata)
        assert "model_info" in metadata
        assert "dataset_info" in metadata
    
    def test_error_handling_robustness(self):
        """Test that core components handle edge cases gracefully."""
        from gradience.bench.task_profiles import get_task_profile_from_config
        
        # Test empty config
        empty_config = {}
        profile = get_task_profile_from_config(empty_config)
        assert profile.name == "seqcls_glue"  # Should default to GLUE
        
        # Test malformed config
        malformed_config = {"task": {"dataset": None}}
        profile = get_task_profile_from_config(malformed_config)
        assert profile.name == "seqcls_glue"  # Should handle gracefully
        
        # Test GSM8K profile with edge case answers
        from gradience.bench.task_profiles.gsm8k_causal_lm import GSM8KCausalLMProfile
        gsm8k_profile = GSM8KCausalLMProfile()
        
        # Test various answer formats
        assert gsm8k_profile._extract_answer("No numbers here") == ""
        assert gsm8k_profile._extract_answer("Multiple 12 numbers 34 here") == "34"
        assert gsm8k_profile._extract_answer("1,000 with commas") == "1000"
    
    def test_performance_smoke_check(self):
        """Verify key operations complete quickly (smoke test for performance)."""
        import time
        
        # Config parsing should be very fast
        start_time = time.time()
        for _ in range(100):
            from gradience.bench.task_profiles import get_task_profile
            profile = get_task_profile("gsm8k_causal_lm")
            assert profile.primary_metric == "exact_match"
        duration = time.time() - start_time
        assert duration < 1.0, f"Config parsing too slow: {duration:.2f}s"
        
        # Environment gathering should be reasonably fast
        start_time = time.time()
        from gradience.bench.protocol import gather_environment_info
        env_info = gather_environment_info()
        duration = time.time() - start_time
        assert duration < 5.0, f"Environment gathering too slow: {duration:.2f}s"
        assert "python_version" in env_info


if __name__ == "__main__":
    unittest.main()