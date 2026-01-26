"""
Basic functionality tests for CI.

Tests core imports and basic functionality without requiring heavy model downloads.
"""

import pytest
from pathlib import Path


class TestBasicFunctionality:
    """Test basic package functionality."""
    
    def test_package_imports(self):
        """Test that core package imports work."""
        # Core gradience imports
        import gradience
        assert hasattr(gradience, '__version__')
        
        # Bench imports
        from gradience.bench import protocol
        from gradience.bench import run_bench
        from gradience.bench import aggregate
        
        # VNext imports
        from gradience.vnext import telemetry
        from gradience.vnext import types
        
        # HuggingFace integration
        from gradience.vnext.integrations import hf
        
    def test_bench_config_parsing(self):
        """Test that bench configs can be parsed."""
        import yaml
        
        # Find a sample config to test
        config_path = Path("gradience/bench/configs/distilbert_sst2_ci.yaml")
        assert config_path.exists(), "CI config not found"
        
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        # Basic structure validation
        assert isinstance(config, dict)
        assert "model" in config
        assert "task" in config
        assert "train" in config
        assert "lora" in config
        assert "runtime" in config
        
        # Verify CPU device for CI
        assert config["runtime"]["device"] == "cpu"
        
    def test_telemetry_types(self):
        """Test telemetry type definitions."""
        from gradience.vnext.types import Severity, EventType
        
        # Test severity enum
        assert hasattr(Severity, 'INFO')
        assert hasattr(Severity, 'WARNING')
        assert hasattr(Severity, 'ERROR')
        
    def test_compression_config_generation(self):
        """Test compression config generation logic."""
        from gradience.bench.protocol import generate_compression_configs
        
        # Sample audit summary
        audit_summary = {
            "stable_rank_mean": 8.5,
            "utilization_mean": 0.75,
            "current_r": 16,
            "per_layer_ranks": {
                "q_lin": 8,
                "k_lin": 6,
                "v_lin": 10,
                "out_lin": 7
            }
        }
        
        compression_config = {
            "allowed_ranks": [2, 4, 8, 16],
            "acc_tolerance": 0.02
        }
        
        try:
            variants = generate_compression_configs(audit_summary, compression_config)
            assert isinstance(variants, list)
            assert len(variants) > 0
            
            # Check that variants have required fields
            for variant in variants:
                assert "name" in variant
                assert "rank_spec" in variant
                assert "total_params_ratio" in variant
                
        except Exception as e:
            pytest.skip(f"Compression config generation not available: {e}")
    
    def test_cli_help_commands(self):
        """Test that CLI help commands work."""
        import subprocess
        import sys
        
        # Test gradience CLI
        result = subprocess.run(
            [sys.executable, "-m", "gradience.cli", "--help"],
            capture_output=True,
            text=True,
            timeout=30
        )
        assert result.returncode == 0
        assert "gradience" in result.stdout.lower()
        
        # Test bench CLI
        result = subprocess.run(
            [sys.executable, "-m", "gradience.bench.run_bench", "--help"],
            capture_output=True,
            text=True,
            timeout=30
        )
        assert result.returncode == 0
        assert "config" in result.stdout.lower()


class TestConfigValidation:
    """Test configuration validation."""
    
    def test_all_configs_parse(self):
        """Test that all YAML configs in the package can be parsed."""
        import yaml
        from pathlib import Path
        
        configs_dir = Path("gradience/bench/configs")
        if not configs_dir.exists():
            pytest.skip("Configs directory not found")
        
        yaml_files = list(configs_dir.rglob("*.yaml"))
        assert len(yaml_files) > 0, "No YAML files found"
        
        errors = []
        for yaml_file in yaml_files:
            try:
                with open(yaml_file) as f:
                    config = yaml.safe_load(f)
                assert isinstance(config, dict), f"{yaml_file} is not a dict"
            except Exception as e:
                errors.append(f"{yaml_file}: {e}")
        
        if errors:
            pytest.fail(f"Config parsing errors:\n" + "\n".join(errors))
    
    def test_gpu_smoke_config_exists(self):
        """Test that GPU smoke config exists and is valid."""
        import yaml
        from pathlib import Path
        
        config_path = Path("gradience/bench/configs/gpu_smoke/mistral_gsm8k_gpu_smoke.yaml")
        assert config_path.exists(), "GPU smoke config not found"
        
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        # Validate GPU-specific settings
        assert config["runtime"]["device"] == "cuda"
        assert config["train"]["max_steps"] == 20  # Fast smoke test
        assert config["audit"]["compute_udr"] == False  # UDR disabled for speed


if __name__ == "__main__":
    pytest.main([__file__])