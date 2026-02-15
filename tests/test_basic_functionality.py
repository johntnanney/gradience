"""
Basic functionality tests for CI.

Tests core imports and basic functionality without requiring heavy model downloads.
"""

import pytest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_CONFIG_DIR = _REPO_ROOT / "gradience" / "bench" / "configs"


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
        config_path = _CONFIG_DIR / "distilbert_sst2_ci.yaml"
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
        
    def test_compression_config_importable(self):
        """Test that compression config generation function is importable."""
        from gradience.bench.protocol import generate_compression_configs

        # generate_compression_configs(probe_dir, config) requires a real probe
        # directory with audit data — just verify it's importable and callable
        assert callable(generate_compression_configs)
    
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
        
        configs_dir = _CONFIG_DIR
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
        
        config_path = _CONFIG_DIR / "gpu_smoke" / "mistral_gsm8k_gpu_smoke.yaml"
        assert config_path.exists(), "GPU smoke config not found"
        
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        # Validate GPU-specific settings
        assert config["runtime"]["device"] == "cuda"
        assert config["train"]["max_steps"] == 20  # Fast smoke test
        assert config["audit"]["compute_udr"] == False  # UDR disabled for speed


if __name__ == "__main__":
    pytest.main([__file__])