"""
Test that configs are accessible via importlib.resources.

This ensures that the packaging setup correctly includes all YAML files
and they can be accessed programmatically in the installed package.
"""

import pytest
import importlib.resources
import gradience.bench.configs
import gradience.bench.policies


class TestPackagingResources:
    """Test config access via importlib.resources."""
    
    def test_main_configs_accessible(self):
        """Test that main config files are accessible."""
        try:
            configs_files = list(importlib.resources.files(gradience.bench.configs).iterdir())
        except Exception as e:
            pytest.fail(f"Failed to access configs directory: {e}")
            
        yaml_files = [f for f in configs_files if f.name.endswith('.yaml')]
        
        # Should have 40+ config files
        assert len(yaml_files) > 40, f"Expected >40 config files, got {len(yaml_files)}"
        
        # Check for some expected configs
        config_names = [f.name for f in yaml_files]
        expected_configs = [
            'distilbert_sst2.yaml',
            'distilbert_sst2_ci.yaml', 
            'mistral_gsm8k_certifiable_seed42.yaml'
        ]
        
        for expected in expected_configs:
            assert expected in config_names, f"Missing expected config: {expected}"
    
    def test_evidence_configs_accessible(self):
        """Test that evidence subdirectory configs are accessible."""
        try:
            evidence_files = list(
                importlib.resources.files(gradience.bench.configs)
                .joinpath('evidence').iterdir()
            )
        except Exception as e:
            pytest.fail(f"Failed to access evidence configs: {e}")
            
        evidence_yamls = [f for f in evidence_files if f.name.endswith('.yaml')]
        
        # Should have at least 3 evidence configs
        assert len(evidence_yamls) >= 3, f"Expected >=3 evidence configs, got {len(evidence_yamls)}"
        
        # Check for expected evidence configs
        evidence_names = [f.name for f in evidence_yamls]
        expected_evidence = ['exp02_distilbert_sst2.yaml']
        
        for expected in expected_evidence:
            assert expected in evidence_names, f"Missing evidence config: {expected}"
    
    def test_gpu_smoke_configs_accessible(self):
        """Test that gpu_smoke subdirectory configs are accessible.""" 
        try:
            gpu_smoke_files = list(
                importlib.resources.files(gradience.bench.configs)
                .joinpath('gpu_smoke').iterdir()
            )
        except Exception as e:
            pytest.fail(f"Failed to access gpu_smoke configs: {e}")
            
        gpu_smoke_yamls = [f for f in gpu_smoke_files if f.name.endswith('.yaml')]
        
        # Should have at least 1 GPU smoke config
        assert len(gpu_smoke_yamls) >= 1, f"Expected >=1 GPU smoke config, got {len(gpu_smoke_yamls)}"
    
    def test_policies_accessible(self):
        """Test that policy files are accessible."""
        try:
            policy_files = list(importlib.resources.files(gradience.bench.policies).iterdir())
        except Exception as e:
            pytest.fail(f"Failed to access policies directory: {e}")
            
        policy_yamls = [f for f in policy_files if f.name.endswith('.yaml')]
        
        # Should have at least 1 policy file
        assert len(policy_yamls) >= 1, f"Expected >=1 policy file, got {len(policy_yamls)}"
        
        # Check for expected policy
        policy_names = [f.name for f in policy_yamls]
        assert 'safe_uniform.yaml' in policy_names, "Missing safe_uniform.yaml policy"
    
    def test_config_content_readable(self):
        """Test that config file contents can be read."""
        try:
            # Test reading a main config
            config_content = (
                importlib.resources.files(gradience.bench.configs)
                .joinpath('distilbert_sst2.yaml')
                .read_text()
            )
            assert len(config_content) > 0, "Config file appears to be empty"
            assert 'model:' in config_content, "Config doesn't contain expected YAML structure"
            
            # Test reading a policy
            policy_content = (
                importlib.resources.files(gradience.bench.policies)
                .joinpath('safe_uniform.yaml')
                .read_text()
            )
            assert len(policy_content) > 0, "Policy file appears to be empty"
            
        except Exception as e:
            pytest.fail(f"Failed to read config content: {e}")
    
    def test_backwards_compatible_access(self):
        """Test that configs are also accessible via traditional file paths."""
        import gradience.bench.configs
        import os
        
        # This should work for backwards compatibility
        config_dir = os.path.dirname(gradience.bench.configs.__file__)
        assert os.path.exists(config_dir), "Config directory not accessible via __file__"
        
        yaml_files = [f for f in os.listdir(config_dir) if f.endswith('.yaml')]
        assert len(yaml_files) > 40, f"Traditional access found {len(yaml_files)} files"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])