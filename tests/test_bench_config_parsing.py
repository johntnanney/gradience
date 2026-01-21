"""
Smoke tests for bench config parsing and task profile routing.

Tests that config parsing routes to correct TaskProfile instances
without requiring GPU or actual model downloads.
"""

import pytest
from gradience.bench.task_profiles import get_task_profile_from_config, get_task_profile
from gradience.bench.task_profiles.gsm8k_causal_lm import GSM8KCausalLMProfile
from gradience.bench.task_profiles.seqcls_glue import GLUESequenceClassificationProfile


class TestBenchConfigParsing:
    """Test bench configuration parsing and task profile routing."""
    
    def test_explicit_profile_routing(self):
        """Test explicit task profile specification."""
        # Test explicit GSM8K profile
        config = {
            "task": {"profile": "gsm8k_causal_lm"}
        }
        profile = get_task_profile_from_config(config)
        assert isinstance(profile, GSM8KCausalLMProfile)
        assert profile.name == "gsm8k_causal_lm"
        assert profile.primary_metric == "exact_match"
        
        # Test explicit GLUE profile
        config = {
            "task": {"profile": "seqcls_glue"}
        }
        profile = get_task_profile_from_config(config)
        assert isinstance(profile, GLUESequenceClassificationProfile)
        assert profile.name == "seqcls_glue"
        assert profile.primary_metric == "accuracy"
    
    def test_backward_compatibility_causal_lm_gsm8k(self):
        """Test backward compatibility: model type + dataset → profile inference."""
        config = {
            "model": {"type": "causal_lm"},
            "task": {"dataset": "gsm8k"}
        }
        profile = get_task_profile_from_config(config)
        assert isinstance(profile, GSM8KCausalLMProfile)
        
        # Test case-insensitive dataset matching
        config = {
            "model": {"type": "causal_lm"},
            "task": {"dataset": "GSM8K"}
        }
        profile = get_task_profile_from_config(config)
        assert isinstance(profile, GSM8KCausalLMProfile)
    
    def test_backward_compatibility_default_seqcls(self):
        """Test backward compatibility: default to sequence classification."""
        # Default model type
        config = {
            "model": {},
            "task": {"dataset": "glue/sst2"}
        }
        profile = get_task_profile_from_config(config)
        assert isinstance(profile, GLUESequenceClassificationProfile)
        
        # Explicit seqcls type
        config = {
            "model": {"type": "seqcls"},
            "task": {"dataset": "glue/cola"}
        }
        profile = get_task_profile_from_config(config)
        assert isinstance(profile, GLUESequenceClassificationProfile)
    
    def test_registry_direct_access(self):
        """Test direct registry access."""
        profile = get_task_profile("gsm8k_causal_lm")
        assert isinstance(profile, GSM8KCausalLMProfile)
        
        profile = get_task_profile("seqcls_glue")
        assert isinstance(profile, GLUESequenceClassificationProfile)
    
    def test_invalid_profile_errors(self):
        """Test that invalid configurations raise helpful errors."""
        # Unknown profile name
        with pytest.raises(ValueError, match="Unknown task profile 'invalid_profile'"):
            get_task_profile("invalid_profile")
        
        # Causal LM with unsupported dataset
        config = {
            "model": {"type": "causal_lm"},
            "task": {"dataset": "unsupported_dataset"}
        }
        with pytest.raises(ValueError, match="No task profile available for causal_lm"):
            get_task_profile_from_config(config)
    
    def test_config_edge_cases(self):
        """Test edge cases in configuration parsing."""
        # Missing task section
        config = {"model": {"type": "seqcls"}}
        profile = get_task_profile_from_config(config)
        assert isinstance(profile, GLUESequenceClassificationProfile)
        
        # Missing model section
        config = {"task": {"dataset": "glue/sst2"}}
        profile = get_task_profile_from_config(config)
        assert isinstance(profile, GLUESequenceClassificationProfile)
        
        # Empty configuration
        config = {}
        profile = get_task_profile_from_config(config)
        assert isinstance(profile, GLUESequenceClassificationProfile)
    
    def test_profile_attributes(self):
        """Test that profiles have required attributes."""
        for profile_name in ["gsm8k_causal_lm", "seqcls_glue"]:
            profile = get_task_profile(profile_name)
            
            # Required attributes
            assert hasattr(profile, "name")
            assert hasattr(profile, "primary_metric")
            assert hasattr(profile, "load")
            assert hasattr(profile, "tokenize")
            assert hasattr(profile, "probe_gate")
            
            # Attributes should have reasonable values
            assert isinstance(profile.name, str)
            assert len(profile.name) > 0
            assert isinstance(profile.primary_metric, str)
            assert len(profile.primary_metric) > 0


if __name__ == "__main__":
    pytest.main([__file__])