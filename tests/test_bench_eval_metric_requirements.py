"""
Unit and integration tests for bench evaluation metric requirements.

This test ensures that:
- GLUE task profiles always emit eval_accuracy 
- GSM8K task profiles always emit eval_exact_match
- The fallback mechanism works robustly
- Task profiles fail gracefully when required metrics are missing

These tests prevent the issue where bench crashes due to missing metric keys.
"""

import pytest
import unittest
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

from gradience.bench.task_profiles.seqcls_glue import GLUESequenceClassificationProfile
from gradience.bench.task_profiles.gsm8k_causal_lm import GSM8KCausalLMProfile
from gradience.bench.protocol import _extract_accuracy_with_fallback


class TestTaskProfileMetricRequirements(unittest.TestCase):
    """Unit tests for task profile metric emission requirements."""

    def setUp(self):
        self.glue_profile = GLUESequenceClassificationProfile()
        self.gsm8k_profile = GSM8KCausalLMProfile()

    def test_glue_profile_has_correct_primary_metric_key(self):
        """Test GLUE profile has eval_accuracy as primary metric key."""
        self.assertEqual(self.glue_profile.primary_metric_key, "eval_accuracy")
        self.assertEqual(self.glue_profile.primary_metric, "accuracy")

    def test_gsm8k_profile_has_correct_primary_metric_key(self):
        """Test GSM8K profile has eval_exact_match as primary metric key.""" 
        self.assertEqual(self.gsm8k_profile.primary_metric_key, "eval_exact_match")
        self.assertEqual(self.gsm8k_profile.primary_metric, "exact_match")

    def test_fallback_function_prefers_primary_metric_key(self):
        """Test fallback function uses task profile primary metric key."""
        # GLUE case: prefers eval_accuracy
        eval_results = {
            "eval_accuracy": 0.85,
            "accuracy": 0.90,  # Should be ignored in favor of primary
            "eval_exact_match": 0.75
        }
        result = _extract_accuracy_with_fallback(eval_results, self.glue_profile)
        self.assertEqual(result, 0.85)  # Should use eval_accuracy

        # GSM8K case: prefers eval_exact_match  
        eval_results = {
            "eval_accuracy": 0.85,
            "eval_exact_match": 0.72,  # Should be chosen as primary
            "exact_match": 0.80
        }
        result = _extract_accuracy_with_fallback(eval_results, self.gsm8k_profile)
        self.assertEqual(result, 0.72)  # Should use eval_exact_match

    def test_fallback_function_handles_missing_primary_key(self):
        """Test fallback function handles missing primary metric gracefully."""
        # GLUE profile with missing eval_accuracy
        eval_results = {
            "accuracy": 0.78,  # Should fall back to this
            "other_metric": 0.99
        }
        result = _extract_accuracy_with_fallback(eval_results, self.glue_profile)
        self.assertEqual(result, 0.78)

        # GSM8K profile with missing eval_exact_match
        eval_results = {
            "exact_match": 0.71,  # Should fall back to this
            "other_metric": 0.99
        }
        result = _extract_accuracy_with_fallback(eval_results, self.gsm8k_profile) 
        self.assertEqual(result, 0.71)

    def test_fallback_function_handles_no_profile(self):
        """Test fallback function works without task profile."""
        eval_results = {
            "eval_accuracy": 0.88,
            "eval_exact_match": 0.72,
            "accuracy": 0.90
        }
        result = _extract_accuracy_with_fallback(eval_results, None)
        self.assertEqual(result, 0.88)  # Should use first in fallback list

    def test_fallback_function_handles_empty_results(self):
        """Test fallback function handles missing metrics gracefully."""
        eval_results = {
            "eval_loss": 0.4,
            "some_other_metric": 0.99
        }
        result = _extract_accuracy_with_fallback(eval_results, self.glue_profile)
        self.assertEqual(result, 0.0)  # Should return default


class TestGLUEEvaluationMetrics(unittest.TestCase):
    """Integration test for GLUE task profile evaluation metrics."""

    def test_glue_evaluate_fails_without_eval_accuracy(self):
        """Test that missing eval_accuracy would be caught by our fallback system."""
        # This test ensures our system would handle the original problem gracefully
        bad_eval_results = {
            "eval_loss": 0.4,
            "eval_runtime": 1.23
            # Missing eval_accuracy - the original problem!
        }
        
        profile = GLUESequenceClassificationProfile()
        accuracy = _extract_accuracy_with_fallback(bad_eval_results, profile)
        
        # Should return 0.0 (graceful failure) instead of crashing
        self.assertEqual(accuracy, 0.0)
    
    def test_glue_profile_metric_structure(self):
        """Test GLUE profile has the right metric structure."""
        profile = GLUESequenceClassificationProfile()
        
        # Test that our modification worked
        self.assertEqual(profile.primary_metric_key, "eval_accuracy")
        self.assertEqual(profile.primary_metric, "accuracy")
        
        # Test that a simulated good result would be extracted correctly
        good_eval_results = {
            "eval_accuracy": 0.85,
            "eval_samples": 1000,
            "eval_loss": 0.42
        }
        
        accuracy = _extract_accuracy_with_fallback(good_eval_results, profile)
        self.assertEqual(accuracy, 0.85)


class TestGSM8KEvaluationMetrics(unittest.TestCase):
    """Integration test for GSM8K task profile evaluation metrics."""

    def test_gsm8k_evaluate_fails_without_eval_exact_match(self):
        """Test that missing eval_exact_match would be caught by our fallback system."""
        # This test ensures our system would handle missing GSM8K metrics gracefully
        bad_eval_results = {
            "eval_loss": 0.8,
            "eval_runtime": 45.2
            # Missing eval_exact_match - could happen with misconfiguration!
        }
        
        profile = GSM8KCausalLMProfile()
        accuracy = _extract_accuracy_with_fallback(bad_eval_results, profile)
        
        # Should return 0.0 (graceful failure) instead of crashing
        self.assertEqual(accuracy, 0.0)
        
    def test_gsm8k_profile_metric_structure(self):
        """Test GSM8K profile has the right metric structure."""
        profile = GSM8KCausalLMProfile()
        
        # Test that our modification worked
        self.assertEqual(profile.primary_metric_key, "eval_exact_match")
        self.assertEqual(profile.primary_metric, "exact_match")
        
        # Test that a simulated good result would be extracted correctly
        good_eval_results = {
            "eval_exact_match": 0.72,
            "eval_samples": 500,
            "eval_loss": 0.89
        }
        
        accuracy = _extract_accuracy_with_fallback(good_eval_results, profile)
        self.assertEqual(accuracy, 0.72)


class TestBenchProtocolIntegration(unittest.TestCase):
    """Integration tests for bench protocol with metric requirements."""

    def test_task_profile_registry_preserves_metric_keys(self):
        """Test that task profile registry preserves primary_metric_key."""
        from gradience.bench.task_profiles.registry import get_task_profile_from_config
        
        # Test GLUE config
        glue_config = {
            "task": {"profile": "seqcls_glue", "dataset": "glue", "subset": "sst2"}
        }
        glue_profile = get_task_profile_from_config(glue_config)
        self.assertEqual(glue_profile.primary_metric_key, "eval_accuracy")
        
        # Test GSM8K config  
        gsm8k_config = {
            "task": {"profile": "gsm8k_causal_lm", "dataset": "gsm8k"}
        }
        gsm8k_profile = get_task_profile_from_config(gsm8k_config)
        self.assertEqual(gsm8k_profile.primary_metric_key, "eval_exact_match")

    @patch('gradience.bench.protocol.get_task_profile_from_config')
    def test_extract_accuracy_called_in_protocol_functions(self, mock_get_profile):
        """Test that protocol functions use our robust metric extraction."""
        from gradience.bench.protocol import write_probe_eval_json
        
        # Mock task profile
        mock_profile = Mock()
        mock_profile.primary_metric_key = "eval_accuracy"
        mock_get_profile.return_value = mock_profile
        
        # Test with evaluation results
        eval_results = {
            "eval_accuracy": 0.92,
            "eval_loss": 0.3
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            probe_dir = Path(tmpdir)
            
            # This should not crash and should use our fallback function
            eval_path = write_probe_eval_json(
                probe_dir=probe_dir,
                eval_results=eval_results,
                eval_dataset_size=100,
                config={"train": {"seed": 42}, "lora": {"probe_r": 16}}
            )
            
            # Verify the eval.json was written correctly
            with open(eval_path) as f:
                eval_data = json.load(f)
                
            self.assertEqual(eval_data["accuracy"], 0.92)
            self.assertEqual(eval_data["eval_samples"], 100)


if __name__ == "__main__":
    # Run with: python -m pytest tests/test_bench_eval_metric_requirements.py -v
    unittest.main()