"""
Specific failure case tests for metric requirements.

This test file contains minimal, fast tests that explicitly check for the 
failure conditions mentioned in the requirements:
- GLUE runs that produce no eval_accuracy
- GSM8K runs that produce no eval_exact_match

These tests are CPU-only and use minimal fixtures to ensure they run fast in CI.
"""

import unittest
from unittest.mock import Mock, patch

from gradience.bench.task_profiles.seqcls_glue import GLUESequenceClassificationProfile
from gradience.bench.task_profiles.gsm8k_causal_lm import GSM8KCausalLMProfile
from gradience.bench.protocol import _extract_accuracy_with_fallback


class TestMetricRequirementFailureCases(unittest.TestCase):
    """Tests that verify our fix handles the specific failure cases."""
    
    def test_glue_missing_eval_accuracy_does_not_crash(self):
        """
        FAILURE CASE: GLUE runs that produce no eval_accuracy.
        
        This test ensures that if a GLUE evaluation somehow doesn't produce 
        eval_accuracy (the original issue), our system handles it gracefully
        instead of crashing during verdict computation.
        """
        profile = GLUESequenceClassificationProfile()
        
        # Simulate the original problem: eval results without eval_accuracy
        broken_eval_results = {
            "eval_loss": 0.42,
            "eval_runtime": 1.23,
            "eval_samples_per_second": 123.45,
            # eval_accuracy is MISSING - this was the original bug!
        }
        
        # Our fallback system should handle this gracefully
        accuracy = _extract_accuracy_with_fallback(broken_eval_results, profile)
        
        # Should not crash and should return 0.0 (indicating failure to find metric)
        self.assertEqual(accuracy, 0.0)
        
        # Test that it would fall back to legacy 'accuracy' key if present
        broken_eval_results_with_fallback = {
            "eval_loss": 0.42,
            "accuracy": 0.78,  # Legacy key that should be found
        }
        
        accuracy = _extract_accuracy_with_fallback(broken_eval_results_with_fallback, profile)
        self.assertEqual(accuracy, 0.78)  # Should find the fallback key
    
    def test_gsm8k_missing_eval_exact_match_does_not_crash(self):
        """
        FAILURE CASE: GSM8K runs that produce no eval_exact_match.
        
        This test ensures that if a GSM8K evaluation doesn't produce 
        eval_exact_match (could happen with misconfiguration), our system 
        handles it gracefully.
        """
        profile = GSM8KCausalLMProfile()
        
        # Simulate problem: GSM8K eval results without eval_exact_match
        broken_eval_results = {
            "eval_loss": 0.89,
            "eval_runtime": 45.67,
            "eval_steps_per_second": 2.1,
            # eval_exact_match is MISSING - potential configuration issue!
        }
        
        # Our fallback system should handle this gracefully
        accuracy = _extract_accuracy_with_fallback(broken_eval_results, profile)
        
        # Should not crash and should return 0.0
        self.assertEqual(accuracy, 0.0)
        
        # Test that it would fall back to legacy 'exact_match' key if present
        broken_eval_results_with_fallback = {
            "eval_loss": 0.89,
            "exact_match": 0.65,  # Legacy key that should be found
        }
        
        accuracy = _extract_accuracy_with_fallback(broken_eval_results_with_fallback, profile)
        self.assertEqual(accuracy, 0.65)  # Should find the fallback key
    
    def test_unknown_task_profile_with_missing_metrics(self):
        """
        EDGE CASE: Unknown task profile or None profile with missing metrics.
        
        This tests the scenario where we have neither a task profile nor
        any of the standard metric keys.
        """
        # Test with None profile (task profile not available)
        broken_eval_results = {
            "some_other_metric": 0.99,
            "eval_loss": 0.5,
            "weird_custom_field": "value"
            # No accuracy metrics at all!
        }
        
        accuracy = _extract_accuracy_with_fallback(broken_eval_results, None)
        
        # Should gracefully return 0.0 instead of crashing
        self.assertEqual(accuracy, 0.0)
    
    def test_fallback_priority_order_is_correct(self):
        """
        Test that the fallback priority order works as expected:
        1. task_profile.primary_metric_key (if available)
        2. eval_accuracy
        3. eval_exact_match  
        4. accuracy
        5. exact_match
        """
        profile = GLUESequenceClassificationProfile()
        
        # Test case where all metrics are present - should prefer primary
        all_metrics_present = {
            "eval_accuracy": 0.85,      # Primary for GLUE - should win
            "eval_exact_match": 0.72,
            "accuracy": 0.90,
            "exact_match": 0.88
        }
        
        result = _extract_accuracy_with_fallback(all_metrics_present, profile)
        self.assertEqual(result, 0.85)  # Should use primary metric
        
        # Test fallback order when primary is missing
        missing_primary = {
            "eval_exact_match": 0.72,   # Second in fallback list
            "accuracy": 0.90,           # Third in fallback list  
            "exact_match": 0.88         # Fourth in fallback list
        }
        
        result = _extract_accuracy_with_fallback(missing_primary, profile)
        self.assertEqual(result, 0.72)  # Should use eval_exact_match (first available)
        
        # Test deeper fallback
        only_legacy = {
            "accuracy": 0.90,           # Third in fallback list
            "exact_match": 0.88         # Fourth in fallback list
        }
        
        result = _extract_accuracy_with_fallback(only_legacy, profile)
        self.assertEqual(result, 0.90)  # Should use accuracy (first available)
    
    @patch('gradience.bench.protocol.get_task_profile_from_config')
    def test_bench_protocol_functions_are_robust_to_missing_metrics(self, mock_get_profile):
        """
        Integration test: Verify bench protocol functions handle missing metrics.
        
        This simulates the original crash scenario but with our fix in place.
        """
        from gradience.bench.protocol import write_probe_eval_json
        import tempfile
        import json
        from pathlib import Path
        
        # Mock GLUE profile
        mock_profile = Mock()
        mock_profile.primary_metric_key = "eval_accuracy"
        mock_get_profile.return_value = mock_profile
        
        # Simulate the original crash scenario: missing eval_accuracy
        broken_eval_results = {
            "eval_loss": 0.42,
            "eval_runtime": 1.23,
            "eval_samples_per_second": 123.45
            # eval_accuracy is MISSING!
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            probe_dir = Path(tmpdir)
            
            # Before our fix, this would crash in compute_verdicts 
            # Now it should gracefully handle the missing metric
            eval_path = write_probe_eval_json(
                probe_dir=probe_dir,
                eval_results=broken_eval_results,
                eval_dataset_size=100,
                config={"train": {"seed": 42}, "lora": {"probe_r": 16}}
            )
            
            # Verify the file was created and contains 0.0 accuracy (graceful failure)
            with open(eval_path) as f:
                eval_data = json.load(f)
                
            self.assertEqual(eval_data["accuracy"], 0.0)  # Graceful failure instead of crash
            self.assertEqual(eval_data["eval_samples"], 100)
            
    def test_specific_reproduce_original_bug_scenario(self):
        """
        Reproduce the exact original bug scenario and verify it's fixed.
        
        Original issue: bench verdict computation crashed when looking for 
        eval_accuracy in GLUE results but finding a different metric name.
        """
        # Simulate exact original issue: GLUE profile expecting eval_accuracy
        # but evaluation results containing different metric names
        profile = GLUESequenceClassificationProfile()
        
        # This could happen if evaluation used a different metric computation
        confusing_eval_results = {
            "test_accuracy": 0.85,      # Wrong prefix
            "validation_accuracy": 0.82, # Different split name
            "acc": 0.88,                # Abbreviated form
            "eval_loss": 0.35
            # No eval_accuracy or standard fallbacks!
        }
        
        # Before our fix: this would cause KeyError in compute_verdicts
        # After our fix: should gracefully return 0.0
        result = _extract_accuracy_with_fallback(confusing_eval_results, profile)
        self.assertEqual(result, 0.0)
        
        # Verify that adding a recognized fallback key works
        confusing_eval_results["accuracy"] = 0.87  # Add fallback key
        result = _extract_accuracy_with_fallback(confusing_eval_results, profile)
        self.assertEqual(result, 0.87)  # Should find the fallback


if __name__ == "__main__":
    # Run with: python -m pytest tests/test_metric_requirements_failure_cases.py -v
    unittest.main()