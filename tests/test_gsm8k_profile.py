"""
Unit tests for GSM8K task profile.

Tests the response-only masking, answer extraction, and formatting
as required by the Bench extension checklist.
"""

import pytest
from unittest.mock import Mock, MagicMock
from transformers import AutoTokenizer

from gradience.bench.task_profiles.gsm8k_causal_lm import GSM8KCausalLMProfile


class TestGSM8KProfile:
    
    def setup_method(self):
        self.profile = GSM8KCausalLMProfile()
    
    def test_answer_extraction_from_standard_format(self):
        """Test answer extraction from #### format."""
        text = "The total cost is $45. #### 45"
        assert self.profile._extract_answer(text) == "45"
        
        text = "So the answer is 1,234. #### 1,234"
        assert self.profile._extract_answer(text) == "1234"  # Comma removed
        
        text = "Final answer: 12.5 #### 12.5"
        assert self.profile._extract_answer(text) == "12.5"
    
    def test_answer_extraction_fallback(self):
        """Test answer extraction fallback to last number."""
        text = "The calculation shows we need 42 items total."
        assert self.profile._extract_answer(text) == "42"
        
        text = "First we have 10, then 20, finally 100."
        assert self.profile._extract_answer(text) == "100"
        
        text = "Cost is $1,500 for everything."
        assert self.profile._extract_answer(text) == "1500"  # Comma removed
    
    def test_answer_extraction_no_numbers(self):
        """Test answer extraction when no numbers found."""
        text = "I don't know the answer to this problem."
        assert self.profile._extract_answer(text) == ""
    
    def test_response_only_masking(self):
        """Test that response-only masking works correctly."""
        # Create a mock tokenizer that returns predictable token IDs
        tokenizer = Mock()
        tokenizer.pad_token = "<pad>"
        tokenizer.eos_token = "</s>"
        
        # Mock tokenization results
        prompt_tokens = {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]}
        completion_tokens = {"input_ids": [4, 5], "attention_mask": [1, 1]}
        
        def mock_tokenize(texts, **kwargs):
            if isinstance(texts, list) and "Question:" in texts[0]:
                # This is prompt tokenization
                return {
                    "input_ids": [prompt_tokens["input_ids"]] * len(texts),
                    "attention_mask": [prompt_tokens["attention_mask"]] * len(texts)
                }
            else:
                # This is completion tokenization
                return {
                    "input_ids": [completion_tokens["input_ids"]] * len(texts),
                    "attention_mask": [completion_tokens["attention_mask"]] * len(texts)
                }
        
        tokenizer.side_effect = mock_tokenize
        
        # Create test data
        raw_ds = {
            "train": Mock()
        }
        
        def mock_map(func, **kwargs):
            # Simulate the tokenization process
            examples = {
                "question": ["What is 2+2?"],
                "answer": ["The answer is 4. #### 4"]
            }
            result = func(examples)
            
            # Verify the structure
            assert "input_ids" in result
            assert "attention_mask" in result
            assert "labels" in result
            
            # Check one example
            input_ids = result["input_ids"][0]
            labels = result["labels"][0]
            
            # Should be: prompt_tokens + completion_tokens
            expected_input_ids = [1, 2, 3, 4, 5]
            assert input_ids == expected_input_ids
            
            # Labels should be: [-100, -100, -100, 4, 5] (prompt masked)
            expected_labels = [-100, -100, -100, 4, 5]
            assert labels == expected_labels
            
            return Mock()
        
        raw_ds["train"].map = mock_map
        
        cfg = {"task": {}, "train": {}}
        
        # This should not raise an exception and should call the map function
        result = self.profile.tokenize(raw_ds, tokenizer, cfg)
        
        # Verify structure
        assert "train" in result
    
    def test_probe_gate(self):
        """Test probe gating logic."""
        cfg = {
            "task": {
                "probe_gate": {
                    "metric": "exact_match",
                    "min_value": 0.15
                }
            }
        }
        
        # Test passing case
        eval_results = {"eval_exact_match": 0.20}
        passed, gate_info = self.profile.probe_gate(eval_results, cfg)
        
        assert passed is True
        assert gate_info["metric"] == "exact_match"
        assert gate_info["value"] == 0.20
        assert gate_info["threshold"] == 0.15
        assert gate_info["passed"] is True
        
        # Test failing case
        eval_results = {"eval_exact_match": 0.10}
        passed, gate_info = self.profile.probe_gate(eval_results, cfg)
        
        assert passed is False
        assert gate_info["value"] == 0.10
        assert gate_info["passed"] is False
    
    def test_dataset_formatting(self):
        """Test that dataset formatting produces expected structure."""
        # Mock dataset with proper length
        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=1000)  # Mock dataset has 1000 items
        mock_dataset.select.return_value = mock_dataset
        
        mock_ds = {
            "train": mock_dataset,
            "test": mock_dataset
        }
        
        # Mock load_dataset
        from unittest.mock import patch
        with patch('gradience.bench.task_profiles.gsm8k_causal_lm.load_dataset') as mock_load:
            mock_load.return_value = mock_ds
            
            cfg = {
                "train": {"train_samples": 100},
                "task": {"eval_max_samples": 50}
            }
            
            result = self.profile.load(cfg)
            
            # Should have train and validation
            assert "train" in result
            assert "validation" in result
            
            # Should call select on datasets to limit size
            mock_dataset.select.assert_called()
    
    def test_primary_metric(self):
        """Test that profile has correct primary metric."""
        assert self.profile.name == "gsm8k_causal_lm"
        assert self.profile.primary_metric == "exact_match"


if __name__ == "__main__":
    pytest.main([__file__])