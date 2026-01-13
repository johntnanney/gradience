"""
Test for Gradience HuggingFace Callback integration.

This test is gated behind GRADIENCE_TEST_HF to keep core CI fast.
Run with: GRADIENCE_TEST_HF=1 python -m unittest tests.test_hf_callback
"""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

# Check if transformers is available
try:
    import transformers
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

# Skip all tests if transformers not available or GRADIENCE_TEST_HF not set
SKIP_HF_TESTS = not HAS_TRANSFORMERS or not os.environ.get("GRADIENCE_TEST_HF")


@unittest.skipIf(SKIP_HF_TESTS, "transformers not installed or GRADIENCE_TEST_HF not set")
class TestGradienceCallback(unittest.TestCase):
    """Test Gradience HF callback without requiring full transformers setup."""

    def setUp(self):
        """Set up test fixtures."""
        self.tmpdir = tempfile.TemporaryDirectory()
        self.output_dir = Path(self.tmpdir.name)

    def tearDown(self):
        """Clean up test fixtures."""
        self.tmpdir.cleanup()

    def test_import_succeeds_with_transformers(self):
        """Test that callback can be imported when transformers is available."""
        from gradience.vnext.integrations.hf import GradienceCallback

    def test_callback_lifecycle_minimal(self):
        """Test minimal callback lifecycle: instantiate -> train_begin -> log -> evaluate -> train_end."""
        from gradience.vnext.integrations.hf import GradienceCallback, GradienceCallbackConfig

        # Create callback with explicit output dir
        config = GradienceCallbackConfig(
            output_dir=str(self.output_dir),
            filename="test_run.jsonl"
        )
        callback = GradienceCallback(config)

        # Mock HF objects (minimal attributes needed)
        mock_args = Mock()
        mock_args.output_dir = str(self.output_dir)
        mock_args.seed = 42
        mock_args.per_device_train_batch_size = 2
        mock_args.gradient_accumulation_steps = 1
        mock_args.max_steps = 10
        mock_args.num_train_epochs = 1.0
        mock_args.learning_rate = 5e-4
        mock_args.weight_decay = 0.01
        mock_args.adam_beta1 = 0.9
        mock_args.adam_beta2 = 0.999
        mock_args.adam_epsilon = 1e-8
        mock_args.optim = "adamw_torch"
        mock_args.fp16 = False
        mock_args.bf16 = False

        mock_state = Mock()
        mock_state.global_step = 5

        mock_control = Mock()

        # Mock model with basic attributes
        mock_model = Mock()
        mock_model.__class__.__name__ = "TestModel"
        # Add minimal config for _best_effort_model_name
        mock_config = Mock()
        mock_config.name_or_path = "test-model"
        mock_model.config = mock_config

        # 1. Test on_train_begin
        callback.on_train_begin(mock_args, mock_state, mock_control, model=mock_model)
        self.assertIsNotNone(callback.writer)
        self.assertIsNotNone(callback._run_id)

        # Check output file was created
        output_file = self.output_dir / "test_run.jsonl" 
        self.assertTrue(output_file.exists())

        # 2. Test on_log
        test_logs = {
            "loss": 1.234,
            "learning_rate": 5e-4,
            "grad_norm": 0.123,
            "epoch": 0.5
        }
        callback.on_log(mock_args, mock_state, mock_control, logs=test_logs)

        # 3. Test on_evaluate  
        test_metrics = {
            "eval_loss": 1.456,
            "eval_accuracy": 0.85,
            "eval_f1": 0.82
        }
        callback.on_evaluate(mock_args, mock_state, mock_control, metrics=test_metrics)

        # 4. Test on_train_end
        callback.on_train_end(mock_args, mock_state, mock_control)
        self.assertIsNone(callback.writer)  # Should be closed

        # 5. Verify file content and validate with TelemetryReader
        self.assertTrue(output_file.exists())
        self.assertTrue(output_file.stat().st_size > 0)

        # Read and validate with TelemetryReader
        from gradience.vnext.telemetry_reader import TelemetryReader
        
        reader = TelemetryReader(output_file, strict_schema=True)
        
        # Test validation passes
        issues = reader.validate()
        self.assertEqual(issues, [], f"Telemetry validation failed: {issues}")
        
        # Test summarize works without crashing
        summary = reader.summarize()
        self.assertIsNotNone(summary)
        
        # Test config roundtrip
        config_snap = reader.latest_config()
        self.assertIsNotNone(config_snap)
        self.assertEqual(config_snap.model_name, "test-model")
        self.assertEqual(config_snap.training.seed, 42)
        self.assertEqual(config_snap.training.batch_size, 2)
        self.assertEqual(config_snap.optimizer.lr, 5e-4)

    def test_callback_handles_none_logs_gracefully(self):
        """Test callback handles None logs without crashing."""
        from gradience.vnext.integrations.hf import GradienceCallback, GradienceCallbackConfig

        config = GradienceCallbackConfig(output_dir=str(self.output_dir))
        callback = GradienceCallback(config)

        # Initialize with minimal mocks (need all numeric attributes to be actual numbers)
        mock_args = Mock()
        mock_args.output_dir = str(self.output_dir)
        mock_args.seed = 42
        mock_args.per_device_train_batch_size = 2
        mock_args.gradient_accumulation_steps = 1
        mock_args.max_steps = 10
        mock_args.num_train_epochs = 1.0
        mock_args.learning_rate = 5e-4
        mock_args.weight_decay = 0.01
        mock_args.adam_beta1 = None  # Optional
        mock_args.adam_beta2 = None  # Optional
        mock_args.adam_epsilon = None  # Optional
        mock_args.optim = "adamw_torch"
        mock_args.fp16 = False
        mock_args.bf16 = False
        
        mock_state = Mock() 
        mock_state.global_step = 1
        mock_control = Mock()
        mock_model = Mock()
        mock_model.__class__.__name__ = "TestModel"

        callback.on_train_begin(mock_args, mock_state, mock_control, model=mock_model)

        # These should not crash with None inputs
        callback.on_log(mock_args, mock_state, mock_control, logs=None)
        callback.on_evaluate(mock_args, mock_state, mock_control, metrics=None)

        callback.on_train_end(mock_args, mock_state, mock_control)

    def test_callback_with_minimal_config(self):
        """Test callback works with default GradienceCallbackConfig."""
        from gradience.vnext.integrations.hf import GradienceCallback

        # Test default config (no params)
        callback = GradienceCallback()
        self.assertIsNotNone(callback.config)
        self.assertEqual(callback.config.filename, "run.jsonl")


if __name__ == "__main__":
    unittest.main()