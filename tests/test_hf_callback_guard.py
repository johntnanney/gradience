"""
Test HF callback with LoRAGuard integration.
"""

import unittest
import tempfile
from pathlib import Path
from unittest.mock import Mock
import torch
import torch.nn as nn

from gradience.vnext.integrations.hf import GradienceCallback, GradienceCallbackConfig


class TinyLoRAModel(nn.Module):
    """Minimal model with LoRA-like parameters."""
    def __init__(self):
        super().__init__()
        self.lora_A = nn.Parameter(torch.randn(4, 8))
        self.lora_B = nn.Parameter(torch.randn(8, 4))
        self.base_weight = nn.Parameter(torch.randn(8, 8))


class TestHFCallbackGuard(unittest.TestCase):
    
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.output_dir = Path(self.tmpdir.name)
    
    def tearDown(self):
        self.tmpdir.cleanup()
    
    def test_guard_disabled_by_default(self):
        """Test that guard is OFF by default."""
        config = GradienceCallbackConfig(output_dir=str(self.output_dir))
        callback = GradienceCallback(config)
        
        self.assertFalse(config.enable_guard)
        self.assertIsNone(callback.guard)
    
    def test_guard_initialization(self):
        """Test guard initializes when enabled."""
        config = GradienceCallbackConfig(
            output_dir=str(self.output_dir),
            enable_guard=True,
            guard_snapshot_every=10,
            guard_ring_size=3,
        )
        callback = GradienceCallback(config)
        
        # Mock HF objects
        mock_args = Mock()
        mock_args.output_dir = str(self.output_dir)
        mock_args.seed = 42
        mock_args.per_device_train_batch_size = 2
        mock_args.gradient_accumulation_steps = 1
        mock_args.learning_rate = 1e-3
        mock_args.weight_decay = 0.0
        mock_args.adam_beta1 = None
        mock_args.adam_beta2 = None
        mock_args.adam_epsilon = None
        mock_args.max_steps = None
        mock_args.num_train_epochs = 1
        mock_args.optim = "adamw"
        mock_args.fp16 = False
        mock_args.bf16 = False
        
        mock_state = Mock()
        mock_state.global_step = 0
        
        mock_control = Mock()
        
        model = TinyLoRAModel()
        
        # Initialize callback (should create guard)
        callback.on_train_begin(mock_args, mock_state, mock_control, model=model)
        
        # Check guard was created
        self.assertIsNotNone(callback.guard)
        self.assertEqual(callback.guard.ring_size, 3)
        self.assertEqual(callback._last_guard_snapshot_step, 0)
    
    def test_guard_snapshots_and_rollback(self):
        """Test guard takes snapshots and can rollback."""
        config = GradienceCallbackConfig(
            output_dir=str(self.output_dir),
            enable_guard=True,
            guard_snapshot_every=5,  # Snapshot every 5 steps
            guard_grad_threshold=10.0,  # Low threshold to trigger
            guard_cooldown_steps=0,  # No cooldown for testing
        )
        callback = GradienceCallback(config)
        
        # Setup
        mock_args = Mock()
        mock_args.output_dir = str(self.output_dir)
        mock_args.seed = 42
        mock_args.per_device_train_batch_size = 2
        mock_args.gradient_accumulation_steps = 1
        mock_args.learning_rate = 1e-3
        mock_args.weight_decay = 0.0
        mock_args.adam_beta1 = None
        mock_args.adam_beta2 = None
        mock_args.adam_epsilon = None
        mock_args.max_steps = None
        mock_args.num_train_epochs = 1
        mock_args.optim = "adamw"
        mock_args.fp16 = False
        mock_args.bf16 = False
        
        mock_state = Mock()
        mock_control = Mock()
        
        model = TinyLoRAModel()
        original_weights = model.lora_A.data.clone()
        
        # Initialize
        callback.on_train_begin(mock_args, mock_state, mock_control, model=model)
        
        # Simulate training steps with normal loss
        for step in [5, 10, 15]:
            mock_state.global_step = step
            logs = {"loss": 1.0, "grad_norm": 5.0}  # Normal values
            callback.on_log(mock_args, mock_state, mock_control, logs=logs, model=model)
        
        # Guard should have taken snapshots
        self.assertEqual(callback.guard.snapshot_count(), 4)  # Initial + 3 snapshots
        
        # Corrupt model weights
        model.lora_A.data.fill_(999.0)
        
        # Trigger rollback with high grad_norm
        mock_state.global_step = 20
        logs = {"loss": 100.0, "grad_norm": 500.0}  # Trigger values
        callback.on_log(mock_args, mock_state, mock_control, logs=logs, model=model)
        
        # Check rollback occurred
        self.assertEqual(callback.guard.n_rollbacks, 1)
        # Weights should be restored (not the corrupted value)
        self.assertFalse(torch.allclose(model.lora_A.data, torch.tensor(999.0)))
    
    def test_guard_telemetry_logging(self):
        """Test guard events are logged to telemetry."""
        config = GradienceCallbackConfig(
            output_dir=str(self.output_dir),
            enable_guard=True,
            guard_snapshot_every=10,
        )
        callback = GradienceCallback(config)
        
        # Setup
        mock_args = Mock()
        mock_args.output_dir = str(self.output_dir)
        mock_args.seed = 42
        mock_args.per_device_train_batch_size = 2
        mock_args.gradient_accumulation_steps = 1
        mock_args.learning_rate = 1e-3
        mock_args.weight_decay = 0.0
        mock_args.adam_beta1 = None
        mock_args.adam_beta2 = None
        mock_args.adam_epsilon = None
        mock_args.max_steps = None
        mock_args.num_train_epochs = 1
        mock_args.optim = "adamw"
        mock_args.fp16 = False
        mock_args.bf16 = False
        
        mock_state = Mock()
        mock_state.global_step = 0
        mock_control = Mock()
        model = TinyLoRAModel()
        
        # Initialize and run
        callback.on_train_begin(mock_args, mock_state, mock_control, model=model)
        
        # Simulate a step that triggers snapshot
        mock_state.global_step = 10
        logs = {"loss": 1.0, "grad_norm": 5.0}
        callback.on_log(mock_args, mock_state, mock_control, logs=logs, model=model)
        
        # End training
        callback.on_train_end(mock_args, mock_state, mock_control)
        
        # Check telemetry file exists
        telemetry_file = self.output_dir / "run.jsonl"
        self.assertTrue(telemetry_file.exists())
        
        # Read and verify guard events are present
        import json
        with open(telemetry_file) as f:
            lines = f.readlines()
            events = [json.loads(line) for line in lines]
            
        # Look for canonical guard events
        guard_alerts = [e for e in events if e.get("event") == "alert" and e.get("code", "").startswith("GUARD_")]
        guard_metrics = [e for e in events if e.get("event") == "metrics" and e.get("kind") == "guard"]
        
        self.assertGreater(len(guard_alerts), 0, "Should have canonical guard alerts in telemetry")
        self.assertGreater(len(guard_metrics), 0, "Should have canonical guard metrics in telemetry")
        
        # Verify at least GUARD_INIT is present
        init_alerts = [e for e in guard_alerts if e.get("code") == "GUARD_INIT"]
        self.assertGreater(len(init_alerts), 0, "Should have GUARD_INIT alert")


if __name__ == "__main__":
    unittest.main()