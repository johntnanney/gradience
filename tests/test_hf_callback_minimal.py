"""
Minimal unit test for HF callback Guard wiring (CI-friendly).

This test verifies basic Guard functionality without heavy dependencies:
- Constructs callback with enable_guard=True
- Calls on_train_begin and on_log with NaN loss
- Asserts callback doesn't crash and emits alert in JSONL
"""

import unittest
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from transformers import TrainingArguments, TrainerState, TrainerControl
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

# Only run tests if both torch and transformers are available
@unittest.skipUnless(HAS_TORCH and HAS_TRANSFORMERS, "Requires torch and transformers")
class TestHFCallbackGuardWiring(unittest.TestCase):
    """Minimal test to verify Guard wiring doesn't crash and produces alerts."""
    
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.output_dir = Path(self.tmpdir.name)
    
    def tearDown(self):
        self.tmpdir.cleanup()
    
    def test_guard_wiring_basic_functionality(self):
        """Test that Guard can be enabled and responds to NaN loss without crashing."""
        from gradience.vnext.integrations.hf import GradienceCallback, GradienceCallbackConfig
        
        # Create tiny mock model with LoRA-like parameters
        class TinyMockModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.lora_A = nn.Parameter(torch.randn(4, 8))
                self.lora_B = nn.Parameter(torch.randn(8, 4))
                self.base_weight = nn.Parameter(torch.randn(8, 8))
        
        # Configure callback with Guard enabled
        config = GradienceCallbackConfig(
            output_dir=str(self.output_dir),
            filename="test.jsonl",
            enable_guard=True,
            guard_snapshot_every=5,
            guard_ring_size=3,
            guard_cooldown_steps=0,  # No cooldown for testing
        )
        
        callback = GradienceCallback(config)
        
        # Create mock HF objects
        mock_args = Mock(spec=TrainingArguments)
        mock_args.output_dir = str(self.output_dir)
        # Set all required attributes to avoid AttributeError
        for attr in [
            'seed', 'per_device_train_batch_size', 'gradient_accumulation_steps',
            'learning_rate', 'weight_decay', 'adam_beta1', 'adam_beta2',
            'adam_epsilon', 'max_steps', 'num_train_epochs', 'optim',
            'fp16', 'bf16'
        ]:
            setattr(mock_args, attr, None)
        
        mock_state = Mock(spec=TrainerState)
        mock_state.global_step = 0
        
        mock_control = Mock(spec=TrainerControl)
        mock_control.should_training_stop = False
        
        model = TinyMockModel()
        
        # Test 1: Initialization should succeed
        try:
            callback.on_train_begin(mock_args, mock_state, mock_control, model=model)
            self.assertIsNotNone(callback.guard, "Guard should be initialized")
        except Exception as e:
            self.fail(f"Guard initialization failed: {e}")
        
        # Test 2: Normal operation should work
        mock_state.global_step = 1
        try:
            callback.on_log(
                mock_args, mock_state, mock_control,
                logs={"loss": 1.0, "grad_norm": 5.0},
                model=model
            )
        except Exception as e:
            self.fail(f"Normal logging failed: {e}")
        
        # Test 3: NaN loss should trigger Guard without crashing
        mock_state.global_step = 5  # This will trigger a snapshot
        try:
            callback.on_log(
                mock_args, mock_state, mock_control,
                logs={"loss": 1.0},  # Normal loss for snapshot
                model=model
            )
        except Exception as e:
            self.fail(f"Snapshot creation failed: {e}")
        
        # Test 4: NaN loss should trigger rollback
        mock_state.global_step = 10
        try:
            callback.on_log(
                mock_args, mock_state, mock_control,
                logs={"loss": float('nan')},  # This should trigger Guard
                model=model
            )
        except Exception as e:
            self.fail(f"NaN handling failed: {e}")
        
        # Test 5: Cleanup should work
        try:
            callback.on_train_end(mock_args, mock_state, mock_control)
        except Exception as e:
            self.fail(f"Cleanup failed: {e}")
        
        # Verify telemetry file was created and contains Guard events
        telemetry_file = self.output_dir / "test.jsonl"
        self.assertTrue(telemetry_file.exists(), "Telemetry file should be created")
        
        # Parse telemetry and look for Guard-related events
        guard_alerts = []
        guard_metrics = []
        
        with open(telemetry_file) as f:
            for line in f:
                if line.strip():
                    event = json.loads(line)
                    
                    # Check for Guard alerts
                    if event.get("event") == "alert":
                        code = event.get("code", "")
                        if code.startswith("GUARD_"):
                            guard_alerts.append(code)
                    
                    # Check for Guard metrics
                    elif (event.get("event") == "metrics" and 
                          event.get("kind") == "guard"):
                        action = event.get("metrics", {}).get("action")
                        if action:
                            guard_metrics.append(action)
        
        # Assertions: Verify Guard detected the NaN and responded appropriately
        self.assertGreater(len(guard_alerts), 0, 
                          "Should have at least one Guard alert (TRIGGERED, ROLLBACK, or ABORT)")
        
        self.assertGreater(len(guard_metrics), 0,
                          "Should have at least one Guard metric (init, snapshot, or rollback)")
        
        # Check for specific canonical expected events
        canonical_codes = {'GUARD_INIT', 'GUARD_TRIGGERED', 'GUARD_ROLLBACK', 'GUARD_SNAPSHOT'}
        actual_codes = set(guard_alerts)
        has_expected = canonical_codes.intersection(actual_codes)
        self.assertTrue(has_expected, 
                       f"Should have canonical Guard alerts, got: {actual_codes}")
        
        print(f"✓ Guard alerts: {guard_alerts}")
        print(f"✓ Guard metrics: {guard_metrics}")
    
    def test_guard_disabled_by_default(self):
        """Verify Guard is disabled by default and doesn't interfere."""
        from gradience.vnext.integrations.hf import GradienceCallback, GradienceCallbackConfig
        
        # Default config should have Guard disabled
        config = GradienceCallbackConfig(output_dir=str(self.output_dir))
        callback = GradienceCallback(config)
        
        self.assertFalse(config.enable_guard, "Guard should be disabled by default")
        self.assertIsNone(callback.guard, "Guard should not be initialized when disabled")
        
        # Should work normally without Guard
        mock_args = Mock()
        mock_args.output_dir = str(self.output_dir)
        for attr in ['seed', 'per_device_train_batch_size', 'gradient_accumulation_steps',
                     'learning_rate', 'weight_decay', 'adam_beta1', 'adam_beta2',
                     'adam_epsilon', 'max_steps', 'num_train_epochs', 'optim',
                     'fp16', 'bf16']:
            setattr(mock_args, attr, None)
        
        mock_state = Mock()
        mock_state.global_step = 1
        mock_control = Mock()
        
        # Should not crash even with NaN
        try:
            callback.on_train_begin(mock_args, mock_state, mock_control, model=Mock())
            callback.on_log(mock_args, mock_state, mock_control, 
                          logs={"loss": float('nan')}, model=Mock())
            callback.on_train_end(mock_args, mock_state, mock_control)
        except Exception as e:
            self.fail(f"Callback should work normally when Guard is disabled: {e}")
    
    def test_guard_with_mock_model_parameters(self):
        """Test Guard works with models that have named_parameters method."""
        from gradience.vnext.integrations.hf import GradienceCallback, GradienceCallbackConfig
        
        config = GradienceCallbackConfig(
            output_dir=str(self.output_dir),
            enable_guard=True,
        )
        callback = GradienceCallback(config)
        
        # Create a mock model with named_parameters method
        mock_model = Mock()
        mock_model.named_parameters.return_value = [
            ("lora_A", Mock(detach=Mock(return_value=Mock(to=Mock(return_value=Mock(clone=Mock(return_value=torch.randn(4, 8))))))))
        ]
        
        mock_args = Mock()
        mock_args.output_dir = str(self.output_dir)
        for attr in ['seed', 'per_device_train_batch_size', 'gradient_accumulation_steps',
                     'learning_rate', 'weight_decay', 'adam_beta1', 'adam_beta2',
                     'adam_epsilon', 'max_steps', 'num_train_epochs', 'optim',
                     'fp16', 'bf16']:
            setattr(mock_args, attr, None)
        
        mock_state = Mock()
        mock_state.global_step = 0
        mock_control = Mock()
        
        # Should work with proper mock model
        try:
            callback.on_train_begin(mock_args, mock_state, mock_control, model=mock_model)
            self.assertIsNotNone(callback.guard, "Guard should be initialized with mock model")
        except Exception as e:
            self.fail(f"Guard should work with mock model: {e}")
    
    def test_anti_thrash_abort_behavior(self):
        """Test anti-thrash: max_rollbacks=1 prevents Guard loops forever."""
        from gradience.vnext.integrations.hf import GradienceCallback, GradienceCallbackConfig
        
        # Create tiny mock model with LoRA-like parameters
        class TinyMockModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.lora_A = nn.Parameter(torch.randn(4, 8))
        
        # Configure anti-thrash settings
        config = GradienceCallbackConfig(
            output_dir=str(self.output_dir),
            filename="anti_thrash_test.jsonl",
            enable_guard=True,
            guard_max_rollbacks=1,  # Only allow 1 rollback
            guard_window_steps=99999,  # Large window
            guard_cooldown_steps=0,  # No cooldown
            guard_snapshot_every=5,
        )
        
        callback = GradienceCallback(config)
        
        # Create mock HF objects
        mock_args = Mock()
        mock_args.output_dir = str(self.output_dir)
        for attr in ['seed', 'per_device_train_batch_size', 'gradient_accumulation_steps',
                     'learning_rate', 'weight_decay', 'adam_beta1', 'adam_beta2',
                     'adam_epsilon', 'max_steps', 'num_train_epochs', 'optim',
                     'fp16', 'bf16']:
            setattr(mock_args, attr, None)
        
        mock_state = Mock()
        mock_control = Mock()
        mock_control.should_training_stop = False
        
        model = TinyMockModel()
        
        # Initialize guard
        callback.on_train_begin(mock_args, mock_state, mock_control, model=model)
        
        # Take a snapshot
        mock_state.global_step = 5
        callback.on_log(mock_args, mock_state, mock_control, 
                       logs={"loss": 1.0}, model=model)
        
        # First NaN trigger - should trigger rollback
        mock_state.global_step = 10
        callback.on_log(mock_args, mock_state, mock_control, 
                       logs={"loss": float('nan')}, model=model)
        
        # Verify first rollback succeeded and training continues
        self.assertFalse(mock_control.should_training_stop, 
                        "Training should continue after first rollback")
        
        # Second NaN trigger - should trigger GUARD_ABORT due to max_rollbacks=1
        mock_state.global_step = 15
        mock_control.should_training_stop = False  # Reset for second test
        callback.on_log(mock_args, mock_state, mock_control, 
                       logs={"loss": float('nan')}, model=model)
        
        # Verify anti-thrash behavior
        self.assertTrue(mock_control.should_training_stop, 
                       "Training should stop after exceeding max_rollbacks")
        
        callback.on_train_end(mock_args, mock_state, mock_control)
        
        # Verify telemetry contains GUARD_ABORT
        telemetry_file = self.output_dir / "anti_thrash_test.jsonl"
        self.assertTrue(telemetry_file.exists(), "Anti-thrash telemetry file should exist")
        
        guard_rollback_count = 0
        guard_abort_found = False
        
        with open(telemetry_file) as f:
            for line in f:
                if line.strip():
                    event = json.loads(line)
                    if event.get("event") == "alert":
                        code = event.get("code")
                        if code == "GUARD_ROLLBACK":
                            guard_rollback_count += 1
                        elif code == "GUARD_ABORT":
                            guard_abort_found = True
                            print(f"✓ Anti-thrash GUARD_ABORT: {event.get('message')}")
        
        # Assertions for anti-thrash behavior
        self.assertEqual(guard_rollback_count, 1, 
                        "Should have exactly 1 GUARD_ROLLBACK before abort")
        self.assertTrue(guard_abort_found, 
                       "Should have GUARD_ABORT when max_rollbacks exceeded")
        
        print(f"✓ Anti-thrash test passed: 1 rollback, then abort with training stop")
    
    def test_enable_guard_false_does_nothing(self):
        """Test safety invariant: enable_guard=False must not interfere with training."""
        from gradience.vnext.integrations.hf import GradienceCallback, GradienceCallbackConfig
        
        # Create tiny mock model with LoRA-like parameters
        class TinyMockModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.lora_A = nn.Parameter(torch.randn(4, 8))
                self.lora_B = nn.Parameter(torch.randn(8, 4))
        
        # Explicitly disable Guard
        config = GradienceCallbackConfig(
            output_dir=str(self.output_dir),
            filename="guard_disabled_test.jsonl",
            enable_guard=False,  # Explicitly disabled
            # Set other guard settings to non-defaults to ensure they're ignored
            guard_max_rollbacks=999,
            guard_cooldown_steps=999,
            guard_snapshot_every=1,
        )
        
        callback = GradienceCallback(config)
        
        # Verify Guard is not initialized
        self.assertFalse(config.enable_guard, "Guard should be explicitly disabled")
        self.assertIsNone(callback.guard, "Guard should not be initialized when disabled")
        
        # Create mock HF objects
        mock_args = Mock()
        mock_args.output_dir = str(self.output_dir)
        for attr in ['seed', 'per_device_train_batch_size', 'gradient_accumulation_steps',
                     'learning_rate', 'weight_decay', 'adam_beta1', 'adam_beta2',
                     'adam_epsilon', 'max_steps', 'num_train_epochs', 'optim',
                     'fp16', 'bf16']:
            setattr(mock_args, attr, None)
        
        mock_state = Mock()
        mock_control = Mock()
        mock_control.should_training_stop = False
        
        model = TinyMockModel()
        
        # Full training lifecycle with problematic inputs should work fine
        try:
            # 1. Initialize
            callback.on_train_begin(mock_args, mock_state, mock_control, model=model)
            self.assertIsNone(callback.guard, "Guard should still be None after initialization")
            
            # 2. Normal training steps
            for step in [1, 2, 3, 4, 5]:
                mock_state.global_step = step
                callback.on_log(mock_args, mock_state, mock_control, 
                               logs={"loss": 1.0 + step * 0.1, "grad_norm": 5.0}, 
                               model=model)
            
            # 3. Problematic inputs that would trigger Guard if enabled
            mock_state.global_step = 10
            callback.on_log(mock_args, mock_state, mock_control, 
                           logs={"loss": float('nan'), "grad_norm": float('inf')}, 
                           model=model)
            
            mock_state.global_step = 15
            callback.on_log(mock_args, mock_state, mock_control, 
                           logs={"loss": float('inf'), "grad_norm": 1e9}, 
                           model=model)
            
            # 4. More normal training
            mock_state.global_step = 20
            callback.on_log(mock_args, mock_state, mock_control, 
                           logs={"loss": 2.0, "grad_norm": 3.0}, 
                           model=model)
            
            # 5. Evaluation
            callback.on_evaluate(mock_args, mock_state, mock_control, 
                               metrics={"eval_loss": 1.5, "eval_accuracy": 0.8})
            
            # 6. Training end
            callback.on_train_end(mock_args, mock_state, mock_control)
            
        except Exception as e:
            self.fail(f"Disabled Guard should not interfere with any training operations: {e}")
        
        # Verify no Guard interference
        self.assertFalse(mock_control.should_training_stop, 
                        "Training should never be stopped when Guard is disabled")
        
        # Verify telemetry contains NO Guard events
        telemetry_file = self.output_dir / "guard_disabled_test.jsonl"
        self.assertTrue(telemetry_file.exists(), "Telemetry file should still be created")
        
        guard_events = []
        train_events = []
        eval_events = []
        
        with open(telemetry_file) as f:
            for line in f:
                if line.strip():
                    event = json.loads(line)
                    event_type = event.get("event")
                    
                    if event_type == "alert" and event.get("code", "").startswith("GUARD_"):
                        guard_events.append(event.get("code"))
                    elif event_type == "metrics" and event.get("kind") == "guard":
                        guard_events.append(f"metrics:{event.get('metrics', {}).get('action')}")
                    elif event_type == "train_step":
                        train_events.append(event)
                    elif event_type == "eval":
                        eval_events.append(event)
        
        # Safety invariant assertions
        self.assertEqual(len(guard_events), 0, 
                        f"Should have NO Guard events when disabled, found: {guard_events}")
        self.assertGreater(len(train_events), 0, 
                          "Should still log normal training events")
        self.assertGreater(len(eval_events), 0, 
                          "Should still log evaluation events")
        
        print(f"✓ Guard disabled safety invariant verified:")
        print(f"  • 0 Guard events (expected: 0)")
        print(f"  • {len(train_events)} train events (expected: >0)")
        print(f"  • {len(eval_events)} eval events (expected: >0)")
        print(f"  • No training interference with NaN/Inf inputs")


if __name__ == "__main__":
    unittest.main()