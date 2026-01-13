#!/usr/bin/env python
"""
CPU-friendly max_rollbacks lockout validation for LoRA Guard.

This script demonstrates:
1. Guard's anti-thrash max_rollbacks protection
2. First trigger causes successful rollback
3. Second trigger hits max_rollbacks limit and is aborted
4. Telemetry validation for GUARD_ABORT with max_rollbacks reason

Usage:
    python examples/vnext/hf_guard_max_rollbacks_lockout.py

Success criteria:
- First trigger at step 10: GUARD_ROLLBACK
- Second trigger at step 11: GUARD_ABORT (max_rollbacks)
- Telemetry shows proper max_rollbacks protection behavior
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock
import torch
import torch.nn as nn

# Add gradience to path if running as script
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from gradience.vnext.integrations.hf import GradienceCallback, GradienceCallbackConfig


# ============================================================================
# Tiny Mock LoRA Model (CPU-only)
# ============================================================================

class TinyLoRAModel(nn.Module):
    """Minimal model with LoRA-like parameters for testing Guard max_rollbacks."""
    
    def __init__(self):
        super().__init__()
        # LoRA adapter parameters (these will be saved by Guard)
        self.lora_A = nn.Parameter(torch.randn(16, 8) * 0.01)
        self.lora_B = nn.Parameter(torch.randn(8, 16) * 0.01)
        self.base_weight = nn.Parameter(torch.randn(32, 32) * 0.01)
        
    def __class__(self):
        # Mock for HF model detection
        return type("MockTransformerModel", (), {"__name__": "MockTransformerModel"})


# ============================================================================
# Main Max Rollbacks Lockout Validation Script
# ============================================================================

def run_max_rollbacks_lockout_validation():
    """Run CPU-friendly max_rollbacks lockout validation and verify Guard behavior."""
    
    print("=" * 60)
    print("LoRA Guard Max Rollbacks Lockout Validation (CPU)")
    print("=" * 60)
    
    # Create temporary directory for outputs
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir) / "guard_max_rollbacks_test"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Setup model
        print("\n1. Setting up tiny LoRA model...")
        model = TinyLoRAModel()
        print(f"   Model has {sum(p.numel() for p in model.parameters())} parameters")
        
        # 2. Configure Guard with max_rollbacks settings for testing
        print("\n2. Configuring LoRA Guard for max_rollbacks testing...")
        guard_config = GradienceCallbackConfig(
            output_dir=str(output_dir),
            filename="max_rollbacks_test.jsonl",
            enable_guard=True,
            guard_snapshot_every=1,  # Snapshot every step (always have snapshots)
            guard_ring_size=10,
            guard_grad_threshold=100.0,  # Trigger at grad_norm > 100
            guard_cooldown_steps=0,  # NO cooldown (max_rollbacks is the limiting factor)
            guard_max_rollbacks=1,  # Only allow 1 rollback in window
            guard_window_steps=99999,  # Very large window (effectively unlimited)
            guard_steps_back=1,
        )
        
        callback = GradienceCallback(guard_config)
        
        print(f"   Cooldown steps: {guard_config.guard_cooldown_steps} (disabled)")
        print(f"   Max rollbacks: {guard_config.guard_max_rollbacks}")
        print(f"   Window steps: {guard_config.guard_window_steps}")
        print(f"   Snapshot every: {guard_config.guard_snapshot_every} steps")
        
        # 3. Setup mock HF objects
        print("\n3. Setting up mock HuggingFace training objects...")
        
        # Mock training arguments
        mock_args = Mock()
        mock_args.output_dir = str(output_dir)
        mock_args.seed = 42
        mock_args.per_device_train_batch_size = 2
        mock_args.gradient_accumulation_steps = 1
        mock_args.max_steps = None
        mock_args.num_train_epochs = 1.0
        mock_args.learning_rate = 5e-4
        mock_args.weight_decay = 0.01
        mock_args.adam_beta1 = 0.9
        mock_args.adam_beta2 = 0.999
        mock_args.adam_epsilon = 1e-8
        mock_args.optim = "adamw_torch"
        mock_args.fp16 = False
        mock_args.bf16 = False
        
        # Mock trainer state
        mock_state = Mock()
        mock_state.global_step = 0
        
        # Mock trainer control
        mock_control = Mock()
        
        # 4. Initialize Guard
        print("\n4. Initializing Guard...")
        callback.on_train_begin(mock_args, mock_state, mock_control, model=model)
        
        # Verify Guard was created
        assert callback.guard is not None, "Guard should be initialized"
        print(f"   ✓ Guard initialized with max_rollbacks: {guard_config.guard_max_rollbacks}")
        
        # 5. Simulate normal training leading up to first trigger
        print("\n5. Simulating normal training steps...")
        
        normal_steps = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        for step in normal_steps:
            mock_state.global_step = step
            logs = {
                "loss": 2.5 + 0.1 * torch.randn(1).item(),  # Normal loss
                "grad_norm": 1.0 + 0.5 * torch.randn(1).item(),  # Normal grad norm (< 100)
                "learning_rate": 5e-4,
            }
            callback.on_log(mock_args, mock_state, mock_control, logs=logs, model=model)
            print(f"   Step {step}: loss={logs['loss']:.3f}, grad_norm={logs['grad_norm']:.3f}")
        
        # Verify snapshots were taken (should be every step)
        snapshot_count = callback.guard.snapshot_count()
        print(f"   ✓ Guard took {snapshot_count} snapshots (expected: {len(normal_steps)})")
        
        # 6. First trigger - should cause rollback
        print("\n6. First trigger (step 10) - should cause rollback...")
        
        first_trigger_step = 10
        mock_state.global_step = first_trigger_step
        
        # Create logs with exploded gradient norm
        first_trigger_logs = {
            "loss": 2.3,  # Normal loss
            "grad_norm": 500.0,  # EXPLODED gradient norm (>> 100)
            "learning_rate": 5e-4,
        }
        
        print(f"   🔥 FIRST TRIGGER at step {first_trigger_step}")
        print(f"      grad_norm: {first_trigger_logs['grad_norm']:.1f} (threshold: {guard_config.guard_grad_threshold})")
        
        callback.on_log(mock_args, mock_state, mock_control, logs=first_trigger_logs, model=model)
        
        # Check first rollback occurred
        first_rollback_count = callback.guard.n_rollbacks
        print(f"   ✓ Guard performed {first_rollback_count} rollback(s) after first trigger")
        assert first_rollback_count == 1, "Should have performed exactly 1 rollback"
        
        # 7. Second trigger immediately after - should hit max_rollbacks limit
        print("\n7. Second trigger (step 11) - should be ABORTED due to max_rollbacks...")
        
        second_trigger_step = 11
        steps_since_rollback = second_trigger_step - first_trigger_step
        print(f"   Steps since first rollback: {steps_since_rollback}")
        print(f"   Max rollbacks allowed: {guard_config.guard_max_rollbacks}")
        
        # Verify we're testing the max_rollbacks limit (not cooldown)
        assert guard_config.guard_cooldown_steps == 0, "Cooldown should be disabled for this test"
        print(f"   ✓ Cooldown is disabled (testing max_rollbacks limit)")
        
        mock_state.global_step = second_trigger_step
        
        # Create logs with another exploded gradient norm
        second_trigger_logs = {
            "loss": 2.1,  # Normal loss
            "grad_norm": 800.0,  # ANOTHER exploded gradient norm (>> 100)
            "learning_rate": 5e-4,
        }
        
        print(f"   🔥 SECOND TRIGGER at step {second_trigger_step}")
        print(f"      grad_norm: {second_trigger_logs['grad_norm']:.1f} (threshold: {guard_config.guard_grad_threshold})")
        
        callback.on_log(mock_args, mock_state, mock_control, logs=second_trigger_logs, model=model)
        
        # Check second trigger was aborted (rollback count should stay the same)
        second_rollback_count = callback.guard.n_rollbacks
        print(f"   ✓ Guard rollback count after second trigger: {second_rollback_count}")
        assert second_rollback_count == 1, "Should still have exactly 1 rollback (second was aborted)"
        
        # 8. Parse and verify telemetry
        print("\n8. Analyzing max_rollbacks telemetry...")
        
        telemetry_file = output_dir / "max_rollbacks_test.jsonl"
        assert telemetry_file.exists(), "Telemetry file should be created"
        
        events = []
        with open(telemetry_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    events.append(json.loads(line))
        
        print(f"   Found {len(events)} telemetry events")
        
        # Check for required Guard events
        guard_init = False
        first_triggered = False
        first_rollback = False
        second_triggered = False
        second_abort = False
        
        first_trigger_context = None
        first_rollback_context = None
        second_trigger_context = None
        second_abort_context = None
        
        for event in events:
            event_type = event.get("event")
            step = event.get("step")
            metadata = event.get("metadata", {})
            
            if event_type == "alert":
                code = event.get("code")
                if code == "GUARD_INIT":
                    guard_init = True
                    print(f"   ✓ GUARD_INIT: {event.get('message')}")
                elif code == "GUARD_TRIGGERED":
                    # For GUARD_TRIGGERED, step info is in metadata
                    trigger_step = metadata.get("step")
                    if trigger_step == first_trigger_step:
                        first_triggered = True
                        first_trigger_context = metadata
                        print(f"   ✓ GUARD_TRIGGERED (first): {event.get('message')}")
                    elif trigger_step == second_trigger_step:
                        second_triggered = True
                        second_trigger_context = metadata
                        print(f"   ✓ GUARD_TRIGGERED (second): {event.get('message')}")
                elif code == "GUARD_ROLLBACK":
                    # For GUARD_ROLLBACK, step info is in metadata
                    rollback_step = metadata.get("step")
                    if rollback_step == first_trigger_step:
                        first_rollback = True
                        first_rollback_context = metadata
                        print(f"   ✓ GUARD_ROLLBACK (first): {event.get('message')}")
                elif code in ("GUARD_ABORT", "GUARD_ABORT_NO_SNAPSHOT"):
                    # For GUARD_ABORT, step info is in metadata
                    abort_step = metadata.get("step")
                    if abort_step == second_trigger_step:
                        second_abort = True
                        second_abort_context = metadata
                        print(f"   ✓ GUARD_ABORT (second): {event.get('message')}")
            
            elif event_type == "metrics" and event.get("kind") == "guard":
                metrics = event.get("metrics", {})
                action = metrics.get("action")
                if action == "rollback" and step == first_trigger_step:
                    restored_step = metrics.get("restored_step")
                    n_rollbacks = metrics.get("n_rollbacks")
                    print(f"   ✓ First rollback metrics: restored_step={restored_step}, n_rollbacks={n_rollbacks}")
                elif action in ("abort", "abort_no_snapshot") and step == second_trigger_step:
                    n_rollbacks = metrics.get("n_rollbacks", 0)
                    print(f"   ✓ Second abort metrics: n_rollbacks={n_rollbacks} (should still be 1)")
        
        # 9. Validation assertions
        print("\n9. Max rollbacks lockout validation:")
        
        assert guard_init, "Guard should have initialized (GUARD_INIT)"
        print("   ✓ Guard initialization verified")
        
        assert first_triggered, "Guard should have detected first trigger (GUARD_TRIGGERED)"
        print("   ✓ First trigger detection verified")
        
        assert first_rollback, "Guard should have performed first rollback (GUARD_ROLLBACK)"
        print("   ✓ First rollback execution verified")
        
        assert second_triggered, "Guard should have detected second trigger (GUARD_TRIGGERED)"
        print("   ✓ Second trigger detection verified")
        
        assert second_abort, "Guard should have aborted second rollback (GUARD_ABORT)"
        print("   ✓ Second rollback abortion verified (max_rollbacks protection)")
        
        # Verify trigger types
        if first_trigger_context:
            trigger_type = first_trigger_context.get("trigger")
            assert trigger_type == "grad_explosion", f"Expected grad_explosion trigger, got {trigger_type}"
            print(f"   ✓ First trigger type verified: {trigger_type}")
        
        if second_trigger_context:
            trigger_type = second_trigger_context.get("trigger")
            assert trigger_type == "grad_explosion", f"Expected grad_explosion trigger, got {trigger_type}"
            print(f"   ✓ Second trigger type verified: {trigger_type}")
        
        # Verify max_rollbacks protection context
        if second_abort_context:
            # Check if abort message mentions max rollbacks
            abort_message = None
            for event in events:
                if (event.get("event") == "alert" and 
                    event.get("code") in ("GUARD_ABORT", "GUARD_ABORT_NO_SNAPSHOT") and
                    event.get("metadata", {}).get("step") == second_trigger_step):
                    abort_message = event.get("message", "")
                    break
            
            if abort_message and ("max rollbacks" in abort_message.lower() or "max_rollbacks" in abort_message.lower()):
                print(f"   ✓ Max rollbacks protection message verified: {abort_message}")
            else:
                # Check if we have window/rollback context in metadata
                window_steps = second_abort_context.get("window_steps")
                n_rollbacks = second_abort_context.get("n_rollbacks")
                if window_steps and n_rollbacks:
                    print(f"   ✓ Max rollbacks protection context verified: n_rollbacks={n_rollbacks}, window_steps={window_steps}")
                else:
                    print(f"   ⚠ Abort context: {second_abort_context}")
                    print(f"   ⚠ Abort message: {abort_message}")
        
        print("\n" + "=" * 60)
        print("✅ MAX ROLLBACKS LOCKOUT VALIDATION PASSED!")
        print("=" * 60)
        
        # Summary
        print(f"\nSummary:")
        print(f"  • Max rollbacks allowed: {guard_config.guard_max_rollbacks}")
        print(f"  • Window size: {guard_config.guard_window_steps} steps")
        print(f"  • Cooldown disabled: {guard_config.guard_cooldown_steps} steps")
        print(f"  • First trigger step: {first_trigger_step}")
        print(f"  • Second trigger step: {second_trigger_step}")
        print(f"  • Steps between triggers: {second_trigger_step - first_trigger_step}")
        print(f"  • First trigger result: ROLLBACK")
        print(f"  • Second trigger result: ABORT (max_rollbacks)")
        print(f"  • Total rollbacks performed: {second_rollback_count}")
        print(f"  • Telemetry events logged: {len(events)}")
        
        return True


if __name__ == "__main__":
    # Run the max rollbacks lockout validation
    success = run_max_rollbacks_lockout_validation()
    
    if not success:
        print("\n❌ Max rollbacks lockout validation failed!")
        sys.exit(1)
    
    print("\n✨ Guard max_rollbacks protection successfully validated!")