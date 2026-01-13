#!/usr/bin/env python
"""
CPU-friendly grad explosion validation for LoRA Guard.

This script demonstrates:
1. Fast Guard testing without real training
2. Fake grad_norm injection to trigger Guard
3. Validation that Guard detects gradient explosions
4. Telemetry verification for grad explosion trigger

Usage:
    python examples/vnext/hf_guard_fault_injection_grad_explosion.py

Success criteria:
- Guard emits GUARD_TRIGGERED with trigger grad_explosion
- Guard emits GUARD_ROLLBACK  
- Telemetry includes context: step, restored_step, trigger, n_rollbacks
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
    """Minimal model with LoRA-like parameters for testing Guard."""
    
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
# Main Validation Script
# ============================================================================

def run_grad_explosion_validation():
    """Run CPU-friendly grad explosion validation and verify Guard behavior."""
    
    print("=" * 60)
    print("LoRA Guard Grad Explosion Validation (CPU)")
    print("=" * 60)
    
    # Create temporary directory for outputs
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir) / "guard_grad_test"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Setup model
        print("\n1. Setting up tiny LoRA model...")
        model = TinyLoRAModel()
        print(f"   Model has {sum(p.numel() for p in model.parameters())} parameters")
        
        # 2. Configure Guard with aggressive settings for testing
        print("\n2. Configuring LoRA Guard for grad explosion testing...")
        guard_config = GradienceCallbackConfig(
            output_dir=str(output_dir),
            filename="grad_explosion_test.jsonl",
            enable_guard=True,
            guard_snapshot_every=5,  # Snapshot every 5 steps
            guard_ring_size=3,
            guard_grad_threshold=100.0,  # Trigger at grad_norm > 100
            guard_cooldown_steps=0,  # No cooldown for testing
            guard_max_rollbacks=3,
            guard_window_steps=50,
            guard_steps_back=1,
        )
        
        callback = GradienceCallback(guard_config)
        
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
        print(f"   ✓ Guard initialized with ring size {callback.guard.ring_size}")
        
        # 5. Simulate training with normal gradient norms
        print("\n5. Simulating normal training steps...")
        
        normal_steps = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        for step in normal_steps:
            mock_state.global_step = step
            logs = {
                "loss": 2.5 + 0.1 * torch.randn(1).item(),  # Normal loss
                "grad_norm": 1.0 + 0.5 * torch.randn(1).item(),  # Normal grad norm (< 100)
                "learning_rate": 5e-4,
            }
            callback.on_log(mock_args, mock_state, mock_control, logs=logs, model=model)
            print(f"   Step {step}: loss={logs['loss']:.3f}, grad_norm={logs['grad_norm']:.3f}")
        
        # Verify snapshots were taken
        snapshot_count = callback.guard.snapshot_count()
        print(f"   ✓ Guard took {snapshot_count} snapshots during normal training")
        
        # 6. Inject gradient explosion
        print("\n6. Injecting gradient explosion...")
        
        explosion_step = 15
        mock_state.global_step = explosion_step
        
        # Create logs with exploded gradient norm
        explosion_logs = {
            "loss": 2.3,  # Normal loss
            "grad_norm": 1500.0,  # EXPLODED gradient norm (>> 100)
            "learning_rate": 5e-4,
        }
        
        print(f"   💥 INJECTING grad explosion at step {explosion_step}")
        print(f"      grad_norm: {explosion_logs['grad_norm']:.1f} (threshold: {guard_config.guard_grad_threshold})")
        
        callback.on_log(mock_args, mock_state, mock_control, logs=explosion_logs, model=model)
        
        # 7. Verify Guard detected and handled explosion
        print("\n7. Verifying Guard response...")
        
        # Check Guard state
        rollback_count = callback.guard.n_rollbacks
        print(f"   ✓ Guard performed {rollback_count} rollback(s)")
        assert rollback_count > 0, "Guard should have performed at least one rollback"
        
        # 8. Parse and verify telemetry
        print("\n8. Analyzing telemetry...")
        
        telemetry_file = output_dir / "grad_explosion_test.jsonl"
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
        guard_triggered = False
        guard_rollback = False
        trigger_context = None
        rollback_context = None
        
        for event in events:
            event_type = event.get("event")
            
            if event_type == "alert":
                code = event.get("code")
                if code == "GUARD_INIT":
                    guard_init = True
                    print(f"   ✓ GUARD_INIT: {event.get('message')}")
                elif code == "GUARD_TRIGGERED":
                    guard_triggered = True
                    trigger_context = event.get("metadata", {})
                    print(f"   ✓ GUARD_TRIGGERED: {event.get('message')}")
                elif code == "GUARD_ROLLBACK":
                    guard_rollback = True
                    rollback_context = event.get("metadata", {})
                    print(f"   ✓ GUARD_ROLLBACK: {event.get('message')}")
            
            elif event_type == "metrics" and event.get("kind") == "guard":
                metrics = event.get("metrics", {})
                action = metrics.get("action")
                if action == "rollback":
                    restored_step = metrics.get("restored_step")
                    n_rollbacks = metrics.get("n_rollbacks")
                    print(f"   ✓ Rollback metrics: restored_step={restored_step}, n_rollbacks={n_rollbacks}")
        
        # 9. Validation assertions
        print("\n9. Final validation:")
        
        assert guard_init, "Guard should have initialized (GUARD_INIT)"
        print("   ✓ Guard initialization verified")
        
        assert guard_triggered, "Guard should have detected trigger (GUARD_TRIGGERED)"
        print("   ✓ Guard trigger detection verified")
        
        assert guard_rollback, "Guard should have performed rollback (GUARD_ROLLBACK)"
        print("   ✓ Guard rollback execution verified")
        
        # Verify trigger was grad_explosion
        if trigger_context:
            trigger_type = trigger_context.get("trigger")
            assert trigger_type == "grad_explosion", f"Expected grad_explosion trigger, got {trigger_type}"
            print(f"   ✓ Trigger type verified: {trigger_type}")
        
        # Verify rollback context
        if rollback_context:
            restored_step = rollback_context.get("restored_step")
            assert restored_step is not None and restored_step <= explosion_step, \
                f"Should restore to step <= {explosion_step}, got {restored_step}"
            print(f"   ✓ Rollback restored to step {restored_step}")
        
        print("\n" + "=" * 60)
        print("✅ GRAD EXPLOSION VALIDATION PASSED!")
        print("=" * 60)
        
        # Summary
        print(f"\nSummary:")
        print(f"  • Normal steps processed: {len(normal_steps)}")
        print(f"  • Snapshots created: {snapshot_count}")
        print(f"  • Explosion injected at step: {explosion_step}")
        print(f"  • Guard rollbacks performed: {rollback_count}")
        print(f"  • Telemetry events logged: {len(events)}")
        
        return True


if __name__ == "__main__":
    # Run the validation
    success = run_grad_explosion_validation()
    
    if not success:
        print("\n❌ Validation failed!")
        sys.exit(1)
    
    print("\n✨ Guard successfully detected and handled gradient explosion!")