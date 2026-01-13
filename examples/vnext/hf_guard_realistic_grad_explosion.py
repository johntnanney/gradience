#!/usr/bin/env python
"""
Realistic grad explosion validation for LoRA Guard with actual gradient computation.

This script demonstrates:
1. Real gradient computation on tiny LoRA model
2. Deliberate loss explosion to create huge grad_norm
3. Guard detection of real gradient explosion
4. Validation that Guard detects computed gradient explosions

Usage:
    python examples/vnext/hf_guard_realistic_grad_explosion.py

This approach:
- Uses actual forward/backward passes
- Computes real gradient norms from model parameters
- Creates realistic grad explosion via loss multiplier
- Still CPU-friendly due to tiny model size
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add gradience to path if running as script
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from gradience.vnext.integrations.hf import GradienceCallback, GradienceCallbackConfig


# ============================================================================
# Realistic Tiny LoRA Model with Forward/Backward
# ============================================================================

class RealisticTinyLoRAModel(nn.Module):
    """Small but realistic LoRA model that can do forward/backward passes."""
    
    def __init__(self, vocab_size=100, hidden_size=32, seq_length=16):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.seq_length = seq_length
        
        # Base model components (would be frozen in real LoRA)
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.linear1 = nn.Linear(hidden_size, hidden_size * 2)
        self.linear2 = nn.Linear(hidden_size * 2, hidden_size)
        self.output = nn.Linear(hidden_size, vocab_size)
        
        # LoRA adapter parameters (these will be saved by Guard)
        self.lora_A1 = nn.Parameter(torch.randn(hidden_size, 8) * 0.01)
        self.lora_B1 = nn.Parameter(torch.randn(8, hidden_size * 2) * 0.01)
        self.lora_A2 = nn.Parameter(torch.randn(hidden_size * 2, 4) * 0.01)
        self.lora_B2 = nn.Parameter(torch.randn(4, hidden_size) * 0.01)
        
    def forward(self, input_ids, labels=None):
        """Forward pass with LoRA adaptations."""
        batch_size, seq_len = input_ids.shape
        
        # Embeddings
        x = self.embedding(input_ids)  # [batch, seq, hidden]
        
        # First layer with LoRA adaptation
        base_out1 = self.linear1(x)  # [batch, seq, hidden*2]
        # LoRA: x @ A @ B
        lora_out1 = x @ self.lora_A1 @ self.lora_B1  # [batch, seq, hidden*2]
        x = F.gelu(base_out1 + lora_out1)
        
        # Second layer with LoRA adaptation  
        base_out2 = self.linear2(x)  # [batch, seq, hidden]
        # LoRA: x @ A @ B
        lora_out2 = x @ self.lora_A2 @ self.lora_B2  # [batch, seq, hidden]
        x = F.gelu(base_out2 + lora_out2)
        
        # Output projection
        logits = self.output(x)  # [batch, seq, vocab]
        
        loss = None
        if labels is not None:
            # Compute cross-entropy loss
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits.view(-1, self.vocab_size), labels.view(-1))
        
        return {"loss": loss, "logits": logits} if loss is not None else logits
    
    def __class__(self):
        # Mock for HF model detection
        return type("MockTransformerModel", (), {"__name__": "MockTransformerModel"})


def compute_grad_norm(model):
    """Compute gradient norm across all model parameters."""
    total_norm = 0.0
    param_count = 0
    
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
            param_count += 1
    
    if param_count == 0:
        return 0.0
    
    return total_norm ** 0.5


def create_synthetic_batch(vocab_size=100, batch_size=4, seq_length=16):
    """Create a synthetic training batch."""
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_length))
    labels = torch.randint(0, vocab_size, (batch_size, seq_length))
    return input_ids, labels


# ============================================================================
# Main Realistic Validation Script
# ============================================================================

def run_realistic_grad_explosion_validation():
    """Run realistic grad explosion validation with actual gradient computation."""
    
    print("=" * 70)
    print("LoRA Guard Realistic Grad Explosion Validation")
    print("=" * 70)
    
    # Create temporary directory for outputs
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir) / "realistic_guard_test"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Setup realistic tiny model
        print("\n1. Setting up realistic tiny LoRA model...")
        model = RealisticTinyLoRAModel(vocab_size=50, hidden_size=16, seq_length=8)
        
        total_params = sum(p.numel() for p in model.parameters())
        lora_params = sum(p.numel() for name, p in model.named_parameters() if 'lora' in name)
        print(f"   Total model parameters: {total_params}")
        print(f"   LoRA parameters: {lora_params}")
        
        # 2. Configure Guard with realistic settings
        print("\n2. Configuring LoRA Guard...")
        guard_config = GradienceCallbackConfig(
            output_dir=str(output_dir),
            filename="realistic_grad_test.jsonl",
            enable_guard=True,
            guard_snapshot_every=3,  # Snapshot every 3 steps
            guard_ring_size=3,
            guard_grad_threshold=50.0,  # Lower threshold for tiny model
            guard_cooldown_steps=0,  # No cooldown for testing
            guard_max_rollbacks=3,
            guard_window_steps=50,
            guard_steps_back=1,
        )
        
        callback = GradienceCallback(guard_config)
        
        # 3. Setup mock HF objects
        print("\n3. Setting up mock training infrastructure...")
        
        mock_args = Mock()
        mock_args.output_dir = str(output_dir)
        mock_args.seed = 42
        mock_args.per_device_train_batch_size = 4
        mock_args.gradient_accumulation_steps = 1
        mock_args.max_steps = None
        mock_args.num_train_epochs = 1.0
        mock_args.learning_rate = 1e-3
        mock_args.weight_decay = 0.01
        mock_args.adam_beta1 = 0.9
        mock_args.adam_beta2 = 0.999
        mock_args.adam_epsilon = 1e-8
        mock_args.optim = "adamw_torch"
        mock_args.fp16 = False
        mock_args.bf16 = False
        
        mock_state = Mock()
        mock_state.global_step = 0
        mock_control = Mock()
        
        # 4. Initialize Guard
        print("\n4. Initializing Guard...")
        callback.on_train_begin(mock_args, mock_state, mock_control, model=model)
        
        assert callback.guard is not None, "Guard should be initialized"
        print(f"   ✓ Guard initialized with threshold {guard_config.guard_grad_threshold}")
        
        # 5. Simulate normal training steps with real gradients
        print("\n5. Simulating normal training with real gradients...")
        
        normal_steps = [1, 2, 3, 4, 5, 6]
        normal_grad_norms = []
        
        for step in normal_steps:
            mock_state.global_step = step
            
            # Create synthetic batch
            input_ids, labels = create_synthetic_batch(
                vocab_size=model.vocab_size, 
                batch_size=2, 
                seq_length=model.seq_length
            )
            
            # Forward pass
            model.zero_grad()
            outputs = model(input_ids, labels=labels)
            loss = outputs["loss"]
            
            # Backward pass to compute gradients
            loss.backward()
            
            # Compute actual gradient norm
            grad_norm = compute_grad_norm(model)
            normal_grad_norms.append(grad_norm)
            
            # Log to Guard
            logs = {
                "loss": loss.item(),
                "grad_norm": grad_norm,
                "learning_rate": mock_args.learning_rate,
            }
            callback.on_log(mock_args, mock_state, mock_control, logs=logs, model=model)
            
            print(f"   Step {step}: loss={loss.item():.3f}, grad_norm={grad_norm:.3f}")
        
        # Verify snapshots were taken
        snapshot_count = callback.guard.snapshot_count()
        avg_normal_grad = sum(normal_grad_norms) / len(normal_grad_norms)
        print(f"   ✓ Guard took {snapshot_count} snapshots")
        print(f"   ✓ Average normal grad_norm: {avg_normal_grad:.3f}")
        
        # 6. Create realistic gradient explosion
        print("\n6. Creating realistic gradient explosion...")
        
        explosion_step = 10
        mock_state.global_step = explosion_step
        
        # Create batch for explosion
        input_ids, labels = create_synthetic_batch(
            vocab_size=model.vocab_size,
            batch_size=2,
            seq_length=model.seq_length
        )
        
        # Forward pass
        model.zero_grad()
        outputs = model(input_ids, labels=labels)
        loss = outputs["loss"]
        
        # EXPLOSION: Multiply loss by huge factor to create gradient explosion
        explosion_multiplier = 1e4
        exploded_loss = loss * explosion_multiplier
        
        print(f"   💥 CREATING gradient explosion at step {explosion_step}")
        print(f"      Original loss: {loss.item():.3f}")
        print(f"      Explosion multiplier: {explosion_multiplier}")
        print(f"      Exploded loss: {exploded_loss.item():.1e}")
        
        # Backward pass with exploded loss
        exploded_loss.backward()
        
        # Compute the exploded gradient norm
        exploded_grad_norm = compute_grad_norm(model)
        print(f"      Computed grad_norm: {exploded_grad_norm:.1e}")
        print(f"      Threshold: {guard_config.guard_grad_threshold}")
        
        # Verify explosion is significant
        explosion_ratio = exploded_grad_norm / avg_normal_grad
        assert exploded_grad_norm > guard_config.guard_grad_threshold, \
            f"Explosion grad_norm {exploded_grad_norm:.1e} should exceed threshold {guard_config.guard_grad_threshold}"
        print(f"      Explosion ratio: {explosion_ratio:.1e}x normal")
        
        # Log exploded gradients to Guard
        explosion_logs = {
            "loss": exploded_loss.item(),
            "grad_norm": exploded_grad_norm,
            "learning_rate": mock_args.learning_rate,
        }
        callback.on_log(mock_args, mock_state, mock_control, logs=explosion_logs, model=model)
        
        # 7. Verify Guard response
        print("\n7. Verifying Guard response...")
        
        rollback_count = callback.guard.n_rollbacks
        print(f"   ✓ Guard performed {rollback_count} rollback(s)")
        assert rollback_count > 0, "Guard should have performed at least one rollback"
        
        # 8. Analyze telemetry
        print("\n8. Analyzing realistic gradient telemetry...")
        
        telemetry_file = output_dir / "realistic_grad_test.jsonl"
        assert telemetry_file.exists(), "Telemetry file should be created"
        
        events = []
        with open(telemetry_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    events.append(json.loads(line))
        
        print(f"   Found {len(events)} telemetry events")
        
        # Check for required Guard events
        guard_triggered = False
        guard_rollback = False
        trigger_context = None
        rollback_context = None
        
        for event in events:
            event_type = event.get("event")
            
            if event_type == "alert":
                code = event.get("code")
                if code == "GUARD_TRIGGERED":
                    guard_triggered = True
                    trigger_context = event.get("metadata", {})
                    print(f"   ✓ GUARD_TRIGGERED: {event.get('message')}")
                elif code == "GUARD_ROLLBACK":
                    guard_rollback = True
                    rollback_context = event.get("metadata", {})
                    print(f"   ✓ GUARD_ROLLBACK: {event.get('message')}")
            
            elif event_type == "metrics" and event.get("kind") == "guard":
                metrics = event.get("metrics", {})
                if metrics.get("action") == "rollback":
                    restored_step = metrics.get("restored_step")
                    n_rollbacks = metrics.get("n_rollbacks")
                    print(f"   ✓ Rollback metrics: restored_step={restored_step}, n_rollbacks={n_rollbacks}")
        
        # 9. Final validation
        print("\n9. Realistic gradient validation:")
        
        assert guard_triggered, "Guard should have detected trigger (GUARD_TRIGGERED)"
        print("   ✓ Guard trigger detection verified")
        
        assert guard_rollback, "Guard should have performed rollback (GUARD_ROLLBACK)"
        print("   ✓ Guard rollback execution verified")
        
        # Verify trigger was grad_explosion with realistic context
        if trigger_context:
            trigger_type = trigger_context.get("trigger")
            assert trigger_type == "grad_explosion", f"Expected grad_explosion trigger, got {trigger_type}"
            print(f"   ✓ Trigger type verified: {trigger_type}")
            
            # Check if realistic grad_norm was logged
            grad_norm_in_context = trigger_context.get("grad_norm")
            if grad_norm_in_context:
                print(f"   ✓ Realistic grad_norm logged: {grad_norm_in_context:.1e}")
        
        print("\n" + "=" * 70)
        print("✅ REALISTIC GRAD EXPLOSION VALIDATION PASSED!")
        print("=" * 70)
        
        # Summary
        print(f"\nSummary:")
        print(f"  • Model parameters: {total_params} (LoRA: {lora_params})")
        print(f"  • Normal training steps: {len(normal_steps)}")
        print(f"  • Average normal grad_norm: {avg_normal_grad:.3e}")
        print(f"  • Explosion multiplier: {explosion_multiplier}x")
        print(f"  • Exploded grad_norm: {exploded_grad_norm:.1e}")
        print(f"  • Explosion ratio: {explosion_ratio:.1e}x")
        print(f"  • Guard rollbacks: {rollback_count}")
        print(f"  • Telemetry events: {len(events)}")
        
        return True


if __name__ == "__main__":
    # Run the realistic validation
    success = run_realistic_grad_explosion_validation()
    
    if not success:
        print("\n❌ Realistic validation failed!")
        sys.exit(1)
    
    print("\n✨ Guard successfully detected realistic gradient explosion!")