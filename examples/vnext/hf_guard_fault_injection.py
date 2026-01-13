#!/usr/bin/env python
"""
Fault injection example for LoRA Guard with HuggingFace Trainer.

This script demonstrates:
1. Using HF Trainer with a tiny dataset (CPU-only)
2. Enabling Guard in GradienceCallback
3. Injecting NaN loss at a specific step
4. Verifying that Guard triggers rollback

Usage:
    python examples/vnext/hf_guard_fault_injection.py
"""

import json
import tempfile
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments, PreTrainedModel, PretrainedConfig

# Add gradience to path if running as script
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from gradience.vnext.integrations.hf import GradienceCallback, GradienceCallbackConfig


# ============================================================================
# Tiny Model and Dataset for Testing
# ============================================================================

class TinyConfig(PretrainedConfig):
    """Minimal config for our tiny model."""
    model_type = "tiny_lora"
    
    def __init__(
        self,
        vocab_size=100,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers


class TinyLoRAModel(PreTrainedModel):
    """
    Tiny model with LoRA-like parameters for testing Guard.
    
    Includes fault injection capability to trigger NaN at specific steps.
    """
    config_class = TinyConfig
    
    def __init__(self, config):
        super().__init__(config)
        
        # Base model parameters (frozen in real LoRA)
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            nn.Linear(config.hidden_size, config.hidden_size)
            for _ in range(config.num_hidden_layers)
        ])
        self.output = nn.Linear(config.hidden_size, config.vocab_size)
        
        # LoRA adapter parameters (these will be saved by Guard)
        self.lora_A = nn.Parameter(torch.randn(config.hidden_size, 8) * 0.01)
        self.lora_B = nn.Parameter(torch.randn(8, config.hidden_size) * 0.01)
        
        # Fault injection flag (set by FaultInjectingTrainer)
        self._inject_nan_now: bool = False
    
    def forward(self, input_ids, labels=None):
        # Simple forward pass
        x = self.embedding(input_ids)
        
        # Apply layers with LoRA adapter
        for layer in self.layers:
            # Base transformation
            base_out = layer(x)
            # LoRA adaptation: x @ lora_A @ lora_B
            lora_out = x @ self.lora_A @ self.lora_B
            x = base_out + lora_out
        
        logits = self.output(x)
        
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits.view(-1, self.config.vocab_size), labels.view(-1))
            
            # FAULT INJECTION: Override loss with NaN if configured
            # We inject based on a global step counter instead of forward calls
            # since the Guard works on HF's global_step
            if hasattr(self, '_inject_nan_now') and self._inject_nan_now:
                print(f"\n🔥 INJECTING NaN loss (Guard should detect this)")
                # Create NaN tensor that maintains gradients
                loss = loss * float('nan')
                self._inject_nan_now = False  # Only inject once
        
        return {"loss": loss, "logits": logits} if loss is not None else logits


class FaultInjectingTrainer(Trainer):
    """Trainer that can inject faults at specific steps."""
    
    def __init__(self, *args, fault_step=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.fault_step = fault_step
    
    def training_step(self, model, inputs, num_items_in_batch=None):
        """Override to inject fault at specific step."""
        # Check if we should inject fault this step
        if self.fault_step is not None and self.state.global_step == self.fault_step:
            model._inject_nan_now = True
        
        return super().training_step(model, inputs, num_items_in_batch)


class TinyDataset(Dataset):
    """Minimal dataset for testing."""
    
    def __init__(self, size=100, seq_length=16, vocab_size=100):
        self.size = size
        self.seq_length = seq_length
        self.vocab_size = vocab_size
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        # Random data for simplicity
        input_ids = torch.randint(0, self.vocab_size, (self.seq_length,))
        labels = torch.randint(0, self.vocab_size, (self.seq_length,))
        return {"input_ids": input_ids, "labels": labels}


# ============================================================================
# Main Test
# ============================================================================

def run_fault_injection_test():
    """Run the fault injection test and verify Guard behavior."""
    
    print("=" * 60)
    print("LoRA Guard Fault Injection Test")
    print("=" * 60)
    
    # Create temporary directory for outputs
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir) / "guard_test"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Setup model and dataset
        print("\n1. Setting up model and dataset...")
        config = TinyConfig()
        model = TinyLoRAModel(config)
        train_dataset = TinyDataset(size=100)
        
        # Configure fault injection
        INJECT_AT_STEP = 15  # Inject NaN at step 15
        
        # 2. Configure Guard
        print("\n2. Configuring LoRA Guard...")
        guard_config = GradienceCallbackConfig(
            output_dir=str(output_dir),
            filename="run.jsonl",
            enable_guard=True,
            guard_snapshot_every=5,  # Snapshot every 5 steps
            guard_ring_size=5,
            guard_grad_threshold=100.0,
            guard_cooldown_steps=10,
            guard_max_rollbacks=3,
            guard_window_steps=50,
            guard_steps_back=1,  # Rollback to most recent snapshot
            guard_prune_newer_on_rollback=True,
        )
        
        callback = GradienceCallback(guard_config)
        
        # 3. Setup HF Trainer
        print("\n3. Setting up HuggingFace Trainer...")
        training_args = TrainingArguments(
            output_dir=str(output_dir),
            max_steps=30,  # Train for 30 steps
            per_device_train_batch_size=4,
            gradient_accumulation_steps=1,
            warmup_steps=0,
            logging_steps=1,  # Log every step for testing
            save_strategy="no",  # Don't save checkpoints
            report_to=[],  # No external reporting
            no_cuda=True,  # CPU only
            fp16=False,
            bf16=False,
        )
        
        trainer = FaultInjectingTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            callbacks=[callback],
            fault_step=INJECT_AT_STEP,
        )
        
        # 4. Run training (will inject NaN and trigger rollback)
        print(f"\n4. Running training (will inject NaN at step {INJECT_AT_STEP})...")
        print("-" * 40)
        
        try:
            trainer.train()
        except Exception as e:
            print(f"Training stopped with: {e}")
        
        # 5. Verify telemetry
        print("\n" + "-" * 40)
        print("\n5. Verifying Guard telemetry...")
        
        telemetry_file = output_dir / "run.jsonl"
        assert telemetry_file.exists(), "Telemetry file not created"
        
        # Parse telemetry
        events = []
        with open(telemetry_file) as f:
            for line in f:
                events.append(json.loads(line))
        
        # Check for canonical events
        guard_init = False
        guard_triggered = False
        guard_rollback = False
        rollback_step = None
        snapshot_events = []
        
        for event in events:
            event_type = event.get("event")
            
            # Check canonical alerts
            if event_type == "alert":
                code = event.get("code")
                if code == "GUARD_INIT":
                    guard_init = True
                    print(f"  ✓ GUARD_INIT: {event.get('message')}")
                elif code == "GUARD_SNAPSHOT":
                    print(f"  ✓ GUARD_SNAPSHOT: {event.get('message')}")
                elif code == "GUARD_TRIGGERED":
                    guard_triggered = True
                    print(f"  ✓ GUARD_TRIGGERED: {event.get('message')}")
                elif code == "GUARD_ROLLBACK":
                    guard_rollback = True
                    rollback_step = event.get("metadata", {}).get("restored_step")
                    print(f"  ✓ GUARD_ROLLBACK: {event.get('message')}")
                elif code in ["GUARD_ABORT", "GUARD_ABORT_NO_SNAPSHOT"]:
                    print(f"  ⚠ {code}: {event.get('message')}")
            
            # Check canonical metrics
            elif event_type == "metrics" and event.get("kind") == "guard":
                metrics = event.get("metrics", {})
                action = metrics.get("action")
                if action == "snapshot":
                    step = event.get("step")
                    snapshot_events.append(step)
                    print(f"  • Snapshot metrics at step {step}")
                elif action == "rollback":
                    print(f"  • Rollback metrics: restored_step={metrics.get('restored_step')}, "
                          f"n_rollbacks={metrics.get('n_rollbacks')}")
                elif action == "init":
                    print(f"  • Init metrics: ring_size={metrics.get('ring_size')}")
        
        # 6. Assertions
        print("\n6. Test Assertions:")
        
        assert guard_init, "Guard should have initialized"
        print("  ✓ Guard initialized (GUARD_INIT)")
        
        assert guard_triggered, "Guard should have detected the trigger"
        print("  ✓ Guard detected trigger (GUARD_TRIGGERED)")
        
        assert guard_rollback, "Guard should have performed rollback"
        print("  ✓ Guard performed rollback (GUARD_ROLLBACK)")
        
        assert rollback_step is not None and rollback_step <= INJECT_AT_STEP, \
            f"Should rollback to step at or before injection (got {rollback_step})"
        print(f"  ✓ Rolled back to step {rollback_step} (at or before fault injection)")
        
        assert len(snapshot_events) >= 2, "Should have at least 2 snapshots before rollback"
        print(f"  ✓ Created {len(snapshot_events)} snapshots before rollback")
        
        # Check that training continued after rollback (optional)
        train_steps = [e.get("step") for e in events if e.get("event") == "train_step"]
        if train_steps:
            max_step = max(train_steps)
            if max_step > INJECT_AT_STEP:
                print(f"  ✓ Training continued after rollback (reached step {max_step})")
        
        print("\n" + "=" * 60)
        print("✅ FAULT INJECTION TEST PASSED!")
        print("=" * 60)
        
        return True


if __name__ == "__main__":
    # Run the test
    success = run_fault_injection_test()
    
    if not success:
        print("\n❌ Test failed!")
        sys.exit(1)
    
    print("\n✨ All tests passed! Guard successfully handled fault injection.")