# LoRA Guard (Experimental)

**⚠️ EXPERIMENTAL PROTOTYPE - NOT RELEASED UNTIL CONFIDENT**

## Critical Warnings

🚨 **Guard is EXPERIMENTAL** - Not recommended for production use  
🚨 **Guard CAN STOP TRAINING** - Will halt training if protection limits exceeded  
🚨 **Guard CAN ROLL BACK WEIGHTS** - Adapter parameters will be reverted to earlier states  
🚨 **Guard DOES NOT FIX ROOT CAUSES** - Cannot fix data bugs, bad objectives, or hyperparameters  
🚨 **ALWAYS VALIDATE WITH EVAL** - Guard actions don't guarantee improved model quality  

## What is Guard?

LoRA Guard is an experimental rollback system that protects LoRA training from gradient explosions and NaN/Inf corruption by automatically taking snapshots and rolling back to stable states when triggers are detected.

**Think of it as an emergency brake, not a steering wheel.**

## Quick Start

### Enable Guard

```python
from gradience.vnext.integrations.hf import GradienceCallback, GradienceCallbackConfig

config = GradienceCallbackConfig(
    output_dir="./training_output",
    enable_guard=True,  # Enable Guard
    guard_snapshot_every=10,  # Snapshot every 10 steps
    guard_grad_threshold=100.0,  # Rollback if grad_norm > 100
)

callback = GradienceCallback(config)

# Add to HuggingFace Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    callbacks=[callback],
)
```

## What Guard Does

### Automatic Snapshots
- Takes periodic snapshots of LoRA adapter weights (not full model)
- Stores on CPU to avoid GPU memory pressure
- Maintains ring buffer of recent snapshots for memory efficiency

### Trigger Detection
Guard rolls back when it detects:
- **Gradient explosion**: `grad_norm > guard_grad_threshold`
- **NaN/Inf gradients**: Any non-finite gradient values
- **Loss corruption**: NaN or Inf loss values

### Rollback Process
1. **Detect trigger** during training step
2. **Restore weights** from most recent snapshot
3. **Continue training** from restored state
4. **Log telemetry** for monitoring

## What Guard CANNOT Do

Guard is NOT a magic fix. It specifically CANNOT:

- **Fix data bugs**: Corrupted or mislabeled data remains corrupted
- **Fix bad objectives**: Wrong loss functions stay wrong
- **Fix hyperparameters**: Bad learning rates remain problematic
- **Improve model quality**: Rolling back doesn't mean better performance
- **Protect full models**: Only LoRA adapter weights are protected
- **Recover from OOM**: Memory errors are beyond Guard's scope
- **Work with non-LoRA**: Only designed for LoRA/PEFT fine-tuning

**Remember**: Guard is a safety net for numerical instability, not a solution for training design problems.

## Anti-Thrash Protection

Guard includes built-in anti-thrash mechanisms to prevent infinite rollback loops:

### Default Settings
```python
guard_max_rollbacks=3,        # Max rollbacks in window
guard_window_steps=50,        # Rolling window size
guard_cooldown_steps=10,      # Steps to wait after rollback
```

### Behavior
- **Cooldown period**: After rollback, Guard waits before next rollback
- **Rolling window**: Tracks rollback frequency over recent steps
- **Max rollbacks**: Aborts if too many rollbacks in window
- **Abort protection**: Training continues even if Guard can't help

## Configuration Options

```python
config = GradienceCallbackConfig(
    # Guard Enable/Disable
    enable_guard=False,  # Guard disabled by default
    
    # Snapshot Settings
    guard_snapshot_every=10,     # Snapshot frequency (steps)
    guard_ring_size=5,           # Max snapshots in ring buffer
    
    # Trigger Thresholds
    guard_grad_threshold=100.0,  # Gradient norm threshold
    
    # Anti-Thrash Protection
    guard_max_rollbacks=3,       # Max rollbacks in window
    guard_window_steps=50,       # Rolling window size
    guard_cooldown_steps=10,     # Cooldown after rollback
    guard_steps_back=1,          # How many snapshots back to restore
    
    # Cleanup Behavior
    guard_prune_newer_on_rollback=True,  # Remove newer snapshots after rollback
)
```

## Telemetry and Monitoring

Guard emits canonical telemetry events with consistent severity levels:

### Alert Severity Pattern

Guard uses a clear escalation pattern for alert severities:

| Severity | Meaning | Alert Codes |
|----------|---------|-------------|
| **INFO** | Normal operation | `GUARD_INIT`, `GUARD_SNAPSHOT`, `GUARD_TRIGGERED` |
| **WARNING** | Intervention taken | `GUARD_ROLLBACK` |
| **ERROR** | Protection failed | `GUARD_ABORT`, `GUARD_ABORT_NO_SNAPSHOT` |

### Alert Codes
- `GUARD_INIT` (INFO): Guard initialized successfully
- `GUARD_SNAPSHOT` (INFO): Snapshot taken
- `GUARD_TRIGGERED` (INFO): Trigger detected (not yet acted on)
- `GUARD_ROLLBACK` (WARNING): ⚠️ Weights rolled back to earlier state
- `GUARD_ABORT` (ERROR): ❌ Cannot rollback (cooldown/limits)
- `GUARD_ABORT_NO_SNAPSHOT` (ERROR): ❌ No snapshot available

### Metrics
Guard metrics use `kind="guard"` with standardized fields:
```json
{
  "event": "metrics",
  "kind": "guard", 
  "metrics": {
    "action": "rollback",
    "ring_size": 5,
    "snapshot_count": 3,
    "memory_mb": 45.2
  }
}
```

## ⚠️ Critical Warnings

### Verify with Evaluation
**Always validate that Guard improves your training outcomes:**

```python
# Monitor these metrics to verify Guard helps:
# 1. Final model quality (eval loss, downstream metrics)
# 2. Training stability (fewer divergent runs)
# 3. Time to convergence (total training time)

# If Guard doesn't improve these metrics, disable it:
config.enable_guard = False
```

### When to Use Guard
- **High-value training runs** where stability is critical
- **Unstable hyperparameters** that occasionally cause explosions  
- **Large-scale training** where restarts are expensive
- **Research experiments** exploring aggressive learning rates

### When NOT to Use Guard
- **Stable training** that never diverges anyway
- **Short experiments** where restarts are cheap
- **Non-LoRA training** (Guard only protects LoRA adapters)
- **Memory-constrained** environments (snapshots use additional memory)

## Examples

See `examples/vnext/hf_guard_fault_injection.py` for a complete working example that demonstrates:
- Guard initialization and configuration
- Fault injection to trigger rollback
- Telemetry verification
- Expected behavior validation

## Implementation Notes

- **CPU snapshots**: Weights copied to CPU to avoid GPU memory pressure
- **LoRA-specific**: Only backs up parameters with "lora" in the name
- **HuggingFace integration**: Designed for HF Trainer workflow
- **Memory efficient**: Ring buffer prevents unbounded memory growth
- **Thread-safe**: Snapshot operations are atomic

## Troubleshooting

### Guard Not Triggering
- Check `guard_grad_threshold` - may be too high
- Verify LoRA parameters exist (names contain "lora")
- Ensure `enable_guard=True` in config

### Excessive Rollbacks
- Increase `guard_cooldown_steps` 
- Decrease `guard_max_rollbacks`
- Check if hyperparameters are fundamentally unstable

### Memory Usage
- Reduce `guard_ring_size`
- Increase `guard_snapshot_every` 
- Monitor `memory_mb` in telemetry metrics

Remember: Guard is a safety net, not a solution to underlying training problems. Always investigate and fix root causes of training instability.