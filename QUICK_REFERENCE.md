# Gradience Quick Reference

## Installation
```bash
pip install gradience[hf]
```

## Minimal Usage
```python
from gradience import GradienceCallback

callback = GradienceCallback(
    out_dir="./gradience_logs",
    guard_enabled=True,
)
trainer = Trainer(..., callbacks=[callback])
trainer.train()
```

## Configuration Presets

### Production (Balanced)
```python
GradienceCallback(
    out_dir="./logs",
    guard_enabled=True,
    guard_snapshot_interval=100,
    spectral_interval=50,
)
```

### Maximum Safety
```python
GradienceCallback(
    out_dir="./logs",
    guard_enabled=True,
    guard_snapshot_interval=50,
    guard_violation_threshold=1e4,
)
```

### Debug Mode (Dense Telemetry)
```python
GradienceCallback(
    out_dir="./logs",
    guard_enabled=True,
    guard_snapshot_interval=10,
    spectral_interval=5,
    telemetry_interval=1,
)
```

## Key Telemetry Fields

| Field | Type | Description |
|-------|------|-------------|
| `train_loss` | float | Current loss |
| `integrity_status` | str | OK/CORRUPT/RECOVERED/ABORT |
| `spectral_kappa` | float | κ̃ (condition proxy) |
| `spectral_slope` | float | Rate of κ̃ change |
| `risk_level` | str | STABLE/LEARNING/WARNING/DANGER/INSTABILITY |
| `guard_n_rollbacks` | int | Total rollbacks |
| `guard_lr_mult` | float | Current LR multiplier |

## Events

| Event | Meaning |
|-------|---------|
| `initialized` | Gradience started |
| `snapshot_saved` | Checkpoint created |
| `rollback_success` | Model restored from snapshot |
| `training_aborted` | Training halted (thrashing/persistent failure) |
| `training_complete` | Training finished normally |

## Analyzing Results

```python
import json

# Read telemetry
with open("gradience_logs/telemetry.jsonl") as f:
    for line in f:
        r = json.loads(line)
        if r.get("event") == "rollback_success":
            print(f"Rollback at step {r['step']}: {r['reason']}")

# Read summary
with open("gradience_logs/summary.json") as f:
    summary = json.load(f)
    print(f"Total rollbacks: {summary['guard']['total_rollbacks']}")
```

## Risk Level Thresholds

| Level | κ̃ Slope | Volatility | Action |
|-------|---------|------------|--------|
| STABLE | < 0.0005 | - | Normal |
| LEARNING | 0.0005-0.002 | - | Normal |
| WARNING | 0.002-0.004 | - | Alert, burst telemetry |
| DANGER | > 0.004 | - | Snapshot now, LR backoff |
| INSTABILITY | - | > 0.003 | Rollback or halt |

## What Gradience Does

- ✅ Detect 100% of NaN/Inf events
- ✅ Recover from transient faults
- ✅ Track training health (κ̃ slope)
- ✅ Abort on persistent instability
- ✅ Zero false positives on healthy training

## What Gradience Does NOT Do

- ❌ Predict exact failure step
- ❌ Fix bad hyperparameters
- ❌ Replace proper tuning
- ❌ Work on all architectures (validated on transformers)
