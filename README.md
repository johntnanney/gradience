# Gradience

**Training Integrity Insurance: Spectral monitoring and guard system for stable deep learning.**

[![PyPI version](https://badge.fury.io/py/gradience.svg)](https://badge.fury.io/py/gradience)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## The Problem

Training neural networks fails in subtle ways:

1. **Silent corruption**: Weights become NaN/Inf but training continues, wasting GPU hours
2. **Late detection**: Loss explosion is visible only after damage is done
3. **No recovery**: When things go wrong, you restart from scratch

## The Solution

Gradience provides **training integrity insurance** through:

### 🔬 Spectral Monitoring
Track **κ̃ (kappa-tilde)**, a condition number proxy that often *leads* loss—showing instability before it manifests.

### 🛡️ Guard System
Automatic checkpoint and rollback with anti-thrash protection and LR dampening.

## Installation

```bash
pip install gradience[hf]
```

## Quick Start

```python
from gradience import GradienceCallback
from transformers import Trainer

callback = GradienceCallback(
    out_dir="./gradience_logs",
    guard_enabled=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    callbacks=[callback],
)
trainer.train()
```

## Demo

Test with fault injection:

```bash
python -m gradience.demo --guard --inject-nan
```

Output:
```
============================================================
GRADIENCE DEMO
============================================================
  Guard:       True
  Inject NaN:  True @ step 50
  
🔥 INJECTING NaN at step 50...
   Corrupted: transformer.h.0.attn.c_attn.weight

============================================================
RESULTS
============================================================
  Duration:  12.3s
  Success:   True

🛡️  Guard:
  Integrity: OK
  Rollbacks: 1
  Recoveries: 1
```

## What You Get

### Telemetry (`telemetry.jsonl`)
```json
{"event": "telemetry", "step": 100, "train_loss": 2.34, "spectral_kappa": 45.2, "risk_level": "LEARNING"}
{"event": "rollback_success", "step": 156, "reason": "NONFINITE_WEIGHTS"}
```

### Summary (`summary.json`)
```json
{
  "total_steps": 1000,
  "final_loss": 1.23,
  "guard": {"total_rollbacks": 2, "integrity_status": "OK"},
  "spectral": {"final_kappa": 23.4, "final_risk_level": "STABLE"}
}
```

## Risk Levels

| Level | κ̃ Slope | Response |
|-------|---------|----------|
| 🟢 STABLE | < 0.0005 | Normal operation |
| 🟢 LEARNING | 0.0005-0.002 | Healthy training |
| 🟡 WARNING | 0.002-0.004 | Burst telemetry, snapshot |
| 🔴 DANGER | > 0.004 | Immediate snapshot, LR backoff |
| 🔴 INSTABILITY | volatility > 0.003 | Rollback or halt |

## Configuration

```python
GradienceCallback(
    out_dir="./gradience_logs",
    
    # Guard (checkpoint/rollback)
    guard_enabled=True,
    guard_snapshot_interval=100,
    guard_violation_threshold=1e6,
    guard_max_rollbacks=3,
    guard_lr_backoff=0.5,
    
    # Spectral monitoring
    spectral_enabled=True,
    spectral_interval=50,
    
    # Telemetry
    telemetry_interval=10,
)
```

## Non-HuggingFace Usage

```python
from gradience import GradienceController

controller = GradienceController(out_dir="./logs")
controller.initialize(model, save_fn, load_fn, get_lr, set_lr)

for step, loss in training_loop:
    if not controller.step(step, loss, model):
        break  # Training aborted

summary = controller.finalize()
```

## Overhead

| Snapshot Interval | Overhead |
|-------------------|----------|
| 50 steps | ~15% |
| 100 steps | ~12% |
| 200 steps | ~8% |

## What Gradience Does

✅ Detect 100% of NaN/Inf events  
✅ Recover from transient faults  
✅ Track training health (κ̃ slope)  
✅ Abort on persistent instability  
✅ Zero false positives on healthy training  

## What Gradience Does NOT Do

❌ Predict exact failure step  
❌ Fix bad hyperparameters  
❌ Replace proper tuning  

## License

MIT

## Telemetry & Privacy

TelemetryWriter redacts strings longer than 256 characters by default.
