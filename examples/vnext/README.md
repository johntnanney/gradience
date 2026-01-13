# Gradience vNext Examples

This directory contains examples demonstrating Gradience vNext features.

## Fault Injection Example

**File**: `hf_guard_fault_injection.py`

Demonstrates LoRA Guard functionality with HuggingFace Trainer by:

1. **Setting up a minimal training scenario** with a tiny LoRA-like model and synthetic dataset
2. **Enabling Guard** with conservative settings (CPU-only)
3. **Injecting NaN loss** at step 15 to trigger Guard
4. **Verifying Guard behavior** through telemetry analysis

### What the test demonstrates:

- **Snapshot creation**: Guard takes snapshots every 5 steps
- **Trigger detection**: Guard detects NaN gradient (from NaN loss)
- **Rollback execution**: Guard restores model to step 15 snapshot
- **Anti-thrash protection**: Guard prevents further rollbacks due to cooldown
- **Telemetry logging**: All events are properly logged to `run.jsonl`

### Expected telemetry events:

- `GUARD_TRIGGERED` (INFO) - when NaN is detected
- `GUARD_ROLLBACK` (WARNING) - successful rollback
- `GUARD_ABORT` (ERROR) - cooldown prevents further rollbacks
- Metrics with `kind="guard"` for snapshots and rollback

### Usage:

```bash
# From gradience root directory:
python examples/vnext/hf_guard_fault_injection.py
```

**Requirements**: `transformers`, `torch` (CPU-only)

This is a **proof tool** for Guard functionality - not intended for CI initially, but demonstrates that Guard successfully protects against training instability in real HF Trainer scenarios.

## CPU-Friendly Grad Explosion Validation

**File**: `hf_guard_fault_injection_grad_explosion.py`

Fast CPU-only validation for gradient explosion detection by:

1. **Mock LoRA model setup** with minimal parameters
2. **Direct callback testing** without HuggingFace Trainer overhead  
3. **Fake grad_norm injection** to trigger Guard efficiently
4. **Telemetry validation** for grad explosion trigger

### What the test demonstrates:

- **Fast execution**: Completes in ~2 seconds (CPU-only)
- **Grad explosion detection**: Guard triggers when `grad_norm > threshold`
- **Rollback execution**: Guard restores to previous snapshot
- **Canonical telemetry**: All events logged with proper trigger context

### Expected telemetry events:

- `GUARD_TRIGGERED` (INFO) - with `trigger=grad_explosion`
- `GUARD_ROLLBACK` (WARNING) - successful rollback to snapshot
- Metrics with `kind="guard"` and `action="rollback"`

### Usage:

```bash
# From gradience root directory:
python examples/vnext/hf_guard_fault_injection_grad_explosion.py
```

**Requirements**: `torch` (CPU-only) - no `transformers` needed

This is optimized for **rapid validation** and suitable for CI testing due to its speed and minimal dependencies.

## Realistic Grad Explosion Validation

**File**: `hf_guard_realistic_grad_explosion.py`

More realistic validation using actual gradient computation by:

1. **Realistic tiny LoRA model** with forward/backward passes
2. **Real gradient computation** from actual loss functions
3. **Loss explosion multiplier** (1e4x) to create genuine grad explosion
4. **CPU-friendly execution** with small model (~3k parameters)

### What the test demonstrates:

- **Real gradients**: Actual forward/backward passes compute grad_norm
- **Loss-driven explosion**: Uses `loss *= 1e4` to create realistic gradient explosion
- **Normal baseline**: Establishes typical grad_norm (~0.3) before explosion
- **Explosion validation**: Verifies 10,000x gradient norm increase triggers Guard

### Expected telemetry events:

- `GUARD_TRIGGERED` (INFO) - with `trigger=grad_explosion` and real grad_norm
- `GUARD_ROLLBACK` (WARNING) - successful rollback to snapshot
- Metrics with computed gradient norms in context

### Usage:

```bash
# From gradience root directory:
python examples/vnext/hf_guard_realistic_grad_explosion.py
```

**Requirements**: `torch` (CPU-only) - no `transformers` needed

This provides **realistic gradient validation** while remaining fast (~3 seconds) and suitable for CI testing. More realistic than fake grad_norm injection but still lightweight.

## CPU-Friendly Cooldown Lockout Validation

**File**: `hf_guard_cooldown_lockout.py`

Fast validation for Guard's anti-thrash cooldown protection by:

1. **First trigger simulation** with fake grad_norm causing rollback
2. **Second trigger within cooldown** causing abort (no rollback)
3. **Cooldown period verification** ensuring anti-thrash protection works
4. **Fastest execution** (~1.4 seconds) using fake logs

### What the test demonstrates:

- **First rollback success**: Guard triggers and rolls back normally
- **Cooldown protection**: Second trigger within 20 steps is aborted
- **Anti-thrash behavior**: Guard prevents infinite rollback loops
- **Telemetry validation**: Both GUARD_ROLLBACK and GUARD_ABORT events

### Expected telemetry sequence:

- Step 10: `GUARD_TRIGGERED` → `GUARD_ROLLBACK` (success)
- Step 15: `GUARD_TRIGGERED` → `GUARD_ABORT` (cooldown protection)
- Metrics showing cooldown context and rollback counts

### Configuration for testing:
```python
guard_cooldown_steps=20     # 20-step cooldown period
guard_max_rollbacks=999     # High limit (cooldown is limiting factor)
guard_snapshot_every=1      # Always have snapshots available
```

### Usage:

```bash
# From gradience root directory:
python examples/vnext/hf_guard_cooldown_lockout.py
```

**Requirements**: `torch` (CPU-only) - no `transformers` needed

This validates **anti-thrash protection** with the fastest execution time, ensuring Guard prevents runaway rollback scenarios in production.

## CPU-Friendly Max Rollbacks Validation

**File**: `hf_guard_max_rollbacks_lockout.py`

Alternative anti-thrash validation testing max_rollbacks limit by:

1. **First trigger simulation** with fake grad_norm causing rollback
2. **Second trigger immediately** hitting max_rollbacks limit  
3. **Max rollbacks verification** ensuring window-based protection works
4. **Fast execution** (~1.4 seconds) using fake logs

### What the test demonstrates:

- **First rollback success**: Guard triggers and rolls back normally
- **Max rollbacks protection**: Second trigger hits limit and is aborted
- **Window-based anti-thrash**: Guard tracks rollbacks over time window
- **Alternative mechanism**: Tests max_rollbacks instead of cooldown

### Expected telemetry sequence:

- Step 10: `GUARD_TRIGGERED` → `GUARD_ROLLBACK` (success)
- Step 11: `GUARD_TRIGGERED` → `GUARD_ABORT` (max_rollbacks exceeded)
- Metrics showing rollback counts and window context

### Configuration for testing:
```python
guard_cooldown_steps=0          # Cooldown disabled  
guard_max_rollbacks=1           # Only 1 rollback allowed in window
guard_window_steps=99999        # Very large window (effectively unlimited)
guard_snapshot_every=1          # Always have snapshots available
```

### Usage:

```bash
# From gradience root directory:
python examples/vnext/hf_guard_max_rollbacks_lockout.py
```

**Requirements**: `torch` (CPU-only) - no `transformers` needed

This validates the **alternative anti-thrash mechanism** with equivalent speed, ensuring Guard has multiple layers of protection against excessive rollbacks.

## Reusable CPU Driver for Guard Testing

**File**: `tests/helpers/hf_callback_driver.py`  
**Example**: `examples/vnext/guard_driver_example.py`

Clean, reusable infrastructure for testing HF callbacks without Trainer overhead:

### Key Features:

- **No HF dependencies**: No model downloads, datasets, or GPU requirements
- **Clean assertions**: Built-in helpers for common Guard validations
- **Predefined scenarios**: Ready-made test cases for common Guard behaviors
- **Fast execution**: Complete scenario testing in ~1.4 seconds
- **Automatic cleanup**: Handles temporary directories and resources

### Basic Usage:

```python
from gradience.vnext.integrations.hf import GradienceCallback
from tests.helpers.hf_callback_driver import HFCallbackDriver, LogEvent

# Create driver
driver = HFCallbackDriver(
    callback_class=GradienceCallback,
    callback_config={
        "enable_guard": True,
        "guard_grad_threshold": 100.0,
        "guard_cooldown_steps": 20,
    }
)

# Define scenario
log_events = [
    LogEvent(step=1, loss=2.5, grad_norm=1.0),    # Normal
    LogEvent(step=5, loss=2.3, grad_norm=500.0),  # Explosion
]

# Run and validate
result = driver.run_events(log_events, scenario_name="test")
result.assert_alert_present("GUARD_TRIGGERED")
result.assert_rollback_count(1)
driver.cleanup()
```

### Predefined Scenarios:

```python
from tests.helpers.hf_callback_driver import (
    create_grad_explosion_scenario,
    create_cooldown_scenario, 
    create_max_rollbacks_scenario
)

# Use ready-made scenarios
scenario = create_cooldown_scenario(first_step=10, second_step=15)
result = driver.run_scenario(scenario)
result.assert_sequence(["GUARD_INIT", "GUARD_TRIGGERED", "GUARD_ROLLBACK", "GUARD_TRIGGERED", "GUARD_ABORT"])
```

### Usage:

```bash
# Run the comprehensive demo:
python examples/vnext/guard_driver_example.py

# Run driver-based tests:
python -m pytest tests/test_guard_with_driver.py -v
```

**Requirements**: `torch` (CPU-only) - no `transformers` needed

This provides a **stable foundation** for Guard testing that's faster and more reliable than full HF Trainer scenarios while maintaining complete validation coverage.