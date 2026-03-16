# Telemetry API

::: gradience.vnext.telemetry

Structured JSONL telemetry for recording and analyzing training dynamics.

## Schema: `gradience.vnext.telemetry/v1`

Each line in a telemetry file is a JSON object with these required fields:

| Field | Type | Description |
|-------|------|-------------|
| `schema` | `str` | Must equal `"gradience.vnext.telemetry/v1"` |
| `ts` | `float` | Unix timestamp (seconds) |
| `run_id` | `str` | Unique run identifier |
| `event` | `str` | Event type (see below) |
| `step` | `int \| null` | Training step number |

Extra keys are allowed on any event for forward compatibility.

## Event types

### `run_start`

Emitted at the beginning of training.

```json
{
  "schema": "gradience.vnext.telemetry/v1",
  "event": "run_start",
  "run_id": "abc123",
  "step": 0,
  "ts": 1710000000.0,
  "config": { "model_name": "mistral-7b", "lora": { "r": 16 } },
  "meta": { "git_hash": "a1b2c3d" }
}
```

### `train_step`

Emitted on each logging step.

```json
{
  "event": "train_step",
  "step": 100,
  "loss": 1.23,
  "lr": 5e-5
}
```

### `eval`

Emitted after evaluation.

```json
{
  "event": "eval",
  "step": 500,
  "split": "test",
  "metrics": { "ppl": 2.3, "accuracy": 0.35, "n": 100 }
}
```

### `metrics`

Emitted for spectral or structural measurements.

```json
{
  "event": "metrics",
  "step": 500,
  "kind": "spectral",
  "metrics": { "stable_rank_mean": 3.2, "utilization_mean": 0.4 }
}
```

### `alert`

Emitted when an issue is detected.

```json
{
  "event": "alert",
  "step": 500,
  "severity": "warning",
  "code": "memorization_gap",
  "message": "Train-test gap exceeds threshold"
}
```

### `recommendation`

Emitted with suggested actions.

```json
{
  "event": "recommendation",
  "step": 500,
  "recommendations": [
    { "severity": "info", "action": "reduce_rank", "message": "Consider r=8" }
  ]
}
```

### `run_end`

Emitted at the end of training.

```json
{
  "event": "run_end",
  "step": 1000,
  "status": "ok"
}
```

## TelemetryWriter

Write telemetry events to a JSONL file.

```python
from gradience.vnext.telemetry import TelemetryWriter
from gradience.vnext.types import ConfigSnapshot

writer = TelemetryWriter("run.jsonl")

# Record training start
writer.write_run_start(config=ConfigSnapshot(model_name="mistral-7b"))

# Record training steps
for step in range(100):
    writer.write_train_step(step=step, loss=2.0 - step * 0.01)

# Record evaluation
writer.write_eval(step=100, split="test", metrics={"accuracy": 0.85})

# End training
writer.write_run_end(status="ok")
```

## TelemetryReader

Read and analyze telemetry from a JSONL file.

```python
from gradience import TelemetryReader

reader = TelemetryReader("run.jsonl")

# Get summary statistics
summary = reader.summary()
print(f"Total steps: {summary['total_steps']}")
print(f"Final loss: {summary['final_train_loss']}")

# Iterate events
for event in reader.events():
    if event["event"] == "alert":
        print(f"Alert at step {event['step']}: {event['message']}")
```

Both `TelemetryWriter` and `TelemetryReader` are re-exported from `gradience.__init__` for convenience:

```python
from gradience import TelemetryWriter, TelemetryReader
```
