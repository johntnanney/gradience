# HuggingFace Integration

Gradience provides a drop-in HuggingFace Trainer callback that records structured telemetry during LoRA fine-tuning.

## One-line integration

```python
from gradience.vnext.integrations.hf import GradienceCallback

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    callbacks=[GradienceCallback()],
)
trainer.train()
```

Telemetry is saved to `{output_dir}/run.jsonl` in the [gradience.vnext.telemetry/v1](../api/telemetry.md) schema.

## Configuration

Use `GradienceCallbackConfig` to customize behavior:

```python
from gradience.vnext.integrations.hf import GradienceCallback, GradienceCallbackConfig

config = GradienceCallbackConfig(
    output_dir="./telemetry",       # Override output location
    filename="training.jsonl",      # Custom filename (default: run.jsonl)
    dataset_name="gsm8k",           # Optional metadata
    task_profile="hard_reasoning",   # Optional task classification
    notes="baseline run",           # Optional notes
)

trainer = Trainer(
    model=model,
    args=training_args,
    callbacks=[GradienceCallback(config=config)],
)
```

### Config options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `output_dir` | `training_args.output_dir` | Where to write the telemetry file |
| `filename` | `"run.jsonl"` | Telemetry filename |
| `dataset_name` | `None` | Optional dataset name for metadata |
| `task_profile` | `None` | Task family: `easy_classification`, `hard_reasoning`, `generation` |
| `notes` | `None` | Free-text notes embedded in telemetry |
| `telemetry_allow_text` | `False` | Allow long text strings in telemetry |
| `telemetry_max_str_len` | `256` | Maximum string length before truncation |

## What gets recorded

The callback records the following telemetry events:

| Event | When | Content |
|-------|------|---------|
| `run_start` | `on_train_begin` | Config snapshot (model, LoRA, optimizer, training params) |
| `train_step` | `on_log` | Loss, learning rate, step number |
| `eval` | `on_evaluate` | Evaluation metrics (loss, accuracy, perplexity) |
| `run_end` | `on_train_end` | Final status |

## Analyzing telemetry

After training, analyze the recorded telemetry:

```bash
# CLI
gradience monitor run.jsonl --verbose

# Python
from gradience import TelemetryReader

reader = TelemetryReader("run.jsonl")
summary = reader.summary()
print(f"Steps: {summary['total_steps']}")
print(f"Final loss: {summary['final_train_loss']}")
```

## Requirements

The HuggingFace integration requires the `[hf]` extra:

```bash
pip install "gradience[hf]"
```

This installs `transformers`, `peft`, and `sentencepiece`.
