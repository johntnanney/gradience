# Gradience Quick Reference (vNext)

## Install

```bash
git clone https://github.com/johntnanney/gradience.git
cd gradience
pip install -e .
pip install torch transformers peft safetensors datasets
```

## Device check

```bash
python -c "import torch; print('cuda available:', torch.cuda.is_available())"
```

## Toy LoRA run (emits telemetry + PEFT adapter)

```bash
# CPU
python examples/vnext/toy_lora_run.py --out runs/toy_run --device cpu

# GPU (Linux + CUDA torch only)
python examples/vnext/toy_lora_run.py --out runs/toy_run --device cuda
```

## Check (pre-flight validation)

```bash
# Directory mode
gradience check --task sst2 --peft-dir runs/toy_run/peft --training-dir runs/toy_run/training

# File mode
gradience check --task sst2 \
  --peft runs/toy_run/peft/adapter_config.json \
  --training runs/toy_run/training/training_args.json

# Verbose
gradience check --task gsm8k --peft-dir <dir> --training-dir <dir> --verbose

# JSON output
gradience check --task sst2 --peft-dir <dir> --training-dir <dir> --json
```

## Monitor (telemetry analysis)

```bash
# Basic
gradience monitor runs/toy_run/run.jsonl

# Verbose (show evidence)
gradience monitor runs/toy_run/run.jsonl --verbose

# JSON summary
gradience monitor runs/toy_run/run.jsonl --json > summary.json
```

## Audit (adapter analysis)

```bash
# Basic audit
gradience audit --peft-dir runs/toy_run/peft

# Show top N wasteful layers
gradience audit --peft-dir runs/toy_run/peft --top-wasteful 10

# JSON output
gradience audit --peft-dir runs/toy_run/peft --json > audit.json

# Append audit metrics into telemetry
gradience audit --peft-dir runs/toy_run/peft --append runs/toy_run/run.jsonl

# Custom paths
gradience audit \
  --peft-dir runs/toy_run/peft \
  --adapter-config path/to/adapter_config.json \
  --weights path/to/adapter_model.safetensors
```

## Combined workflow

```bash
# 1) Check before training
gradience check --task sst2 --peft-dir config/peft --training-dir config/training

# 2) Run training (your code)
python train.py --output runs/experiment

# 3) Monitor telemetry
gradience monitor runs/experiment/run.jsonl --verbose

# 4) Audit adapter
gradience audit --peft-dir runs/experiment/peft --top-wasteful 5

# 5) Append audit + re-monitor
gradience audit --peft-dir runs/experiment/peft --append runs/experiment/run.jsonl
gradience monitor runs/experiment/run.jsonl --verbose
```

## Python API — TelemetryWriter

```python
from gradience.vnext.telemetry import TelemetryWriter
from gradience.vnext.types import Severity

with TelemetryWriter("run.jsonl") as tw:
    tw.run_start(config, meta={})
    tw.train_step(1, loss=2.3, lr=5e-5)
    tw.eval(100, split="test", metrics={"accuracy": 0.92, "n": 100})
    tw.metrics(100, kind="lora_audit", metrics={"utilization_mean": 0.18})
    tw.alert(severity=Severity.WARNING, code="LR_HIGH", message="Learning rate may be too high")
    tw.run_end(status="ok")
```

## Python API — TelemetryReader

```python
from gradience.vnext.telemetry_reader import TelemetryReader

r = TelemetryReader("run.jsonl")

# Iterate events
for e in r.iter_events(event_type="eval"):
    print(f"Step {e['step']} split={e['split']} metrics={e.get('metrics')}")

# Get latest
cfg = r.latest_config()
test_eval = r.latest_eval(split="test")

# Summarize
signals = r.summarize()
```

## Troubleshooting

```bash
# CUDA not available (common on macOS)
python examples/vnext/toy_lora_run.py --device cpu

# Command not found (entrypoint not installed)
pip install -e .

# Missing datasets
pip install datasets
```
