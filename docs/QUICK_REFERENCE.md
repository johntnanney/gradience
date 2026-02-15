# Gradience Quick Reference (vNext)

*A research instrument for studying LoRA training dynamics, rank structure, and spectral geometry.*

## Install

```bash
# Core package (audit, monitor, merge-audit)
pip install gradience

# Full benchmarking suite
pip install "gradience[bench]"
```

## HuggingFace Integration

```python
from gradience.vnext.integrations.hf import GradienceCallback
trainer.add_callback(GradienceCallback())
```

```bash
# Try the minimal example
python examples/vnext/hf_trainer_example.py
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

## Check (experiment validation)

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
# Summary of training dynamics
gradience monitor runs/toy_run/run.jsonl

# Detailed evidence with diagnostic signals
gradience monitor runs/toy_run/run.jsonl --verbose

# JSON summary (for downstream analysis scripts)
gradience monitor runs/toy_run/run.jsonl --json > summary.json
```

## Audit (spectral analysis of adapter weights)

```bash
# Spectral audit — singular value decomposition of all LoRA matrices
gradience audit --peft-dir runs/toy_run/peft

# Show top N underutilized layers by rank
gradience audit --peft-dir runs/toy_run/peft --top-wasteful 10

# JSON output (for plotting, cross-run comparison)
gradience audit --peft-dir runs/toy_run/peft --json > audit.json

# Append audit metrics into telemetry for joint analysis
gradience audit --peft-dir runs/toy_run/peft --append runs/toy_run/run.jsonl

# Custom paths
gradience audit \
  --peft-dir runs/toy_run/peft \
  --adapter-config path/to/adapter_config.json \
  --weights path/to/adapter_model.safetensors
```

## Truncate (rank structure analysis via SVD)

```bash
# Reduce rank and measure retained spectral energy
gradience truncate --peft-dir runs/toy_run/peft --out-dir runs/compressed --rank 8

# With detailed per-layer energy retention
gradience truncate --peft-dir adapter_r16 --out-dir adapter_r8 --rank 8 --verbose

# JSON output (energy curves, per-layer statistics)
gradience truncate --peft-dir adapter --out-dir compressed --rank 4 --json

# Save detailed spectral report
gradience truncate --peft-dir adapter --out-dir adapter_r6 --rank 6 --report compression.json
```

## Research workflow

```bash
# 1) Validate experiment configuration
gradience check --task sst2 --peft-dir config/peft --training-dir config/training

# 2) Run training (your code)
python train.py --output runs/experiment

# 3) Analyze training dynamics from telemetry
gradience monitor runs/experiment/run.jsonl --verbose

# 4) Spectral audit — identify rank utilization patterns
gradience audit --peft-dir runs/experiment/peft --top-wasteful 5

# 5) Append audit + re-analyze joint signals
gradience audit --peft-dir runs/experiment/peft --append runs/experiment/run.jsonl
gradience monitor runs/experiment/run.jsonl --verbose

# 6) Optional: Test compression hypothesis (how much rank is actually used?)
gradience truncate --peft-dir runs/experiment/peft --out-dir runs/experiment/peft_compressed --rank 8
```

## Extracting spectral trajectories for plotting

```bash
# Dump per-layer spectral data to JSON for external analysis
gradience audit --peft-dir runs/seed_42/peft --json > seed_42_spectra.json

# Extract training dynamics as JSON for plotting loss/lr curves
gradience monitor runs/seed_42/run.jsonl --json > seed_42_dynamics.json
```

```python
# Example: load spectral data for plotting
import json, matplotlib.pyplot as plt

with open("seed_42_spectra.json") as f:
    spectra = json.load(f)
# Plot singular value distributions, energy retention, etc.
```

## Comparing geometric signatures across seeds

```bash
# Run spectral audit on each seed
for seed in 42 123 7; do
  gradience audit --peft-dir runs/seed_${seed}/peft --json > spectra_${seed}.json
done

# Compare training dynamics across seeds
for seed in 42 123 7; do
  gradience monitor runs/seed_${seed}/run.jsonl --json > dynamics_${seed}.json
done

# Diff summaries to find structural divergence
diff <(jq '.summary' spectra_42.json) <(jq '.summary' spectra_123.json)
```

## Correlating metrics with downstream evaluation

```bash
# 1) Extract spectral audit metrics
gradience audit --peft-dir runs/experiment/peft --json > spectra.json

# 2) Extract training signals
gradience monitor runs/experiment/run.jsonl --json > signals.json

# 3) Run your downstream eval
python eval.py --adapter runs/experiment/peft --output eval_results.json

# 4) Join in analysis script — correlate spectral energy with task accuracy
python analysis/correlate_spectra_eval.py \
  --spectra spectra.json \
  --signals signals.json \
  --eval eval_results.json
```

## Configuration presets

```python
from gradience.vnext.integrations.hf import GradienceCallback

# Dense telemetry — for detailed training dynamics studies
trainer.add_callback(GradienceCallback())  # default: captures every step

# Lightweight monitoring — for large parameter sweeps
trainer.add_callback(GradienceCallback())  # stream to per-run JSONL, analyze post-hoc

# Diagnostic mode — verbose alerts for debugging anomalies
trainer.add_callback(GradienceCallback())  # combine with --verbose on monitor
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
