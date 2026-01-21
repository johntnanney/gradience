# Gradience

**Telemetry-first observability for LoRA / PEFT fine-tuning.**

## Quick Start (One Line)

```python
from transformers import Trainer
from gradience.vnext.integrations.hf import GradienceCallback

trainer = Trainer(..., callbacks=[GradienceCallback()])
trainer.train()

# Telemetry: <output_dir>/run.jsonl
```

That's it! Gradience now tracks your training. 

### After Training - Copy & Paste These Commands:

```bash
# View training summary with recommendations
gradience monitor <output_dir>/run.jsonl --verbose

# Analyze LoRA adapter efficiency (if using PEFT)
gradience audit --peft-dir <output_dir>/adapter --layers --json

# Combine telemetry + audit for complete analysis
gradience audit --peft-dir <output_dir>/adapter --append <output_dir>/run.jsonl
gradience monitor <output_dir>/run.jsonl --verbose
```

**Run the complete example:**
```bash
python examples/vnext/golden_path.py

# Then analyze it:
gradience monitor outputs/run.jsonl --verbose
```

📋 **See [CLI_CHEATSHEET.md](CLI_CHEATSHEET.md) for more copy-paste commands**

---

## What is Gradience?

Gradience is a *flight recorder + mechanic* for LoRA runs:

- **Flight recorder:** emits a stable, line-by-line JSONL telemetry stream (`gradience.vnext.telemetry/v1`)
- **Mechanic:** reads that telemetry + audits adapters to produce conservative, testable recommendations

The guiding idea is simple: **constrained updates tend to generalize better**, and Gradience helps you detect when you've left that regime.

> This release ships the canonical API under `gradience.vnext` and the stable schema `gradience.vnext.telemetry/v1`.

---

## What Gradience is

- **`gradience check`**: pre-flight validation of a PEFT + training config
- **`gradience monitor`**: summarize a run JSONL (gap, basic diagnostics, recommendations)
- **`gradience audit`**: analyze a PEFT adapter for rank waste + “suggested rank” compression hints

---

## What Gradience is NOT

- Not AutoML (it won’t tune your hyperparameters for you)
- Not a training framework (it sits *next to* your stack)
- Not an oracle (it doesn’t “predict success from spectra”)
- Not a replacement for evaluation (recommendations are **hypotheses**; verify on held-out eval)

---

## Install

### 📦 Recommended Setup (Contributors & Users)

**One canonical path - boring and predictable:**

```bash
git clone https://github.com/johntnanney/gradience.git
cd gradience

python -m venv .venv
source .venv/bin/activate

pip install -U pip
pip install -e ".[hf,dev]"
```

This gives you:
- ✅ **HuggingFace integration**: `transformers`, `peft`, `datasets`, `accelerate`, `safetensors` 
- ✅ **Development tools**: `pytest`, `ruff`, `mypy`, `pre-commit`
- ✅ **Core functionality**: `torch`, `numpy`, `pyyaml`

### Alternative Installation Options

```bash
# Minimal core only (torch, numpy, pyyaml)
pip install -e .

# HuggingFace integration only
pip install -e ".[hf]"

# Benchmarking suite
pip install -e ".[bench]"

# Everything (all dependencies)  
pip install -e ".[all]"

# Fast downloads (add hf_transfer)
pip install -e ".[hf,fast]"
```

### 🚀 Quick Development Setup

```bash
# Use Makefile for convenience
make setup        # Creates venv and installs [hf,dev]
make setup-cache  # Configure storage (prevents "disk quota exceeded")
make verify-version  # Verify installation
```

> 💾 **Storage tip**: Run `make setup-cache` to prevent disk space issues in cloud environments. See [docs/storage_and_caching.md](docs/storage_and_caching.md) for details.

### What's Included in Each Extra

- **`[hf]`**: HuggingFace integration (`transformers>=4.35.0`, `peft>=0.7.0`, `datasets>=2.14.0`, `accelerate>=0.20.0`, `safetensors>=0.4.0`, `sentencepiece`, `protobuf`)
- **`[bench]`**: Benchmarking suite (includes `[hf]` + `scikit-learn`, `pandas`)
- **`[dev]`**: Development tools (`pytest>=7.0.0`, `pytest-cov`, `ruff>=0.1.0`, `mypy>=1.5.0`, `pre-commit>=3.0.0`, `build`, `twine`)
- **`[fast]`**: Performance enhancement (`hf_transfer>=0.1.4` for faster downloads)
- **`[all]`**: Everything above combined

> 💡 **Note**: PyTorch is always required but left to user choice for CPU/GPU compatibility

## HuggingFace Integration

The [Quick Start](#quick-start-one-line) above shows the minimal integration. For more control:

### Custom Configuration

```python
from transformers import Trainer
from gradience.vnext.integrations.hf import GradienceCallback, GradienceCallbackConfig

config = GradienceCallbackConfig(
    output_dir="./my_runs",          # Custom telemetry location
    filename="experiment.jsonl",     # Custom filename
    dataset_name="your_dataset",     # Optional metadata
    task_profile="easy_classification", 
    notes="testing rank 16 vs 32"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    callbacks=[GradienceCallback(config)]
)
trainer.train()
```

### Try It Now

```bash
# Minimal integration demo
python examples/vnext/hf_trainer_example.py

# Complete example with all options
python examples/vnext/hf_trainer_run.py
```

**Examples:**
- [`hf_trainer_example.py`](examples/vnext/hf_trainer_example.py) - Minimal "one line" integration
- [`hf_trainer_run.py`](examples/vnext/hf_trainer_run.py) - Full configuration example

**Note:** The callback writes telemetry to `training_args.output_dir/run.jsonl` by default.

---

## Common Workflows (Copy & Paste)

### Standard HuggingFace + PEFT Training

```bash
# After your trainer.train() completes:

# 1. Quick check - did training go well?
gradience monitor ./output/run.jsonl

# 2. Detailed analysis with recommendations
gradience monitor ./output/run.jsonl --verbose

# 3. Check if your LoRA rank is wasteful
gradience audit --peft-dir ./output/adapter --layers

# 4. Get rank compression suggestions
gradience audit --peft-dir ./output/adapter --suggest-per-layer --json

# 5. Complete workflow: append audit to telemetry, then analyze
gradience audit --peft-dir ./output/adapter --append ./output/run.jsonl
gradience monitor ./output/run.jsonl --verbose
```

### Pre-flight Config Validation

```bash
# Before training - validate your config
gradience check --task text_generation \
    --peft adapter_config.json \
    --training training_args.json \
    --verbose

# Or from output directories
gradience check --task text_generation \
    --peft-dir ./peft_config \
    --training-dir ./trainer_config
```

### Debugging a Failed Run

```bash
# Check for gradient explosions, NaN losses, etc.
gradience monitor failed_run/run.jsonl --verbose --json | \
    python -m json.tool | grep -A5 "alerts"

# Look for Guard interventions (if enabled)
grep "GUARD_" failed_run/run.jsonl | head -20
```

---

## Quickstart (golden path)

### 1) Run a toy LoRA example (writes telemetry + a PEFT adapter)

```bash
# CPU (works anywhere)
python examples/vnext/toy_lora_run.py --out runs/toy_run --device cpu

# GPU (Linux + CUDA)
python examples/vnext/toy_lora_run.py --out runs/toy_run --device cuda
```

Outputs:

- Telemetry: `runs/toy_run/run.jsonl`
- PEFT dir: `runs/toy_run/peft/`
- Training args: `runs/toy_run/training/training_args.json`

### 2) Check config (pre-flight)

```bash
gradience check --task sst2 --peft-dir runs/toy_run/peft --training-dir runs/toy_run/training
```

### 3) Monitor run (telemetry)

```bash
gradience monitor runs/toy_run/run.jsonl --verbose
```

### 4) Audit adapter (efficiency)

```bash
gradience audit --peft-dir runs/toy_run/peft --top-wasteful 10
```

### 5) Append audit to telemetry, then re-monitor

```bash
gradience audit --peft-dir runs/toy_run/peft --append runs/toy_run/run.jsonl
gradience monitor runs/toy_run/run.jsonl --verbose
```

---

## Core commands

```bash
# Check config before training
gradience check --task <task> --peft-dir <dir> --training-dir <dir> [--verbose] [--json]

# Monitor a run JSONL
gradience monitor <run.jsonl> [--verbose] [--json]

# Audit a PEFT adapter directory
gradience audit --peft-dir <dir> [--top-wasteful N] [--json]

# Audit with Update Dominance Ratio (UDR) metrics
gradience audit --peft-dir <dir> --base-model <model_name> [--json]

# Append audit stats into an existing run JSONL
gradience audit --peft-dir <dir> --append <run.jsonl>
```

---

## QLoRA / quantized base models (caveat)

Gradience can audit **LoRA and QLoRA** adapters (the adapter matrices are still learned in full precision).
However, in QLoRA the adapter may partly **compensate quantization error** in the base model, so interpretation is more ambiguous.

Practical guidance:
- Treat **utilization / suggested rank** as **compression hints**, not guarantees — verify with eval.
- Scaled update magnitude stats (e.g. `alpha/r × ||ΔW||`) remain comparable, but “dominance vs base”
  is harder to define under quantization.

Tip: include quantization settings in `run_start.meta` (bits / scheme / compute dtype) so runs can be compared apples-to-apples.

---

## Telemetry & privacy defaults

Gradience telemetry is local JSONL. Treat it as **sensitive**.

Defaults are intentionally conservative:
- **TelemetryWriter redacts strings >256 characters by default.**
- To log raw text, you must explicitly opt in (e.g. `--telemetry-allow-text`). This is dangerous; treat JSONL as sensitive.
- The LoRA auditor prefers `adapter_model.safetensors` when present.

---

## Python API (minimal)

```python
from gradience.vnext.telemetry import TelemetryWriter
from gradience.vnext.types import Severity

with TelemetryWriter("run.jsonl") as tw:
    tw.run_start(config, meta={"experiment": "demo"})
    tw.train_step(1, loss=2.3, lr=5e-5)
    tw.eval(100, split="test", metrics={"accuracy": 0.92, "n": 100})
    tw.alert(severity=Severity.WARNING, code="LR_HIGH", message="Learning rate may be too high")
    tw.run_end(status="ok")
```

Read + summarize:

```python
from gradience.vnext.telemetry_reader import TelemetryReader

r = TelemetryReader("run.jsonl")
signals = r.summarize()
latest_test = r.latest_eval(split="test")
```

---

## API stability

Gradience is an applied research codebase with a strong preference for **reproducible outputs** over frozen internals.
To keep things usable without blocking refactors, we define "stable surfaces" in tiers.

### ✅ Stable (public API)

These are the interfaces we aim to keep backward-compatible across minor releases (and will call out loudly if we break):

**Command-line interfaces**
- `gradience audit ...`
- `gradience monitor ...`
- `python -m gradience.bench.run_bench --config <yaml> --output <dir>`
- `python -m gradience.bench.aggregate <run_dir>... --output <dir>`

**Bench config schema**
- The top-level structure and semantics of Bench YAML configs (e.g., `model`, `task`, `lora`, `train`, `compression`, `runtime`, `seed`).

**Canonical artifacts (schema + meaning of core keys)**
- `audit.json`
- `bench.json`, `bench.md`
- `bench_aggregate.json`, `bench_aggregate.md`

We may add new fields freely, but we avoid renaming/removing core keys without a clear migration path.

**Python wrappers**
- `gradience.api` (thin wrappers around the stable CLI/module entrypoints).
  If you want to call Gradience from Python, prefer this module over importing internal utilities.

### 🟨 Provisional (public, but may evolve)

These are intended for external use, but may change as we learn what's actually needed:

- `gradience.vnext.integrations.hf.GradienceCallback` (Hugging Face Trainer integration)
- `gradience.bench.task_profiles.*` (task/model routing and evaluation abstraction)

We'll document changes in release notes, but we reserve the right to adjust these interfaces more aggressively than the "Stable" tier.

### 🧪 Experimental (opt-in, not guaranteed)

- Anything under `gradience/vnext/experimental/*` (e.g., Guard prototypes)
- Features labeled experimental in docstrings/README

These may change, move, or be removed without notice. Use only if you're willing to validate behavior yourself.

### 🚫 Internal (no compatibility promise)

Everything else is internal implementation detail, including (but not limited to):
- `gradience.bench.protocol` and helper functions
- low-level spectral utilities and telemetry plumbing

If you import these directly, you're depending on internals and should expect breakage.

### Reproducibility tip

For published results, pin a tag:
- `pip install git+https://github.com/johntnanney/gradience.git@vX.Y.Z`

and cite the corresponding `bench_aggregate.json/md` artifacts.

---

## Documentation

- **API stability guide:** `docs/api_stability.md` (stable interfaces and migration patterns)
- **User manual:** `USER_MANUAL.md`
- **Quick reference:** `QUICK_REFERENCE.md`
- **Telemetry contract:** `gradience/vnext/SCHEMA.md`
- **Examples:** `examples/vnext/`

---

## License

Apache 2.0 — see `LICENSE`.
