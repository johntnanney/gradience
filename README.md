# Gradience

**Telemetry-first observability for LoRA / PEFT fine-tuning.**

Gradience is a *flight recorder + mechanic* for LoRA runs:

- **Flight recorder:** emits a stable, line-by-line JSONL telemetry stream (`gradience.vnext.telemetry/v1`)
- **Mechanic:** reads that telemetry + audits adapters to produce conservative, testable recommendations

The guiding idea is simple: **constrained updates tend to generalize better**, and Gradience helps you detect when you’ve left that regime.

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

## Install (from source)

```bash
git clone https://github.com/johntnanney/gradience.git
cd gradience
python -m pip install -U pip
pip install -e .
```

For the toy example + HF/PEFT tooling:

```bash
pip install torch transformers peft safetensors datasets
```

**macOS note:** `--device cuda` requires a CUDA-enabled PyTorch build (typically Linux + NVIDIA GPU). On macOS, use `--device cpu`.

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

## Documentation

- **User manual:** `docs/USER_MANUAL.md`
- **Quick reference:** `QUICK_REFERENCE.md`
- **Telemetry contract:** `gradience/vnext/SCHEMA.md`
- **Examples:** `examples/vnext/`

---

## License

Apache 2.0 — see `LICENSE`.
