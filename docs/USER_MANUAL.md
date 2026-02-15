# Gradience Experiment Guide (vNext)

Gradience is a **spectral instrumentation framework** for studying training dynamics in fine-tuning runs, with a focus on **LoRA/PEFT**. It emits a stable JSONL stream ("flight recorder") and provides structured observations and analysis based on spectral decomposition of learned updates.

This guide documents the **canonical vNext release**:
- **Telemetry schema:** `gradience.vnext.telemetry/v1`
- **Core methodology:** `configure --> observe --> measure --> analyze`
- **Research stance:** *measure rigorously, interpret carefully, always validate empirically.*

> If you just want copy-paste commands, use **QUICK_REFERENCE.md**.
> If you want the project overview + a complete example, start with **README.md**.

---

## 1. Research methodology

### 1.1 What Gradience measures

Gradience provides instrumentation for a structured empirical workflow:

1) **Configure:** validate that your experimental setup is well-formed *before* committing compute.
2) **Observe:** collect structured telemetry (JSONL) throughout training.
3) **Measure:** apply spectral decomposition to the learned adapter, producing per-layer and aggregate statistics.
4) **Analyze:** correlate spectral structure with training outcomes to characterize how the adapter learned.

### 1.2 Spectral metrics as empirical probes

Spectral metrics (stable rank, energy rank, singular value distributions) are empirical measurements of the structure that training produced. They tell you *what happened* geometrically inside the adapter:

- How many effective dimensions does the learned update occupy?
- Is spectral mass concentrated in a few directions, or spread broadly?
- How does spectral structure vary across layers and module types?

Correlating these measurements with downstream performance across experimental conditions **is the research program**. Gradience provides the measurement infrastructure; interpreting the relationship between spectral structure and task performance is the investigator's work.

Gradience can surface observations like:
- "Train/test gap suggests memorization."
- "Spectral mass is concentrated in 3 of 16 rank dimensions across most layers."
- "Layer-to-layer variation in effective rank is unusually high."

These are measurements to interpret, not prescriptions to follow blindly.

---

## 2. Quick Start

### 2.1 HuggingFace Integration (Recommended)

If you're using HuggingFace Trainer, add telemetry with one line:

```python
from gradience.vnext.integrations.hf import GradienceCallback

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    callbacks=[GradienceCallback()]  # <-- Add this line
)
```

This automatically:
- Captures training configuration (model, dataset, LoRA config, hyperparameters)
- Logs training metrics (loss, learning rate, evaluation results)
- Writes telemetry to `training_args.output_dir/run.jsonl`
- Uses stable schema `gradience.vnext.telemetry/v1`

For custom configuration:

```python
from gradience.vnext.integrations.hf import GradienceCallback, GradienceCallbackConfig

config = GradienceCallbackConfig(
    dataset_name="glue/cola",           # Optional dataset identifier
    task_profile="easy_classification", # Optional task difficulty
    notes="Rank ablation seed=42"       # Optional experiment notes
)

trainer.add_callback(GradienceCallback(config))
```

### 2.2 Complete Examples

**Minimal demo (recommended first try):**
```bash
python examples/vnext/hf_trainer_example.py
# Shows: one line integration, next steps, complete workflow
# Outputs: ./gradience_example_output/run.jsonl + adapter files
```

**Full-featured example:**
```bash
python examples/vnext/hf_trainer_run.py
# Shows: detailed configuration, custom telemetry settings
# Outputs: ./hf_example_output/run.jsonl + adapter files
```

Both examples:
- Train tiny models on CPU (no GPU required)
- Generate telemetry automatically with vNext schema
- Produce audit-ready PEFT adapters
- Show complete workflow: train --> observe --> measure

---

## 3. Installation

### 3.1 Quick install (recommended)

```bash
# Core package -- includes torch + safetensors (audit, monitor, merge-audit)
pip install gradience

# Full benchmarking suite -- adds transformers, datasets, peft
pip install "gradience[bench]"
```

### 3.2 Install from source (for contributors)

```bash
git clone https://github.com/gradience-ai/gradience.git
cd gradience
pip install -e ".[dev]"
```

> **Important:** Always create your virtual environment from the repo root (the directory that contains `pyproject.toml`). This prevents import issues and ensures correct package installation.

### 3.3 Device note
- `--device cpu` works anywhere.
- `--device cuda` requires a CUDA-enabled PyTorch build (typically Linux + NVIDIA GPU). On macOS, CUDA is generally not available.

---

## 4. Replicating a complete experiment

This is the shortest path to a fully-instrumented experiment. The same flow appears in the README/quick reference.

### 4.1 Run a toy LoRA experiment (emits telemetry + PEFT dir)
```bash
# CPU
python examples/vnext/toy_lora_run.py --out runs/toy_run --device cpu

# GPU (Linux + CUDA torch)
python examples/vnext/toy_lora_run.py --out runs/toy_run --device cuda
```

Outputs:
```
runs/toy_run/
  run.jsonl                      # telemetry stream
  peft/                          # PEFT adapter artifacts
  training/training_args.json    # training config snapshot
```

### 4.2 Validate experiment configuration
```bash
gradience check --task sst2 --peft-dir runs/toy_run/peft --training-dir runs/toy_run/training
```

### 4.3 Summarize experimental observations
```bash
gradience monitor runs/toy_run/run.jsonl --verbose
```

### 4.4 Measure spectral structure of the adapter
```bash
gradience audit --peft-dir runs/toy_run/peft --top-wasteful 10
```

### 4.5 Append spectral measurements into telemetry and re-summarize
```bash
gradience audit --peft-dir runs/toy_run/peft --append runs/toy_run/run.jsonl
gradience monitor runs/toy_run/run.jsonl --verbose
```

### 4.6 Designing controlled experiments

For rigorous results, structure your experiments to isolate variables:

**Seed variation:** Run the same configuration across multiple seeds to distinguish signal from noise. Spectral structure that is consistent across seeds reflects genuine task geometry; structure that varies is likely an artifact of optimization trajectory.

**Rank ablation:** Train adapters at r={2,4,8,16,32} and measure both spectral structure and downstream performance. The relationship between provisioned rank, effective rank, and task accuracy reveals how much capacity the task actually requires.

**Layer-level analysis:** Compare spectral profiles across attention components (Q/K/V/O) and layer depth. Systematic patterns (e.g., deeper layers consistently showing higher effective rank) suggest something about how the model distributes task-relevant computation.

**Controlled comparisons:** When comparing conditions (learning rates, datasets, base models), keep everything else fixed. Log all configuration in `run_start.meta` so that post-hoc analysis can correctly attribute differences.

---

## 5. CLI reference

### 5.1 `gradience check`
**Purpose:** Validate experiment configuration before committing compute.

**Typical usage**
```bash
gradience check --task <task> --peft-dir <peft_out_dir> --training-dir <training_dir>
```

**Input options**
- `--peft` and `--training` can point at explicit files (e.g. `adapter_config.json` and `training_args.json`)
- `--peft-dir` and `--training-dir` auto-detect common filenames

**Output options**
- `--verbose`: print rationale/evidence
- `--json`: machine-readable output

### 5.2 `gradience monitor`
**Purpose:** Read a vNext JSONL file and summarize experimental observations and analysis.

```bash
gradience monitor <run.jsonl> [--verbose] [--json]
```

What monitor typically surfaces:
- Latest eval metrics (train/test/val)
- Gap ratio (when possible)
- LoRA spectral audit stats (if present)
- Analysis and observations (policy engine output)

### 5.3 `gradience audit`
**Purpose:** Apply spectral decomposition to a PEFT adapter and characterize rank structure.

```bash
gradience audit --peft-dir <dir> [--top-wasteful N] [--json]
```

Append mode:
```bash
gradience audit --peft-dir <dir> --append <run.jsonl>
```

Notes:
- The auditor prefers **`adapter_model.safetensors`** when present.
- JSON output includes summary metrics + optional per-layer rows.

### 5.4 `gradience truncate`
**Purpose:** Compress a LoRA adapter using SVD truncation to reduce parameter count while preserving performance.

```bash
gradience truncate --peft-dir <source_dir> --out-dir <target_dir> --rank <target_rank>
```

**Key options**
- `--rank <k>`: Target rank (must be smaller than original)
- `--alpha-mode {keep_ratio,keep_alpha}`: Alpha scaling behavior (default: `keep_ratio`)
- `--dtype {fp16,bf16,fp32}`: Output weight precision (default: `fp16`)
- `--report <path>`: Save detailed compression report as JSON
- `--verbose`: Show per-module energy retention statistics
- `--json`: Machine-readable output

**Output**
- Creates complete PEFT adapter directory at `--out-dir`
- Shows: input/output ranks, mean energy retained, parameter reduction
- Automatic `truncation_report.json` with per-module statistics

**Example workflow**
```bash
# 1) Characterize spectral structure of current adapter
gradience audit --peft-dir ./adapter_r16 --top-wasteful 5

# 2) Truncate to a rank informed by spectral analysis
gradience truncate --peft-dir ./adapter_r16 --out-dir ./adapter_r8 --rank 8 --verbose

# 3) Validate: measure downstream performance of truncated adapter
model = PeftModel.from_pretrained(base_model, "./adapter_r8")
```

**Technical notes**
- Uses fast QR-based SVD to avoid materializing large matrices
- Preserves all non-LoRA weights (task heads, modules_to_save)
- Output adapter is drop-in compatible with PEFT/Transformers
- Energy retention indicates approximation quality (higher = better)

---

## 6. Telemetry: what is logged

### 6.1 File format
Telemetry is **JSONL**: one JSON object per line.

Each event has:
- `schema` (must equal `gradience.vnext.telemetry/v1`)
- `ts` (unix timestamp)
- `run_id`
- `event`
- `step` (optional; may be null)
- plus an **event-specific payload**

### 6.2 Stable event names
The vNext contract guarantees the following stable event names:
- `run_start`
- `train_step`
- `eval`
- `metrics`
- `alert`
- `recommendation`
- `run_end`

### 6.3 Minimal metric keys (stable spine)
Gradience's analysis layer assumes these core metric keys when present:
- `loss`
- `ppl`
- `accuracy`
- `n`

Additional metrics should live in:
- `metrics(kind=...)` blocks, or
- `extras` dictionaries (forward compatible).

---

## 7. Privacy & safety defaults (important)

Gradience telemetry is local JSONL. Treat it as **sensitive**.

### 7.1 Redaction default
**TelemetryWriter redacts strings longer than 256 characters by default.**

This is a guardrail against accidentally logging prompts, dataset examples, or other raw text.

### 7.2 Opt-in text logging (dangerous)
To log raw text, you must explicitly opt in (e.g. `--telemetry-allow-text` in scripts that support it). If you do this, treat JSONL as sensitive and avoid uploading it to public places.

### 7.3 What Gradience does not log (by design)
- Training examples / prompts / labels (unless you explicitly opt in)
- Model weights
- Raw gradients

Gradience is intended to log **scalars + structured metadata**, not data.

---

## 8. Python API: TelemetryWriter

### 8.1 Minimal usage
```python
from gradience.vnext.telemetry import TelemetryWriter
from gradience.vnext.types import Severity

with TelemetryWriter("run.jsonl") as tw:
    tw.run_start(config, meta={"experiment": "demo"})

    tw.train_step(1, loss=2.3, lr=5e-5)

    tw.eval(100, split="test", metrics={"accuracy": 0.92, "n": 100})

    tw.alert(
        severity=Severity.WARNING,
        code="LR_HIGH",
        message="Learning rate may be too high for this setup",
        step=100,
        context={"lr": 5e-4},
    )

    tw.run_end(status="ok")
```

### 8.2 Common logging patterns
- **Log eval metrics** at the end of each epoch or at a fixed cadence.
- **Log train_step** scalars periodically (loss / lr) if you want learning curves.
- Use `metrics(kind="...")` to attach structured metric blocks (e.g. `lora_audit`, `spectral`, `structural`).

---

## 9. Python API: TelemetryReader

TelemetryReader is the "other half": it streams JSONL safely, validates schema, and produces a summary snapshot suitable for analysis.

### 9.1 Minimal usage
```python
from gradience.vnext.telemetry_reader import TelemetryReader

r = TelemetryReader("run.jsonl")

# Iterate events (optionally filter by type)
for e in r.iter_events(event_type="eval"):
    print(e.get("step"), e.get("split"), e.get("metrics"))

# Latest config + latest eval
cfg = r.latest_config()
test_eval = r.latest_eval(split="test")

# One-shot summary
signals = r.summarize()
```

### 9.2 What `summarize()` returns
A **SignalSnapshot** (or dict with the same information) that includes:
- latest eval metrics by split
- gap ratios when train + test metrics exist
- attached metric blocks like `lora_audit` (if present)

This is what monitor/analysis consumes.

---

## 10. Observations and analysis

### 10.1 Observation objects
Gradience's analysis engine emits **Recommendation** objects. These are designed to be:
- **human-readable**
- **testable** (verifiable by running eval or changing one variable)
- **empirically grounded** (tied to specific measurements)

Typical fields include:
- severity (`info` / `warning` / `error` / `critical`)
- code (stable identifier)
- message (what was observed)
- why (rationale)
- confidence (0-1)
- evidence (structured context)

### 10.2 `config_ok`
Gradience may emit a `config_ok` informational observation **only if there are no actionable findings**. This keeps output less noisy.

---

## 11. Spectral audit: what it reveals about training dynamics

### 11.1 What the audit characterizes

The spectral audit decomposes the learned LoRA update matrices to characterize the geometry of what training produced. Rather than treating the adapter as a black box, spectral analysis reveals its internal structure: how many independent directions the model actually used, how spectral energy is distributed, and how this varies across layers and module types.

The core questions:
- "How many effective dimensions does the learned update occupy?"
- "Is spectral mass concentrated (a few dominant directions) or diffuse (many directions contributing equally)?"
- "How does spectral structure vary across layers -- do deeper layers learn differently from shallow ones?"
- "What is the relationship between provisioned rank, effective rank, and downstream performance?"

### 11.2 Key metrics and what they reveal

**Stable rank** (per layer + aggregate)
The ratio of squared Frobenius norm to squared spectral norm. This measures the effective dimensionality of the update without requiring a hard threshold. A stable rank of 2.1 in a rank-16 adapter means the learned update is effectively two-dimensional -- the model concentrated its learning into a low-dimensional subspace. Comparing stable rank across layers reveals where the model invested capacity.

**Utilization** = stable_rank / r
A normalized view of how much of the provisioned rank capacity was used. Low utilization across all layers suggests the task may require fewer dimensions than were provisioned. High utilization in specific layers suggests those layers are capacity-constrained and may benefit from more rank. The *pattern* of utilization across layers is often more informative than any single value.

**Energy rank at 90%** (`k@90%`) and its distribution (p50/p90)
The number of singular directions needed to capture 90% of the update's energy. This is a more conservative measure than stable rank: it asks "how many directions carry most of the signal?" The distribution across layers (p50 vs p90) reveals whether spectral structure is consistent or heterogeneous across the model.

**Suggested ranks**
  - `suggested_r_global_median`: smallest r in {1,2,4,8,16,32} that covers median k@90%
  - `suggested_r_global_90`: smallest r in {1,2,4,8,16,32} that covers p90 k@90% (tail coverage)

These are derived from the spectral measurements and represent starting points for rank ablation experiments, not definitive answers. The median suggestion addresses typical layers; the p90 suggestion addresses the most demanding layers. The gap between them indicates how heterogeneous the spectral structure is.

### 11.3 Scaled update magnitude

Audit also logs *scaled update magnitude* ingredients, such as:
- `delta_sigma_max_scaled_*`
- `delta_frob_norm_scaled_*`

These measure how large the learned update is in absolute terms (scaled by layer dimensions to enable cross-layer comparison). They are useful for understanding which layers changed most during training, independently of spectral shape. A layer with large update magnitude but low stable rank learned a strong but simple signal; a layer with large update magnitude and high stable rank learned something more complex.

---

## 12. QLoRA / quantized base models (caveat)

Gradience can audit LoRA **and** QLoRA adapters mechanically, because the adapters are still low-rank matrices learned in higher precision.

However, under QLoRA the adapter may implicitly do two jobs:
1) task adaptation
2) quantization error compensation

This dual role has methodological implications for interpreting spectral structure. Some of the effective rank may be "spent" on compensating for quantization artifacts rather than representing task-relevant structure. This makes spectral metrics a composite measurement of two distinct phenomena.

Practical guidance for controlled studies:
- Compare spectral structure between LoRA (full-precision base) and QLoRA (quantized base) on the same task to isolate the quantization compensation component
- Treat suggested ranks as **starting points for ablation**, not as direct compression targets
- Verify with held-out eval after any rank modification
- Log quantization metadata in `run_start.meta` (quantization scheme, bits, group size) so post-hoc analysis can properly condition on these variables
- Consider that rank needs under QLoRA may be legitimately higher than under LoRA for the same task, precisely because of the compensation role

---

## 13. Troubleshooting

### 13.1 "gradience: command not found"
Make sure you installed the repo in editable mode:

```bash
pip install -e .
```

### 13.2 Missing dependencies (datasets, transformers, peft)
Install required packages:

```bash
pip install torch transformers peft safetensors datasets
```

### 13.3 "Torch not compiled with CUDA enabled"
You're using a CPU-only PyTorch build. Either:
- run with `--device cpu`, or
- install a CUDA-enabled torch build (Linux + NVIDIA GPU)

### 13.4 Audit can't find adapter weights
Gradience expects a PEFT output directory with:
- `adapter_config.json`
- `adapter_model.safetensors` (preferred) or equivalent adapter weights

---

## 14. Versioning policy

- The telemetry schema ID is **`gradience.vnext.telemetry/v1`**.
- Breaking schema changes require a bump to **`/v2`**.
- New optional fields and new `metrics(kind=...)` blocks are allowed in v1.

---

## Appendix A: Glossary

- **Telemetry (JSONL):** event stream of structured observations from a training run
- **Gap:** ratio between train and test metrics (often used as a memorization signal)
- **Stable rank:** effective dimensionality measure based on the ratio of squared Frobenius norm to squared spectral norm
- **Energy rank (`k@90%`):** number of singular directions needed to explain 90% of spectral energy
- **Utilization:** stable_rank / r (normalized measure of rank capacity usage)
- **Spectral structure:** the distribution of singular values in a learned update matrix, revealing the geometry of what training produced
- **PEFT:** Parameter-Efficient Fine-Tuning
- **LoRA:** Low-Rank Adaptation (learned low-rank update to weights)
- **QLoRA:** LoRA over a quantized base model

---

## Appendix B: Where to look in the repo

- `examples/vnext/` -- runnable examples
- `gradience/vnext/SCHEMA.md` -- telemetry contract
- `gradience/vnext/telemetry.py` -- TelemetryWriter
- `gradience/vnext/telemetry_reader.py` -- TelemetryReader
- `gradience/vnext/policy/` -- analysis engine
- `gradience/vnext/audit/` -- spectral auditor
- `gradience/bench/` -- internal validation framework used to calibrate defaults and validate analysis (see `gradience/bench/README.md`)
