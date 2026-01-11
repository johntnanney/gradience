# Gradience User Manual (vNext)

Gradience is a **telemetry-first monitor** for fine-tuning runs, with a focus on **LoRA/PEFT**. It emits a stable JSONL stream (“flight recorder”) and provides conservative, testable recommendations (“mechanic”) based on what was observed.

This manual documents the **canonical vNext release**:
- **Telemetry schema:** `gradience.vnext.telemetry/v1`
- **Core workflow:** `check → monitor → audit`
- **Key design stance:** *observe first, recommend conservatively, always verify with eval.*

> If you just want copy-paste commands, use **QUICK_REFERENCE.md**.  
> If you want the project overview + golden path, start with **README.md**.

---

## 1. Mental model

### 1.1 What Gradience does
Gradience is built around a simple loop:

1) **Pre-flight:** check whether your configuration looks risky *before* you burn GPU hours.  
2) **Run:** collect structured telemetry (JSONL).  
3) **Post-flight:** summarize what happened and provide conservative next-step recommendations.  
4) **Efficiency pass (LoRA):** audit adapters to detect oversized ranks and suggest safe compression trials.

### 1.2 What Gradience does *not* do
- It does **not** claim that spectral metrics predict final performance.
- It does **not** replace evaluation.
- It does **not** auto-tune your run end-to-end.

Gradience can tell you things like:
- “Train/test gap suggests memorization.”
- “Adapter seems over-provisioned; consider trying smaller rank.”
- “Your run looks consistent with typical stable behavior.”

It should *not* be treated as:
- “This will definitely improve accuracy.”
- “This configuration is optimal.”

---

## 2. Installation

### 2.1 Install from source (recommended for now)

```bash
git clone https://github.com/johntnanney/gradience.git
cd gradience
python -m pip install -U pip
pip install -e .
```

### 2.2 Common runtime dependencies (for HF/PEFT examples)
```bash
pip install torch transformers peft safetensors datasets
```

### 2.3 Device note
- `--device cpu` works anywhere.
- `--device cuda` requires a CUDA-enabled PyTorch build (typically Linux + NVIDIA GPU). On macOS, CUDA is generally not available.

---

## 3. Golden path (end-to-end)

This is the shortest “it works” loop. The same flow appears in the README/quick reference. fileciteturn2file1 fileciteturn2file0

### 3.1 Run a toy LoRA experiment (emits telemetry + PEFT dir)
```bash
# CPU
python examples/vnext/toy_lora_run.py --out runs/toy_run --device cpu

# GPU (Linux + CUDA torch)
python examples/vnext/toy_lora_run.py --out runs/toy_run --device cuda
```

Outputs:
```
runs/toy_run/
├── run.jsonl                      # telemetry
├── peft/                          # PEFT adapter artifacts
└── training/training_args.json    # training config snapshot
```

### 3.2 Check the config (pre-flight)
```bash
gradience check --task sst2 --peft-dir runs/toy_run/peft --training-dir runs/toy_run/training
```

### 3.3 Monitor the telemetry (post-flight summary)
```bash
gradience monitor runs/toy_run/run.jsonl --verbose
```

### 3.4 Audit the adapter (efficiency)
```bash
gradience audit --peft-dir runs/toy_run/peft --top-wasteful 10
```

### 3.5 Append the audit into telemetry and re-monitor
```bash
gradience audit --peft-dir runs/toy_run/peft --append runs/toy_run/run.jsonl
gradience monitor runs/toy_run/run.jsonl --verbose
```

---

## 4. CLI reference

### 4.1 `gradience check`
**Purpose:** Pre-flight validation for a LoRA config + training args.

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

### 4.2 `gradience monitor`
**Purpose:** Read a vNext JSONL file and summarize signals + recommendations.

```bash
gradience monitor <run.jsonl> [--verbose] [--json]
```

What monitor typically surfaces:
- Latest eval metrics (train/test/val)
- Gap ratio (when possible)
- LoRA audit stats (if present)
- Recommendations (policy engine output)

### 4.3 `gradience audit`
**Purpose:** Analyze a PEFT adapter directory for rank waste and compression suggestions.

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

---

## 5. Telemetry: what is logged

### 5.1 File format
Telemetry is **JSONL**: one JSON object per line.

Each event has:
- `schema` (must equal `gradience.vnext.telemetry/v1`)
- `ts` (unix timestamp)
- `run_id`
- `event`
- `step` (optional; may be null)
- plus an **event-specific payload**

### 5.2 Stable event names
The vNext contract guarantees the following stable event names:
- `run_start`
- `train_step`
- `eval`
- `metrics`
- `alert`
- `recommendation`
- `run_end`

### 5.3 Minimal metric keys (stable spine)
Gradience’s policy layer assumes these core metric keys when present:
- `loss`
- `ppl`
- `accuracy`
- `n`

Additional metrics should live in:
- `metrics(kind=...)` blocks, or
- `extras` dictionaries (forward compatible).

---

## 6. Privacy & safety defaults (important)

Gradience telemetry is local JSONL. Treat it as **sensitive**.

### 6.1 Redaction default
**TelemetryWriter redacts strings longer than 256 characters by default.**

This is a guardrail against accidentally logging prompts, dataset examples, or other raw text.

### 6.2 Opt-in text logging (dangerous)
To log raw text, you must explicitly opt in (e.g. `--telemetry-allow-text` in scripts that support it). If you do this, treat JSONL as sensitive and avoid uploading it to public places.

### 6.3 What Gradience does not log (by design)
- Training examples / prompts / labels (unless you explicitly opt in)
- Model weights
- Raw gradients

Gradience is intended to log **scalars + structured metadata**, not data.

---

## 7. Python API: TelemetryWriter

### 7.1 Minimal usage
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

### 7.2 Common logging patterns
- **Log eval metrics** at the end of each epoch or at a fixed cadence.
- **Log train_step** scalars periodically (loss / lr) if you want learning curves.
- Use `metrics(kind="...")` to attach structured metric blocks (e.g. `lora_audit`, `spectral`, `structural`).

---

## 8. Python API: TelemetryReader

TelemetryReader is the “other half”: it streams JSONL safely, validates schema, and produces a summary snapshot suitable for policy decisions.

### 8.1 Minimal usage
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

### 8.2 What `summarize()` returns
A **SignalSnapshot** (or dict with the same information) that includes:
- latest eval metrics by split
- gap ratios when train + test metrics exist
- attached metric blocks like `lora_audit` (if present)

This is what monitor/policy consumes.

---

## 9. Recommendations and alerts

### 9.1 Recommendation objects
Gradience’s policy engine emits **Recommendation** objects. At a minimum, these are intended to be:
- **human-readable**
- **testable** (easy to verify by running eval / changing one variable)
- **conservative** (avoid overclaiming)

Typical fields include:
- severity (`info` / `warning` / `error` / `critical`)
- code (stable identifier)
- message (what to do)
- why (rationale)
- confidence (0–1)
- evidence (structured context)

### 9.2 `config_ok`
Gradience may emit a `config_ok` informational recommendation **only if there are no actionable recommendations**. This keeps output less noisy.

---

## 10. LoRA audit: what it measures

### 10.1 What audit is trying to answer
LoRA audit is an **efficiency auditor**. It tries to answer:
- “Is this adapter rank obviously oversized?”
- “Are most layers using only a small fraction of rank capacity?”
- “What rank might cover most layers vs worst-case layers?”

### 10.2 Key metrics
Common audit metrics include:
- **Stable rank** of the adapter update (per layer + aggregate)
- **Utilization** = stable_rank / r (a rough “how much of rank is used”)
- **Energy rank** at 90% energy (`k@90%`) and its distribution (p50/p90)
- **Suggested ranks**
  - `suggested_r_global_median`: smallest r in {1,2,4,8,16,32} that covers median k@90%
  - `suggested_r_global_90`: smallest r in {1,2,4,8,16,32} that covers p90 k@90% (tail coverage)

Interpretation rule of thumb:
- `suggested_r_global_median` is a “most layers” hint
- `suggested_r_global_90` is a “tail safety” hint  
Always verify with eval.

### 10.3 Dominance ingredients (scaled update magnitude)
Audit also logs *scaled update magnitude* ingredients, such as:
- `delta_sigma_max_scaled_*`
- `delta_frob_norm_scaled_*`

These are useful for comparing “how large the update is” across adapters/configs without needing to compute ratios against the base weight (which is especially ambiguous under quantization).

---

## 11. QLoRA / quantized base models (caveat)

Gradience can audit LoRA **and** QLoRA adapters mechanically, because the adapters are still low-rank matrices learned in higher precision.

However, under QLoRA the adapter may implicitly do two jobs:
1) task adaptation  
2) quantization error compensation  

This makes “rank utilization → overprovisioned” interpretations less direct.

Practical guidance:
- Treat utilization/suggested rank as **compression hints**
- Verify with held-out eval after compression trials
- Log quantization metadata in `run_start.meta` so runs can be compared by quantization scheme

---

## 12. Troubleshooting

### 12.1 “gradience: command not found”
Make sure you installed the repo in editable mode:

```bash
pip install -e .
```

### 12.2 Missing dependencies (datasets, transformers, peft)
Install required packages:

```bash
pip install torch transformers peft safetensors datasets
```

### 12.3 “Torch not compiled with CUDA enabled”
You’re using a CPU-only PyTorch build. Either:
- run with `--device cpu`, or
- install a CUDA-enabled torch build (Linux + NVIDIA GPU)

### 12.4 Audit can’t find adapter weights
Gradience expects a PEFT output directory with:
- `adapter_config.json`
- `adapter_model.safetensors` (preferred) or equivalent adapter weights

---

## 13. Versioning policy

- The telemetry schema ID is **`gradience.vnext.telemetry/v1`**.
- Breaking schema changes require a bump to **`/v2`**.
- New optional fields and new `metrics(kind=...)` blocks are allowed in v1.

---

## Appendix A: Glossary

- **Telemetry (JSONL):** event stream of structured observations
- **Gap:** ratio between train and test (often used as a memorization signal)
- **Stable rank:** rank-like measure based on Frobenius norm and spectral norm
- **Energy rank (`k@90%`):** how many singular directions are needed to explain 90% of energy
- **Utilization:** stable_rank / r (rough “how much rank is used”)
- **PEFT:** Parameter-Efficient Fine-Tuning
- **LoRA:** Low-Rank Adaptation (learned low-rank update to weights)
- **QLoRA:** LoRA over a quantized base model

---

## Appendix B: Where to look in the repo

- `examples/vnext/` — runnable examples
- `gradience/vnext/SCHEMA.md` — telemetry contract
- `gradience/vnext/telemetry.py` — TelemetryWriter
- `gradience/vnext/telemetry_reader.py` — TelemetryReader
- `gradience/vnext/policy/` — recommendation engine
- `gradience/vnext/audit/` — LoRA auditor
