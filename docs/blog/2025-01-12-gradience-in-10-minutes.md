# Gradience in 10 Minutes: Flight Recorder + Mechanic for LoRA

*Published: 2025-01-12*

LoRA fine-tuning fails quietly in two boring ways:

1. You overfit (train/test gap grows)
2. You waste capacity (rank is oversized)

Gradience is a telemetry-first, conservative, non-invasive toolkit that helps you see those failure modes without pretending to predict outcomes. It's a simple workflow:

- **check** — validate config before you burn compute
- **monitor** — summarize a run from its telemetry
- **audit** — analyze the adapter for waste

---

## The Golden Path (copy-paste verbatim)
```bash
git clone https://github.com/johntnanney/gradience.git
cd gradience && pip install -e .
pip install torch transformers peft safetensors datasets

python examples/vnext/toy_lora_run.py --out runs/toy --device cpu

gradience check --task sst2 --peft-dir runs/toy/peft --training-dir runs/toy/training
gradience monitor runs/toy/run.jsonl --verbose
gradience audit --peft-dir runs/toy/peft --top-wasteful 10
```

That's it. No dashboards. No accounts. No secret sauce.

**Already have a PEFT adapter?** Skip the toy run:
```bash
gradience audit --peft-dir ./your-adapter --top-wasteful 10
```

---

## What You Get

### 1) A run log you can grep/parse

After the toy run you'll have:

- `run.jsonl` — telemetry log (the core artifact)
- `peft/` — adapter config + weights
- `training/` — training args snapshot

That `run.jsonl` is the core artifact. Everything else (monitor, later `audit --append`) is just different ways of reading/writing structured events into that log. (Privacy note: strings >256 chars are redacted by default; raw text logging requires explicit opt-in.)

---

### 2) A "flight recorder" summary (monitor)

Run:
```bash
gradience monitor runs/toy/run.jsonl --verbose
```

Example output (your numbers will vary):
```
========================================================================
GRADIENCE MONITOR
========================================================================
File: runs/toy/run.jsonl
Model:   distilbert-base-uncased
Dataset: glue/sst2
Profile: easy_classification

Latest eval signals:
  Train PPL: 1.88
  Test  PPL: 2.04
  Gap:       1.09x
  Train Acc: 68.8%
  Test  Acc: 50.0%

Diagnostics:
  Stable rank (mean): 1.47
  Utilization (mean): 18.4%

Recommendations (…):
  ...
```

**What to look for:**

- Gap > 1.5x: you're probably memorizing
- Test accuracy ≈ random chance: fix your task setup before tuning LoRA
- Utilization < 25%: you're likely over-provisioned (more in the audit section)

---

### 3) A "mechanic's report" (audit)

Now run:
```bash
gradience audit --peft-dir runs/toy/peft --top-wasteful 10
```

Example output (your numbers will vary):
```
========================================================================
GRADIENCE LoRA AUDIT
========================================================================
PEFT dir: runs/toy/peft
Config:   runs/toy/peft/adapter_config.json
Weights:  runs/toy/peft/adapter_model.safetensors

Summary:
  LoRA params: 294.9K
  Layers:      24
  Stable rank (mean):    1.47
  Utilization (mean):    18.4%
  Energy rank k@90% (p50/p90): 2.5/6

Most wasteful layers (lowest utilization, top 3):
  01. attn r=8   util=12.7%  sr=1.02  k@90%=1  ...
  02. attn r=8   util=12.8%  sr=1.02  k@90%=1  ...
  03. attn r=8   util=12.8%  sr=1.03  k@90%=1  ...
```

The "aha" most people get here: you're often paying for rank you're not using. Audit doesn't say "this will work." It says "this adapter update looks low-dimensional; consider compressing rank and verify with eval."

---

## What Gradience is NOT

- **Not AutoML** (it won't tune hyperparameters for you)
- **Not predictive** (it won't claim "this run will hit 83.2%")
- **Not a replacement for eval** (recommendations are hypotheses—verify on held-out data)

It instruments. You decide.

---

## Next Post

What the LoRA audit is actually telling you about your adapter—and how to use "suggested rank" conservatively (median vs worst-case tails) without turning it into superstition.

---

## Try It

Run the toy script. Then paste the machine-readable output somewhere (gist, issue, or reply):
```bash
gradience monitor runs/toy/run.jsonl --json
```

That single JSON blob is enough to reproduce the high-level story of your run.

---

**GitHub:** [github.com/johntnanney/gradience](https://github.com/johntnanney/gradience)
