# Gradience Bench Extension: Mistral-7B + GSM8K (Dense Step-by-Step Guide)

This guide turns Bench from **encoder + classification** into **decoder + generation** with **response-only loss** and **GSM8K exact-match evaluation**, while keeping the existing Bench protocol intact (probe → audit → compress → eval → aggregate).

> Goal: produce **per-seed** `bench.json/bench.md` and a **certifiable** `bench_aggregate.json/bench_aggregate.md` for **Mistral-7B + GSM8K**.

---

## Step 0 — Pre-flight checklist (do this first)

1. **Confirm you have GPU + bf16 support** (A40 is good).
2. **Confirm your audit is scalable on 7B** (LoRA SVD should be low-rank; avoid full 4k×4k SVD per module if possible).
3. **Disable full-model checkpoints** for 7B runs:
   - Save **adapter-only** (`adapter_model.safetensors`, `adapter_config.json`)
   - Prefer `save_strategy="no"` (or very sparse saving)

---

## Step 1 — Extend config schema minimally (YAML)

Create new config files in `gradience/bench/configs/`:

- `mistral_gsm8k_screening_seed42.yaml`
- `mistral_gsm8k_certifiable_seed42.yaml`
- `mistral_gsm8k_certifiable_seed123.yaml`
- `mistral_gsm8k_certifiable_seed456.yaml`

Use this minimal structure (adapt to your existing schema keys):

```yaml
model:
  name: mistralai/Mistral-7B-v0.1
  type: causal_lm              # NEW: seqcls vs causal_lm
  torch_dtype: bf16
  gradient_checkpointing: true
  use_cache: false

task:
  dataset: gsm8k
  subset: main
  profile: gsm8k_causal_lm     # NEW: routes to GSM8K pipeline
  eval_max_samples: 500        # important for stable deltas

  generation:
    max_new_tokens: 128
    do_sample: false
    temperature: 0.0
    num_beams: 1

  probe_gate:
    metric: exact_match        # NEW
    min_value: 0.15            # placeholder; calibrate after screening

training:
  max_steps: 1500              # screening might be 200–500
  per_device_train_batch_size: 1
  per_device_eval_batch_size: 8
  gradient_accumulation_steps: 16
  learning_rate: 1.0e-4
  warmup_ratio: 0.03
  weight_decay: 0.0
  logging_steps: 10
  save_strategy: "no"          # strongly recommended for 7B
  report_to: "none"

lora:
  r_probe: 32
  lora_alpha: 64
  lora_dropout: 0.05
  target_modules: ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]

run_type: "mistral_gsm8k_certifiable_v0.1_seed42"
seed: 42
```

**Notes**
- Keep `eval_max_samples >= 500` if you want the ±2.5% safety rule to mean anything.
- You can start with a **screening** config at `max_steps: 200–500` to prove the pipeline.

---

## Step 2 — Add task profile routing (small abstraction)

Create:

- `gradience/bench/task_profiles/base.py`
- `gradience/bench/task_profiles/seqcls_glue.py` (wrap existing SST-2/QNLI logic)
- `gradience/bench/task_profiles/gsm8k_causal_lm.py` (new)

Define a minimal interface (keep it boring):

```python
# gradience/bench/task_profiles/base.py
from typing import Protocol, Dict, Any

class TaskProfile(Protocol):
    name: str
    primary_metric: str  # "accuracy" or "exact_match"

    def load(self, cfg: Dict[str, Any]) -> Dict[str, Any]: ...
    def tokenize(self, raw_ds, tokenizer, cfg: Dict[str, Any]): ...
    def build_trainer(self, model, tokenizer, tokenized_ds, cfg: Dict[str, Any], callbacks): ...
    def evaluate(self, model, tokenizer, tokenized_ds, cfg: Dict[str, Any]) -> Dict[str, Any]: ...
    def probe_gate(self, probe_eval: Dict[str, Any], cfg: Dict[str, Any]): ...
```

Then in your bench `protocol.py`, replace hard-coded assumptions like “accuracy from logits” with:

- `profile = get_profile(cfg)`
- `eval_metrics = profile.evaluate(...)`
- `probe_ok, gate_context = profile.probe_gate(eval_metrics, cfg)`

Add `get_profile(cfg)` in something like `gradience/bench/task_profiles/__init__.py`:

```python
def get_profile(cfg):
    name = cfg["task"]["profile"]
    if name == "gsm8k_causal_lm":
        from .gsm8k_causal_lm import GSM8KCausalLMProfile
        return GSM8KCausalLMProfile()
    # default existing:
    from .seqcls_glue import GlueSeqClsProfile
    return GlueSeqClsProfile()
```

---

## Step 3 — Implement GSM8K formatting + response-only masking

Create `gradience/bench/datasets/gsm8k.py` (or put inside the task profile).

### 3.1 Load dataset

Use `datasets`:

- dataset: `gsm8k`
- subset: `main`

### 3.2 Format each example (prompt + answer)

Canonical format:

```text
Question: {question}
Answer:
{answer}
```

> Using GSM8K’s `answer` field verbatim is simplest for a first evidence point.

### 3.3 Tokenize with **response-only loss**

For each example:

1. Tokenize prompt → `prompt_ids`
2. Tokenize answer → `answer_ids`
3. Concatenate:
   - `input_ids = prompt_ids + answer_ids`
   - `labels = [-100]*len(prompt_ids) + answer_ids`
4. `attention_mask = [1]*len(input_ids)`
5. Truncate carefully:
   - cap prompt length first so you don’t delete the answer entirely

Pseudo-code:

```python
prompt = f"Question: {q}\nAnswer:\n"
answer = a

prompt_ids = tok(prompt, add_special_tokens=False).input_ids
answer_ids = tok(answer, add_special_tokens=False).input_ids

input_ids = prompt_ids + answer_ids
labels = [-100]*len(prompt_ids) + answer_ids
```

### 3.4 Collator that preserves labels

Write a tiny collator that pads:

- `input_ids`
- `attention_mask`
- `labels` (pad with `-100`)

Do **not** rely on a collator that re-creates labels.

---

## Step 4 — Model + Trainer for causal LM (Mistral)

In `gsm8k_causal_lm.py` profile:

- Load with `AutoModelForCausalLM.from_pretrained`
- Set:
  - `model.gradient_checkpointing_enable()` if configured
  - `model.config.use_cache = False` during training
- Apply PEFT LoRA config to model
- Use `Trainer` with:
  - your custom collator
  - `remove_unused_columns=False` (so labels survive)
  - careful batch sizes (7B)

---

## Step 5 — Evaluation via generation + exact match

Create `gradience/bench/eval/gsm8k.py`:

### 5.1 Deterministic generation

Use:

- `do_sample=False`
- `temperature=0.0`
- `num_beams=1`
- `max_new_tokens` from config

### 5.2 Answer extraction

Implement:

- Prefer text after `####`
- Fallback: last number regex

Then:

- `exact_match = (pred_num == gold_num)`

### 5.3 Return eval dict

Return something like:

```json
{
  "exact_match": 0.23,
  "n": 500,
  "generation_max_new_tokens": 128
}
```

---

## Step 6 — Make Bench metric-agnostic (small report change)

In your canonical bench reports, add:

- `primary_metric_name`
- and use it for probe baseline + deltas

Example fields:

```json
"primary_metric": {"name": "exact_match", "probe_value": 0.23}
```

Or minimally:
- keep `accuracy` but label it in the report as `exact_match` (less ideal).

Update verdict computation:

- `delta = variant_metric - probe_metric`
- safety policy applies to `delta` in the same way

---

## Step 7 — Probe gate for GSM8K (task-specific)

Do **not** reuse SST-2’s `0.75`.

Instead, make the gate config-driven:

```yaml
probe_gate:
  metric: exact_match
  min_value: 0.15
```

Then in profile:

```python
metric = cfg["task"]["probe_gate"]["metric"]
minv = cfg["task"]["probe_gate"]["min_value"]
ok = eval_metrics[metric] >= minv
```

Tip: start with a low gate for screening, then raise it once you see stable numbers.

---

## Step 8 — Disk + checkpoint discipline for pods

For 7B:

- Ensure `save_strategy="no"` or extremely sparse
- Ensure adapter-only save path is stable (no deep checkpoint hunting)
- Ensure your “rank heterogeneity check” looks for adapter weights at:
  - `variant_dir/peft/adapter_model.safetensors` OR latest checkpoint adapter

If your bench currently saves into `checkpoint-*`, add a “latest checkpoint” resolver (you already did this for earlier outsider drills).

---

## Step 9 — Test plan (don’t burn GPU time)

### 9.1 Unit tests (CPU)

Add tests for:

- response-only masking:
  - prompt labels are `-100`
  - answer labels are not `-100`
- answer extraction:
  - handles `#### 42`
  - handles fallback last-number
- dataset formatter returns non-empty prompt/answer

### 9.2 Tiny GPU smoke (A40)

Create a smoke config:

- `train_max_samples: 64`
- `eval_max_samples: 50`
- `max_steps: 20–50`

Confirm:

- run completes
- telemetry exists
- eval returns `exact_match`
- audit runs (and does not OOM)

Only then run screening/certifiable.

---

## Step 10 — Run protocol (screening → certifiable → aggregate)

### 10.1 Screening (1 seed)

```bash
python -m gradience.bench.run_bench \
  --config gradience/bench/configs/mistral_gsm8k_screening_seed42.yaml \
  --output bench_runs/mistral_gsm8k_screen_seed42
```

Confirm:
- probe passes gate
- audit produces suggestions
- variants run
- eval produces exact_match
- canonical report exists

### 10.2 Certifiable (3 seeds)

```bash
python -m gradience.bench.run_bench \
  --config gradience/bench/configs/mistral_gsm8k_certifiable_seed42.yaml \
  --output bench_runs/mistral_gsm8k_seed42

python -m gradience.bench.run_bench \
  --config gradience/bench/configs/mistral_gsm8k_certifiable_seed123.yaml \
  --output bench_runs/mistral_gsm8k_seed123

python -m gradience.bench.run_bench \
  --config gradience/bench/configs/mistral_gsm8k_certifiable_seed456.yaml \
  --output bench_runs/mistral_gsm8k_seed456
```

### 10.3 Aggregate

```bash
python gradience/bench/aggregate.py \
  bench_runs/mistral_gsm8k_seed42 \
  bench_runs/mistral_gsm8k_seed123 \
  bench_runs/mistral_gsm8k_seed456 \
  --output bench_runs/mistral_gsm8k_agg
```

Your citable artifacts:
- `bench_runs/mistral_gsm8k_agg/bench_aggregate.json`
- `bench_runs/mistral_gsm8k_agg/bench_aggregate.md`

---

## Step 11 — What to record for credibility

When you publish:
- exact git tag/commit
- GPU type (A40), driver/CUDA versions
- transformers/torch/peft versions
- your exact config YAML(s)
- aggregate JSON/MD

---

## Step 12 — Known “gotchas” (save yourself pain)

- **If exact_match is noisy**, increase `eval_max_samples`.
- **If training OOMs**, reduce sequence length / batch size, increase grad accumulation, enable gradient checkpointing.
- **If audit is too slow**, you need a low-rank SVD path (r×r-based) for large models.
- **If disk explodes**, disable saving checkpoints and save adapters only.

---

## Quick summary

You are adding:
1) a **task profile** for causal LM generation  
2) **response-only loss** masking  
3) **generation-based eval** for GSM8K exact match  
4) small report generalization so Bench isn’t hardcoded to “accuracy”

Everything else (probe→audit→compress→eval→aggregate) stays the same.

