# GPU-Side Inference Spec v0.1

**Project:** Benchmark Accuracy as Measurement: Reliability and Tolerance Schedules for LLM Evaluation
**Repository path:** `papers/benchmark_reliability_study/`
**Spec version:** v0.1 (initial draft)
**Date:** 2026-04-25
**Status:** ready for execution after RunPod allocation
**Companion spec:** `SPEC_CPU_v0_2.md` (CPU pipeline; defines the input/output data contracts this GPU work consumes/produces)
**Pre-registration tag dependency:** `v1_1_1_LOCKED`

---

## 1. Scope

This spec defines the GPU-side inference work that produces the raw run outputs the CPU pipeline normalizes and analyzes. It is the Phase 4 of the §16-task inventory in SPEC_CPU_v0_2.

The GPU side is **inference-only**: it reads the locked manifests + locked prompts + pinned HF datasets, runs three open language models against ~672 conditions, and writes per-condition output directories conformant to the data contracts in SPEC_CPU_v0_2 §5.2–5.4.

**Out of scope:** model fine-tuning, prompt construction (prompts are locked at v1_1_1), few-shot exemplar selection (locked at `preregistration/appendices/fewshot_draws_LOCKED.json`), benchmark-item curation (datasets pinned at HF revisions). All of these are upstream of this spec.

**Architectural principle:** the GPU side must be a contract-conformant black box. The CPU pipeline does not know how outputs are produced — it knows what schema they conform to. Replacing the GPU implementation (e.g., switching to lm-evaluation-harness later) must not require any CPU-side change.

---

## 2. Hardware

### 2.1 GPU class

**Recommended:** RTX 4090 (24 GB VRAM) on RunPod.

**Why this choice (not A4000 or A100):**

The three primary models max at 1.5B parameters; in bfloat16, weights total ~3 GB. With KV cache and activations for batch inference, working memory stays under 8 GB. So memory is not the binding constraint — bf16 throughput is.

| GPU | VRAM | bf16 tensor cores | RunPod $/hr (community) | 30-hr pass cost |
|---|---|---|---|---|
| RTX A4000 | 16 GB | mid | ~$0.20 | $6 |
| **RTX 4090** | **24 GB** | **high** | **~$0.40** | **$12** |
| A100 40GB | 40 GB | very high | ~$1.50 | $45 |

The 4090 is the throughput-per-dollar sweet spot for our scale. The A4000 is workable but ~2× slower, pushing total inference closer to 50-60 GPU-hours. The A100 is overpowered for 1.5B models — its bf16 advantage doesn't translate proportionally at small model size, so cost per condition is worse.

If we later run the optional Mistral-7B extension (deferred per pre-registration §14.3), the 4090's 24 GB still accommodates it (Mistral-7B-bf16 ≈ 14 GB + KV cache headroom). Same hardware choice covers both primary and extension cases without an upgrade.

### 2.2 Pod configuration

- GPU: 1× RTX 4090 (24 GB)
- vCPU: ≥ 8 cores (for tokenization throughput)
- System RAM: ≥ 32 GB
- Disk: ≥ 100 GB ephemeral + ≥ 50 GB persistent volume mounted at `/workspace`
- Network: standard RunPod allocation; HF Hub downloads sustained at ~50-200 MB/s

### 2.3 Cost ceiling

Primary pass budget: **$15** (allows for 30-40 GPU-hours at community pricing). Hard cap for cost protection: $30.

Optional 7B extension (post-primary, budget-permitting): additional $20-30 for a reference subset. Not part of v1.1.1 commitment.

---

## 3. Software environment

### 3.1 Base image

RunPod's PyTorch 2.4 / Python 3.11 / CUDA 12.4 image:

```
runpod/pytorch:2.4.0-py3.11-cuda12.4.0-devel-ubuntu22.04
```

(or the closest equivalent at execution time; pin the exact tag in the runbook.)

### 3.2 Python dependencies

Install in this order after pod start:

```bash
pip install --upgrade pip
pip install \
  transformers==4.46.0 \
  accelerate==1.1.0 \
  sentencepiece==0.2.0 \
  datasets==4.8.0 \
  huggingface_hub==1.12.0 \
  pyarrow==15.0.2 \
  pandas==2.2.3 \
  pyyaml==6.0.2 \
  jsonschema==4.23.0 \
  numpy==1.26.4 \
  scipy==1.13.1 \
  tqdm==4.67.0
```

Pin to exact patch versions; commit a `requirements.gpu.lock` at execution time recording `pip freeze` output. Any deviation from these pins is a `deviations.md` entry per CPU §11.

### 3.3 Hugging Face authentication

Some datasets require HF authentication (Winogrande gates behind community license). Set environment variable at pod start:

```bash
export HF_TOKEN=<read-only-token>
```

The token must be read-only (HF profile → Settings → Access Tokens → "Read" type). Document the token's source in the runbook; the token itself is not committed to the repository.

Before run start, verify license acceptance for each gated dataset by attempting a minimal `load_dataset(name, revision=hash)` call.

### 3.4 Hugging Face cache

Mount the persistent volume at `/workspace` to `/root/.cache/huggingface` so model and dataset downloads survive pod restarts:

```bash
mkdir -p /workspace/hf_cache
ln -sfn /workspace/hf_cache /root/.cache/huggingface
```

Estimated cache size: ~10 GB (3 models × ~3 GB + dataset metadata). The persistent volume retains this between sessions.

### 3.5 Repository

Clone `gradience` at the locked tag:

```bash
cd /workspace
git clone https://github.com/<repo>/gradience.git
cd gradience
git checkout v1_1_1_LOCKED
git submodule update --init --recursive
```

The GPU script lives at `papers/benchmark_reliability_study/scripts/gpu_inference.py` (to be implemented per this spec).

---

## 4. Architecture: GPU/CPU contract

### 4.1 Inputs (read by GPU side)

| Input | Path | Source |
|---|---|---|
| Conditions manifest (primary) | `manifests/conditions_primary.csv` | CPU §8.2; produced by `01_build_manifests.py` |
| Conditions manifest (secondary) | `manifests/conditions_gsm8k.csv` | same |
| Few-shot manifest | `manifests/fewshot_manifest.csv` | CPU §8.3; produced by `02_draw_fewshot_examples.py` |
| Few-shot lock | `preregistration/appendices/fewshot_draws_LOCKED.json` | same |
| Prompt files | `prompts/<benchmark_id>/<prompt_id>.txt` | locked at v1.1 |
| Prompt manifest | `manifests/prompt_manifest.csv` | CPU §8.4; produced by `03_validate_prompts.py` |
| Configs | `configs/*.yaml` | locked at v1.1.1 |
| Schemas | `schemas/*.schema.json` | for output validation |

### 4.2 Outputs (written by GPU side)

For every condition with `condition_status == "pending"` in a manifest:

```
runs/raw/{run_id}/
  run_metadata.json     # SPEC_CPU §5.2 schema
  item_outputs.jsonl    # SPEC_CPU §5.3 schema (G&P scoring rules)
  item_scores.jsonl     # SPEC_CPU §5.4 schema (LL scoring rules)
```

Where `run_id == condition.condition_id` (must equal — checked at normalize time).

Exactly one of `item_outputs.jsonl` or `item_scores.jsonl` per run, determined by the condition's `scoring_rule_id`.

### 4.3 Atomicity

Each condition's output directory is **atomic**: either the directory exists with all required files, or it does not exist. No partial directories.

Implementation: write to `runs/raw/.tmp/{run_id}/`, validate every file against its schema, then `mv .tmp/{run_id} {run_id}`. If the validate step fails, leave the `.tmp` directory and emit a failure log; subsequent runs see no `runs/raw/{run_id}/` and re-run the condition.

---

## 5. Inference backend: custom Python loop

### 5.1 Overall structure

`scripts/gpu_inference.py` (single-file driver):

```
1. Parse args (manifest path, output dir, optional --resume, optional --filter)
2. Load configs via gradience_study.config.load_config()
3. Load conditions manifest; filter to pending only (or to --filter scope)
4. Group conditions by model_id (so we load each model exactly once)
5. For each model_id:
     a. Load model + tokenizer at pinned hf_revision
     b. Move to GPU in bfloat16
     c. For each condition in this model's group:
          i. Resume check: skip if runs/raw/{run_id}/run_metadata.json exists
          ii. Load benchmark items at pinned dataset_version_hash
          iii. Render prompt with few-shot examples (read from fewshot_manifest)
          iv. Dispatch to LL or G&P scorer based on scoring_rule_id
          v. Write outputs atomically to runs/raw/.tmp/{run_id}/
          vi. Validate outputs against schemas
          vii. Atomic rename to runs/raw/{run_id}/
     d. Free model from GPU memory
6. Emit summary (n completed, n failed, total time)
```

Total implementation: ~400-600 lines.

### 5.2 Prompt rendering

For each item, render the prompt template by substituting placeholders:

```python
def render_prompt(template_text: str, item: dict, fewshot_items: list[dict],
                  benchmark_spec, prompt_spec) -> str:
    fewshot_block = render_fewshot_block(fewshot_items, benchmark_spec)
    choices_block = render_choices_block(item, benchmark_spec)
    return (template_text
            .replace("{{fewshot_examples}}", fewshot_block)
            .replace("{{question}}", item[benchmark_spec.question_field])
            .replace("{{choices}}", choices_block)
            .replace("{{answer_instruction}}", get_answer_instruction(prompt_spec))
            )
```

Where:
- `fewshot_block` formats the few-shot exemplars as "Question: ...\nA) ...\nB) ...\nAnswer: <gold letter>\n\n" repeated k times
- `choices_block` formats the item's choices as "A) <text>\nB) <text>\nC) <text>\nD) <text>" (or analogous for non-4-choice benchmarks)
- For benchmarks where the prompt template doesn't use a placeholder, the substitution is a no-op (the placeholder doesn't appear in the template)

The `rendered_prompt_hash` field in each output row is `sha256(rendered_prompt)` — this gives the CPU pipeline a stable per-item prompt fingerprint for the reproducibility trace.

### 5.3 LL scoring (length-normalized log-likelihood)

For each (model, item, choices) triple:

```python
def score_ll_norm(model, tokenizer, prompt: str, choices: list[str]) -> dict:
    """Compute length-normalized LL for each choice's letter continuation.

    Our locked prompts present lettered choices in the body and end with
    "Answer:". We score the continuation likelihood of each letter token
    (with leading space).
    """
    # For each choice index, the candidate continuation is " A", " B", ... or
    # " " + ascii_uppercase[i]
    candidates = [f" {chr(ord('A') + i)}" for i in range(len(choices))]
    
    prompt_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
    
    scores = {}
    token_counts = {}
    for letter_idx, candidate in enumerate(candidates):
        cand_ids = tokenizer.encode(candidate, add_special_tokens=False)
        # Compute log P(candidate | prompt)
        full_ids = torch.cat([prompt_ids, torch.tensor([cand_ids]).to(model.device)], dim=1)
        with torch.no_grad():
            logits = model(full_ids).logits  # [1, seq_len, vocab]
        # Sum log-probs of candidate tokens (positions are [prompt_len-1 ... prompt_len+len(cand_ids)-2])
        prompt_len = prompt_ids.shape[1]
        log_probs = F.log_softmax(logits[0, prompt_len-1 : prompt_len-1 + len(cand_ids)], dim=-1)
        token_log_probs = [log_probs[t, cand_ids[t]].item() for t in range(len(cand_ids))]
        sum_log_prob = sum(token_log_probs)
        normalized = sum_log_prob / len(cand_ids)  # length-normalized
        choice_letter = chr(ord('A') + letter_idx)
        scores[choice_letter] = normalized
        token_counts[choice_letter] = len(cand_ids)
    
    selected = max(scores, key=scores.get)
    return {
        "choices": [chr(ord('A') + i) for i in range(len(choices))],
        "choice_scores": scores,
        "choice_token_counts": token_counts,
        "normalization": "length_normalized",
        "selected_answer": selected,
    }
```

Per-item output emitted to `item_scores.jsonl` per CPU §5.4 schema.

**Choice count handling:**
- ARC-Challenge: typically 4 choices (occasionally 3 or 5; use actual count from item)
- HellaSwag: 4 choices
- TruthfulQA-MC: variable (4-13 from `mc1_targets`); use actual count
- Winogrande: 2 choices
- MMLU: 4 choices

The candidate letters are generated dynamically from the actual choice count; no hardcoded "ABCD".

### 5.4 G&P scoring (generate and parse)

For each (model, item) pair:

```python
def score_generate_parse(model, tokenizer, prompt: str, gold_answer: str,
                          benchmark_spec, max_new_tokens: int = 32) -> dict:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,           # greedy
            temperature=0,             # deterministic
            num_beams=1,
            pad_token_id=tokenizer.eos_token_id,
        )
    # Strip the prompt prefix to get just the generation
    generated_ids = outputs[0, inputs.input_ids.shape[1]:]
    raw_generation = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    parse_status, parsed_answer = parse_answer(
        raw_generation, benchmark_spec, gold_answer
    )
    is_correct = (parsed_answer == gold_answer) if parse_status == "parsed" else False
    
    return {
        "raw_generation": raw_generation,
        "parsed_answer": parsed_answer,
        "parse_status": parse_status,
        "is_correct": is_correct,
        "generation_length_tokens": int(generated_ids.shape[0]),
    }
```

**Parse strategies** (per benchmark, locked at v1.1.1 in `scoring_rules.yaml`):

For constrained-choice benchmarks (regex `benchmark_specific_regex`):
- Extract first standalone uppercase letter A-Z from `raw_generation`. If found and within the choice space → `parsed`; if no letter → `unparseable`; if multiple distinct letters → `multiple_answers`; if letter outside choice space → `invalid_choice`; if generation is empty → `empty`.

For GSM8K strict (`exact_match_final_number`):
- Extract the final integer in the generation matching `\d+(?:\.\d+)?$` (numeric, optionally decimal). If found and equals gold (after normalization) → `parsed`; if no number → `unparseable`; if number doesn't match → `parsed` but `is_correct=False`.

For GSM8K permissive (`permissive_number_regex`):
- Extract any integer or formatted number from the last 5 lines of the generation matching `[\d,]+(?:\.\d+)?` (allows commas, decimal points). Same semantics for status.

Parse-status enum is exactly:
```
parsed | unparseable | empty | multiple_answers | invalid_choice | runtime_error
```

Per CPU §5.3 schema.

### 5.5 Generation parameters

Locked in `scoring_rules.yaml`:

| scoring_rule_id | max_new_tokens | decoding | temperature |
|---|---|---|---|
| ll_norm | n/a (no generation) | n/a | n/a |
| generate_parse | 32 | greedy | 0 |
| generate_parse_strict | 256 | greedy | 0 |
| generate_parse_permissive | 256 | greedy | 0 |

These values are the locked contract; the GPU script reads them from `scoring_rules.yaml` rather than hardcoding.

---

## 6. Per-condition execution detail

### 6.1 Loading model and dataset

For a given condition row with `model_id` and `benchmark_id`:

```python
model_spec = merged.models[model_id]
benchmark_spec = merged.benchmarks[benchmark_id]

# Load model exactly once per (run, model_id); reuse across conditions
model = AutoModelForCausalLM.from_pretrained(
    model_spec.hf_name,
    revision=model_spec.hf_revision,
    torch_dtype=torch.bfloat16,
    device_map="cuda",
)
tokenizer = AutoTokenizer.from_pretrained(
    model_spec.hf_name,
    revision=model_spec.hf_revision,
)

# Load dataset items at pinned revision
ds_kwargs = dict(
    revision=benchmark_spec.dataset_version_hash,
    split=benchmark_spec.eval_split,
)
if benchmark_spec.benchmark_id == "mmlu_panel":
    ds_kwargs["name"] = "all"  # see CPU IMPLEMENTATION_DEVIATIONS.md D-X
else:
    ds_kwargs["name"] = benchmark_spec.hf_config

items = load_dataset(benchmark_spec.hf_dataset, **ds_kwargs)
if benchmark_spec.benchmark_id == "mmlu_panel":
    items = items.filter(lambda x: x["subject"] in benchmark_spec.subjects)
```

### 6.2 Few-shot exemplar lookup

Read `manifests/fewshot_manifest.csv` for the benchmark + subject + seed; collect the corresponding exemplar item IDs; load those items from the benchmark's `fewshot_source_split`.

```python
fewshot_items = lookup_fewshot_for_condition(
    condition_id=row.condition_id,
    benchmark_spec=benchmark_spec,
    seed_id=row.seed_id,
    subject_id=row.subject_id,
    fewshot_manifest_path="manifests/fewshot_manifest.csv",
)
```

Few-shot items must be loaded from the `fewshot_source_split` (typically `train` or `dev`), not the eval split. The CPU pipeline already enforces this at draw time; the GPU side just looks up the IDs.

### 6.3 Prompt rendering and per-item scoring

For each item in the eval set:

```python
rendered_prompt = render_prompt(template_text, item, fewshot_items,
                                  benchmark_spec, prompt_spec)
prompt_text_hash = sha256(template_text).hexdigest()
rendered_prompt_hash = sha256(rendered_prompt).hexdigest()

if scoring_rule.applies_to == "ll_norm":
    output = score_ll_norm(model, tokenizer, rendered_prompt, choices)
    # ... emit to item_scores.jsonl row
elif "generate_parse" in scoring_rule.id:
    output = score_generate_parse(model, tokenizer, rendered_prompt,
                                     gold, benchmark_spec)
    # ... emit to item_outputs.jsonl row
```

### 6.4 Run metadata

After all items in the condition are scored:

```python
run_metadata = {
    "run_id": condition.condition_id,
    "condition_id": condition.condition_id,
    "model_id": model_spec.model_id,
    "hf_name": model_spec.hf_name,
    "hf_revision": model_spec.hf_revision,
    "benchmark_id": benchmark_spec.benchmark_id,
    "subject_id": condition.subject_id,
    "prompt_id": condition.prompt_id,
    "seed_id": condition.seed_id,
    "fewshot_k": condition.fewshot_k,
    "scoring_rule_id": condition.scoring_rule_id,
    "task_type": benchmark_spec.task_type,
    "num_items_expected": int(condition.expected_num_items),
    "num_items_completed": len(scored_items),
    "inference_backend": "custom_transformers_v0.1",
    "python_version": "3.11.x",
    "transformers_version": transformers.__version__,
    "torch_version": torch.__version__,
    "lm_eval_version": None,
    "device": "cuda",
    "dtype": "bfloat16",
    "started_at": iso_utc(start_time),
    "finished_at": iso_utc(end_time),
    "status": "complete" if len(scored_items) == condition.expected_num_items else "partial",
    "notes": "",
}
```

---

## 7. Resumability and error handling

### 7.1 Resume protocol

On each invocation, before processing a condition, check:

```python
existing = Path(f"runs/raw/{condition.condition_id}/run_metadata.json")
if existing.exists():
    skip(condition.condition_id)
    continue
```

This makes the script idempotent: re-running picks up where the previous run left off.

### 7.2 Per-condition failure semantics

If a single condition fails (model OOM, dataset load error, schema validation rejection):

1. Log full traceback to `runs/failures.jsonl` (one JSON line per failure: condition_id, exception_class, exception_message, traceback, timestamp).
2. Do not write `runs/raw/{run_id}/`. The failed condition has no directory; the next run will retry it.
3. Continue to the next condition. Do not abort the whole pass.

### 7.3 OOM recovery

If `torch.cuda.OutOfMemoryError` is caught:
1. Drop batch size by half for this condition; retry.
2. If batch size 1 OOMs, log failure, skip condition, continue.

In practice, all three primary models on a 4090 should never OOM. But the recovery path exists for safety.

### 7.4 Whole-script crashes

If the script crashes (out of process control), pod restart picks up where it left off via the resume protocol.

---

## 8. Batch sizing

Inference can batch multiple items per forward pass. Batch sizing depends on prompt length and choice count.

Recommended starting batch sizes (tune at runtime):

| Benchmark | LL scoring batch | G&P scoring batch |
|---|---|---|
| arc_challenge | 16 | 8 |
| hellaswag | 8 (longer prompts) | 4 |
| truthfulqa_mc | 16 | 8 |
| mmlu_panel | 16 | 8 |
| winogrande | 16 | 8 |
| gsm8k | n/a (no LL) | 4 (256-token gen) |

The script monitors actual GPU memory after first batch and adjusts. If memory headroom > 20%, doubles batch size for next batch. If OOM, halves and retries.

---

## 9. Pre-flight checks

Before the main loop, verify:

1. **All model weights loadable.** For each model_id in models.yaml, attempt `AutoModelForCausalLM.from_pretrained(name, revision=hash, torch_dtype=torch.bfloat16, device_map="cuda")`, then immediately free. Verify weight loading completes without errors.

2. **All dataset splits loadable.** For each benchmark_id, attempt `load_dataset(name, hf_config, split=split, revision=hash)` for both `eval_split` and `fewshot_source_split` (where applicable). Verify item counts approximately match `expected_num_items`.

3. **All prompt files readable and SHA-256 matches.** For each prompt_id, read `prompts/<benchmark>/<prompt_id>.txt`, compute SHA-256, verify it matches the locked `content_hash` in `prompts.yaml`.

4. **Few-shot lock readable.** Load `preregistration/appendices/fewshot_draws_LOCKED.json`, verify config_hash matches the current locked configs.

5. **Schemas readable.** Load each of the 6 JSON schemas; verify they parse.

6. **Output directory writable.** Create `runs/raw/.tmp/`; write a tiny test file; remove.

If any pre-flight check fails, exit before the main loop. The pre-flight cost is ~5-10 minutes but catches most production-blocking issues before they consume GPU time.

---

## 10. Run-time monitoring

### 10.1 Per-condition timing log

For each condition, log to stdout:

```
[2026-04-25T18:23:14Z] [condition_001/672] arc_challenge__pythia_410m__P1_original__s42__ll_norm: 1172 items, 47.3s, 24.8 items/s
```

### 10.2 Periodic summary

Every 25 conditions, print a summary:
- Conditions completed: N / 672
- Total elapsed: HH:MM:SS
- Estimated remaining: HH:MM:SS (linear extrapolation)
- Estimated total cost: $X (at locked $/hr)
- Failures: N (with condition IDs listed)

### 10.3 Status file

Maintain `runs/status.json` updated after each condition:
```json
{
  "completed": [...condition_ids...],
  "failed": [...condition_ids...],
  "pending": [...condition_ids...],
  "current_condition": "...",
  "started_at": "...",
  "last_updated": "...",
  "estimated_total_hours": ...
}
```

---

## 11. Output handling and upload

### 11.1 During run

Outputs land at `runs/raw/<run_id>/` on the pod's local volume. Total size at completion: ~50-200 MB across all 672 conditions.

### 11.2 Periodic checkpoint upload

Every 100 conditions completed (or every hour, whichever comes first):

```bash
cd /workspace/gradience/papers/benchmark_reliability_study
tar -czf runs_checkpoint_$(date +%s).tar.gz runs/raw/ runs/status.json runs/failures.jsonl
# Upload to RunPod's object store or via rclone to user's destination
```

### 11.3 Final upload

After the run completes, upload the final tar.gz to the user's workstation. The CPU pipeline's `04_normalize_outputs.py` consumes `runs/raw/` directly after extraction.

### 11.4 Pod teardown

Once the upload is verified by the user (manual or scripted check), tear down the pod. The persistent volume `/workspace/hf_cache` can stay for future runs; the ephemeral volume is automatically released.

---

## 12. Verification protocol

Before declaring the GPU pass complete:

1. **Sample verification.** Pick 5 random conditions from `runs/raw/`. For each, manually verify:
   - `run_metadata.json` validates against `schemas/run_metadata.schema.json`
   - The `item_outputs.jsonl` or `item_scores.jsonl` validates against its schema
   - `num_items_completed` in metadata matches the line count of the JSONL

2. **CPU normalizer dry-run.** On the user's workstation, after pulling `runs/raw/` back:
   ```bash
   python scripts/04_normalize_outputs.py \
     --conditions manifests/conditions_primary.csv \
     --raw-dir runs/raw/ \
     --schemas-dir schemas/ \
     --out runs/normalized/item_level_primary.parquet
   ```
   Verify exit code 0; verify normalized parquet row count equals sum of `expected_num_items` across complete conditions.

3. **Reproducibility trace step.** Run `98_reproducibility_trace.py` per CPU §13 with the GPU outputs in place. Verify trace status `pass`.

If any verification fails, do not proceed to Phase 5 (analysis). Re-run failed conditions or surface the failure for diagnosis.

---

## 13. Cost and time estimates

### 13.1 Time per condition (rough order-of-magnitude on RTX 4090, bf16)

| Benchmark | LL forward passes | LL time | G&P generation | G&P time |
|---|---|---|---|---|
| arc_challenge | 4700 | 30-60 s | 1172 × 32 tok | 60-120 s |
| hellaswag | 40000 | 5-8 min | 10042 × 32 tok | 8-15 min |
| truthfulqa_mc | 5000 | 30-60 s | 817 × 32 tok | 30-60 s |
| mmlu_panel | 600 (per subj) | 5-15 s | 100 (per subj) × 32 tok | 5-15 s |
| winogrande | 2500 | 20-40 s | 1267 × 32 tok | 30-60 s |
| gsm8k | n/a | n/a | 1319 × 256 tok | 15-30 min |

### 13.2 Total estimate

- Primary tier (600 conditions): ~22-32 GPU-hours
- Secondary tier GSM8K (72 conditions): ~3-7 GPU-hours
- Total: **25-40 GPU-hours**
- Cost at RTX 4090 community pricing ($0.40/hr): **$10-16**
- Time wall-clock: ~36-48 hours of pod uptime (with overhead)

### 13.3 Cost protection

If at any 5-condition checkpoint the running cost projects above $30 for the primary pass, halt and surface for review. This is a hard cap.

---

## 14. Runbook (step-by-step on RunPod)

### 14.1 Pre-flight (workstation, before pod allocation)

- [ ] Confirm `v1_1_1_LOCKED` tag is on the remote
- [ ] Generate read-only HF token; verify it can read all 6 datasets
- [ ] Pull the latest manifests from the locked tag

### 14.2 Pod allocation

- [ ] RunPod: launch RTX 4090 community-tier pod
- [ ] Persistent volume: 50 GB at `/workspace`
- [ ] Base image: PyTorch 2.4 CUDA 12.4
- [ ] Wait for pod IP/SSH access

### 14.3 Pod setup

```bash
# On the pod
cd /workspace
git clone https://github.com/<user>/gradience.git
cd gradience
git checkout v1_1_1_LOCKED

# Set HF token
export HF_TOKEN=<token>

# Install pinned deps
pip install --upgrade pip
pip install -r papers/benchmark_reliability_study/requirements.gpu.lock

# Verify GPU available
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# Expected: True, NVIDIA GeForce RTX 4090
```

### 14.4 Pre-flight checks (scripted)

```bash
cd papers/benchmark_reliability_study
python scripts/gpu_inference.py --preflight-only --config configs/study_config.yaml
```

Verify all 6 checks per §9 pass.

### 14.5 Run

```bash
nohup python scripts/gpu_inference.py \
  --config configs/study_config.yaml \
  --conditions manifests/conditions_primary.csv \
  --conditions-secondary manifests/conditions_gsm8k.csv \
  --fewshot manifests/fewshot_manifest.csv \
  --out-dir runs/raw/ \
  > runs/inference.log 2>&1 &

# Monitor in another shell:
tail -f runs/inference.log
```

### 14.6 Periodic checks

Every few hours:
- Check `runs/status.json` for progress
- Check `runs/failures.jsonl` for any failures
- Verify cost is on track (~$0.40/hr × elapsed hours)

### 14.7 Completion

```bash
# Final tar
tar -czf runs_final.tar.gz runs/raw/ runs/status.json runs/failures.jsonl runs/inference.log

# Upload to user's workstation (e.g., via rsync, rclone, or RunPod CLI)
rsync -avz runs_final.tar.gz user@workstation:/path/to/papers/benchmark_reliability_study/
```

### 14.8 Verification (workstation)

```bash
cd /Users/john/code/gradience/papers/benchmark_reliability_study
tar -xzf runs_final.tar.gz

# Run §12 verification protocol
python scripts/04_normalize_outputs.py --conditions manifests/conditions_primary.csv \
  --raw-dir runs/raw/ --schemas-dir schemas/ \
  --out runs/normalized/item_level_primary.parquet

python scripts/98_reproducibility_trace.py --config configs/study_config.yaml \
  --manifests-dir manifests/ --raw-dir runs/raw/ \
  --normalized-dir runs/normalized/ --analysis-dir analysis/ \
  --sample-n 5 --seed 20260424 \
  --out reports/reproducibility_trace.md
```

If both succeed: GPU phase is complete. Move to Phase 5 (analysis pipeline).

### 14.9 Pod teardown

After upload + verification:

```bash
# Stop pod from RunPod web UI or CLI
runpod stop <pod-id>
runpod delete <pod-id>
```

---

## 15. Open questions / decisions for implementation time

These are not blockers for the spec; they're judgment calls that will need to be made during implementation:

1. **MMLU subject filter timing.** Current spec: load `mmlu/all` and filter to our 5 panel subjects in Python. Alternative: load each panel subject as a separate config (5 separate `load_dataset` calls). The "all" approach is simpler; the per-subject approach is faster (less data loaded). Recommend: start with "all"; switch to per-subject if pre-flight is slow.

2. **HellaSwag prompt length.** HellaSwag prompts can be long (context + 4 candidate continuations × ~50 tokens). Verify single-batch fits in VRAM at the chosen batch size; if not, drop batch size for HellaSwag specifically.

3. **Few-shot exemplar rendering for non-standard formats.** HellaSwag, Winogrande, and GSM8K have different "natural" few-shot formats than MCQ benchmarks. The prompt templates handle this through their internal structure; verify each renders cleanly during pre-flight.

4. **Tokenizer chat-template handling for Qwen2.5-Instruct.** Instruction-tuned models often expect a chat-template format (`<|im_start|>system\n...<|im_end|>`). Decision: do we apply chat formatting, or treat Qwen2.5 as a base-style continuation model? Recommend: treat as base-style (i.e., don't apply chat template) for consistency with Pythia and to keep prompt format identical across models. This is itself a measurement-design choice and should be noted in the manuscript's §5.1 model-selection discussion.

5. **Random number generation.** The script doesn't sample (greedy decoding), so RNG state shouldn't affect outputs. But verify by setting `torch.manual_seed(20260424)` at script start — should be a no-op for greedy generation but defends against accidental sampling code.

6. **Disk I/O during run.** Writing JSONL line-by-line per item is fine performance-wise. Verify with strace if throughput drops; switch to in-memory buffering with periodic flush if needed.

---

## 16. Pinned library versions (target)

To be confirmed against RunPod's PyTorch image at execution time and locked in `requirements.gpu.lock`:

```
torch==2.4.0
transformers==4.46.0
accelerate==1.1.0
sentencepiece==0.2.0
datasets==4.8.0
huggingface_hub==1.12.0
pyarrow==15.0.2
pandas==2.2.3
pyyaml==6.0.2
jsonschema==4.23.0
numpy==1.26.4
scipy==1.13.1
tqdm==4.67.0
```

CUDA: 12.4
Python: 3.11.x

---

## 17. Non-negotiables

Mirroring CPU §13:

1. No prompt modifications during inference. Prompts are locked at v1.1.1; any deviation is a `deviations.md` entry, not a runtime fix.
2. No silent dropping of conditions. Every condition either succeeds (writes complete `runs/raw/{run_id}/`) or fails (logged in `runs/failures.jsonl` with traceback).
3. No silent dropping of items. Every item gets a row in the JSONL output, even if `parse_status != "parsed"`.
4. Greedy decoding only. No sampling. `do_sample=False, temperature=0`.
5. No ad-hoc retries that change the input. If a condition fails on first attempt, the retry must be byte-identical (same prompt, same fewshot, same model state).
6. No batch-size adjustments that change which items go in which forward pass. Batch sizing is for memory efficiency only; output values are batch-invariant.
7. No mixing of inference-tool versions across the run. `requirements.gpu.lock` is committed once and not touched mid-run.

---

## 18. Success criterion

The GPU pass succeeds when:

1. All conditions in `manifests/conditions_primary.csv` and `manifests/conditions_gsm8k.csv` either have a complete `runs/raw/{run_id}/` directory or are documented in `runs/failures.jsonl` with a clear traceback.
2. The CPU normalizer (`04_normalize_outputs.py`) processes all complete runs without exit-code 3 failures.
3. The reproducibility trace (`98_reproducibility_trace.py`) reports trace status `pass` on a 5-condition recompute sample.
4. Total cost is at or below the $30 hard cap.

The bar for "success" is contract-conformance, not "all 672 conditions completed without failure." A 5% failure rate is acceptable if those failures are well-documented and the remaining conditions support the analysis. The CPU normalizer's exit-3 hard-fails on a single condition's contract violation; success means we don't hit those hard-fails on the conditions we did complete.

---

## 19. Implementation status

This spec describes work not yet started. The next steps to implement:

1. Write `scripts/gpu_inference.py` per §5 (~400-600 lines).
2. Write `requirements.gpu.lock` per §3.2.
3. Write integration tests against the small fixture data already in `tests/fixtures/` (skip the actual GPU inference; use mock model objects).
4. Spin up RunPod pod per §14.
5. Execute pre-flight + main loop.
6. Verify per §12 + §14.8.
7. Tag `v1_1_1_INFERENCE_COMPLETE` once verification passes (or document failures and proceed under v1.1.1 with deviations).

---

*End of GPU spec v0.1. Companion to SPEC_CPU_v0_2.md. Ready for review and execution.*
