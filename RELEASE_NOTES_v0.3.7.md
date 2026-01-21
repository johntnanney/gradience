# Gradience v0.3.7 Release Notes

## 🎯 Major Feature: Bench Causal LM Support (Mistral + GSM8K)

Extends Gradience Bench from **encoder + classification** to **decoder + generation** while preserving the core **probe → audit → compress → eval → aggregate** protocol.

### What's New

**Task Profile System**
- Clean abstraction layer for different model types and evaluation tasks
- `GLUESequenceClassificationProfile` (existing functionality)  
- `GSM8KCausalLMProfile` (new causal LM pipeline)
- Automatic profile detection with backward compatibility

**Causal Language Model Support**
- Response-only loss training (prompts masked with `-100`, completions learned)
- Generation-based evaluation with exact match scoring
- Support for `mistralai/Mistral-7B-v0.1` with bf16 precision
- Gradient checkpointing and memory optimizations for 7B models

**GSM8K Mathematical Reasoning**
- Canonical format: `Question: {question}\nAnswer: {answer}`
- Exact match evaluation with `#### number` pattern extraction
- Fallback to last number regex for robustness
- Task-specific probe gates (15% exact match threshold)

### New Configurations

Ready-to-run configs for A40 certification:

```
gradience/bench/configs/mistral_gsm8k_screening_seed42.yaml       # Fast validation
gradience/bench/configs/mistral_gsm8k_certifiable_seed42.yaml     # Seed 1/3
gradience/bench/configs/mistral_gsm8k_certifiable_seed123.yaml    # Seed 2/3  
gradience/bench/configs/mistral_gsm8k_certifiable_seed456.yaml    # Seed 3/3
```

### Memory & Performance Optimizations

- **Gradient checkpointing**: Reduces memory usage for large models
- **Disabled model cache**: Prevents memory accumulation during training
- **Adapter-only saving**: No full model checkpoints (`save_strategy: "no"`)
- **bf16 precision**: Faster training with reduced memory footprint

### API Changes

**Backward Compatible**: All existing GLUE/SST-2/QNLI configs continue to work unchanged.

**New Config Schema** (optional):
```yaml
model:
  type: causal_lm  # NEW: "seqcls" vs "causal_lm"
  gradient_checkpointing: true
  use_cache: false

task:
  profile: gsm8k_causal_lm  # NEW: explicit task profile
  generation:               # NEW: generation parameters
    max_new_tokens: 128
    do_sample: false
  probe_gate:              # NEW: task-specific gates
    metric: exact_match
    min_value: 0.15
```

### Usage

```bash
# Quick validation (5-10 min on A40)
python -m gradience.bench.run_bench \
  --config gradience/bench/configs/mistral_gsm8k_screening_seed42.yaml \
  --output bench_runs/mistral_gsm8k_screen

# Full certification (3 seeds → aggregate)  
python -m gradience.bench.run_bench \
  --config gradience/bench/configs/mistral_gsm8k_certifiable_seed42.yaml \
  --output bench_runs/mistral_gsm8k_seed42

python gradience/bench/aggregate.py \
  bench_runs/mistral_gsm8k_seed42 \
  bench_runs/mistral_gsm8k_seed123 \
  bench_runs/mistral_gsm8k_seed456 \
  --output bench_runs/mistral_gsm8k_agg
```

### Testing

Added comprehensive unit tests:
```bash
pytest tests/test_gsm8k_profile.py -v
```

Tests cover response-only masking, answer extraction, probe gating, and dataset formatting.

## Technical Details

**Implementation Philosophy**: Minimal, reproducible, policy-driven. The same compression policies and audit framework now work seamlessly across encoder and decoder models.

**Requirements**: 
- A40 (or equivalent) GPU for 7B model training
- bf16 support recommended
- Transformers ≥4.20.0, PEFT ≥0.4.0

**Git Reference**: Use tag `v0.3.7` for reproducible A40 runs

---

**Ready for**: Mistral-7B + GSM8K certification runs, blog post validation, and scaling to other causal LM tasks.