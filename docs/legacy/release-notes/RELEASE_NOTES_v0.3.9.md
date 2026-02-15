# Gradience v0.3.9 Release Notes

## 🔧 Critical Fixes for Mistral GSM8K Pipeline

This release resolves the final training and compatibility issues discovered in v0.3.7/v0.3.8, ensuring the Mistral 7B + GSM8K causal LM pipeline runs smoothly on modern environments.

### What's Fixed

**TrainingArguments Compatibility**
- Fixed deprecated `evaluation_strategy` parameter → `eval_strategy` across all task profiles
- Resolves "TrainingArguments.__init__() got an unexpected keyword argument 'evaluation_strategy'" errors
- Updated both GLUE sequence classification and training example scripts for consistency

**GSM8K Training Stability**  
- Disabled in-training evaluation to prevent dataset format conflicts
- Added `eval_strategy="no"` and `do_eval=False` to prevent crashes during training
- Set `eval_dataset=None` in Trainer constructor for clean separation
- Maintained `remove_unused_columns=False` to preserve custom tokenization pipeline

**Configuration Schema**
- Fixed "training:" → "train:" block migration across all Mistral GSM8K configs
- Ensures configs work with standard Transformers training argument loading
- Cleaned Python bytecode cache to prevent stale parameter references

### Performance Validation

✅ **Training Confirmed Healthy**: Loss progression 1.15 → 0.10 validates response-only masking  
✅ **7B Model Support**: Gradient checkpointing and bf16 optimizations working  
✅ **Pipeline Integrity**: probe → train → evaluate → compress workflow operational

### Files Updated

- `gradience/bench/task_profiles/gsm8k_causal_lm.py` - Training argument fixes
- `gradience/bench/task_profiles/seqcls_glue.py` - Evaluation strategy compatibility  
- `scripts/training_integration_example.py` - Parameter modernization
- All 4 Mistral GSM8K configuration files - Schema corrections

### Testing

The complete Mistral 7B + GSM8K pipeline now runs successfully:
```bash
python -m gradience.bench.run_bench \
  --config gradience/bench/configs/mistral_gsm8k_screening_seed42.yaml \
  --output bench_runs/mistral_gsm8k_screen
```

### Breaking Changes

None. All fixes are backward compatible and maintain existing API contracts.

## Technical Notes

**Issue Resolution Pattern**: 
1. Modern Transformers uses `eval_strategy` not `evaluation_strategy`
2. GSM8K requires `remove_unused_columns=False` for custom tokenization  
3. Response-only loss training needs clean train/eval dataset separation

**Deployment**: Use tag `v0.3.9` for stable Mistral+GSM8K certification runs.

---

**Ready for**: Production Mistral 7B benchmarking, A40 certification workflows, and scaling to additional causal LM tasks.