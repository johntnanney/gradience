# Gradience v0.3.10 Release Notes

## 🔧 Critical Fixes for Production Bench Pipeline

This release resolves multiple critical issues discovered during Mistral 7B + GSM8K benchmarking, ensuring robust operation across all model types and save strategies.

### What's Fixed

**Adapter Save & Audit Reliability**
- **NEW**: `_save_peft_adapter_only()` - Safe adapter-only saving with guardrails against full model saves
- **NEW**: `_unwrap_model_for_save()` - Proper model unwrapping for accelerator/DataParallel
- Save adapters after both probe and variant training regardless of `save_strategy` setting
- Pre-audit validation ensures `adapter_config.json` exists before audit begins
- Clear diagnostic errors for missing adapter files

**Audit & Rank Suggestion Fixes**  
- Add `current_r` to audit summary for rank suggestion compatibility
- Fix "Could not infer current_r" error by explicitly providing probe rank
- Resolves audit failures when `stable_rank_mean=0.0` and `utilization_mean=0.0`

**GSM8K & Generation Task Compatibility**
- Fix `KeyError: 'eval_loss'` for generation-based tasks that don't compute classification loss  
- Use `eval_results.get("eval_loss")` instead of `eval_results["eval_loss"]` for graceful handling
- Full compatibility with GSM8K causal LM evaluation pipeline

**CI & Testing Infrastructure**
- Add `transformers>=4.20.0`, `peft>=0.4.0`, and `datasets` to unit test dependencies
- Resolves "No module named X" CI failures when tests import bench protocol modules
- Ensures comprehensive test coverage across all task profiles

### Performance & Reliability 

✅ **End-to-End Pipeline**: Complete Mistral 7B + GSM8K benchmarking now works reliably  
✅ **Memory Safety**: Only saves small LoRA adapters (~MB), never full models (~GB)  
✅ **Cross-Task Support**: Works with both classification (GLUE) and generation (GSM8K) tasks  
✅ **Save Strategy Independence**: Functions correctly with any `save_strategy` setting

### Files Updated

- `gradience/bench/protocol.py` - Core adapter saving, audit fixes, and eval_loss compatibility
- `.github/workflows/tests.yml` - CI dependency updates for comprehensive testing

### Breaking Changes

None. All fixes are backward compatible and maintain existing API contracts.

## Technical Notes

**Issue Resolution Pattern**:
1. `save_strategy="no"` → adapter files missing → audit fails → **FIXED**: Force save adapters  
2. Missing `current_r` in audit → rank suggestions fail → **FIXED**: Add probe rank explicitly
3. Missing `eval_loss` in generation tasks → KeyError → **FIXED**: Use `.get()` for optional fields
4. Missing ML dependencies in CI → test failures → **FIXED**: Install full ML stack

**Deployment**: Use tag `v0.3.10` for stable, production-ready bench runs across all model types.

**Tested Scenarios**:
- ✅ Mistral 7B + GSM8K (screening & certification configs)
- ✅ DistilBERT + SST-2 (existing GLUE pipeline)  
- ✅ `save_strategy="no"` configurations
- ✅ CI/CD pipeline with comprehensive dependencies

---

**Ready for**: Production Mistral+GSM8K benchmarking, multi-seed certification runs, and reliable bench automation across diverse model architectures.