# UDR/SDI Release QA Checklist

## Pre-Release Validation ✅

### Core Functionality
- [ ] **Unit tests pass**: Run `python3 tests/test_udr_correctness.py`
- [ ] **Integration tests pass**: Run `PYTHONPATH=/Users/john/code/gradience python3 test_udr_pipeline.py`
- [ ] **Bench smoke test passes**: Run `PYTHONPATH=/Users/john/code/gradience python3 test_bench_udr_smoke.py`
- [ ] **Simple verification passes**: Run `python3 test_bench_udr_simple_check.py`

### Behavioral Contract
- [ ] **UDR computation**: Verify ||ΔW||₂ / ||W_base||₂ produces expected values
- [ ] **SDI transformation**: Verify log₁₀(UDR + ε) scaling is correct
- [ ] **Epsilon protection**: Small base norms don't cause division by zero
- [ ] **Graceful degradation**: Missing base models return None UDR (not crash)
- [ ] **Memory efficiency**: r×r operations avoid dense BA formation

### Cache System
- [ ] **Cache write/read**: Base norms survive filesystem roundtrip
- [ ] **Cache corruption**: Invalid files trigger recomputation (not crash)
- [ ] **Cache invalidation**: Different models don't share cache entries
- [ ] **Path handling**: Relative and absolute cache paths work correctly

### CLI Integration
- [ ] **New flags work**: `--base-model`, `--base-norms-cache`, `--no-udr`
- [ ] **Backward compatibility**: Existing commands work without UDR flags
- [ ] **Help text**: New options appear in `--help` output
- [ ] **Error handling**: Invalid base model IDs fail gracefully

### Bench Integration
- [ ] **Config parsing**: `audit.base_model` in YAML configs is recognized
- [ ] **Parameter passing**: UDR params flow from config to audit_lora_peft_dir
- [ ] **Report instrumentation**: UDR metrics appear in bench.json when available
- [ ] **Optional sections**: Reports work normally when UDR unavailable
- [ ] **Sample configs**: distilbert_sst2_with_udr.yaml demonstrates usage

## Documentation Quality

### User-Facing Docs
- [ ] **README coverage**: UDR section explains what it measures and how to use it
- [ ] **CLI examples**: Working commands for common UDR workflows
- [ ] **Interpretation guide**: Cheat sheet for UDR/SDI values
- [ ] **Error scenarios**: Common failure modes and solutions documented

### Code Documentation
- [ ] **Behavioral contract**: UDR function guarantees are explicit
- [ ] **Mathematical formulation**: ||ΔW||₂ / ||W_base||₂ formula is documented
- [ ] **Performance notes**: r×r optimization approach is explained
- [ ] **API stability**: Public interface changes are noted

## Compatibility & Regression

### Backward Compatibility
- [ ] **Existing audit calls**: Work without UDR parameters
- [ ] **Dataclass evolution**: LoRALayerAudit maintains field order
- [ ] **Default values**: New fields have sensible defaults (0.0, None)
- [ ] **JSON serialization**: to_dict() includes new fields appropriately

### Cross-Platform
- [ ] **CPU devices**: UDR works on CPU-only setups
- [ ] **Memory constraints**: Large models don't OOM during base norm computation
- [ ] **Path separators**: Cache paths work on different filesystems

## Production Readiness

### Performance
- [ ] **Timing overhead**: UDR adds <20% to audit runtime
- [ ] **Memory overhead**: Peak memory usage stays reasonable
- [ ] **Cache effectiveness**: Base norm reuse provides speedup

### Error Handling
- [ ] **Network failures**: HuggingFace download errors are handled
- [ ] **Disk space**: Cache write failures degrade gracefully
- [ ] **Invalid models**: Non-existent base models fail with clear messages

### Monitoring
- [ ] **UDR availability**: Reports indicate when UDR was computed vs skipped
- [ ] **Cache statistics**: Hit/miss rates available for debugging
- [ ] **Timing breakdown**: Audit phases can be profiled if needed

## Manual Verification

### Sample Outputs
- [ ] **Known good case**: DistilBERT SST-2 produces reasonable UDR values (~0.1-1.0)
- [ ] **Edge case**: Very small adapters produce UDR values > 1.0
- [ ] **No base model**: Audit completes with UDR=None in all layers

### End-to-End Workflow
1. [ ] Create fresh environment
2. [ ] Install dependencies 
3. [ ] Run `python3 -m gradience.cli audit <peft-dir> --base-model distilbert-base-uncased`
4. [ ] Verify output JSON contains UDR fields
5. [ ] Run Bench with sample UDR config
6. [ ] Verify bench.json contains instrumentation block

## Sign-off

- [ ] **All tests pass** on clean environment
- [ ] **Documentation reviewed** for accuracy
- [ ] **Breaking changes identified** (none expected for this release)
- [ ] **Performance impact measured** and acceptable
- [ ] **Ready for production** deployment

---

**Release Checklist Completed By**: _________________  
**Date**: _________________  
**Version**: UDR/SDI Initial Release  
**Notes**: _________________