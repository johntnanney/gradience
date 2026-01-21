# CPU-Only Smoke Tests

**TL;DR: Contributors can verify core logic without GPUs in ~6 seconds.**

This document describes the CPU-only smoke test suite that allows contributors to verify they haven't broken core pipeline logic without requiring GPUs or model downloads.

## ⚡ Quick Start

```bash
# Run all CPU smoke tests (recommended)
python scripts/run_ci_smoke_tests.py

# Run with timing information
python scripts/run_ci_smoke_tests.py --timing

# Run with verbose output
python scripts/run_ci_smoke_tests.py --verbose

# Run specific test suites
python -m pytest tests/test_cpu_smoke_comprehensive.py -v
python -m pytest tests/test_bench_config_parsing.py -v
```

## 📋 Test Coverage

The smoke test suite verifies:

### 1. **Config Parsing → TaskProfile Routing**
- `tests/test_bench_config_parsing.py`
- Verifies config → TaskProfile routing works correctly
- Tests both explicit profile specification and backward compatibility
- Covers edge cases and error handling

### 2. **GSM8K Formatting & Response-Only Masking** 
- `tests/test_gsm8k_profile.py` (already existed)
- Verifies answer extraction from `#### 42` format
- Tests response-only masking for causal LM training
- Validates probe gating logic

### 3. **Audit with Minimal Artifacts**
- `tests/test_audit_json_invariants.py` (already existed) 
- `tests/test_cpu_smoke_comprehensive.py`
- Tests audit pipeline with minimal LoRA adapter artifacts
- Verifies audit → rank suggestion pipeline
- No GPU required, uses synthetic torch tensors

### 4. **Aggregator Backward Compatibility**
- `tests/test_bench_aggregator_compatibility.py`
- Tests aggregation handles old-style (string) vs new-style (dict) task fields
- Verifies statistical calculations work correctly
- Tests mixed variant aggregation scenarios

### 5. **Comprehensive Pipeline Integration**
- `tests/test_cpu_smoke_comprehensive.py`
- End-to-end config → profile → tokenization pipeline
- Protocol metadata inclusion (git, environment, model/dataset revisions)
- Error handling robustness
- Performance smoke checks

## 🚀 CI Integration

### Performance Target
- **Target**: Under 5 minutes for CI
- **Actual**: ~6 seconds (well under target!)
- **Optimizations**: Parallel execution, minimal setup, focused tests

### GitHub Actions Example
```yaml
name: CPU Smoke Tests
on: [push, pull_request]
jobs:
  smoke-tests:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    - uses: actions/setup-python@v4
      with:
        python-version: '3.10'
    - run: pip install -e ".[dev]"
    - run: python scripts/run_ci_smoke_tests.py --timing
```

### Pre-commit Hook Example
```bash
#!/bin/sh
# .git/hooks/pre-commit
echo "Running CPU smoke tests..."
python scripts/run_ci_smoke_tests.py --timing || {
    echo "❌ Smoke tests failed. Fix before committing."
    exit 1
}
echo "✅ Smoke tests passed!"
```

## 💡 Design Principles

### 1. **No External Dependencies**
- No GPU required
- No model downloads
- No internet connectivity
- Uses minimal synthetic data

### 2. **Fast Feedback**
- Complete suite runs in seconds
- Fails fast with clear error messages
- Focuses on logic, not integration

### 3. **Comprehensive Coverage**
- Tests all critical pipeline components
- Covers backward compatibility scenarios
- Validates error handling edge cases

### 4. **Contributor Friendly**
- Clear test names and documentation
- Easy to run locally
- Provides actionable feedback

## 📁 Test Files Reference

| Test File | Purpose | Runtime |
|-----------|---------|---------|
| `test_cpu_smoke_comprehensive.py` | End-to-end pipeline verification | ~4s |
| `test_bench_config_parsing.py` | Config → TaskProfile routing | ~3s |
| `test_gsm8k_profile.py` | GSM8K formatting & masking | ~1s |
| `test_audit_json_invariants.py` | Audit structure validation | ~1s |
| `test_bench_aggregator_compatibility.py` | Aggregation compatibility | ~2s |
| `test_cli_smoke.py` | Basic CLI functionality | ~1s |
| `test_validation_protocol.py` | Core validation logic | ~1s |

## 🔧 Common Issues

### Test Failures
```bash
# Debug specific test
python -m pytest tests/test_bench_config_parsing.py::TestBenchConfigParsing::test_explicit_profile_routing -v

# Run without optimizations for better error messages  
python -m pytest tests/ --tb=long -v
```

### Performance Issues
```bash
# Check what's taking time
python -m pytest tests/ --durations=10

# Use parallel execution for larger test suites
pip install pytest-xdist
python -m pytest tests/ -n auto
```

## 🎯 Adding New Smoke Tests

When adding new pipeline components:

1. **Add to appropriate test file** or create new focused test file
2. **Keep tests fast** - use minimal synthetic data
3. **Test edge cases** - empty configs, malformed input, etc.
4. **Update smoke runner** - add to `scripts/run_ci_smoke_tests.py`
5. **Verify timing** - ensure total suite stays under 1 minute

### Example Test Pattern
```python
def test_new_component_smoke(self):
    """Test new component works without GPU/downloads."""
    # Use minimal synthetic data
    config = {"minimal": "config"}
    
    # Test core logic
    result = new_component.process(config)
    
    # Verify structure/behavior
    assert "required_field" in result
    assert result["required_field"] > 0
```

---

**Remember**: These tests verify **logic**, not **training quality**. For integration testing with real models, use the full bench test suite.