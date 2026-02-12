# CI Gates for "Pip Install Ready"

This document describes the CI gates that ensure Gradience is ready for PyPI distribution.

## Overview

The `pip-install-ready.yml` workflow validates that:

1. **Base Install** - Minimal dependencies, fast CLI, clean imports
2. **Bench Extras** - Full ML stack works with CPU-only smoke tests  
3. **Packaging Correctness** - Wheels include all configs, importlib.resources works
4. **Performance** - CLI help commands meet speed requirements

## Test Matrix

### A. Base Install Test
- **Platforms**: Ubuntu, macOS, Windows
- **Python versions**: 3.9, 3.10, 3.11, 3.12
- **Validates**:
  - `pip install .` works without heavy ML dependencies
  - `import gradience` works and shows version
  - `gradience --help` completes within 3 seconds
  - No torch/transformers in base dependencies

### B. Bench Extras Test  
- **Platforms**: Ubuntu
- **Python versions**: 3.9, 3.11, 3.12
- **Validates**:
  - `pip install .[bench]` works with CPU PyTorch
  - `gradience-bench --help` works within 15 seconds (first run)
  - All YAML configs are accessible programmatically
  - CPU-only smoke test validation

### C. Packaging Correctness
- **Platforms**: Ubuntu  
- **Python versions**: 3.10, 3.12
- **Validates**:
  - `python -m build` creates wheel and sdist
  - Wheel installation works in fresh environment
  - Console scripts work from installed wheel
  - All configs accessible via `importlib.resources`
  - Evidence and policy subdirectories included
  - Fresh install test (no source interference)

### D. Performance Requirements
- **Platform**: Ubuntu
- **Python version**: 3.11
- **Requirements**:
  - `gradience --help` < 3 seconds (base CLI)
  - `gradience-bench --help` < 15 seconds (first run with ML deps)
  - `gradience-bench --help` < 5 seconds (subsequent runs)

## Local Testing

Use the local test script to validate before pushing:

```bash
# Test all scenarios
python scripts/test_pip_install_ready.py

# Test specific scenario
python scripts/test_pip_install_ready.py --test base
python scripts/test_pip_install_ready.py --test bench
python scripts/test_pip_install_ready.py --test packaging
python scripts/test_pip_install_ready.py --test performance
```

## Key Features Validated

### ✅ Minimal Base Dependencies
- Core package installs with only: pyyaml, numpy, packaging, rich
- No forced torch/transformers download (580MB+ saved)
- Fast CLI help without ML library loading

### ✅ Optional Heavy Dependencies  
- ML dependencies only loaded with `[bench]` extras
- CPU-only PyTorch support for CI environments
- Graceful degradation when ML libs unavailable

### ✅ Complete Config Bundling
- 44+ YAML config files included in wheel
- Evidence and GPU smoke subdirectories shipped
- Policy files accessible via importlib.resources
- Backwards compatible file-path access

### ✅ Professional Console Scripts
- `gradience` and `gradience-bench` commands work out-of-box
- Proper entry points in pyproject.toml
- Cross-platform compatibility (Windows/macOS/Linux)

### ✅ Lazy Loading Performance
- Base CLI help < 3s (no ML deps loaded)
- Bench CLI acceptable on first run (< 15s)
- Fast subsequent calls (< 5s) with import caching

## Success Criteria

All tests must pass for the package to be considered "pip install ready". The workflow validates:

- ✅ Clean installation across platforms and Python versions
- ✅ Functional console scripts without source code present  
- ✅ Complete config and policy file inclusion
- ✅ Performance requirements for CLI responsiveness
- ✅ CPU-only operation for broad compatibility

This ensures users can `pip install gradience` on any supported platform and immediately have a working, complete installation with fast CLI access and all necessary configuration files.