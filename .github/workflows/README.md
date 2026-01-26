# GitHub Actions CI

This directory contains the GitHub Actions workflows for Gradience.

## CI Workflow (`ci.yml`)

Runs on every push and PR to prevent regressions in core functionality.

### What it tests:

**🐍 Python Compatibility**
- Python 3.10, 3.11, 3.12
- Cross-platform (Ubuntu)

**🧪 Code Quality**
- `ruff` linting
- `mypy` type checking  
- Unit tests with `pytest`
- Code coverage reporting

**⚙️ Core Invariants**
- Package imports work correctly
- Config parsing (all YAML files)
- Compression config generation
- CLI help commands

**🏃‍♂️ CPU Bench Smoke Test**
- Runs `distilbert_sst2_ci.yaml` with `--smoke` flag
- Validates full pipeline: train → audit → compress → evaluate
- Checks that `bench.json`, `bench.md`, `runs.json` are generated
- Validates JSON structure and compressed variants

### Why CPU-only?

GitHub Actions runners don't have GPUs, so we focus on:
- ✅ **Config parsing** - catches YAML syntax errors
- ✅ **Audit writing** - ensures telemetry/audit logic works  
- ✅ **Compression config generation** - validates rank suggestion algorithms
- ✅ **Aggregation invariants** - checks bench.json structure
- ✅ **End-to-end pipeline** - but with DistilBERT (fast) instead of Mistral

This catches "it worked on my pod" bugs without requiring expensive GPU time.

### Local Testing

Run the same checks locally:

```bash
# Quick validation
python test_ci_locally.py

# Full pytest
pytest tests/ -v

# Manual smoke test  
python -m gradience.bench.run_bench \
  --config gradience/bench/configs/distilbert_sst2_ci.yaml \
  --output test_output \
  --smoke --ci
```

### Performance

- **Unit tests**: ~30 seconds
- **CPU smoke test**: ~3-5 minutes  
- **Total CI time**: ~6-8 minutes per Python version

The smoke test uses:
- DistilBERT (117M params vs Mistral's 7B)
- 50 steps (vs 200+ for full runs)
- 200 train samples (vs 1000s)
- CPU device (no CUDA setup time)