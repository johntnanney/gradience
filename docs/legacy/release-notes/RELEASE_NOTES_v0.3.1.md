# Gradience v0.3.1 Release Notes

## 🎯 Major Addition: Bench v0.1 - LoRA Compression Validation Framework

Gradience v0.3.1 introduces **Bench**, a comprehensive validation harness for testing LoRA compression recommendations in controlled experiments.

### What is Bench?

Bench turns "suggested rank reductions" into testable claims by running complete train→audit→compress→retrain→evaluate workflows. It provides empirical validation for Gradience's LoRA efficiency recommendations.

**Core workflow:**
1. Train probe adapter (high-rank baseline)
2. Audit → generate compression suggestions  
3. Retrain with compressed ranks
4. Evaluate and compare performance
5. Generate detailed reports (JSON + Markdown)

### Key Features

**🔬 Safe Uniform Baseline Policy**
- **Pass Rate**: ≥67% of seeds must pass accuracy tolerance (Δ ≥ -2.5%)
- **Worst-Case Protection**: Worst-performing seed must have Δ ≥ -2.5%
- Machine-readable policy: `gradience/bench/policies/safe_uniform.yaml`

**📊 Certifiable v0.1 Results** (DistilBERT/SST-2)
- **uniform_median**: 61% compression, 100% pass rate, worst Δ = -1.0% ✅
- Validated across 3 seeds × 500 training steps

**🛠️ Production-Ready CLI**
```bash
# Basic benchmark
python -m gradience.bench.run_bench \
  --config configs/distilbert_sst2.yaml \
  --output results/run_001

# CI mode (fails if strategies don't pass)  
python -m gradience.bench.run_bench \
  --config configs/distilbert_sst2.yaml \
  --output ci_run --ci

# Smoke test (faster)
python -m gradience.bench.run_bench \
  --config configs/distilbert_sst2.yaml \
  --output smoke_test --smoke
```

### Bench Components

**Configuration System**
- YAML-based experiment configs with smoke/full modes
- Device override support (CPU/CUDA/MPS)
- Reproducible seeds and hyperparameters

**Compression Strategies**
- `uniform_median`: Compress all layers to median suggested rank
- `uniform_p90`: Compress all layers to 90th percentile (tail-safe)
- Automatic control run detection when no compression possible

**Comprehensive Reporting**
- `bench.json`: Machine-readable results with verdicts
- `bench.md`: Human-readable summary with recommendations
- `bench_internal.json`: Detailed internal state for debugging

### Validation Infrastructure

**Policy Compliance**
- Automated verdict generation against Safe Uniform Baseline
- Pass/fail determination based on empirical accuracy deltas
- Multi-seed validation with statistical confidence

**CI Integration**
- Exit codes for automated testing
- Configurable failure thresholds
- Progress monitoring and artifact management

## 🔧 Package Improvements

**Fixed Critical Dependencies**
- Added missing `pyyaml` dependency (fixes Bench YAML config loading)
- Moved HuggingFace dependencies to core (`transformers`, `peft`, `datasets`)
- Ensures clean install works for all Bench functionality

**Enhanced Distribution**
- Verified package builds correctly with `twine check`
- Tested clean virtual environment installation
- All CLI commands work from installed package

## 📁 New Directory Structure

```
gradience/bench/
├── run_bench.py              # Main CLI entry point
├── protocol.py               # Core benchmark protocol
├── report.py                 # Report generation
├── configs/                  # Experiment configurations
│   ├── distilbert_sst2*.yaml
│   └── ...
├── policies/                 # Validation policies
│   ├── safe_uniform.yaml     # Machine-readable policy
│   └── README.md
└── README.md                 # Bench documentation
```

## ⚠️ Important Notes

**Task/Model Dependency**
- Current baselines validated specifically for DistilBERT on SST-2
- Always validate on your specific task/model before production use
- Bench provides the framework - baselines are empirically determined

**Resource Requirements**
- CPU mode works everywhere (no GPU required for basic validation)
- CUDA mode available for Linux + NVIDIA setups
- Smoke mode enables faster testing with reduced training steps

## 🚀 Getting Started

**Installation**
```bash
pip install gradience  # or pip install dist/gradience-0.3.1-py3-none-any.whl
```

**Quick validation run**
```bash
# Ensure Bench is working
python -m gradience.bench.run_bench --help

# Run a quick smoke test
python -m gradience.bench.run_bench \
  --config gradience/bench/configs/distilbert_sst2.yaml \
  --output test_run --smoke --device cpu
```

**Integration with existing workflow**
```bash
# After training with Gradience telemetry
gradience audit --peft-dir my_adapter/ --top-wasteful 5
# Use Bench to validate compression suggestions
python -m gradience.bench.run_bench \
  --config my_config.yaml --output validation_run
```

## 📚 Documentation

- **Bench Guide**: `gradience/bench/README.md`
- **Policy Details**: `VALIDATION_POLICY.md`
- **User Manual**: `USER_MANUAL.md`
- **Quick Reference**: `QUICK_REFERENCE.md`

## 🔄 Migration Notes

**No Breaking Changes**
- All existing Gradience vNext APIs remain unchanged
- Telemetry schema (`gradience.vnext.telemetry/v1`) unchanged
- Existing audit/monitor/check commands work as before

**New Optional Workflow**
- Bench is additive - use it to validate compression recommendations
- Integrates with existing telemetry via audit append mode
- CI/testing workflows can now include empirical validation

---

**Full Changelog**: [v0.3.0...v0.3.1](https://github.com/johntnanney/gradience/compare/v0.3.0...v0.3.1)

For detailed technical documentation, see the [Bench README](gradience/bench/README.md) and [validation policy](VALIDATION_POLICY.md).