# Gradience

**Spectral audit and evidence-based compression for LoRA adapters.** Detect rank waste, generate compression candidates, and validate with multi-seed benchmarks.

## Who it's for

- **ML engineers shipping LoRA adapters** - Audit efficiency and compress without quality loss
- **Researchers comparing compression regimes** - Generate reproducible evidence with multi-seed validation
- **Teams deploying at scale** - Reduce adapter storage and inference costs systematically

## What you get

- **Audit reports** with spectral analysis, rank utilization, and compression recommendations
- **Validated compression candidates** with tolerance thresholds and quality gates
- **Multi-seed benchmark artifacts** proving compression strategies work reliably

## Install

### Quick Install
```bash
# Core package (audit, monitor, compress)
pip install gradience

# Full benchmarking suite (includes transformers, datasets, peft)  
pip install "gradience[bench]"

# Development tools
pip install "gradience[dev]"
```

### PyTorch Installation (Important)

**For CPU-only usage:**
```bash
# Install PyTorch CPU first (recommended)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install "gradience[bench]"
```

**For CUDA GPU usage:**
```bash
# Install PyTorch with CUDA support first (replace cu121 with your CUDA version)
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install "gradience[bench]"
```

**Auto PyTorch (may not choose optimal version):**
```bash
pip install "gradience[bench]"  # Will install PyTorch automatically
```

💡 **Common Issue**: If you see `ModuleNotFoundError: No module named 'datasets'` or `'transformers'`, you installed the base package. **Fix**: `pip install "gradience[bench]"`

📖 **[Complete Installation Guide](https://github.com/gradience-ai/gradience/blob/main/docs/install.md)** - GPU setup, RunPod, troubleshooting, cache configuration

## Verify installation (10 seconds)

```bash
# Install base package (lightweight)
pip install gradience

# Test basic functionality - no ML dependencies needed
gradience --help                # < 3 seconds
gradience audit --help          # Show audit command options  
gradience --version             # Confirm installation

# Test with sample data (no training required)
python -c "
import gradience
print(f'✅ Gradience {gradience.__version__} installed successfully')
print('Base package ready for audit workflows')

# Test JSON parsing (works with any audit.json file)
import json
sample_audit = {
    'audit_timestamp': '2026-02-06T10:30:00.000000',
    'probe_rank': 16,
    'summary': {
        'stable_rank_mean': 2.27,
        'utilization_mean': 0.142,
        'energy_rank_90_p50': 8.0
    }
}
print('📊 Sample audit data parsed successfully')
print(f'   Mean stable rank: {sample_audit[\"summary\"][\"stable_rank_mean\"]}')
print(f'   Rank utilization: {sample_audit[\"summary\"][\"utilization_mean\"]:.1%}')
"
```

**Zero drama**: These commands work immediately after `pip install gradience` with no additional setup.

## Quickstart: 60-second CPU test

```bash
# Install with ML dependencies
pip install "gradience[bench]"

# List available configs (guaranteed in package)
python -c "
import importlib.resources
configs = importlib.resources.files('gradience.bench.configs')
print('Available configs:')
for f in sorted(configs.iterdir()):
    if f.name.endswith('.yaml'):
        print(f'  {f.name}')
"

# Copy and run fast CPU benchmark (~60 seconds)
python -c "
import importlib.resources, shutil
configs = importlib.resources.files('gradience.bench.configs')
shutil.copy(configs / 'distilbert_sst2_ci.yaml', './cpu_test.yaml')
print('Config copied to cpu_test.yaml')
"

gradience bench cpu_test.yaml --output-dir results/

# Examine results
cat results/bench.json  # Main results
cat results/audit.json  # Detailed analysis
```

**Performance expectations**: First run ~60-90s (model download + training), subsequent runs ~30s (cached).

## What gets produced

Every benchmark run generates:

- **`bench.json`** - Complete validation results with compression verdicts
- **`bench.md`** - Human-readable report with recommendations  
- **`compression_configs.json`** - Generated compression candidates (rank suggestions)
- **`audit.json`** - Spectral analysis of the probe adapter (utilization, energy, rank waste)
- **`bench_aggregate.json`** - Multi-seed statistical summary (for research)

## Core commands

```bash
# Audit an existing LoRA adapter
gradience audit --peft-dir /path/to/adapter --json

# Monitor training telemetry
gradience monitor training_run.jsonl --verbose

# Generate rank compression suggestions
gradience audit --peft-dir /path/to/adapter --suggest-per-layer

# Run complete benchmark protocol
gradience-bench --config config.yaml --output results/

# Validate config before training
gradience check --peft adapter_config.json --training training_args.json
```

## Concepts

- **Probe adapter** - High-rank (r=16+) baseline to establish performance ceiling
- **Audit** - Spectral analysis revealing stable rank, utilization, and energy distribution  
- **Candidate generation** - Automated compression config generation (Tier A/B, second rung policies)
- **Validation policy** - Quality tolerance thresholds with multi-seed statistical verification

📖 **[Complete documentation](https://github.com/gradience-ai/gradience/tree/main/docs/)** | **[API reference](https://github.com/gradience-ai/gradience/blob/main/docs/api_stability.md)** | **[Benchmark guide](https://github.com/gradience-ai/gradience/blob/main/docs/bench_guide.md)**

## Integration with HuggingFace Transformers

Add telemetry to your existing training with one line:

```python
from transformers import Trainer
from gradience.vnext.integrations.hf import GradienceCallback

trainer = Trainer(..., callbacks=[GradienceCallback()])
trainer.train()

# Telemetry saved to: <output_dir>/run.jsonl
```

Then analyze your run:

```bash
# Quick overview
gradience monitor output/run.jsonl

# Detailed analysis with recommendations  
gradience monitor output/run.jsonl --verbose

# Audit the resulting adapter
gradience audit --peft-dir output/adapter --layers --json
```

## Example workflow

```bash
# 1. Train with telemetry (or use existing adapter)
python your_training_script.py  # with GradienceCallback

# 2. Quick sanity check
gradience monitor output/run.jsonl

# 3. Audit adapter efficiency 
gradience audit --peft-dir output/adapter --suggest-per-layer

# 4. Run compression benchmark (generates candidates + validates)
gradience-bench --config my_config.yaml --output benchmark_results/

# 5. Deploy compressed adapter with confidence
cp benchmark_results/compression_configs.json production/
```

## Configuration

Benchmark configs specify model, task, compression policies, and validation criteria:

```yaml
model:
  name_or_path: "distilbert-base-uncased"
  
task:
  name: "sst2"
  type: "classification"
  
lora:
  r: 16              # Probe rank
  target_modules: ["q_lin", "v_lin"]
  
compression:
  policies: ["energy_p90", "knee_p90", "uniform"]
  tolerance: 0.02    # 2% quality loss threshold
  
runtime:
  device: "auto"
  seed: [42, 43, 45]  # Multi-seed validation
```

## Requirements

- **Python 3.9+**
- **PyTorch** (install separately or via `[bench]` extra)  
- **Optional**: transformers, peft, datasets (for benchmarking)

## Performance

- **CLI help**: < 3 seconds (no ML dependencies loaded)
- **Base install**: < 10MB (minimal dependencies)
- **CPU-only operation**: Full functionality without GPU requirements

## What we test in CI

Every release is validated with comprehensive CI gates:

- **Package integrity**: Base install, bench extras, and console scripts work correctly across Python 3.9-3.12
- **Performance guarantees**: CLI help commands complete within documented timing requirements  
- **PyPI readiness**: Wheel contents, config accessibility via importlib.resources, and quickstart functionality

📋 **[Complete CI test matrix](https://github.com/gradience-ai/gradience/blob/main/.github/workflows/pip-install-ready.yml)** - See exactly what we validate before each release  
📋 **[CI Gates documentation](https://github.com/gradience-ai/gradience/blob/main/CI_GATES.md)** - Detailed testing requirements and performance targets

## What Gradience is NOT

- ❌ **Not AutoML** - Won't tune hyperparameters for you
- ❌ **Not a training framework** - Works alongside your existing stack  
- ❌ **Not an oracle** - Provides evidence-based recommendations to validate
- ❌ **Not a replacement for evaluation** - Always verify on your target metrics

## API Stability

**Stable interfaces** (backwards compatible):
- CLI commands (`gradience audit`, `gradience-bench`, etc.)
- Config schema (YAML structure)
- Output artifacts (`audit.json`, `bench.json`, `bench.md`)

**Experimental features** are clearly marked and may change.

## Examples

- **[Minimal integration](https://github.com/gradience-ai/gradience/blob/main/examples/vnext/toy_lora_run.py)** - Add telemetry to any training script
- **[Complete benchmark](https://github.com/gradience-ai/gradience/blob/main/examples/vnext/full_benchmark.py)** - End-to-end validation workflow
- **[Custom policies](https://github.com/gradience-ai/gradience/tree/main/examples/config/)** - Define compression strategies

## Documentation

Complete documentation available on GitHub:

- **[Installation Guide](https://github.com/gradience-ai/gradience/blob/main/docs/install.md)** - Complete setup guide with troubleshooting
- **[CLI Reference](https://github.com/gradience-ai/gradience/blob/main/docs/cli.md)** - Complete command-line reference and examples
- **[Configuration Reference](https://github.com/gradience-ai/gradience/blob/main/docs/configs.md)** - YAML config schema and examples
- **[Artifacts & Evidence](https://github.com/gradience-ai/gradience/blob/main/docs/artifacts.md)** - Understanding benchmark outputs
- **[Troubleshooting](https://github.com/gradience-ai/gradience/blob/main/docs/troubleshooting.md)** - Common issues and solutions
- **[RunPod Guide](https://github.com/gradience-ai/gradience/blob/main/docs/runpod.md)** - Optional cloud GPU setup (RunPod-specific)

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](https://github.com/gradience-ai/gradience/blob/main/LICENSE) for details.

## Citation

If you use Gradience in your research, please cite:

```bibtex
@software{gradience2026,
  title = {Gradience: Evidence-Based LoRA Compression for Language Models},
  author = {Nanney, John T.},
  year = {2026},
  url = {https://github.com/gradience-ai/gradience},
  note = {Version 0.9.0}
}
```

**APA Style:** Nanney, J. T. (2026). *Gradience: Evidence-based LoRA compression for language models* (Version 0.9.0) [Computer software]. https://github.com/gradience-ai/gradience

**Note for maintainers:** Update version numbers in citation when releasing new versions. Current version should match `pyproject.toml`.

## Changelog

See [CHANGELOG.md](https://github.com/gradience-ai/gradience/blob/main/CHANGELOG.md) for version history and [GitHub Releases](https://github.com/gradience-ai/gradience/releases) for detailed release notes.

## Security & Responsible Use

Gradience is designed for research and development of efficient language models. Users should:

- **Validate outputs**: Always verify compression results meet your accuracy requirements
- **Resource awareness**: Monitor compute and storage usage, especially in cloud environments  
- **Data privacy**: Ensure compliance with data handling requirements for your datasets
- **Model licensing**: Respect licensing terms of base models and datasets used

For security issues, please see our [security policy](https://github.com/gradience-ai/gradience/blob/main/SECURITY.md).