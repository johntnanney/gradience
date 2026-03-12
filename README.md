# Gradience

**Spectral metrics as empirical probes into the geometry of LoRA training.** Measure rank evolution, detect phase transitions, and study how spectral structure relates to generalization -- with reproducible, multi-seed experimental infrastructure.

> If you use Gradience in your research, please cite:
>
> ```bibtex
> @software{gradience2026,
>   title = {Gradience: Spectral Analysis of Low-Rank Adaptation Dynamics},
>   author = {Nanney, John T.},
>   year = {2026},
>   url = {https://github.com/johntnanney/gradience},
>   note = {Version 0.11.0}
> }
> ```

## Research questions Gradience helps you investigate

- **How does effective rank evolve during fine-tuning?** Track stable rank, energy concentration, and utilization ratios across layers and training steps.
- **What geometric signatures distinguish runs that generalize from runs that don't?** Compare spectral profiles across seeds, hyperparameter sweeps, and task families.
- **Can spectral metrics detect phase transitions in training dynamics?** Monitor when rank structure shifts abruptly and correlate with loss landscape changes.
- **How much of the learned subspace is actually used?** Quantify per-layer rank waste and map the gap between allocated and effective dimensionality.
- **When two adapters are merged, do their learned subspaces align or interfere?** Measure principal angles, directional agreement, and magnitude balance between adapter pairs.

## Who it's for

- **ML researchers studying training dynamics** -- Use spectral measurements to probe how low-rank structure emerges and evolves during fine-tuning
- **Researchers comparing adaptation strategies** -- Generate reproducible, statistically rigorous evidence across seeds, ranks, and tasks
- **Practitioners managing adapter inventories** -- Screen adapter quality, assess merge risk, and run preflight checks before deployment

## What you get

- **Adapter QA** -- Structural eligibility screening for individual LoRA adapters, with machine-readable artifacts
- **Merge-risk reporting** -- Pairwise geometric compatibility analysis with per-layer verdicts, strategy recommendations, and risk levels
- **Inventory preflight** -- Aggregated summary across adapters and merge pairs, with strict-QA gating for deployment workflows
- **Spectral measurements** -- Per-layer SVD analysis yielding stable rank, energy concentration, utilization ratios, and rank waste quantification
- **Training telemetry** -- Structured JSONL recording of spectral evolution across training steps
- **Merge compatibility analysis** -- Principal angle and directional agreement measurements between adapter pairs, with per-layer geometric characterization
- **Reproducible experimental infrastructure** -- Multi-seed benchmarking with statistical aggregation, tolerance-based validation, and automated candidate generation

## Install

### Quick Install
```bash
# Core package -- includes torch + safetensors (spectral analysis, merge analysis)
pip install gradience

# Full experimental suite -- adds transformers, datasets, peft
pip install "gradience[bench]"

# Development tools
pip install "gradience[dev]"
```

### PyTorch Variant Selection (Optional)

`pip install gradience` auto-installs a default PyTorch build. To use a specific variant (CPU-only or a particular CUDA version), install PyTorch **before** Gradience:

```bash
# CPU-only (smaller download)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install gradience

# CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install gradience
```

**Common issue**: If you see `ModuleNotFoundError: No module named 'datasets'` or `'transformers'`, you need the bench extras. **Fix**: `pip install "gradience[bench]"`

**[Complete Installation Guide](https://github.com/johntnanney/gradience/blob/main/docs/install.md)** -- GPU setup, RunPod, troubleshooting, cache configuration

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
print(f'Gradience {gradience.__version__} installed successfully')
print('Base package ready for spectral analysis workflows')

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
print('Sample audit data parsed successfully')
print(f'   Mean stable rank: {sample_audit[\"summary\"][\"stable_rank_mean\"]}')
print(f'   Rank utilization: {sample_audit[\"summary\"][\"utilization_mean\"]:.1%}')
"
```

**Zero drama**: These commands work immediately after `pip install gradience` with no additional setup.

## Quickstart: reproduce a spectral analysis in 60 seconds

```bash
# Install with ML dependencies
pip install "gradience[bench]"

# List available experiment configs (included in package)
python -c "
import importlib.resources
configs = importlib.resources.files('gradience.bench.configs')
print('Available configs:')
for f in sorted(configs.iterdir()):
    if f.name.endswith('.yaml'):
        print(f'  {f.name}')
"

# Copy and run a fast CPU experiment (~60 seconds)
python -c "
import importlib.resources, shutil
configs = importlib.resources.files('gradience.bench.configs')
shutil.copy(configs / 'distilbert_sst2_ci.yaml', './cpu_test.yaml')
print('Config copied to cpu_test.yaml')
"

gradience-bench --config cpu_test.yaml --output results/

# Examine results
cat results/bench.json  # Spectral measurements + validation results
cat results/audit.json  # Per-layer spectral analysis
```

**Performance expectations**: First run ~60-90s (model download + training), subsequent runs ~30s (cached).

## Preflight artifacts

The standard preflight workflow produces machine-readable QA and risk artifacts:

- **`*_qa.json`** -- Adapter QA artifact (`gradience.adapter_qa/v1`): structural summary, eligibility status, behavioral evidence
- **`*_report.json`** -- Merge risk report (`gradience.merge_qa_report/v1`): pairwise risk level, recommended strategy, dominant issue
- **`inventory_summary.json`** -- Inventory summary (`gradience.inventory_summary/v1`): aggregated counts across adapters and pairs

## Experimental artifacts

Benchmark runs produce additional experimental records:

- **`audit.json`** -- Per-layer spectral decomposition: stable rank, energy concentration, utilization ratios, rank waste
- **`bench.json`** -- Full validation results with per-seed metrics and statistical comparisons
- **`bench.md`** -- Human-readable summary of findings
- **`bench_aggregate.json`** -- Multi-seed statistical aggregation (mean, std, confidence intervals, effect sizes)
- **`merge_audit.json`** -- Per-layer geometric compatibility between two adapters (principal angles, directional agreement, magnitude balance)
- **`merge_audit.md`** -- Human-readable merge compatibility report with per-layer characterization

## Core commands

```bash
# Measure spectral structure of an existing LoRA adapter
gradience audit --peft-dir /path/to/adapter --json

# Record training telemetry
gradience monitor training_run.jsonl --verbose

# Analyze per-layer rank utilization
gradience audit --peft-dir /path/to/adapter --suggest-per-layer

# Measure geometric compatibility between two adapters
gradience merge-audit --adapter-a ./adapter_a --adapter-b ./adapter_b --output-dir ./merge_out

# Run a complete experimental protocol
gradience-bench --config config.yaml --output results/

# Validate config before running
gradience check --peft adapter_config.json --training training_args.json
```

## Experimental methodology

Gradience implements a four-stage experimental protocol for studying rank structure in LoRA adapters:

1. **Probe** -- Train a high-rank adapter (r=16 or higher) to establish an unconstrained baseline.
   This captures the full spectral structure the task can express before any dimensionality constraints are imposed.

2. **Measure** -- Decompose each adapter weight matrix via SVD. Extract stable rank,
   energy concentration at the 90th percentile, and per-layer utilization ratios.
   A layer using 2 of 16 available rank dimensions reveals that the learned transformation
   is intrinsically low-dimensional for that layer.

3. **Derive** -- Generate rank configurations from the spectral measurements. Multiple
   derivation policies (energy-based, knee detection, uniform) each map metrics to
   per-layer rank settings differently, producing a family of configurations
   to test rather than a single point estimate.

4. **Validate** -- Train each derived configuration with multiple seeds (3+) and compare
   against the probe baseline using a tolerance threshold (e.g., 2% accuracy loss).
   Configurations where all seeds pass are Tier A (statistically reliable).
   Configurations where 2 of 3 seeds pass are Tier B (suggestive but not conclusive).
   Multi-seed validation separates genuine spectral findings from initialization artifacts.

The result is a set of rank configurations with statistical evidence linking spectral
measurements to downstream performance.

## Key concepts

- **Probe adapter** -- High-rank (r=16+) baseline capturing unconstrained spectral structure
- **Spectral audit** -- SVD-based measurement of stable rank, utilization, and energy distribution per layer
- **Candidate derivation** -- Mapping spectral measurements to rank configurations via multiple policies
- **Validation protocol** -- Multi-seed comparison against probe baseline with tolerance thresholds and effect sizes
- **Merge audit** -- Geometric compatibility analysis between adapter pairs via principal angles, directional agreement, and magnitude balance -- characterizes each layer as aligned, redundant, conflicting, or imbalanced

**[Complete documentation](https://github.com/johntnanney/gradience/tree/main/docs/)** | **[API reference](https://github.com/johntnanney/gradience/blob/main/PUBLIC_API.md)** | **[Artifacts & Evidence](https://github.com/johntnanney/gradience/blob/main/docs/artifacts.md)**

## Integration with HuggingFace Transformers

Instrument your training loop with one line to capture spectral telemetry:

```python
from transformers import Trainer
from gradience.vnext.integrations.hf import GradienceCallback

trainer = Trainer(..., callbacks=[GradienceCallback()])
trainer.train()

# Telemetry saved to: <output_dir>/run.jsonl
```

Then analyze the recorded data:

```bash
# Quick overview of training dynamics
gradience monitor output/run.jsonl

# Detailed spectral analysis
gradience monitor output/run.jsonl --verbose

# Measure the resulting adapter
gradience audit --peft-dir output/adapter --layers --json
```

## Default workflow: preflight QA

The standard Gradience workflow screens adapters and merge pairs before deployment:

```bash
# 1. Audit each adapter
gradience audit-adapter --peft-dir ./adapter_a --out qa/adapter_a_qa.json
gradience audit-adapter --peft-dir ./adapter_b --out qa/adapter_b_qa.json

# 2. Assess merge risk (with source QA context)
gradience merge-audit --adapter-a ./adapter_a --adapter-b ./adapter_b \
    --source-a-qa qa/adapter_a_qa.json --source-b-qa qa/adapter_b_qa.json \
    --emit-report reports/ab_report.json

# 3. Aggregate inventory
gradience summarize-inventory --qa-dir qa/ --report-dir reports/ \
    --emit-report inventory/summary.json

# 4. Gate on quality (optional strict mode)
gradience merge-audit --adapter-a ./adapter_a --adapter-b ./adapter_b \
    --source-a-qa qa/adapter_a_qa.json --source-b-qa qa/adapter_b_qa.json --strict-qa
```

See **[Getting Started: Preflight](https://github.com/johntnanney/gradience/blob/main/docs/getting-started-preflight.md)** for the full walkthrough and **[Source QA Workflow](https://github.com/johntnanney/gradience/blob/main/docs/source_qa_workflow.md)** for interpretation examples.

## Experimental workflow

For research into spectral training dynamics:

```bash
# 1. Train with telemetry (or use existing adapter)
python your_training_script.py  # with GradienceCallback

# 2. Inspect training dynamics
gradience monitor output/run.jsonl

# 3. Measure spectral structure
gradience audit --peft-dir output/adapter --suggest-per-layer

# 4. Run experimental protocol (derive candidates + validate with multiple seeds)
gradience-bench --config my_config.yaml --output experiment_results/

# 5. Examine statistical evidence
cat experiment_results/bench_aggregate.json
```

## Configuration

Experiment configs specify model, task, and validation criteria:

```yaml
model:
  name_or_path: "distilbert-base-uncased"

task:
  name: "sst2"
  type: "classification"

lora:
  r: 16              # Probe rank
  target_modules: ["q_lin", "v_lin"]

runtime:
  device: "auto"
  seed: [42, 43, 45]  # Multi-seed validation
```

See **[Configuration Reference](https://github.com/johntnanney/gradience/blob/main/docs/configs.md)** for the full schema including experimental options.

## Requirements

- **Python 3.10+**
- **PyTorch** and **safetensors** (auto-installed with `pip install gradience`)
- **Optional**: transformers, peft, datasets (for benchmarking via `pip install "gradience[bench]"`)

## Performance

- **CLI help**: < 3 seconds (no ML dependencies loaded)
- **Base install**: < 10MB (minimal dependencies)
- **CPU-only operation**: Full functionality without GPU requirements

## What we test in CI

Every release is validated with comprehensive CI gates:

- **Package integrity**: Base install, bench extras, and console scripts work correctly across Python 3.10-3.12
- **Performance guarantees**: CLI help commands complete within documented timing requirements
- **PyPI readiness**: Wheel contents, config accessibility via importlib.resources, and quickstart functionality

**[Complete CI test matrix](https://github.com/johntnanney/gradience/blob/main/.github/workflows/pip-install-ready.yml)** -- See exactly what we validate before each release
**[CI Gates documentation](https://github.com/johntnanney/gradience/blob/main/CI_GATES.md)** -- Detailed testing requirements and performance targets

## What Gradience is

Gradience is a **preflight QA and merge-risk layer** for LoRA adapter decisions, backed by spectral measurement:

- **A structural screening tool** -- It audits individual adapters for spectral health and screens merge pairs for geometric compatibility
- **A merge-risk reporting system** -- It produces machine-readable risk artifacts with per-layer verdicts, strategy recommendations, and strict-QA gating
- **An inventory preflight layer** -- It aggregates adapter-level and pair-level judgments into deployment-ready summaries
- **A research instrument** -- It computes spectral metrics (stable rank, energy concentration, utilization) from adapter weights for training dynamics research
- **A companion to your training stack** -- It instruments and analyzes; it does not replace your trainer, optimizer, or evaluation pipeline
- **A source of evidence, not prescriptions** -- Spectral measurements inform your analysis. The interpretation is yours

## API Stability

**Stable interfaces** (backwards compatible):
- CLI commands (`gradience audit-adapter`, `gradience merge-audit`, `gradience summarize-inventory`, `gradience-bench`, etc.)
- Frozen artifact schemas: `gradience.adapter_qa/v1`, `gradience.merge_qa_report/v1`, `gradience.inventory_summary/v1`
- Python API: `gradience.api.audit_adapter()`, `gradience.api.merge_risk_report()`, `gradience.api.summarize_inventory()`
- Config schema (YAML structure)
- Output artifacts (`audit.json`, `bench.json`, `bench.md`, `merge_audit.json`, `merge_audit.md`)

**Experimental features** (including spectral compression) are clearly marked and may change.

## Examples

- **[Minimal integration](https://github.com/johntnanney/gradience/blob/main/examples/vnext/toy_lora_run.py)** -- Add telemetry to any training script
- **[HF Trainer integration](https://github.com/johntnanney/gradience/blob/main/examples/vnext/hf_trainer_example.py)** -- End-to-end training with spectral telemetry
- **[Experiment configs](https://github.com/johntnanney/gradience/tree/main/examples/configs/)** -- Define experimental protocols

## Documentation

Complete documentation available on GitHub:

- **[Theoretical Foundations](https://github.com/johntnanney/gradience/blob/main/docs/THEORY.md)** -- Mathematical framework and open questions
- **[Empirical Findings](https://github.com/johntnanney/gradience/blob/main/docs/FINDINGS.md)** -- Results obtained with Gradience
- **[Research Roadmap](https://github.com/johntnanney/gradience/blob/main/docs/ROADMAP.md)** -- Open questions and planned investigations
- **[Experiment Guide](https://github.com/johntnanney/gradience/blob/main/docs/USER_MANUAL.md)** -- Designing and running spectral studies
- **[Statistical Methodology](https://github.com/johntnanney/gradience/blob/main/docs/VALIDATION_POLICY.md)** -- Validation rigor requirements
- **[Spectral Analysis Policies](https://github.com/johntnanney/gradience/blob/main/docs/RANK_POLICIES_GUIDE.md)** -- Interpretive guide for rank metrics
- **[Installation Guide](https://github.com/johntnanney/gradience/blob/main/docs/install.md)** -- Complete setup guide with troubleshooting
- **[Source QA Workflow](https://github.com/johntnanney/gradience/blob/main/docs/source_qa_workflow.md)** -- Assess adapter quality before merging
- **[CLI Reference](https://github.com/johntnanney/gradience/blob/main/docs/cli.md)** -- Complete command-line reference and examples
- **[Configuration Reference](https://github.com/johntnanney/gradience/blob/main/docs/configs.md)** -- YAML config schema and examples
- **[Artifacts & Evidence](https://github.com/johntnanney/gradience/blob/main/docs/artifacts.md)** -- Understanding experimental outputs
- **[Troubleshooting](https://github.com/johntnanney/gradience/blob/main/docs/troubleshooting.md)** -- Common issues and solutions

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](https://github.com/johntnanney/gradience/blob/main/LICENSE) for details.

## Citation

**APA Style:** Nanney, J. T. (2026). *Gradience: Spectral analysis of low-rank adaptation dynamics* (Version 0.11.0) [Computer software]. https://github.com/johntnanney/gradience

**Note for maintainers:** Update version numbers in citation when releasing new versions. Current version should match `pyproject.toml`.

## Changelog

See [CHANGELOG.md](https://github.com/johntnanney/gradience/blob/main/CHANGELOG.md) for version history and [GitHub Releases](https://github.com/johntnanney/gradience/releases) for detailed release notes.

## Security & Responsible Use

Gradience is designed for research into the spectral structure of fine-tuned language models. Users should:

- **Validate findings**: Always verify spectral measurements against downstream task metrics
- **Resource awareness**: Monitor compute and storage usage, especially in cloud environments
- **Data privacy**: Ensure compliance with data handling requirements for your datasets
- **Model licensing**: Respect licensing terms of base models and datasets used

For security issues, please see our [security policy](https://github.com/johntnanney/gradience/blob/main/SECURITY.md).
