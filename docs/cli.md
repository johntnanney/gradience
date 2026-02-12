# CLI Reference

**Complete command-line reference for Gradience core and benchmarking tools.**

## Overview

Gradience provides two main CLI tools:

- **`gradience`** - Core audit, monitor, and analysis tools
- **`gradience-bench`** - Complete benchmarking protocol with compression validation

## Main CLI: gradience

### Usage
```bash
gradience [-h] {verify,report,check,audit,explain,truncate,monitor} ...
```

**Description**: Spectral telemetry and restraint-first diagnostics for neural network training

### Subcommands

| Command | Purpose |
|---------|---------|
| `verify` | Verify installation and dependencies |
| `report` | Generate report from telemetry |
| `check` | Validate config and emit restraint-first recommendations |
| `audit` | Audit PEFT LoRA adapter for rank/utilization waste |
| `explain` | Explain disagreement analysis for specific layer from audit JSON |
| `truncate` | SVD truncate PEFT LoRA adapter to smaller rank |
| `monitor` | Analyze vNext telemetry JSONL and emit alerts/recommendations |

### gradience audit

**Most commonly used subcommand for analyzing LoRA efficiency.**

```bash
gradience audit --peft-dir PEFT_DIR [OPTIONS]
```

#### Required Arguments
- `--peft-dir PEFT_DIR` - Path to PEFT output directory (containing adapter_config.* and weights)

#### Optional Arguments

**Output Control:**
- `--json` - Output JSON instead of pretty text
- `--layers` - Include per-layer audit rows in JSON output (can be large)
- `--suggest-per-layer` - Include per-layer rank suggestions (requires `--layers`)

**Analysis Options:**
- `--top-wasteful N` - Print N most wasteful layers (lowest utilization). 0 disables
- `--top-singular-values N` - Include top-k singular values per layer in JSON output
- `--base-model BASE_MODEL` - Base model ID for UDR computation (e.g., 'microsoft/DialoGPT-medium')
- `--base-norms-cache PATH` - Save/load base model norms cache (speeds up repeated audits)
- `--no-udr` - Skip UDR computation even if base model available

**Rank Policy Configuration:**
- `--rank-policies POLICIES` - Rank selection policies. Comma/space-separated. Default: `energy@0.90,knee,erank`
  - Available: `energy@0.90`, `energy@0.95`, `knee`, `erank`, `oht`

**Advanced Options:**
- `--importance-quantile Q` - Quantile threshold for energy filtering (default: 0.75)
- `--importance-uniform-mult-gate G` - Uniform multiplier gate (default: 1.5)
- `--importance-metric {energy_share,frobenius_norm,param_weighted}` - Importance calculation method
- `--disagreement-rationale {full,flagged_only}` - Detail level for JSON rationale

**Integration:**
- `--append JSONL_FILE` - Append audit metrics to existing vNext telemetry JSONL

#### Examples

```bash
# Basic audit with recommendations
gradience audit --peft-dir ./my_adapter

# JSON output with per-layer analysis
gradience audit --peft-dir ./my_adapter --json --layers

# Rank compression suggestions
gradience audit --peft-dir ./my_adapter --suggest-per-layer --json

# Audit with base model for UDR computation
gradience audit --peft-dir ./my_adapter --base-model distilbert-base-uncased

# Show top 5 most wasteful layers
gradience audit --peft-dir ./my_adapter --top-wasteful 5

# Append to existing training telemetry
gradience audit --peft-dir ./my_adapter --append ./training_run.jsonl
```

### gradience monitor

**Analyze training telemetry from GradienceCallback.**

```bash
gradience monitor FILE [OPTIONS]
```

#### Required Arguments
- `FILE` - Path to vNext telemetry JSONL file

#### Optional Arguments
- `--gap-threshold RATIO` - Train/test PPL ratio threshold for memorization warning (default: 1.5)
- `--strict-schema` - Fail on validation issues instead of skipping bad lines
- `--verbose` - Print rationale/evidence and telemetry issues
- `--json` - Output JSON instead of pretty text

#### Examples

```bash
# Quick overview of training run
gradience monitor ./output/run.jsonl

# Detailed analysis with rationale
gradience monitor ./output/run.jsonl --verbose

# JSON output for automation
gradience monitor ./output/run.jsonl --json --verbose
```

### gradience check

**Pre-flight config validation.**

```bash
gradience check [CONFIG] [OPTIONS]
```

#### Arguments
- `CONFIG` - Path to config JSON/YAML (optional if using --peft/--training)

#### Options
- `--peft PATH` - Path to PEFT adapter_config.json/yaml
- `--training PATH` - Path to training_args.json/yaml
- `--peft-dir DIR` - Auto-detect adapter_config.json in directory
- `--training-dir DIR` - Auto-detect training_args.json in directory
- `--task TASK` - Task name (e.g., gsm8k, sst2)
- `--model MODEL` - Override model name
- `--dataset DATASET` - Override dataset name
- `--verbose` - Print rationale/evidence
- `--json` - JSON output

## Benchmarking CLI: gradience-bench

**Complete LoRA compression benchmarking protocol.**

### Usage

```bash
python -m gradience.bench.run_bench --config CONFIG --output OUTPUT [OPTIONS]
```

Or using the console script:
```bash
gradience-bench --config CONFIG --output OUTPUT [OPTIONS]
```

### Required Arguments

- `--config CONFIG` - Path to YAML config file (e.g., `configs/distilbert_sst2.yaml`)
- `--output OUTPUT` - Output directory for benchmark results

### Optional Arguments

#### Mode Control
- `--smoke` - Run in smoke mode (uses `smoke_*` limits from config for faster testing)
- `--ci` - CI mode: exit non-zero if compression strategies FAIL
- `--full-mode` - Full mode: test all policy variants (default: fast mode)

#### Runtime Control
- `--device {cpu,mps,cuda}` - Override device from config
- `--verbose, -v` - Verbose output during execution
- `--resume` - Resume from completed stages (skips expensive operations)

#### Candidate Control
- `--max-candidates N` - Maximum number of compression candidates to test (default: 4)

#### Artifact Management
- `--keep-artifacts` - Keep all artifacts (default: True)
- `--clean-on-pass` - Clean artifacts if all strategies pass (future feature)

### Mode Differences

#### Fast Mode (Default)
- **Policies tested**: `energy_p90`, `knee_p90`, `erank_p90` only
- **Candidates**: ~3 compression variants
- **Speed**: Fastest benchmark execution
- **Use case**: Standard validation, CI pipelines

```bash
# Fast mode (default)
gradience-bench --config config.yaml --output results/
```

#### Full Mode
- **Policies tested**: All available policy variants
- **Candidates**: All policies + legacy suggestions (capped at `--max-candidates`)
- **Speed**: Slower due to more compression variants
- **Use case**: Research, comprehensive comparison

```bash
# Full mode
gradience-bench --config config.yaml --output results/ --full-mode
```

### Resume Functionality

The `--resume` flag allows restarting interrupted benchmarks without re-running expensive stages:

#### How Resume Works
1. **State tracking**: Progress saved to `<output>/stage_state.json`
2. **Stage detection**: Automatically detects completed stages
3. **Expensive operations skipped**: Probe training (90+ minutes), model downloads
4. **Safe restart**: Validates existing outputs before resuming

#### Resume Examples

```bash
# Start benchmark
gradience-bench --config config.yaml --output results/exp001/

# If interrupted, resume from last completed stage
gradience-bench --config config.yaml --output results/exp001/ --resume
```

#### State Storage
Resume state stored in `<output>/stage_state.json`:
```json
{
  "completed_stages": ["probe_training", "probe_audit"],
  "current_stage": "compression_candidate_generation",
  "timestamp": "2024-01-15T10:30:00Z",
  "config_hash": "abc123...",
  "artifacts": {
    "probe_r16/": "completed",
    "audit.json": "exists",
    "compression_configs.json": "pending"
  }
}
```

#### Resume Safety
- **Config validation**: Ensures config hasn't changed
- **Artifact verification**: Checks required files exist
- **Stage dependencies**: Validates prerequisite stages completed
- **Clean restart**: Falls back to full run if state corrupted

### Exit Codes

Gradience uses exit codes for automation and CI integration:

| Exit Code | Meaning | When It Occurs |
|-----------|---------|----------------|
| **0** | Success | All operations completed successfully |
| **1** | General failure | Config errors, missing files, benchmark failures |
| **2** | Undertrained probe | Probe didn't reach quality threshold (non-smoke mode) |
| **130** | User interruption | Ctrl+C / SIGINT received |

#### CI Mode Exit Codes

When using `--ci` flag:

| Exit Code | Meaning | Condition |
|-----------|---------|-----------|
| **0** | CI PASS | At least one compression strategy passed quality gates |
| **1** | CI FAIL | No compression strategies passed, or all strategies failed |

#### Exit Code Examples

```bash
# Check exit code in shell scripts
gradience-bench --config config.yaml --output results/ --ci
if [ $? -eq 0 ]; then
    echo "✅ Compression strategies validated"
else
    echo "❌ Compression validation failed"
    exit 1
fi
```

```python
# Check exit code in Python
import subprocess
result = subprocess.run([
    "gradience-bench", "--config", "config.yaml", 
    "--output", "results/", "--ci"
])

if result.returncode == 0:
    print("✅ Benchmark passed")
elif result.returncode == 2:
    print("⚠️ Probe undertrained (extend training)")
else:
    print("❌ Benchmark failed")
```

### Examples

#### Basic Benchmark
```bash
# Standard CPU benchmark
gradience-bench --config configs/distilbert_sst2.yaml --output results/run_001
```

#### Fast Development Testing
```bash
# Quick smoke test
gradience-bench --config configs/distilbert_sst2_ci.yaml \
                --output smoke_test \
                --smoke \
                --device cpu
```

#### Production Validation
```bash
# CI mode with specific device
gradience-bench --config configs/production_config.yaml \
                --output validation/$(date +%Y%m%d_%H%M%S) \
                --ci \
                --device cuda \
                --verbose
```

#### Research Mode
```bash
# Full policy comparison
gradience-bench --config configs/research_config.yaml \
                --output research/full_comparison \
                --full-mode \
                --max-candidates 8 \
                --verbose
```

#### Resume Interrupted Run
```bash
# Resume after interruption
gradience-bench --config configs/long_experiment.yaml \
                --output experiments/exp_001 \
                --resume \
                --verbose
```

### Configuration

Benchmark behavior controlled via YAML config files:

```yaml
# Example config structure
model:
  name_or_path: "distilbert-base-uncased"

task:
  name: "sst2"
  type: "classification"

lora:
  probe_r: 16
  target_modules: ["q_lin", "v_lin"]

compression:
  fast_mode: true              # Override CLI --full-mode
  max_candidates: 4            # Override CLI --max-candidates
  candidate_policies:          # Override default fast mode policies
    - "energy_p90"
    - "knee_p90"
  acc_tolerance: 0.02          # Quality loss threshold

runtime:
  device: "auto"
  smoke_max_steps: 50          # For --smoke mode
```

### Performance Tips

#### For Faster Development
```bash
# Use smoke mode + CPU + resume
gradience-bench --config config.yaml --output dev_test/ --smoke --device cpu --resume
```

#### For CI Pipelines  
```bash
# Fast mode + CI validation
gradience-bench --config ci_config.yaml --output ci_results/ --ci --device cpu
```

#### For Production Validation
```bash
# Full mode + resume capability
gradience-bench --config prod_config.yaml --output prod_val/ --full-mode --resume --ci
```

## Integration Examples

### Shell Scripts

```bash
#!/bin/bash
set -e

# Run benchmark with error handling
echo "🚀 Starting benchmark..."
if gradience-bench \
    --config "configs/production.yaml" \
    --output "results/$(date +%Y%m%d_%H%M%S)" \
    --ci \
    --device cuda; then
    echo "✅ Benchmark passed - compression strategies validated"
    # Deploy compressed models
    ./deploy_compressed_models.sh
else
    echo "❌ Benchmark failed - keeping original models"
    exit 1
fi
```

### Python Integration

```python
import subprocess
import json
from pathlib import Path

def run_gradience_audit(peft_dir: Path) -> dict:
    """Run audit and return JSON results."""
    result = subprocess.run([
        "gradience", "audit", 
        "--peft-dir", str(peft_dir),
        "--json", "--layers"
    ], capture_output=True, text=True)
    
    if result.returncode != 0:
        raise RuntimeError(f"Audit failed: {result.stderr}")
    
    return json.loads(result.stdout)

def run_benchmark(config: Path, output: Path) -> bool:
    """Run benchmark and return success status."""
    result = subprocess.run([
        "gradience-bench",
        "--config", str(config),
        "--output", str(output),
        "--ci", "--verbose"
    ], capture_output=True, text=True)
    
    return result.returncode == 0
```

### GitHub Actions

```yaml
name: LoRA Compression Validation
on: [push, pull_request]

jobs:
  validate-compression:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    
    - name: Setup Python
      uses: actions/setup-python@v4
      with:
        python-version: "3.11"
    
    - name: Install Gradience
      run: |
        pip install torch --index-url https://download.pytorch.org/whl/cpu
        pip install "gradience[bench]"
    
    - name: Run compression benchmark
      run: |
        gradience-bench \
          --config configs/ci_validation.yaml \
          --output ci_results/ \
          --ci \
          --device cpu \
          --smoke
    
    - name: Upload results
      if: always()
      uses: actions/upload-artifact@v3
      with:
        name: benchmark-results
        path: ci_results/
```

## Troubleshooting

### Common Issues

**Long help times**: First run of `gradience-bench --help` loads ML libraries (~15s normal)
**Resume failures**: Delete `stage_state.json` to force clean restart
**Exit code 2**: Extend training steps or use `--smoke` for testing
**Device errors**: Use `--device cpu` for compatibility

### Debug Commands

```bash
# Verify installation
gradience verify

# Test with minimal config
gradience-bench --config configs/distilbert_sst2_ci.yaml --output debug_test/ --smoke --verbose

# Check stage state
cat results/stage_state.json

# Clean restart
rm results/stage_state.json && gradience-bench --config config.yaml --output results/ --resume
```