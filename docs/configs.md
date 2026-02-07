# Configuration Reference

**Complete guide to Gradience benchmark configuration files.**

## Overview

Gradience benchmarks are configured using YAML files that specify models, tasks, LoRA settings, compression policies, and runtime options. This document describes the complete configuration schema with examples.

## Where Configs Live When Installed

**All configs ship with the package** - no need to clone repositories or download additional files:

```python
# Configs are installed with the package
import gradience.bench.configs

# List available configs programmatically
import os
config_dir = os.path.dirname(gradience.bench.configs.__file__)
configs = [f for f in os.listdir(config_dir) if f.endswith('.yaml')]
print(f"Available configs: {len(configs)} files")

# Or via importlib.resources (Python 3.9+)
import importlib.resources
config_files = list(importlib.resources.files(gradience.bench.configs).iterdir())
yaml_configs = [f for f in config_files if f.name.endswith('.yaml')]
```

**Config location in installed package:**
- **Main configs**: `gradience/bench/configs/*.yaml`
- **Evidence pack**: `gradience/bench/configs/evidence/*.yaml`  
- **GPU smoke tests**: `gradience/bench/configs/gpu_smoke/*.yaml`
- **Policies**: `gradience/bench/policies/*.yaml`

## Schema Overview

Benchmark configs use this top-level structure:

```yaml
bench_version: "0.1"        # Config schema version
model: {}                   # Model configuration
task: {}                    # Task/dataset configuration  
lora: {}                    # LoRA adapter settings
compression: {}             # Compression policies
train: {}                   # Training configuration
runtime: {}                 # Runtime/device settings
audit: {}                   # Audit configuration (optional)
```

## Complete Schema Reference

### Model Configuration

```yaml
model:
  name: "distilbert-base-uncased"    # HuggingFace model ID or local path
  type: "sequence_classification"     # Model type (optional)
  torch_dtype: "auto"                # Precision: auto, float16, bfloat16, float32
  gradient_checkpointing: false      # Enable gradient checkpointing for memory
  use_cache: true                    # Use KV cache (set false for training)
```

**Model Types:**
- `"sequence_classification"` - BERT-style classification models
- `"causal_lm"` - GPT-style generative models
- `"auto"` - Auto-detect from model config

### Task Configuration

#### Classification Tasks
```yaml
task:
  dataset: "glue"                    # Dataset name
  subset: "sst2"                     # Dataset subset
  metric: "accuracy"                 # Primary evaluation metric
  eval_max_samples: null             # Limit eval samples (null = all)
  
  # Quality gate for probe training
  probe_gate:
    metric: "accuracy"               # Metric to check
    min_value: 0.75                 # Minimum value to pass
```

#### Generation Tasks
```yaml
task:
  dataset: "gsm8k"
  subset: "main"
  profile: "gsm8k_causal_lm"        # Task profile for evaluation
  eval_max_samples: 500
  
  # Generation parameters
  generation:
    max_new_tokens: 128             # Max tokens to generate
    do_sample: false                # Deterministic generation
    temperature: 0.0                # Sampling temperature
    num_beams: 1                    # Beam search width
  
  probe_gate:
    metric: "exact_match"
    min_value: 0.15                 # Lower threshold for harder tasks
```

### LoRA Configuration

```yaml
lora:
  probe_r: 16                       # Probe rank (baseline for compression)
  alpha: 16                         # LoRA scaling factor (typically = r)
  dropout: 0.0                      # LoRA dropout rate
  
  # Target modules (model-specific)
  target_modules:
    - "q_lin"                       # Query projection (DistilBERT)
    - "k_lin"                       # Key projection
    - "v_lin"                       # Value projection
    - "out_lin"                     # Output projection
  
  # For decoder models (GPT, Llama, Mistral):
  # target_modules:
  #   - "q_proj"
  #   - "k_proj" 
  #   - "v_proj"
  #   - "o_proj"
  #   - "gate_proj"    # MLP gates (Llama/Mistral)
  #   - "up_proj"      # MLP up projection
  #   - "down_proj"    # MLP down projection
```

### Compression Configuration

```yaml
compression:
  # Allowed ranks for compression candidates
  allowed_ranks: [1, 2, 4, 8, 16, 32]
  
  # Quality tolerance (how much accuracy loss is acceptable)
  acc_tolerance: 0.005              # 0.5% accuracy drop threshold
  
  # Mode control
  fast_mode: true                   # Use fast mode (energy_p90, knee_p90, erank_p90 only)
  max_candidates: 4                 # Maximum compression candidates to test
  
  # Explicit policy selection (overrides fast_mode defaults)
  candidate_policies:               
    - "energy_p90"                  # Energy-based rank selection at 90th percentile
    - "knee_p90"                    # Knee detection at 90th percentile
    - "erank_p90"                   # Effective rank at 90th percentile
```

**Available Compression Policies:**
- `"energy_p90"`, `"energy_p95"` - Energy-based rank selection
- `"knee"`, `"knee_p90"` - Elbow/knee detection in singular values
- `"erank"`, `"erank_p90"` - Effective rank (entropy-based)
- `"oht"` - Optimal hard threshold
- `"uniform"` - Uniform rank across all layers

### Training Configuration

```yaml
train:
  # Training control
  seed: 42                          # Random seed
  max_steps: 500                    # Maximum training steps
  eval_steps: 100                   # Evaluate every N steps
  
  # Learning rate and optimization
  learning_rate: 0.00005           # Learning rate (5e-5)
  lr: 0.00005                      # Alias for learning_rate
  warmup_ratio: 0.03               # Warmup ratio (3% of total steps)
  weight_decay: 0.01               # L2 regularization
  
  # Batch sizes
  per_device_train_batch_size: 8   # Training batch size per device
  per_device_eval_batch_size: 32   # Evaluation batch size per device
  gradient_accumulation_steps: 1   # Gradient accumulation
  
  # Data control
  train_samples: null              # Limit training samples (null = all)
  eval_samples: null               # Limit eval samples (null = all)
  
  # Logging and checkpointing
  logging_steps: 10                # Log every N steps
  save_strategy: "no"              # Checkpoint strategy: "no", "steps", "epoch"
  report_to: "none"                # Logging backend: "none", "wandb", "tensorboard"
```

### Runtime Configuration

```yaml
runtime:
  device: "auto"                   # Device: "auto", "cpu", "cuda", "mps"
  
  # Smoke test overrides (when using --smoke flag)
  smoke_max_steps: 150            # Reduced steps for smoke tests
  smoke_train_samples: 800        # Reduced training samples
  smoke_eval_samples: 200         # Reduced eval samples
  
  # Artifact management
  keep_adapter_weights: false     # Keep adapter weights after benchmark
  keep_checkpoints: false         # Keep training checkpoints
```

**Device Options:**
- `"auto"` - Use GPU if available, fallback to CPU
- `"cpu"` - Force CPU usage
- `"cuda"` - Use CUDA GPU (fails if not available)
- `"mps"` - Use Apple Metal Performance Shaders (Apple Silicon)

### Audit Configuration (Optional)

```yaml
audit:
  base_model: "distilbert-base-uncased"  # Base model for UDR computation
  compute_udr: true                      # Compute Update Dominance Ratio
  base_norms_cache: "./cache/norms"      # Cache location for base model norms
```

### Additional Options

```yaml
# Optional metadata
run_type: "my_experiment_v1.0"      # Experiment identifier
bench_version: "0.1"                # Config schema version

# Multi-seed experiments (advanced)
seed: [42, 43, 45]                  # Multiple seeds for statistical validation
```

## Canonical Examples

### Example 1: CPU-Friendly CI Testing

**File**: `gradience/bench/configs/distilbert_sst2_ci.yaml`

```yaml
bench_version: "0.1"

model:
  name: "distilbert-base-uncased"

task:
  dataset: "glue"
  subset: "sst2"
  metric: "accuracy"

train:
  seed: 0
  max_steps: 200
  eval_steps: 50
  lr: 0.00005
  weight_decay: 0.0
  per_device_train_batch_size: 8
  per_device_eval_batch_size: 32

lora:
  probe_r: 16
  alpha: 16
  dropout: 0.0
  target_modules: ["q_lin","k_lin","v_lin","out_lin"]

compression:
  allowed_ranks: [1,2,4,8,16,32]
  acc_tolerance: 0.005

runtime:
  device: "cpu"
  smoke_max_steps: 50
  smoke_train_samples: 200
  smoke_eval_samples: 200
  keep_adapter_weights: false
  keep_checkpoints: false
```

**Usage:**
```bash
gradience-bench --config gradience/bench/configs/distilbert_sst2_ci.yaml \
                --device cpu \
                --output ./demo_run \
                --smoke
```

### Example 2: GPU-Optimized Smoke Test

**File**: `gradience/bench/configs/distilbert_sst2_gpu_smoke.yaml`

```yaml
bench_version: "0.1"

model:
  name: "distilbert-base-uncased"

task:
  dataset: "glue"
  subset: "sst2"
  metric: "accuracy"

train:
  seed: 42
  max_steps: 300
  eval_steps: 100
  lr: 0.00005
  weight_decay: 0.01
  per_device_train_batch_size: 16  # Larger for GPU
  per_device_eval_batch_size: 64

lora:
  probe_r: 16
  alpha: 16
  dropout: 0.0
  target_modules: ["q_lin","k_lin","v_lin","out_lin"]

compression:
  allowed_ranks: [1,2,4,8,16,32]
  acc_tolerance: 0.005
  fast_mode: true

runtime:
  device: "auto"
  smoke_max_steps: 150
  smoke_train_samples: 500
  smoke_eval_samples: 500
  keep_adapter_weights: false
  keep_checkpoints: false
```

**Usage:**
```bash
gradience-bench --config gradience/bench/configs/distilbert_sst2_gpu_smoke.yaml \
                --output ./gpu_demo \
                --smoke
```

## Production Examples

### Example 3: Full Research Benchmark

```yaml
bench_version: "0.1"

model:
  name: "distilbert-base-uncased"
  gradient_checkpointing: true

task:
  dataset: "glue"
  subset: "sst2"
  metric: "accuracy"
  probe_gate:
    metric: "accuracy"
    min_value: 0.75

train:
  seed: 42
  max_steps: 1000
  eval_steps: 200
  learning_rate: 0.00005
  warmup_ratio: 0.1
  weight_decay: 0.01
  per_device_train_batch_size: 16
  per_device_eval_batch_size: 64
  train_samples: 8000
  eval_samples: 1000

lora:
  probe_r: 32                      # Higher rank for more compression headroom
  alpha: 32
  dropout: 0.05
  target_modules: ["q_lin","k_lin","v_lin","out_lin"]

compression:
  allowed_ranks: [2,4,8,12,16,24,32]
  acc_tolerance: 0.01             # Stricter tolerance
  fast_mode: false                # Test all policies
  max_candidates: 6

audit:
  base_model: "distilbert-base-uncased"
  compute_udr: true
  base_norms_cache: "./cache/distilbert_norms"

runtime:
  device: "auto"
  keep_adapter_weights: true      # Keep for analysis
  keep_checkpoints: false
```

### Example 4: Large Model Template

```yaml
bench_version: "0.1"

model:
  name: "mistralai/Mistral-7B-v0.1"
  type: "causal_lm"
  torch_dtype: "bfloat16"
  gradient_checkpointing: true
  use_cache: false

task:
  dataset: "gsm8k"
  subset: "main"
  profile: "gsm8k_causal_lm"
  eval_max_samples: 500
  generation:
    max_new_tokens: 128
    do_sample: false
    temperature: 0.0
  probe_gate:
    metric: "exact_match"
    min_value: 0.15

lora:
  probe_r: 32
  alpha: 64
  dropout: 0.05
  target_modules:
    - "q_proj"
    - "k_proj"
    - "v_proj"
    - "o_proj"
    - "gate_proj"
    - "up_proj"
    - "down_proj"

compression:
  allowed_ranks: [2,4,8,16,20,24,32]
  acc_tolerance: 0.025
  fast_mode: true

train:
  max_steps: 1500
  per_device_train_batch_size: 1
  per_device_eval_batch_size: 8
  gradient_accumulation_steps: 16
  learning_rate: 0.0001
  warmup_ratio: 0.03
  train_samples: 2000

runtime:
  device: "cuda"
  smoke_max_steps: 500
  smoke_train_samples: 500
```

## Using Configs from Installed Package

### Method 1: Direct Path Reference
```bash
# Works after pip install gradience[bench]
gradience-bench --config gradience/bench/configs/distilbert_sst2_ci.yaml --output results/
```

### Method 2: Programmatic Access
```python
import os
import gradience.bench.configs

# Find config files
config_dir = os.path.dirname(gradience.bench.configs.__file__)
ci_config = os.path.join(config_dir, "distilbert_sst2_ci.yaml")
gpu_config = os.path.join(config_dir, "distilbert_sst2_gpu_smoke.yaml")

# Verify files exist
assert os.path.exists(ci_config), "CI config not found in package"
assert os.path.exists(gpu_config), "GPU config not found in package"
```

### Method 3: Copy and Customize
```python
import shutil
import os
import gradience.bench.configs

# Copy shipped config to customize
config_dir = os.path.dirname(gradience.bench.configs.__file__)
source_config = os.path.join(config_dir, "distilbert_sst2_ci.yaml")
custom_config = "./my_custom_config.yaml"

shutil.copy(source_config, custom_config)
print(f"Config copied to {custom_config} - now customize it!")
```

## Config Validation

### Pre-flight Check
```bash
# Validate config before running benchmark
gradience check --config my_config.yaml --verbose
```

### Schema Validation
```python
# Validate config programmatically
import yaml
from pathlib import Path

def validate_config(config_path):
    """Basic config validation."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Check required sections
    required_sections = ["model", "task", "lora", "train"]
    for section in required_sections:
        assert section in config, f"Missing required section: {section}"
    
    # Check required fields
    assert "name" in config["model"], "Missing model.name"
    assert "dataset" in config["task"], "Missing task.dataset"
    assert "probe_r" in config["lora"], "Missing lora.probe_r"
    
    print(f"✅ Config {config_path} is valid")

# Validate shipped configs
validate_config("gradience/bench/configs/distilbert_sst2_ci.yaml")
```

## Common Configuration Patterns

### CPU Development
- Small batch sizes (8-16)
- Shorter training (200-500 steps)
- Device: "cpu"
- Fast mode enabled

### GPU Training  
- Larger batch sizes (16-64)
- Mixed precision (bfloat16)
- Gradient checkpointing
- Device: "auto" or "cuda"

### CI/Testing
- Smoke mode overrides
- Fast mode enabled
- Minimal artifact retention
- Conservative quality gates

### Research
- Full mode for comprehensive comparison
- Higher probe ranks (32+)
- UDR computation enabled
- Multi-seed configurations

### Production
- Strict quality gates
- Resume enabled
- Artifact retention for analysis
- Conservative compression tolerances

## Troubleshooting

### Config Not Found
```bash
# Verify config ships with package
python -c "import gradience.bench.configs; print(gradience.bench.configs.__file__)"

# List available configs
python -c "import os, gradience.bench.configs; print(os.listdir(os.path.dirname(gradience.bench.configs.__file__)))"
```

### Invalid Config
```bash
# Check config syntax
gradience check --config your_config.yaml --verbose
```

### Missing Model
```bash
# Check if model exists on HuggingFace
python -c "from transformers import AutoModel; AutoModel.from_pretrained('your-model-name')"
```

### Device Issues
```bash
# Test device compatibility
gradience-bench --config config.yaml --device cpu --smoke --output test_run/
```