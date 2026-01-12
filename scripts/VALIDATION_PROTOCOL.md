# Validation Protocol for Rank Suggestions

This directory contains a CPU-only validation protocol for testing rank compression recommendations from Gradience.

## Overview

The validation protocol follows this workflow:

1. **Train Probe** - Train a LoRA adapter at high rank (r=16 or r=32) on a tiny dataset
2. **Generate Suggestions** - Use `gradience audit --suggest-per-layer` to get rank recommendations  
3. **Retrain with Strategies** - Test three different compression strategies:
   - `uniform_p90`: Single rank based on global 90th percentile (conservative)
   - `module_p90`: Per-module-type ranks based on 90th percentiles (moderate)
   - `per_layer`: Full per-layer rank pattern (experimental)
4. **Compare Results** - Evaluate parameter reduction vs. performance trade-offs

## Quick Start

```bash
# Run with defaults (tiny-distilbert, r=16)
./scripts/validate_suggestions.sh

# Custom configuration
./scripts/validate_suggestions.sh --probe-r 32 --model tiny-bert --dataset cola

# Quick validation (skip detailed evaluation) 
./scripts/validate_suggestions.sh --quick
```

## Files

### Core Scripts

- `validation_protocol.py` - Main validation protocol implementation
- `validate_suggestions.sh` - Convenient shell wrapper
- `training_integration_example.py` - Example integrations for different frameworks

### Configuration

The protocol uses minimal configurations designed for CPU validation:
- Very small datasets (100 training samples)
- Short training (50 steps)
- Small models (tiny variants)
- CPU-optimized settings

## Usage Examples

### Basic Validation

```bash
# Validate rank suggestions for DistilBERT
python scripts/validation_protocol.py \
    --model tiny-distilbert \
    --dataset tiny \
    --probe-r 16 \
    --base-dir ./validation_runs
```

### Custom Training Integration

To integrate with your training framework, modify the `_get_training_command()` method in `validation_protocol.py`:

```python
def _get_training_command(self, config_path: Path, output_dir: Path) -> List[str]:
    return [
        "python", "your_training_script.py",
        f"--config={config_path}",
        f"--output-dir={output_dir}",
        "--do-train", "--do-eval"
    ]
```

See `training_integration_example.py` for patterns with:
- Hugging Face Transformers + PEFT
- Custom PyTorch training loops
- Different model architectures

## Output Structure

```
validation_runs/
└── validation_{model}_{dataset}_r{probe_r}/
    ├── evaluation.json              # Summary and recommendations
    ├── probe_audit.json            # Full audit data from probe
    ├── probe_r16/                  # Original high-rank model
    │   ├── peft/                   # LoRA adapter
    │   ├── training_config.json    # Training configuration
    │   └── trainer_state.json      # Training metrics
    ├── retrain_uniform_p90/        # Conservative uniform strategy
    ├── retrain_module_p90/         # Moderate per-module strategy  
    └── retrain_per_layer/          # Experimental per-layer strategy
```

## Key Metrics

The protocol tracks:
- **Parameter reduction**: `(original_params - compressed_params) / original_params`
- **Training success**: Whether each strategy completed without errors
- **Memory estimates**: Approximate memory usage for each strategy

## Interpretation

### Strategy Safety Levels

1. **uniform_p90** (Safest)
   - Uses single rank across all layers
   - Based on global 90th percentile
   - Least risk of performance degradation
   - Moderate parameter reduction

2. **module_p90** (Moderate)
   - Different ranks per module type (attention, MLP, etc.)
   - Based on 90th percentiles within module types
   - Balanced risk/reward
   - Better parameter reduction

3. **per_layer** (Experimental)  
   - Unique rank per layer based on individual analysis
   - Maximum parameter reduction potential
   - Highest risk of performance issues
   - Requires careful validation

### Recommendations

The protocol generates recommendations like:
```
• Safest approach: uniform_p90 (35% reduction)
• Best reduction: per_layer (67% reduction)
• Per-layer experimental: true (67% reduction)
```

Use these to guide your compression strategy choice.

## Extending the Protocol

### Adding New Models

1. Add model mapping in `_get_model_path()`:
```python
model_map = {
    "your-model": "path/to/your/model",
    # ...
}
```

2. Update target modules in `create_tiny_training_config()`:
```python
"lora_target_modules": ["your", "target", "modules"]
```

### Adding New Datasets

1. Add dataset mapping in `_get_dataset_config()`:
```python
dataset_map = {
    "your-dataset": "dataset_name_or_path",
    # ...
}
```

2. Update task configuration in `_get_task_name()`.

### Custom Training Scripts

Replace the mock training in `_get_training_command()` with your actual training command. The script expects:
- PEFT adapter saved to `{output_dir}/peft/`
- Config at `{output_dir}/peft/adapter_config.json`
- Weights at `{output_dir}/peft/adapter_model.{bin|safetensors}`

## Troubleshooting

### Common Issues

1. **Import errors**: Make sure Gradience is installed and importable
2. **Training failures**: Check that your training script handles the generated config
3. **Missing audit data**: Ensure PEFT outputs are in expected format
4. **Memory issues**: Reduce batch size or sequence length for your hardware

### Debugging

Enable verbose mode for detailed logging:
```bash
python scripts/validation_protocol.py --verbose
```

Check individual output directories for training logs and error messages.

## Performance Notes

This protocol is designed for **validation only**, not production training:
- Uses tiny datasets and models
- Very short training runs  
- Minimal hyperparameter tuning
- CPU-optimized (no GPU required)

For production use, scale up the training configuration while keeping the same validation methodology.