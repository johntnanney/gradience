# Gradience vNext Examples

Copy-pasteable examples demonstrating Gradience integrations with popular ML frameworks.

## Examples

### `hf_trainer_example.py` - Minimal "One Line" Integration ⭐

**What it does:**
- Demonstrates the simplest possible Gradience integration  
- Trains a tiny model in ~30 seconds on CPU
- Shows exact next steps: monitor → audit → suggestions
- Perfect for copy-pasting into blog posts and documentation

**Usage:**
```bash
python examples/vnext/hf_trainer_example.py
# Outputs: ./gradience_example_output/run.jsonl + adapter files
# Shows: Complete workflow commands to run next
```

**Key features:**
- ✅ **One line integration**: `callbacks=[GradienceCallback()]`
- ✅ **Instant gratification**: Runs in 30 seconds on any CPU
- ✅ **Copy-pasteable**: Blog-ready example with clear next steps
- ✅ **Educational**: Shows the core value proposition

### `hf_trainer_run.py` - Complete HuggingFace + PEFT Integration

**What it does:**
- Minimal CPU-friendly training example with HuggingFace Trainer + LoRA
- Automatically generates Gradience telemetry during training
- Produces audit-ready PEFT adapters 

**Requirements:**
```bash
pip install transformers datasets peft
```

**Usage:**
```bash
# Run the example
python examples/vnext/hf_trainer_run.py

# Check outputs
ls hf_example_output/
# → run.jsonl (telemetry), adapter_config.json, adapter_model.bin

# Audit the trained model
python -m gradience audit --peft-dir hf_example_output

# Get rank suggestions  
python -m gradience audit --peft-dir hf_example_output --layers --suggest-per-layer --json
```

**Key features:**
- ✅ **Drop-in integration**: Just add `GradienceCallback` to your trainer
- ✅ **CPU-optimized**: Runs on any machine, no GPU required
- ✅ **Minimal example**: ~50 lines of copy-pasteable code
- ✅ **Full workflow**: Training → telemetry → audit → suggestions

**Copy-paste integration:**
```python
from gradience.vnext.integrations.hf import GradienceCallback, GradienceCallbackConfig

# Add to your existing trainer
config = GradienceCallbackConfig(
    dataset_name="your_dataset",
    task_profile="your_task_type", 
    notes="your experiment notes"
)
trainer.add_callback(GradienceCallback(config))
```

## Output Structure

All examples produce this structure:
```
{example}_output/
├── run.jsonl                 # Gradience telemetry (vNext format)
├── adapter_config.json       # PEFT configuration  
├── adapter_model.bin         # LoRA weights
└── trainer_state.json        # Framework training state
```

## Integration Patterns

### 1. **Minimal Integration** (for existing projects)
```python
trainer.add_callback(GradienceCallback())
# Writes to training_args.output_dir/run.jsonl
```

### 2. **Custom Configuration**
```python
config = GradienceCallbackConfig(
    output_dir="./my_experiment",
    dataset_name="glue/cola",
    task_profile="easy_classification",
    notes="Testing rank compression"
)
trainer.add_callback(GradienceCallback(config))
```

### 3. **Privacy-conscious**
```python
config = GradienceCallbackConfig(
    telemetry_allow_text=False,     # Redact text fields
    telemetry_max_str_len=128       # Limit string lengths
)
```

## Next Steps

After running examples:

1. **Monitor training**: `python -m gradience monitor {output_dir}/run.jsonl`
2. **Audit efficiency**: `python -m gradience audit --peft-dir {output_dir}`  
3. **Get suggestions**: `python -m gradience audit --peft-dir {output_dir} --layers --suggest-per-layer --json`
4. **Validate compression**: Use `scripts/validate_suggestions.sh`

## Troubleshooting

**Import errors**: Make sure dependencies are installed and you're in the correct environment.

**Memory issues**: Examples are CPU-optimized but you can reduce batch size further if needed.

**Permission errors**: Examples write to `./` - make sure you have write permissions.

**Framework versions**: Examples work with recent versions of transformers/peft. Check compatibility if you see API errors.
