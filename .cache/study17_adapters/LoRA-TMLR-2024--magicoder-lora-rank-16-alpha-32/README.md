# Tiny LoRA Adapter Example

This is a minimal synthetic LoRA adapter for testing and demonstration purposes.

## 📋 Specifications

- **Base Model**: microsoft/DialoGPT-small (117M parameters)
- **LoRA Rank (r)**: 8
- **Alpha**: 16
- **Target Modules**: c_attn, c_proj (attention layers)
- **Dropout**: 0.1
- **Total LoRA Parameters**: 221,184

## 🎯 Purpose

This adapter is **synthetic** (random weights) and designed for:

- **Testing audit functionality** without downloading real models
- **Demonstrating LoRA structure** and file formats
- **Quick smoke tests** during development
- **Schema validation** for PEFT compatibility

## 🚀 Usage Examples

### Audit the Adapter
```bash
# Basic audit
gradience audit examples/adapters/tiny_lora/

# With rank suggestions
gradience audit examples/adapters/tiny_lora/ --suggest

# JSON output for programmatic use
gradience audit examples/adapters/tiny_lora/ --json
```

### Expected Audit Output
```bash
# Should show:
# - 6 layers (DialoGPT-small depth)
# - 2 modules per layer (c_attn, c_proj)  
# - Rank 8 for all modules
# - Utilization and energy rank metrics
# - Conservative compression suggestions
```

### Integration with Tests
```python
from gradience.vnext.audit import audit_lora_peft_dir

# Use in tests
result = audit_lora_peft_dir("examples/adapters/tiny_lora/")
assert result.config.r == 8
assert len(result.layers) == 12  # 6 layers × 2 modules
```

## ⚠️ Important Notes

- **Synthetic weights**: This adapter was not trained and will not produce meaningful outputs
- **Testing only**: Use real adapters for actual model compression
- **Minimal size**: Kept small for fast testing and repository cleanliness
- **Standard format**: Compatible with HuggingFace PEFT library

## 📁 Files

- `adapter_config.json`: PEFT configuration (task type, target modules, hyperparameters)
- `adapter_model.safetensors`: LoRA weight matrices (A and B matrices for each target module)
- `README.md`: This documentation

## 🔍 Inspection

```bash
# View config
cat examples/adapters/tiny_lora/adapter_config.json | jq .

# Check file sizes
ls -lh examples/adapters/tiny_lora/

# Verify tensor shapes (requires Python)
python -c "
from safetensors.torch import load_file
weights = load_file('examples/adapters/tiny_lora/adapter_model.safetensors')
for name, tensor in weights.items():
    print(f'{name}: {tensor.shape}')
"
```