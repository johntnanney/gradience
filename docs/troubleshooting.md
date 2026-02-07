# Troubleshooting Guide

**Symptoms → fixes for common Gradience issues.**

This guide covers the most frequent problems encountered when using Gradience, with practical solutions based on real user experiences.

## Installation & Dependencies

### "No module named 'datasets'"

**Symptoms:**
```
ImportError: No module named 'datasets'
ModuleNotFoundError: No module named 'transformers'
```

**Cause:** You installed the base package (`pip install gradience`) which doesn't include ML dependencies.

**Fix:**
```bash
# Install with bench extras (bench)
pip install "gradience[bench]"

# Or if already installed: (bench)
pip install datasets transformers torch

# Or upgrade existing installation: (bench)
pip install --upgrade "gradience[bench]"
```

**Why this happens:** The base package is lightweight for audit-only workflows. Benchmarking requires the `[bench]` extras which include transformers, datasets, and other ML dependencies.

### "No module named 'transformers'"

**Symptoms:**
```
ImportError: No module named 'transformers'
AttributeError: module 'transformers' has no attribute 'AutoModel'
```

**Cause:** You installed the base package without ML dependencies.

**Fix:**
```bash
# Install bench extras (includes transformers) (bench)
pip install "gradience[bench]"

# Verify installation (bench)
python -c "import transformers; print(transformers.__version__)"

# Or upgrade existing installation: (bench)
pip install --upgrade "gradience[bench]"
```

**Prevention:** Always install `gradience[bench]` for any benchmarking workflows. The base package is only for standalone audit operations.

### Package installation fails with dependency conflicts

**Symptoms:**
```
ERROR: pip's dependency resolver does not currently consider all the packages that are installed
ERROR: Could not find a version that satisfies the requirement torch>=1.9.0
```

**Cause:** Conflicting package versions or Python environment issues.

**Fix:**
```bash
# Create clean environment (base)
python -m venv fresh-env
source fresh-env/bin/activate  # On Windows: fresh-env\Scripts\activate

# Install PyTorch first for better compatibility
pip install torch --index-url https://download.pytorch.org/whl/cpu  # or cu121 for CUDA

# Install with specific Python version (bench)
pip install --upgrade pip
pip install "gradience[bench]"

# Or force upgrade all dependencies (bench)
pip install --upgrade --force-reinstall "gradience[bench]"
```

**PyTorch Note:** Installing PyTorch separately before `gradience[bench]` often resolves dependency conflicts. Use the appropriate index URL for your hardware (CPU vs CUDA).

## CLI & Performance Issues

### "gradience bench --help" times out or takes 30+ seconds **(bench)**

**Symptoms:**
```bash
$ gradience bench --help  # (bench)
# Hangs for 30+ seconds before showing help
```

**Cause:** Lazy imports loading heavy ML dependencies on first help call.

**Fix:**
This is a **known issue**. Workarounds:
```bash
# First run will be slow (normal) (bench)
gradience bench --help

# Subsequent runs are fast (bench)
gradience bench --help  # Now fast

# Or use base command for quick help (base)
gradience --help  # Always fast
```

**Technical note:** This occurs because help generation triggers import of transformers/torch libraries. This is expected behavior for ML tools.

### CLI commands not found after installation

**Symptoms:**
```bash
$ gradience
command not found: gradience
```

**Cause:** Console scripts not installed or PATH issues.

**Fix:**
```bash
# Verify installation (base)
pip show gradience

# Check if script installed (base)
which gradience

# If missing, reinstall with --force-reinstall (base)
pip install --force-reinstall gradience

# Or use module syntax (base)
python -m gradience.cli --help
```

## Adapter & Model Issues

### "adapter_model.bin missing" or "No such file or directory"

**Symptoms:**
```
FileNotFoundError: [Errno 2] No such file or directory: '/path/to/adapter_model.bin'
RuntimeError: Error loading adapter weights
```

**Cause:** Missing adapter files for testing or benchmarking.

**Fix:**

**For testing/development:**
```bash
# Generate sample adapters using test suite (bench)
python -m pytest tests/test_adapters.py -v
# This creates sample adapters in tests/fixtures/

# Or create minimal test adapter (bench)
python -c "
import torch
from pathlib import Path
Path('test_adapter').mkdir(exist_ok=True)
torch.save({'weight': torch.randn(10, 10)}, 'test_adapter/adapter_model.bin')
"
```

**For production benchmarks:**
```bash
# Run full bench process to generate adapters (bench)
gradience bench configs/distilbert_sst2_ci.yaml --output-dir results/
# This creates real adapters as part of the benchmark process

# Or download pre-trained adapters from HuggingFace (bench)
huggingface-cli download microsoft/DialoGPT-medium --include="adapter*"
```

### "PEFT adapter not found" or adapter loading errors

**Symptoms:**
```
ValueError: Can't find adapter config file
RuntimeError: Adapter weights do not match model architecture
```

**Cause:** Incompatible adapter or missing configuration.

**Fix:**
```bash
# Verify adapter structure (bench)
ls -la your_adapter_path/
# Should contain: adapter_config.json, adapter_model.bin (or .safetensors)

# Check adapter compatibility (bench)
python -c "
import json
with open('your_adapter_path/adapter_config.json') as f:
    config = json.load(f)
    print('Base model:', config.get('base_model_name_or_path'))
    print('PEFT type:', config.get('peft_type'))
"

# Regenerate adapter if corrupted (bench)
rm -rf problematic_adapter_path/
# Re-run bench to recreate
```

## Storage & Environment Issues

### "No space left on device" (disk full)

**Symptoms:**
```
OSError: [Errno 28] No space left on device
torch.RuntimeError: Could not allocate tensor
```

**Cause:** HuggingFace cache or results filling disk.

**Fix:**

**Emergency cleanup:**
```bash
# Check disk usage
df -h

# Clean HuggingFace cache
rm -rf ~/.cache/huggingface/*
# Or if using custom cache location:
rm -rf $HF_HUB_CACHE/*

# Clean old results
find . -name "results*" -type d -mtime +7 -exec rm -rf {} \;

# Clean Python cache
find . -name "__pycache__" -type d -exec rm -rf {} \; 2>/dev/null
```

**Prevention:**
```bash
# Set cache to larger drive
export HF_HUB_CACHE="/path/to/large/drive/hf_cache"
export HF_HOME="/path/to/large/drive/hf_home"

# Add to ~/.bashrc for persistence
echo 'export HF_HUB_CACHE="/path/to/large/drive/hf_cache"' >> ~/.bashrc
```

### RunPod/Cloud: "/workspace disk full"

**Symptoms:**
```
No space left on device (workspace)
ModelNotFoundError during download
```

**Cause:** Models downloading to wrong location.

**Fix:**
```bash
# Set cache to persistent storage BEFORE any imports
export HF_HOME="/workspace/hf_cache"
export HF_HUB_CACHE="/workspace/hf_cache/hub"
mkdir -p $HF_HOME

# Emergency: move existing cache
mv ~/.cache/huggingface/* /workspace/hf_cache/ 2>/dev/null || true

# Verify environment
env | grep HF_
```

### CUDA out of memory **(gpu)**

**Symptoms:**
```
RuntimeError: CUDA out of memory
torch.cuda.OutOfMemoryError
```

**Cause:** Batch size too large or GPU memory fragmentation.

**Fix:**
```bash
# Reduce batch sizes in config (bench)
# Edit your config file:
train:
  per_device_train_batch_size: 2  # Reduce from 8
  per_device_eval_batch_size: 4   # Reduce from 16

# Clear GPU memory (gpu)
python -c "
import torch
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print('GPU cache cleared')
"

# Kill competing processes (gpu)
nvidia-smi  # Find competing processes
kill -9 <pid_of_competing_process>
```

## Session Management & Resume Issues

### SSH disconnect kills long-running experiments

**Symptoms:**
```
Connection to remote host lost
Experiment terminated unexpectedly
Partial results only
```

**Cause:** Network instability terminating SSH session.

**Fix:**

**Use tmux (recommended):**
```bash
# Start experiment in tmux
tmux new-session -d -s experiment1
tmux attach-session -t experiment1

# Inside tmux, run experiment
gradience bench config.yaml --output-dir results/

# Detach with Ctrl+B, then D
# Experiment continues running

# Reconnect later
ssh user@host
tmux attach-session -t experiment1
```

**Use built-in resume functionality:**
```bash
# Initial run
gradience bench config.yaml --output-dir results/

# If interrupted, resume from where it left off
gradience bench config.yaml --output-dir results/ --resume

# Check what's resumable
ls results/  # Look for partial state files
```

### "--resume not working" or "No resumable state found"

**Symptoms:**
```
RuntimeError: No resumable state found in output directory
ValueError: Cannot resume from corrupted state
```

**Cause:** Missing state files or corrupted resume data.

**Fix:**
```bash
# Check for resume files
ls -la results/
# Look for: .gradience_state, partial_results.json, etc.

# If state corrupted, restart experiment
rm -rf results/.gradience_state
gradience bench config.yaml --output-dir results/
# Restarts from beginning

# Or use different output directory
gradience bench config.yaml --output-dir results_retry/
```

### tmux session lost or "no sessions"

**Symptoms:**
```bash
$ tmux list-sessions
no server running
failed to connect to server
```

**Cause:** tmux server crashed or was killed.

**Fix:**
```bash
# Check for dead sessions
tmux list-sessions

# If no sessions, restart experiment
tmux new-session -d -s recovery
tmux attach-session -t recovery

# Use resume if partial results exist
gradience bench config.yaml --output-dir results/ --resume

# Install tmux if missing (on cloud instances)
apt-get update && apt-get install -y tmux
```

## Configuration & Validation Issues

### "Configuration file not found"

**Symptoms:**
```
FileNotFoundError: Configuration file 'config.yaml' not found
ConfigError: Invalid configuration path
```

**Cause:** Wrong path or missing config file.

**Fix:**
```bash
# Use absolute path
gradience bench /full/path/to/config.yaml

# Or check available configs
find . -name "*.yaml" -type f

# Use bundled configs (guaranteed available)
python -c "
import importlib.resources
configs = importlib.resources.files('gradience.bench.configs')
print('Available configs:', list(configs.iterdir()))
"

# Copy bundled config to modify
python -c "
import importlib.resources
import shutil
configs = importlib.resources.files('gradience.bench.configs')
shutil.copy(configs / 'distilbert_sst2_ci.yaml', './my_config.yaml')
"
```

### "Model not found" or HuggingFace download failures

**Symptoms:**
```
OSError: microsoft/DialoGPT-medium does not appear to be a 🤗 model identifier
requests.exceptions.ConnectionError: Failed to download
```

**Cause:** Network issues or incorrect model identifier.

**Fix:**
```bash
# Test internet connectivity
curl -I https://huggingface.co

# Verify model exists
curl -I https://huggingface.co/api/models/distilbert-base-uncased

# Try alternative model name
gradience bench config.yaml  # Check config for typos

# Use offline mode if model already cached
HF_HUB_OFFLINE=1 gradience bench config.yaml

# Clear corrupted cache
rm -rf ~/.cache/huggingface/hub/models--*your-model*
# Re-download will start fresh
```

## Performance & Memory Issues

### Extremely slow training or evaluation

**Symptoms:**
```
Training taking hours for small datasets
CPU at 100% but GPU idle
Very slow progress bars
```

**Cause:** CPU-only mode when GPU expected, or inefficient batch sizes.

**Fix:**
```bash
# Check GPU availability
python -c "
import torch
print('CUDA available:', torch.cuda.is_available())
print('GPU count:', torch.cuda.device_count())
if torch.cuda.is_available():
    print('GPU name:', torch.cuda.get_device_name(0))
"

# Force GPU usage in config
runtime:
  device: "cuda"  # or "auto"

# Increase batch sizes for GPU
train:
  per_device_train_batch_size: 16  # Increase from 2
  per_device_eval_batch_size: 32   # Increase from 4

# Check GPU utilization during run
nvidia-smi -l 1  # Monitor in real-time
```

### "RuntimeError: Expected all tensors to be on the same device"

**Symptoms:**
```
RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!
```

**Cause:** Mixed device placement in multi-GPU or CPU/GPU setup.

**Fix:**
```bash
# Use single GPU
CUDA_VISIBLE_DEVICES=0 gradience bench config.yaml

# Or force CPU mode
runtime:
  device: "cpu"

# Clear any cached model states
rm -rf ~/.cache/huggingface/transformers/*
```

## Getting Additional Help

### Enable debug logging

```bash
# Run with verbose logging
GRADIENCE_LOG_LEVEL=DEBUG gradience bench config.yaml

# Or enable in Python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Generate diagnostic report

```bash
# System info
python -c "
import sys, torch, transformers
print('Python:', sys.version)
print('PyTorch:', torch.__version__)
print('Transformers:', transformers.__version__)
print('CUDA available:', torch.cuda.is_available())
"

# Package versions
pip show gradience datasets transformers torch
```

### Common support resources

- **GitHub Issues**: Report bugs at https://github.com/gradience-ai/gradience/issues
- **Documentation**: Check docs/ directory for detailed guides
- **CLI Help**: Use `gradience --help` and `gradience <command> --help`
- **Examples**: See gradience/bench/configs/ for working configurations

---

**Pro Tip**: Most issues stem from missing dependencies (`[bench]` extras) or environment setup. Start with `pip install "gradience[bench]"` and proper cache configuration.