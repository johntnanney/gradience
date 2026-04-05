# RunPod Cloud GPU Guide

**Optional guide for RunPod users. Gradience works on any machine - this is just cloud-specific convenience.**

RunPod provides affordable GPU access for model compression experiments. This guide covers RunPod-specific setup, management, and artifact extraction.

## Prerequisites

- RunPod account with GPU pod access
- Basic Linux command line familiarity
- SSH client for secure copy operations

## Quick Start

1. **Launch Pod**: Select PyTorch template with CUDA support
2. **Setup Environment**: Use automated script or manual installation
3. **Run Experiments**: Execute bench commands in tmux sessions
4. **Extract Results**: SCP artifacts to local machine
5. **Clean Up**: Manage disk space and stop billing

For a resumable single-block adjudication/fingerprint run (train → preflight → merge/eval),
use the dedicated workflow:
[`docs/workflows/runpod_gpu_resumable_pipeline.md`](workflows/runpod_gpu_resumable_pipeline.md)
with
[`scripts/runpod/run_resumable_gpu_pipeline.py`](../scripts/runpod/run_resumable_gpu_pipeline.py).

## Environment Setup

### Automated Setup

Use the provided environment setup script:

```bash
# Download and run setup script (base)
curl -fsSL https://raw.githubusercontent.com/gradience-ai/gradience/main/scripts/runpod_setup.sh | bash

# Or if already cloned: (base)
./scripts/runpod_setup.sh
```

The script handles:
- Python environment creation
- Gradience installation with bench extras **(bench)**
- HuggingFace cache configuration
- tmux installation and basic config

### Manual Setup

```bash
# Install system dependencies (base)
apt-get update && apt-get install -y tmux git

# Create Python environment (base)
python -m venv gradience-env
source gradience-env/bin/activate

# Install Gradience (bench)
pip install "gradience[bench]"

# Configure HuggingFace cache (base)
export HF_HOME="/workspace/hf_cache"
export HF_HUB_CACHE="/workspace/hf_cache/hub"
mkdir -p $HF_HOME

# Add to ~/.bashrc for persistence (base)
echo 'export HF_HOME="/workspace/hf_cache"' >> ~/.bashrc
echo 'export HF_HUB_CACHE="/workspace/hf_cache/hub"' >> ~/.bashrc
```

## tmux Session Management

**Important**: tmux is a system package (`apt-get`), not a Python package (`pip`).

### Install tmux

```bash
# On RunPod (Ubuntu-based)
apt-get update && apt-get install -y tmux

# Verify installation
tmux --version
```

### Basic tmux Workflow

```bash
# Start new session for experiment (base)
tmux new-session -d -s experiment1

# Attach to session (base)
tmux attach-session -t experiment1

# Inside tmux: run long experiments (bench)
source gradience-env/bin/activate
gradience bench configs/distilbert_sst2_gpu.yaml --output-dir /workspace/results

# Detach (Ctrl+B, then D) (base)
# Session continues running in background

# List sessions (base)
tmux list-sessions

# Kill session when done (base)
tmux kill-session -t experiment1
```

### tmux Best Practices

- **One session per experiment**: Avoid conflicts and easy monitoring
- **Meaningful session names**: `experiment1`, `bert-large`, `gpt2-eval`
- **Regular detaching**: Preserve work if connection drops
- **Clean session management**: Kill completed sessions to save memory

## Disk Space Management

RunPod pods have limited disk space. Regular cleanup prevents storage issues.

### HuggingFace Cache Cleanup

```bash
# Check cache size
du -sh $HF_HOME

# Clean old/unused models (be careful!)
rm -rf $HF_HOME/hub/models--*unused-model*

# Or clear entire cache (re-downloads everything)
rm -rf $HF_HOME && mkdir -p $HF_HOME
```

### Repository Cleanup

```bash
# Clean git repositories
find /workspace -name ".git" -type d -exec du -sh {} \;
rm -rf /workspace/old-repo-clone

# Clean Python cache
find /workspace -name "__pycache__" -type d -exec rm -rf {} \; 2>/dev/null
find /workspace -name "*.pyc" -delete
```

### Experiment Results Management

```bash
# Check results disk usage
du -sh /workspace/results

# Archive old results before cleanup
tar -czf old-results-$(date +%Y%m%d).tar.gz /workspace/results/old-experiment
rm -rf /workspace/results/old-experiment

# Extract important artifacts only
mkdir -p /workspace/artifacts
cp /workspace/results/*/bench.json /workspace/artifacts/
cp /workspace/results/*/audit.json /workspace/artifacts/
```

## Artifact Extraction

### SCP to Local Machine

```bash
# From your local machine: (base)

# Copy specific artifacts (base)
scp root@runpod-ip:/workspace/results/bench.json ./local-results/

# Copy entire results directory (base)
scp -r root@runpod-ip:/workspace/results ./local-results/

# Copy compressed archive (base)
scp root@runpod-ip:/workspace/artifacts.tar.gz ./
tar -xzf artifacts.tar.gz
```

### Batch Artifact Collection

Create extraction script on RunPod:

```bash
# On RunPod: create extraction script
cat > /workspace/collect_artifacts.sh << 'EOF'
#!/bin/bash
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
ARCHIVE_NAME="gradience_artifacts_${TIMESTAMP}.tar.gz"

# Create temporary directory for artifacts
mkdir -p /tmp/artifacts

# Collect key artifacts
find /workspace/results -name "bench.json" -exec cp {} /tmp/artifacts/bench_{}.json \;
find /workspace/results -name "audit.json" -exec cp {} /tmp/artifacts/audit_{}.json \;
find /workspace/results -name "compression_configs.json" -exec cp {} /tmp/artifacts/compression_{}.json \;

# Create archive
tar -czf "/workspace/${ARCHIVE_NAME}" -C /tmp artifacts

# Cleanup
rm -rf /tmp/artifacts

echo "Artifacts archived to: /workspace/${ARCHIVE_NAME}"
echo "Download with: scp root@\$(curl -s https://ipv4.icanhazip.com):/workspace/${ARCHIVE_NAME} ./"
EOF

chmod +x /workspace/collect_artifacts.sh
```

### Using the Collection Script

```bash
# On RunPod
./collect_artifacts.sh

# Output shows download command:
# Download with: scp root@192.168.1.100:/workspace/gradience_artifacts_20240206_143022.tar.gz ./

# Run the scp command from your local machine
```

## Cost Optimization

### Monitoring Usage

```bash
# Check GPU utilization (gpu)
nvidia-smi

# Monitor disk usage (base)
df -h /workspace

# Check running processes (base)
ps aux | grep python
```

### Stopping vs Pausing

- **Pause pod**: Preserves disk state, reduced billing rate
- **Stop pod**: Destroys disk state, no billing
- **Always stop** when experiment complete unless continuing work

### Efficient Workflows

1. **Prepare locally**: Test configs on CPU before GPU runs
2. **Batch experiments**: Queue multiple runs in same session
3. **Extract early**: Copy artifacts immediately after completion
4. **Clean aggressively**: Remove data as soon as extracted

## Troubleshooting

### Common Issues

**tmux not found**:
```bash
apt-get update && apt-get install -y tmux
```

**Out of disk space**:
```bash
# Emergency cleanup
rm -rf $HF_HOME/hub/models--*
docker system prune -f
```

**Connection lost**:
```bash
# Reconnect and reattach
ssh root@runpod-ip
tmux list-sessions
tmux attach-session -t experiment1
```

**CUDA out of memory**:
```bash
# Reduce batch size in config
# Check GPU memory: nvidia-smi
# Kill competing processes: pkill -f python
```

### Getting Help

- Check RunPod documentation for platform issues
- Use `gradience --help` for tool-specific questions  
- Gradience logs available in experiment output directories

---

**Remember**: This is optional cloud-specific guidance. Gradience works on any machine with Python 3.9+ and optional GPU support.
