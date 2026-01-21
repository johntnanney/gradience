# RunPod Setup Guide for Mechanism Testing Experiment

This guide walks through setting up and running the mechanism testing experiment on RunPod GPU cloud instances.

## 1. RunPod Instance Setup

### Recommended Configuration
- **GPU**: RTX 4090 (24GB VRAM) or A100 (40GB+)
- **CPU**: 8+ cores 
- **RAM**: 32GB+
- **Storage**: 100GB+ (for models, datasets, results)
- **Template**: PyTorch 2.1+ or Ubuntu with CUDA

### Instance Selection
```bash
# Choose template with:
- CUDA 11.8+ or 12.x
- Python 3.10+
- PyTorch 2.0+
- Git installed
```

## 2. Environment Setup

### Connect and Initial Setup
```bash
# Connect via SSH or web terminal
ssh root@<runpod-ip> -p <port>

# Update system
apt update && apt upgrade -y

# Install required system packages
apt install -y git curl wget vim htop

# Verify CUDA
nvidia-smi
python3 -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

### Clone Repository
```bash
# Clone your gradience repository
git clone https://github.com/your-username/gradience.git
cd gradience

# Or if using a specific branch
git clone -b mechanism-testing https://github.com/your-username/gradience.git
cd gradience
```

### Python Environment Setup
```bash
# Create virtual environment (recommended)
python3 -m venv venv_gradience
source venv_gradience/bin/activate

# Install gradience with all dependencies
pip install -e .[hf,bench,dev]

# Verify installation
python3 -c "import gradience; print('Gradience version:', gradience.__version__)"
```

## 3. HuggingFace Authentication

### Setup HF Token
```bash
# Install HF CLI if not present
pip install huggingface_hub

# Login with your token
huggingface-cli login
# Enter your HF token when prompted

# Verify access to Mistral
python3 -c "from transformers import AutoTokenizer; print('✓ Mistral access OK')"
```

### Alternative: Environment Variable
```bash
# If you prefer environment variable
export HF_TOKEN="your_huggingface_token_here"
echo "export HF_TOKEN='your_token'" >> ~/.bashrc
```

## 4. Storage Configuration

### Set HuggingFace Cache Location
```bash
# Configure cache to use RunPod storage efficiently
export HF_HOME="/workspace/hf_cache"
export HF_HUB_CACHE="/workspace/hf_cache/hub"  
export HF_DATASETS_CACHE="/workspace/hf_cache/datasets"

# Add to bashrc for persistence
echo "export HF_HOME='/workspace/hf_cache'" >> ~/.bashrc
echo "export HF_HUB_CACHE='/workspace/hf_cache/hub'" >> ~/.bashrc
echo "export HF_DATASETS_CACHE='/workspace/hf_cache/datasets'" >> ~/.bashrc

# Create directories
mkdir -p /workspace/hf_cache/{hub,datasets}
```

### Verify Storage Space
```bash
# Check available space
df -h /workspace
du -sh /workspace/hf_cache

# Monitor during download
watch -n 5 'df -h /workspace && du -sh /workspace/hf_cache'
```

## 5. Pre-download Models (Recommended)

```bash
# Pre-download Mistral-7B to avoid timeouts during experiment
python3 -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

print('Downloading Mistral-7B...')
tokenizer = AutoTokenizer.from_pretrained('mistralai/Mistral-7B-v0.1')
model = AutoModelForCausalLM.from_pretrained(
    'mistralai/Mistral-7B-v0.1',
    torch_dtype=torch.bfloat16,
    device_map='auto'
)
print('✓ Download complete')
"

# Pre-download GSM8K dataset
python3 -c "
from datasets import load_dataset
print('Downloading GSM8K...')
dataset = load_dataset('gsm8k', 'main')
print('✓ GSM8K downloaded')
"
```

## 6. Experiment Configuration

### Create RunPod-Specific Config
```bash
# Copy and modify the experiment config
cp experiments/mechanism_test_config.yaml experiments/runpod_config.yaml

# Edit for RunPod environment
vim experiments/runpod_config.yaml
```

### RunPod Configuration Adjustments
```yaml
# experiments/runpod_config.yaml
env:
  output_base_dir: "/workspace/mechanism_test_results"
  torch_dtype: "bfloat16"
  device_map: "auto" 
  max_memory_per_gpu: "20GB"  # Adjust based on GPU (4090=22GB, A100=38GB)
  
train:
  # Smaller batch if memory constrained
  per_device_train_batch_size: 4  # Reduce from 8 if needed
  gradient_accumulation_steps: 2  # Increase to maintain effective batch size
```

### Memory Optimization Settings
```bash
# Set PyTorch memory management
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
export TORCH_USE_CUDA_DSA=1

# Add to bashrc
echo "export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256" >> ~/.bashrc
```

## 7. Running the Experiment

### Quick Test Run
```bash
# Test with single seed first (faster validation)
python3 -m gradience.cli bench \
    --config experiments/runpod_config.yaml \
    --output /workspace/test_run \
    --verbose
```

### Full Experiment
```bash
# Make sure scripts are executable
chmod +x experiments/run_mechanism_test.sh

# Modify script for RunPod paths
sed -i 's|experiments/mechanism_test_results|/workspace/mechanism_test_results|g' experiments/run_mechanism_test.sh
sed -i 's|experiments/mechanism_test_config.yaml|experiments/runpod_config.yaml|g' experiments/run_mechanism_test.sh

# Run full experiment (6-8 hours)
./experiments/run_mechanism_test.sh
```

### Monitor Progress
```bash
# In separate terminal/tmux session
# Monitor GPU usage
watch -n 2 nvidia-smi

# Monitor disk usage  
watch -n 10 'df -h /workspace'

# Monitor experiment logs
tail -f /workspace/mechanism_test_results/seed_*/run_log.txt
```

## 8. Background Execution (Recommended)

### Using tmux
```bash
# Install tmux if not present
apt install tmux

# Start tmux session
tmux new-session -d -s mechanism_test

# Attach to session
tmux attach-session -t mechanism_test

# Run experiment in tmux
./experiments/run_mechanism_test.sh

# Detach: Ctrl+B, then D
# Reattach later: tmux attach-session -t mechanism_test
```

### Using nohup (Alternative)
```bash
# Run in background with logging
nohup ./experiments/run_mechanism_test.sh > /workspace/experiment.log 2>&1 &

# Monitor progress
tail -f /workspace/experiment.log
```

## 9. Results Analysis

### Run Statistical Analysis
```bash
# After experiment completes
python3 experiments/analyze_mechanism_test.py \
    /workspace/mechanism_test_results/aggregated_results.json
```

### Download Results
```bash
# Compress results for download
tar -czf /workspace/mechanism_test_results.tar.gz /workspace/mechanism_test_results/

# Download via RunPod file manager or SCP
# scp root@<runpod-ip>:/workspace/mechanism_test_results.tar.gz ./
```

## 10. Troubleshooting

### Common Issues

**CUDA Out of Memory**
```bash
# Reduce batch size in config
per_device_train_batch_size: 2
gradient_accumulation_steps: 4

# Or use gradient checkpointing
gradient_checkpointing: true
```

**Model Download Fails**
```bash
# Check HF token
huggingface-cli whoami

# Manually download with retries
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('mistralai/Mistral-7B-v0.1', resume_download=True)
"
```

**Disk Space Issues**
```bash
# Clean cache if needed
rm -rf /workspace/hf_cache/hub/models--*/blobs/*.tmp.*

# Move cache to larger volume if available
mv /workspace/hf_cache /mnt/data/hf_cache
ln -s /mnt/data/hf_cache /workspace/hf_cache
```

### Monitoring Commands
```bash
# System resources
htop

# GPU utilization
nvidia-smi -l 2

# Disk usage
du -sh /workspace/* | sort -h

# Process monitoring
ps aux | grep python
```

## 11. Expected Timeline

**Setup Time**: 30-60 minutes
- Instance creation: 5 min
- Environment setup: 20 min
- Model download: 15-30 min

**Experiment Runtime**: 6-8 hours
- Per seed: 2-3 hours
- 3 seeds total
- Plus audit/evaluation overhead

**Total**: ~8-9 hours end-to-end

## 12. Cost Estimation

**RTX 4090 Instance (~$0.50/hour)**
- Setup + Experiment: ~9 hours = $4.50
- Storage: ~100GB = $1-2
- **Total**: ~$6-7

**A100 Instance (~$2.00/hour)**  
- Faster but more expensive: ~$18-20 total

Choose based on budget vs. speed preference.

## 13. Success Verification

After completion, you should have:
```
/workspace/mechanism_test_results/
├── seed_42/
│   ├── bench.json
│   └── bench.md  
├── seed_43/
├── seed_44/
├── aggregated_results.json
└── mechanism_analysis.json
```

The analysis will show whether audit-guided per-layer ranks provide real benefit beyond heterogeneity!