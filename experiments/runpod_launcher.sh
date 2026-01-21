#!/bin/bash
# RunPod Launcher for Mechanism Testing Experiment
# 
# This script automates the complete setup and execution of the mechanism
# testing experiment on RunPod GPU instances.

set -euo pipefail

echo "🚀 RunPod Mechanism Testing Launcher"
echo "===================================="

# Configuration
WORKSPACE="/workspace"
HF_CACHE_DIR="$WORKSPACE/hf_cache"
RESULTS_DIR="$WORKSPACE/mechanism_test_results"
REPO_DIR="$WORKSPACE/gradience"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

log_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

log_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

log_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Step 1: System Setup
setup_system() {
    log_info "Setting up system environment..."
    
    # Update system
    apt update -qq
    apt install -y git curl wget vim htop tmux tree > /dev/null 2>&1
    
    # Verify CUDA
    if nvidia-smi > /dev/null 2>&1; then
        log_success "CUDA available"
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    else
        log_error "CUDA not available - this experiment requires GPU"
        exit 1
    fi
    
    log_success "System setup complete"
}

# Step 2: Configure Storage
setup_storage() {
    log_info "Configuring storage and cache directories..."
    
    # Create cache directories
    mkdir -p "$HF_CACHE_DIR"/{hub,datasets}
    mkdir -p "$RESULTS_DIR"
    
    # Set environment variables
    cat >> ~/.bashrc << EOF
export HF_HOME='$HF_CACHE_DIR'
export HF_HUB_CACHE='$HF_CACHE_DIR/hub'
export HF_DATASETS_CACHE='$HF_CACHE_DIR/datasets'
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
export TORCH_USE_CUDA_DSA=1
EOF
    
    # Apply to current session
    export HF_HOME="$HF_CACHE_DIR"
    export HF_HUB_CACHE="$HF_CACHE_DIR/hub"
    export HF_DATASETS_CACHE="$HF_CACHE_DIR/datasets"
    export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
    export TORCH_USE_CUDA_DSA=1
    
    # Check available space
    local available_space=$(df -BG "$WORKSPACE" | tail -1 | awk '{print $4}' | sed 's/G//')
    if [ "$available_space" -lt 80 ]; then
        log_warning "Only ${available_space}GB available - experiment needs ~100GB"
    fi
    
    log_success "Storage configured - Cache: $HF_CACHE_DIR"
}

# Step 3: Setup Repository
setup_repository() {
    log_info "Setting up gradience repository..."
    
    # Check if we're already in gradience repo
    if [ "$(basename $PWD)" = "gradience" ] && [ -f "pyproject.toml" ]; then
        log_info "Already in gradience repository"
        REPO_DIR="$PWD"
    else
        # Clone or use existing
        if [ ! -d "$REPO_DIR" ]; then
            log_info "Cloning gradience repository..."
            read -p "Enter repository URL (or press Enter for default): " repo_url
            if [ -z "$repo_url" ]; then
                repo_url="https://github.com/your-username/gradience.git"
                log_warning "Using placeholder URL - update with your actual repo"
            fi
            git clone "$repo_url" "$REPO_DIR"
        fi
        cd "$REPO_DIR"
    fi
    
    # Setup Python environment
    log_info "Setting up Python environment..."
    python3 -m venv venv_gradience
    source venv_gradience/bin/activate
    
    # Install with dependencies
    pip install -U pip setuptools wheel
    pip install -e .[hf,bench,dev]
    
    # Verify installation
    python3 -c "import gradience; print('Gradience version:', gradience.__version__)" || {
        log_error "Failed to install gradience"
        exit 1
    }
    
    log_success "Repository setup complete"
}

# Step 4: HuggingFace Authentication
setup_huggingface() {
    log_info "Setting up HuggingFace authentication..."
    
    # Check if already logged in
    if huggingface-cli whoami > /dev/null 2>&1; then
        log_success "Already logged in to HuggingFace"
        return
    fi
    
    echo "Please login to HuggingFace to access Mistral-7B model:"
    echo "1. Get your token from: https://huggingface.co/settings/tokens"
    echo "2. Ensure you have access to: mistralai/Mistral-7B-v0.1"
    echo ""
    
    # Interactive login
    huggingface-cli login || {
        log_error "HuggingFace login failed"
        exit 1
    }
    
    # Verify model access
    python3 -c "
from transformers import AutoTokenizer
try:
    AutoTokenizer.from_pretrained('mistralai/Mistral-7B-v0.1')
    print('✅ Mistral access verified')
except Exception as e:
    print(f'❌ Mistral access failed: {e}')
    exit(1)
" || {
        log_error "Cannot access Mistral-7B model"
        exit 1
    }
    
    log_success "HuggingFace authentication complete"
}

# Step 5: Pre-download Models
preload_models() {
    log_info "Pre-downloading models and datasets..."
    
    echo "Downloading Mistral-7B (this may take 15-30 minutes)..."
    python3 -c "
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import sys

print('📥 Downloading Mistral-7B tokenizer...')
tokenizer = AutoTokenizer.from_pretrained('mistralai/Mistral-7B-v0.1')

print('📥 Downloading Mistral-7B model...')
model = AutoModelForCausalLM.from_pretrained(
    'mistralai/Mistral-7B-v0.1',
    torch_dtype=torch.bfloat16,
    device_map='auto'
)
print('✅ Mistral-7B download complete')

# Free memory
del model
torch.cuda.empty_cache()
" || {
        log_error "Failed to download Mistral-7B"
        exit 1
    }
    
    echo "Downloading GSM8K dataset..."
    python3 -c "
from datasets import load_dataset
print('📥 Downloading GSM8K...')
dataset = load_dataset('gsm8k', 'main')
print('✅ GSM8K download complete')
" || {
        log_error "Failed to download GSM8K"
        exit 1
    }
    
    log_success "Model pre-loading complete"
}

# Step 6: Configure Experiment
setup_experiment() {
    log_info "Configuring experiment for RunPod environment..."
    
    # Create RunPod-specific config
    if [ ! -f "experiments/runpod_config.yaml" ]; then
        cp experiments/mechanism_test_config.yaml experiments/runpod_config.yaml
        
        # Update paths for RunPod
        sed -i "s|experiments/mechanism_test_results|$RESULTS_DIR|g" experiments/runpod_config.yaml
        
        # Adjust memory settings based on GPU
        local gpu_memory=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
        local max_memory=$((gpu_memory * 85 / 100))  # Use 85% of GPU memory
        
        sed -i "s|max_memory_per_gpu: \"14GB\"|max_memory_per_gpu: \"${max_memory}MB\"|g" experiments/runpod_config.yaml
        
        log_success "Created RunPod configuration: experiments/runpod_config.yaml"
    fi
    
    # Update launcher script paths
    sed -i "s|experiments/mechanism_test_results|$RESULTS_DIR|g" experiments/run_mechanism_test.sh
    sed -i "s|experiments/mechanism_test_config.yaml|experiments/runpod_config.yaml|g" experiments/run_mechanism_test.sh
    
    # Make scripts executable
    chmod +x experiments/run_mechanism_test.sh
    chmod +x experiments/analyze_mechanism_test.py
    
    log_success "Experiment configuration complete"
}

# Step 7: Quick Validation
validate_setup() {
    log_info "Validating setup with quick test..."
    
    # Test basic functionality
    python3 -c "
from gradience.bench.protocol import generate_compression_configs, _create_shuffled_rank_pattern

# Test shuffle function
original = {'q_proj': 8, 'v_proj': 4, 'k_proj': 16}
shuffled = _create_shuffled_rank_pattern(original, 42)

print('Original:', original)
print('Shuffled:', shuffled)
print('Same ranks:', sorted(original.values()) == sorted(shuffled.values()))
print('Different assignment:', original != shuffled)
print('✅ Shuffle mechanism working')
" || {
        log_error "Setup validation failed"
        exit 1
    }
    
    log_success "Setup validation passed"
}

# Step 8: Launch Options
show_launch_options() {
    log_success "🎉 Setup Complete! Choose your next action:"
    echo ""
    echo "1. 🧪 Quick Test (single seed, fast validation)"
    echo "2. 🚀 Full Experiment (3 seeds, 6-8 hours)" 
    echo "3. 📊 Analysis Only (if experiment already complete)"
    echo "4. 🖥️  Interactive Shell (manual control)"
    echo ""
    
    read -p "Enter choice (1-4): " choice
    
    case $choice in
        1)
            launch_quick_test
            ;;
        2) 
            launch_full_experiment
            ;;
        3)
            launch_analysis
            ;;
        4)
            launch_interactive
            ;;
        *)
            log_error "Invalid choice"
            exit 1
            ;;
    esac
}

# Launch functions
launch_quick_test() {
    log_info "🧪 Starting quick test run..."
    
    # Single seed test
    python3 -m gradience.cli bench \
        --config experiments/runpod_config.yaml \
        --output "$RESULTS_DIR/quick_test" \
        --verbose || {
        log_error "Quick test failed"
        exit 1
    }
    
    log_success "✅ Quick test completed successfully!"
    log_info "Results in: $RESULTS_DIR/quick_test"
    
    echo ""
    echo "Next steps:"
    echo "1. Review results in $RESULTS_DIR/quick_test"
    echo "2. Run full experiment: ./experiments/run_mechanism_test.sh"
}

launch_full_experiment() {
    log_info "🚀 Launching full mechanism testing experiment..."
    log_warning "This will take 6-8 hours - consider using tmux!"
    
    echo ""
    echo "Launch options:"
    echo "1. 🖥️  Foreground (blocks terminal)"
    echo "2. 📱 Tmux session (detachable)"
    echo "3. 🌙 Background with nohup"
    echo ""
    
    read -p "Enter choice (1-3): " launch_choice
    
    case $launch_choice in
        1)
            ./experiments/run_mechanism_test.sh
            ;;
        2)
            tmux new-session -d -s mechanism_test './experiments/run_mechanism_test.sh'
            log_success "Experiment launched in tmux session 'mechanism_test'"
            echo "Attach with: tmux attach-session -t mechanism_test"
            echo "Monitor with: tail -f $RESULTS_DIR/seed_*/run_log.txt"
            ;;
        3)
            nohup ./experiments/run_mechanism_test.sh > "$WORKSPACE/experiment.log" 2>&1 &
            local pid=$!
            log_success "Experiment launched in background (PID: $pid)"
            echo "Monitor with: tail -f $WORKSPACE/experiment.log"
            echo "Check progress: tail -f $RESULTS_DIR/seed_*/run_log.txt"
            ;;
        *)
            log_error "Invalid choice"
            exit 1
            ;;
    esac
}

launch_analysis() {
    log_info "📊 Running analysis..."
    
    if [ -f "$RESULTS_DIR/aggregated_results.json" ]; then
        python3 experiments/analyze_mechanism_test.py "$RESULTS_DIR/aggregated_results.json"
    else
        log_error "No aggregated results found at $RESULTS_DIR/aggregated_results.json"
        echo "Run the experiment first or check the results directory."
    fi
}

launch_interactive() {
    log_success "🖥️  Launching interactive shell"
    echo "You're now in the gradience repository with everything set up."
    echo "Available commands:"
    echo "  - Quick test: python3 -m gradience.cli bench --config experiments/runpod_config.yaml --output /tmp/test"
    echo "  - Full experiment: ./experiments/run_mechanism_test.sh"
    echo "  - Analysis: python3 experiments/analyze_mechanism_test.py <results.json>"
    echo ""
    bash
}

# Main execution
main() {
    log_info "Starting RunPod setup for mechanism testing experiment..."
    
    setup_system
    setup_storage
    setup_repository
    setup_huggingface
    preload_models
    setup_experiment
    validate_setup
    show_launch_options
}

# Run main function
main "$@"