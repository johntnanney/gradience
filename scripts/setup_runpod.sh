#!/bin/bash
# RunPod Environment Setup Script for Gradience Validation
#
# Usage: 
#   1. Upload gradience codebase to /workspace/gradience
#   2. Run: chmod +x /workspace/gradience/scripts/setup_runpod.sh
#   3. Run: /workspace/gradience/scripts/setup_runpod.sh
#   4. Run experiments with: python -m gradience.bench.run_bench --config <config> --output <output>

set -e

echo "🚀 Setting up RunPod environment for Gradience validation..."

# Navigate to workspace
cd /workspace

# Check if gradience is present
if [ ! -d "gradience" ]; then
    echo "❌ Gradience codebase not found in /workspace/gradience"
    echo "   Please upload your gradience codebase first"
    exit 1
fi

cd gradience

# Install dependencies if needed
echo "📦 Installing Python dependencies..."
pip install -e ".[bench]" --quiet

# Verify installation
echo "🔍 Verifying Gradience installation..."
if python -c "import gradience.bench; print('✅ Gradience import successful')"; then
    echo "✅ Gradience package ready"
else
    echo "❌ Gradience package installation failed"
    exit 1
fi

# Check GPU availability
echo "🔍 Checking GPU availability..."
python -c "
import torch
if torch.cuda.is_available():
    print(f'✅ CUDA available: {torch.cuda.get_device_name(0)}')
    print(f'   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB')
else:
    print('❌ CUDA not available')
    exit(1)
"

# Check configuration files
echo "🔍 Checking experiment configurations..."
configs=(
    "runpod_distilbert_sst2_dev.yaml"
    "runpod_distilbert_sst2.yaml"
    "runpod_mistral_gsm8k_dev.yaml"
    "runpod_mistral_gsm8k_public.yaml"
)

for config in "${configs[@]}"; do
    if [ -f "$config" ]; then
        echo "   ✅ $config"
    else
        echo "   ❌ $config missing"
        echo "      Please ensure configuration files are uploaded"
        exit 1
    fi
done

# Create output directory
echo "📁 Creating output directories..."
mkdir -p /workspace/experiments
echo "   ✅ /workspace/experiments ready"

# Display ready message
echo ""
echo "🎉 RunPod environment ready for Gradience validation!"
echo ""
echo "🚀 Quick Start Commands:"
echo ""
echo "# DistilBERT Development (4 min, \$0.06):"
echo "python -m gradience.bench.run_bench \\"
echo "  --config runpod_distilbert_sst2_dev.yaml \\"
echo "  --output /workspace/experiments/distilbert_dev"
echo ""
echo "# DistilBERT Full Validation (15 min, \$0.23):"
echo "python -m gradience.bench.run_bench \\"
echo "  --config runpod_distilbert_sst2.yaml \\"
echo "  --output /workspace/experiments/distilbert_full"
echo ""
echo "# Mistral Development (30 min, \$0.45):"
echo "python -m gradience.bench.run_bench \\"
echo "  --config runpod_mistral_gsm8k_dev.yaml \\"
echo "  --output /workspace/experiments/mistral_dev"
echo ""
echo "# Mistral Public Demo (150 min, \$2.25):"
echo "python -m gradience.bench.run_bench \\"
echo "  --config runpod_mistral_gsm8k_public.yaml \\"
echo "  --output /workspace/experiments/mistral_public"
echo ""
echo "🔍 Monitor progress with:"
echo "tail -f /workspace/experiments/<output>/seed_*/progress.txt"
echo "tail -f /workspace/experiments/<output>/seed_*/heartbeat.log"
echo ""
echo "📊 Analyze results:"
echo "python scripts/analyze_public_artifacts.py /workspace/experiments/<output>"
echo ""
echo "📄 Raw results:"
echo "cat /workspace/experiments/<output>/bench_aggregate.json"
echo "cat /workspace/experiments/<output>/bench_aggregate.md"