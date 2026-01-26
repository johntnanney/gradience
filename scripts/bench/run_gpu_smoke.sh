#!/usr/bin/env bash
set -euo pipefail

# GPU Smoke Test - Fast validation of full GPU pipeline
# Tests: model loading, training, audit, compression, evaluation in ~minutes
# 
# Usage: scripts/bench/run_gpu_smoke.sh [--output OUTPUT_DIR]

# Parse arguments
OUTPUT_DIR=""
while [[ $# -gt 0 ]]; do
  case $1 in
    --output)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    -h|--help)
      echo "Usage: $0 [--output OUTPUT_DIR]"
      echo ""
      echo "GPU smoke test for gradience bench pipeline validation"
      echo ""
      echo "Options:"
      echo "  --output DIR     Output directory (default: runs/gpu_smoke_TIMESTAMP)"
      echo "  -h, --help       Show this help"
      echo ""
      echo "Environment variables:"
      echo "  GRADIENCE_GPU_SMOKE_STEPS    Override max_steps (default: 20)"
      echo "  GRADIENCE_GPU_SMOKE_TRAIN    Override train_samples (default: 32)"
      echo "  GRADIENCE_GPU_SMOKE_EVAL     Override eval_samples (default: 64)"
      exit 0
      ;;
    *)
      echo "Unknown option $1"
      echo "Use -h or --help for usage information"
      exit 1
      ;;
  esac
done

# Always run from repo root
ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$ROOT"

echo "========================================================================"
echo "GRADIENCE GPU SMOKE TEST"
echo "========================================================================"
echo "repo: $ROOT"
echo ""

# Basic preflight checks
echo "🔍 Preflight checks..."

# Check Python dependencies
python3 - <<'PY'
import importlib, sys
required_modules = ["torch", "transformers", "peft", "safetensors", "datasets"]
missing = []
for module in required_modules:
    try:
        importlib.import_module(module)
    except Exception:
        missing.append(module)

if missing:
    print("❌ Missing dependencies:", ", ".join(missing))
    print("Install with: pip install torch transformers peft safetensors datasets")
    sys.exit(1)

print("✅ Python dependencies available")
PY

# Check CUDA availability
CUDA_AVAILABLE="$(python3 - <<'PY'
try:
    import torch
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        current_device = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(current_device)
        print(f"✅ CUDA available: {gpu_count} GPU(s), current: {gpu_name}")
        print("true")
    else:
        print("❌ CUDA not available - GPU smoke test requires CUDA")
        print("false")
except Exception as e:
    print(f"❌ Error checking CUDA: {e}")
    print("false")
PY
)"

if [[ "${CUDA_AVAILABLE##*$'\n'}" != "true" ]]; then
    echo ""
    echo "💡 This test requires a CUDA-capable GPU."
    echo "   For CPU testing, use: scripts/smoke.sh"
    exit 1
fi

# Check gradience bench CLI
if ! python3 -m gradience.bench.run_bench --help >/dev/null 2>&1; then
    echo "❌ gradience.bench.run_bench not available"
    echo "Install gradience with: pip install -e ."
    exit 1
fi
echo "✅ gradience.bench.run_bench available"
echo ""

# Setup output directory
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${OUTPUT_DIR:-runs/gpu_smoke_${TIMESTAMP}}"

echo "📁 Output directory: $OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

# Environment overrides (for debugging/tuning)
MAX_STEPS="${GRADIENCE_GPU_SMOKE_STEPS:-20}"
TRAIN_SAMPLES="${GRADIENCE_GPU_SMOKE_TRAIN:-32}" 
EVAL_SAMPLES="${GRADIENCE_GPU_SMOKE_EVAL:-64}"

echo "📊 Test parameters:"
echo "   max_steps: $MAX_STEPS"
echo "   train_samples: $TRAIN_SAMPLES"
echo "   eval_samples: $EVAL_SAMPLES"
echo ""

# Create temporary config with overrides if needed
CONFIG_PATH="gradience/bench/configs/gpu_smoke/mistral_gsm8k_gpu_smoke.yaml"
TEMP_CONFIG=""

if [[ "$MAX_STEPS" != "20" ]] || [[ "$TRAIN_SAMPLES" != "32" ]] || [[ "$EVAL_SAMPLES" != "64" ]]; then
    echo "🔧 Creating temporary config with overrides..."
    TEMP_CONFIG="$OUTPUT_DIR/config_override.yaml"
    
    # Use Python to safely modify YAML
    python3 - <<PY "$CONFIG_PATH" "$TEMP_CONFIG" "$MAX_STEPS" "$TRAIN_SAMPLES" "$EVAL_SAMPLES"
import yaml, sys
config_path, temp_path, max_steps, train_samples, eval_samples = sys.argv[1:6]

with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# Apply overrides
config['train']['max_steps'] = int(max_steps)
config['train']['train_samples'] = int(train_samples)  
config['train']['eval_samples'] = int(eval_samples)
config['task']['eval_max_samples'] = int(eval_samples)

with open(temp_path, 'w') as f:
    yaml.dump(config, f, default_flow_style=False)
    
print(f"✅ Temporary config created: {temp_path}")
PY
    CONFIG_PATH="$TEMP_CONFIG"
fi

echo ""
echo "🚀 Starting GPU smoke test..."
echo ""

# Run the bench protocol
echo "---- Running bench protocol ----"
START_TIME=$(date +%s)

python3 -m gradience.bench.run_bench \
    --config "$CONFIG_PATH" \
    --output "$OUTPUT_DIR" \
    2>&1 | tee "$OUTPUT_DIR/bench_log.txt"

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo ""
echo "⏱️  Bench completed in ${DURATION}s"

# Validate outputs
echo ""
echo "🔍 Validating outputs..."

REQUIRED_FILES=(
    "bench.json"
    "bench.md"  
    "runs.json"
)

MISSING_FILES=()
for file in "${REQUIRED_FILES[@]}"; do
    if [[ -f "$OUTPUT_DIR/$file" ]]; then
        echo "✅ $file"
    else
        echo "❌ $file (missing)"
        MISSING_FILES+=("$file")
    fi
done

# Check for any compressed variant results
VARIANT_DIRS=($(find "$OUTPUT_DIR" -maxdepth 1 -name "*_r*" -type d 2>/dev/null || true))
if [[ ${#VARIANT_DIRS[@]} -gt 0 ]]; then
    echo "✅ Found ${#VARIANT_DIRS[@]} compressed variant(s)"
    for dir in "${VARIANT_DIRS[@]}"; do
        variant_name="$(basename "$dir")"
        if [[ -f "$dir/eval.json" ]]; then
            echo "   ✅ $variant_name/eval.json"
        else
            echo "   ❌ $variant_name/eval.json (missing)"
            MISSING_FILES+=("$variant_name/eval.json")
        fi
    done
else
    echo "❌ No compressed variants found"
    MISSING_FILES+=("compressed_variants")
fi

# Validate bench.json structure
if [[ -f "$OUTPUT_DIR/bench.json" ]]; then
    python3 - <<'PY' "$OUTPUT_DIR/bench.json"
import json, sys
bench_path = sys.argv[1]

try:
    with open(bench_path, 'r') as f:
        bench = json.load(f)
    
    required_keys = ['config', 'audit', 'compression', 'results']
    missing_keys = [key for key in required_keys if key not in bench]
    
    if missing_keys:
        print(f"❌ bench.json missing keys: {missing_keys}")
        sys.exit(1)
        
    if 'variants' not in bench['results'] or not bench['results']['variants']:
        print("❌ bench.json has no variant results")
        sys.exit(1)
        
    print(f"✅ bench.json structure valid ({len(bench['results']['variants'])} variants)")
    
except Exception as e:
    print(f"❌ bench.json validation failed: {e}")
    sys.exit(1)
PY
    if [[ $? -ne 0 ]]; then
        MISSING_FILES+=("valid_bench.json")
    fi
fi

echo ""

# Final results
if [[ ${#MISSING_FILES[@]} -eq 0 ]]; then
    echo "🎉 GPU SMOKE TEST PASSED"
    echo ""
    echo "✅ Pipeline validation complete:"
    echo "   • Model loading (Mistral-7B-v0.1 on CUDA)"
    echo "   • Training with gradient checkpointing"
    echo "   • LoRA probe training and audit" 
    echo "   • Compression configuration generation"
    echo "   • Compressed variant retraining"
    echo "   • Evaluation and verdict generation"
    echo "   • Report generation (JSON + Markdown)"
    echo ""
    echo "📁 Artifacts in: $OUTPUT_DIR"
    echo "   📄 bench.json      - Complete benchmark results"
    echo "   📄 bench.md        - Human-readable report" 
    echo "   📄 runs.json       - Execution metadata"
    echo "   📂 *_r*/           - Compressed variant directories"
    echo "   📄 bench_log.txt   - Complete execution log"
    echo ""
    echo "🚀 Ready for GPU deployment!"
else
    echo "❌ GPU SMOKE TEST FAILED"
    echo ""
    echo "Missing or invalid outputs:"
    for file in "${MISSING_FILES[@]}"; do
        echo "   - $file"
    done
    echo ""
    echo "Check log: $OUTPUT_DIR/bench_log.txt"
    exit 1
fi