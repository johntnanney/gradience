#!/bin/bash
#
# Quick validation script for rank suggestions
# 
# Usage:
#   ./scripts/validate_suggestions.sh [--probe-r 16] [--model tiny-distilbert] [--quick]
#

set -e

# Default parameters
PROBE_R=16
MODEL="tiny-distilbert"
DATASET="tiny"
BASE_DIR="./validation_runs"
QUICK_MODE=false
VERBOSE=true

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --probe-r)
            PROBE_R="$2"
            shift 2
            ;;
        --model)
            MODEL="$2" 
            shift 2
            ;;
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --base-dir)
            BASE_DIR="$2"
            shift 2
            ;;
        --quick)
            QUICK_MODE=true
            shift
            ;;
        --quiet)
            VERBOSE=false
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --probe-r R       Probe rank (8, 16, 32) [default: 16]"
            echo "  --model M         Model type (tiny-bert, tiny-distilbert, tiny-gpt2) [default: tiny-distilbert]" 
            echo "  --dataset D       Dataset (tiny, cola, sst2) [default: tiny]"
            echo "  --base-dir DIR    Output directory [default: ./validation_runs]"
            echo "  --quick           Skip detailed evaluation"
            echo "  --quiet           Reduce logging output"
            echo "  -h, --help        Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

# Validate parameters
if [[ ! "$PROBE_R" =~ ^(8|16|32)$ ]]; then
    echo "Error: --probe-r must be 8, 16, or 32"
    exit 1
fi

if [[ ! "$MODEL" =~ ^(tiny-bert|tiny-distilbert|tiny-gpt2)$ ]]; then
    echo "Error: --model must be tiny-bert, tiny-distilbert, or tiny-gpt2"
    exit 1
fi

# Print configuration
if [ "$VERBOSE" = true ]; then
    echo "🚀 Starting validation protocol"
    echo "   Model: $MODEL"
    echo "   Dataset: $DATASET" 
    echo "   Probe rank: $PROBE_R"
    echo "   Output: $BASE_DIR"
    echo "   Quick mode: $QUICK_MODE"
    echo
fi

# Check dependencies
if ! command -v python &> /dev/null; then
    echo "Error: Python not found"
    exit 1
fi

# Check if gradience is available
if ! python -c "import gradience" &> /dev/null; then
    echo "Error: Gradience module not available"
    echo "Make sure you're in the correct environment and gradience is installed"
    exit 1
fi

# Create base directory
mkdir -p "$BASE_DIR"

# Build command
CMD="python scripts/validation_protocol.py"
CMD="$CMD --model $MODEL"
CMD="$CMD --dataset $DATASET"
CMD="$CMD --probe-r $PROBE_R" 
CMD="$CMD --base-dir $BASE_DIR"

if [ "$VERBOSE" = true ]; then
    CMD="$CMD --verbose"
fi

# Run validation protocol
if [ "$VERBOSE" = true ]; then
    echo "🔬 Running validation protocol..."
    echo "Command: $CMD"
    echo
fi

if eval "$CMD"; then
    # Success - show results
    RESULTS_DIR="$BASE_DIR/validation_${MODEL}_${DATASET}_r${PROBE_R}"
    
    if [ "$VERBOSE" = true ]; then
        echo
        echo "✅ Validation completed successfully!"
        echo "📁 Results: $RESULTS_DIR"
        echo
    fi
    
    # Show summary if evaluation exists
    if [ -f "$RESULTS_DIR/evaluation.json" ] && [ "$QUICK_MODE" = false ]; then
        echo "📊 Summary:"
        if command -v jq &> /dev/null; then
            # Use jq for pretty formatting if available
            jq -r '.recommendations[]' "$RESULTS_DIR/evaluation.json" | sed 's/^/   • /'
            echo
            echo "📈 Parameter Reductions:"
            jq -r '.parameter_reductions | to_entries[] | "   • \(.key): \(.value * 100 | round)%"' "$RESULTS_DIR/evaluation.json" 2>/dev/null || echo "   (detailed metrics in evaluation.json)"
        else
            # Fallback without jq
            echo "   (see $RESULTS_DIR/evaluation.json for detailed results)"
        fi
        echo
    fi
    
    # Show key files
    if [ "$VERBOSE" = true ]; then
        echo "📄 Key files:"
        echo "   • $RESULTS_DIR/evaluation.json (summary)"
        echo "   • $RESULTS_DIR/probe_audit.json (audit data)" 
        echo "   • $RESULTS_DIR/probe_r${PROBE_R}/ (original model)"
        echo "   • $RESULTS_DIR/retrain_*/ (compressed models)"
        echo
    fi
    
    # Quick validation check
    if [ -d "$RESULTS_DIR/retrain_uniform_p90" ] && [ -d "$RESULTS_DIR/retrain_per_layer" ]; then
        echo "🎯 Validation strategies completed:"
        echo "   ✓ Uniform P90 (conservative)"
        echo "   ✓ Per-layer (experimental)" 
        if [ -d "$RESULTS_DIR/retrain_module_p90" ]; then
            echo "   ✓ Module P90 (moderate)"
        fi
        echo
    fi
    
else
    echo "❌ Validation failed"
    exit 1
fi