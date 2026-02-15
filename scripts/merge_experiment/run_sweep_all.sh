#!/bin/bash
# ===========================================================================
# run_sweep_all.sh — Parameter sweep for merge experiments
#
# Runs a grid of 21 merge configurations varying:
#   - Merge coefficients (5 configs)
#   - Output rank (8 configs)
#   - TIES trim fraction (5 configs)
#   - Predibase vs. self-trained cross-comparison (3 configs)
#
# Prerequisites:
#   - v0.9.9 experiment has completed (trained adapters exist)
#   - pip install -e ".[bench]"
#   - GPU available (A40 recommended)
#
# Usage:
#   cd /workspace/gradience
#   bash scripts/merge_experiment/run_sweep_all.sh
#
# Environment overrides:
#   PYTHON_BIN         Python binary (default: python)
#   SWEEP_WORKSPACE    Working directory (default: /workspace/merge_sweep)
#   EXPERIMENT_WORKSPACE  Where v0.9.9 results live (default: /workspace/merge_experiment)
#   BASE_MODEL         Base model (default: mistralai/Mistral-7B-v0.1)
#   QUICK_SAMPLES      Quick-screen samples (default: 100)
#   FULL_SAMPLES       Full eval samples (default: 500)
#   TOP_K              Top configs for full eval (default: 5)
#   TIME_BUDGET        Time budget in minutes for quick-screen (default: 0 = unlimited)
# ===========================================================================

set -e
set -o pipefail

SCRIPT_DIR="scripts/merge_experiment"
PY="${PYTHON_BIN:-python}"
SWEEP_WORKSPACE="${SWEEP_WORKSPACE:-/workspace/merge_sweep}"
EXPERIMENT_WORKSPACE="${EXPERIMENT_WORKSPACE:-/workspace/merge_experiment}"
BASE_MODEL="${BASE_MODEL:-mistralai/Mistral-7B-v0.1}"
QUICK_SAMPLES="${QUICK_SAMPLES:-100}"
FULL_SAMPLES="${FULL_SAMPLES:-500}"
TOP_K="${TOP_K:-5}"
TIME_BUDGET="${TIME_BUDGET:-0}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

mkdir -p "$SWEEP_WORKSPACE"

echo "============================================================"
echo "  Merge Parameter Sweep"
echo "  $(date)"
echo "  Sweep workspace: $SWEEP_WORKSPACE"
echo "  Experiment workspace: $EXPERIMENT_WORKSPACE"
echo "  Python: $PY"
echo "============================================================"

# ---------------------------------------------------------------------------
# Phase 0: Verify environment
# ---------------------------------------------------------------------------
echo ""
echo "--- Phase 0: Environment Check ---"

$PY -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')" || {
    echo "ERROR: PyTorch not available"
    exit 1
}

$PY -c "import gradience; print(f'Gradience {gradience.__version__}')" || {
    echo "ERROR: Gradience not installed"
    exit 1
}

echo ""
echo "Environment OK"

# ---------------------------------------------------------------------------
# Phase 0.5: Verify adapter directories
# ---------------------------------------------------------------------------
echo ""
echo "--- Phase 0.5: Verifying Adapters ---"

# Trained adapters (from v0.9.9 re-run)
TRAINED_A_DIR="$EXPERIMENT_WORKSPACE/trained_adapters/mnli"
TRAINED_B_DIR="$EXPERIMENT_WORKSPACE/trained_adapters/qnli"
TRAINED_OK=false

if [ -d "$TRAINED_A_DIR" ] && [ -d "$TRAINED_B_DIR" ]; then
    if [ -f "$TRAINED_A_DIR/adapter_config.json" ] && [ -f "$TRAINED_B_DIR/adapter_config.json" ]; then
        echo "  Trained adapters: OK"
        echo "    MNLI: $TRAINED_A_DIR"
        echo "    QNLI: $TRAINED_B_DIR"
        TRAINED_OK=true
    else
        echo "  WARNING: Trained adapter directories exist but missing adapter_config.json"
    fi
else
    echo "  WARNING: Trained adapters not found"
    echo "    Expected: $TRAINED_A_DIR"
    echo "    Expected: $TRAINED_B_DIR"
    echo "    Run v0.9.9 experiment first (bash scripts/merge_experiment/run_all.sh)"
fi

# Predibase adapters (from download phase)
PREDIBASE_A_DIR="$EXPERIMENT_WORKSPACE/adapters/adapter_a"
PREDIBASE_B_DIR="$EXPERIMENT_WORKSPACE/adapters/adapter_b"
PREDIBASE_OK=false

if [ -d "$PREDIBASE_A_DIR" ] && [ -d "$PREDIBASE_B_DIR" ]; then
    if [ -f "$PREDIBASE_A_DIR/adapter_config.json" ] && [ -f "$PREDIBASE_B_DIR/adapter_config.json" ]; then
        echo "  Predibase adapters: OK"
        echo "    MNLI: $PREDIBASE_A_DIR"
        echo "    QNLI: $PREDIBASE_B_DIR"
        PREDIBASE_OK=true
    else
        echo "  WARNING: Predibase adapter directories exist but missing adapter_config.json"
    fi
else
    echo "  WARNING: Predibase adapters not found (cross-comparison will be skipped)"
fi

if [ "$TRAINED_OK" = false ] && [ "$PREDIBASE_OK" = false ]; then
    echo ""
    echo "ERROR: No valid adapters found. Run v0.9.9 experiment first."
    exit 1
fi

# ---------------------------------------------------------------------------
# Phase 1: Run sweep
# ---------------------------------------------------------------------------
echo ""
echo "--- Phase 1: Running Parameter Sweep ---"

SWEEP_ARGS=""
if [ "$TRAINED_OK" = true ]; then
    SWEEP_ARGS="$SWEEP_ARGS --trained-a-dir $TRAINED_A_DIR --trained-b-dir $TRAINED_B_DIR"
fi
if [ "$PREDIBASE_OK" = true ]; then
    SWEEP_ARGS="$SWEEP_ARGS --predibase-a-dir $PREDIBASE_A_DIR --predibase-b-dir $PREDIBASE_B_DIR"
fi

$PY $SCRIPT_DIR/run_sweep.py \
    --workspace "$SWEEP_WORKSPACE" \
    --base-model "$BASE_MODEL" \
    $SWEEP_ARGS \
    --quick-samples "$QUICK_SAMPLES" \
    --full-samples "$FULL_SAMPLES" \
    --top-k "$TOP_K" \
    --time-budget "$TIME_BUDGET" \
    2>&1 | tee "$SWEEP_WORKSPACE/sweep_log_${TIMESTAMP}.txt"

# ---------------------------------------------------------------------------
# Phase 2: Package results
# ---------------------------------------------------------------------------
echo ""
echo "--- Phase 2: Packaging Results ---"

ARCHIVE="merge_sweep_results_${TIMESTAMP}.tar.gz"
cd /workspace
tar czf "$ARCHIVE" \
    merge_sweep/results/ \
    merge_sweep/sweep_log_*.txt \
    2>/dev/null || true

echo "  Results archived: /workspace/$ARCHIVE"
echo ""
echo "  Download with:"
echo "    scp root@<runpod-ip>:/workspace/$ARCHIVE ./"
echo ""

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo "============================================================"
echo "  SWEEP COMPLETE"
echo "  $(date)"
echo "============================================================"
echo ""
echo "  Key artifacts:"
echo "    $SWEEP_WORKSPACE/results/sweep_results.json"
echo "    $SWEEP_WORKSPACE/results/sweep_results.md"
echo ""
echo "  Remember to stop the pod when done!"
echo "============================================================"
