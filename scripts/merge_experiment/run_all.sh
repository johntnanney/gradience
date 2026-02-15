#!/bin/bash
# ===========================================================================
# run_all.sh — Two-path merge experiment: Predibase first, train-your-own fallback
#
# Strategy:
#   1. Download Predibase adapters (predibase/glue_mnli, predibase/glue_qnli)
#   2. Validate templates: run 100 samples, check ≥70% accuracy
#   3. If validation passes → proceed with Predibase templates
#   4. If validation fails → train rank-8 adapters (~45 min), use custom templates
#   5. Full experiment: audit → merge (3 strategies) → evaluate → collect
#
# Usage:
#   cd /workspace/gradience
#   bash scripts/merge_experiment/run_all.sh
#
# Prerequisites:
#   - pip install -e ".[bench]"
#   - GPU available (A40 recommended)
#   - HuggingFace login (huggingface-cli login)
#
# Environment overrides:
#   PYTHON_BIN      Python binary (default: python)
#   MERGE_WORKSPACE Working directory (default: /workspace/merge_experiment)
#   ADAPTER_A_REPO  MNLI adapter HF repo (default: predibase/glue_mnli)
#   ADAPTER_B_REPO  QNLI adapter HF repo (default: predibase/glue_qnli)
#   BASE_MODEL      Base model (default: mistralai/Mistral-7B-v0.1)
#   MAX_EVAL_SAMPLES Eval samples per dataset (default: 500)
#   VALIDATION_SAMPLES Template validation samples (default: 100)
#   VALIDATION_THRESHOLD Min accuracy to pass (default: 0.70)
# ===========================================================================

set -e
set -o pipefail

SCRIPT_DIR="scripts/merge_experiment"
WORKSPACE="${MERGE_WORKSPACE:-/workspace/merge_experiment}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Configurable python binary: RunPod uses "python", local Mac uses "python3"
PY="${PYTHON_BIN:-python}"

# Model and adapter defaults
BASE_MODEL="${BASE_MODEL:-mistralai/Mistral-7B-v0.1}"
ADAPTER_A_REPO="${ADAPTER_A_REPO:-predibase/glue_mnli}"
ADAPTER_B_REPO="${ADAPTER_B_REPO:-predibase/glue_qnli}"

# Experiment parameters
OUTPUT_RANK="${OUTPUT_RANK:-8}"
OUTPUT_ALPHA="${OUTPUT_ALPHA:-16.0}"
MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-500}"
VALIDATION_SAMPLES="${VALIDATION_SAMPLES:-100}"
VALIDATION_THRESHOLD="${VALIDATION_THRESHOLD:-0.70}"

mkdir -p "$WORKSPACE"

echo "============================================================"
echo "  Merge Experiment — Two-Path Strategy"
echo "  $(date)"
echo "  Workspace: $WORKSPACE"
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
    echo "ERROR: Gradience not installed. Run: pip install -e '.[bench]'"
    exit 1
}

$PY -c "from gradience.vnext.merge import merge_audit, execute_merge; print('merge imports OK')" || {
    echo "ERROR: merge modules not available"
    exit 1
}

$PY -c "import transformers, peft, datasets; print('bench deps OK')" || {
    echo "ERROR: Bench dependencies not installed. Run: pip install -e '.[bench]'"
    exit 1
}

echo "Adapter A repo: $ADAPTER_A_REPO"
echo "Adapter B repo: $ADAPTER_B_REPO"
echo "Base model: $BASE_MODEL"
echo "Output rank: $OUTPUT_RANK"
echo "Max eval samples: $MAX_EVAL_SAMPLES"
echo "Validation samples: $VALIDATION_SAMPLES"
echo "Validation threshold: $VALIDATION_THRESHOLD"

# HF cache setup
if [ -n "$HF_HOME" ]; then
    echo "HF_HOME=$HF_HOME"
elif [ -d "/workspace/hf_cache" ]; then
    export HF_HOME="/workspace/hf_cache"
    export HF_HUB_CACHE="/workspace/hf_cache/hub"
    echo "Auto-detected HF cache at /workspace/hf_cache"
else
    echo "HF_HOME not set, using default: ~/.cache/huggingface"
fi

echo ""
echo "Environment OK"

# ---------------------------------------------------------------------------
# Phase 0.5: Download Predibase adapters
# ---------------------------------------------------------------------------
echo ""
echo "--- Phase 0.5: Downloading Predibase Adapters ---"

ADAPTERS_DIR="$WORKSPACE/adapters"
ADAPTER_A_DIR="$ADAPTERS_DIR/adapter_a"
ADAPTER_B_DIR="$ADAPTERS_DIR/adapter_b"

$PY -c "
from huggingface_hub import snapshot_download
from pathlib import Path

adapters_dir = Path('$ADAPTERS_DIR')
adapters_dir.mkdir(parents=True, exist_ok=True)

for name, repo in [('adapter_a', '$ADAPTER_A_REPO'), ('adapter_b', '$ADAPTER_B_REPO')]:
    dest = adapters_dir / name
    if dest.exists() and any(dest.iterdir()):
        print(f'  {name} already cached: {dest}')
        continue
    print(f'  Downloading {repo} -> {dest}')
    snapshot_download(repo_id=repo, local_dir=str(dest))
    print(f'  Done: {dest}')
"

echo "Adapters ready."

# ---------------------------------------------------------------------------
# Phase 1: Template validation (45-minute timebox)
# ---------------------------------------------------------------------------
echo ""
echo "--- Phase 1: Template Validation ---"
echo "  Testing Predibase prompt templates on $VALIDATION_SAMPLES samples..."
echo "  Threshold: $VALIDATION_THRESHOLD"

TEMPLATE_MODE="predibase"
VALIDATION_RESULT="$WORKSPACE/validation_result.json"

if $PY $SCRIPT_DIR/validate_templates.py \
    --base-model "$BASE_MODEL" \
    --adapter-a-dir "$ADAPTER_A_DIR" \
    --adapter-b-dir "$ADAPTER_B_DIR" \
    --threshold "$VALIDATION_THRESHOLD" \
    --n-samples "$VALIDATION_SAMPLES" \
    --output "$VALIDATION_RESULT" \
    2>&1 | tee "$WORKSPACE/validation_log_${TIMESTAMP}.txt"; then

    echo ""
    echo "  VALIDATION PASSED — using Predibase templates"
    TEMPLATE_MODE="predibase"
else
    echo ""
    echo "  VALIDATION FAILED — falling back to adapter training"

    # -----------------------------------------------------------------------
    # Phase 1.5: Train fallback adapters
    # -----------------------------------------------------------------------
    echo ""
    echo "--- Phase 1.5: Training Fallback Adapters ---"
    echo "  This will take ~45 minutes..."

    TRAINED_DIR="$WORKSPACE/trained_adapters"

    $PY $SCRIPT_DIR/train_adapters.py \
        --base-model "$BASE_MODEL" \
        --output-dir "$TRAINED_DIR" \
        2>&1 | tee "$WORKSPACE/training_log_${TIMESTAMP}.txt"

    # Override adapter dirs to point to trained adapters
    ADAPTER_A_DIR="$TRAINED_DIR/mnli"
    ADAPTER_B_DIR="$TRAINED_DIR/qnli"
    TEMPLATE_MODE="custom"

    echo ""
    echo "  Training complete. Using custom word-label templates."
fi

echo ""
echo "  Template mode: $TEMPLATE_MODE"
echo "  Adapter A: $ADAPTER_A_DIR"
echo "  Adapter B: $ADAPTER_B_DIR"

# ---------------------------------------------------------------------------
# Phases 2-5: Full experiment (audit → merge → evaluate → collect)
# ---------------------------------------------------------------------------
echo ""
echo "--- Phases 2-5: Full Experiment ---"

$PY $SCRIPT_DIR/run_experiment.py \
    --workspace "$WORKSPACE" \
    --base-model "$BASE_MODEL" \
    --adapter-a-dir "$ADAPTER_A_DIR" \
    --adapter-b-dir "$ADAPTER_B_DIR" \
    --output-rank "$OUTPUT_RANK" \
    --output-alpha "$OUTPUT_ALPHA" \
    --max-eval-samples "$MAX_EVAL_SAMPLES" \
    --template-mode "$TEMPLATE_MODE" \
    2>&1 | tee "$WORKSPACE/experiment_log_${TIMESTAMP}.txt"

# ---------------------------------------------------------------------------
# Package results
# ---------------------------------------------------------------------------
echo ""
echo "--- Packaging Results ---"

ARCHIVE="merge_experiment_results_${TIMESTAMP}.tar.gz"
cd /workspace
tar czf "$ARCHIVE" \
    merge_experiment/results/ \
    merge_experiment/*_log_*.txt \
    merge_experiment/validation_result.json \
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
echo "  EXPERIMENT COMPLETE"
echo "  $(date)"
echo "  Template mode used: $TEMPLATE_MODE"
echo "============================================================"
echo ""
echo "  Key artifacts:"
echo "    $WORKSPACE/results/experiment_results.json"
echo "    $WORKSPACE/results/audit/merge_audit.json"
echo "    $WORKSPACE/validation_result.json"
echo "    $WORKSPACE/results/merged_*/adapter_config.json"
echo "    $WORKSPACE/results/evaluations/*.json"
echo ""
echo "  Remember to stop the pod when done!"
echo "============================================================"
