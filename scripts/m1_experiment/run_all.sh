#!/usr/bin/env bash
#
# M1 Controlled Interference Experiment -- Master Orchestrator
#
# Runs all 5 phases sequentially. Each phase is independently runnable
# and resumable (skips completed work).
#
# Usage:
#   bash scripts/m1_experiment/run_all.sh [--smoke]
#
# Environment:
#   Expects to run on RunPod with CUDA available.
#   Install: pip install "gradience[bench]" lm-eval scipy scikit-learn

set -euo pipefail

# Allow lm-eval to execute generated code for pass@1 benchmarks (mbpp, humaneval)
export HF_ALLOW_CODE_EVAL="1"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${SCRIPT_DIR}/m1_config.yaml"

# Pass --smoke flag through
EXTRA_ARGS="${*}"

echo "=================================================================="
echo "  M1 Controlled Interference Experiment"
echo "  Config: ${CONFIG}"
echo "  Args: ${EXTRA_ARGS:-none}"
echo "  Started: $(date)"
echo "=================================================================="

echo ""
echo "--- Phase 1: Train Adapters ---"
python "${SCRIPT_DIR}/phase1_train.py" --config "${CONFIG}" ${EXTRA_ARGS}

echo ""
echo "--- Phase 2: Pairwise Merge-Audit ---"
python "${SCRIPT_DIR}/phase2_audit.py" --config "${CONFIG}" ${EXTRA_ARGS}

echo ""
echo "--- Phase 3: Execute Merges ---"
python "${SCRIPT_DIR}/phase3_merge.py" --config "${CONFIG}" ${EXTRA_ARGS}

echo ""
echo "--- Phase 4: Evaluate ---"
python "${SCRIPT_DIR}/phase4_evaluate.py" --config "${CONFIG}" ${EXTRA_ARGS}

echo ""
echo "--- Phase 5: Analyze ---"
python "${SCRIPT_DIR}/phase5_analyze.py" --config "${CONFIG}"

echo ""
echo "=================================================================="
echo "  M1 experiment complete!"
echo "  Results: /workspace/m1/analysis/"
echo "  Finished: $(date)"
echo "=================================================================="
