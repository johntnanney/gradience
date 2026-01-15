#!/usr/bin/env bash
set -euo pipefail

# Gradience Outsider Bench Drill - External User Simulation & Acceptance Test
#
# This script simulates a fresh external user experience and validates critical
# Gradience bench behaviors end-to-end. It serves as both an acceptance test
# and a confidence-building tool for releases.
#
# Validates:
#   • Smoke mode: exits 0 even with undertrained probe (UNDERTRAINED_SMOKE status)
#   • Cert mode: multi-seed validation completes successfully  
#   • Aggregation: smoke runs excluded by default, cert aggregate generated
#   • Fresh environment: clean clone, fresh venv, minimal dependencies
#
# Usage:
#   ./scripts/outsider_bench_drill.sh <ref> [out_dir] [smoke_only]
#
# Examples:
#   ./scripts/outsider_bench_drill.sh v0.3.5                    # Full test (smoke + cert)
#   ./scripts/outsider_bench_drill.sh v0.3.5 /tmp/test         # Full test with custom output
#   ./scripts/outsider_bench_drill.sh v0.3.5 /tmp/test true    # Smoke-only mode (faster)
#   ./scripts/outsider_bench_drill.sh master "" true           # Smoke-only with default output

REF="${1:-master}"
OUTROOT="${2:-}"
SMOKE_ONLY="${3:-false}"  # Optional third parameter for smoke-only mode

PYTHON="${PYTHON:-python3}"

# Create a fresh workspace
WORKDIR="$(mktemp -d 2>/dev/null || mktemp -d -t gradience_outsider)"
REPO_DIR="$WORKDIR/gradience"
VENV_DIR="$WORKDIR/.venv"

if [[ -n "${OUTROOT}" ]]; then
  OUTDIR="${OUTROOT}"
else
  OUTDIR="$WORKDIR/out"
fi

echo "========================================"
echo "Gradience Outsider Bench Drill"
echo "========================================"
echo "Ref:     ${REF}"
echo "Workdir: ${WORKDIR}"
echo "Outdir:  ${OUTDIR}"
echo

command -v git >/dev/null || { echo "ERROR: git is required"; exit 1; }
command -v "${PYTHON}" >/dev/null || { echo "ERROR: ${PYTHON} not found"; exit 1; }

# Clone + checkout ref
git clone https://github.com/johntnanney/gradience.git "${REPO_DIR}"
cd "${REPO_DIR}"
git checkout "${REF}"

# Safety: must be repo root (pyproject.toml must exist)
if [[ ! -f "pyproject.toml" ]]; then
  echo "ERROR: pyproject.toml not found. Not at repo root?"
  echo "PWD: $(pwd)"
  exit 1
fi

# Fresh venv
"${PYTHON}" -m venv "${VENV_DIR}"
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
python -m pip install -U pip

# Install Gradience + bench deps using extras (v0.3.5+)
# This automatically pulls in transformers, peft, datasets, safetensors, accelerate
if python -m pip install -e ".[bench]"; then
  echo "[install] ✅ Installed with extras: .[bench]"
  echo "[install] Includes: transformers, peft, datasets, safetensors, accelerate"
else
  echo "[install] ⚠️  Extras 'bench' not available; falling back to manual installs."
  echo "[install] This may indicate an older version or installation issue."
  python -m pip install -e .
  python -m pip install transformers peft datasets safetensors accelerate
fi

# Sanity checks
python -c "import gradience; print('gradience import OK:', gradience.__file__)"
if command -v gradience >/dev/null; then
  echo "[sanity] gradience CLI found: $(command -v gradience)"
  gradience --help >/dev/null
else
  echo "[sanity] WARNING: gradience CLI not found in venv PATH"
fi

python -m gradience.bench.run_bench --help >/dev/null

mkdir -p "${OUTDIR}"

echo
echo "----------------------------------------"
echo "1) Smoke bench (NON-BLOCKING: pipeline sanity only)"
echo "----------------------------------------"

set +e
python -m gradience.bench.run_bench \
  --config gradience/bench/configs/distilbert_sst2.yaml \
  --output "${OUTDIR}/bench_smoke" \
  --smoke
SMOKE_RC=$?
set -e

echo "[smoke] Exit code: ${SMOKE_RC}"

# Validate smoke mode behavior (v0.3.5+)
if [[ $SMOKE_RC -eq 0 ]]; then
  echo "[smoke] ✅ Expected: smoke mode never hard-fails (exit 0)"
else
  echo "[smoke] ❌ Unexpected: smoke mode should exit 0 even with undertrained probe"
  echo "[smoke] This may indicate a regression in smoke mode implementation."
fi

# Check bench.json status
SMOKE_BENCH="${OUTDIR}/bench_smoke/bench.json"
if [[ -f "$SMOKE_BENCH" ]]; then
  SMOKE_STATUS=$(python3 -c "
import json, sys
try:
    with open('${SMOKE_BENCH}') as f: 
        data = json.load(f)
    print(data.get('status', 'UNKNOWN'))
except: 
    print('ERROR')
" 2>/dev/null || echo "ERROR")
  
  echo "[smoke] Status: ${SMOKE_STATUS}"
  
  case "$SMOKE_STATUS" in
    "UNDERTRAINED_SMOKE")
      echo "[smoke] ✅ Expected: UNDERTRAINED_SMOKE (probe failed but continuing in smoke mode)"
      ;;
    "COMPLETE")
      echo "[smoke] ✅ Acceptable: COMPLETE (probe passed in smoke mode)"
      ;;
    "UNDERTRAINED")
      echo "[smoke] ❌ Unexpected: should be UNDERTRAINED_SMOKE in smoke mode, not UNDERTRAINED"
      ;;
    *)
      echo "[smoke] ⚠️  Unexpected status: ${SMOKE_STATUS}"
      ;;
  esac
else
  echo "[smoke] ❌ No bench.json found at ${SMOKE_BENCH}"
fi

if [[ "${SMOKE_ONLY}" == "true" ]]; then
  echo
  echo "🚀 SMOKE-ONLY MODE: Skipping cert runs for faster validation"
  echo
else
  echo
  echo "----------------------------------------"
  echo "2) Cert seeds (42/123/456)"
  echo "----------------------------------------"
  for seed in 42 123 456; do
    python -m gradience.bench.run_bench \
      --config "gradience/bench/configs/distilbert_sst2_certifiable_seed${seed}.yaml" \
      --output "${OUTDIR}/cert_seed${seed}"
  done

  echo
  echo "----------------------------------------"
  echo "3) Aggregate"
  echo "----------------------------------------"
  python gradience/bench/aggregate.py \
    "${OUTDIR}/cert_seed42" \
    "${OUTDIR}/cert_seed123" \
    "${OUTDIR}/cert_seed456" \
    --output "${OUTDIR}/cert_agg"
fi

# Validate aggregate results (only if not smoke-only)
if [[ "${SMOKE_ONLY}" != "true" ]]; then
  AGG_MD="${OUTDIR}/cert_agg/bench_aggregate.md"
  AGG_JSON="${OUTDIR}/cert_agg/bench_aggregate.json"

  if [[ -f "$AGG_MD" && -f "$AGG_JSON" ]]; then
    echo "[aggregate] ✅ Aggregate files created successfully"
    
    # Check that smoke runs are properly excluded 
    if ! grep -q "bench_smoke" "$AGG_MD" 2>/dev/null; then
      echo "[aggregate] ✅ Expected: smoke runs excluded from aggregation by default"
    else
      echo "[aggregate] ⚠️  Unexpected: smoke runs found in aggregation (should be excluded by default)"
    fi
    
    # Check for expected content
    if grep -q "Certifiable" "$AGG_MD" 2>/dev/null; then
      echo "[aggregate] ✅ Expected: 'Certifiable' content found in aggregate report"
    else
      echo "[aggregate] ⚠️  Missing: 'Certifiable' content not found in aggregate report"
    fi
    
  else
    echo "[aggregate] ❌ Missing aggregate output files"
    echo "  Expected: ${AGG_MD}"
    echo "  Expected: ${AGG_JSON}"
  fi
else
  echo "[aggregate] ⏭️  Skipped in smoke-only mode"
fi

echo
echo "----------------------------------------"
echo "4) Package Artifacts"
echo "----------------------------------------"

# Create tarball with all results
if [[ "${SMOKE_ONLY}" == "true" ]]; then
  ARTIFACT_NAME="gradience_outsider_drill_${REF}_smoke_$(date +%Y%m%d_%H%M%S).tgz"
  ARTIFACT_PATH="${OUTDIR}/${ARTIFACT_NAME}"
  tar -czf "${ARTIFACT_PATH}" -C "${OUTDIR}" bench_smoke
else
  ARTIFACT_NAME="gradience_outsider_drill_${REF}_$(date +%Y%m%d_%H%M%S).tgz"
  ARTIFACT_PATH="${OUTDIR}/${ARTIFACT_NAME}"
  tar -czf "${ARTIFACT_PATH}" -C "${OUTDIR}" \
    bench_smoke \
    cert_seed42 \
    cert_seed123 \
    cert_seed456 \
    cert_agg
fi

echo "[artifacts] Created: ${ARTIFACT_PATH}"

# Extract key metrics for final summary
SMOKE_STATUS=""
CERT_POLICY_STATUS=""
CERT_COMPRESSION=""
CERT_PASS_RATE=""

if [[ -f "${OUTDIR}/bench_smoke/bench.json" ]]; then
  SMOKE_STATUS=$(python3 -c "
import json
try:
    with open('${OUTDIR}/bench_smoke/bench.json') as f: 
        data = json.load(f)
    print(data.get('status', 'UNKNOWN'))
except: 
    print('ERROR')
" 2>/dev/null || echo "ERROR")
fi

if [[ -f "${OUTDIR}/cert_agg/bench_aggregate.json" ]]; then
  CERT_INFO=$(python3 -c "
import json
try:
    with open('${OUTDIR}/cert_agg/bench_aggregate.json') as f: 
        data = json.load(f)
    policy = data.get('policy_compliance', {})
    best = data.get('best_compression', {})
    print(f\"{policy.get('status', 'UNKNOWN')}|{best.get('compression_ratio', 'N/A')}|{policy.get('pass_rate', 'N/A')}\")
except: 
    print('ERROR|N/A|N/A')
" 2>/dev/null || echo "ERROR|N/A|N/A")
  
  IFS='|' read -r CERT_POLICY_STATUS CERT_COMPRESSION CERT_PASS_RATE <<< "$CERT_INFO"
fi

echo
echo "=================================================================="
echo "🎯 GRADIENCE OUTSIDER DRILL COMPLETE"
echo "=================================================================="
echo "REF: ${REF}"
echo "ARTIFACT: ${ARTIFACT_PATH}"
echo
echo "SMOKE MODE:"
echo "  Status: ${SMOKE_STATUS:-N/A}"
echo "  Expected: UNDERTRAINED_SMOKE or COMPLETE (exit 0 in both cases)"
echo
echo "CERTIFICATION MODE:"
echo "  Policy Status: ${CERT_POLICY_STATUS:-N/A}"  
echo "  Best Compression: ${CERT_COMPRESSION:-N/A}"
echo "  Pass Rate: ${CERT_PASS_RATE:-N/A}"
echo
echo "KEY ARTIFACTS:"
echo "  • Smoke: ${OUTDIR}/bench_smoke/bench.json"
echo "  • Aggregate: ${OUTDIR}/cert_agg/bench_aggregate.md"
echo "  • Package: ${ARTIFACT_PATH}"
echo
if [[ "${SMOKE_ONLY}" == "true" ]]; then
  # Smoke-only mode: only check smoke status
  if [[ "${SMOKE_STATUS}" =~ ^(UNDERTRAINED_SMOKE|COMPLETE)$ ]]; then
    echo "✅ ACCEPTANCE TEST: PASSED (SMOKE-ONLY)"
    echo "   Gradience ${REF} smoke test successful - basic functionality confirmed"
  else
    echo "❌ ACCEPTANCE TEST: FAILED (SMOKE-ONLY)" 
    echo "   Smoke test issues - review outputs before proceeding"
  fi
else
  # Full mode: check both smoke and cert
  if [[ "${SMOKE_STATUS}" =~ ^(UNDERTRAINED_SMOKE|COMPLETE)$ && "${CERT_POLICY_STATUS}" =~ ^(COMPLIANT|PARTIAL)$ ]]; then
    echo "✅ ACCEPTANCE TEST: PASSED"
    echo "   Gradience ${REF} is ready for external users"
  else
    echo "❌ ACCEPTANCE TEST: FAILED"
    echo "   Issues detected - review outputs before release"
  fi
fi
echo "=================================================================="
echo

if [[ "${SMOKE_ONLY}" != "true" && -f "${OUTDIR}/cert_agg/bench_aggregate.md" ]]; then
  echo "Preview (first 120 lines of aggregate report):"
  sed -n '1,120p' "${OUTDIR}/cert_agg/bench_aggregate.md" || true
else
  echo "Preview (smoke bench.json):"
  cat "${OUTDIR}/bench_smoke/bench.json" 2>/dev/null || echo "No smoke results found"
fi