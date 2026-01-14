#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./scripts/outsider_bench_drill.sh <ref> [out_dir]
#
# Examples:
#   ./scripts/outsider_bench_drill.sh v0.3.4 /tmp/gradience_outsider
#   ./scripts/outsider_bench_drill.sh master
#   ./scripts/outsider_bench_drill.sh 289c054

REF="${1:-master}"
OUTROOT="${2:-}"

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

# Install Gradience + bench deps
# Prefer extras if available; fallback to manual deps if not.
if python -m pip install ".[bench]"; then
  echo "[install] Installed with extras: .[bench]"
else
  echo "[install] Extras 'bench' not available; falling back to manual installs."
  python -m pip install .
  python -m pip install torch transformers peft datasets safetensors pyyaml
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

if [[ $SMOKE_RC -ne 0 ]]; then
  echo "[smoke] Non-zero exit (${SMOKE_RC}). Common cause: probe quality gate in smoke mode."
  echo "[smoke] Continuing to cert seeds anyway."
fi

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

echo
echo "✅ Outsider drill complete."
echo "Aggregate report:"
echo "  ${OUTDIR}/cert_agg/bench_aggregate.md"
echo "  ${OUTDIR}/cert_agg/bench_aggregate.json"
echo

echo "Preview (first 120 lines):"
sed -n '1,120p' "${OUTDIR}/cert_agg/bench_aggregate.md" || true