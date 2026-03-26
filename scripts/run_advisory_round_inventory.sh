#!/usr/bin/env bash
set -euo pipefail

# Usage: bash scripts/run_advisory_round_inventory.sh <inventory_id> <adapter_a> <qa_a> [<adapter_b> <qa_b> ...]
# Adapters and QA files are passed as alternating pairs.

INVENTORY_ID="${1:?Usage: $0 <inventory_id> <adapter_a> <qa_a> [adapter_b qa_b ...]}"
shift

RUN_ROOT="results/task_advisory_round/${INVENTORY_ID}"

if [[ -e "${RUN_ROOT}" && "${OVERWRITE:-0}" != "1" ]]; then
  echo "Error: ${RUN_ROOT} already exists. Set OVERWRITE=1 to reuse."
  exit 1
fi

mkdir -p \
  "${RUN_ROOT}/qa" \
  "${RUN_ROOT}/pair_reports" \
  "${RUN_ROOT}/inventory" \
  "${RUN_ROOT}/neighborhoods" \
  "${RUN_ROOT}/notes"

# Parse adapter/qa pairs from args
declare -a ADAPTERS=()
declare -a QA_FILES=()
declare -a ADAPTER_NAMES=()

while [[ $# -ge 2 ]]; do
  adapter_path="$1"
  qa_path="$2"
  shift 2

  if [[ ! -d "${adapter_path}" ]]; then
    echo "Error: missing adapter directory: ${adapter_path}"
    exit 1
  fi
  if [[ ! -f "${qa_path}" ]]; then
    echo "Error: missing QA file: ${qa_path}"
    exit 1
  fi

  adapter_name=$(basename "${adapter_path}")
  ADAPTERS+=("${adapter_path}")
  QA_FILES+=("${qa_path}")
  ADAPTER_NAMES+=("${adapter_name}")

  # Copy QA to run dir
  cp "${qa_path}" "${RUN_ROOT}/qa/${adapter_name}_qa.json"
done

N=${#ADAPTERS[@]}
echo "Inventory: ${INVENTORY_ID}"
echo "Adapters: ${N}"
for i in $(seq 0 $((N-1))); do
  echo "  [${i}] ${ADAPTER_NAMES[$i]} <- ${ADAPTERS[$i]}"
done

# Run all pairwise merge reports
echo ""
echo "Running pairwise merge reports..."
for i in $(seq 0 $((N-1))); do
  for j in $(seq $((i+1)) $((N-1))); do
    pair_id="${ADAPTER_NAMES[$i]}_vs_${ADAPTER_NAMES[$j]}"
    out_json="${RUN_ROOT}/pair_reports/${pair_id}.json"
    out_txt="${RUN_ROOT}/pair_reports/${pair_id}.terminal.txt"

    echo "  Pair: ${pair_id}"
    gradience merge-audit \
      --adapter-a "${ADAPTERS[$i]}" \
      --adapter-b "${ADAPTERS[$j]}" \
      --source-a-qa "${RUN_ROOT}/qa/${ADAPTER_NAMES[$i]}_qa.json" \
      --source-b-qa "${RUN_ROOT}/qa/${ADAPTER_NAMES[$j]}_qa.json" \
      --qa-report \
      --emit-report "${out_json}" \
      > "${out_txt}"
  done
done

# Inventory summary
echo ""
echo "Running inventory summary..."
gradience summarize-inventory \
  --qa-dir "${RUN_ROOT}/qa" \
  --report-dir "${RUN_ROOT}/pair_reports" \
  --emit-report "${RUN_ROOT}/inventory/inventory_summary.json" \
  > "${RUN_ROOT}/inventory/inventory_summary.terminal.txt"

# Neighborhoods
echo ""
echo "Running neighborhoods..."
gradience suggest-neighborhoods \
  --qa-dir "${RUN_ROOT}/qa" \
  --report-dir "${RUN_ROOT}/pair_reports" \
  --emit-report "${RUN_ROOT}/neighborhoods/neighborhoods.json" \
  > "${RUN_ROOT}/neighborhoods/neighborhoods.terminal.txt"

# Corpus entry
echo ""
echo "Appending corpus entry..."
DATE_ISO=$(date +%F)
RUN_DATE=$(date +%Y%m%d)
RUN_ID="${INVENTORY_ID}_${RUN_DATE}"

python3 scripts/append_corpus_entry.py \
  --run-id "${RUN_ID}" \
  --date "${DATE_ISO}" \
  --base-model "distilbert-base-uncased" \
  --qa-dir "${RUN_ROOT}/qa" \
  --report-dir "${RUN_ROOT}/pair_reports" \
  --neighborhood-report "${RUN_ROOT}/neighborhoods/neighborhoods.json" \
  --note "advisory round 01 inventory" \
  --corpus-root results/corpus \
  > "${RUN_ROOT}/notes/corpus_append.log" 2>&1 || true

echo ""
echo "Inventory ${INVENTORY_ID} complete: ${RUN_ROOT}"
