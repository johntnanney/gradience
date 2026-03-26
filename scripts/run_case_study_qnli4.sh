#!/usr/bin/env bash
set -euo pipefail

INVENTORY_ID="${1:-case_study_qnli4_realmix}"
RUN_DATE="${2:-$(date +%Y%m%d)}"
DATE_ISO="${DATE_ISO:-$(date +%F)}"
BASE_MODEL="${BASE_MODEL:-distilbert-base-uncased}"

RUN_ROOT="results/case_study/${INVENTORY_ID}"

if [[ -e "${RUN_ROOT}" && "${OVERWRITE:-0}" != "1" ]]; then
  echo "Error: ${RUN_ROOT} already exists. Set OVERWRITE=1 to reuse/overwrite files in place."
  exit 1
fi

mkdir -p \
  "${RUN_ROOT}/qa" \
  "${RUN_ROOT}/pair_reports" \
  "${RUN_ROOT}/core_space" \
  "${RUN_ROOT}/inventory" \
  "${RUN_ROOT}/neighborhoods" \
  "${RUN_ROOT}/notes"

# Fixed 4-adapter set (Role A/B/C/D)
ADAPTER_A="results/core_space_benchmark/real_adapters/final_uniform_median_r16"
ADAPTER_B="results/real_inventory_runs/20260317/cycle03_qnli_all_eligible_triplet/adapters/qnli_probe_elig"
ADAPTER_C="results/real_inventory_runs/20260317/cycle03_qnli_mixed_behavior_triplet/adapters/qnli_uniform_weak"
ADAPTER_D="results/real_inventory_runs/20260317/cycle03_qnli_all_eligible_triplet/adapters/qnli_per_layer_elig"

QA_A_SRC="results/real_inventory_runs/20260317/cycle03_real_adapter_triplet/qa/final_uniform_median_r16_qa.json"
QA_B_SRC="results/real_inventory_runs/20260317/cycle03_qnli_all_eligible_triplet/qa/qnli_probe_elig_qa.json"
QA_C_SRC="results/real_inventory_runs/20260317/cycle03_qnli_mixed_behavior_triplet/qa/qnli_uniform_weak_qa.json"
QA_D_SRC="results/real_inventory_runs/20260317/cycle03_qnli_all_eligible_triplet/qa/qnli_per_layer_elig_qa.json"

for p in "${ADAPTER_A}" "${ADAPTER_B}" "${ADAPTER_C}" "${ADAPTER_D}"; do
  if [[ ! -d "${p}" ]]; then
    echo "Error: missing adapter directory: ${p}"
    exit 1
  fi
done

for p in "${QA_A_SRC}" "${QA_B_SRC}" "${QA_C_SRC}" "${QA_D_SRC}"; do
  if [[ ! -f "${p}" ]]; then
    echo "Error: missing source QA artifact: ${p}"
    exit 1
  fi
done

QA_A="${RUN_ROOT}/qa/final_uniform_median_r16_qa.json"
QA_B="${RUN_ROOT}/qa/qnli_probe_elig_qa.json"
QA_C="${RUN_ROOT}/qa/qnli_uniform_weak_qa.json"
QA_D="${RUN_ROOT}/qa/qnli_per_layer_elig_qa.json"

cp "${QA_A_SRC}" "${QA_A}"
cp "${QA_B_SRC}" "${QA_B}"
cp "${QA_C_SRC}" "${QA_C}"
cp "${QA_D_SRC}" "${QA_D}"

cat > "${RUN_ROOT}/notes/inventory_selection_note.md" <<EOF
# Inventory Selection Note

Inventory id: ${INVENTORY_ID}
Run date: ${RUN_DATE}

Roles:
- Role A (strong anchor): final_uniform_median_r16
- Role B (related secondary): qnli_probe_elig
- Role C (weak/exclusion-worthy): qnli_uniform_weak
- Role D (ambiguous/non-obvious): qnli_per_layer_elig

Adapter paths:
- ${ADAPTER_A}
- ${ADAPTER_B}
- ${ADAPTER_C}
- ${ADAPTER_D}

QA source artifacts:
- ${QA_A_SRC}
- ${QA_B_SRC}
- ${QA_C_SRC}
- ${QA_D_SRC}
EOF

run_pair() {
  local pair_id="$1"
  local adapter_a="$2"
  local adapter_b="$3"
  local qa_a="$4"
  local qa_b="$5"
  local out_json="${RUN_ROOT}/pair_reports/${pair_id}.json"
  local out_txt="${RUN_ROOT}/pair_reports/${pair_id}.terminal.txt"

  gradience merge-audit \
    --adapter-a "${adapter_a}" \
    --adapter-b "${adapter_b}" \
    --source-a-qa "${qa_a}" \
    --source-b-qa "${qa_b}" \
    --qa-report \
    --emit-report "${out_json}" \
    > "${out_txt}"
}

# Full 4-adapter pair matrix (6 reports)
run_pair "a_final_vs_b_probe" "${ADAPTER_A}" "${ADAPTER_B}" "${QA_A}" "${QA_B}"
run_pair "a_final_vs_c_weak" "${ADAPTER_A}" "${ADAPTER_C}" "${QA_A}" "${QA_C}"
run_pair "a_final_vs_d_perlayer" "${ADAPTER_A}" "${ADAPTER_D}" "${QA_A}" "${QA_D}"
run_pair "b_probe_vs_c_weak" "${ADAPTER_B}" "${ADAPTER_C}" "${QA_B}" "${QA_C}"
run_pair "b_probe_vs_d_perlayer" "${ADAPTER_B}" "${ADAPTER_D}" "${QA_B}" "${QA_D}"
run_pair "c_weak_vs_d_perlayer" "${ADAPTER_C}" "${ADAPTER_D}" "${QA_C}" "${QA_D}"

# Ambiguous-pair deep inspection (core-space on one pair only)
AMBIG_PAIR_ID="b_probe_vs_d_perlayer"
gradience merge-audit \
  --adapter-a "${ADAPTER_B}" \
  --adapter-b "${ADAPTER_D}" \
  --source-a-qa "${QA_B}" \
  --source-b-qa "${QA_D}" \
  --qa-report \
  --compute-core-space \
  --emit-report "${RUN_ROOT}/core_space/${AMBIG_PAIR_ID}_core_space.json" \
  > "${RUN_ROOT}/core_space/${AMBIG_PAIR_ID}_core_space.terminal.txt"

cat > "${RUN_ROOT}/notes/ambiguous_pair_note.md" <<EOF
# Ambiguous Pair Note

Selected pair: qnli_probe_elig x qnli_per_layer_elig
Pair report id: ${AMBIG_PAIR_ID}

Why selected:
- ordinary pair-risk is plausible but not fully decisive
- practical merge candidate worth deeper structural inspection
- core-space may clarify whether pair should stay in active candidate set

Core-space report:
- ${RUN_ROOT}/core_space/${AMBIG_PAIR_ID}_core_space.json
EOF

gradience summarize-inventory \
  --qa-dir "${RUN_ROOT}/qa" \
  --report-dir "${RUN_ROOT}/pair_reports" \
  --emit-report "${RUN_ROOT}/inventory/inventory_summary.json" \
  > "${RUN_ROOT}/inventory/inventory_summary.terminal.txt"

gradience suggest-neighborhoods \
  --qa-dir "${RUN_ROOT}/qa" \
  --report-dir "${RUN_ROOT}/pair_reports" \
  --emit-report "${RUN_ROOT}/neighborhoods/neighborhoods.json" \
  > "${RUN_ROOT}/neighborhoods/neighborhoods.terminal.txt"

RUN_ID="${INVENTORY_ID}_${RUN_DATE}"
python3 scripts/append_corpus_entry.py \
  --run-id "${RUN_ID}" \
  --date "${DATE_ISO}" \
  --base-model "${BASE_MODEL}" \
  --qa-dir "${RUN_ROOT}/qa" \
  --report-dir "${RUN_ROOT}/pair_reports" \
  --neighborhood-report "${RUN_ROOT}/neighborhoods/neighborhoods.json" \
  --downstream-outcome "${RUN_ROOT}/core_space/${AMBIG_PAIR_ID}_core_space.json" \
  --note "case-study inventory run" \
  --note "ambiguous pair: ${AMBIG_PAIR_ID}" \
  --corpus-root results/corpus \
  > "${RUN_ROOT}/notes/corpus_append.log"

cat > "${RUN_ROOT}/notes/final_decision_note.md" <<EOF
# Final Decision Note

Inventory id: ${INVENTORY_ID}
Corpus run id: ${RUN_ID}

Fill after inspecting outputs:
- which adapter was excluded or downgraded?
- which pair was narrowed/deferred/cautioned?
- what neighborhood structure changed the action plan?
- how is the final action set narrower than naive exploration?
EOF

mkdir -p docs/internal
if [[ ! -f docs/internal/case_study_inventory_candidates.md && -f docs/internal/templates/case-study-inventory-candidates-template.md ]]; then
  cp docs/internal/templates/case-study-inventory-candidates-template.md docs/internal/case_study_inventory_candidates.md
fi
if [[ ! -f docs/internal/case_study_ambiguous_pair_note.md && -f docs/internal/templates/case-study-ambiguous-pair-note-template.md ]]; then
  cp docs/internal/templates/case-study-ambiguous-pair-note-template.md docs/internal/case_study_ambiguous_pair_note.md
fi
if [[ ! -f docs/internal/case_study_publication_summary.md && -f docs/internal/templates/case-study-publication-summary-template.md ]]; then
  cp docs/internal/templates/case-study-publication-summary-template.md docs/internal/case_study_publication_summary.md
fi

echo "Case study run complete: ${RUN_ROOT}"
echo "Corpus manifest: results/corpus/manifests/${RUN_ID}.json"
echo "Next: fill ${RUN_ROOT}/notes/final_decision_note.md and docs/internal/case_study_publication_summary.md"
