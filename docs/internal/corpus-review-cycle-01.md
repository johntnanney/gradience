# Corpus Review Cycle-01 Runbook

Status: active  
Start date: 2026-03-17

## Purpose

Run real adapter inventories through the current preflight + advanced optional tier, append validated entries to the corpus, and produce the first evidence-backed review memo.

This cycle prioritizes observation over invention.

## Hard Guardrails (Freeze)

During cycle-01, do not change:

- default workflow behavior
- strict-QA semantics
- default recommendation logic
- neighborhood grouping logic
- core-space formulas/status thresholds

Allowed work:

- running inventories
- appending corpus entries
- summarizing corpus behavior
- writing review/decision memos
- bugfixes only if they block reproducible execution

## Inputs Per Inventory

Each inventory run must include:

- adapter directories (PEFT)
- pair list to audit
- base model identifier
- output run id

Optional:

- list of ambiguous pairs to run with core-space
- downstream outcome files

## Required Output Layout

Use one run directory per inventory:

```text
results/real_inventory_runs/<YYYYMMDD>/<inventory_id>/
  qa/
  reports/
  inventory/
  neighborhoods/
  notes/
```

## Step 1 — Run QA + Pair Reports

Set shell variables first:

```bash
RUN_DATE=20260317
INVENTORY_ID=<inventory_id>
RUN_DIR="results/real_inventory_runs/${RUN_DATE}/${INVENTORY_ID}"
mkdir -p "${RUN_DIR}/qa" "${RUN_DIR}/reports" "${RUN_DIR}/inventory" "${RUN_DIR}/neighborhoods" "${RUN_DIR}/notes"
```

### 1A) Adapter QA artifacts

Run once per adapter:

```bash
gradience audit-adapter \
  --peft-dir <adapter_path> \
  --base-model <base_model_id> \
  --out "${RUN_DIR}/qa/<adapter_name>_qa.json"
```

### 1B) Pair reports (default path)

Run once per pair:

```bash
gradience merge-audit \
  --adapter-a <adapter_a_path> \
  --adapter-b <adapter_b_path> \
  --source-a-qa "${RUN_DIR}/qa/<adapter_a_name>_qa.json" \
  --source-b-qa "${RUN_DIR}/qa/<adapter_b_name>_qa.json" \
  --qa-report \
  --emit-report "${RUN_DIR}/reports/<pair_id>.json"
```

### 1C) Ambiguous pairs (optional core-space)

Only for ambiguous pairs:

```bash
gradience merge-audit \
  --adapter-a <adapter_a_path> \
  --adapter-b <adapter_b_path> \
  --source-a-qa "${RUN_DIR}/qa/<adapter_a_name>_qa.json" \
  --source-b-qa "${RUN_DIR}/qa/<adapter_b_name>_qa.json" \
  --qa-report \
  --compute-core-space \
  --emit-report "${RUN_DIR}/reports/<pair_id>_core_space.json"
```

## Step 2 — Inventory + Neighborhood Outputs

### 2A) Inventory summary

```bash
gradience summarize-inventory \
  --qa-dir "${RUN_DIR}/qa" \
  --report-dir "${RUN_DIR}/reports" \
  --emit-report "${RUN_DIR}/inventory/inventory_summary.json"
```

### 2B) Neighborhood suggestion report

```bash
gradience suggest-neighborhoods \
  --qa-dir "${RUN_DIR}/qa" \
  --report-dir "${RUN_DIR}/reports" \
  --emit-report "${RUN_DIR}/neighborhoods/neighborhoods.json"
```

## Step 3 — Append Corpus Entry (strict)

Use one manifest per inventory. Fail on any malformed or missing artifact.

```bash
python3 scripts/append_corpus_entry.py \
  --run-id "${INVENTORY_ID}_${RUN_DATE}" \
  --date 2026-03-17 \
  --qa-dir "${RUN_DIR}/qa" \
  --report-dir "${RUN_DIR}/reports" \
  --neighborhood-report "${RUN_DIR}/neighborhoods/neighborhoods.json" \
  --note "cycle-01 real inventory run" \
  --corpus-root results/corpus
```

If available, append downstream outcomes:

```bash
python3 scripts/append_corpus_entry.py \
  --run-id "${INVENTORY_ID}_${RUN_DATE}" \
  --date 2026-03-17 \
  --qa-dir "${RUN_DIR}/qa" \
  --report-dir "${RUN_DIR}/reports" \
  --neighborhood-report "${RUN_DIR}/neighborhoods/neighborhoods.json" \
  --downstream-outcome "${RUN_DIR}/notes/<outcome_file>.json" \
  --note "cycle-01 real inventory run" \
  --corpus-root results/corpus \
  --overwrite
```

## Step 4 — Build Corpus Summary Snapshot

After at least 3 inventories are appended:

```bash
python3 scripts/summarize_corpus.py \
  --corpus-root results/corpus \
  --emit-json results/corpus/summary_cycle01.json \
  --emit-md results/corpus/summary_cycle01.md
```

## Step 5 — Write Memos from Templates

Create memo files for this cycle:

```bash
cp docs/internal/templates/corpus-review-memo-template.md \
  docs/internal/corpus-review-memo-2026-03.md

cp docs/internal/templates/selective-calibration-decision-template.md \
  docs/internal/selective-calibration-decision-2026-03.md
```

Then fill both files using:

- `results/corpus/summary_cycle01.json`
- `results/corpus/summary_cycle01.md`
- manifest entries in `results/corpus/manifests/`
- selected representative run artifacts under `results/real_inventory_runs/`

Companion quick checklist: `docs/internal/investigator-checklist-cycle-01.md`

## Acceptance Gates

Cycle-01 is complete only when all gates pass:

- [ ] at least 3 real inventories appended to corpus
- [ ] all cycle-01 manifests strict-load (`gradience.corpus_manifest/v1`)
- [ ] each inventory has QA, pair reports, inventory summary, and neighborhoods report
- [ ] corpus summary snapshot emitted (`summary_cycle01.json` and `.md`)
- [ ] corpus review memo completed
- [ ] selective calibration decision memo completed
- [ ] explicit decision recorded: `no_change` or one scoped `targeted_calibration`

## Recommended Decision Rule for Cycle-01

Default to `no_change` unless corpus evidence clearly supports one narrow adjustment with low blast radius.

If evidence is mixed, choose `defer` and run one additional collection cycle.
