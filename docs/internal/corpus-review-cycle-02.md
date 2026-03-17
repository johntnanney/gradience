# Corpus Review Cycle-02 Runbook

Status: planned  
Planned start: `<YYYY-MM-DD>`

## Purpose

Run a second corpus review cycle with broader real-inventory diversity, then produce an evidence-backed review memo and single calibration decision.

Cycle-02 focus:
- broaden inventory coverage
- track low-risk/core-space disagreement cases explicitly
- keep default behavior frozen unless evidence is strong and narrow

## Hard Guardrails (Freeze)

During cycle-02, do not change:

- default workflow behavior
- strict-QA semantics
- default recommendation logic
- neighborhood grouping logic
- core-space formulas/status thresholds

Allowed work:

- collecting and appending real inventories
- corpus summarization and memo updates
- documentation and review artifacts
- bugfixes only if they block reproducible execution

## Diversity Targets (Cycle-02)

Collect `3–5` inventories with explicit coverage:

- at least `1` inventory with clearer non-checkpoint adapter identity names
- at least `1` inventory with semantically varied adapters
- at least `1` inventory likely to produce medium/high pair risk mix
- at least `1` inventory likely to produce non-singleton neighborhood grouping

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
RUN_DATE=<YYYYMMDD>
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

## Step 3 — Append Corpus Entry Immediately

Append one manifest as soon as each inventory completes:

```bash
python3 scripts/append_corpus_entry.py \
  --run-id "${INVENTORY_ID}_${RUN_DATE}" \
  --date <YYYY-MM-DD> \
  --qa-dir "${RUN_DIR}/qa" \
  --report-dir "${RUN_DIR}/reports" \
  --neighborhood-report "${RUN_DIR}/neighborhoods/neighborhoods.json" \
  --note "cycle-02 real inventory run" \
  --corpus-root results/corpus
```

Rule:
- do not defer append to end-of-cycle batch
- fail fast on any malformed or missing artifact

## Step 4 — Build Corpus Summary Snapshot

After inventory `#3` is appended:

```bash
python3 scripts/summarize_corpus.py \
  --corpus-root results/corpus \
  --emit-json results/corpus/summary_cycle02.json \
  --emit-md results/corpus/summary_cycle02.md
```

Update the review memo immediately after this first snapshot.

## Step 5 — Cycle-02 Memo and Decision Files

Use these files:

- `docs/internal/corpus-review-memo-2026-04.md`
- `docs/internal/selective-calibration-decision-2026-04.md`

Required additional tracker in memo:

- low-risk/core-space mismatch population:
  - `pair_risk=low` and `core_space.status in {marginal, incompatible}`

## Acceptance Gates

Cycle-02 is complete only when all gates pass:

- [ ] at least `3` real inventories appended (`4–5` preferred)
- [ ] diversity targets are covered
- [ ] all cycle-02 manifests strict-load (`gradience.corpus_manifest/v1`)
- [ ] corpus summary snapshot emitted (`summary_cycle02.json` and `.md`)
- [ ] low-risk/core-space mismatch tracker filled in memo
- [ ] explicit decision recorded (`no_change` or one narrow `targeted_calibration`)

## Recommended Decision Rule for Cycle-02

Default to `no_change` unless one issue is:

- repeated
- specific
- clearly evidenced
- small enough to fix in one move
