# Corpus Review Cycle-03 Runbook

Status: planned  
Planned start: `<YYYY-MM-DD>`

## Purpose

Run a third corpus review cycle focused on targeted diversity acquisition, then record one evidence-backed calibration decision.

Cycle-03 focus:
- acquire inventories that stress neighborhood grouping diversity
- test strict-QA in a mixed behavioral-evidence middle case
- continue explicit low-risk/core-space mismatch tracking
- follow through on corpus identity hardening so adapter-instance counting is trustworthy

## Hard Guardrails (Freeze)

During cycle-03, do not change:

- default workflow behavior
- strict-QA semantics
- default recommendation logic
- neighborhood grouping logic
- core-space formulas/status thresholds

Allowed work:

- collecting and appending real inventories
- corpus summarization and memo updates
- identity hardening implementation/follow-through
- documentation and review artifacts
- bugfixes only if they block reproducible execution

## Targeted Diversity Gates (Required)

Cycle-03 must include all four gates:

1. at least one inventory likely to produce non-singleton neighborhoods
2. at least one inventory with mixed behavioral evidence so strict-QA gets a real middle-case test
3. continued explicit tracking of low-risk/core-space mismatch cases
4. follow-through on corpus identity hardening so adapter-instance counting is trustworthy across runs

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

```bash
gradience audit-adapter \
  --peft-dir <adapter_path> \
  --base-model <base_model_id> \
  --out "${RUN_DIR}/qa/<adapter_name>_qa.json"
```

### 1B) Pair reports (default path)

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

```bash
gradience summarize-inventory \
  --qa-dir "${RUN_DIR}/qa" \
  --report-dir "${RUN_DIR}/reports" \
  --emit-report "${RUN_DIR}/inventory/inventory_summary.json"

gradience suggest-neighborhoods \
  --qa-dir "${RUN_DIR}/qa" \
  --report-dir "${RUN_DIR}/reports" \
  --emit-report "${RUN_DIR}/neighborhoods/neighborhoods.json"
```

## Step 3 — Append Corpus Entry Immediately

```bash
python3 scripts/append_corpus_entry.py \
  --run-id "${INVENTORY_ID}_${RUN_DATE}" \
  --date <YYYY-MM-DD> \
  --qa-dir "${RUN_DIR}/qa" \
  --report-dir "${RUN_DIR}/reports" \
  --neighborhood-report "${RUN_DIR}/neighborhoods/neighborhoods.json" \
  --note "cycle-03 real inventory run" \
  --corpus-root results/corpus
```

Rules:
- do not defer append to end-of-cycle batch
- fail fast on malformed or missing artifacts

## Step 4 — Build Corpus Summary Snapshot

After inventory `#3` is appended:

```bash
python3 scripts/summarize_corpus.py \
  --corpus-root results/corpus \
  --emit-json results/corpus/summary_cycle03.json \
  --emit-md results/corpus/summary_cycle03.md
```

## Step 5 — Cycle-03 Memo and Decision Files

Use these files:

- `docs/internal/corpus-review-memo-2026-05.md`
- `docs/internal/selective-calibration-decision-2026-05.md`

Required memo tracker:

- low-risk/core-space mismatch population:
  - `pair_risk=low` and `core_space.status in {marginal, incompatible}`

Required memo section:

- strict-QA middle-case analysis for mixed behavioral-evidence inventories

## Step 6 — Identity Hardening Follow-through

Before closing cycle-03:

- document current identity-hardening status (`implemented`, `in-flight`, or `blocked`)
- if implemented, report adapter-instance counting method used in summaries
- if in-flight/blocked, record a concrete next action and owner

Reference:
- `docs/internal/corpus-identity-hardening-note.md`

## Acceptance Gates

Cycle-03 is complete only when all gates pass:

- [ ] at least `3` real inventories appended (`4–5` preferred)
- [ ] at least one inventory likely to produce non-singleton neighborhoods
- [ ] at least one inventory with mixed behavioral evidence (strict-QA middle-case)
- [ ] low-risk/core-space mismatch tracker filled in memo
- [ ] identity hardening follow-through status recorded
- [ ] all cycle-03 manifests strict-load (`gradience.corpus_manifest/v1`)
- [ ] corpus summary snapshot emitted (`summary_cycle03.json` and `.md`)
- [ ] explicit decision recorded (`no_change` or one narrow `targeted_calibration`)

## Recommended Decision Rule for Cycle-03

Default to `no_change` unless one issue is:

- repeated
- specific
- clearly evidenced
- cross-inventory consistent
- small enough to fix in one move
