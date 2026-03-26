# Corpus Foundation (Internal)

## Purpose

The corpus is a repo-native accumulation layer for validated inventory runs.

Each entry links:
- adapter QA artifacts (`gradience.adapter_qa/v1`)
- pair reports (`gradience.merge_qa_report/v1`)
- one neighborhood report (`gradience.merge_neighborhoods/v1`)

The goal is evidence accumulation, not policy automation.

## Layout

Default root:

```text
results/corpus/
  manifests/
    <run_id>.json
```

Manifest schema id:

```text
gradience.corpus_manifest/v1
```

## Append Flow

Use `scripts/append_corpus_entry.py` to add one entry.

Example:

```bash
python3 scripts/append_corpus_entry.py \
  --run-id inventory_safe_small_20260317 \
  --date 2026-03-17 \
  --qa-dir examples/inventories/inventory_safe_small/qa \
  --report-dir examples/inventories/inventory_safe_small/reports \
  --neighborhood-report results/neighborhood_eval/baseline_v1/inventory_safe_small/neighborhood_report.json \
  --note "safe fixture baseline"
```

Behavior:
- validates every referenced artifact before writing
- fails fast on malformed or missing inputs
- writes one manifest JSON only after full validation
- does not silently register partial entries

## Summary Flow

Use `scripts/summarize_corpus.py` to inspect accumulated entries.

Example:

```bash
python3 scripts/summarize_corpus.py \
  --corpus-root results/corpus \
  --emit-json results/corpus/summary.json \
  --emit-md results/corpus/summary.md
```

The summary reports:
- inventory count
- adapter instance count (identity-safe unique across manifests)
- unique adapter count (identity-safe alias for backward compatibility)
- unique adapter display-name count (human-readable labels)
- pair report count
- strategy distribution
- dominant issue distribution
- strict block candidate pair count
- neighborhood totals (groups, excluded, boundary warnings)

### Adapter identity semantics

`scripts/summarize_corpus.py` separates display names from instance identity:

- display names (`adapter_names`) are retained for readability only
- adapter-instance totals are computed from deterministic identity keys

Identity key source order:

1. optional explicit instance id (`adapter_instance_ids[index]` in manifest)
2. optional explicit instance id in QA artifact (`adapter.instance_id`)
3. canonicalized QA `adapter.path`
4. canonicalized QA artifact path

Dedupe semantics:

- corpus-level adapter-instance totals dedupe by identity key across manifests
- repeated references to the same adapter instance are counted once

## Valid Entry Contract

A valid `gradience.corpus_manifest/v1` entry must include:
- `run_id`
- `date` (ISO date/time)
- `base_model`
- `adapter_names` (non-empty)
- `qa_artifact_paths` (non-empty)
- `pair_report_paths` (non-empty)
- `neighborhood_report_path`

Optional:
- `downstream_outcome_paths`
- `notes`

Every referenced artifact path must exist and strict-load under its schema.

## Review Assets

For corpus-backed review cycles, use:

- `docs/internal/corpus-review-cycle-01.md` (execution runbook)
- `docs/internal/corpus-review-cycle-02.md` (execution runbook)
- `docs/internal/corpus-review-cycle-03.md` (execution runbook)
- `docs/internal/templates/corpus-review-memo-template.md`
- `docs/internal/templates/selective-calibration-decision-template.md`
- `docs/internal/corpus-identity-hardening-note.md`
