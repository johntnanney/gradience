# Design Note: Batch / Repeated Preflight Ergonomics (Project E)

**Date:** 2026-03-26
**Status:** In progress
**Roadmap reference:** CPU-Only Roadmap, Phase 2, Project E

## Problem

Running Gradience across multiple inventories or repeated snapshots of the same
inventory currently requires manual orchestration:

```bash
# Manual workflow today:
gradience summarize-inventory --qa-dir v1/qa --report-dir v1/reports \
  --emit-bundle runs/run_001

gradience summarize-inventory --qa-dir v2/qa --report-dir v2/reports \
  --emit-bundle runs/run_002 --previous-run runs/run_001

# ...and so on, manually tracking run IDs and previous-run links
```

There is no way to: auto-discover the previous run, compare all runs at once,
or see an aggregate view across the run history.

## Design

Three additions, layered from smallest to largest:

### 1. Auto-discover previous run (`--auto-previous`)

When `--emit-bundle <DIR>` is used, if no `--previous-run` is given,
automatically look for a `latest` symlink in the bundle's parent directory.
If found, use it as the previous run.

This eliminates the manual `--previous-run` chaining. The first run in a
directory has no previous. The second run finds the first via `latest`.
The third finds the second. No user action needed.

Implementation: a single function `discover_previous_run(bundle_parent)` in
`run_bundle.py`. Called from `cmd_summarize_inventory()` when
`--previous-run` is not explicitly provided.

### 2. Cross-run summary (`gradience batch-summary`)

A new CLI command that reads all `preflight_summary.json` files under a
directory tree and produces a comparison table.

```bash
gradience batch-summary --run-dir runs/
```

Output: a markdown table showing each run's key metrics (adapter count,
pair count, retained candidates, reduction%, evidence ratio, risk
distribution) side by side.

Implementation: a new module `gradience/vnext/inventory/batch.py` with:
- `collect_run_summaries(root_dir)` — discovers all preflight_summary.json files
- `build_batch_summary(summaries)` — aggregates into a cross-run table structure
- `format_batch_summary(batch)` — terminal-formatted table
- `emit_batch_summary_md(batch, path)` — writes markdown + JSON

### 3. Batch summary output files

When `--emit-report` is provided to `batch-summary`:
- `batch_summary.json` — machine-readable array of per-run metrics
- `batch_summary.md` — human-readable table with trend indicators

## What this does NOT do

- No batch orchestration loop (user still runs each preflight individually)
- No config file for batch definitions (unnecessary complexity)
- No statistical trend analysis (overkill for current usage patterns)

The point is visibility across runs, not automation of runs.

## Completion signal

A user can:
1. Run repeated preflights without manually tracking `--previous-run`
2. See all runs side-by-side in one table
3. Quickly spot whether the candidate set is narrowing, broadening, or stable
