# Phase 5 Dry-Run Workplan — 2026-04-26

**Trigger.** GPU run mid-flight (266/672 conditions complete after the 2026-04-26 22:55 UTC budget-driven scope amendment). Want to exercise the full Phase 5 analysis pipeline (`scripts/04` → `99`) on the partial GPU output now, so any pipeline-integration issues surface while there's still time to fix them — rather than at the moment Phase 5 launches in earnest after GPU completion.

**Goal.** Catch code-level integration issues (script crashes, schema mismatches, dependency drift) and partial-data-specific edge cases (sparse cells, empty cells, regime-classification on mid-completion data) on a known-incomplete corpus, where the resulting numbers are not for interpretation. Output: a clean end-to-end run log + any patches needed before the real Phase 5.

**Out of scope.** Interpreting any of the resulting numbers. The partial corpus is missing pythia_1_4b's primary panel completion checks beyond the GSM8K cells, all of pythia_410m beyond 42 conditions, and all of qwen2_5_1_5b. None of the H1/H2/H3/H4 decisions is supposed to land here.

---

## Sandbox-side verification (already done)

Scripts 04 + 05 ran clean against the in-tree test fixtures (`tests/fixtures/raw/arc_challenge__pythia_410m__P1_original__s42__{ll_norm,generate_parse}/`):

- `04_normalize_outputs.py` — produced `item_level.parquet` with 8 rows from 2 conditions; complete=2, normalized_ok=2, failed=0, skipped=0. Plumbing handles `condition_id`, `run_metadata.json`, schemas, and the empty-subject-id case correctly.
- `05_make_condition_scores.py` — produced `condition_level.csv` with 2 rows. Confirmed parseability_rate is NaN for ll_norm conditions (correct behavior; script 07 handles this case explicitly at lines 110–114, defaulting to `g_theory` regime when parseability data is empty).
- One **pandas FutureWarning** surfaced in `05_make_condition_scores.py:119` on `.fillna(False)` (object-dtype downcasting). Cosmetic; not blocking but worth a small follow-up patch (`pd.set_option('future.no_silent_downcasting', True)` at script top, or change to `.fillna(False).infer_objects(copy=False)`).

The static-analysis pass on `06`–`10`, `98`, `99` (below) identifies the partial-data risk areas the rsync-and-run pass should look for.

## Partial-data risk areas (from static analysis)

Each is a place where the partial-corpus data shape could trigger a code path that wasn't covered by the test-fixture suite.

**Risk A — Empty cells (qwen_2_5_1_5b at 0/224).** Scripts that operate per-cell will produce empty output for cells with no data. Static analysis confirms graceful handling: `06`'s cascade falls through to a Level 4 fallback; `07` returns `"g_theory"` regime by default if parseability data is empty; `08` returns an empty DataFrame on `sub.empty`; `09` has explicit empty checks at lines 236, 368, 446, 452. Watch for: warnings about empty cells should be informational, not errors.

**Risk B — Mid-completion cells (pythia_410m partial).** Some (model, benchmark, prompt, seed, scoring_rule) tuples missing. The variance-components fit on such a cell will use only the conditions that completed, producing biased estimates. **The pipeline will not crash — but the resulting numbers are not interpretable.** The dry-run's job is to confirm the pipeline runs to completion and emits clearly-labeled outputs, *not* to interpret the numbers. Output sanity check: every cell-level entry in `tolerance_by_cell.csv` should carry an `n_conditions_used` column or equivalent; if not, that's a documentation gap to fix.

**Risk C — Manifest condition_id format mismatch.** Real GPU output uses condition_ids with **four underscores** for the empty-subject-id slot (`arc_challenge____pythia_1_4b__P1_original__s123__generate_parse`); test fixtures use **two underscores**. Verified: `04_normalize_outputs.py` reads condition_id from the manifest CSV and looks for the matching directory in `--raw-dir`, so the format is whatever the manifest says. Risk fires only if the manifest and the on-disk directory names disagree.

**Risk D — Cells with only one scoring rule available.** A cell with only `ll_norm` (or only `generate_parse`) conditions will produce `parseability_rate = NaN` for half the design. Script `07` handles this at lines 110–114 (defaults to `g_theory` regime when parseability data is empty). For cells expected to be `parse_failure_dominated` (base-model G&P), but where the partial run has only completed the LL conditions, the regime classification will be wrong. **This is expected on partial data and recovers automatically when the full G&P conditions complete.**

**Risk E — Bootstrap with sparse data.** Bootstrap resampling on a cell with 8 conditions (instead of the full 24) will produce wider CIs but should compute. Watch for: any nan-on-resample warnings in `08`'s output; verify `n_resamples` parameter still completes within reasonable time.

**Risk F — `analysis_config.yaml` resolution.** Script `06` requires `--config` pointing at a YAML with the `tolerance.parse_failure_threshold: 0.30` entry per the v1.1.2 amendment. Verify the partial-data run uses the locked config (config_hash `fbc4a5dd`), not a stale one.

**Risk G — `condition_status` column in manifest.** The full manifest has `condition_status: pending` for all rows pre-run (per D-17). The partial-data run needs to either (a) update `condition_status: complete` for the 266 done conditions before script 04 runs, or (b) script 04 needs to gracefully skip pending-but-no-data rows. Verified: the test-fixture run had `condition_status: complete` set in the synthetic manifest. **For the partial-data dry run, the manifest must be updated to reflect what's actually completed**, or script 04 will treat all 672 conditions as expected-to-have-data and fail on the 406 incomplete ones.

---

## Concrete dry-run sequence (user's machine)

The pod has the partial outputs at `/workspace/study/runs/raw/`. Sandbox can't reach the pod. Run on your Mac.

### Step 1 — Rsync partial outputs from pod

```bash
mkdir -p /Users/john/code/gradience/papers/benchmark_reliability_study/runs_dryrun_2026_04_26/raw
rsync -avh --progress \
  -e "ssh -p 10024 -i ~/.ssh/id_ed25519" \
  root@213.173.109.14:/workspace/study/runs/raw/ \
  /Users/john/code/gradience/papers/benchmark_reliability_study/runs_dryrun_2026_04_26/raw/
```

Approx size estimate: 266 conditions × ~50 KB raw item-output JSONL ≈ 13 MB. Should complete in seconds-to-minutes depending on link.

### Step 2 — Build a partial manifest reflecting actually-completed conditions

```bash
cd /Users/john/code/gradience/papers/benchmark_reliability_study
RUNS=runs_dryrun_2026_04_26
mkdir -p $RUNS/manifests

# Header from the master manifest:
head -1 manifests/conditions_primary.csv > $RUNS/manifests/conditions_primary.csv

# Keep rows whose condition_id has a matching directory in the rsync'd raw/:
awk -F, 'NR>1 {cmd="test -d '$RUNS'/raw/" $1 " && echo " $0; system(cmd)}' \
  manifests/conditions_primary.csv | \
  sed 's/,pending,/,complete,/' >> $RUNS/manifests/conditions_primary.csv

# Same for GSM8K (only pythia_1_4b's 24 GSM8K conditions; per the post-cut state):
head -1 manifests/conditions_gsm8k.csv > $RUNS/manifests/conditions_gsm8k.csv
awk -F, 'NR>1 {cmd="test -d '$RUNS'/raw/" $1 " && echo " $0; system(cmd)}' \
  manifests/conditions_gsm8k.csv | \
  sed 's/,pending,/,complete,/' >> $RUNS/manifests/conditions_gsm8k.csv

# Verify counts:
wc -l $RUNS/manifests/conditions_primary.csv $RUNS/manifests/conditions_gsm8k.csv
# Expected: ~242 + ~24 (header lines) = ~266 + 2 header rows.
```

### Step 3 — Run scripts 04 → 10 + 98 + 99

```bash
cd /Users/john/code/gradience/papers/benchmark_reliability_study
RUNS=runs_dryrun_2026_04_26
mkdir -p $RUNS/{normalized,analysis,reports}

# Script 04 — normalize raw outputs
python3 scripts/04_normalize_outputs.py \
  --conditions $RUNS/manifests/conditions_primary.csv \
  --raw-dir $RUNS/raw \
  --schemas-dir schemas \
  --out $RUNS/normalized/item_level_primary.parquet \
  --config configs/study_config.yaml \
  2>&1 | tee $RUNS/04_log.txt

# Script 05 — condition-level scores
python3 scripts/05_make_condition_scores.py \
  --item-level $RUNS/normalized/item_level_primary.parquet \
  --out $RUNS/normalized/condition_level_primary.csv \
  2>&1 | tee $RUNS/05_log.txt

# Script 06 — variance components (requires statsmodels). NOTE: --config
# expects configs/study_config.yaml (the merged-config entry that includes
# models, benchmarks, prompts, scoring_rules, analysis_config), NOT
# configs/analysis_config.yaml directly.
python3 scripts/06_variance_components.py \
  --item-level $RUNS/normalized/item_level_primary.parquet \
  --condition-level $RUNS/normalized/condition_level_primary.csv \
  --config configs/study_config.yaml \
  --out-dir $RUNS/analysis/variance_components/ \
  2>&1 | tee $RUNS/06_log.txt

# Script 07 — tolerance schedule (regime-aware per v1.1.2)
python3 scripts/07_tolerance_schedule.py \
  --condition-level $RUNS/normalized/condition_level_primary.csv \
  --variance-components $RUNS/analysis/variance_components/ \
  --analysis-config configs/analysis_config.yaml \
  --out-dir $RUNS/analysis/tolerance_schedules/ \
  2>&1 | tee $RUNS/07_log.txt

# Script 08 — ranking stability
python3 scripts/08_ranking_stability.py \
  --condition-level $RUNS/normalized/condition_level_primary.csv \
  --out-dir $RUNS/analysis/ranking_stability/ \
  2>&1 | tee $RUNS/08_log.txt

# Script 09 — MMLU subject decomposition
python3 scripts/09_mmlu_subject_decomp.py \
  --item-level $RUNS/normalized/item_level_primary.parquet \
  --condition-level $RUNS/normalized/condition_level_primary.csv \
  --out-dir $RUNS/analysis/mmlu_subjects/ \
  2>&1 | tee $RUNS/09_log.txt

# Script 10 — GSM8K case
python3 scripts/04_normalize_outputs.py \
  --conditions $RUNS/manifests/conditions_gsm8k.csv \
  --raw-dir $RUNS/raw \
  --schemas-dir schemas \
  --out $RUNS/normalized/item_level_gsm8k.parquet \
  --config configs/study_config.yaml
python3 scripts/05_make_condition_scores.py \
  --item-level $RUNS/normalized/item_level_gsm8k.parquet \
  --out $RUNS/normalized/condition_level_gsm8k.csv
python3 scripts/10_gsm8k_case.py \
  --item-level $RUNS/normalized/item_level_gsm8k.parquet \
  --condition-level $RUNS/normalized/condition_level_gsm8k.csv \
  --analysis-config configs/analysis_config.yaml \
  --out-dir $RUNS/analysis/gsm8k_case/ \
  2>&1 | tee $RUNS/10_log.txt

# Script 98 — reproducibility trace
python3 scripts/98_reproducibility_trace.py \
  --config configs/study_config.yaml \
  --manifests-dir $RUNS/manifests \
  --raw-dir $RUNS/raw \
  --out $RUNS/reports/reproducibility_trace_dryrun.md \
  2>&1 | tee $RUNS/98_log.txt

# Script 99 — final report
python3 scripts/99_make_report.py \
  --analysis-dir $RUNS/analysis \
  --tables-dir $RUNS/analysis \
  --figures-dir $RUNS/analysis \
  --out $RUNS/reports/cpu_pipeline_report_dryrun.md \
  --config configs/study_config.yaml \
  2>&1 | tee $RUNS/99_log.txt
```

### Step 4 — Inspect outputs for the seven risk areas

```bash
RUNS=runs_dryrun_2026_04_26
# Risk A: empty-cell handling
grep -i "empty\|no data\|skipped\|qwen" $RUNS/0[6-9]_log.txt $RUNS/10_log.txt

# Risk B: incomplete-cell labeling
head -3 $RUNS/analysis/tolerance_schedules/tolerance_by_cell.csv
# Look for n_conditions or similar count column; if missing, file a fix.

# Risk D + F: regime classification
awk -F, 'NR>1 {print $NF}' $RUNS/analysis/tolerance_schedules/tolerance_by_cell.csv | sort | uniq -c
# Should see g_theory and parse_failure_dominated counts both nonzero.

# Risk E: bootstrap warnings
grep -i "bootstrap\|nan\|insufficient" $RUNS/0[6-9]_log.txt $RUNS/10_log.txt

# Risk G: manifest status conversion
grep -c ",complete," $RUNS/manifests/conditions_primary.csv
# Should be ~242 (266 minus 24 GSM8K).

# Final integrity:
ls -la $RUNS/analysis/*/ $RUNS/reports/
```

---

## Acceptance criteria

- [ ] All eight pipeline scripts complete with exit 0 (or document specific cell-level failures and confirm they're partial-data artifacts, not script bugs).
- [ ] Logs are clean of unhandled exceptions and silent crashes.
- [ ] `tolerance_by_cell.csv` has both regime values represented (some `g_theory`, some `parse_failure_dominated`) — confirms the regime split machinery is alive on real data.
- [ ] Empty-cell paths produce informational messages, not errors.
- [ ] Bootstrap CIs computed for at least one cell where the partial design has ≥ 12 conditions; verify CIs are reasonable (non-degenerate) on that cell.
- [ ] `99_make_report.py` produces a report file even though some analyses have empty inputs — the report should explicitly note "partial corpus, not for interpretation" status.

If any of the seven risks fires as a real bug rather than expected partial-data behavior, file a fix patch in `scripts/` before Phase 5 launches in earnest. Patches should land before GPU run completes (~35–40 hours from 2026-04-26 22:55 UTC) so that real Phase 5 starts on a verified pipeline.

---

## What this dry-run does *not* test

- Numerical correctness of variance-components estimates on real-distribution data (the test fixtures were synthetic; the partial corpus is real but incomplete).
- Final tolerance-schedule values (depend on full data).
- Ranking-stability outcomes (sparse model coverage).
- The figure-generation portion of the pipeline (figures are scripted but not bundled with the supplementary per the trim discipline; figure scripts live elsewhere).

These will be exercised at real Phase 5 launch. The dry-run's purpose is to ensure that launch starts cleanly, not to anticipate its results.

---

## After the dry-run

Whether it passes clean or surfaces patches:

1. Add a row to `RESEARCH_INVENTORY.md` Section 9 (Drafting and submission milestones) recording the dry-run date, outcome, and any patches landed.
2. If patches landed, append to `IMPLEMENTATION_DEVIATIONS.md` (D-19+) with the script name and the patch description.
3. If the dry-run's `99_make_report.py` produces output that's interesting beyond the test of the pipeline itself, archive it as `runs_dryrun_2026_04_26/REPORT_archive.md` for comparison against real Phase 5 output.

The dry-run output is throwaway. The dry-run's *fixes*, if any, are what matters.
