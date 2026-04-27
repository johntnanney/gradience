# Phase 5 Dry-Run Report — 2026-04-26

**Status:** NOT pipeline-ready for Phase 5. Two blocking issues surfaced; one critical (pod-vs-local-repo drift), one severe (script 06 wall-clock at real scale).

**Disposition of all results in this report:** _partial corpus, not for interpretation._
The dry-run consumed 248 primary + 24 GSM8K conditions out of the planned 600 + 24
(post-Cut 2 design), captured during a mid-flight GPU run. Numerical outputs from
07–10 / 98 / 99 below were derived from a synthetic `aggregate_vc.csv` placeholder
because script 06 did not complete. The numbers exist solely to confirm that the
pipeline plumbing routes data to the expected output shapes; they have no
inferential meaning.

---

## 1. Per-script exit codes and log line counts

| Script | Exit | Log lines | Notes |
|---|---|---|---|
| `04_normalize_outputs.py` (primary) | **3** | 99 | 96 of 248 conditions failed `PartialRunCompletion` (MMLU panel: `num_items_completed` reports actual subject size 100–282 vs `expected_num_items=100` in manifest). 152 conditions normalized to `item_level_primary.parquet` (524,396 item rows). |
| `05_make_condition_scores.py` (primary) | 0 | 4 | Wrote 152 condition rows. **FutureWarning** at line 119 (`.fillna(False)` object-dtype downcasting) — cosmetic, as expected. |
| `06_variance_components.py` | **143 (SIGTERM)** | 1 | **Killed after 15:07 wall-clock** with no per-benchmark progress logs. Process was at 1100 % CPU on 101 BLAS threads, fitting `level_1` MixedLM on `arc_challenge` (63,288 items × 4 crossed random effects, including `item_id` with ~1 172 levels). Downstream pipeline exercised with synthetic placeholder `aggregate_vc.csv`. |
| `07_tolerance_schedule.py` | 0 | 3 | 16 cells, 5 benchmarks, H1 confirmed. **Output schema is missing the v1.1.2 `regime` column** (see Risk D + F). |
| `08_ranking_stability.py` | 0 | 6 | 3 of 5 benchmarks gracefully skipped (mmlu_panel, truthfulqa_mc, winogrande — fewer than 2 models with data on the partial corpus); empty-cell handling is informational, not error. |
| `09_mmlu_subject_decomp.py` | 0 | 5 | Documented graceful fallback (LinAlg `Singular matrix` → ANOVA fallback) on the very thin MMLU corpus (1 model × 1 subject after partial-corpus filtering). |
| `04_normalize_outputs.py` (GSM8K) | 0 | 2 | All 24 GSM8K conditions normalized clean (31,656 item rows). |
| `05_make_condition_scores.py` (GSM8K) | 0 | 2 | 24 condition rows. |
| `10_gsm8k_case.py` | 0 | 4 | Wrote tolerance, extraction-sensitivity, parseability outputs. |
| `98_reproducibility_trace.py` | **5** | 1 | 3 / 16 reproducibility-critical artifacts failed: (a) variance-components re-derivation hit the script's own 120 s `subprocess.run` timeout — same root cause as 06 above; (b) tolerance-schedule re-derivation differs (expected — synthetic VC input); (c) one MMLU recompute-sample condition skipped (no stored accuracy because 04 dropped it). 3 of 5 sampled conditions re-computed bit-exact. |
| `99_make_report.py` | 0 | 1 | Report written to `reports/cpu_pipeline_report_dryrun.md`. |

**Eight tee'd log files** in `logs/`: `04_log.txt`, `05_log.txt`, `06_log.txt`,
`07_log.txt`, `08_log.txt`, `09_log.txt`, `10_log.txt`, `98_log.txt`, `99_log.txt`,
plus `04_gsm8k_log.txt`, `05_gsm8k_log.txt`, `exit_codes.txt`.

---

## 2. Risk-area inspection results

### Risk A — Empty cells (qwen, missing benchmarks)

**PASS.** Empty-cell paths produced informational warnings, not errors. Script 08
emitted three `WARNING` lines (`Benchmark X: fewer than 2 models with data;
skipping.`) and continued. Per-cell coverage on the partial corpus:

|                     | arc_challenge | hellaswag | mmlu_panel | truthfulqa_mc | winogrande |
|---|---|---|---|---|---|
| pythia_1_4b         | 24 | 24 | 24* | 8 | 24 |
| pythia_410m         | 24 | 18 | 0 | 0 | 0 |
| qwen2_5_1_5b_instr  | 6  | 0  | 0 | 0 | 0 |

*MMLU panel for pythia_1_4b: 24 conditions in the manifest, but only 2 subjects
(`international_law`, `miscellaneous`) survived script 04 normalization due to
the partial-completion mismatch (Risk B / blocker patch P-1 below).
_(Counts above are partial-corpus, not final design coverage.)_

### Risk B — Mid-completion / partial-cell labeling

**PASS on schema, FAIL on documentation accuracy.** `tolerance_by_cell.csv` does
include an `n_conditions` column (varies 6–24 across cells), satisfying the
"document partial-cell sample size" requirement. The workplan's grep for a
`regime` column in `tolerance_by_cell.csv` returned the column header itself —
because the column does not exist (see Risk D + F). _No patch needed for Risk B
itself._

### Risk C — Manifest condition_id format mismatch

**PASS.** All 248 + 24 condition_ids built with the four-underscore empty-subject
slot match on-disk directory names. `04_normalize_outputs.py` consumed all 272
without any "directory not found" errors.

### Risk D — Cells with only one scoring rule available

**FAIL — workplan claim is false.** The workplan states "Script 07 handles this
at lines 110–114 (defaults to `g_theory` regime when parseability data is empty)."
A grep for `regime`, `parse_failure_dominated`, `g_theory`, and
`parse_failure_threshold` in the pod's `/workspace/study/scripts/07_tolerance_schedule.py`
returned **zero matches**. The pod's script has no regime classification at all.

The local repo's `papers/benchmark_reliability_study/scripts/07_tolerance_schedule.py`
does have the regime logic (function `_determine_regime` at line 94, branches at
lines 304–317, output column at lines 383 / 407). **The pod is running stale
v1.1 scripts; the v1.1.2 amendment is committed locally but not deployed.** This
is the primary blocker — see patch P-1 below.

### Risk E — Bootstrap with sparse data

**PASS.** No NaN / insufficient-data warnings in any of the 0[6-9] / 10 logs.
Bootstrap CIs are computed for cells down to n = 6 (qwen / arc) — wide CIs, e.g.
`tolerance_single_ci_lower=0.264, ci_upper=0.668` for that cell, but finite and
non-degenerate. The `tolerance_by_cell.csv` has 16 cells, all with bootstrap
intervals populated.

### Risk F — `analysis_config.yaml` resolution / config_hash check

**FAIL.** The pod's `configs/analysis_config.yaml` has no `tolerance.parse_failure_threshold`
entry; its top-line comment says "v1.1-LOCKED" (not v1.1.2). The pod's
config_hash (short, full-config) is `89ce3f1f`, **not** the workplan's expected
`fbc4a5dd`. The local repo's commit `480c213` ("amend lock to v1.1.2") modified
four study files (`configs/analysis_config.yaml`, `gradience_study/config.py`,
`scripts/07_tolerance_schedule.py`, `preregistration/prereg_v1_1_LOCKED.md`); none
of them have made it to the pod. Same root cause as Risk D — patch P-1.

### Risk G — Manifest `condition_status` conversion

**PASS.** Both filtered manifests carry `condition_status: complete` for every
data-bearing row (248 primary, 24 GSM8K), exactly matching the on-disk directory
count. Script 04 read the status field and processed all rows without
"pending-but-no-data" complaints.

---

## 3. Patches landed during the dry-run

**No patches landed in this dry-run.** Findings recorded only — patches must be
landed before Phase 5.

The dry-run produced one helper script (`runs_dryrun_2026_04_26/synth_vc.py`,
~50 LOC) used to fabricate a placeholder `aggregate_vc.csv` after script 06 was
killed. It is not a script-tree patch; it is a dry-run-only artifact and lives in
the run directory, not in `scripts/`.

---

## 4. Other observations

### Workplan invocation errors found while dry-running

These are workplan documentation errors, not script bugs. The dry-run's
invocations corrected each one ad-hoc.

- **08:** the workplan invocation omits `--analysis-config`. Script 08 requires
  it (`argparse` error otherwise). Corrected to
  `--analysis-config configs/study_config.yaml`.
- **09:** the workplan invocation passes `--condition-level`. Script 09 does not
  accept that argument (`--item-level` and `--analysis-config` only).
  Corrected.
- **07:** the workplan invocation passes `--variance-components <DIR>/`. Script
  07 expects a CSV file path (`<DIR>/aggregate_vc.csv`). Corrected.
- **98:** the workplan invocation omits `--normalized-dir` (required). Corrected.

The workplan should be patched in line with these for the real Phase 5 runbook;
none of these are pipeline bugs.

### Live GPU process (PID 3823) status

GPU process remained healthy and undisturbed throughout the dry-run. Snapshot
times: started dry-run with 272 raw dirs, ended with 274 (2 new conditions
completed during dry-run wall-clock). PID 3823 etime ~ 40 min, %CPU 82, %MEM
1.1 % at end of dry-run. Read-only access to `runs/raw/` worked via symlink
from `/workspace/study_dryrun/raw → /workspace/study/runs/raw`.

### Notable healthy paths

- Script 04 GSM8K: 24 / 24 conditions normalized clean — no partial-completion
  issue on GSM8K because `gpu_inference.py` and the GSM8K manifest agree on
  expected item counts (~1 319 per condition).
- Script 04 primary: the 152 normalized non-MMLU conditions parsed without
  schema-validation errors. The 96 failures cluster entirely in `mmlu_panel`.
- Script 99: produced a complete report file even though 06 / 98 / 09 inputs
  were partly synthetic or failure-state. Section 1 ("Pipeline execution
  summary") and Section 2 ("Headline numbers") rendered.

---

## 5. Blocking patches required before Phase 5

### P-1 (CRITICAL, must-land) — Sync v1.1.2 amendment to the pod

**Symptom.** Risks D + F both fire. Pod scripts are at v1.1; local repo is at
v1.1.2 (commit `480c213`). The pod's `07_tolerance_schedule.py` has no regime
classification; its `analysis_config.yaml` has no `parse_failure_threshold`;
its config_hash is `89ce3f1f`, not the workplan's expected `fbc4a5dd`.

**Fix.** Push commit `480c213` (or its descendant) to the pod's `/workspace/study/`
working tree. The four files to update:

- `configs/analysis_config.yaml` (adds `tolerance.parse_failure_threshold: 0.30`)
- `gradience_study/config.py` (adds the field to `ToleranceConfig`)
- `scripts/07_tolerance_schedule.py` (adds `_determine_regime` and the
  `regime` column on `tolerance_by_cell.csv`)
- `preregistration/prereg_v1_1_LOCKED.md` (documentation only)

**Verification after sync.** Re-run script 07 on the dry-run inputs; the resulting
`tolerance_by_cell.csv` should have a `regime` column with mixed values
(`g_theory` / `parse_failure_dominated`); `98_reproducibility_trace.py` should
emit `config_hash` = `fbc4a5dd` (or whatever value the v1.1.2 hash actually is —
the workplan claims `fbc4a5dd`, which should be re-verified post-sync).

**GPU-process compatibility note.** The v1.1.2 amendment did **not** touch
`gpu_inference.py` or any input the GPU process is consuming. The live PID 3823
run is safe to continue under v1.1; only the analysis pipeline (07+) needs
v1.1.2.

### P-2 (SEVERE) — `06_variance_components.py` wall-clock at real scale

**Symptom.** On the partial corpus (524 396 items, 5 benchmarks), the level_1
MixedLM fit stalled with no per-benchmark progress logs after 15 min wall-clock
(1 100 % CPU on 101 threads, no I/O). Script 98's own re-derivation step
hit its 120 s subprocess timeout against the same code path. At full Phase 5
scale (~1 600 conditions × 3 models, with the largest benchmark `hellaswag`
~10 000 unique items), this fit is likely to take many hours per benchmark or
fail entirely.

**Why it stalls.** `_fit_item_level_lpm` constructs a crossed-random-effects
LPM via `statsmodels.MixedLM` with `vc_formula = {fac: f"0 + C({fac})" for fac
in random_effects}` over `prompt_id`, `seed_id`, `scoring_rule_id`, **`item_id`**.
The `item_id` factor expands to ~1 172 dummy columns for `arc_challenge` (and
~10 000 for `hellaswag`). statsmodels MixedLM with that many random-effect
levels on hundreds of thousands of rows is known to be slow / fragile.

**Suggested fix paths (not landed; pick before Phase 5).**

1. **Add periodic progress logging inside the cascade fit** so 98's timeout +
   any future operator wall-clock budget have something to look at. Even an
   `[INFO] level_1: fit started ({n_items} items, {n_re} RE)` before
   `model.fit()` would have made the dry-run informative.
2. **Drop `item_id` from level_1's RE list** if the spec permits — a Generalized
   theory of generalizability cell typically treats item as fixed, not random,
   when items are common across persons. The current cascade puts item in
   level_1, level_2, level_3, but never level_4. Whether to demote item to
   fixed effects is a methodological call; document the change in a deviation
   note.
3. **Alternatively** raise 98's `subprocess.run(..., timeout=120)` to something
   commensurate with the level_1 fit budget (e.g. `timeout=3600`), and accept
   that 98 will be a long-running step.
4. **As a last resort** flag this benchmark-by-benchmark as falling to level_4
   (aggregate G-theory only); the cascade already supports that fallback at
   exit code 4.

**Decision required.** A non-numeric one — option 2 is the right call if the
spec already treats item as a fixed effect for tolerance-schedule purposes;
option 1 + 3 is a no-methodology-change compromise. Out of scope for the
dry-run; recommend a 30-min decision meeting before Phase 5 launch.

---

## 6. Pipeline plumbing assessment

| Script | Plumbing OK? | Notes |
|---|---|---|
| 04 | ✓ | Runs to completion; partial-completion errors are informative, not crashes. |
| 05 | ✓ | Runs clean (modulo cosmetic FutureWarning). |
| 06 | ✗ | Does not complete on real-scale data. **Patch P-2.** |
| 07 | ✗ | Pod version is stale (v1.1, missing regime logic). **Patch P-1.** |
| 08 | ✓ | Empty-cell handling correct; runs clean. |
| 09 | ✓ | Singular-matrix fallback works. |
| 10 | ✓ | GSM8K case-study pipeline runs clean. |
| 98 | ✓ (degrades) | Runs to completion; reports failure status appropriately. **Patch P-2 indirectly affects this** (06 timeout). |
| 99 | ✓ | Produces report even with partial / failed inputs. |

---

## 7. Acceptance-criteria checklist (workplan §"Acceptance criteria")

- [ ] **All eight pipeline scripts complete with exit 0 (or document specific cell-level failures).** _Not met._ Script 06 was killed after 15 min; script 04 exited 3 due to MMLU manifest mismatch; script 98 exited 5 (failure trace). All three have documented root causes (P-1 manifest issue for 04; P-2 fit-time issue for 06; both for 98).
- [ ] **Logs clean of unhandled exceptions and silent crashes.** _Met._ All non-zero exits surfaced through structured `[ERROR]` log lines; no `Traceback` / `CRITICAL` lines in any log.
- [ ] **`tolerance_by_cell.csv` has both `g_theory` and `parse_failure_dominated`.** _Not met._ The `regime` column is missing entirely (Risk D + F + P-1).
- [x] **Empty-cell paths produce informational messages, not errors.** _Met._ Risk A above.
- [x] **Bootstrap CIs computed for at least one cell with ≥ 12 conditions, non-degenerate.** _Met._ All 12 cells with `n_conditions ≥ 18` have finite, non-zero CIs.
- [x] **`99_make_report.py` produces a report file.** _Met._ `cpu_pipeline_report_dryrun.md` written, 7.3 KB.

---

## 8. Recommendation

**NOT pipeline-ready for Phase 5 launch.**

Two patches must land before Phase 5:

1. **P-1 (must, before any Phase 5 step that consumes 07 output).** Sync the
   v1.1.2 amendment from the local repo to the pod. This is mechanical (one
   `git pull` or `scp` of four files) and unblocks the regime classification
   that the SPEC + IMPLEMENTATION_DEVIATIONS D-09 v1.1.2 mandate.
2. **P-2 (should, before script 06 is run on the full corpus).** Decide and
   land one of the four 06 wall-clock options above. Without this, Phase 5
   either takes hours waiting on a single MixedLM fit, or fails opaquely.

Additionally:

- The 96 MMLU-panel `PartialRunCompletion` errors in script 04 reflect a
  manifest-vs-runner disagreement: the manifest's `expected_num_items=100`
  for MMLU subjects, but `gpu_inference.py` is generating the full subject
  size (171–282 items per subject). Either the manifest must be updated to
  the per-subject true sizes, or `gpu_inference.py` must truncate. This is
  arguably a third blocker, but it is a **GPU-side** issue and is outside the
  Phase 5 analysis pipeline. The MMLU rows simply drop out of normalization
  for now; whatever decision is made here, it should land before MMLU is
  re-run.

The workplan deviations in §4 above (08, 09, 07, 98 invocation errors) should
be folded back into the workplan / runbook; they are not script bugs but they
will trip the next operator if uncorrected.

---

## 9. Files in this run directory

```
runs_dryrun_2026_04_26/
├── DRY_RUN_REPORT.md            (this file)
├── synth_vc.py                  helper: synthesize placeholder aggregate_vc.csv
├── logs/
│   ├── 04_log.txt               99 lines (96 PartialRunCompletion errors)
│   ├── 04_gsm8k_log.txt         2 lines (24/24 OK)
│   ├── 05_log.txt               4 lines
│   ├── 05_gsm8k_log.txt         2 lines
│   ├── 06_log.txt               1 line (killed at 15:07)
│   ├── 07_log.txt               3 lines
│   ├── 08_log.txt               6 lines (3 empty-cell warnings)
│   ├── 09_log.txt               5 lines (singular-matrix fallback)
│   ├── 10_log.txt               4 lines
│   ├── 98_log.txt               1 line (FAIL summary)
│   ├── 99_log.txt               1 line
│   └── exit_codes.txt           summary
├── reports/
│   ├── cpu_pipeline_report_dryrun.md
│   └── reproducibility_trace_dryrun.md
└── analysis_outputs/analysis/
    ├── variance_components/    aggregate_vc.csv (SYNTHETIC, 32 rows / 8 cells)
    │                            item_level_vc.csv (stub), model_convergence_report.csv (stub)
    ├── tolerance_schedules/    tolerance_by_cell.csv (16 rows, no regime col)
    │                            tolerance_by_benchmark_summary.csv, h1_test.json
    ├── ranking_stability/      kendall_tau_by_benchmark.csv (2 rows; 3 benchmarks skipped)
    │                            ranking_reversals.csv (0 rows), pairwise_win_probabilities.csv
    ├── mmlu_subjects/          mmlu_subject_accuracy_matrix.csv (1×1), variance_components, h4_test.json
    └── gsm8k_case/             gsm8k_tolerance_schedule.csv (2 rows), extraction_sensitivity, parseability
```

The pod's working tree at `/workspace/study_dryrun/` contains the same files
plus the original (large) parquet outputs; it can be deleted at the operator's
convenience after this report is reviewed.

---

## 10. Suggested follow-ups

1. Add a row to `RESEARCH_INVENTORY.md` §9 with this dry-run's date and
   outcome (NOT-READY, two blockers).
2. Append `IMPLEMENTATION_DEVIATIONS.md` D-19 documenting the v1.1.2 pod-sync
   miss and the P-2 wall-clock issue.
3. Apply P-1 (mechanical). Re-run scripts 07 / 98 on the same dry-run inputs to
   verify the `regime` column and config_hash; archive the deltas as
   `runs_dryrun_2026_04_26/POST_P1_DELTA.md`.
4. Schedule the P-2 decision meeting (≤ 30 min) before GPU completion.
5. Once both patches land, re-trigger this dry-run on whatever GPU output is
   available at that moment as a confidence check. The synthesis step
   (`synth_vc.py`) is not needed once 06 completes within budget.
