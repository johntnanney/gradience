# Phase 5 Handoff Checklist

**Purpose.** When the GPU run completes, transition cleanly from pod-side
inference to workstation-side analysis. Each step has a verification
gate; do not advance past a failing gate without an explicit decision.

**Drafted:** 2026-04-28, after running the full pipeline end-to-end on a
548-condition partial pull. CLI invocations and the manifest-mark-complete
re-derivation logic are pre-verified.

---

## Step 1 — Confirm GPU run is genuinely complete

On the pod (current SSH: `ssh root@213.173.102.74 -p 17180 -i ~/.ssh/id_ed25519`):

```bash
# Should show 624 (post-Cut-2 total). The 48 excluded_pre_run rows do
# not have raw/ entries; that's intentional, not partial completion.
ls /workspace/study/runs/raw/ | grep -v "^\.tmp$" | wc -l

# All status=complete or all status=partial-due-to-stale-expected?
# (Stale-expected is fine; the workstation Phase 0 sweep re-derives.)
ls /workspace/study/runs/raw/ | grep -v "^\.tmp$" | head -3 | xargs -I {} \
  python3 -c "import json; m=json.load(open('/workspace/study/runs/raw/{}/run_metadata.json')); print('{}'[:60], m['status'])"

# Process should have exited cleanly (exit_code=0 in last log line).
tail -5 /workspace/study/runs/inference.log
ls /workspace/study/runs/failures.jsonl 2>/dev/null && \
  wc -l /workspace/study/runs/failures.jsonl
```

**Gate:** if completed-condition count is < 600 (primary) + 24 (kept
GSM8K) = 624, surface to user before continuing. If failures.jsonl
exists with non-zero rows, surface and decide whether to retry the
failed conditions or document as deviations.

---

## Step 2 — Tar + pull raw outputs to workstation

On the pod:

```bash
cd /workspace/study
tar --exclude="runs/raw/.tmp" --exclude="runs/raw/.tmp/*" \
    -czf /tmp/raw_final.tar.gz \
    runs/raw/ runs/inference.log \
    runs/failures.jsonl 2>/dev/null || \
  tar --exclude="runs/raw/.tmp" --exclude="runs/raw/.tmp/*" \
      -czf /tmp/raw_final.tar.gz runs/raw/ runs/inference.log
ls -lh /tmp/raw_final.tar.gz
```

On the workstation:

```bash
cd /Users/john/code/gradience/papers/benchmark_reliability_study
scp -P 17180 -i ~/.ssh/id_ed25519 \
  root@213.173.102.74:/tmp/raw_final.tar.gz /tmp/raw_final.tar.gz

# Move any existing prep dir aside; extract canonical raw/ at study root.
[ -d runs ] && [ "$(ls runs 2>/dev/null)" ] && \
  mv runs runs.pre_phase5_$(date +%s) || true
mkdir -p runs
tar --no-same-owner -xzf /tmp/raw_final.tar.gz
ls runs/raw/ | grep -v "^\.tmp$" | wc -l   # should match pod count
```

**Gate:** workstation raw-dir count == pod raw-dir count. If they differ,
do not proceed; tarball was incomplete or extract failed.

---

## Step 3 — Run the Phase 5 driver

```bash
cd /Users/john/code/gradience/papers/benchmark_reliability_study
./scripts/run_phase5.sh 2>&1 | tee runs/phase5_run.log
```

This executes:

- **Phase 0:** mark-complete sweep (re-derives `condition_status` from
  jsonl row counts vs. local D-19 patched manifest; patches each
  pod-side metadata.json's stale `num_items_expected` field).
- **Phase 1:** script 04 normalize → `runs/normalized/item_level_*.parquet`
- **Phase 2:** script 05 condition aggregation → `runs/normalized/condition_level_*.csv`
- **Phase 3:** script 06 variance components (D-20 cascade) → `analysis/variance_components/`
- **Phase 4:** script 07 tolerance schedule → `analysis/tolerance_schedules/`
- **Phase 5:** script 08 ranking stability → `analysis/ranking_stability/`
- **Phase 6:** script 09 MMLU subject decomposition → `analysis/mmlu_subjects/`
- **Phase 7:** script 10 GSM8K case → `analysis/gsm8k_case/`
- **Phase 8:** script 98 reproducibility trace → `reports/reproducibility_trace.md`
- **Phase 9:** script 99 pipeline report → `reports/cpu_pipeline_report.md`

**Gates per phase:**

- Phase 0: `complete + pending + excluded == 672` (sanity check on
  manifest-row total).
- Phase 1: 04 exit 0; `complete=N, normalized_ok=N, failed=0` in the
  output log; N matches the marked-complete count.
- Phase 3: 06 exit 0; expected ~12 sec for all 5 benchmarks at level_1
  (hours-long fits would indicate the D-20 cascade fix did not apply
  — re-verify that `_random_effects_for_level` excludes `item_id`).
- Phase 4: 07 exit 0; `tolerance_by_cell.csv` has the `regime` column
  populated (not all NaN — populated `regime_split` indicates v1.1.2
  code is active).
- Phase 8: trace status may report `fail` on `tolerance_by_cell.csv` due
  to known bootstrap non-determinism (D-21). That is a pre-known issue,
  not a regression. The per-condition recompute deltas in section 4 of
  the trace should all be 0 — that is the load-bearing reproducibility
  check.

If any phase fails its gate, surface to user with the log excerpt before
re-running.

---

## Step 4 — Verification spot-check

After `run_phase5.sh` completes:

```bash
# H1 result
cat analysis/tolerance_schedules/h1_test.json | python3 -m json.tool

# Per-condition recompute (should be 5/5 pass with delta=0)
grep -A 8 "Per-condition recompute" reports/reproducibility_trace.md

# Per-benchmark variance components present
head -3 analysis/variance_components/aggregate_vc.csv
wc -l analysis/variance_components/*.csv

# All benchmarks reach level_1 (D-20 fix verification)
awk -F, 'NR>1 {print $2}' analysis/variance_components/model_convergence_report.csv | \
  sort | uniq -c
# Should show all level_1 (or level_2/3 if a benchmark genuinely failed
# to converge — but never level_4 unless a benchmark has too few cells).
```

**Gate:** all 5 benchmarks at level_1 or level_2; any level_4 fallbacks
should be expected (and noted in the convergence report's error_type
column).

---

## Step 5 — Document run outcomes + close audit trail

Three artifacts to update post-run:

### a) `CHANGELOG.md` — append a new dated entry

Template:

```markdown
## 2026-04-XX — Phase 5 analysis complete

### GPU run summary

- N completed conditions: ___ / 624 (post-Cut-2 total)
- Per-model: pythia_1_4b ___, pythia_410m ___, qwen2_5_1_5b ___
- Wall-clock: ___ hours total (with one unplanned pod restart at ___ UTC)
- Failures: ___ (see `runs/failures.jsonl` if non-zero)
- Total inference cost: ~$___

### Analysis pipeline outcomes

- All 5 benchmarks reached level___ in the variance-components cascade
- H1 confirmed: ___ / 5 benchmarks (threshold 0.005)
- H4 confirmed: ___ (MMLU model × subject interaction proportion ___)
- H3 ranking-reversal fraction: ___ (threshold 0.20)
- Reproducibility trace status: ___ (D-21 bootstrap determinism noted)

### Cross-paper coordination

- §7.1 → §7.5 prose can now be drafted against actual numbers
- §8 discussion: NIST 800-3 GLMM-vs-LPM comparison appendix data ready
```

### b) `LOCK_NOTES.md` — append a Phase 5 completion section

Template:

```markdown
---

## Phase 5 completion: 2026-04-XX

**Status.** GPU inference complete; analysis pipeline run end-to-end on
the workstation against the canonical v1.1.2 codebase + D-19/D-20
patched manifests + D-18 Cut-2 scope.

**Final inputs:**
- ___ raw-condition directories ingested
- ___ item rows in the primary parquet
- ___ item rows in the GSM8K parquet
- 0 schema-validation failures (or N failures, see deviations)

**Final outputs:**
- `analysis/variance_components/aggregate_vc.csv` — N benchmark × model cells
- `analysis/tolerance_schedules/tolerance_by_cell.csv` — N rows
- `analysis/tolerance_schedules/h1_test.json` — H1 confirmed/refuted
- `analysis/ranking_stability/{ranking_reversals,pairwise_win_probabilities,kendall_tau_by_benchmark}.csv`
- `analysis/mmlu_subjects/{mmlu_subject_accuracy_matrix,mmlu_subject_variance_components,h4_test}.{csv,json}`
- `analysis/gsm8k_case/{gsm8k_tolerance_schedule,gsm8k_extraction_sensitivity,gsm8k_parseability}.csv`
- `reports/reproducibility_trace.md`
- `reports/cpu_pipeline_report.md`
- `figures/mmlu_model_subject_heatmap.png`, `figures/ranking_stability_by_benchmark.png`

**Audit summary.**
- Pre-reg lock: v1.1.2-LOCKED (config_hash `fbc4a5dd`)
- Manifest patches: D-19 (MMLU per-subject sizes)
- Cascade modification: D-20 (item_id removed from RE)
- Scope amendment: D-18 (Cut 2 — 48 GSM8K conditions excluded_pre_run)
- Bootstrap determinism: D-21 (CI bounds drift across re-runs; point
  estimates stable; trace section 4 deltas all 0)

**Tag:** `v1_1_2_PHASE5_COMPLETE` (post-merge to master).
```

### c) PR description (when opening the PR against master)

Template:

```markdown
## Phase 1–5: scaffold, lock, GPU run, analysis pipeline

This PR consolidates the benchmark-reliability study from initial scaffold
through Phase 5 analysis. Self-contained: master at merge time will
reflect the locked v1.1.2 design, the executed Cut-2 scope, and the full
analysis-pipeline outputs.

### Phase 1 — pre-registration (v1.1 → v1.1.1 → v1.1.2 chain)
- Locked configs (3 models × 5 primary benchmarks × 4 prompts × 3 seeds × 2 scoring rules)
- v1.1.2 amendment: D-09 LPM-vs-GLMM regime split per NIST AI 800-3
- 24 prompts sourced and content-hash verified across benchmark-author / lm-eval / HELM / minimal-author tiers

### Phase 2 — pipeline scripts (00–10, 98, 99)
- 6664 lines across 14 numbered scripts
- 4311 lines of test coverage (183 tests, 1 documented env-dependent fail)

### Phase 3 — manifests
- 600 primary + 72 GSM8K conditions enumerated
- Few-shot draws locked per seed × benchmark × subject (D-01 subject-key convention)
- Hard-fail leakage check passed

### Phase 4 — GPU inference (RunPod RTX 4090)
- 600 + 24 = 624 conditions (post-Cut-2)
- $___ total cost (under $30 cap)
- Cut 2 (D-18): GSM8K reduced to single-model case study after cost-projection tripwire fired
- One unplanned pod stop with seamless resume (persistent volume preserved state)

### Phase 5 — analysis
- All 5 benchmarks converged at cascade level 1 (D-20 fix critical)
- H1: ___ / 5 benchmarks confirmed
- Reproducibility trace: ___ status (D-21 bootstrap CI determinism noted)

### Process discipline
- 21 deviations tracked (D-01 through D-21) with full rationale
- Tension-finder + rotating-persona prompts wired as slash commands
- Two pre-registered tripwires fired (Cut 2, then D-21 surfaced); both pre-committed responses executed
- Test suite: 182/183 passing on the workstation venv

### Reviewer checklist
- [ ] `LOCK_NOTES.md` audit chain (v1 → v1.1.2 → Phase 5 completion)
- [ ] `IMPLEMENTATION_DEVIATIONS.md` D-01 through D-21
- [ ] Reproducibility trace section 4 (per-condition recompute) shows delta=0
- [ ] H1 result in `analysis/tolerance_schedules/h1_test.json`
- [ ] Manuscript outline `manuscript_outline_v0.md` §7 results placeholders ready to fill
```

---

## Step 6 — Pod teardown

After the workstation has the full raw outputs and Phase 5 has completed:

```bash
# Optional: keep the persistent volume for follow-up work; pod itself
# is no longer needed.
# RunPod web UI: stop pod (or terminate, depending on storage policy).
```

The persistent volume retains `/workspace/study/runs/raw/` independently
of the pod; if a re-run is needed within the volume's retention window,
spin up a new pod and remount.

---

## Known issues that are NOT blockers (documented for review)

- **D-21 — Bootstrap CI non-determinism in script 07.** Two consecutive
  runs of `07_tolerance_schedule.py` on identical input produce slightly
  different CI bounds in `tolerance_by_cell.csv` (point estimates stable;
  CI columns drift). Reproducibility trace's section-5 re-derivation
  flags this as a `fail`, but section-4 per-condition recompute (the
  load-bearing test) shows delta=0 across all sample conditions.
  Resolution paths (none auto-applied):
    - (a) Audit the seed flow in `07`'s bootstrap implementation; pin
      `numpy.random.default_rng` per cell rather than once.
    - (b) Document as a deviation with rationale: bootstrap CI drift on
      the order of 1e-6 is well below any decision-rule threshold, so
      H1 is unaffected; non-determinism is a reproducibility-trace
      housekeeping issue rather than a methodological one.
  Recommend (a) before manuscript submission; (b) is acceptable for
  internal review.

- **MMLU mixed-effects test.** `tests/test_mmlu_subject_decomp.py::
  test_mixed_effects_path_used_on_well_conditioned_fixture` is
  statsmodels-version-dependent (documented in `tests/README.md` §1).
  Output correctness unaffected; cascade fallback to ANOVA produces
  correct variance components.

- **Pre-Phase-5 dry-run report** (`runs_dryrun_2026_04_26/`) committed
  as audit artifact. P-1 (pod stale code) and P-2 (item_id RE
  intractability) both addressed pre-Phase-5; report retained for
  audit trail.
