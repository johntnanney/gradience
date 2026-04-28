# Benchmark Reliability Study — CHANGELOG

This file records substantive program-side work on the benchmark-reliability
paper (Thesis B / N135). Daily research-review entries that flag external
literature for this paper live in `RESEARCH_INVENTORY.md` Section 7
(at the repo root); this file records work *on the paper itself*.

---

## 2026-04-26 — Drafting milestone: §1–§6 committed

### Pre-registration state

- v1.1.2-LOCKED (config hash `fbc4a5dd`); D-09 regime split applied
  (parse_failure_threshold = 0.30 in `analysis_config.yaml`).
- Lock-amendment chain: v1 (2026-04-24) → v1.1-draft (2026-04-24) →
  v1.1-LOCKED (2026-04-25) → v1.1.1-LOCKED (2026-04-25) →
  v1.1.2-LOCKED (2026-04-26). Full audit in `LOCK_NOTES.md`.

### Outline + manuscript

- Manuscript outline drafted at `manuscript_outline_v0.md` (~600 lines):
  section structure, abstract sketch, citation-staging table, cross-paper
  coordination notes, open decisions list. Six revisions applied in a
  reviewer-proofing pass: portability-claim defense, construct-hierarchy
  surfacing earlier, "earn its keep" principle for parallel-work citation,
  anticipated-objections subsection, §1.2 opener differentiation, §2/§4
  reorder.
- Working manuscript committed at `manuscript/draft_v1.tex`. 16pp,
  four-pass clean compile (BasicTeX + lmodern + microtype + natbib).
  - §1 Introduction (full prose; reporting-gap motivation, construct-
    hierarchy reframe, parallel-development register, contribution claims,
    "what this paper does not do").
  - §2 Prompt-sensitivity baseline (full prose).
  - §3 Framework setup (full prose; six subsections; §3.5 tolerance-
    schedule construction is the load-bearing distinctive contribution).
  - §4 Parallel-development register (full prose; co-developed register,
    distinguishing inferential targets, "what is cited and what is not"
    methodological note).
  - §5 Pre-registered design (full prose; materials, mixed-effects
    cascade, decision rules, regime split / v1.1.2 amendment).
  - §6 Pipeline implementation (full prose; thirteen-script structure,
    three-layer provenance, test suite status).
  - §7-§9 placeholders (wait for Phase 5).
- Bibliography at `manuscript/references.bib`. Verified attributions:
  Bean et al. 2025 (corrected from misattributed `reuel2025measuring`
  via 2026-04-26 arXiv-API verification — 42 authors, first author
  Andrew M. Bean); Camuffo et al. 2026 (full author list verified).

### Cross-paper coordination

- N134 (precursor) submission state: post-Tier-1.5; tarball at
  `tmlr_main_submission_v2.tar.gz` ready to upload; OpenReview fields
  staged at `papers/n134_workshop/openreview_submission_fields.md`.
- The benchmark-reliability paper's §8.3 ("Relationship to N134")
  awaits the N134 submission for a stable cross-paper-anchor target.
- Verbatim or near-verbatim convergences flagged with N134:
  "discovery-like in the narrower reporting sense" register
  (parse-failure regime split mirrors N134's rank-on-residuals
  observation); post-hoc analysis register ("hypothesis-generating
  rather than confirmatory" matches N134 EDIT-18); FAMILY_B-equivalent
  capacity caveat (mixed-effects cascade is high-capacity by design,
  mirrors N134 EDIT-17).

### Pipeline + test suite

- Pipeline implemented at `scripts/00–10` + `scripts/98`, `scripts/99`.
- Test suite: 181 passing, 2 skipped (GPU-only paths) at v1.1.2 lock.
  Task #47 currently in_progress; resolution covered in
  `POST_DRAFTING_WORKPLAN.md` Item 1.

### Phase status

- GPU run continues. Cost-projection tripwire at $29 fired at the
  32h45m elapsed checkpoint (projected $31 on trailing pace, $29.6 on
  per-model scaling); pre-committed Cut 2 executed at 22:55 UTC. GSM8K
  Tier 2 reduced from 3-model (72 conditions) to 1-model case study
  (pythia_1_4b's 24 conditions, completed pre-cut). Pythia_410m's and
  qwen2_5_1_5b's 48 GSM8K conditions removed from the run manifest.
  Inference resumed cleanly at PID 3823. Audit: D-18 in deviations,
  budget-driven scope amendment in LOCK_NOTES, §7.6 framing note
  staged in manuscript_outline_v0.md.
- Phase 5 analysis pipeline waits for GPU completion (revised total
  cost projection ~$27, well inside the $30 cap post-cut).
- §7 (results), §8 (discussion), §9 (conclusion) drafting waits for
  Phase 5 outputs. §7.6 prose-draft must reflect 1-model GSM8K scope.

---

## 2026-04-28 — Phase 4 GPU run complete + Phase 5 analysis pipeline executed

### GPU run summary

- **624 / 624 conditions complete** (post-Cut-2 total: 600 primary + 24
  pythia_1_4b GSM8K). 0 failures.
- Per-model: pythia_1_4b 224, pythia_410m 200, qwen2_5_1_5b_instruct 200.
- Wall-clock: ~74 hours total (with one unplanned pod stop at ~2026-04-26
  evening UTC; persistent volume preserved state; resumed cleanly on a
  new pod at 213.173.102.74:17180).
- Estimated final inference cost: ~$18, well inside the $30 cap and
  inside the $29 tripwire margin (the tripwire fired at the pre-Cut-2
  projection of ~$31; Cut 2 saved sufficient headroom).

### Phase 5 analysis pipeline

Driven by `scripts/run_phase5.sh`. All 9 phases ran cleanly on the
canonical 624-condition raw output:

- 1,024,512 primary item rows + 31,656 GSM8K item rows normalized.
- 600 primary + 24 GSM8K condition rows aggregated.
- Variance components (script 06, with D-20 cascade): 4 of 5 benchmarks
  converged at level_1 (arc_challenge, hellaswag, mmlu_panel,
  truthfulqa_mc). Winogrande did not converge at level_1 or level_2 and
  descended to level_3 (drops `seed_id` from the random-effects list);
  level_3 converged. Zero level_4 fallbacks. Per-benchmark fit time
  under 1.1s.
- Tolerance schedule (script 07, regime-aware): 30 cells (5 benchmarks ×
  3 models × 2 scoring rules).
- Ranking stability (script 08): ran clean.
- MMLU subject decomposition (script 09): mixed-effects path used.
- GSM8K case study (script 10): 24 condition rows (pythia_1_4b only,
  Cut 2 in effect).
- Reproducibility trace (script 98): per-condition recompute (section
  4) shows delta=0.00e+00 for all 5 sample conditions — the load-bearing
  reproducibility check passes. Section 5 reports `tolerance_by_cell.csv`
  re-derivation `fail` per the pre-known D-21 bootstrap CI determinism
  issue.
- Pipeline report (script 99): assembled.

### Pre-registered hypothesis test outcomes

Reported here as the JSONs report them; manuscript-side interpretation
still pending §7-§8 drafting.

- **H1 — confirmed.** Bootstrap lower bound of cross-model median
  single-occasion tolerance > 0.005 for **5 of 5** primary benchmarks
  (n_required = 3). See `analysis/tolerance_schedules/h1_test.json`.
- **H4 — not confirmed.** MMLU model × subject interaction proportion
  = 0.0046, below the 0.1 threshold. See
  `analysis/mmlu_subjects/h4_test.json`.
- H2 (generalizability), H3 (ranking reversals): outputs in respective
  analysis/ subdirs; manuscript §7.4 to engage.

### Manifest state

- `manifests/conditions_primary.csv`: 600 rows, all `condition_status =
  complete` after Phase 0 mark-complete sweep (re-derives status from
  jsonl-row count vs. local D-19 patched manifest, sidestepping
  pod-side stale-expected metadata).
- `manifests/conditions_gsm8k.csv`: 24 rows complete (pythia_1_4b),
  48 rows excluded_pre_run (Cut 2).

### Cross-paper coordination state

- §7.1 → §7.5 prose can now be drafted against actual numbers.
- §8 discussion: NIST 800-3 GLMM-vs-LPM appendix data is ready (the
  D-09 v1.1.2 regime split, D-20 cascade modification, and the
  winogrande level-3 descent give the appendix concrete material).
- §7.6 GSM8K case (single-model scope per Cut 2): manuscript outline
  already updated to reflect 1-model scope.
- N134 paper (precursor): submission-staged at `tmlr_main_submission_v2.tar.gz`;
  no cross-paper dependencies blocking benchmark-reliability §7-§9.

### Pipeline + audit-trail additions

- `scripts/run_phase5.sh` — driver with verified CLI invocations
  (corrected from the dry-run workplan's mismatches).
- `PHASE5_HANDOFF.md` — operator checklist with gates and templates.
- `IMPLEMENTATION_DEVIATIONS.md` D-21 — bootstrap CI non-determinism
  in script 07; documented with mitigation paths.
- `runs/inference.log` and `runs/phase5_run.log` — committed via
  `git add -f` as audit trail (parent `runs/` is gitignored for size).

### Phase status

- Phase 5 analysis: **complete.**
- Manuscript §7-§9 drafting: ready to begin.
- Pre-submission gate (§13.2 reproducibility trace = `pass`): cleared
  on 2026-04-28 (see follow-up entry below).

---

## 2026-04-28 (later) — D-21 + D-22 fixed; H2 + H3 closed; trace passes

Three close-out fixes after the initial Phase 5 entry above:

### D-21 fix (bootstrap CI non-determinism)

`scripts/07_tolerance_schedule.py` line 351 used Python's built-in
`hash((b_id, m_id, s_id))` to derive per-cell bootstrap seed offsets.
Python randomizes string hashes per process via `PYTHONHASHSEED`, so
consecutive invocations produced different cell seeds → different
bootstrap CIs. Replaced with `hashlib.sha256(cell_key).digest()[:4]`
for deterministic offset derivation. Two consecutive runs now produce
bit-identical `tolerance_by_cell.csv`. Also patched `98`'s section 3
to load both `conditions_primary.csv` and `conditions_gsm8k.csv`
(previously only primary, causing 24 GSM8K raw dirs to appear as
`raw_without_manifest`). **Reproducibility trace now passes (18
artifacts, 0 failures); SPEC §13.2 gate cleared.**

### D-22 fix (ranking-stability pivot keyed on model-baked condition_id)

`scripts/08_ranking_stability.py::pivot_condition_scores` pivoted on
`condition_id`, but our schema bakes `model_id` into the id, so each
row had accuracy for exactly one model column → `dropna(how="any")`
removed everything → 0 condition pairs for kendall tau, 0 reversal
candidates. Fix: build a model-stripped cell key from
`(subject_id, prompt_id, seed_id, scoring_rule_id)` and pivot on that.
Re-run produced 5 kendall tau cells (276–7140 pairs each), 15 reversal
cells, 15 win-probability cells. Added `h3_test.json` emission for
parity with `h1_test.json`/`h4_test.json`.

### H2 (generalizability) — confirmed

New script `scripts/11_generalizability.py` (the existing scripts only
loaded the H2 threshold but never used it). Computes G coefficients
under 4 averaging schemes per benchmark; tests H2 per pre-reg §3.3.

**H2 confirmed: 4 of 5 primary benchmarks have G_single < 0.80**
(threshold per `analysis_config.h2_generalizability_threshold`):

| benchmark | G_single |
|---|---:|
| arc_challenge | 0.564 |
| mmlu_panel | 0.301 |
| truthfulqa_mc | 0.049 |
| winogrande | 0.405 |
| hellaswag | 0.953 (above threshold) |

Output at `analysis/generalizability/{generalizability_coefficients.csv, h2_test.json}`.

### H3 (ranking reversal) — confirmed (post-D-22)

**H3 confirmed: 5 of 5 primary benchmarks have at least one
model-pair with condition-reversal fraction exceeding the 0.20
threshold.** The pythia_1_4b vs pythia_410m pair drives the result on
4 of 5 benchmarks (small overall_mean_diff between similarly-sized
base models → high condition-by-condition reversal rate).

| benchmark | n_pairs_exceeding |
|---|---:|
| arc_challenge | 1 |
| hellaswag | 1 |
| mmlu_panel | 1 |
| truthfulqa_mc | 3 |
| winogrande | 1 |

Output at `analysis/ranking_stability/h3_test.json`.

### Sanity sweep (clean)

- VC proportions sum to exactly 1.0 for every (benchmark, model) cell
  (15 cells; full partition consistency).
- Tolerance-schedule regime distribution: 23 g_theory + 7
  parse_failure_dominated (regime split is doing real work).
- Single-occasion licensed precision: **30 of 30 cells require
  interval reporting** (no two-decimal license available at single
  occasion).
- Full-design (24 conditions averaged): 29/30 still require interval;
  1 cell licenses two-decimal accuracy. The single-occasion → averaged
  contrast is the paper's prescriptive lever per pre-reg §7.2.
- Median single-occasion tolerance: **0.21** (vs. 0.005 H1 threshold);
  full-design median: 0.038.

### Pre-registered hypothesis-test outcomes — final

| Hypothesis | Result | Source |
|---|---|---|
| **H1** (single-occasion tolerance > 0.005 for ≥ 3/5 benchmarks) | **confirmed** (5/5) | `analysis/tolerance_schedules/h1_test.json` |
| **H2** (single-occasion G < 0.80 for ≥ 3/5 benchmarks) | **confirmed** (4/5) | `analysis/generalizability/h2_test.json` |
| **H3** (≥ 1 pair with reversal fraction > 0.20 for ≥ 3/5 benchmarks) | **confirmed** (5/5) | `analysis/ranking_stability/h3_test.json` |
| **H4** (MMLU model × subject interaction proportion ≥ 0.10) | **not confirmed** (0.0046) | `analysis/mmlu_subjects/h4_test.json` |

### Phase status (updated)

- Phase 5 analysis: **complete and reproducibility-trace-passing.**
- All four pre-registered hypothesis tests resolved.
- D-21 and D-22 documented; cascade descent + regime split + bootstrap
  determinism all verified.
- Manuscript §7–§9 drafting: ready to begin against actual results.

---

(Future entries follow this format, dated and section-organized per
the program's drafting cadence.)
