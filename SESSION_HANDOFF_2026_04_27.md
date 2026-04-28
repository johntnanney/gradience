# Session Handoff — 2026-04-27

**Purpose.** Single document that consolidates state-of-program at end-of-session 2026-04-27 for fresh-session pickup. Self-contained: a fresh agent or fresh user-session should be able to navigate from this document without prior context.

**Scope.** Covers both papers (N134 / Thesis A and benchmark-reliability / Thesis B), the cross-paper coordination state, and all process tooling (daily research review, tension-finder, rotating-persona, inventory). Includes pointers to every load-bearing artifact rather than reproducing them.

---

## 1. Headline state

**N134 (TMLR submission target):**
- 17pp clean compile, post-Tier-1.5 reviewer-proofing pass, post-Reuel→Bean cite-correction.
- Tarball at `papers/n134_workshop/tmlr_main_submission_v2.tar.gz` (117 KB), verified by fresh-extract four-pass compile.
- Supplementary tarball at `papers/n134_workshop/supplementary_bundle.tar.gz` (908 KB, dated 2026-04-23, still current).
- OpenReview submission-fields document staged at `papers/n134_workshop/openreview_submission_fields.md`.
- **Pending Task #35: OpenReview upload (user-side, ~15 min).** Only remaining step before TMLR review queue.

**Benchmark-reliability (Thesis B / N135):**
- Pre-registration at v1.1.2-LOCKED, config hash `fbc4a5dd`.
- **Phase 5 canonical run completed cleanly** — all 9 phases ran, results in hand.
- Draft manuscript at `papers/benchmark_reliability_study/manuscript/draft_v1.tex`, 16pp clean compile (§1–§6 full prose; §7–§9 placeholders awaiting Phase 5 prose).
- Test suite: 182/183 passing on workstation post-D-19/D-20 patches (Task #47 complete).
- Total GPU cost: ~$18 (well under $30 cap; budget tripwire fired and Cut 2 executed cleanly at 2026-04-26 22:55 UTC).

**Program-wide:**
- Research inventory current through 2026-04-27 daily second-pass; **seven-voice parallel-development register** crystallized (Messing, NIST 800-2/800-3, Bean et al., Camuffo et al., Brittlebench, Signal and Noise, BenchRisk).
- Two new sub-registers crystallized: IRT-extension future-work (6 papers); Construct-validity-extension future-work (4 papers).
- Tension-finder v2 in place; first dry-runs validated silence-discipline.
- 18 deviations recorded in `IMPLEMENTATION_DEVIATIONS.md` (D-01 through D-21).

---

## 2. Phase 5 results summary

Results reported at the descriptive level; interpretation deferred to §8 of the manuscript when drafted.

**Pre-registered hypothesis tests:**

| Hypothesis | Threshold | Observed | Decision |
|---|---|---|---|
| **H1 (tolerance schedule)** | 3 of 5 primary benchmarks exceed ±0.005 single-occasion tolerance | **5 of 5 exceed** | **Confirmed (with margin)** |
| **H2 (generalizability)** | 3 of 5 single-occasion gen. coefficients < 0.80 | TBD per `analysis/variance_components/` JSONs | TBD |
| **H3 (ranking stability)** | 2 of 5 pairwise reversal fractions > 20% | TBD per `analysis/ranking_stability/` JSONs | TBD |
| **H4 (MMLU subject interaction)** | Model × subject interaction proportion ≥ 0.10 | **0.0046** (two orders of magnitude below) | **Not confirmed** |

**Cascade convergence (variance-components script 06, post-D-20 patch):**
- ARC-Challenge, HellaSwag, MMLU panel, TruthfulQA-MC: converged at level_1 (full random-effects structure) ✓
- Winogrande: did_not_converge at levels 1–2; converged at level_3 (drops `seed_id` random effect). Real cascade descent (Winogrande exemplar-selection variance is small enough at this sample size that statsmodels treats `seed_id` as singular), not a hang.
- **Zero level_4 fallbacks.** ANOVA fallback never needed.

**Reproducibility trace (script 98 / Section 4):**
- 5/5 sample conditions: delta = 0.00e+00 (bit-identical) ✓
- Section 5 acknowledges D-21: bootstrap CI drift on `tolerance_by_cell.csv` — point estimates stable, CI bounds environment-sensitive. Substantively a smaller-scale instance of N134's rank-on-residuals observation.

**Pipeline scale:**
- 1,024,512 primary item rows; 31,656 GSM8K item rows.
- 600 primary condition rows; 24 GSM8K condition rows (1-model post-Cut-2 scope).
- 30 tolerance cells (5 benchmarks × 3 models × 2 averaging levels); 60 aggregate variance-components cells.

**Cost:** pre-restart ~$16 + post-restart ~5 hrs × $0.40 ≈ ~$18 spent on GPU. Inside cap by ~$12.

---

## 3. Today's accomplishments by stream

### N134 final-mile (Tasks #33, #34 complete; #35 pending user-side)

- **Tier 1.5 reviewer-proofing pass** completed across `draft_v2_thesis_b.tex`: EDIT-13 through EDIT-22 applied. Page count held at 17pp.
- **Reuel→Bean cite-key correction** caught at pre-submission cite-verification pass (arXiv API confirmed first author Andrew M. Bean, 42 authors total; Anka Reuel not in author list). Cascade landed across `references.bib`, `draft_v2_thesis_b.tex`, `RESEARCH_INVENTORY.md` Section 2, and `pre_submission_edit_spec_tier_1_5.md` historical-document annotation.
- **App E / F / G prose rewrites** to drop pointers to non-shipping supplementary artifacts (caught by tension-finder narrow-scope dry run): summary now self-contained for review purposes.
- **Tarball assembly + verified four-pass compile in fresh extract** at `tmlr_main_submission_v2.tar.gz`.
- **OpenReview submission-fields document** staged with title, abstract, keywords, code/data availability statements, conflict-of-interest field structure, and pre-upload checklist.
- **MacTeX compile verification (Task #33)**: BasicTeX + collection-latexextra + collection-fontsrecommended toolchain confirmed working on user's machine.

### Benchmark-reliability paper

**Manuscript drafting** (`papers/benchmark_reliability_study/manuscript/draft_v1.tex`):
- Manuscript outline drafted at `manuscript_outline_v0.md` with reviewer-proofing revisions (six revisions: portability-claim defense, construct-hierarchy surfacing, "earn its keep" principle, anticipated-objections section, §1.2 differentiation, §2/§4 reorder).
- Full prose committed for §1 (introduction with reporting-gap motivation, construct-hierarchy reframe, parallel-development register, contribution claims, "what this paper does not do"), §2 (prompt-sensitivity baseline), §3 (framework setup, 6 subsections), §4 (parallel-development register, 3 subsections), §5 (pre-registered design with regime-split narrative), §6 (pipeline implementation summary).
- §3.5 deferred-update note registered for the signal/generalizability vocabulary convergence; landing recommended at §8.1 methodological-implications when §8 drafts.
- §4.1 updated twice during the session: sixth-voice update (Heineman et al.); seventh-voice update (BenchRisk via 2026-04-27 second-pass review).
- **§7, §8, §9 still placeholders** awaiting Phase 5 prose pass.

**Pre-registration and protocol:**
- v1.1.2-LOCKED with regime-split (D-09 → v1.1.2 amendment for parse-failure-dominated cells; threshold = 0.30 parseability).
- Budget-driven scope amendment 2026-04-26 22:55 UTC: GSM8K Tier 2 reduced to single-model case study (pythia_1_4b only) per pre-committed Cut 2; all three follow-ups landed (D-18 in deviations log; LOCK_NOTES amendment; §7.6 scope note in outline).

**Pipeline patches (during dry-run pass):**
- D-19: 288 MMLU manifest rows patched with actual per-subject sizes (171 / 378 / 545 / 282 / 100) from `cais/mmlu @ c30699e8`.
- D-20: `item_id` removed from levels 1–3 of variance-components cascade (was confounding the variance fit). 11/11 variance-components tests pass after fix.

**Test suite:** 182/183 passing on workstation venv (Python 3.13.7, pandas 3.0, numpy 2.4, statsmodels 0.14.6). One env-dependent failure (`test_mixed_effects_path_used_on_well_conditioned_fixture`) documented as known.

**Phase 5 canonical run** (this session, just before handoff): all 9 phases ran cleanly, results in hand. See §2 above.

### Process tooling

- **Tension-finder prompt v2** at `research_review/tension_finder_prompt.md` with refinements driven by v1 dry-run experience (scope-override examples, manuscript-vs-supplementary-pointer pattern, expanded clean-check examples, scope-and-clean-check matching, triggering-cadence section).
- **Daily research review** continued at v2 calibration: 2026-04-27 morning pass (3 flagged: Signal and Noise HIGH, Freiesleben HIGH, Growing Pains MEDIUM); 2026-04-27 second pass (2 flagged: BenchRisk HIGH, Kearns MEDIUM). Combined day total: 5 flags / 18 candidates examined.
- **Rotating-persona prompt** (Item 2B from POST_DRAFTING_WORKPLAN): status uncertain — was being resumed by relay agent earlier in session; not confirmed landed. Worth checking at fresh-session opening.

### Inventory and audit-trail hygiene

- `RESEARCH_INVENTORY.md` updated through 2026-04-27 second pass: HIGH section now 5 papers; IRT-extension sub-register crystallized with 6 papers; Construct-validity-extension sub-register crystallized with 4 papers; Section 6 outstanding queue empty (next-day fetch resolved); Section 7 trajectory current; Section 8 note 1 updated to 7-voice parallel-development register with NIST-alignment note added.
- `RESEARCH_INVENTORY.md` Section 9 — Drafting and submission milestones — populated with three 2026-04-26 entries.
- `papers/benchmark_reliability_study/CHANGELOG.md` seeded with 2026-04-26 milestone entry (per POST_DRAFTING_WORKPLAN Item 4).
- `papers/benchmark_reliability_study/POST_DRAFTING_WORKPLAN.md` covers the four post-drafting items (test suite, process tooling, inventory update, CHANGELOG seed).
- `papers/benchmark_reliability_study/PHASE5_DRY_RUN_WORKPLAN.md` was the basis for the dry-run that surfaced D-19 + D-20.
- `papers/benchmark_reliability_study/PHASE5_HANDOFF.md` (mentioned by relay agent) — should contain templates for CHANGELOG / LOCK_NOTES / PR description; recommended starting point for the operational steps below.

---

## 4. Document inventory

### N134 / Thesis A

| Document | Path | Purpose |
|---|---|---|
| Manuscript | `papers/n134_workshop/draft_v2_thesis_b.tex` | 17pp post-Tier-1.5 |
| Compiled PDF | `papers/n134_workshop/draft_v2_thesis_b.pdf` | 466 KB |
| Bibliography | `papers/n134_workshop/references.bib` | Post-Reuel→Bean rename |
| Tarball (current) | `papers/n134_workshop/tmlr_main_submission_v2.tar.gz` | 117 KB, ready to upload |
| Tarball (stale) | `papers/n134_workshop/tmlr_main_submission.tar.gz` | Apr 23, pre-Tier-1.5; superseded |
| Supplementary tarball | `papers/n134_workshop/supplementary_bundle.tar.gz` | 908 KB, separate upload |
| OpenReview fields | `papers/n134_workshop/openreview_submission_fields.md` | Title/abstract/keywords/COI/checklist |
| Internal memo | `papers/n134_workshop/internal_memo.md` | Strategic stance |
| Internal summary | `papers/n134_workshop/internal_summary.md` | Claims/limits/next-implications |
| Tier 1 spec | `papers/n134_workshop/pre_submission_edit_spec.md` | Historical, executed |
| Tier 1.5 spec | `papers/n134_workshop/pre_submission_edit_spec_tier_1_5.md` | Historical, executed |
| Revision notes | `papers/n134_workshop/revision_notes.md` | Running log |
| CHANGELOG | `papers/n134_workshop/CHANGELOG.md` | ~10 KB; reference format for benchmark-reliability CHANGELOG |
| Style files | `papers/n134_workshop/tmlr.sty`, `tmlr.bst`, `fancyhdr.sty` | TMLR template |
| Figures | `papers/n134_workshop/figures/` | h1_decision, layer_depth_trend, four_method_forest |

### Benchmark-reliability / Thesis B

| Document | Path | Purpose |
|---|---|---|
| Manuscript | `papers/benchmark_reliability_study/manuscript/draft_v1.tex` | 16pp; §1–§6 prose; §7–§9 placeholders |
| Bibliography | `papers/benchmark_reliability_study/manuscript/references.bib` | Includes bean, camuffo, romanou, sui, heineman, mcgregor entries |
| Outline | `papers/benchmark_reliability_study/manuscript_outline_v0.md` | Section structure, citation staging, cross-paper notes |
| Pre-registration (locked) | `papers/benchmark_reliability_study/preregistration/prereg_v1_1_LOCKED.md` | v1.1.2 |
| Lock notes | `papers/benchmark_reliability_study/LOCK_NOTES.md` | v1 → v1.1 → v1.1.1 → v1.1.2 + budget-amendment |
| Deviations | `papers/benchmark_reliability_study/IMPLEMENTATION_DEVIATIONS.md` | D-01 through D-21 |
| CHANGELOG | `papers/benchmark_reliability_study/CHANGELOG.md` | 2026-04-26 milestone entry; needs Phase 5 entry next |
| CPU spec | `papers/benchmark_reliability_study/SPEC_CPU_v0_2.md` | Pipeline specification |
| GPU spec | `papers/benchmark_reliability_study/SPEC_GPU_v0_1.md` | Inference specification |
| Analysis config | `papers/benchmark_reliability_study/configs/analysis_config.yaml` | parse_failure_threshold = 0.30 |
| Manifest (primary) | `papers/benchmark_reliability_study/manifests/conditions_primary.csv` | 600 conditions, post-D-19 patched |
| Manifest (GSM8K) | `papers/benchmark_reliability_study/manifests/conditions_gsm8k.csv` | 24 conditions (pythia_1_4b only) post-Cut-2 |
| Pipeline scripts | `papers/benchmark_reliability_study/scripts/00–10`, `98`, `99`, `gpu_inference.py` | 13 scripts |
| Test suite | `papers/benchmark_reliability_study/tests/` | 182/183 passing; tests/README.md current |
| Workplans | `POST_DRAFTING_WORKPLAN.md`, `PHASE5_DRY_RUN_WORKPLAN.md`, `PHASE5_HANDOFF.md` | Operational guides |
| Phase 5 results dir | `papers/benchmark_reliability_study/analysis/` (workstation) | Variance components, tolerance schedules, ranking stability, MMLU subjects, GSM8K case |

### Program-wide

| Document | Path | Purpose |
|---|---|---|
| Project instructions | `CLAUDE.md` | Codebase-level project state |
| Research inventory | `RESEARCH_INVENTORY.md` | External-work tracking; current through 2026-04-27 second pass |
| Daily reviews | `research_review/2026-04-25.md`, `2026-04-26.md`, `2026-04-27.md` | Daily literature-scan reports |
| Tension audit | `research_review/tension_audit_2026-04-26.md` | First on-demand cross-document audit (clean) |
| Tension-finder prompt | `research_review/tension_finder_prompt.md` | v2 (refined post-dry-run) |
| Daily-review prompt | `research_review/daily_review_prompt.md` | Reusable agent prompt |
| Rotating-persona prompt | `research_review/rotating_persona_prompt.md` (?) | Status uncertain; check at session open |
| Session handoff | `SESSION_HANDOFF_2026_04_27.md` | This document |

---

## 5. Outstanding work, prioritized

### Load-bearing (gates downstream work)

1. **N134 OpenReview upload (Task #35).** User-side, ~15 min. Tarball + supplementary + OpenReview fields all staged. After this lands, N134 enters TMLR's rolling review (~3-month cycle to first decision). Marks the close of the N134 work cycle.

2. **Three operational steps from end-of-Phase-5 (still pending despite "Run 1–3" instruction earlier in session):**
   - **(a) CHANGELOG.md and LOCK_NOTES.md updates** with Phase 5 final numbers. Templates in `PHASE5_HANDOFF.md` (per relay agent's note) ready to fill in.
   - **(b) Commit + push all session work** as one cohesive feature-branch update. Branch: `papers/benchmark_reliability_study/scaffold-and-pipeline`.
   - **(c) Open PR against master** with description template from `PHASE5_HANDOFF.md`.
   The work itself is operational/mechanical; the templates make it ~15–20 min total. Should land before any §7+ drafting begins so the data-state and documentation-state stay synchronized.

3. **§7–§9 prose drafting** of benchmark-reliability paper. With Phase 5 data in hand, §7 (results, six subsections), §8 (discussion, five subsections), and §9 (conclusion) are now drafting-eligible. Estimated 5–8 hours of focused prose work to land 22–24pp manuscript. Load-bearing paragraph: §8.3 cross-paper move (the substrate-portability claim made empirically). Recommended starting point: §7.3 tolerance-schedule.

4. **Appendix structure** for benchmark-reliability paper: App A (pre-reg verbatim), App B (pipeline detail), App C (LPM-vs-GLMM methodological side-by-side per D-09 v1.1.2), App D (per-cell tolerance tables), App E (ranking-stability detail), App F (MMLU subject-decomp detail). ~1–2 hrs.

### Ripe now (agent-doable, lower priority)

5. **§3.5 vocabulary update** — registered as outline note; recommended landing in §8.1 methodological-implications when §8 drafts (~1 paragraph, signal/generalizability convergence acknowledgment).

6. **Rotating-persona prompt status check.** Was being worked on by relay agent earlier in session; status uncertain. If file exists at `research_review/rotating_persona_prompt.md`, mark Item 2B complete in POST_DRAFTING_WORKPLAN. If not, finish drafting (~60–90 min per Item 2B spec).

7. **N134 post-acceptance prep** (~30–45 min). Camera-ready de-anonymization checklist (every `% ANON:` marker → restored content), reviewer-response template, arXiv-preprint variant of tarball (`\usepackage[preprint]{tmlr}` substitution).

8. **Task list cleanup.** 51 task entries with 50 completed + 1 pending. Older N134 sub-task entries (#1–#32) are recorded in CHANGELOGs and inventory Section 9; could be deleted as historical. Optional cleanup; not blocking.

### Deferred / low priority

9. **Tomorrow's daily research review** runs autonomously. Should drop below 10% steady-state flagging rate now that 2025-Q3 / 2026-Q1 backlog is largely cleared.

10. **Tension-finder rerun** before pre-OpenReview submission. Clean state expected given recent runs.

11. **§3.5 explicit edit in `draft_v1.tex`** (rather than the deferred §8.1 landing) if you'd prefer the §3 vocabulary in the same section as the rest of the framework setup. Either landing is defensible; the §8.1 landing is what was last recommended.

---

## 6. Recommended fresh-session opening

A fresh session can pick up from this document. Suggested opening prompts depending on user priority:

**If priority is closing N134:**
> "Run the three operational steps from PHASE5_HANDOFF.md (CHANGELOG / LOCK_NOTES updates, commit + push, PR), then walk through OpenReview upload for N134."

**If priority is benchmark-reliability §7–§9 drafting:**
> "Start drafting §7 of `papers/benchmark_reliability_study/manuscript/draft_v1.tex` against the Phase 5 results in `papers/benchmark_reliability_study/analysis/`. Begin with §7.3 tolerance-schedule (the load-bearing empirical paragraph). Use the writeup-register sketch from SESSION_HANDOFF_2026_04_27.md §3 stream notes."

**If priority is process tooling:**
> "Check whether `research_review/rotating_persona_prompt.md` exists from earlier session work. If yes, mark Item 2B in POST_DRAFTING_WORKPLAN complete. If no, finish drafting per the workplan's Item 2B specification."

**If priority is research-process continuation:**
> "Tomorrow's daily research review should run with the v2 prompt. Targeted query suggestion: backlog should be largely cleared after 2026-04-27 second pass; expect <10% flagging rate. One fetch candidate flagged for next-day if deferred items emerge."

---

## 7. Open questions worth resurfacing in next session

1. **Rotating-persona prompt landing status.** If not landed, the POST_DRAFTING_WORKPLAN Item 2B is the spec to finish.

2. **§3.5 vocabulary update placement.** Outline note registered; recommended §8.1 landing. User-decidable when §8 drafts.

3. **§7 register confirmation.** The writeup-register sketch (descriptive-only §7; interpretive §8 with cross-paper move) is my recommendation; user may prefer a different shape (e.g., interpretive notes within §7 subsections rather than batched in §8).

4. **Appendix C (LPM-vs-GLMM)** is named in the outline as load-bearing per D-09 v1.1.2; whether to draft it as a methodological-defense appendix or fold into §3.4 reliability-machinery prose is a structural choice.

5. **Venue decision for benchmark-reliability paper.** Outline lists TMLR (default, methodological track continuity) vs. NeurIPS Datasets and Benchmarks Track (Bean / BenchRisk venue) vs. ICLR Position Track (probably wrong venue). Phase 5 results are now in hand; venue decision can be settled.

6. **Task list pruning.** 50 completed entries from N134 work cycle could be deleted. Cosmetic; not blocking.

---

## 8. Cross-paper coordination state

The two papers' coordination is currently aligned at:

- **Construct-validity register** — N134 §2 invokes Messick (1989/1995); benchmark-reliability §3.1 invokes Cronbach-Meehl (1955) and Messick. Both papers' bibliographies include Messick, Cronbach-Meehl, Cronbach (alpha), Shrout-Fleiss (ICC), Brennan (G-theory).

- **"Discovery-like in the narrower reporting sense" register** — N134 §6.4 (post-Tier-1.5 EDIT-16) frames the rank-on-residuals observation; benchmark-reliability §5.4 / §7.6 (when drafted) frames the parse-failure regime split + the post-Cut-2 GSM8K 1-model scope; D-21 (bootstrap CI drift) is a small instance of the same register applied to analytical artifacts. Three (or four) instances now establish the register as load-bearing across both papers.

- **Post-hoc analysis register** — N134 §7.1 (post-Tier-1.5 EDIT-18) substitutes "no evidential weight" with "hypothesis-generating rather than confirmatory evidential status"; benchmark-reliability `prereg_v1_1_LOCKED.md` §11 commits to the matched language. Cross-paper consistent.

- **FAMILY_B-equivalent capacity caveat** — N134 §5.2 + §7 limitations (post-Tier-1.5 EDIT-17) acknowledge family-pair residualization as a high-capacity baseline by design; benchmark-reliability §8.2 (when drafted) and §8.5 anticipated-objections (already outlined) name the mixed-effects cascade as analogously high-capacity relative to per-cell N. Cross-paper consistent.

- **Parallel-development register** — N134 EDIT-22 §1.2 paragraph engages 5 voices (Messing, NIST 800-2, NIST 800-3, Bean, Camuffo); benchmark-reliability §4.1 engages 7 voices (the N134 five + Brittlebench + Heineman et al. + BenchRisk). The benchmark-reliability paper is later and engages more voices because the corpus has continued developing; both papers retire "first to propose" framing for "co-developed register, distinctive prescriptive contribution."

- **Vocabulary convergence note** — benchmark-reliability §4.1 acknowledges that Heineman et al.'s "signal" ↔ G-theory's "generalizability coefficient" is a vocabulary convergence; same observation queued for §3.5 / §8.1 manuscript landing. N134 doesn't engage this directly because its substrate doesn't use the signal/noise vocabulary.

The §8.3 cross-paper section in the benchmark-reliability paper, when drafted, is the load-bearing paragraph that makes the framework-portability claim of §1.3 contribution claim 1 empirically defensible. The two papers' joint output (N134's H1 null + rank-on-residuals + qualitative reproducibility; benchmark-reliability's H1 confirmed + H4 null + bit-identical reproducibility + regime-split + bootstrap CI drift) constitutes the empirical content of the substrate-portability claim.

---

## 9. Final state checksum

End-of-session 2026-04-27:

- **N134**: tarball-ready; one user-side step (OpenReview upload) to TMLR submission.
- **Benchmark-reliability**: §1–§6 prose drafted (16pp clean); §7–§9 placeholders; Phase 5 data in hand; three operational steps (CHANGELOG / commit / PR) pending closing-the-loop.
- **Inventory**: current through 2026-04-27 second pass; 7-voice parallel-development register; 6-paper IRT sub-register; 4-paper construct-validity sub-register; 18 deviations recorded; daily-review trajectory current.
- **Tooling**: tension-finder v2 + daily-review prompt + (possibly) rotating-persona prompt operational.
- **Cost**: ~$18 GPU; well inside $30 cap.
- **Tasks**: 50 completed, 1 pending (#35 OpenReview upload).

Fresh session can pick up from this document at any of §6's recommended openings. The program's load-bearing state is auditable from the documents this handoff points at; this document itself is the index.
