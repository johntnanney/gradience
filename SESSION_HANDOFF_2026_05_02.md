# Session Handoff — 2026-05-02

**Purpose.** Single document consolidating state-of-program at end-of-session 2026-05-02 for fresh-session pickup. Self-contained: a fresh agent or fresh user-session should be able to navigate from this document without prior context.

**Scope.** Covers both papers (N134 / Thesis A and benchmark-reliability / Thesis B), the cross-paper coordination state, and pending operational work. Builds on `SESSION_HANDOFF_2026_04_27.md` (the prior handoff, covering the Phase 5 closeout and editorial work through 2026-04-27); the present handoff covers §7–§9 drafting and Tier-1 appendix work in this session.

---

## 1. Headline state

**N134 (TMLR submission target):**

- 17pp clean compile, post-Tier-1.5 reviewer-proofing pass, post-Reuel→Bean cite-correction. **Unchanged from 2026-04-27 handoff.**
- Tarball at `papers/n134_workshop/tmlr_main_submission_v2.tar.gz` (117 KB), tarball-source reproducibility verified during the 2026-04-28 cleanup against branch `papers/n134_workshop/tier-1-5-final` (now merged into master at commit `03f8ace`).
- OpenReview submission-fields document staged at `papers/n134_workshop/openreview_submission_fields.md`.
- **Pending Task #35: OpenReview upload (user-side, ~15 min).** Only remaining step before TMLR review queue. Strategically: should be uploaded soon to start the ~3-month TMLR cycle in parallel with benchmark-reliability paper development.

**Benchmark-reliability (Thesis B / N135):**

- Pre-registration at v1.1.2-LOCKED, config hash `fbc4a5dd`. Unchanged.
- Phase 5 canonical run completed and audit-clean. 18-artifact reproducibility trace passes. SPEC §13.2 gate cleared. Tag `v1_1_2_PHASE5_COMPLETE` on master at commit `b8ebcf8`.
- **Draft manuscript at 41pp clean compile.** §1–§9 prose drafted; abstract empirical-findings sentence updated post-Phase-5; five figures/tables embedded in main body; three Tier-1 appendices (App. A pre-registration record, App. B pipeline + reproducibility trace, App. C LPM-vs-GLMM); 0 citation warnings, 0 reference warnings.
- **Session work uncommitted on the working tree.** See `MANUSCRIPT_COMMIT_2026_05_02.md` for the commit + PR spec; coding agents will execute Bucket-style branch + commits next.
- All four pre-registered hypotheses decided: H1 confirmed (5/5), H2 confirmed (4/5), H3 confirmed (5/5), H4 not confirmed (0.0046; pre-registered null).
- Total deviations: 22 (D-01 through D-22). D-21 closed by deterministic-hash fix; D-22 closed by model-stripped pivot-key fix.

**Program-wide:**

- **Cross-paper strategic decision settled (2026-05-02):** two papers, separate consecutive TMLR submissions, N134 first, benchmark-reliability second. Reasoning: external triangulation (two independent peer-review streams) is stronger evidence for the substrate-portability claim than internal coherence (one combined paper); reviewer-expertise distribution favors focused submissions; current draft is structurally optimized for the two-paper path.
- Research inventory current through 2026-04-27 second pass. **Not updated this session;** no new daily reviews surfaced.
- Tension-finder v2 in place; rotating-persona prompt at `research_review/rotating_persona_prompt.md`.

---

## 2. Today's accomplishments

### Benchmark-reliability §7–§9 drafting

**§7 (Empirical results), six subsections drafted against Phase 5 outputs:**

- §7.1 Variance components: cascade convergence trace (4/5 at level_1, Winogrande at level_3), scoring-rule dominance pattern (12/15 cells > 50%), in-frame interpretive close on the variance-decomposition's role in the §3.4 reliability defense.
- §7.2 Generalizability coefficients: H2 confirmed at 4/5; HellaSwag exception ($g_{\mathrm{single}} = 0.953$); TruthfulQA-MC's persistent low reliability across averaging schemes ($g_{\mathrm{full}} = 0.290$) flagged as a measurement-instrument finding.
- §7.3 Tolerance schedule: H1 confirmed at 5/5 with the load-bearing paragraph and expanded interpretive close (per user feedback for "more in-frame work" — settled at register-confirmation step). The schedule licenses interval-required for 30/30 cells at single-occasion and 29/30 at full-design.
- §7.4 Ranking stability: H3 confirmed at 5/5; close-skill / cross-skill asymmetry surfaced (within-family Pythia-1.4B vs. Pythia-410M reverses at 21–54%; cross-lineage Pythia-vs-Qwen pairs reverse 0% on most benchmarks); Brittlebench comparison anchor preserved.
- §7.5 MMLU subject decomposition: H4 pre-registered null at 0.0046 (~1/20 of threshold); subject-level dominance of additive main effects.
- §7.6 GSM8K case: single-model post-Cut-2 scope on Pythia-1.4B; strict (tolerance 0.003, licenses 2-decimal) vs. permissive (tolerance 0.007, interval-required) extraction contrast as regime-split-at-its-starkest, in the "discovery-like in the narrower reporting sense" register per cross-paper convention.

**§8 (Discussion), five subsections drafted:**

- §8.1 Methodological implications: field-level reading-discipline move; Heineman et al. signal/G-coefficient vocabulary acknowledgment (closes the §3.5 vocabulary-update note from the outline).
- §8.2 Limitations: small-to-mid-scale regime, six-benchmark panel, constrained measurement universe, FAMILY_B-equivalent capacity caveat per cross-paper convention with N134.
- §8.3 Relationship to precursor: the load-bearing cross-paper substrate-portability move grounding §1.3 contribution claim 1. Joint empirical content (precursor's H1 null + rank-on-residuals + qualitative reproducibility ↔ this paper's H1/H2/H3 confirmed + H4 null + regime split + GSM8K extraction-rule sensitivity) framed as the empirical content of the portability claim.
- §8.4 Future work: five extensions (frontier-scale, chain-of-thought, IRT-style item-level, adversarial-perturbation universe, cross-model GSM8K).
- §8.5 Anticipated objections: four objections ported from outline (two-demonstrations-not-portability, regime-extrapolation, regime-split-looks-ad-hoc, no-contamination-assessment).

**§9 (Conclusion):** brief 2-paragraph close. H1–H4 outcomes plus substrate-portability framing.

**Abstract:** empirical-findings sentence updated post-Phase-5 (replaces `[TBD]` placeholder).

### Figures and tables (minimum-compelling-set)

Five artifacts produced and embedded:

- **Figure 1** (`figures/variance_components_stacked.pdf`): per-cell variance components stacked bar across 15 cells; scoring-rule dominance pattern visible immediately. Generated via Python (matplotlib + pandas) against `analysis/variance_components/aggregate_vc.csv`.
- **Figure 2** (`figures/tolerance_calibration.pdf`): log-log scatter of single-occasion vs. full-design tolerance per cell, colored by regime, with reference lines at ±0.005 and ±0.05 thresholds plus *y* = *x* diagonal. The most rhetorically powerful image in the paper.
- **Table 1**: per-benchmark generalizability coefficients across four averaging schemes.
- **Table 2**: per-cell tolerance schedule (30 rows × 6 columns); the schedule's empirical incarnation.
- **Table 3**: per-pair ranking-reversal fractions (15 rows × 6 columns) with H3 decision marks.

Tier 2 figures and tables (Figures 4–6, Tables 4–7) are deferred per the outline-tiered approach; underlying CSVs in supplementary materials carry the equivalent content.

### Tier-1 appendices

Three appendices drafted (~11pp combined):

- **App. A** (Pre-registration record, selected sections): version history, pre-registered hypotheses and decision rules, measurement universe and admissibility rules, materials, design and data hierarchy, pre-registered analyses, deviations protocol. Sections omitted: §1 Purpose (duplicates main-body §1), §2 Theoretical framing (duplicates §3), §8 Confound decomposition (duplicates §3.6), §9 Scope claims (duplicates §1.4 and §8.2), §10 Timeline and budget (operational), §12 Deliverables (operational), §14 Open Questions (resolved at v1.1-draft). Full pre-reg in supplementary materials at the v1.1.2 lock tag.
- **App. B** (Pipeline implementation and reproducibility trace): pipeline architecture (13-script CPU-pre / GPU / CPU-post), configuration provenance and lock-amendment chain (v1.1 → v1.1.2 + post-Phase-4 budget amendment), 22 implementation deviations (7 in full prose for manuscript-relevant ones; 15 in summary table for benign ones), cascade convergence trace per benchmark, 18-artifact reproducibility trace summary, test suite. Resolves the `\label{app:pipeline}` forward reference from §7.1.
- **App. C** (LPM-vs-GLMM methodological side-by-side): defends the v1.1.2 regime-split amendment per D-09 v1.1.2. LPM/GLMM background, agreement region (cells with parseability ≥ 0.30), disagreement region (cells with parseability < 0.30), NIST AI 800-3 endorsement and amendment trigger, empirical demonstration on the present panel. Cites Hellevik 2009 for the LPM/GLMM agreement region (new bib entry).

Tier-2 appendices (App. D per-cell tolerance with bootstrap CIs, App. E ranking-stability detail, App. F MMLU subject-decomposition detail) deferred per the appendix-tiering recommendation; not load-bearing for review, deferrable to post-acceptance editorial pass or to a separate revision cycle.

### Strategic decision: cross-paper submission structure

User decided after substantive analysis to keep N134 and the benchmark-reliability paper as separate consecutive TMLR submissions (rather than merging into one big paper). Reasoning recorded in conversation: external triangulation is stronger evidence for the substrate-portability claim than internal coherence; reviewer-expertise distribution favors focused submissions; the current draft is structurally optimized for the two-paper path. Decision settled in this session.

### Process-tooling

No new process-tooling work this session. Daily research review not run today (next-day expected).

---

## 3. Document inventory

### N134 / Thesis A (unchanged from 2026-04-27 handoff)

| Document | Path | Purpose |
|---|---|---|
| Manuscript | `papers/n134_workshop/draft_v2_thesis_b.tex` | 17pp post-Tier-1.5 |
| Compiled PDF | `papers/n134_workshop/draft_v2_thesis_b.pdf` | 466 KB |
| Tarball (current) | `papers/n134_workshop/tmlr_main_submission_v2.tar.gz` | 117 KB, ready to upload |
| Supplementary tarball | `papers/n134_workshop/supplementary_bundle.tar.gz` | 908 KB |
| OpenReview fields | `papers/n134_workshop/openreview_submission_fields.md` | Title/abstract/keywords/COI/checklist |

### Benchmark-reliability / Thesis B

| Document | Path | Purpose |
|---|---|---|
| Manuscript | `papers/benchmark_reliability_study/manuscript/draft_v1.tex` | **41pp; §1–§9 prose; 5 figures/tables; 3 Tier-1 appendices** |
| Compiled PDF | `papers/benchmark_reliability_study/manuscript/draft_v1.pdf` | Build artifact (will be gitignored after commit cycle) |
| Bibliography | `papers/benchmark_reliability_study/manuscript/references.bib` | Hellevik 2009 added this session |
| Outline | `papers/benchmark_reliability_study/manuscript_outline_v0.md` | Section structure, citation staging, cross-paper notes |
| Pre-registration (locked) | `papers/benchmark_reliability_study/preregistration/prereg_v1_1_LOCKED.md` | v1.1.2 |
| Lock notes | `papers/benchmark_reliability_study/LOCK_NOTES.md` | v1 → v1.1 → v1.1.1 → v1.1.2 + budget-amendment + Phase 5 completion |
| Deviations | `papers/benchmark_reliability_study/IMPLEMENTATION_DEVIATIONS.md` | D-01 through D-22 |
| CHANGELOG | `papers/benchmark_reliability_study/CHANGELOG.md` | Through Phase 5 completion |
| Phase 5 results dir | `papers/benchmark_reliability_study/analysis/` | All Phase 5 outputs (CSVs + JSONs) |
| Figure 1 | `papers/benchmark_reliability_study/figures/variance_components_stacked.pdf` | New this session; gitignored (un-ignore in commit cycle) |
| Figure 2 | `papers/benchmark_reliability_study/figures/tolerance_calibration.pdf` | New this session; gitignored (un-ignore in commit cycle) |

### Program-wide

| Document | Path | Purpose |
|---|---|---|
| Project instructions | `CLAUDE.md` | Codebase-level project state |
| Research inventory | `RESEARCH_INVENTORY.md` | Current through 2026-04-27 second pass |
| Daily reviews | `research_review/2026-04-25.md`, `2026-04-26.md`, `2026-04-27.md` | Daily literature-scan reports |
| Tension-finder prompt | `research_review/tension_finder_prompt.md` | v2 |
| Daily-review prompt | `research_review/daily_review_prompt.md` | Reusable agent prompt |
| Rotating-persona prompt | `research_review/rotating_persona_prompt.md` | Item 2B from POST_DRAFTING_WORKPLAN |
| Prior session handoff | `SESSION_HANDOFF_2026_04_27.md` | Phase 5 closeout |
| 2026-04-28 cleanup spec | `WORKING_TREE_CLEANUP_2026_04_28.md` | Three-bucket partition (executed; on master) |
| **This session handoff** | `SESSION_HANDOFF_2026_05_02.md` | **§7–§9 drafting + figures + appendices** |
| **This session commit spec** | `MANUSCRIPT_COMMIT_2026_05_02.md` | **Branch + commits + PR for next-session execution** |
| Bootstrap doc | `START_HERE.md` | Fresh-agent entry point |

---

## 4. Outstanding work, prioritized

### Load-bearing (gates downstream work)

1. **Execute the manuscript-v1 commit cycle** per `MANUSCRIPT_COMMIT_2026_05_02.md`. Three commits on a fresh feature branch (`papers/benchmark_reliability_study/manuscript-v1`) off master. ~15–20 min via coding agents. Establishes the §7–§9 work on git as a reviewable PR.

2. **N134 OpenReview upload (Task #35).** User-side, ~15 min. Tarball + supplementary + OpenReview fields all staged. Strategically: should land soon to put N134 into the ~3-month TMLR cycle in parallel with benchmark-reliability paper development.

### Next major work cycle

3. **Editorial pass on benchmark-reliability draft** (Tier-1.5-equivalent reviewer-proofing). ~3–4 hours focused work. Items to address:
   - Five overfull-hbox warnings (minor typography; lines 542–561, 599–610, 813–826, 907–908, 1019–1020).
   - §6 test-count discrepancy (181 passing vs. 182/183 from handoff; minor pre-Phase-5 prose detail).
   - Cross-paper register coherence audit against N134's draft (verify the four cross-paper registers — construct-validity, "discovery-like in narrower sense", post-hoc framing, FAMILY_B-equivalent — are phrasing-level consistent).
   - Citation completeness check (every claim that needs a citation has one; every cited work has a `references.bib` entry).
   - Page-count assessment for TMLR style (currently 41pp; TMLR is permissive but 35–40pp main + appendices is the typical comfortable range).
   - Optional: redo App. A in fully verbatim mode if the selected-sections approach feels under-thorough at editorial-pass review.

4. **Submit benchmark-reliability paper to TMLR** after editorial pass and (optionally) waiting briefly for N134 review feedback to inform the final §8.3 framing. Submission tarball assembly, OpenReview fields preparation.

### Ripe now (lower priority)

5. **Tier-2 appendices** (App. D, E, F) if length budget allows or if reviewer pressure warrants. Reference-quality data dumps; supplementary materials carry equivalent content for now. ~2–3 hours if produced as a set.

6. **Daily research review for 2026-05-02 (or next operating day).** Expected steady-state low flag rate (<10%) per the prior calibration discipline. Should be checked at session start.

7. **Tension-finder rerun** before TMLR submission of benchmark-reliability paper. Clean state expected given the recent register-conscious drafting.

8. **Task list cleanup.** 50+ completed historical entries; cosmetic; not blocking.

### Deferred / low priority

9. **N134 post-acceptance prep** (~30–45 min). Camera-ready de-anonymization checklist, reviewer-response template, arXiv-preprint variant tarball. Not relevant until N134 enters revision-or-acceptance state.

10. **Reproducible figure-generation script.** The two figures landed via a one-shot Python script not committed; if reviewers want figure regenerability, a small `scripts/figures.py` would land on a follow-on.

---

## 5. Recommended fresh-session opening

A fresh session can pick up from this document. Suggested opening prompts depending on user priority:

**If priority is closing N134 + executing the commit cycle:**

> "Run the `MANUSCRIPT_COMMIT_2026_05_02.md` spec to commit and PR the §7–§9 + figures + appendices work, then walk through OpenReview upload for N134."

**If priority is the benchmark-reliability editorial pass:**

> "Start the Tier-1.5-equivalent editorial pass on `papers/benchmark_reliability_study/manuscript/draft_v1.tex`. Items in priority order: overfull-hbox warnings (5 instances), §6 test-count discrepancy, cross-paper register coherence audit against N134's draft, citation completeness check, page-count assessment for TMLR. ~3–4 hours focused work."

**If priority is research-process continuation:**

> "Run today's daily research review with the v2 prompt. Steady-state low flag rate expected (<10%); backlog cleared after 2026-04-27 second pass."

**If priority is Tier-2 appendices:**

> "Produce the Tier-2 appendices (App. D per-cell tolerance with bootstrap CIs from `tolerance_by_cell.csv`; App. E ranking-stability detail with the existing `figures/ranking_stability_by_benchmark.png`; App. F MMLU subject-decomposition detail with the existing `figures/mmlu_model_subject_heatmap.png`). ~2–3 hours."

---

## 6. Open questions worth resurfacing in next session

1. **TMLR same-author related-submission policy.** The two-paper consecutive-submission strategy assumes TMLR is permissive about related submissions in close temporal proximity. Worth verifying before the second submission lands.

2. **Whether App. A should be fully verbatim or selected-sections.** Currently selected-sections; reviewer-friendly but introduces editorial-decision overhead. Verbatim option (~25pp) is more conservative but bloats the appendix.

3. **Reproducible figure-generation script.** Whether to land one before TMLR submission (defensible move for a measurement-discipline paper) or defer to post-acceptance.

4. **§3.5 vocabulary update placement.** I addressed it in §8.1 (Heineman et al. signal/G-coefficient acknowledgment) per the outline note. The outline alternative was to update §3.5 directly. Either is defensible; current placement is in §8.1.

5. **Page count and venue fit.** 41pp at v1; TMLR is permissive but a Tier-1.5 editorial pass might compress to 35–38pp. Worth deciding whether to compress, expand, or hold at 41 before submission.

6. **N134 cite-key normalization.** The references.bib for benchmark-reliability uses `anonymized2026n134` for the precursor. Once N134 is accepted (assuming it is), the cite key should normalize to the canonical author-year form. Editorial-pass-time cleanup; not urgent.

---

## 7. Cross-paper coordination state

Cross-paper coordination at the phrasing level is intentional per `CLAUDE.md` program policy. The four registers and their status as of end-of-session 2026-05-02:

- **Construct-validity register** — Both papers cite Messick (1989, 1995) and Cronbach-Meehl (1955); benchmark-reliability §3.1 + N134 §2 invoke construct-validity foundations identically. Status: consistent.
- **"Discovery-like in the narrower reporting sense" register** — N134 §6.4 (post-Tier-1.5 EDIT-16) frames the rank-on-residuals observation. Benchmark-reliability §7.6 frames the GSM8K extraction-rule contrast in the same register. The framing is now load-bearing in both papers. Status: consistent.
- **Post-hoc analysis register** — N134 §7.1 (post-Tier-1.5 EDIT-18) substitutes "no evidential weight" with "hypothesis-generating rather than confirmatory evidential status." Benchmark-reliability App. A §A.7 (deviations protocol) commits to the same language; §7.5 prose handles the H4 null in the same register. Status: consistent.
- **FAMILY_B-equivalent capacity caveat** — N134 §5.2 + §7 limitations acknowledge family-pair residualization as a high-capacity baseline by design. Benchmark-reliability §8.2 acknowledges the mixed-effects cascade as analogously high-capacity relative to per-cell N. Status: consistent.

**Vocabulary convergence note (Heineman et al.):** Acknowledged in benchmark-reliability §8.1 (signal ↔ generalizability coefficient mapping). N134 doesn't engage this directly because its substrate doesn't use the signal/noise vocabulary. Status: settled.

The §8.3 cross-paper section in the benchmark-reliability paper is the load-bearing paragraph that makes the §1.3 contribution claim 1 (substrate-portability) empirically defensible. The two papers' joint output (N134's H1 null + rank-on-residuals + qualitative reproducibility; benchmark-reliability's H1/H2/H3 confirmed + H4 null + bit-identical reproducibility + regime split + GSM8K extraction-rule sensitivity) constitutes the empirical content of the substrate-portability claim. The current §8.3 prose is structurally optimized for the two-paper-consecutive-submission strategy.

---

## 8. Final state checksum

End-of-session 2026-05-02:

- **N134**: tarball-ready; one user-side step (Task #35 OpenReview upload) to TMLR submission. Unchanged from 2026-04-27 handoff.
- **Benchmark-reliability**: 41pp draft (30 main + 11 Tier-1 appendices); §1–§9 fully drafted; five figures/tables embedded; compile clean (0 citation warnings, 0 reference warnings). Session work currently uncommitted on working tree.
- **Cross-paper strategic decision**: settled at separate consecutive submissions; current draft structurally optimized.
- **Inventory**: current through 2026-04-27 second pass; 7-voice parallel-development register; daily-review trajectory current through 2026-04-27.
- **Tooling**: tension-finder v2 + daily-review prompt + rotating-persona prompt all operational.
- **Tasks**: 50 completed historical + 1 pending (#35 OpenReview upload) + 5 created this session (all completed except #14 which finished the compile verify).

Fresh session can pick up from this document at any of §5's recommended openings. The program's load-bearing state is auditable from the documents this handoff points at; this document itself is the index.

---

*End of session handoff 2026-05-02.*
