# Tension Audit — 2026-04-26

**Trigger:** First dry run of `tension_finder_prompt.md` (2026-04-26). No specific lock amendment or submission imminent; this run is calibration of the prompt's silence-discipline and load-bearing-only commitments.

**Corpus:** Full default corpus per Phase 1.

- `/Users/john/code/gradience/CLAUDE.md` (read)
- `/Users/john/code/gradience/RESEARCH_INVENTORY.md` (read)
- `/Users/john/code/gradience/papers/n134_workshop/internal_memo.md` (read)
- `/Users/john/code/gradience/papers/n134_workshop/internal_summary.md` (read)
- `/Users/john/code/gradience/papers/n134_workshop/draft_v2_thesis_b.tex` (read)
- `/Users/john/code/gradience/papers/n134_workshop/pre_submission_edit_spec.md` (read)
- `/Users/john/code/gradience/papers/n134_workshop/pre_submission_edit_spec_tier_1_5.md` (read)
- `/Users/john/code/gradience/papers/n134_workshop/revision_notes.md` (read)
- `/Users/john/code/gradience/papers/n134_workshop/references.bib` (sampled at the EDIT-22 region)
- `/Users/john/code/gradience/papers/benchmark_reliability_study/preregistration/prereg_v1_1_LOCKED.md` (read)
- `/Users/john/code/gradience/papers/benchmark_reliability_study/LOCK_NOTES.md` (read)
- `/Users/john/code/gradience/papers/benchmark_reliability_study/IMPLEMENTATION_DEVIATIONS.md` (read)
- `/Users/john/code/gradience/papers/benchmark_reliability_study/SPEC_CPU_v0_2.md` (sampled by grep at scoring/prereg/random-effects sections)
- `/Users/john/code/gradience/papers/benchmark_reliability_study/SPEC_GPU_v0_1.md` (sampled by grep at scoring/prereg sections)
- `/Users/john/code/gradience/papers/benchmark_reliability_study/configs/analysis_config.yaml` (read)
- `/Users/john/code/gradience/research_review/2026-04-25.md` (read, including 00:44 UTC second pass)
- `/Users/john/code/gradience/research_review/2026-04-26.md` (read)

**Findings count:** 0 tensions, 1 minor.

---

## Findings

No tensions identified that pass all three filter questions (grounded in document text, not already program-tracked, load-bearing).

The audit identified roughly 25 candidate areas of potential tension during the read. All were dropped on filter-2 (already tracked elsewhere in the program: in editorial specs, deviation logs, lock-amendment notes, the research inventory's parallel-development discussion, or the revision-notes log) or filter-3 (stylistic/cosmetic). The program is — at this snapshot — running a disciplined commitment-tracking trail; the audit is reporting silence as information, per the prompt's calibration.

---

## Minor / stylistic items

- **`SPEC_CPU_v0_2.md` end-of-spec sentence still references `preregistration_v1.md` §3.10 (the v1 draft) as the gating condition for implementation work** (line 1948: "*No implementation work should begin until v1.1 pre-registration is locked and the open-question resolution (§3.10 of `preregistration_v1.md`) is complete.*"). The locked pre-registration is now `preregistration/prereg_v1_1_LOCKED.md`, and the §14 open questions there are all marked resolved. The reference is stale rather than wrong (the gate it names *was* cleared at v1.1 lock), but a fresh implementer reading SPEC_CPU end-to-end would be pointed at a non-canonical file. Stylistic / housekeeping; does not affect any decision.

---

## What was checked but found clean

- **Pre-reg lock chain (v1.1 → v1.1.1 → v1.1.2) vs. `configs/analysis_config.yaml`.** H3 ranking-reversal threshold consistent at `0.20` across `prereg_v1_1_LOCKED.md` §3.3, `LOCK_NOTES.md` §5, and `analysis_config.yaml`. `parse_failure_threshold: 0.30` consistent between `LOCK_NOTES.md` v1.1.2 amendment, `IMPLEMENTATION_DEVIATIONS.md` D-09 v1.1.2 resolution, and `analysis_config.yaml` `tolerance.parse_failure_threshold`. Mixed-effects cascade Level-1 random-effects list (`prompt, seed, scoring_rule, item, model_prompt_interaction, model_scoring_rule_interaction`) consistent between `analysis_config.yaml`, `SPEC_CPU_v0_2.md` §10.1, and `prereg_v1_1_LOCKED.md` §7.1.

- **Scoring-rule commitment.** Both `prereg_v1_1_LOCKED.md` §5.5 and `SPEC_CPU_v0_2.md` §6 (scoring-rules registry) commit to the LL + G&P pair (`ll_norm`, `generate_parse`, plus GSM8K's strict / permissive variants); no scoring-rule contradiction across pre-reg, spec, configs.

- **N134 contribution-claim language vs. parallel-development register.** Manuscript §1.2 EDIT-22 paragraph cites Messing, NIST 800-2/800-3, Reuel et al., and Camuffo et al., and explicitly retreats from "first to propose" framing. `RESEARCH_INVENTORY.md` Section 8 records the same retreat ("must be retired entirely (not merely softened)") and identifies the decimal-place precision tolerance schedule as the still-distinctive contribution. Manuscript §1.2's claim — "the tolerance-schedule prescription is what the present paper still distinctively contributes" — is consistent with the inventory's framing.

- **N134 contribution claim (iii) vs. §6 title and §6.4 epistemic register.** Tier 1 EDIT-01 retreated from "previously unnamed" to "underreported precision limitation" across abstract / contribution claim (iii) / §6 title; Tier 1.5 EDIT-16 then retreated §6.4 "discovery" language to "discovery-like in the narrower reporting sense." All four locations align on the reporting-register framing.

- **Reliability regime-scope caveat coordination.** EDIT-05 surfaced the same-task-vs-cross-task caveat from Appendix D into §4.2; Tier 1.5 EDIT-20 softened "framework prescription" phrasing. Manuscript §4.2 (lines ~604–608) and Appendix D (the SEM-transferability caveat at lines ~1273–1303) now name the same regime-scope limitation in compatible registers.

- **D-09 (LPM-not-logistic) vs. NIST AI 800-3 citation in N134.** N134 §1.2 cites NIST 800-3 as endorsing GLMM for AI variance decomposition; the benchmark study's D-09 v1.1.2 resolution uses LPM only in `g_theory` regime cells and routes parse-failure-dominated cells to sample-SD tolerance. The two papers are not in tension on this — N134 names NIST as parallel-development context; the benchmark study explicitly justifies its method choice against the NIST endorsement in `IMPLEMENTATION_DEVIATIONS.md` D-09's v1.1.2 resolution paragraph and its `LOCK_NOTES.md` v1.1.2 amendment.

- **Brittlebench (Romanou et al. 2026 v2) coverage decision.** `RESEARCH_INVENTORY.md` Section 4 HIGH explicitly records the program's deliberate decision to cite Brittlebench in the benchmark-reliability manuscript but *not* in N134's EDIT-22 paragraph (rationale: model-level brittleness vs. measurement-tolerance framings). The decision is documented; absence of Brittlebench from N134 is intentional, not drift.

- **Cross-paper "second worked demonstration" framing.** `prereg_v1_1_LOCKED.md` §13 names the benchmark study as the second worked demonstration of the framework; N134 `internal_memo.md` §strategic-implications and `internal_summary.md` §next-implications both name benchmark-evaluation reliability as the strongest next substrate. Aligned across both papers' internal documents and the locked pre-reg.

- **Post-hoc analysis register.** N134 §7.1 EDIT-18 substituted "no evidential weight" with "hypothesis-generating rather than confirmatory evidential status." The benchmark study's `prereg_v1_1_LOCKED.md` §11 deviation protocol now uses the matched language: "Post-hoc analyses are permitted but must be labeled as such and assigned hypothesis-generating rather than confirmatory evidential status (cf. the N134 paper's §7.1 post-hoc framing)." Cross-paper consistent.

- **Pre-reg §11 vs. `IMPLEMENTATION_DEVIATIONS.md`.** Pre-reg §11 specifies that any deviation from spec-committed parameters requires a `deviations.md` entry. `IMPLEMENTATION_DEVIATIONS.md` exists at `papers/benchmark_reliability_study/IMPLEMENTATION_DEVIATIONS.md` (a renamed-but-functionally-identical instance) and tracks D-01 through D-17. The naming difference is documented in `LOCK_NOTES.md` §9.

- **Internal author preference vs. paper claims.** `internal_memo.md` and `internal_summary.md` both record the author's preferred forward direction as "(a) substrate generalization + (c) framework extension." Neither document claims this is in the manuscript; `internal_summary.md` explicitly says the preference is "unstated in the paper." The manuscript's §9 conclusion paragraph "Three directions follow" lists activation-informed analogues, framework-prescription evolution, and voluntary-adoption invitation — agnostic across (a)/(b)/(c). No tension; the internal preference is honestly out of scope of the manuscript.

- **Daily-review rolling state.** `research_review/2026-04-26.md` notes (item 5) that the v1.1.2 lock is fresh and that none of today's findings affect the resolution. `LOCK_NOTES.md` v1.1.2 amendment cites the 2026-04-25 second-pass review for the NIST 800-3 surfacing; the temporal chain is consistently recorded.
