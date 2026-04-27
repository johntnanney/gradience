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

(Future entries follow this format, dated and section-organized per
the program's drafting cadence.)
