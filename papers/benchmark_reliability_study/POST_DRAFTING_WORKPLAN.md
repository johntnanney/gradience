# Post-Drafting Workplan — 2026-04-26

**Context.** The N134 paper has reached the OpenReview-upload step (Task #35); the benchmark-reliability paper's draftable-now sections (§1–§6) are committed prose at `manuscript/draft_v1.tex` (16pp, four-pass clean compile). The GPU run continues; Phase 5 analysis waits for completion. This workplan covers four items that can be executed in the GPU window without waiting on Phase 5: a test-suite resolution sweep on Task #47, two process-tooling artifacts, an inventory-trajectory update, and a CHANGELOG seed for the benchmark-reliability paper folder.

Each item is specified to be picked up and executed independently by Claude or the user. Acceptance criteria are concrete; estimated times reflect actual focused work, not calendar elapsed.

---

## Item 1 — Test Suite Resolution (Task #47, currently in_progress)

**Goal.** Move Task #47 (T-01 fixture generation + T-02 full test suite) from in_progress to completed, with a clear pass/fail report, documented fixtures, and any remaining gaps explicitly enumerated.

**Why this is ripe now.** The earlier reports cited 181 passing / 2 skipped at v1.1.2 lock. Survey of `tests/` and `tests/fixtures/` shows 17 test files and a substantial fixture set (configs, mock_analysis, pathological_all_parse_fail, pathological_near_tied_ll, tiny_* fixtures, plus parquet fixtures for GSM8K and MMLU subject decomp). The 2-skipped number suggests the GPU-only paths can't be exercised in CI; everything else is exercisable. So Task #47 is closer to "verify and document" than "build new fixtures." Worth resolving rather than leaving in_progress.

**Concrete steps.**

1. **Run the full suite from a clean state.**
   ```bash
   cd /Users/john/code/gradience/papers/benchmark_reliability_study
   python -m pytest tests/ -v --tb=short 2>&1 | tee /tmp/test_run_2026_04_26.log
   ```
   Capture the pass/fail count and any newly-introduced failures.

2. **Inventory the fixture set.** For each fixture under `tests/fixtures/`, confirm:
   - The fixture is referenced by at least one test (use `grep -r <fixture_name> tests/`).
   - The fixture's generation logic is recorded in `tests/fixtures/make_fixtures.py` (or documented if generated outside that script).
   - Any unused fixtures are either removed or marked as future-use with an inline comment.

3. **Classify the 2 skipped tests.** Open each and confirm the skip reason. Acceptable skip reasons: GPU-only paths, network-dependent tests, optional-dependency tests. If a skip reason is "TODO" or "implement later," promote that to a tracked task.

4. **Document the test-suite state in `tests/README.md`.** If the file doesn't exist, create it. Sections:
   - **Suite scope** — which scripts are covered, which aren't (and why).
   - **Fixture inventory** — table mapping fixture file → test(s) that use it → generation source.
   - **Run instructions** — the `pytest` invocation, expected pass count, expected skips.
   - **Known limitations** — GPU-only paths, parse-failure-regime corner cases, etc.

5. **Update Task #47.** Status → completed. Add a one-line note to the task naming the final pass count and the README anchor.

**Acceptance criteria.**

- [ ] Full-suite run captured in `/tmp/test_run_2026_04_26.log` with explicit pass count.
- [ ] All fixtures under `tests/fixtures/` accounted for in the README inventory.
- [ ] Skip reasons explicitly documented (no opaque skips).
- [ ] `tests/README.md` exists and contains the four sections above.
- [ ] Task #47 marked completed with the final pass count noted.

**Estimated time.** 30–45 min focused. The bulk of the work is documentation; the test suite itself is mostly already in place.

**Dependencies.** None. Independent of GPU run, N134 submission, and benchmark-reliability §7-§9 drafting.

---

## Item 2 — Process Tooling

Two artifacts. Neither blocks anything; both are infrastructure investments that pay off across future research-program work.

### Item 2A — Tension-Finder v2 Refinements

**Goal.** Refine `research_review/tension_finder_prompt.md` based on the first dry-run experience (2026-04-26 audit on the full corpus + the narrower-scope rerun on N134), preserving the silence-discipline that worked while encoding the patterns the dry runs surfaced.

**What the dry runs revealed.**

- The "What was checked but found clean" section worked exactly as designed; the narrower-scope rerun's clean-checks list (parse_failure_threshold consistency, mixed-effects cascade matching, contribution-claim alignment, post-hoc framing convergence) is the kind of high-coverage low-noise output the prompt aims for.
- The 25-candidates-to-1-minor ratio on the full corpus + the 4-pointer-tensions on the narrow N134 scope confirms the silence-discipline holds in practice. The narrow scope did surface real tensions (App E/F/G pointers to non-shipping artifacts) that the full-corpus run did not — different scopes catch different things.
- The taxonomy (six tension types) was used straightforwardly; agent reports labeled findings cleanly.
- One thing the prompt did **not** explicitly cover: the narrow-scope run found tensions between the manuscript and the supplementary bundle's `README.md` "Not attached" list. This is a specific kind of cross-document tension worth naming as a sub-pattern.

**Concrete refinements for v2.**

1. **Add a "scope override examples" section** at the top of the prompt. The current Phase 1 says "if the user specifies a subset, use only what they name" but doesn't give example scope strings. Add:
   - Pre-OpenReview-submission scope: `draft_v2_thesis_b.tex`, `references.bib`, `revision_notes.md`, `supplementary/README.md`, the four supplementary files.
   - Pre-lock-amendment scope: pre-reg + analysis_config.yaml + relevant scripts.
   - Cross-paper scope: both manuscripts + their internal docs.

2. **Encode the manuscript-vs-supplementary-pointer pattern** as a sub-pattern under Type 3 (Stale commitment). Add an example:
   > "Manuscript appendix points to a supplementary artifact (e.g., 'Full trace in the supplementary materials') that the supplementary `README.md` 'Not attached' list explicitly excludes. This is a Type 3 tension: the pointer is stale relative to the supplementary's actual scope."

3. **Add example clean-checks for the report's "What was checked" section.** Concrete examples encode the pattern better than the current abstract guidance does. Add 4–6 example clean-checks like:
   > - "Pre-reg lock chain (vN → vN.x → vN.y) vs. analysis config: load-bearing parameters consistent."
   > - "Manuscript contribution claim (i) vs. abstract: aligned."
   > - "Cross-paper framing on [topic]: consistent across A and B."

4. **Add a "scope-and-clean-check matching" guidance.** The narrow-scope run's clean-checks were tighter than the full-corpus run's because the narrower scope licensed more specific clean-checks. Encode this: "On a narrow scope, the clean-checks should be specifically about the documents in scope; on a broader scope, the clean-checks should be cross-document consistency claims."

5. **Add a triggering-cadence section.** The current prompt is non-scheduled-on-purpose. v2 can keep that while naming a few canonical trigger moments (pre-lock, pre-submission, pre-amendment, post-major-edit-pass) more explicitly.

**File location.** Replace `research_review/tension_finder_prompt.md` in place; record the v1→v2 diff in a header comment block. Or write `research_review/tension_finder_prompt_v2.md` and deprecate v1. Recommend in-place replacement with a header comment that records the v1-experience refinements; the v1 version lives in git history.

**Acceptance criteria.**

- [ ] v2 prompt incorporates the four refinements above.
- [ ] v2 still satisfies the silence-discipline test: a clean-day report with zero findings is a valid output.
- [ ] Header comment block records the v1→v2 changes and the dry-run experience that drove them.
- [ ] First post-v2 run (whenever it happens) produces output of equal or better signal-to-noise than the v1 dry runs.

**Estimated time.** 30–45 min focused.

### Item 2B — Rotating-Persona Weekly Review Prompt

**Goal.** Draft `research_review/rotating_persona_prompt.md` — a weekly (not daily) prompt where each session adopts a fixed methodological persona and writes one focused critique of the program's current state from that angle.

**Why this is needed.** The tension-finder catches inconsistencies; it doesn't catch monoculture-of-perspective. The rotating-persona prompt forces variance by adopting outside-frame voices in turn. Lower frequency than daily; different epistemic shape from tension-finder.

**Persona roster (starting).**

The persona is the constraint that forces variance. Six are enough to cycle through quarterly without immediate repetition; rotate manually based on whichever angle would be most useful that week.

1. **Bayesian skeptic.** Reads the program's pre-registration discipline through the lens of subjective probability. Pushes back on threshold-test inference; asks why decision rules aren't framed as Bayes factors or posterior bounds.
2. **NIST-policy reader.** Reads everything through the AI 800-2 / AI 800-3 voluntary practices lens. Asks: where does the program comply with NIST guidance, where does it deviate, and is the deviation defensible?
3. **Philosopher of measurement.** Pushes the construct-validity register hardest. Asks: when the program says "measurement universe," is it using the term in the same sense Cronbach and Meehl did? Where does the operationalization-vs-construct distinction wobble?
4. **Frequentist statistician hostile to GLMM.** Asks: when does the variance-components decomposition assume random effects normality? Does it? On which cells does the assumption fail? Why hasn't the program reported a cell where GLMM-vs-LPM disagree on direction?
5. **ML-systems practitioner suspicious of psychometric framing.** Reads the program from the engineering side. Asks: is this overhead worth it? What does measurement discipline actually buy a practitioner trying to ship a benchmark? Is the prescriptive contribution operationally feasible?
6. **Applied benchmarking person who only cares about ranking.** Pushes against the variance-decomposition framing entirely. Asks: if rankings are roughly stable across most cells, why does the field need any of this? Where is the benchmark on which the framework actually changes a published claim?

**Output discipline.** Each session: one persona, one focused critique, ~500–800 words. The critique names what the persona's pressure surfaces — not as a list of grievances, but as a structured argument the program would need to engage. The session does not propose answers; it surfaces questions in the persona's voice for the program to answer.

**Silence/variance balance.** Unlike the tension-finder, the rotating-persona prompt expects an output every run — silence is less informative when the constraint is "adopt this voice." But the critique should be substantive; if the persona has nothing to push on this week, the agent picks a different persona.

**Output location.** Dated files at `research_review/persona_<persona-name>_<YYYY-MM-DD>.md`, e.g., `research_review/persona_bayesian_skeptic_2026-05-03.md`.

**Concrete steps to draft the prompt.**

1. Write the prompt's opening — same calibration register as the daily-review and tension-finder prompts.
2. Define the persona-selection mechanism (user picks at invocation; if not specified, agent picks one not used in the past four weeks).
3. Write the persona roster with a one-paragraph framing of each persona's commitments, characteristic pressure, and what the persona would and would not push on.
4. Write the output schema (preamble naming the persona; the critique itself; questions for the program to answer; a "what this persona explicitly does not push on" boundary).
5. Add anti-patterns (the persona is a constraint, not a costume; don't caricature; the critique must be one a real practitioner of that persona could endorse).

**Acceptance criteria.**

- [ ] `research_review/rotating_persona_prompt.md` exists.
- [ ] Six personas named with substantive framing.
- [ ] Output schema enforces structured critique + questions + boundary.
- [ ] Anti-patterns explicitly forbid caricature.
- [ ] First trial run (whenever launched) produces a critique that a real practitioner of the chosen persona could endorse — not a strawman.

**Estimated time.** 60–90 min focused.

---

## Item 3 — RESEARCH_INVENTORY.md Section 7 Trajectory Update

**Goal.** Add today's drafting milestone to the program's longitudinal record.

**Why.** Section 7 of `RESEARCH_INVENTORY.md` currently tracks daily-review trajectory only. Today's drafting work (N134 Tier 1.5 + Reuel→Bean rename + tarball; benchmark-reliability §1-§6 prose; tension-finder dry runs) is substantial enough that future audits should be able to find it. The current Section 7 schema doesn't have a slot for non-review entries; the cleanest move is to add a new entry to Section 7 that uses a different format, or to add a new Section 9 "Drafting and submission milestones" alongside the existing daily-review trajectory.

**Recommendation:** add a new **Section 9 — Drafting and submission milestones** rather than overload Section 7. The two trajectories track different things and should remain visually distinct.

**Concrete steps.**

1. Read the current `RESEARCH_INVENTORY.md` to confirm Section 8 is the last section.
2. Add Section 9 with this structure:

   ```markdown
   ## Section 9 — Drafting and submission milestones

   Substantive program-side work distinct from daily research-review activity:
   manuscript drafting, lock amendments, submission events.

   | Date | Paper | Milestone | Notes |
   |---|---|---|---|
   | 2026-04-26 | N134 / Thesis A | Tier 1.5 reviewer-proofing pass complete | EDIT-13 through EDIT-22 applied; Reuel→Bean cite-key rename caught at pre-submission verification; tarball at `tmlr_main_submission_v2.tar.gz` (117 KB) verified by fresh-extract four-pass compile; OpenReview submission-fields document staged at `papers/n134_workshop/openreview_submission_fields.md`. Page count held at 17pp. |
   | 2026-04-26 | N135 / Thesis B | v1.1.2 lock applied; manuscript outline + draft_v1 §1–§6 committed | D-09 regime-split (parse_failure_threshold = 0.30) per NIST 800-3 + early Phase 4 GPU output. Outline at `papers/benchmark_reliability_study/manuscript_outline_v0.md`; working .tex at `manuscript/draft_v1.tex`; references.bib drafted with `bean2025measuring` cite key. 16pp four-pass clean. §7-§9 wait for Phase 5. |
   | 2026-04-26 | (program-wide) | Tension-finder prompt drafted + dry-run executed | Prompt at `research_review/tension_finder_prompt.md`. Two runs: full corpus (1 minor finding, no load-bearing tensions); narrow N134 scope (4 pointer-tension blockers found and resolved at App E/F/G; numerical claims clean). |
   ```

3. Confirm the file ends cleanly and re-grep for any cross-section references that might need updating.

**Acceptance criteria.**

- [ ] Section 9 exists with the three 2026-04-26 entries.
- [ ] Format consistent with Section 7's table register.
- [ ] No cross-section refs broken.

**Estimated time.** 10–15 min.

**Dependencies.** None. Read + Edit only.

---

## Item 4 — Benchmark-Reliability CHANGELOG.md Seed

**Goal.** Create `papers/benchmark_reliability_study/CHANGELOG.md` documenting today's substantial drafting work, in the same register as `papers/n134_workshop/CHANGELOG.md` (which is ~10 KB and serves as the reference format).

**Why.** The benchmark-reliability paper folder doesn't have a CHANGELOG yet (verified: `ls CHANGELOG*` returned nothing). The N134 folder's CHANGELOG was load-bearing for the Tier 1.5 pass — every editorial decision was traceable to a CHANGELOG entry. The benchmark-reliability paper will need the same trail as it moves through pre-Phase-5 drafting → results integration → editorial passes → submission.

**Concrete content for the seed CHANGELOG.**

```markdown
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

- GPU run continues. Cost projection ~$27–31; tripwire at $29 with
  pre-committed fall-back (drop GSM8K symmetrically across remaining
  models if tripwire fires).
- Phase 5 analysis pipeline waits for GPU completion.
- §7 (results), §8 (discussion), §9 (conclusion) drafting waits for
  Phase 5 outputs.

---

(Future entries follow this format, dated and section-organized per
the program's drafting cadence.)
```

**Concrete steps.**

1. Create `papers/benchmark_reliability_study/CHANGELOG.md` with the content above.
2. Optionally: copy the section-organization structure (Pre-registration / Outline + manuscript / Cross-paper coordination / Pipeline + test suite / Phase status) into `papers/n134_workshop/CHANGELOG.md` as a comparison register, if N134's CHANGELOG uses a different format and harmonization would help. (Skip if the format mismatch is small and not worth the touch.)

**Acceptance criteria.**

- [ ] `papers/benchmark_reliability_study/CHANGELOG.md` exists.
- [ ] First entry covers 2026-04-26 with the five sections (pre-reg, outline + manuscript, cross-paper, pipeline + test suite, phase status).
- [ ] Format consistent enough with N134's CHANGELOG that future readers can navigate both with one mental model.

**Estimated time.** 15–20 min.

**Dependencies.** None.

---

## Suggested Order of Operations

If executing the workplan in one session:

1. **Item 4 (CHANGELOG seed)** — 15 min. Cheap and finishes a deliverable. Records today's work before details fade.
2. **Item 3 (inventory Section 9 update)** — 10 min. Cheap, complements Item 4.
3. **Item 1 (test suite)** — 30–45 min. Documentation-heavy; finishes a long-standing in_progress task.
4. **Item 2A (tension-finder v2)** — 30–45 min. While the dry-run experience is fresh.
5. **Item 2B (rotating-persona prompt)** — 60–90 min. Most substantial; can stretch.

Total focused-time budget: 2.5–3 hours. Easily fits the GPU-run window with margin.

If only some items get done, prioritize 4 → 3 → 1 (the deliverables that close existing tracking gaps) before 2A → 2B (the infrastructure investments that pay off downstream but are not currently blocking anything).

If resuming this workplan in a future session, the items are independent and can be picked up in any order; each item's acceptance criteria are self-contained.
