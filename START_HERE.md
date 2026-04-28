# START HERE — Fresh Session Bootstrap

**For a fresh agent or fresh user-session opening this repository.** This document is the entry point. Read it first; then read the documents it points at, in the order specified; then begin work.

The repository is the working state of an active research program (two papers in flight; one cross-paper framework; ongoing daily-review and tooling streams). The program has accumulated substantial state across multiple documents over the last several sessions. This document does not duplicate that state — it tells you where to find it.

---

## 1. First action: read the session handoff

Before anything else, read this file:

**`/Users/john/code/gradience/SESSION_HANDOFF_2026_04_27.md`**

(Use the Read tool with the absolute path. The file is ~600 lines, fully self-contained, and indexes every load-bearing artifact in the program.)

The handoff document covers, in nine sections:

1. Headline state (N134, benchmark-reliability, program-wide)
2. Phase 5 results summary (descriptive only)
3. Today's accomplishments by stream
4. Document inventory (three tables: N134, benchmark-reliability, program-wide)
5. Outstanding work, prioritized
6. Recommended fresh-session opening prompts (four canonical entry points)
7. Open questions worth resurfacing
8. Cross-paper coordination state
9. Final state checksum

After reading the handoff, you will know what state the program is in, what's pending, and how to navigate from here.

---

## 2. Working directory and tool conventions

**Working directory:** `/Users/john/code/gradience/`

**Tool conventions:**

- **Read** — for any file. Use absolute paths. The file tools have access to the entire repository at `/Users/john/code/gradience/`.
- **Edit** — for in-place modifications. You must Read a file before editing it (the harness enforces this).
- **Write** — for creating new files only, or full rewrites. Do NOT use Write to modify a file you've already Read; use Edit instead.
- **Glob / Grep** — for searching the codebase.
- **Bash (`mcp__workspace__bash`)** — for shell commands. Note: the bash sandbox has separate path mappings:
  - `/Users/john/code/gradience/` → `/sessions/peaceful-practical-ptolemy/mnt/gradience/` (in bash sandbox)
  - The bash sandbox can READ files in the user's directory but may not be able to WRITE to existing files (e.g., `pdflatex` writing `.aux` files often fails). For compile-and-verify workflows, copy files to a fresh `mktemp -d` directory first.
  - Some operations (rsync to remote pods, file deletion in user's directory) cannot be done from sandbox; will require user-side execution.

**Pre-installed Python packages (in bash sandbox):** `pandas`, `numpy`, `statsmodels`, `scipy`, `pyarrow`, `jsonschema`, `pyyaml`, `scikit-learn`. If a missing package is needed, install with `pip install --break-system-packages <package>`.

**LaTeX compile in sandbox:** BasicTeX (TeX Live 2022). For compiling benchmark-reliability `draft_v1.tex`, the preamble already includes `\usepackage{lmodern}` and `\usepackage[T1]{fontenc}` to satisfy `microtype`. Compile from a temp directory: copy `.tex` + `.bib`, run four-pass (`pdflatex` → `bibtex` → `pdflatex` × 2). N134 uses `tmlr.sty` (loads `lmodern` internally).

---

## 3. Document reading order (priority)

After reading the handoff document, the next layer of context depends on what work the user wants done. Use the handoff's §6 "Recommended fresh-session opening" to determine which path applies.

**For any path:** read these foundational documents to understand the program:

1. **`/Users/john/code/gradience/CLAUDE.md`** — project-level instructions and architecture commitments.
2. **`/Users/john/code/gradience/RESEARCH_INVENTORY.md`** — external-work tracking; current through 2026-04-27 second-pass review.

**For N134 OpenReview-upload work** (the only N134 work cycle pending closure):

3. `/Users/john/code/gradience/papers/n134_workshop/openreview_submission_fields.md` — staged title, abstract, keywords, COI fields, pre-upload checklist.
4. `/Users/john/code/gradience/papers/n134_workshop/draft_v2_thesis_b.pdf` — final compiled PDF (17pp, ~466 KB).

**For benchmark-reliability §7–§9 drafting** (the next major work cycle):

3. `/Users/john/code/gradience/papers/benchmark_reliability_study/manuscript_outline_v0.md` — section structure, citation staging, cross-paper coordination notes, post-Phase-5 update notes.
4. `/Users/john/code/gradience/papers/benchmark_reliability_study/manuscript/draft_v1.tex` — current 16pp draft with §1–§6 prose and §7–§9 placeholders.
5. `/Users/john/code/gradience/papers/benchmark_reliability_study/preregistration/prereg_v1_1_LOCKED.md` — pre-registration committing to the analyses §7 reports.
6. `/Users/john/code/gradience/papers/benchmark_reliability_study/LOCK_NOTES.md` — lock-amendment chain (v1 → v1.1 → v1.1.1 → v1.1.2 + budget-amendment).
7. `/Users/john/code/gradience/papers/benchmark_reliability_study/IMPLEMENTATION_DEVIATIONS.md` — D-01 through D-21.
8. `/Users/john/code/gradience/papers/benchmark_reliability_study/CHANGELOG.md` — manuscript-level milestone log.
9. `/Users/john/code/gradience/papers/benchmark_reliability_study/PHASE5_HANDOFF.md` — operational templates for CHANGELOG / LOCK_NOTES / PR description (per relay agent's note in last session).
10. `/Users/john/code/gradience/papers/benchmark_reliability_study/analysis/` — Phase 5 result JSONs (variance components, tolerance schedules, ranking stability, MMLU subjects, GSM8K case).

**For the three-operational-steps cleanup** (CHANGELOG / commit / PR — pending despite earlier "Run 1–3" instruction):

3. `/Users/john/code/gradience/papers/benchmark_reliability_study/PHASE5_HANDOFF.md` — templates for the operational steps.
4. `/Users/john/code/gradience/papers/benchmark_reliability_study/CHANGELOG.md` — current state; needs Phase 5 milestone entry.
5. `/Users/john/code/gradience/papers/benchmark_reliability_study/LOCK_NOTES.md` — needs final-cost amendment record.
6. Branch context: feature branch `papers/benchmark_reliability_study/scaffold-and-pipeline`. Last commit `95488c4` per relay agent's report.

**For process-tooling continuation**:

3. `/Users/john/code/gradience/research_review/tension_finder_prompt.md` — v2 prompt; on-demand cross-document audit.
4. `/Users/john/code/gradience/research_review/daily_review_prompt.md` — daily literature-scan prompt.
5. `/Users/john/code/gradience/research_review/rotating_persona_prompt.md` — status uncertain; check existence at session open. If missing, see `papers/benchmark_reliability_study/POST_DRAFTING_WORKPLAN.md` Item 2B for the spec.
6. `/Users/john/code/gradience/research_review/2026-04-27.md` — most recent daily review (3 HIGH + 2 MEDIUM flags across morning + second-pass).

---

## 4. Program conventions to honor

Honor these without re-deriving them.

### Editorial-marker conventions

- **`% EDIT: <date> — <rationale>`** in `.tex` files marks editorial changes with date and rationale. Continues across editorial passes; do not strip.
- **`% ANON:`** in `.tex` files marks content anonymized for review. The `%-prefixed` form is a comment that does not render. ANON-marker restoration happens at camera-ready, not during review submission.
- **Tier-1, Tier-1.5, Tier-2 editorial passes** are the program's editorial discipline; each pass has its own spec document under `papers/n134_workshop/`. Specs are historical-document artifacts after execution; do not rewrite the spec to match post-execution state — annotate when corrections land.

### Documentation discipline

- **Pre-registration commitments are load-bearing.** Anything in `prereg_v1_1_LOCKED.md` or `LOCK_NOTES.md` amendments must be honored unless an amendment is added; post-hoc deviations require an entry in `IMPLEMENTATION_DEVIATIONS.md` (D-NN format).
- **Cross-paper coordination at the phrasing level is intentional.** Verbatim or near-verbatim convergences between N134 and benchmark-reliability prose (the "discovery-like in the narrower reporting sense" register, the post-hoc "hypothesis-generating rather than confirmatory" framing, the FAMILY_B-equivalent capacity caveat) are part of the program's coherence; do not paraphrase them out.
- **Auditability commitments override convenience.** Manifest CSVs, lock states, and amendment chains are version-tracked deliberately. When a patch makes a regenerable artifact non-regenerable, the deviation log is the audit trail — do not skip recording.

### Reporting discipline

- **Don't-interpret-during-results-reporting discipline.** When reporting Phase 5 numerical results, report what the JSONs say, not what they mean. Interpretation belongs to §8 of the manuscript, not §7. Match the user's register on this — they explicitly request "report what the JSONs say, not what they mean" when in production-discipline mode. Interpretation mode ("for the sake of argument, how would you interpret...") is a separate explicit conversational mode.
- **Pre-registered decisions are equal-weight regardless of outcome.** A confirmed hypothesis and a pre-registered null are both reportable findings; do not over-interpret confirmations or apologize for nulls.
- **The "discovery-like in the narrower reporting sense" register** (post-Tier-1.5 EDIT-16 in N134) is the canonical phrasing for findings the framework's discipline surfaces. Multiple instances exist across both papers; cross-paper coherence relies on the phrasing matching.

### Skepticism defaults for research-process work

- **Daily research review** defaults to skeptical: most days yield zero or one flagged paper. Calibration discipline forbids inflating criticality to look productive. The recent backlog-clearing pattern (multiple HIGH flags per pass) is expected to subside; steady-state flagging rate is <10%.
- **Tension-finder** defaults to silence: if nothing is in tension, the report says "What was checked but found clean" and lists what was checked. Padding the findings list with stretches violates the silence-discipline principle.

---

## 5. The four canonical fresh-session openings (from handoff §6)

Pick one based on user priority:

**(a) Closing N134 (TMLR submission):**
> "Run the three operational steps from `PHASE5_HANDOFF.md` (CHANGELOG / LOCK_NOTES updates, commit + push, PR), then walk through OpenReview upload for N134."

**(b) Benchmark-reliability §7–§9 drafting:**
> "Start drafting §7 of `papers/benchmark_reliability_study/manuscript/draft_v1.tex` against the Phase 5 results in `papers/benchmark_reliability_study/analysis/`. Begin with §7.3 tolerance-schedule (the load-bearing empirical paragraph). Use the writeup-register sketch from `SESSION_HANDOFF_2026_04_27.md` §3 stream notes."

**(c) Process tooling (rotating-persona prompt):**
> "Check whether `research_review/rotating_persona_prompt.md` exists from earlier session work. If yes, mark Item 2B in `POST_DRAFTING_WORKPLAN` complete. If no, finish drafting per the workplan's Item 2B specification."

**(d) Research-process continuation:**
> "Tomorrow's daily research review should run with the v2 prompt. Backlog should be largely cleared after 2026-04-27 second pass; expect <10% flagging rate."

---

## 6. State of pending tasks

The task list at session start has 51 entries. Of these:

- **50 completed.**
- **1 pending:** **Task #35 — OpenReview form + upload (TMLR submission).** User-side. Tarball + supplementary + OpenReview fields all staged.

The 50 completed entries include older N134 sub-tasks (#1–#32) that are recorded in `papers/n134_workshop/CHANGELOG.md` and `RESEARCH_INVENTORY.md` Section 9; they could be deleted as historical cleanup, but this is cosmetic and not blocking.

---

## 7. State of pending operational work (not in the task list)

These items were identified in the previous session but not closed:

1. **Three operational steps from end-of-Phase-5** (CHANGELOG / LOCK_NOTES updates with final numbers; commit + push session work as cohesive feature-branch update; open PR against master). Templates in `PHASE5_HANDOFF.md`. ~15–20 min total.

2. **§7–§9 drafting** of benchmark-reliability paper. ~5–8 hours focused work to land 22–24pp manuscript. Phase 5 data in hand.

3. **§3.5 vocabulary update** in `draft_v1.tex` — registered as outline note; recommended landing in §8.1 methodological-implications when §8 drafts.

4. **Rotating-persona prompt status check** — was being worked on by relay agent earlier; status uncertain.

5. **N134 post-acceptance prep** (~30–45 min). Camera-ready de-anonymization checklist; reviewer-response template; arXiv-preprint variant tarball.

6. **Appendix structure** for benchmark-reliability paper — App A through App F. ~1–2 hrs.

The handoff document's §5 "Outstanding work, prioritized" has the full list with priority ordering.

---

## 8. Critical user preferences to honor

The user's stated preferences (from the conversation history):

- **Audience: PhD philosopher, model-development focus.** Approach questions from advanced conceptual level; do not assume expert-level coding knowledge.
- **Format: prose over bullet points** for substantive responses. Lists where they aid clarity, prose where they don't.
- **Register: substantive depth.** The user appreciates conceptual reasoning and methodological reflection embedded in technical work.
- **Brevity: tight responses.** No padding, no excessive preamble, no over-explanation. Match the user's register.

---

## 9. If you're an agent reading this for the first time

Your suggested first message after reading the handoff is something like:

> "I've read `SESSION_HANDOFF_2026_04_27.md` and `START_HERE.md`. The program is at end-of-session 2026-04-27 with N134 awaiting OpenReview upload, benchmark-reliability §7–§9 drafting next, and the three Phase 5 operational steps (CHANGELOG / commit / PR) pending. What would you like to start with — the N134 close-out, the benchmark-reliability drafting, the operational steps, or something else?"

Then proceed based on the user's response, navigating from the handoff document's priority list and the canonical opening prompts.

---

## 10. Cross-references

| Need | File |
|---|---|
| Full state-of-program | `SESSION_HANDOFF_2026_04_27.md` |
| Project conventions | `CLAUDE.md` |
| External-work tracking | `RESEARCH_INVENTORY.md` |
| N134 manuscript | `papers/n134_workshop/draft_v2_thesis_b.tex` |
| N134 OpenReview prep | `papers/n134_workshop/openreview_submission_fields.md` |
| Benchmark-reliability manuscript | `papers/benchmark_reliability_study/manuscript/draft_v1.tex` |
| Benchmark-reliability outline | `papers/benchmark_reliability_study/manuscript_outline_v0.md` |
| Pre-registration | `papers/benchmark_reliability_study/preregistration/prereg_v1_1_LOCKED.md` |
| Lock notes | `papers/benchmark_reliability_study/LOCK_NOTES.md` |
| Deviations log | `papers/benchmark_reliability_study/IMPLEMENTATION_DEVIATIONS.md` |
| Phase 5 operational templates | `papers/benchmark_reliability_study/PHASE5_HANDOFF.md` |
| Phase 5 dry-run audit | `papers/benchmark_reliability_study/PHASE5_DRY_RUN_WORKPLAN.md` |
| Post-drafting workplan | `papers/benchmark_reliability_study/POST_DRAFTING_WORKPLAN.md` |
| Tension-finder prompt | `research_review/tension_finder_prompt.md` |
| Daily-review prompt | `research_review/daily_review_prompt.md` |
| Most recent daily review | `research_review/2026-04-27.md` |

---

This document is the entry point. The handoff document is the index. The artifacts the handoff points at are the working state. Begin by reading the handoff.
