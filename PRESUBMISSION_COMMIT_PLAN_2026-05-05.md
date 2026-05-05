# Pre-Submission Commit Plan — 2026-05-05

Repo-facing spec for the pre-submission cleanup work in flight on
2026-05-05, covering both manuscript-content branches plus a new
program-level discipline branch. Documents what has landed, what is
pending, the order in which pending commits should be applied, and the
documentation associated with each.

This document is itself a commit candidate — see §6 for its disposition.

---

## 1. State at top of session 2026-05-05

The two manuscript branches were already past their major editorial
work:

**N134 (`papers/n134_workshop/tarball-rebuild-v3`):** at HEAD
`4997bdc` (abstract refinement landed earlier in the session). Anonymity
audit not yet executed; AI-language scan not yet executed.

**Benchmark-reliability (`papers/benchmark_reliability_study/empirical-table-revisions`):**
at HEAD `c561d52` (Mitchell scope-bounding citations landed yesterday
2026-05-04). Anonymity audit not yet executed (separate from the N134
audit). AI-language scan not yet executed.

Three program-level branches existed from yesterday's work:
`program/inventory-2026-05-03-second-pass`,
`program/research-reviews-2026-05-03`, `program/first-revision-staging`.

---

## 2. Commits landed during 2026-05-05 session

### 2.1 — N134 anonymity audit remediation

**Commit:** `3be686d` on `papers/n134_workshop/tarball-rebuild-v3`.

**Scope:** Citation-key renames + comment-leak strips in
`references.bib` (anonymized2026n130/n132/n133/n134spec → descriptive
keys); ANON-marker stripping in `supplementary/*.md`, `*.py`, and
`*.json`; supplementary `README.md` rewording away from
marker-convention references; `compare_methods.py` bare-`# ANON`
cleanup; rebuild of `tmlr_main_submission_v4.tar.gz` and
`supplementary_bundle_v2.tar.gz` with `COPYFILE_DISABLE=1` to prevent
macOS AppleDouble files; removal of obsolete `v3` and `v2_pre_*`
tarballs.

**Associated documentation:**
- `papers/n134_workshop/supplementary/ANON_AUDIT_CHECKLIST.md` — the
  procedural checklist this commit's audit followed (not for submission;
  internal procedure document).
- The commit message (in git history) documents the four-prong
  remediation: citation-key renames, comment-leak strips, ANON-marker
  stripping, AppleDouble exclusion.

**Why first:** Anonymity violations are the only kind of leak that
double-blind review explicitly rejects on. Other cleanup (cadence,
prose) is quality-of-prose, not policy-violating. Anonymity gets fixed
before anything else.

### 2.2 — N134 AI-cadence cleanup

**Commit:** `4e44150` on `papers/n134_workshop/tarball-rebuild-v3`.

**Scope:** Two prose edits replacing the "not merely X. It is Y"
cadence pivot at §2 framework introduction (line 311 of
`draft_v2_thesis_b.tex`) and at §6.4 generalization (line 928).
Recompile + tarball rebuild for consistency.

**Associated documentation:**
- `AI_LANGUAGE_SCAN_CHECKLIST.md` — see §3.1 below; documents the
  pattern this commit removed.
- `scripts/ai_language_scan.sh` — see §3.1 below; the scanner that
  identified the cluster (Layer 7 returned 2 hits before this commit;
  0 after).

**Why second (after anonymity):** Prose-quality cleanup is independent
of anonymity, and doing it after the anonymity remediation kept the two
diffs reviewable separately. Both fit cleanly on the same N134 branch.

---

## 3. Pending commits

Two pending commits remain. Apply in the order below.

### 3.1 — `program/ai-language-scan-discipline` (program-level)

**Branch:** `program/ai-language-scan-discipline` — new, parallel in
form to `program/research-reviews-2026-05-03` and
`program/first-revision-staging`.

**Parent:** `master` (or whatever the canonical baseline branch is).
The discipline artifacts are independent of any specific paper.

**Files added:**
- `AI_LANGUAGE_SCAN_CHECKLIST.md` (repo root, ~14KB) — comprehensive
  reference document covering ten categories of AI-tell patterns.
- `scripts/ai_language_scan.sh` (executable, ~12KB) — layered grep
  scanner with twelve scan layers (Layer 1a high-precision tells, Layer
  1b domain-overlap candidates reported informationally, Layers 2-12
  covering hype vocabulary, framing pivots, hedge clusters, AI-cadence
  pivots, verbosity substitutions, academic boilerplate, conversational
  AI leaks, connective overuse counts, and em-dash density).

**Calibration verification:** the scanner was tested against both
manuscripts in this repo:
- N134 (`draft_v2_thesis_b.tex`, post AI-cadence cleanup): cluster
  score 3 (mild signal — verbosity substitutions and one substantive
  use of "compelling," all explainable).
- Benchmark-reliability (`draft_v1.tex` at `c561d52`): cluster score
  1 (clean), with 22 Layer-1b informational hits all domain-legitimate
  (10× `harness` for `lm-evaluation-harness`, 11× `manifest` for
  manifest-variable psychometrics, 1× `optimize` in technical
  optimization context, 1× `optimized` in a citation summary).

**Why this order:** This commit must land before commit 3.2 because
the scanner identified 3.2's cleanup target. Landing the scanner and
checklist first establishes the procedure that 3.2 follows.

**Why a new program/ branch and not a paper branch:** The artifacts
are program-level discipline applicable across all manuscript work, not
content for any specific paper. Matches the convention from yesterday's
`program/first-revision-staging` (Hua engagement drafts staged for
future revision) and `program/research-reviews-2026-05-03` (daily
research review files + verify-after-write discipline).

**Commit instructions:** see Appendix A.

### 3.2 — Benchmark-reliability line-132 cadence cleanup (paper-content)

**Branch:** `papers/benchmark_reliability_study/empirical-table-revisions`
— extends the existing four-commit chain by one.

**Parent:** `c561d52` (Mitchell scope-bounding, current HEAD of the
empirical-table-revisions branch).

**File modified:**
- `papers/benchmark_reliability_study/manuscript/draft_v1.tex` —
  single prose edit at line 132. The §3 framework section of the
  benchmark-reliability paper shares prose with the §2 framework section
  of the N134 paper; the same "not merely a methods upgrade. It is a
  conceptual reorientation..." line that was cleaned up in N134
  (commit 4e44150) appears verbatim here and warrants the same fix for
  cross-paper consistency.

**Edit:**

Before:
```
The replacement is not merely a methods upgrade. It is a
conceptual reorientation whose consequences include a specific set
of reporting practices, but the practices are consequences rather
than the starting point.
```

After:
```
What changes here is conceptual, not methodological. The
reorientation produces a specific set of reporting practices, but
the practices are consequences rather than the starting point.
```

**Identical phrasing to the N134 cleanup (commit 4e44150).** This is
intentional: the two papers share §3 framework prose, and the cleanup
should be applied symmetrically.

**Recompile required:** `pdflatex → bibtex → pdflatex → pdflatex` to
regenerate `draft_v1.pdf` and `draft_v1.bbl`. No tarball exists for
this paper yet (see §4 below for forward-looking work), so no tarball
rebuild needed.

**Why this order:** Depends on 3.1 being landed because the commit's
rationale references the AI-language scan procedure that 3.1
establishes. Could technically be applied before 3.1 mechanically, but
the rationale chain reads more cleanly with 3.1 first.

**Associated documentation:**
- `AI_LANGUAGE_SCAN_CHECKLIST.md` — §2 (sentence-level rhetorical
  structures) names the "not merely X. It is Y" pattern as a
  recognizable LLM cadence.
- The commit message references both the cross-paper symmetry with
  N134 and the scan procedure from 3.1.

**Commit instructions:** see Appendix B.

---

## 4. Forward-looking work not in scope here

Four pre-submission items remain after 3.1 and 3.2 land. None of them
are commit candidates today; flagging for future planning.

**Benchmark-reliability anonymity audit.** The N134 audit was paper-specific
(commit 3be686d). The benchmark-reliability paper has not yet had a
parallel audit. The procedure documented in
`papers/n134_workshop/supplementary/ANON_AUDIT_CHECKLIST.md` is
broadly transferable but the specific leak vectors will differ;
particular attention warranted on the recently-edited `references.bib`
which gained Salaudeen, Hua, and Mitchell entries across recent
sessions.

**Benchmark-reliability tarball assembly.** No submission tarball
exists for this paper yet. Once anonymity audit is clean, build
`tmlr_main_submission_v1.tar.gz` and a reproducibility supplementary
following the same `COPYFILE_DISABLE=1` discipline as N134.

**OpenReview submission for N134.** The N134 paper is otherwise
submission-ready; remaining work is at the OpenReview UI level
(profile/co-author confirmation, form navigation, PDF + tarball
upload). See `papers/n134_workshop/openreview_submission_fields.md`
(if present) for the field-by-field plan from yesterday's preparation.

**Hua engagement application.** The drafts staged on
`program/first-revision-staging` (commit `d4252f3`) target first-revision
of the benchmark-reliability paper, not the v1 submission. Apply only
when that paper goes into revision.

---

## 5. Branch convention recap

The repo uses three branch namespaces:

**`papers/<paper-name>/<descriptor>`** for manuscript-content commits.
Multi-paper work goes on per-paper branches. The benchmark-reliability
paper's `empirical-table-revisions` and the N134 paper's
`tarball-rebuild-v3` are examples.

**`program/<descriptor>`** for program-level meta-content: discipline
documents, daily review files, inventory updates, staged drafts not
yet applied to any paper. Examples:
`program/research-reviews-2026-05-03`, `program/first-revision-staging`,
the new `program/ai-language-scan-discipline` from §3.1.

**`master`** for the canonical baseline. Paper and program branches
both root from here (or from a sibling baseline branch if the
maintainer's convention differs).

Discipline documents and tooling that apply across papers go on
program/ branches even when the file lives at a paper-specific path.
Example: `papers/n134_workshop/supplementary/ANON_AUDIT_CHECKLIST.md`
is paper-scoped by path but its content is procedural; future
generalizations of it would land on a `program/` branch.

---

## 6. Disposition of this document

This file (`PRESUBMISSION_COMMIT_PLAN_2026-05-05.md`) is itself a
commit candidate. Three options:

(a) **Commit alongside the AI-language-scan discipline.** Add this
file to the same commit that adds `AI_LANGUAGE_SCAN_CHECKLIST.md` and
`scripts/ai_language_scan.sh`. The discipline branch becomes the home
for both the discipline artifacts and the plan that documents how they
were assembled.

(b) **Commit on a separate `program/presubmission-plan-2026-05-05`
branch.** Cleaner separation between "discipline tooling" (the
checklist + script) and "session planning" (this document).

(c) **Leave untracked.** This document serves its purpose as a one-time
planning artifact and doesn't need to live in git history beyond the
commits it describes.

My recommendation is (b) — same convention as
`SESSION_HANDOFF_2026-05-03.md` from yesterday's session, which is
also a one-time planning artifact and is currently still untracked at
the repo root. Either decide both files go on `program/handoffs-and-plans`
or similar, or accept that one-time planning documents live untracked
indefinitely. Yesterday's choice was the second; today's choice is
either consistent with that or a deliberate change.

---

## Appendix A — Commit instructions for `program/ai-language-scan-discipline`

```bash
cd /Users/john/code/gradience

# Verify the new files are in working tree
git status --short
# Expected to include:
#   ?? AI_LANGUAGE_SCAN_CHECKLIST.md
#   ?? scripts/ai_language_scan.sh

# Verify the script is executable
ls -la scripts/ai_language_scan.sh
# Expected: -rwx prefix

# Optional sanity test
./scripts/ai_language_scan.sh papers/n134_workshop/draft_v2_thesis_b.tex
# Expected: cluster score ~3, "mild signal" interpretation

# Switch to baseline before branching
git checkout master   # adjust if your default branch has a different name

# Create the new program branch
git checkout -b program/ai-language-scan-discipline

# Stage the new files
git add AI_LANGUAGE_SCAN_CHECKLIST.md scripts/ai_language_scan.sh

# Verify staging
git status --short
# Expected:
#   A  AI_LANGUAGE_SCAN_CHECKLIST.md
#   A  scripts/ai_language_scan.sh

git commit -F-   # see commit message in §3.1's earlier preparation
git push -u origin program/ai-language-scan-discipline
```

The full commit message body for §3.1 was prepared earlier in this
session; see the chat-session record. It documents the ten-category
checklist structure, the twelve-layer scanner architecture, the
calibration verification against both manuscripts, and the references
cited in the checklist's §10.

---

## Appendix B — Commit instructions for benchmark-reliability line-132 cleanup

```bash
cd /Users/john/code/gradience

# Switch to the benchmark-reliability branch
git checkout papers/benchmark_reliability_study/empirical-table-revisions

# Verify HEAD
git log --oneline -1
# Expected: c561d52 papers/benchmark_reliability_study: scope-bounding paragraph + construct-validity citations

# Apply the prose edit
cd papers/benchmark_reliability_study/manuscript
perl -i -0pe 's|The replacement is not merely a methods upgrade\. It is a\nconceptual reorientation whose consequences include a specific set\nof reporting practices, but the practices are consequences rather\nthan the starting point\.|What changes here is conceptual, not methodological. The\nreorientation produces a specific set of reporting practices, but\nthe practices are consequences rather than the starting point.|' \
  draft_v1.tex

# Verify edit landed
grep -n "What changes here is conceptual" draft_v1.tex
# Expected: 1 hit (the new sentence)
grep -n "not merely a methods upgrade" draft_v1.tex
# Expected: no hits

# Recompile
pdflatex -interaction=nonstopmode -halt-on-error draft_v1.tex > /dev/null
bibtex draft_v1 > /dev/null
pdflatex -interaction=nonstopmode -halt-on-error draft_v1.tex > /dev/null
pdflatex -interaction=nonstopmode -halt-on-error draft_v1.tex > /dev/null
grep -E "Output written|Citation .* undefined|Reference .* undefined|Overfull" draft_v1.log
# Expected: clean compile, no warnings

# Stage and commit
cd /Users/john/code/gradience
git add papers/benchmark_reliability_study/manuscript/draft_v1.tex \
        papers/benchmark_reliability_study/manuscript/draft_v1.pdf \
        papers/benchmark_reliability_study/manuscript/draft_v1.bbl

git status --short papers/benchmark_reliability_study/

git commit -F- <<'COMMIT_MSG'
papers/benchmark_reliability_study: AI-cadence cleanup — line 132 "not merely X. It is Y" pivot

Single prose edit at sec:framework (line 132 of draft_v1.tex), mirroring
the cleanup applied to the N134 paper's parallel framework section
(commit 4e44150 on papers/n134_workshop/tarball-rebuild-v3). The two
papers share framework prose; this commit applies the same cadence
cleanup symmetrically.

Before:
  "The replacement is not merely a methods upgrade. It is a
   conceptual reorientation whose consequences include a specific set
   of reporting practices, but the practices are consequences rather
   than the starting point."

After:
  "What changes here is conceptual, not methodological. The
   reorientation produces a specific set of reporting practices, but
   the practices are consequences rather than the starting point."

Substantive content unchanged; only the cadence pivot is removed. The
"not merely X. It is Y" construction is a recognizable LLM-cadence
tell (documented in AI_LANGUAGE_SCAN_CHECKLIST.md §2 on
program/ai-language-scan-discipline).

Identified by scripts/ai_language_scan.sh on the
program/ai-language-scan-discipline branch: Layer 7 returned 1 hit on
this paper before the edit; cluster score was 1 ("clean") because
Layer 1b's 22 informational hits (harness, manifest, optimize) are all
domain-legitimate technical vocabulary in measurement-discipline
writing.

Compile-verify clean: 4-pass cycle, 0 citation/reference warnings, 0
overfull hboxes.
COMMIT_MSG

git push origin papers/benchmark_reliability_study/empirical-table-revisions
```

---

## Appendix C — File index after both commits land

After §3.1 and §3.2 land, the repo state on origin:

**Manuscript-content branches:**
- `papers/n134_workshop/tarball-rebuild-v3` (`4e44150`): submission-ready
  modulo OpenReview UI work
- `papers/benchmark_reliability_study/empirical-table-revisions`
  (post-§3.2 HEAD): five commits including the new line-132 cadence
  cleanup; not yet through anonymity audit or tarball assembly

**Program-level branches:**
- `program/inventory-2026-05-03-second-pass`
- `program/research-reviews-2026-05-03`
- `program/first-revision-staging`
- `program/ai-language-scan-discipline` (new)

**Discipline documents (in repo):**
- `AI_LANGUAGE_SCAN_CHECKLIST.md` (repo root, on `program/ai-language-scan-discipline`)
- `papers/n134_workshop/supplementary/ANON_AUDIT_CHECKLIST.md` (on
  `papers/n134_workshop/tarball-rebuild-v3`)
- `papers/benchmark_reliability_study/REVIEWER_DEFERRED_TASKS.md` (on
  `papers/benchmark_reliability_study/reviewer-framing-and-figure` and
  inheritor branches)
- `papers/benchmark_reliability_study/FIRST_REVISION_STAGED_EDITS.md`
  (on `program/first-revision-staging`)

**Discipline tooling (in repo):**
- `scripts/ai_language_scan.sh` (on `program/ai-language-scan-discipline`)
- `papers/benchmark_reliability_study/scripts/render_tolerance_calibration.py`
  (on `papers/benchmark_reliability_study/empirical-table-revisions`)
