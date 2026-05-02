# Manuscript Commit Spec — 2026-05-02

**Purpose.** Commit the current session's benchmark-reliability manuscript work (§7–§9 prose, figures and tables, three Tier-1 appendices) on a fresh feature branch off `master`, push to origin, open PR.

**Audience.** Coding agents executing the commit. Self-contained.

**Status at start of work.** Working tree on `master` with the following uncommitted state:

- `papers/benchmark_reliability_study/manuscript/draft_v1.tex` — modified, +567 line diff. §7–§9 prose drafted; abstract empirical-findings updated; `\graphicspath` added; five figures/tables embedded (Figures 1, 2; Tables 1, 2, 3 in main body); three appendices added (App. A pre-registration, App. B pipeline + reproducibility trace, App. C LPM-vs-GLMM).
- `papers/benchmark_reliability_study/manuscript/references.bib` — modified, +15 line diff. New bib entry `hellevik2009linear` for App. C citation.
- `papers/benchmark_reliability_study/manuscript/draft_v1.pdf` — untracked, compiled artifact (should be gitignored, not committed).
- `papers/benchmark_reliability_study/figures/variance_components_stacked.pdf` — present in working tree but currently gitignored under repo-root `.gitignore` line 188 (`figures/`).
- `papers/benchmark_reliability_study/figures/tolerance_calibration.pdf` — same gitignore status.
- `MANUSCRIPT_COMMIT_2026_05_02.md` — untracked, this spec document itself; program-level operational artifact.
- `SESSION_HANDOFF_2026_05_02.md` — untracked, end-of-session handoff for fresh-session pickup; program-level artifact.

The two new figure PDFs are not visible to `git status` because the directory is gitignored. They need explicit handling (un-ignore via `.gitignore` edit, mirroring the n134 pattern at lines 195–196). The two program-level markdown documents are visible to `git status` and land on the same branch as a fourth atomic commit (§6 below).

**Constraint.** Do not rewrite history on any commit already pushed to origin. All cleanup adds new commits on a fresh feature branch.

---

## 1. Pre-flight verification

```bash
cd /Users/john/code/gradience
git branch --show-current        # expect: master
git log --oneline -3              # tip should be b8ebcf8 (Bucket A merge)
git status --short                # expect: 2 modified + 1 untracked (draft_v1.pdf)
```

Expected `git status --short` output:

```
 M papers/benchmark_reliability_study/manuscript/draft_v1.tex
 M papers/benchmark_reliability_study/manuscript/references.bib
?? MANUSCRIPT_COMMIT_2026_05_02.md
?? SESSION_HANDOFF_2026_05_02.md
?? papers/benchmark_reliability_study/manuscript/draft_v1.pdf
```

If anything else is dirty, **stop and report**.

Verify the figure files exist in the working tree (gitignored):

```bash
ls -la papers/benchmark_reliability_study/figures/*.pdf
```

Expected: `variance_components_stacked.pdf` and `tolerance_calibration.pdf` both present.

---

## 2. Create the feature branch

```bash
git checkout -b papers/benchmark_reliability_study/manuscript-v1
```

All subsequent commits land on this branch; `master` is unchanged until merge.

---

## 3. Commit 1 — `.gitignore` updates

Two changes:

**(a) Repo-root `.gitignore`:** un-ignore the benchmark-reliability `figures/` directory, mirroring the n134 pattern at lines 195–196. Insert after the `figures/` ignore line (line 188) or in the un-ignore block at lines 195–202:

```
!papers/benchmark_reliability_study/figures/
!papers/benchmark_reliability_study/figures/**
```

**(b) New manuscript-local `.gitignore`** at `papers/benchmark_reliability_study/manuscript/.gitignore`, mirroring `papers/n134_workshop/.gitignore`:

```
*.aux
*.log
*.bbl
*.blg
*.out
*.toc
*.synctex.gz
_build/
draft_v1.pdf
```

Verify after editing:

```bash
git check-ignore -v papers/benchmark_reliability_study/figures/variance_components_stacked.pdf
# Expected: NO output (no longer ignored)

git check-ignore -v papers/benchmark_reliability_study/manuscript/draft_v1.pdf
# Expected: matches manuscript/.gitignore (now ignored)
```

Stage and commit:

```bash
git add .gitignore papers/benchmark_reliability_study/manuscript/.gitignore
git status                       # verify only the two .gitignore files staged
git commit -m "papers/benchmark_reliability_study: un-ignore figures/, add manuscript/.gitignore for build artifacts"
```

---

## 4. Commit 2 — Generated figures

The two figure PDFs are now visible to `git add` (post .gitignore update).

```bash
git add papers/benchmark_reliability_study/figures/variance_components_stacked.pdf \
        papers/benchmark_reliability_study/figures/tolerance_calibration.pdf
git status                       # verify only these two files staged
git commit -m "papers/benchmark_reliability_study: figures 1 and 3 (variance components stacked bar, tolerance calibration plot)"
```

The figures were generated via a Python script (matplotlib + pandas) operating on `analysis/variance_components/aggregate_vc.csv` and `analysis/tolerance_schedules/tolerance_by_cell.csv` respectively. The generation script is not committed (one-shot artifact); regenerating requires running matplotlib against those CSVs. If a reproducible figure-generation script becomes important, that's a follow-on task for the editorial-pass cycle.

---

## 5. Commit 3 — Manuscript prose and bibliography

```bash
git add papers/benchmark_reliability_study/manuscript/draft_v1.tex \
        papers/benchmark_reliability_study/manuscript/references.bib
git status                       # verify only these two files staged
git commit -m "papers/benchmark_reliability_study: §7–§9 prose, figures and tables, three Tier-1 appendices"
```

The commit body content (long-form description for the commit message body):

```
- §7 (Empirical results): six subsections drafted against Phase 5 outputs. §7.1 variance components (4/5 cascade at level_1, Winogrande at level_3); §7.2 generalizability (H2 confirmed 4/5); §7.3 tolerance schedule (H1 confirmed 5/5; load-bearing paragraph with expanded in-frame interpretive close); §7.4 ranking stability (H3 confirmed 5/5; close-skill / cross-skill asymmetry surfaced); §7.5 MMLU subject decomposition (H4 pre-registered null at proportion 0.0046); §7.6 GSM8K case (single-model post-Cut-2 scope; strict-vs-permissive extraction contrast as regime-split-at-its-starkest, in the discovery-like-in-the-narrower-reporting-sense register).

- §8 (Discussion): five subsections. §8.1 methodological implications (field-level reading-discipline move + Heineman et al. signal/G-coefficient vocabulary acknowledgment); §8.2 limitations (FAMILY_B-equivalent caveat per cross-paper convention with N134); §8.3 relationship to precursor (cross-paper substrate-portability move; LOAD-BEARING for §1.3 contribution claim 1); §8.4 future work (five extensions); §8.5 anticipated objections (four objections ported from outline).

- §9 (Conclusion): brief; H1–H4 outcomes plus substrate-portability framing.

- Abstract empirical-findings sentence updated post-Phase-5 (replaces [TBD] placeholder).

- Five figures/tables embedded in main body: Figure 1 (variance components stacked bar), Figure 2 (tolerance calibration plot), Table 1 (generalizability coefficients), Table 2 (per-cell tolerance schedule, 30 rows), Table 3 (per-pair ranking reversals, 15 rows).

- Three Tier-1 appendices: App. A (pre-registration record, selected sections; see preamble for omitted-sections list); App. B (pipeline implementation + lock-amendment chain + 22 deviations + cascade trace + reproducibility trace + test suite); App. C (LPM-vs-GLMM methodological side-by-side, defending the v1.1.2 regime-split amendment per D-09 v1.1.2).

- references.bib: added hellevik2009linear entry for App. C citation.

- \graphicspath{{../figures/}{figures/}} added to preamble for tarball-flat structure compatibility.

Compile state: 41 pages (30 main body + 11 appendices), 0 citation warnings, 0 reference warnings, all forward references resolved.
```

---

## 5b. Commit 4 — Program-level documents (this spec + session handoff)

Two program-level markdown artifacts are committed on the same feature branch as a fourth atomic commit. They document this session's work and provide the fresh-session pickup point for the next session.

```bash
git add MANUSCRIPT_COMMIT_2026_05_02.md SESSION_HANDOFF_2026_05_02.md
git status                       # verify only these two files staged
git commit -m "program: 2026-05-02 manuscript commit spec + session handoff"
```

These could alternatively land on a separate program-level branch (mirroring the prior 2026-04-28 cleanup pattern's Bucket C). Including them on the manuscript-v1 branch is defensible because they document this work cycle specifically; separating them would be more conservative branch-scope discipline. Either pattern works; the included-on-branch pattern is the simpler default.

---

## 6. Compile verification

Verify the branch's HEAD compiles cleanly:

```bash
cd papers/benchmark_reliability_study/manuscript
TMPDIR=$(mktemp -d)
cp draft_v1.tex references.bib "$TMPDIR/"
mkdir -p "$TMPDIR/figures"
cp ../figures/variance_components_stacked.pdf ../figures/tolerance_calibration.pdf "$TMPDIR/figures/"
cd "$TMPDIR"
pdflatex -interaction=nonstopmode draft_v1.tex > /dev/null 2>&1
bibtex draft_v1 > /dev/null 2>&1
pdflatex -interaction=nonstopmode draft_v1.tex > /dev/null 2>&1
pdflatex -interaction=nonstopmode draft_v1.tex > /dev/null 2>&1
pdfinfo draft_v1.pdf | grep -i pages          # expect: Pages: 41
grep -c "Citation.*undefined" draft_v1.log    # expect: 0
grep -c "Reference.*undefined" draft_v1.log    # expect: 0
```

If page count, citation warnings, or reference warnings differ from expected, **stop and report** — the branch HEAD is not what was intended.

---

## 7. Push and PR

```bash
git push -u origin papers/benchmark_reliability_study/manuscript-v1
```

Manual PR-create URL (since `gh` is not authenticated, unless `gh auth login` was run separately):

```
https://github.com/johntnanney/gradience/compare/master...papers/benchmark_reliability_study/manuscript-v1
```

PR title:

```
papers/benchmark_reliability_study: §7–§9 prose, figures and tables, three Tier-1 appendices
```

PR description body (paste verbatim into GitHub's web UI):

```markdown
## §7–§9 prose + figures and tables + three Tier-1 appendices

This PR consolidates the manuscript prose and supporting artifacts for the benchmark-reliability paper, building on the Phase 5 closeout merged at b8ebcf8 (Bucket A). The draft is now compile-clean at 41 pages with 0 citation warnings and 0 reference warnings.

## Summary

- **§7 (Empirical results):** six subsections drafted against Phase 5 outputs. H1, H2, H3 confirmed; H4 is a pre-registered null. Each subsection follows a descriptive primary clause structure with brief in-frame interpretive close (heavier interpretive lifting reserved for §8). §7.3 (tolerance schedule) is the load-bearing subsection with the expanded interpretive close; §7.6 (GSM8K) names the regime-split phenomenon in the "discovery-like in the narrower reporting sense" register per cross-paper convention with N134.
- **§8 (Discussion):** five subsections. §8.3 (relationship to precursor) is the load-bearing cross-paper substrate-portability move grounding §1.3 contribution claim 1.
- **§9 (Conclusion):** brief; H1–H4 outcomes plus substrate-portability framing.
- **Abstract empirical-findings sentence updated** post-Phase-5.
- **Five figures and tables in main body:** Figure 1 (variance components stacked bar), Figure 2 (tolerance calibration plot), Tables 1–3 (generalizability, per-cell tolerance, ranking reversals).
- **Three Tier-1 appendices:** App. A (pre-registration record, selected sections), App. B (pipeline + reproducibility trace + 22 deviations + cascade trace + test suite), App. C (LPM-vs-GLMM methodological side-by-side defending the v1.1.2 regime-split amendment).
- **`.gitignore` updates** un-ignore the benchmark-reliability figures directory and add a manuscript-local .gitignore for build artifacts (mirrors the n134 convention).

## Key files for review

- `papers/benchmark_reliability_study/manuscript/draft_v1.tex` — full draft (30pp main + 11pp appendices)
- `papers/benchmark_reliability_study/manuscript/references.bib` — bibliography (one new entry: hellevik2009linear)
- `papers/benchmark_reliability_study/figures/variance_components_stacked.pdf` — Figure 1
- `papers/benchmark_reliability_study/figures/tolerance_calibration.pdf` — Figure 2

## Reviewer checklist

- [ ] §7 prose claims align with Phase 5 numerical outputs (cross-checked at session end; spot-check recommended).
- [ ] §8.3 cross-paper substrate-portability framing is consistent with the program-level cross-paper register conventions in CLAUDE.md (construct-validity, "discovery-like in the narrower reporting sense", post-hoc framing as "hypothesis-generating rather than confirmatory evidential status", FAMILY_B-equivalent capacity caveat).
- [ ] App. A redaction list (§§ omitted) is appropriate; reviewer comfortable with the pre-reg being available in supplementary materials for the redacted sections.
- [ ] App. B's structured deviation table (Table for D-01–D-05, D-07, D-08, D-10–D-17) plus full prose for D-06, D-09, D-18, D-19, D-20, D-21, D-22 covers the manuscript-relevant deviations adequately.
- [ ] App. C empirical demonstration on the present panel is empirical-only (cites Hellevik 2009 for the LPM-vs-GLMM agreement region; does not require an independent GLMM run on the data).
- [ ] Compile state: 41 pages, 0 citation warnings, 0 reference warnings.

## What this PR does NOT do

- **Editorial pass (Tier-1.5-equivalent reviewer-proofing).** Five overfull-hbox warnings remain; minor typography. Deferred to a separate editorial-pass cycle before submission.
- **App. D, E, F (Tier-2 appendices).** Reference-quality data dumps deferred per the appendix-tiering recommendation. The underlying CSV files in supplementary materials carry equivalent content for reviewers willing to look.
- **§6 test-count discrepancy** (181 vs. 182/183 passing). Minor pre-Phase-5 prose detail; deferred to editorial pass.
- **TMLR submission.** Submission tarball assembly is a separate downstream task, blocked by the editorial pass.

## What follows this PR

- N134 OpenReview upload (Task #35; user-side; ~15 min). Independent of this PR.
- Editorial pass on benchmark-reliability draft (Tier-1.5-equivalent reviewer-proofing). ~3–4 hours focused work; produces the submission-ready draft.
- TMLR submission of benchmark-reliability paper after editorial pass and (optionally) review-feedback signals from N134's first cycle.
```

---

## 8. End-state verification

After all commits and the PR is opened:

```bash
git log --oneline master..HEAD                  # expect: 4 commits
git status --short                              # working tree clean
git rev-parse origin/papers/benchmark_reliability_study/manuscript-v1  # matches local HEAD
```

**Report back:** branch name, four commit hashes, PR URL, working-tree-clean confirmation, compile result (page count, warning count).

---

## 9. If anything goes wrong

Stop and report rather than improvise if any of the following:

- Pre-flight `git status` shows files outside the expected 2 modified + 1 untracked.
- The `.gitignore` un-ignore doesn't resolve the figures (`git check-ignore` still reports them as ignored after the edit).
- The compile produces non-zero citation or reference warnings on the branch HEAD.
- The page count is not 41.
- Any commit fails to apply.
- `git push` fails (likely auth or non-fast-forward).

---

**End of spec.**
