# Working-Tree Cleanup Spec — 2026-04-28

**Purpose.** Partition the current uncommitted working-tree state of `/Users/john/code/gradience/` into three logically-separate work streams, each with its own git provenance. After execution, the working tree is clean and three PRs are open against `master`.

**Audience.** Coding agents executing the cleanup. This document is self-contained — read top to bottom, execute in order, report at the verification gates.

**Status at start of work.** Branch `papers/benchmark_reliability_study/scaffold-and-pipeline` at HEAD `51328af`. Five pushed commits ahead of master via the prior session's Phase 5 closeout. Working tree has six modified files and ten untracked paths spanning three logical work streams. (The tenth untracked file is this spec document itself, `WORKING_TREE_CLEANUP_2026_04_28.md`, which is included in Bucket C below.)

**Constraint.** Do **not** rewrite history on any commit already pushed to origin (`fb96295`..`51328af`). All cleanup adds new commits; never amend or rebase published commits.

---

## 1. Decision recap (already settled)

The §1–§6 manuscript prose for the benchmark-reliability paper folds into the current PR (Bucket A below) rather than a separate follow-on branch. The PR's scope expands from "scaffold + pipeline" to "scaffold + pipeline + Phase 5 closeout + §1–§6 prose." This is honest about what the branch contains, avoids working-tree limbo for the prose during §7 drafting, and lets reviewers sanity-check that prose claims match Phase 5 numbers in a single review pass.

The N134 final-mile changes go to a fresh branch off `master` (Bucket B). The program-level artifacts (handoff, START_HERE, daily reviews, inventory updates) go to a fresh branch off `master` (Bucket C).

---

## 2. Pre-flight verification

Run before touching anything:

```bash
cd /Users/john/code/gradience
git log --oneline -2          # tip should be 51328af
git status --short             # exactly 6 modified + 9 untracked, listed below
git remote -v                  # confirm origin
git branch -a | grep master    # confirm master exists locally and on origin
```

Expected `git status --short` output:

```
 M RESEARCH_INVENTORY.md
 M papers/benchmark_reliability_study/figures/ranking_stability_by_benchmark.png
 M papers/benchmark_reliability_study/manuscript_outline_v0.md
 M papers/n134_workshop/draft_v2_thesis_b.tex
 M papers/n134_workshop/pre_submission_edit_spec_tier_1_5.md
 M papers/n134_workshop/references.bib
?? SESSION_HANDOFF_2026_04_27.md
?? START_HERE.md
?? WORKING_TREE_CLEANUP_2026_04_28.md
?? papers/benchmark_reliability_study/manuscript/
?? papers/n134_workshop/internal_memo.md
?? papers/n134_workshop/internal_summary.md
?? papers/n134_workshop/openreview_submission_fields.md
?? papers/n134_workshop/pre_submission_edit_spec.md
?? research_review/2026-04-26.md
?? research_review/2026-04-27.md
```

If anything else is dirty (extra files, unexpected modifications), **stop and report** before proceeding.

---

## 3. Bucket A — benchmark-reliability (current branch)

**Branch:** `papers/benchmark_reliability_study/scaffold-and-pipeline` (already checked out).

**Files (all benchmark-reliability):**

- `papers/benchmark_reliability_study/manuscript/draft_v1.tex` — new; §1–§6 prose, §7–§9 placeholders
- `papers/benchmark_reliability_study/manuscript/references.bib` — new; bibliography for the manuscript
- `papers/benchmark_reliability_study/manuscript_outline_v0.md` — modified, +2 lines
- `papers/benchmark_reliability_study/figures/ranking_stability_by_benchmark.png` — modified, regenerated post-H3 fix (5 KB → 51 KB)

**Action:**

```bash
git add papers/benchmark_reliability_study/manuscript/ \
        papers/benchmark_reliability_study/manuscript_outline_v0.md \
        papers/benchmark_reliability_study/figures/ranking_stability_by_benchmark.png
git status                     # verify only the four paths above are staged
git commit -m "papers/benchmark_reliability_study: §1–§6 manuscript prose + Phase-5-driven outline and figure updates"
git push
```

**Verification gate A:**

- `git log --oneline -1` → tip is the new commit
- `git status` → working tree no longer shows the four Bucket A paths
- `git status` should still show the Bucket B (N134) and Bucket C (program-level) entries — confirm those are unchanged

**PR description update (mechanical):**

If a PR is already open against `master` for this branch, replace the description body with §9 below. If no PR is open yet, use §9's body when opening it. The `gh` CLI is unauthenticated; either run `gh auth login` (~30 sec, device flow) and use `gh pr edit` / `gh pr create`, or paste the body via the GitHub web UI at:

```
https://github.com/johntnanney/gradience/compare/master...papers/benchmark_reliability_study/scaffold-and-pipeline
```

---

## 4. Bucket B — N134 final-mile (new branch off master)

**Branch to create:** `papers/n134_workshop/tier-1-5-final` (off `master`).

**Files (all N134):**

- Modified: `papers/n134_workshop/draft_v2_thesis_b.tex` (~277 line diff: +234, −43), `papers/n134_workshop/references.bib` (+72 lines, Reuel→Bean correction cascade), `papers/n134_workshop/pre_submission_edit_spec_tier_1_5.md` (+2 lines, historical-document annotation)
- Untracked: `papers/n134_workshop/internal_memo.md`, `papers/n134_workshop/internal_summary.md`, `papers/n134_workshop/openreview_submission_fields.md`, `papers/n134_workshop/pre_submission_edit_spec.md`

**Mechanics.** The cleanest approach: temporarily commit the remaining working-tree changes on the current branch (Buckets B + C together) as a "wip" commit, switch to master, create the new N134 branch, cherry-pick + filter, then drop the wip commit on the original branch.

```bash
# On scaffold-and-pipeline, after Bucket A is committed and pushed:
git add papers/n134_workshop/ \
        RESEARCH_INVENTORY.md \
        SESSION_HANDOFF_2026_04_27.md \
        START_HERE.md \
        WORKING_TREE_CLEANUP_2026_04_28.md \
        research_review/2026-04-26.md \
        research_review/2026-04-27.md
git commit -m "WIP: stash N134 + program-level for partition (do not push)"
WIP_COMMIT=$(git rev-parse HEAD)

# Create N134 branch off master
git checkout master
git checkout -b papers/n134_workshop/tier-1-5-final

# Restore N134 files only, from the WIP commit
git checkout "$WIP_COMMIT" -- papers/n134_workshop/
git status                     # should show only papers/n134_workshop/ paths staged
```

**Suggested commit grouping** (logical, agent may consolidate into one commit if cleaner):

1. **Tier 1 historical record** — `pre_submission_edit_spec.md` only. Commit message: `papers/n134_workshop: archive Tier 1 pre-submission edit spec`
2. **Tier 1.5 reviewer-proofing pass + Reuel→Bean cite correction** — `draft_v2_thesis_b.tex`, `references.bib`, `pre_submission_edit_spec_tier_1_5.md`. Commit message: `papers/n134_workshop: Tier 1.5 reviewer-proofing pass + Reuel→Bean cite correction`
3. **OpenReview prep** — `openreview_submission_fields.md`, `internal_memo.md`, `internal_summary.md`. Commit message: `papers/n134_workshop: OpenReview submission fields + internal strategic docs`

**If consolidating into a single commit:** acceptable; commit message: `papers/n134_workshop: Tier 1.5 reviewer-proofing pass + OpenReview submission prep`

**Push:**

```bash
git push -u origin papers/n134_workshop/tier-1-5-final
```

**Verification gate B (tarball reproducibility):**

The OpenReview tarball at `papers/n134_workshop/tmlr_main_submission_v2.tar.gz` (gitignored, intentionally — `.gitignore` line 132 covers `*.tar.gz`) was built from the working-tree state now on this branch. Verify:

- Check `papers/n134_workshop/BUILD.md` for the tarball-build script or recipe
- Regenerate the tarball from this branch's HEAD
- Diff the regenerated tarball's contents against the existing `tmlr_main_submission_v2.tar.gz` (extract both, run `diff -r` on the extracted directories)
- Source files (`.tex`, `.bib`, `.sty`, `.bst`, figures, README) must match exactly. Build artifacts (timestamps, byte-ordering of metadata) may differ; that is fine.

**If source files differ between the regenerated tarball and the staged tarball, stop and report.** That means the working-tree state and the staged tarball are not in sync, and the OpenReview upload would have a weak audit trail.

**Open PR** against master from `papers/n134_workshop/tier-1-5-final`. Description: short — "post-Tier-1.5 reviewer-proofing pass, Reuel→Bean cite correction, OpenReview submission-fields staging. The compiled tarball at `tmlr_main_submission_v2.tar.gz` (gitignored) is reproducible from this branch's HEAD; verified via tarball-diff against staged version."

---

## 5. Bucket C — program-level (new branch off master)

**Branch to create:** `program/session-2026-04-27-handoff` (off `master`).

**Files (all program-level):**

- Modified: `RESEARCH_INVENTORY.md` (+36 lines, current through 2026-04-27 second-pass review)
- Untracked: `SESSION_HANDOFF_2026_04_27.md`, `START_HERE.md`, `WORKING_TREE_CLEANUP_2026_04_28.md` (this spec), `research_review/2026-04-26.md`, `research_review/2026-04-27.md`

**Mechanics:**

```bash
# From the WIP commit on scaffold-and-pipeline:
git checkout master
git checkout -b program/session-2026-04-27-handoff
git checkout "$WIP_COMMIT" -- RESEARCH_INVENTORY.md \
                              SESSION_HANDOFF_2026_04_27.md \
                              START_HERE.md \
                              WORKING_TREE_CLEANUP_2026_04_28.md \
                              research_review/2026-04-26.md \
                              research_review/2026-04-27.md
git status                     # should show only the six paths above staged
git commit -m "program: 2026-04-27 session handoff + 2026-04-28 cleanup spec + bootstrap doc + daily reviews + inventory updates"
git push -u origin program/session-2026-04-27-handoff
```

**Open PR** against master. Description: short — "Inventory updates through 2026-04-27 second pass; daily-review reports for 2026-04-26 and 2026-04-27; session handoff document, fresh-session bootstrap doc, and 2026-04-28 working-tree cleanup spec (operational record)."

---

## 6. Final cleanup — drop the WIP commit on scaffold-and-pipeline

After Buckets B and C are pushed:

```bash
git checkout papers/benchmark_reliability_study/scaffold-and-pipeline
git reset --hard "$WIP_COMMIT^"   # drops only the WIP commit, keeps Bucket A commit
git log --oneline -2               # tip is now the Bucket A commit; WIP is gone
git status                         # working tree clean
```

**Important:** the WIP commit was a local-only commit (never pushed). `git reset --hard` is safe here because it touches no published history. If for any reason the WIP commit was pushed, **stop and report** — recovery requires `git push --force-with-lease` after confirming no other clones depend on it.

---

## 7. Sequencing

1. Bucket A: commit + push on current branch, update PR description (or open PR with §9's body).
2. WIP commit Buckets B + C together on current branch.
3. Bucket B: branch off master, restore from WIP, commit, push, verify tarball, open PR.
4. Bucket C: branch off master, restore from WIP, commit, push, open PR.
5. Drop WIP commit on current branch.

Buckets B and C can be done in either order (steps 3 and 4 are independent — they touch disjoint paths).

---

## 8. End-state verification

After all five steps:

- `git status` on `papers/benchmark_reliability_study/scaffold-and-pipeline` → **working tree clean**
- `git log --oneline master..HEAD` on that branch → 7 original commits + 1 Bucket-A commit = 8 commits
- `git log --oneline master..papers/n134_workshop/tier-1-5-final` → 1–3 commits (depending on grouping choice)
- `git log --oneline master..program/session-2026-04-27-handoff` → 1 commit
- All three branches pushed to origin
- All three PRs open against master
- Bucket B verification gate passed (tarball source matches regenerated tarball source)

**Report back:** three branch names confirmed pushed, three PR URLs, working-tree-clean confirmation on the current branch, tarball-diff result for Bucket B.

---

## 9. Updated PR description body (Bucket A's PR — supersedes prior staged version)

Paste this verbatim into the GitHub PR description for the current PR (or the new one if not yet opened):

```markdown
## Phase 1–5: scaffold, lock chain, GPU run, analysis pipeline, §1–§6 manuscript prose

This PR consolidates the benchmark-reliability study from initial scaffold through Phase 5 analysis closeout, and adds the §1–§6 manuscript prose drafted against the Phase 5 results.

## Summary

- **GPU run complete:** 624 / 624 conditions, 0 failures, ~$18 inference cost, ~74h wall-clock (with one unplanned pod restart cleanly recovered via the persistent `/workspace` volume).
- **Phase 5 analysis** ran end-to-end via `scripts/run_phase5.sh`. All 9 phases clean.
- **All four pre-registered hypotheses decided:**
  - **H1 confirmed** (5 of 5 primary benchmarks exceed ±0.005 single-occasion tolerance; median tolerance 0.21)
  - **H2 confirmed** (4 of 5 primary benchmarks have single-occasion generalizability < 0.80; arc 0.56, mmlu 0.30, truthfulqa 0.05, winogrande 0.40; hellaswag 0.95 above)
  - **H3 confirmed** (5 of 5 primary benchmarks have ≥1 model pair with reversal fraction > 0.20)
  - **H4 not confirmed** (MMLU model × subject interaction proportion 0.0046, two orders of magnitude below the 0.10 threshold; pre-registered null)
- **Reproducibility trace passes:** 18 artifacts, 0 failures. SPEC §13.2 gate cleared.
- **22 deviations tracked** (D-01 through D-22). D-21 (bootstrap CI drift) closed by deterministic-hash fix in script 07; D-22 (script-08 pivot key including model_id) closed by model-stripped cell key.
- **Sanity properties verified:** VC proportions sum to exactly 1.0 across all 15 (benchmark, model) cells; regime split active (23 g_theory + 7 parse_failure_dominated); 30/30 cells require interval reporting at single occasion.
- **§1–§6 manuscript prose** drafted against the Phase 5 results. §7–§9 remain placeholders pending the next drafting cycle.

## Key files for review

- `papers/benchmark_reliability_study/LOCK_NOTES.md` — v1 → v1.1.2 → Phase 5 completion audit chain
- `papers/benchmark_reliability_study/IMPLEMENTATION_DEVIATIONS.md` — D-01 through D-22 with full rationale
- `papers/benchmark_reliability_study/reports/reproducibility_trace.md` — section 4 deltas all 0; section 5 passes (D-21 closed by fix)
- `papers/benchmark_reliability_study/analysis/tolerance_schedules/h1_test.json` — H1 result
- `papers/benchmark_reliability_study/analysis/generalizability/h2_test.json` — H2 result
- `papers/benchmark_reliability_study/analysis/ranking_stability/h3_test.json` — H3 result
- `papers/benchmark_reliability_study/analysis/mmlu_subjects/h4_test.json` — H4 result
- `papers/benchmark_reliability_study/analysis/variance_components/model_convergence_report.csv` — cascade trace (4/5 converged at level_1; winogrande at level_3, real cascade descent)
- `papers/benchmark_reliability_study/manuscript/draft_v1.tex` — §1–§6 prose; §7–§9 placeholders
- `papers/benchmark_reliability_study/manuscript_outline_v0.md` — section structure, citation staging, Phase-5-driven outline updates
- `papers/benchmark_reliability_study/PHASE5_HANDOFF.md` — operator checklist + post-run templates
- `papers/benchmark_reliability_study/scripts/run_phase5.sh` — Phase 5 driver
- `papers/benchmark_reliability_study/scripts/11_generalizability.py` — H2 closure script (added during Phase 5 closeout)

## Reviewer checklist

- [ ] LOCK_NOTES.md audit chain (v1 → v1.1 → v1.1.1 → v1.1.2 → Phase 5 completion)
- [ ] IMPLEMENTATION_DEVIATIONS.md D-01 through D-22, with attention to D-09 (LPM-vs-GLMM regime split), D-21 resolution, and D-22 resolution
- [ ] reports/reproducibility_trace.md — confirm section 4 (per-condition recompute) reports delta = 0 across 5/5 sample conditions; confirm section 5 (bootstrap CI) passes post-D-21 fix
- [ ] H1, H2, H3, H4 hypothesis JSONs match the summary above
- [ ] manuscript/draft_v1.tex §1–§6 prose: claims align with Phase 5 numbers; cross-paper register consistency with N134 (construct-validity invocation, parallel-development register, post-hoc framing, FAMILY_B-equivalent capacity caveat)

## What this PR does NOT do

- **Manuscript §7–§9 prose drafting** (next work cycle, scoped for a follow-on PR; Phase 5 data is now in hand to drive it)
- **Appendix structure** (App A–F sketched in `manuscript_outline_v0.md`; drafting deferred to §7+ cycle)
- **N134 paper integration** (separate paper, separate branch; cross-paper §8.3 substrate-portability paragraph written against this paper's findings will land with §7+ drafting)
- **Tag `v1_1_2_PHASE5_COMPLETE`** (recommended post-merge)
```

---

## 10. Optional sanity check (recommended, ~5 min)

Before opening the Bucket A PR (or before merging it if already open), do a quick scan of `manuscript/draft_v1.tex` §1–§6 prose against the Phase 5 numbers and pre-registered hypothesis decisions:

- The prose was drafted before some Phase 5 closeout work landed (specifically: H2 closure via the new `11_generalizability.py`, H3 fix via D-22, D-21 resolution).
- If any §1–§6 paragraph references hypotheses as "pending" or "to be reported in §7" with framing that suggests the Phase 5 results were not yet known, that is normal — §7 is where outcomes get reported. Do NOT pull H1/H2/H3/H4 results into §1–§6.
- If any §1–§6 paragraph asserts a Phase 5 result that contradicts the actual outcomes (e.g., references "the H2 generalizability analysis (forthcoming)" in a way that implies a result), flag it for the §7 drafting cycle.
- If any §1–§6 paragraph names D-21 as an open caveat (e.g., "bootstrap CI drift remains an open methodological note"), that prose is now stale — D-21 is closed by fix. Note this for the §7 drafting cycle; the cross-paper register (handoff §8) no longer treats D-21 as an instance of the "discovery-like in the narrower reporting sense" register.

This is a sanity scan, not a rewrite. Drift between §1–§6 prose and Phase 5 results becomes the explicit subject of §7 drafting; it is not a blocker for the Bucket A PR.

---

## 11. After the cleanup

The next work cycle is one of:

- **N134 OpenReview upload** (Task #35, user-side, ~15 min). Tarball + supplementary tarball + `openreview_submission_fields.md` all staged. Unblocks the N134 paper into TMLR's review queue.
- **Pod teardown** (user-side). The GPU pod is idle but billing at $0.40/hr; `runs/raw/` is committed-via-tarball-equivalents (analysis CSVs + metadata JSONs are on git via Bucket A's predecessor commits); the persistent volume can be released.
- **§7–§9 manuscript drafting**. Phase 5 data in hand; recommended starting point §7.3 (tolerance schedule, the load-bearing empirical paragraph). 5–8 hours focused work to land 22–24pp manuscript.

These are not part of this cleanup spec; they're the next sessions' work.

---

## 12. If anything goes wrong

If the agent encounters any of the following, **stop and report** rather than improvising:

- Pre-flight `git status` shows files outside the expected 6 modified + 9 untracked.
- `git push` fails for any branch (likely auth issue or non-fast-forward).
- The Bucket B tarball-diff verification shows source-file differences.
- `git reset --hard` to drop the WIP commit would touch any commit that is on origin (it shouldn't — verify with `git log origin/papers/benchmark_reliability_study/scaffold-and-pipeline`).
- Any merge conflict during `git checkout` from the WIP commit — there should be none, since the target branches (`master`-based) do not contain any of the partitioned files.

---

**End of spec.**
