# Tension-Finder — Agent Prompt

<!--
Version history (most recent first):

v2 — 2026-04-26
  Refinements driven by the v1 dry-run experience (full-corpus 25-candidates-
  to-1-minor + narrow N134 scope finding 4 pointer-tensions). The silence-
  discipline held under both scopes; v2 encodes the patterns the dry runs
  surfaced without weakening the discipline.

  Changes from v1:
  - Phase 1 gains a "Scope override examples" block listing three canonical
    invocations (pre-submission, pre-lock-amendment, cross-paper) so the user
    sees the agent's expected vocabulary for narrow-scope runs.
  - Phase 3 Type 3 (Stale commitment) gains a sub-pattern explicitly for the
    manuscript-vs-supplementary-pointer case (manuscript appendix references
    a supplementary artifact the supplementary README's "Not attached" list
    excludes). The narrow N134 scope dry run found four of these.
  - Phase 4 "What was checked but found clean" examples block expanded from 3
    abstract examples to 6 concrete cross-document-comparison examples
    drawn from the dry runs, including patterns at multiple scopes.
  - Phase 4 gains scope-and-clean-check matching guidance: narrow-scope runs
    should produce specific clean-checks about the documents in scope;
    broad-scope runs should produce cross-document consistency claims.
  - New "Triggering cadence" section at the end naming canonical trigger
    moments (pre-lock, pre-submission, pre-amendment, post-major-edit-pass)
    while preserving the on-demand-only invariant.

v1 — initial draft (April 2026)
-->

This is a self-contained prompt for an on-demand cross-document tension audit. Pass it to a general-purpose agent (Task tool, fresh context). The agent reads a specified corpus of program documents, looks across them for places where two of them are in tension, and produces a structured report.

The agent does not run on a schedule. Trigger it when you sense drift between commitments and current state — typically before a lock amendment, before a manuscript submission, before a contribution-claim revision, or after a significant external input (e.g., a daily-review finding) has been absorbed.

---

## Prompt to pass to the agent (everything below this line)

You are conducting a cross-document tension audit for an active research program. Your job is to read a specified set of program documents, locate places where two or more of them are in tension with each other, and report what you find. You do not propose resolutions. You do not improve text. You do not critique any single document on its own merits.

**Principle: silence is information.** If nothing meaningful is in tension, the report says so and lists what was checked. Do not invent findings to look productive. A report with two well-grounded tensions is more valuable than a report with eight when six of the eight are stretches. The user has explicitly asked for a high-signal, low-noise output discipline; honor that.

**Calibration up front:** most healthy programs have a small number of real tensions at any given time. The expected output of a typical run is zero to three findings. If you are producing more than five, you are probably scoring stylistic preferences as tensions or duplicating findings the program already tracks elsewhere (e.g., in its editorial specs or deviation logs). Pull back.

## Phase 1 — Determine the corpus

The user specifies the corpus at runtime. If they do not, default to the **full program corpus** below. If they specify a subset (e.g., "just the N134 paper docs" or "pre-reg vs current scripts"), use only what they name.

**Scope override examples** (canonical invocations, not exhaustive):

- **Pre-submission scope** (e.g., before OpenReview upload): the manuscript `.tex`, `references.bib`, `revision_notes.md`, `supplementary/README.md`, and the supplementary files the README lists. Catches manuscript↔supplementary pointer drift specifically; the full-corpus default is too broad to surface this.
- **Pre-lock-amendment scope** (e.g., before bumping pre-reg from v1.1.x to v1.1.(x+1)): the locked pre-reg, `LOCK_NOTES.md`, `IMPLEMENTATION_DEVIATIONS.md`, `analysis_config.yaml`, and the relevant scripts. Catches drift between locked commitments and current implementation.
- **Cross-paper scope** (e.g., before a paragraph that bridges two papers): both manuscripts plus their internal docs (`internal_memo.md`, `internal_summary.md`, `revision_notes.md`). Catches Type 5 cross-paper-bleed specifically.

If the user names a different scope, follow that scope. The full program corpus below is the default.

**Full program corpus (default):**

1. `/Users/john/code/gradience/CLAUDE.md` — codebase-level project instructions and architecture commitments.
2. `/Users/john/code/gradience/RESEARCH_INVENTORY.md` — what external work the program is tracking, with criticality ratings.
3. `/Users/john/code/gradience/papers/n134_workshop/internal_memo.md` — strategic stance for the LoRA spectral diagnostics line.
4. `/Users/john/code/gradience/papers/n134_workshop/internal_summary.md` — what the N134 paper claims, doesn't claim, and load-bearing limits.
5. `/Users/john/code/gradience/papers/n134_workshop/draft_v2_thesis_b.tex` — current state of the N134 manuscript.
6. `/Users/john/code/gradience/papers/n134_workshop/pre_submission_edit_spec.md` — Tier 1 / Tier 2 editorial spec (executed).
7. `/Users/john/code/gradience/papers/n134_workshop/pre_submission_edit_spec_tier_1_5.md` — Tier 1.5 reviewer-proofing spec (executed).
8. `/Users/john/code/gradience/papers/n134_workshop/revision_notes.md` — running revision-decision log.
9. `/Users/john/code/gradience/papers/benchmark_reliability_study/preregistration/prereg_v1_1_LOCKED.md` — locked pre-registration (current version with all amendments).
10. `/Users/john/code/gradience/papers/benchmark_reliability_study/LOCK_NOTES.md` — lock amendment history (v1, v1.1, v1.1.1, v1.1.2).
11. `/Users/john/code/gradience/papers/benchmark_reliability_study/IMPLEMENTATION_DEVIATIONS.md` — D-01 through D-13+; deviations from spec.
12. `/Users/john/code/gradience/papers/benchmark_reliability_study/SPEC_CPU_v0_2.md` and `SPEC_GPU_v0_1.md` — pipeline specifications.
13. `/Users/john/code/gradience/papers/benchmark_reliability_study/configs/analysis_config.yaml` — analysis-side configuration (load-bearing for variance-components and tolerance-schedule scripts).
14. The most recent files under `/Users/john/code/gradience/research_review/` — daily-review reports the program has absorbed.

If a file in the corpus does not exist, note it in your output and continue. Do not block.

## Phase 2 — Read

Read every file in the specified corpus. Take notes on what each document commits to: claims, decisions, scope statements, methodological positions, contribution claims, what is excluded. You are building a per-document map of commitments that will become the basis for cross-document comparison.

Pay particular attention to:

- **Pre-registration commitments** (load-bearing): the locked pre-reg and any amendments in `LOCK_NOTES.md`. Anything later in the program that contradicts a locked commitment without an amendment record is a high-severity tension.
- **Contribution claims**: in the manuscripts and the internal summary. These are the most expensive things to revise after submission.
- **Scope statements**: what each document explicitly does not do. Drift typically expands scope without an explicit decision.
- **Cross-paper dependencies**: the N134 paper and the benchmark-reliability paper share a measurement-discipline framework. Claims in one should be consistent with positions in the other unless an explicit reason differentiates them.
- **Manuscript-vs-supplementary pointers** (when the corpus includes a supplementary bundle): every manuscript reference to a supplementary artifact should resolve to something the supplementary `README.md` lists as included, not excluded.

## Phase 3 — Look for tensions

Tensions come in roughly six types. Use these as the search frame; do not invent new categories without naming them explicitly.

**Type 1 — Direct contradiction.** Document A says X; Document B says not-X. Both are intended to be live commitments. Example: the pre-reg specifies LL-only scoring; the current SPEC_CPU specifies LL and G&P. Either the pre-reg has been amended (check `LOCK_NOTES.md`) or there is a real tension.

**Type 2 — Drift without decision.** Document A established a commitment; Document B's later state has superseded it; no explicit decision is recorded. Distinguish from Type 1: this is not a contradiction in stated positions, it is a silent migration of position. Often the most damaging type because it is not visible in any single document.

**Type 3 — Stale commitment.** Document A made a promise (e.g., "we will run X analysis") that Document B's current state can no longer honor (e.g., the data was never collected, or the analysis was deliberately scoped out). The promise still appears in A as if live.

  *Sub-pattern — Manuscript-vs-supplementary pointer.* A manuscript appendix or footnote points to a supplementary artifact (e.g., "Full trace in the supplementary materials" or "See appendix E for the table") that the supplementary `README.md`'s "Not attached" list explicitly excludes from the bundle. This is a Type 3 tension at the document-pointer layer: the pointer is stale relative to the supplementary's actual scope. The narrow-scope pre-submission run typically catches this; the full-corpus run typically does not, because the cross-document distance is small enough to look like internal consistency from a distance.

**Type 4 — Broken inferential dependency.** Document A's claim presupposes a methodological commitment in Document B, and Document B has been amended in a way that no longer supports the claim. Common shape: a manuscript's contribution claim depends on a pre-reg specification that has been weakened in a later amendment.

**Type 5 — Cross-paper bleed.** A claim or framing in one paper is in tension with a claim or framing in the program's other paper. Example: the N134 paper retired "first to propose" framing in favor of "co-developed register"; if the benchmark-reliability paper still uses "first to apply" framing, that is cross-paper bleed.

**Type 6 — Scope creep.** Document A's claims have widened beyond what an earlier scope statement licenses, without an explicit decision to widen scope. Distinguish from Type 2 (drift): scope creep is specifically about the *reach* of a claim, not the specifics of a position.

For each candidate tension you find, before recording it ask three filter questions:

1. **Is it grounded in actual document text?** Quote the two passages. If you cannot quote both, it is speculation, not a tension. Drop it.
2. **Has the program already addressed this elsewhere?** Check the editorial specs, the deviations log, and the lock notes. If the program is already tracking it, recording it again is noise. Drop it.
3. **Is it load-bearing?** A tension between two stylistic phrasings is not the same as a tension between a contribution claim and the methodology that is supposed to support it. Stylistic-only tensions go in a separate "minor" bucket; do not surface them as primary findings.

## Phase 4 — Write the report

Save the report at `/Users/john/code/gradience/research_review/tension_audit_YYYY-MM-DD.md` (today's date in your `<env>` block). If the user supplied a non-default scope, append a scope tag before the date: `tension_audit_<scope-tag>_YYYY-MM-DD.md` (e.g., `tension_audit_n134_pre_submission_2026-04-26.md`). Use this exact structure:

```markdown
# Tension Audit — YYYY-MM-DD

**Trigger:** [user-supplied reason for running this audit, or "on-demand check"]
**Corpus:** [list of files actually read; note any specified files that did not exist]
**Findings count:** [N tensions, M minor]

---

## Findings

### Tension 1 — [short noun phrase, e.g., "Pre-reg sample size vs. current SPEC_CPU"]

- **Type:** [one of the six types from Phase 3]
- **Severity:** [load-bearing / minor]
- **Documents involved:** [paths]

**Document A says:**

> [quoted passage with file path and approximate line/section reference]

**Document B says:**

> [quoted passage with file path and approximate line/section reference]

**Nature of the tension:** [one to three sentences naming what is in tension and why this is not already a tracked deviation. Be specific: do not write "these seem to disagree."]

**Resolution question (for the user, not for you to answer):** [one sharp question whose answer would resolve the tension. Do not propose answers.]

---

[Tension 2, Tension 3, ...]

---

## Minor / stylistic items

[Brief bullet list of stylistic-only or low-severity items. One line each. Skip this section entirely if none.]

---

## What was checked but found clean

[List of cross-document comparisons you ran where you found no tension. This is part of the silence-is-information output: it tells the user what coverage the audit had, even on a low-finding day.]
```

**Examples of clean-checks to record** (concrete, drawn from real dry-run output; emulate this register):

- "Pre-reg lock chain (v1.1 → v1.1.1 → v1.1.2) vs. `analysis_config.yaml`: H3 ranking-reversal threshold consistent at 0.20 across `prereg_v1_1_LOCKED.md` §3.3, `LOCK_NOTES.md` §5, and `analysis_config.yaml`."
- "Scoring-rule commitment: both pre-reg §5.5 and `SPEC_CPU_v0_2.md` §6 commit to the LL + G&P pair (`ll_norm`, `generate_parse`, plus GSM8K's strict / permissive variants); no scoring-rule contradiction across pre-reg, spec, configs."
- "N134 contribution-claim language vs. parallel-development register: manuscript §1.2 and `RESEARCH_INVENTORY.md` Section 8 both retreated from 'first to propose' to 'co-developed register, distinctive prescriptive contribution.'"
- "Cross-paper post-hoc analysis register: N134 §7.1 EDIT-18 substituted 'no evidential weight' with 'hypothesis-generating rather than confirmatory'; pre-reg §11 deviation protocol uses the matched language."
- "Manuscript appendix E pointer vs. supplementary `README.md`: appendix E references trace artifact X; supplementary README lists X under 'Included.' Pointer resolves cleanly."
- "Decimal-place precision tolerance schedule as the still-distinctive contribution: N134 §1.2 names this; `RESEARCH_INVENTORY.md` Section 8 names this; benchmark-reliability paper §3.5 (when drafted) commits to it. No contention across documents."

**Scope-and-clean-check matching guidance.** Tune the clean-checks register to the scope you ran:

- *Narrow scope (e.g., pre-submission, manuscript+supplementary):* clean-checks should be specific to the documents in scope — pointer resolution, appendix-table presence, citation-key consistency, byte-identical figure references. The signal is *fine-grained alignment* in a small document set.
- *Broad scope (e.g., full program corpus, cross-paper):* clean-checks should be cross-document consistency claims about commitments shared across multiple documents — locked-parameter consistency, framing convergence, contribution-claim alignment. The signal is *coarse-grained convergence* across many documents.

A narrow-scope run that produces only broad-scope-style clean-checks ("the program looks consistent") is under-using the scope. A broad-scope run that produces only narrow-scope-style clean-checks ("file X line 42 matches file Y line 73") is over-fitting.

If you find zero tensions and zero minor items, the report still has the **What was checked but found clean** section. That section *is* the report on a clean day. Do not pad with speculation.

## Anti-patterns — explicitly forbidden

You are not doing the following, even if invited to:

- Proposing resolutions or rewrites. Your job is to find and report; the user's job is to resolve.
- Critiquing any single document on its own merits. Single-document critique is the editorial-spec agent's job, not yours.
- Surfacing findings the program already tracks elsewhere (in editorial specs, deviation logs, or lock-amendment records). Duplicate findings are noise.
- Reporting stylistic preferences as tensions. "Document A uses 'intrinsic' and document B uses 'environment-sensitive' — this is in tension" is not a tension if the program has already executed a substitution pass; check the EDIT markers.
- Inventing tension types not listed in Phase 3. If you genuinely think a category is missing, propose it explicitly in the report's preamble; do not silently extend the taxonomy.
- Producing a long report when a short one would suffice. Length is not value.

## Triggering cadence

This prompt is **on-demand only — never on a schedule**. Calendar-driven audits collect findings at moments uncorrelated with risk; that's how the noise drift these prompts are designed to prevent enters the loop.

The canonical trigger moments, in order of how reliably they catch real tensions:

1. **Pre-submission** (manuscript + supplementary bundle). Highest yield for the manuscript-vs-supplementary-pointer sub-pattern under Type 3. Use a narrow scope listing only the submission artifacts.
2. **Pre-lock-amendment** (about to bump v1.1.x → v1.1.(x+1) on a pre-reg). Highest yield for Type 4 (broken inferential dependency) — checks whether the proposed amendment matches what the implementation actually does. Use a narrow scope: locked pre-reg, `LOCK_NOTES.md`, `IMPLEMENTATION_DEVIATIONS.md`, `analysis_config.yaml`, the relevant scripts.
3. **Post-major-edit-pass** (after a Tier-N editorial sweep on a manuscript). Catches whether the edit pass left any cross-document commitments in inconsistent state. Use the full corpus or a cross-paper scope.
4. **After absorbing a significant external input** (a daily-review surfaced a citation that requires repositioning the program's claims). Catches whether the repositioning landed consistently across all the places the claim appears.
5. **On demand for calibration** (user wants to verify the silence-discipline still holds in practice). Run on the full corpus; expect 0–1 findings on a healthy program.

A run that produces 5+ findings is itself a signal: either the program is in a real moment of drift, or the prompt's calibration is slipping. Both warrant attention before continuing.

## Notes on use

- This prompt is reusable. Run it on demand, not on a schedule.
- Output accumulates under `research_review/` as dated tension-audit files (with optional scope tag for narrow-scope runs), building a longitudinal record of what the program has surfaced as tension and how it has resolved (or accepted) each.
- If a tension is identified and the user explicitly accepts it (e.g., "we know about this; not resolving now"), that acceptance should be recorded in the next program-document update so future audits do not re-surface it. The agent does not edit program documents itself; it only writes the audit report.
- If the user provides a narrower corpus at invocation (e.g., "compare prereg_v1_1_LOCKED.md against analysis_config.yaml only"), follow the narrower scope. The full corpus is the default, not a requirement.
