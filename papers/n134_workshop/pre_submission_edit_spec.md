# Pre-Submission Editorial Pass — Edit Spec

**Context.** External editorial review (2026-04-23) surfaced a set of revision suggestions for `draft_v2_thesis_b.tex`. This document triages those suggestions, specifies the concrete edits worth executing before OpenReview submission, and provides line-targeted before/after text for each.

**Scope.** Editorial revisions to the main manuscript only. Does not cover A4 packaging (MacTeX verification, tarball), supplementary bundle (already tagged `v2-anonymized-supplementary`), or any template-port matters.

**Goal.** Strengthen the paper's measurement-discipline spine at every level (abstract, intro, main text, conclusion) and reduce the surface area for predictable reviewer attacks. All Tier 1 edits are independently defensible on their own merits; the cumulative effect is that the paper is read as a normative-methodological paper with a worked empirical example rather than as a spectral-merging paper with a methodological preface.

---

## Marker convention

A new comment-marker convention for this pass, parallel to the existing `% ANON:` (anonymization) and `% TMLR:` (template-port) conventions:

```
% EDIT: 2026-04-23 — <what changed, why, which editorial suggestion #>
```

Place immediately above each edited region. Rules:

- Use `% EDIT:` only for substantive content changes (claim-bearing or framing-bearing). Word-level polish without a claim-shift doesn't need a marker; git diff is sufficient audit for those.
- When an edit touches an already-ANON'd region, extend the existing `% ANON:` block with an additional line noting the EDIT on top; don't replace the ANON comment.
- When an edit touches a pre-existing paragraph with no prior marker, add a fresh `% EDIT:` block.
- `grep '% EDIT:' draft_v2_thesis_b.tex` should enumerate exactly the lines touched in this pass.
- `% EDIT:` comments are retained at camera-ready (they're not anonymization artifacts) but may be stripped post-acceptance if desired — they serve as an audit trail for the pre-submission revision, not for camera-ready restoration.

---

## Priority tiers overview

**Tier 1 (must-do pre-submission, 5 edits, ~2 hours total):**

| ID | Target | Summary | Cost |
|----|--------|---------|------|
| EDIT-01 | Abstract + §6 title + Contribution claim (iii) | Temper "previously unnamed" language | ~20 min |
| EDIT-02 | §1.1 around line 116 | Defang Zhou et al. critique (illustrative, not accusatory) | ~15 min |
| EDIT-03 | End of §1.4 (Road map) or end of §1.3 (Contribution claims) | Hierarchy sentence: case study is stress test, not vindication | ~15 min |
| EDIT-04 | §5.4 around line 687 | Regime-specific null, not broad "binding constraint" | ~15 min |
| EDIT-05 | §4.2 around line 473 | Surface same-task-vs-cross-task reliability caveat from Appendix D | ~30 min |

**Tier 2 (high value if time allows, ~4–5 edits, 2–4 hours):**

| ID | Target | Summary | Cost |
|----|--------|---------|------|
| EDIT-06 | Abstract | Tighten around three outputs (framework, null, precision pathology) | 1–2 hrs |
| EDIT-07 | End of §1.3 | Concrete reader-payoff preview (three scientific changes) | 30 min |
| EDIT-08 | §1.2 or §2 opening | Conciliatory sentence about ML statistical sophistication | 10 min |
| EDIT-09 | §5.3 | Task-boundary replication reframed as construct-scope refinement | 1–2 hrs |
| EDIT-10 | §8 (Conclusion) | Concrete norm-change close | 30 min |

**Tier 3 (defer to revision round or skip):**

| ID | Target | Disposition |
|----|--------|-------------|
| EDIT-11 | §2 compact framework table | Defer unless revision round requests; current §2 prose adequately itemizes the four components |
| EDIT-12 | Selective manifesto-phrase softening | Do surgically, not globally; preserve the voice sentences that do philosophical work |

---

# Tier 1 edits (concrete)

## EDIT-01 — Temper "previously unnamed" language

**Editorial suggestion:** #5 (temper novelty claim); also affects #3 (abstract) and #6 section title.

**Rationale.** The phrase "previously unnamed" appears in three load-bearing locations: the abstract (line 89), the third contribution claim (line 213), and the §6 section title (line 694). A skeptical reviewer from a numerical-methods background can cite decades of prior work on rank-correlation near-tie instability and floating-point nondeterminism — the mathematical phenomenon is not novel, and "previously unnamed" invites an avoidable reviewer attack. The correctly defensible claim is not "this is a new mathematical phenomenon" but "this is a reporting-level measurement constraint that ML diagnostic practice does not currently treat as such." The fix is to move the novelty claim from the phenomenon to the reporting-register framing.

### EDIT-01a — Abstract (line 88–91)

**Current:**
```
being subjected to measurement-discipline scrutiny --- surfaces a
previously unnamed property of the primary test statistic: rank-based
correlations on small-$N$ residuals have intrinsic floating-point
precision of order $10^{-2}$, comparable to their sampling variability.
```

**Proposed:**
```
% EDIT: 2026-04-23 — editorial #5: "previously unnamed property" →
% EDIT: "underreported precision limitation" to move novelty claim from
% EDIT: phenomenon (mathematically old) to reporting register (new).
being subjected to measurement-discipline scrutiny --- surfaces an
underreported precision limitation of the primary test statistic:
rank-based correlations on small-$N$ residuals have intrinsic
floating-point precision of order $10^{-2}$, comparable to their
sampling variability.
```

### EDIT-01b — §1.3 Contribution claim (iii), line 213

**Current:**
```
  \item A specific previously-unnamed measurement property of partial
    Spearman correlations on small-$N$ residuals, surfaced by the
    measurement-discipline scrutiny the framework prescribes, with
    implications for any rank-based diagnostic metric reported at
    small sample sizes.
```

**Proposed:**
```
% EDIT: 2026-04-23 — editorial #5: contribution claim iii reframed
% EDIT: from phenomenon-novelty to reporting-register novelty.
  \item Identification of an underreported precision limitation of
    partial Spearman correlations on small-$N$ residuals as a
    reporting-level measurement constraint: the floating-point
    precision of rank-based diagnostic correlations at these sample
    sizes is comparable to their sampling variability, a fact that
    measurement-discipline scrutiny surfaces and that ordinary
    reporting conventions do not.
```

### EDIT-01c — §6 section title, line 694

**Current:**
```
\section{A Previously-Unnamed Measurement Property}
```

**Proposed:**
```
% EDIT: 2026-04-23 — editorial #5: section title reframes novelty claim
% EDIT: from phenomenon to reporting-register constraint.
\section{A Reporting-Level Precision Constraint for Rank-Based Residual Correlations}
```

(Also check the `\label{sec:rank-observation}` reference on this section and confirm no downstream `\ref{}` calls rely on the title's exact wording. The label stays the same; only the rendered title changes.)

### EDIT-01d — Opening sentence of §6 (check after title change)

The §6 body text should also be checked for any "this previously unnamed..." or "the previously unnamed..." sentence-level references at the section opening. If present, substitute `previously unnamed` → `under-reported precision` or similar.

---

## EDIT-02 — Defang Zhou et al. critique

**Editorial suggestion:** #8 (illustrative, not accusatory).

**Rationale.** The paper's §1.1 anchors the reporting-gap argument on a specific named critique of Zhou et al. (2026) — naming the paper, the reported correlation value, and the missing measurement infrastructure. This is a known reviewer-poisoning vector: a reviewer who is Zhou, collaborates with Zhou, or has cited Zhou approvingly may experience the critique as hostile and score the paper accordingly. The fix is to add one clarifying sentence that frames the example as illustrative of field-wide reporting norms, not as a criticism of the paper's execution.

### EDIT-02 — §1.1, inserted after line 135 (after the `...rather than as the central claim of an instrument whose reliability, precision, and confound structure have been independently characterized.` sentence)

**Current (lines 129–136):**
```
The point is not that $0.572$ is wrong ---
it may well be the best available estimate from the reported
experiment --- but that the reporting convention treats a
point-estimate correlation as a self-standing measurement rather
than as the central claim of an instrument whose reliability,
precision, and confound structure have been independently
characterized.
```

**Proposed (extend with one sentence):**
```
% EDIT: 2026-04-23 — editorial #8: added illustrative-not-accusatory
% EDIT: clarifier to reduce reviewer attack surface on a specific
% EDIT: named-paper critique (reviewer-poisoning risk).
The point is not that $0.572$ is wrong ---
it may well be the best available estimate from the reported
experiment --- but that the reporting convention treats a
point-estimate correlation as a self-standing measurement rather
than as the central claim of an instrument whose reliability,
precision, and confound structure have been independently
characterized. We use this example not as a critique of the
paper's execution under current norms but as an illustration of
the norms themselves: the field currently permits a predictive
correlation to function as the primary evidential object without
requiring the accompanying measurement infrastructure that would
calibrate its interpretation.
```

---

## EDIT-03 — Hierarchy sentence at end of intro

**Editorial suggestion:** #2 (paper hierarchy made explicit).

**Rationale.** The single highest-leverage framing edit in the paper. A meaningful fraction of reviewers will read the paper as a spectral-merging paper with a methodological preface, which makes the null an embarrassment instead of a demonstration. Explicit statement that the case study is a stress test of the framework, not a vindication of the diagnostic, preempts the misreading. Best placement is at the end of §1.3 (Contribution claims) or the end of §1.4 (Road map), where the reader has just seen the contribution hierarchy and is about to enter the body of the paper.

### EDIT-03 — End of §1.4 Road map, inserted after line 229

**Current (lines 220–230):**
```
\subsection{Road map}

Section~\ref{sec:framework} develops the framework.
Section~\ref{sec:case-study} introduces the worked example's setting.
Sections~\ref{sec:applying}--\ref{sec:rank-observation} present the
worked example. Section~\ref{sec:generalizing} returns to the
framework and generalizes from the case study to broader reporting
practice. Section~\ref{sec:objections} addresses limitations and
common objections to the thesis. Section~\ref{sec:conclusion}
concludes.
```

**Proposed (add a concluding paragraph):**
```
\subsection{Road map}

Section~\ref{sec:framework} develops the framework.
Section~\ref{sec:case-study} introduces the worked example's setting.
Sections~\ref{sec:applying}--\ref{sec:rank-observation} present the
worked example. Section~\ref{sec:generalizing} returns to the
framework and generalizes from the case study to broader reporting
practice. Section~\ref{sec:objections} addresses limitations and
common objections to the thesis. Section~\ref{sec:conclusion}
concludes.

% EDIT: 2026-04-23 — editorial #2: explicit framework-vs-case-study
% EDIT: hierarchy to prevent the paper being read as a spectral-merging
% EDIT: paper with a methodological preface.
The empirical case study is not offered as evidence that spectral
diagnostics solve LoRA-merge prediction. It is offered as a stress
test of the framework under deliberately unfavorable conditions: a
pre-registered diagnostic fails its primary prediction, a strong
confound explains most outcome variance, and the headline statistic
proves less numerically precise than ordinary reporting conventions
would imply. The case is useful precisely because it demonstrates
what measurement discipline prevents authors from claiming; the
framework succeeds to the extent that the case study produces these
refusals rather than rescues.
```

---

## EDIT-04 — Regime-specific humility on §5.4 binding constraint

**Editorial suggestion:** #10 (four-method comparison humility).

**Rationale.** The sentence at lines 687–690 already contains a hedge ("one plausible interpretation, not a definitive claim"), but the framing — "weight-space spectral geometry itself... appears to be the binding constraint" — still generalizes from 45 pairs, one backbone, one rank, one merge operation to a claim about the entire class of weight-space spectral diagnostics. A skeptical reviewer will correctly observe that "binding constraint" overstates the scope. The fix is to frame the finding as a regime-specific null: within the tested regime, changing the spectral triage score did not rescue prediction, but this is evidence about the regime, not about the class of methods.

### EDIT-04 — §5.4, lines 687–691

**Current:**
```
The framework reading: weight-space spectral geometry itself,
rather than any specific algorithmic choice within it, appears to
be the binding constraint on per-pair prediction at this sample
size --- one plausible interpretation, not a definitive claim.
Activation-informed methods
```

**Proposed:**
```
% EDIT: 2026-04-23 — editorial #10: reframe broad "binding constraint"
% EDIT: claim as regime-specific null; evidence is about the tested
% EDIT: regime, not about the class of weight-space spectral methods.
The framework reading: within the tested regime — 45 cross-task pairs
on a single decoder backbone at a single rank under the specified
merge operation — changing the weight-space spectral triage score did
not rescue per-pair prediction. The appropriate interpretation is a
regime-specific null for this class of static weight-space diagnostics,
not evidence against activation-informed, behavioral, or learned
mergeability predictors.
Activation-informed methods
```

---

## EDIT-05 — Main-text reliability regime-scope caveat

**Editorial suggestion:** #11 (bring Appendix D caveat into main text).

**Rationale.** The same-task-vs-cross-task regime-scope limitation on the reliability estimate is one of the paper's clearest instances of measurement-discipline refusal: the ICC is valid only for the regime that generated it; the primary H1 test operates on cross-task pairs, for which no direct reliability estimate exists. This caveat embodies the paper's thesis. It currently lives in Appendix D and is only glancingly referenced in the main-text §4.2 (the main text mentions "same-task adapter pairs" provide replicate structure but doesn't flag the scope limitation on downstream use). Surfacing it explicitly in the main text strengthens the spine.

### EDIT-05 — §4.2, inserted after line 477 (end of current paragraph)

**Current (lines 464–478):**
```
\subsection{Reliability considerations at pre-registration time}

% ANON: "The N134 design" → "The present study's design"
The present study's design commits to three seeds per task across eight
pre-registered tasks, yielding 24 adapters. This design supports
cross-seed reliability estimation for $S_{\mathrm{H1}}$ as an
instrument: same-task adapter pairs (three same-task pairs per task,
24 total) provide the replicate structure that a cross-seed ICC
estimate requires. The resulting estimate,
$\hat{\rho}_{\mathrm{ICC}} = 0.566$ with SEM $= 0.014$, is
reported at Appendix~\ref{app:reliability}; for the purpose of the
pre-registered H1 test, the relevant observation is that the design
was calibrated to support this estimate before data collection began,
not that the estimate was examined post-hoc to reassure reviewers.
```

**Proposed (extend with a follow-up paragraph):**
```
\subsection{Reliability considerations at pre-registration time}

% ANON: "The N134 design" → "The present study's design"
The present study's design commits to three seeds per task across eight
pre-registered tasks, yielding 24 adapters. This design supports
cross-seed reliability estimation for $S_{\mathrm{H1}}$ as an
instrument: same-task adapter pairs (three same-task pairs per task,
24 total) provide the replicate structure that a cross-seed ICC
estimate requires. The resulting estimate,
$\hat{\rho}_{\mathrm{ICC}} = 0.566$ with SEM $= 0.014$, is
reported at Appendix~\ref{app:reliability}; for the purpose of the
pre-registered H1 test, the relevant observation is that the design
was calibrated to support this estimate before data collection began,
not that the estimate was examined post-hoc to reassure reviewers.

% EDIT: 2026-04-23 — editorial #11: surface same-task-vs-cross-task
% EDIT: regime-scope caveat from Appendix D into main text — this
% EDIT: refusal-to-overgeneralize is the paper's thesis in compressed form.
The reliability estimate validates only the same-task regime that
supplies its replicate structure. The H1 test concerns cross-task
pairs, for which the present design contains no direct reliability
estimate. We therefore report the same-task ICC as an instrument-
validation constraint, not as a universal reliability coefficient
for $S_{\mathrm{H1}}$: the estimate licenses the claim that the score
is moderately stable when the same task is scored under different
seeds; it does not license the claim that the score is moderately
stable across tasks. Naming what the coefficient does not cover is
itself a framework prescription.
```

(The phrase "Naming what the coefficient does not cover is itself a framework prescription" is the journal-prose variant of the manifesto-style line in Appendix D that EDIT-12 flags for selective softening. This placement consolidates the thesis-compressed sentence in the main text rather than repeating it in two registers.)

---

# Tier 2 edits (wording proposals, no full before-text)

## EDIT-06 — Abstract tightening (editorial #3)

**Current abstract** is dense; condenses five threads (psychometrics analogy, LoRA case, informative null, task-boundary replication, precision pathology) into one long sentence. Proposed structure: (1) framework claim; (2) null under confound control; (3) precision pathology; (4) meta-claim about what the framework produces. Drop or subordinate the three-architectures-two-metric-families result (keep it in §5.3 only).

**Candidate tighter closing (replacing lines 82–95):**

```
We demonstrate the framework through a pre-registered decoder-scale
study of spectral LoRA-merging diagnostics ($N = 45$ cross-task
adapter pairs on Mistral-7B-v0.3). Under the framework's discipline,
the case study does three things: it produces a clean pre-registered
null on per-pair merge prediction under family-confound control; it
exposes that the headline statistic's floating-point precision at
this sample size is of order $10^{-2}$, comparable to its sampling
variability, so ordinary third-decimal reporting is spurious; and it
bounds the diagnostic's reliability with an explicit regime-scope
caveat. The case study illustrates the paper's central claim:
measurement discipline applied to ML diagnostics changes what can
responsibly be claimed — not by decorating point estimates with error
bars, but by forcing refusals the default reporting view would not.
```

Execute after Tier 1 is in place, because EDIT-01 (underreported-precision reframe) affects the abstract's phrasing and EDIT-03 (hierarchy) makes the abstract's spine-language more natural.

## EDIT-07 — Reader payoff preview at end of §1.3 (editorial #4)

**Placement:** after the enumerate ending on line 218 (before §1.4 Road map, line 220).

**Proposed paragraph:**

```
% EDIT: 2026-04-23 — editorial #4: concrete preview of what the
% EDIT: framework changes in the worked example, to orient the reader
% EDIT: before the body of the paper.
In the worked example, the framework changes the scientific
interpretation in three specific ways. A diagnostic relationship that
an unstructured presentation would license as informative becomes a
clean null once family-pair identity is residualized out
(Section~\ref{sec:empirical-h1}). A reliability estimate that a
default report would quote as a universal property of the instrument
is shown to license only the regime that generated it
(Section~\ref{sec:applying-reliability}, Appendix~\ref{app:reliability}).
A headline statistic that default reporting would quote to three
decimal places is shown to have intrinsic numerical precision at its
second decimal, turning the third digit into an unsupported claim
(Section~\ref{sec:rank-observation}). These are not cosmetic
additions to the report; they alter the claims the study is allowed
to make.
```

## EDIT-08 — Conciliatory sentence placed earlier (editorial #6)

**Placement:** in §1.2 (The psychometric analog), inserted after line 194 (end of the paragraph ending "systematic expectation that every reported diagnostic value carries this measurement-theoretic context."). The existing text (lines 195–198) already disclaims conceptual originality; the conciliatory sentence should sit alongside.

**Proposed sentence:**

```
% EDIT: 2026-04-23 — editorial #6: conciliatory framing — ML has
% EDIT: statistical sophistication; the gap is specifically an
% EDIT: integrated measurement argument, not statistical naivety.
The claim this paper makes is narrower than a comparison of
statistical sophistication between fields: ML diagnostic reporting
often lacks an integrated measurement argument that ties score
meaning, reliability, precision, and confound structure to the
substantive interpretation of the reported number. Component
methods are present; their coordinated application is not.
```

## EDIT-09 — §5.3 task-boundary replication as construct-scope refinement (editorial #9)

**Current framing** (lines 640–650) ends with "what replicates is the qualitative structure — same > cross with zero overlap — not any specific numerical scale. The framework's construct-validity discipline is what enforces this restraint." This is already partially doing the editor's suggested reframing, but it doesn't explicitly state the construct-scope-narrowing move.

**Proposed addition** (append after line 650, before the end of §5.3):

```
% EDIT: 2026-04-23 — editorial #9: explicit construct-scope narrowing
% EDIT: frame so the task-boundary result serves the measurement-
% EDIT: discipline thesis rather than living alongside the per-pair null.
Framework-wise, this is a construct-scope refinement rather than a
standalone positive result. The diagnostic family's construct
validity is narrowed by the joint pattern across §\ref{sec:empirical-h1}
and this section: the spectral score is informative for population-
level task-boundary structure, but not for per-pair merge-risk
prediction under the tested regime. A report that quoted only the
task-boundary replication as evidence for the diagnostic would be
operating on a broader construct claim than the evidence warrants;
a report that quoted only the per-pair null would miss a real
regularity at a different level of aggregation. The framework's
discipline is to license each claim at its proper scope.
```

## EDIT-10 — Concrete norm-change close in conclusion (editorial #13)

**Placement:** replace or extend the paragraph at lines 918–926 (the "Three directions follow" paragraph) with a concrete prescriptive close.

**Proposed conclusion ending (replacing lines 918–926 or appended after them):**

```
% EDIT: 2026-04-23 — editorial #13: concrete norm-change close; turns
% EDIT: the paper from observation to prescription.
A minimal measurement-disciplined diagnostic report, in the sense
this paper argues for, should include: a construct statement that
distinguishes the theoretical object from its operationalization; a
reliability estimate matched to the regime of intended use; a
precision tolerance schedule for the reported quantities; and a
pre-registered decomposition against obvious alternatives. A
diagnostic that cannot yet supply these elements may still be useful,
but its evidential status should be described as exploratory rather
than quantitative. The purpose of the framework is not to slow ML
diagnostics down. It is to make their claims durable enough to
matter — so that a reported value, once published, can be read by
the field as a bounded measurement rather than as a decimal to be
trusted by convention.
```

---

# Tier 3 (defer or skip)

## EDIT-11 — §2 compact framework table (editorial #7)

**Disposition: defer.** The current §2 (lines 231–409) already itemizes the four framework components with subsection headers (`What the score indicates`, `How stable the indication is`, `What precision the indication supports`, `What else could explain the signal`). The hierarchy is legible in the section structure. A compact table would aid reviewer memory and citability, but the marginal benefit over the existing structure is smaller than the cost of wordsmithing table cells to match the registers used in §4 and §5. If the revision round requests a summary table, it's a natural revision-cycle addition; if not, skip.

## EDIT-12 — Selective manifesto-phrase softening (editorial #12)

**Disposition: selective surgical.** The editor's suggested phrase softenings are not globally correct; some of the voice sentences do philosophical work the khaki variants lose. Recommended surgical edits:

- **Line 1062 ("Naming what the estimate does not license is itself the framework move")**: if EDIT-05 is executed as written, this sentence becomes redundant with the new main-text reliability caveat. Consider deleting the Appendix D sentence entirely, or retaining as "Explicitly bounding the coefficient's regime of validity is part of the proposed reporting discipline." Execute after EDIT-05.
- **Lines 766–767 ("the third decimal would have been an unstated promise the data could not keep")**: **keep.** This sentence does philosophical and rhetorical work — it frames precision overclaims in moral-epistemic terms that the rest of the paper's register rests on. The editor's khaki variant ("implied a level of reproducibility not supported by the statistic under the observed numerical conditions") is more neutral but less memorable and drops the moral weight. The paper's voice here is part of what makes it distinctive.
- **"Framework move" / "discipline in action" elsewhere**: check for occurrences via `grep -n "framework move\|discipline in action\|central claim"`. Soften only where the phrase reads as internal jargon rather than as a pointer to a named concept.

---

# Process

## Order of operations

1. **Tier 1 in sequence** — EDIT-01 through EDIT-05. Order matters slightly: EDIT-01 affects abstract/title/contribution claim wording that later edits may reference. EDIT-05 establishes main-text reliability language that affects EDIT-12's Appendix D decision.
2. **Compile checkpoint** — after Tier 1, run the full four-pass MacTeX compile with microtype enabled (per Plan 1 from the A4 spec). Verify: page count (target: still 16, acceptable: 15 or 17); zero undefined citations; title block renders correctly; §6 title renders with its new wording.
3. **Tier 2 if time permits** — EDIT-06 through EDIT-10. EDIT-06 (abstract) should execute after Tier 1 because Tier 1 provides the language; other Tier 2 edits are independent.
4. **Compile checkpoint after Tier 2** — same checks; page count may drift to 17 or 18 after Tier 2 inserts; if 18, consider whether any Tier 2 edit is cuttable or whether a small trim elsewhere reclaims a page.
5. **Commit separately** — one commit for Tier 1, a second for Tier 2 if executed. Commit message: `papers/n134_workshop: pre-submission editorial pass (Tier N)`. Do not overwrite the `v2-anonymized-supplementary` tag; the editorial pass is a downstream commit on the same branch.

## Verification commands after each tier

```bash
cd /Users/john/code/gradience/papers/n134_workshop/

# Enumerate edits made
grep -n '% EDIT:' draft_v2_thesis_b.tex

# Verify existing conventions intact
grep -c '% ANON:' draft_v2_thesis_b.tex   # should be >= pre-pass count
grep -c '% TMLR:' draft_v2_thesis_b.tex   # should equal pre-pass count (unchanged)

# Compile
rm -f *.aux *.bbl *.blg *.log *.out *.toc
pdflatex -interaction=nonstopmode draft_v2_thesis_b.tex
bibtex draft_v2_thesis_b
pdflatex -interaction=nonstopmode draft_v2_thesis_b.tex
pdflatex -interaction=nonstopmode draft_v2_thesis_b.tex

# Page count and warnings
pdfinfo draft_v2_thesis_b.pdf | grep Pages
grep -E "(undefined|Undefined|^\!)" draft_v2_thesis_b.log
```

## Integration with existing marker conventions

After the editorial pass, the source will carry four independent marker conventions:

- `% ANON:` — anonymization edits, restore at camera-ready
- `% TMLR:` — template-port edits, retain at camera-ready (revert only if redirecting)
- `% EDIT:` — pre-submission editorial pass edits, retain at camera-ready (markers may be stripped if desired)
- `% TODO:` — pre-existing author working notes (e.g., the §1.1 second-example TODO at line 138); not a convention of this pass, but present in the source

All four are independently grepable. `grep '% ANON:\|% TMLR:\|% EDIT:\|% TODO:' draft_v2_thesis_b.tex` enumerates all marked regions for a full audit.

---

# Time budget

| Phase | Budget | Cumulative |
|-------|--------|------------|
| Tier 1 (5 edits) | 2 hrs | 2 hrs |
| Compile checkpoint + log review | 30 min | 2.5 hrs |
| Tier 2 (if executed, 5 edits) | 2–4 hrs | 4.5–6.5 hrs |
| Compile checkpoint #2 | 30 min | 5–7 hrs |
| Commit + tag | 10 min | 5.2–7.2 hrs |

**Minimum defensible pass (Tier 1 only):** ~2.5 hrs.
**Recommended pass (Tier 1 + selective Tier 2):** ~4 hrs.
**Full pass (Tier 1 + Tier 2 + selective Tier 3):** ~6–7 hrs.

Decision gate remains 2026-04-28 (TMLR start vs. NeurIPS pivot). Executing Tier 1 is compatible with any A4 timeline; executing Tier 2 requires a full day's work and should be committed to only if the decision gate is not at risk.

---

# Notes on editorial suggestions not adopted

For completeness, the editorial suggestions not producing edits above:

- **Suggestion #1** — praise, no action.
- **Suggestion #7 (framework table)** — deferred per EDIT-11 disposition.
- **Suggestion #14** — the editor's own priority list; substantially consistent with this spec's Tier 1/Tier 2 split.

Any future editorial suggestion that revises these dispositions should be added to a new revision of this file rather than edited in place, so the reasoning is preserved across revisions.
