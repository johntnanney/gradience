# Pre-Submission Editorial Pass — Tier 1.5 Edit Spec (Reviewer-Proofing)

**Context.** Second external editorial review (2026-04-23) of the post-Tier-1/Tier-2 draft. The editor's own assessment: "this is now in the zone where the remaining work is not identity-level revision. It is reviewer-proofing." This spec enumerates the concrete edits that reviewer-proofing pass warrants.

**Relationship to prior spec.** Extends `pre_submission_edit_spec.md`. All edits in this file continue the `% EDIT: 2026-04-23` marker convention introduced there. Edit IDs continue from EDIT-12 (the last ID used in the prior spec), so there is no numbering collision.

**Critical difference from the prior pass.** One of these edits (EDIT-13, the §6.2 Spearman formula) is not a "would be better" language edit but a potential technical-correctness issue. It is the first edit that must land before submission regardless of time budget.

---

## Priority tiers overview

**Tier 1.5-A (must-do, non-negotiable):**

| ID | Target | Summary | Cost |
|----|--------|---------|------|
| EDIT-13 | §6.2 around line 819 | Spearman per-flip coefficient retreat — technical correctness | ~20 min + verification |
| EDIT-14 | §1.3 line 261, §2.3 lines 423/429/435/439, §9 line 1018 | Replace "intrinsic numerical precision" with regime-specific phrasing | ~20 min |
| EDIT-15 | §6.3 lines 838–839 | Soften n=200 stabilization sentence | ~10 min |
| EDIT-16 | §6.4 lines 875–877 | Soften "discovery" language | ~10 min |
| EDIT-17 | §5.2 around line 662 + §7 limitations | FAMILY_B capacity clarification (two-site insertion) | ~30 min |

**Tier 1.5-B (should-do, small cost, real value):**

| ID | Target | Summary | Cost |
|----|--------|---------|------|
| EDIT-18 | §7.1 line 914 | "No evidential weight" → "hypothesis-generating not confirmatory" | ~10 min |
| EDIT-19 | §9 conclusion line 1042 | "Exploratory rather than quantitative" → "confirmatory/decision-supporting" | ~10 min |
| EDIT-20 | §4.2 line 552 (self-correction) | Soften "framework prescription" phrase EDIT-05 introduced | ~10 min |
| EDIT-22 | end of §1.2 + `references.bib` | Parallel-development related-work section engaging Messing, NIST 800-2/800-3, NeurIPS 2025 construct-validity review, Camuffo et al. | ~45 min |

**Tier 1.5-C (conditional on bandwidth, high value if executed):**

| ID | Target | Summary | Cost |
|----|--------|---------|------|
| EDIT-21 | §2, inserted before §2.1 (line 343) | Compact four-component framework table | 1–2 hrs |

**Total budget:**
- Tier 1.5-A only: ~1.5 hrs
- Tier 1.5-A + B (including expanded EDIT-22): ~2.75 hrs
- Tier 1.5-A + B + C: 4–5 hrs

---

# Tier 1.5-A (must-do)

## EDIT-13 — §6.2 Spearman per-flip coefficient retreat (CRITICAL)

**Editorial suggestion:** second-round item 5 (potential formula / framing error).

**Rationale.** The paper currently claims in §6.2:

> "Each rank-pair flip shifts Spearman by $2/(n(n^2-1)) \approx 2.2 \times 10^{-5}$; flipping on the order of a few hundred near-tied pairs accumulates drift of the observed magnitude."

Two issues. First, "rank-pair flip" is language more natural to Kendall's τ (which counts inversions) than to Spearman's ρ (which depends on squared rank differences via $\rho = 1 - 6\sum d_i^2 / n(n^2-1)$). Second, on the standard no-ties Spearman formula, a single adjacent-rank swap in one variable shifts $\sum d_i^2$ by 2, which propagates to a ρ change of $12/(n(n^2-1))$, roughly $1.3 \times 10^{-4}$ at $n=45$ — an order of magnitude larger than the paper's $2.2 \times 10^{-5}$ figure. A reviewer with numerical-statistics background will notice, and the paper's own thesis ("don't make unsupported precise claims") amplifies the cost of a wrong precise claim *within* §6.

**Recommended action:** Give up the micro-step algebra and retreat to qualitative wording. The qualitative point — that many small rank perturbations can accumulate to 10⁻² drift at n=45 — is what actually matters for the paper's argument. Do **not** substitute a new precise formula; the coefficient I computed above should be verified against a reference before being used in the source text, and the safer course is to not commit to one at all.

### EDIT-13 — §6.2, lines 816–828

**Current:**
```
Spearman correlation is a rank-based statistic: it depends on the
pairwise ordering of its inputs, not their magnitudes. At $n = 45$
residuals where many values are close together, a floating-point
perturbation too small to shift the sum of squares by more than its
15th decimal place can still flip the rank order of a few near-tied
residual pairs. Each rank-pair flip shifts Spearman by
$2/(n(n^2-1)) \approx 2.2 \times 10^{-5}$; flipping on the order of
a few hundred near-tied pairs accumulates drift of the observed
magnitude. Across numerical environments, the committed value and
the reproduced value are \emph{both correct}: each is computed
faithfully from the same input data by the same algorithm under
different floating-point paths in \texttt{numpy.linalg.lstsq}.
Neither value is more right than the other. The statistic is
numerically ill-conditioned on residuals of this size at this
sample count.
```

**Proposed:**
```
% EDIT: 2026-04-23 — editorial (second round) #5: retreat from precise
% EDIT: per-flip algebra (potential formula inaccuracy; framing more
% EDIT: natural to Kendall's tau than to Spearman). The paper's
% EDIT: qualitative point — that many small rank perturbations
% EDIT: accumulate to order-10^-2 drift at n=45 — is preserved;
% EDIT: fake precision is dropped.
Spearman correlation is a rank-based statistic: it depends on the
pairwise ordering of its inputs, not their magnitudes. At $n = 45$
residuals where many values are close together, a floating-point
perturbation too small to shift the sum of squares by more than its
15th decimal place can still flip the rank order of a few near-tied
residual pairs. Small rank-order changes alter the squared
rank-difference term in Spearman's statistic; at this sample size,
many such changes can accumulate into movement at the $10^{-2}$
scale. The exact contribution of any single near-tie depends on the
surrounding rank configuration, and we do not attempt a closed-form
per-flip accounting; what matters for the paper's argument is that
the observed environment-to-environment drift is consistent with
this sensitivity. Across numerical environments, the committed value
and the reproduced value are \emph{both correct}: each is computed
faithfully from the same input data by the same algorithm under
different floating-point paths in \texttt{numpy.linalg.lstsq}.
Neither value is more right than the other. The statistic is
numerically ill-conditioned on residuals of this size at this
sample count.
```

**Verification required before committing this edit.** Check the claim that "Spearman depends on squared rank differences, not inversions" against a standard reference (Kendall & Gibbons, *Rank Correlation Methods*; or the SciPy `scipy.stats.spearmanr` docstring, which handles ties via midrank correction and does not use the simple $1 - 6\sum d_i^2/n(n^2-1)$ form when ties are present). If the current implementation uses a tie-correcting formula, the sensitivity analysis is subtler than the paper states under either the original or the proposed wording; the proposed wording's qualitative retreat is safer under both regimes.

---

## EDIT-14 — Replace "intrinsic numerical precision" with regime-specific phrasing

**Editorial suggestion:** second-round item 3 (intrinsic language).

**Rationale.** "Intrinsic" language implies a universal property of Spearman itself. The observed effect is regime-specific: it depends on residualization, near-ties, sample size, numerical environment, and implementation path. The paper's claim is stronger and more defensible when stated precisely — "environment-sensitive" or "effective reproducibility" capture what is actually being argued. Affects five locations.

### EDIT-14a — §1.3 line 261

**Current (inside the EDIT-07 reader-payoff paragraph):**
```
A headline statistic that default reporting would quote to three
decimal places is shown to have intrinsic numerical precision at its
second decimal, turning the third digit into an unsupported claim
(Section~\ref{sec:rank-observation}).
```

**Proposed:**
```
A headline statistic that default reporting would quote to three
decimal places is shown to have effective numerical reproducibility
only at its second decimal under ordinary environment variation,
turning the third digit into an unsupported claim
(Section~\ref{sec:rank-observation}).
```

### EDIT-14b — §2.3 around lines 421–440 (three occurrences)

Three occurrences of "intrinsic" in §2.3 (What precision the indication supports). All describe the statistic's property as intrinsic; all should be softened consistently.

- Line 423: `intrinsic numerical properties of the statistic at the operative` → `environment-sensitive numerical properties of the statistic at the operative`
- Line 429: `intrinsic-statistic precision each contribute, and which one` → `statistic-level reproducibility each contribute, and which one`
- Line 435: `intrinsic floating-point precision of approximately $\pm 0.01$ at` → `environment-sensitive floating-point reproducibility of approximately $\pm 0.01$ at`
- Line 439: `it is about the statistic's intrinsic precision on residuals at this` → `it is about the statistic's environment-sensitive reproducibility on residuals at this`

Add one `% EDIT:` block at the top of §2.3 noting these four substitutions are a coordinated pass on the same reframe; individual occurrences then do not each need separate comment blocks.

**Recommended marker block above §2.3 opening:**
```
% EDIT: 2026-04-23 — editorial (second round) #3: replace "intrinsic"
% EDIT: with "environment-sensitive" / "effective reproducibility" in
% EDIT: four occurrences of §2.3. Reason: the observed effect depends
% EDIT: on residualization, near-ties, sample size, and implementation
% EDIT: path, not a universal Spearman property.
```

### EDIT-14c — §9 conclusion around line 1018

**Current:**
```
limitation of the headline statistic (rank-based correlation on
small-$N$ residuals has intrinsic floating-point precision of order
$10^{-2}$) that unstructured reporting conventions do not flag.
```

**Proposed:**
```
limitation of the headline statistic (rank-based correlation on
small-$N$ residuals has effective reproducibility of order $10^{-2}$
across numerical environments) that unstructured reporting conventions
do not flag.
```

---

## EDIT-15 — §6.3 n=200 stabilization sentence

**Editorial suggestion:** second-round item 4 (n=200 is too confident without simulation).

**Rationale.** The paper claims "At n = 200 or larger the statistic's intrinsic precision would stabilize substantially." This is an empirically or analytically unsupported claim about larger-sample behavior. The paper's entire thesis is about not making unsupported claims; having one inside §6 is self-undermining. Retreat to a weaker statement.

### EDIT-15 — §6.3, lines 838–839

**Current:**
```
At $n = 200$ or larger the statistic's intrinsic precision
would stabilize substantially; at $n = 45$, it does not.
```

**Proposed:**
```
% EDIT: 2026-04-23 — editorial (second round) #4: retreat from
% EDIT: unsupported n=200 stabilization claim. The paper's own
% EDIT: thesis precludes making confident claims without evidence;
% EDIT: we leave the threshold unspecified.
At larger sample sizes, and especially when residual ranks are less
dominated by near-ties, this source of drift should typically
diminish; the present study does not estimate a general stabilization
threshold.
```

---

## EDIT-16 — §6.4 "discovery" language

**Editorial suggestion:** second-round item 6 (soften "discovery").

**Rationale.** The phrase "measurement property unnamed in the ML methods literature prior to this study" invites a reviewer to cite prior numerical-statistics work on rank-correlation instability. The more defensible framing is "discovery-like in the narrower reporting sense": the issue was not visible in the original analysis and is rarely named in ML diagnostic reports, surfaced here by the framework. That is what the paper actually establishes.

### EDIT-16 — §6.4, lines 875–879

**Current:**
```
what measurement-disciplined reporting yields. The three split into
two epistemic kinds. The rank-on-residuals property is a
\emph{discovery}: a measurement property unnamed in the ML methods
literature prior to this study, surfaced by the framework's
reproducibility discipline.
```

**Proposed:**
```
% EDIT: 2026-04-23 — editorial (second round) #6: soften "discovery"
% EDIT: language to "discovery-like in the narrower reporting sense"
% EDIT: to avoid claiming priority over numerical-statistics
% EDIT: literature. The contribution is in the reporting register,
% EDIT: not in the mathematical phenomenon.
what measurement-disciplined reporting yields. The three split into
two epistemic kinds. The rank-on-residuals property is
\emph{discovery-like in the narrower reporting sense}: a measurement
constraint that was not visible in the original analysis and is
rarely named in ML diagnostic reports, surfaced here by the
framework's reproducibility discipline.
```

---

## EDIT-17 — FAMILY_B capacity clarification

**Editorial suggestion:** second-round item 7 (FAMILY_B is not "too powerful").

**Rationale.** The family-pair decomposition explains 88.1% of outcome variance using 28 family-pair cells against N = 45 observations. A skeptical reviewer will correctly observe that FAMILY_B is a high-capacity baseline relative to the sample size — and will ask whether the decomposition simply absorbs the dataset, leaving nothing for the diagnostic to beat. The paper's answer, implicit in the pre-registration, is: yes, deliberately — the claim is about *incremental* diagnostic information beyond an obvious family confound, not about estimating a deployable predictor. Making this explicit in both §5.2 and the limitations section defends against the attack vector without changing any numbers.

### EDIT-17a — §5.2 "Why this is an informative null, framework-wise", appended to the existing paragraph at line 662

**Current (lines 661–671):**
```
Task-family-pair identity explains 88.1\% of outcome variance on
its own; $S_{\mathrm{H1}}$ contributes $+0.003$. Without the
framework's confound-decomposition discipline, the raw correlation
(also negative, smaller magnitude) would have been reported without
context, and the reader would have had no way to assess what
fraction of any predictive claim was actually family identity
operating under a different name. The informativeness of the null
is legible because the decomposition is pre-registered; it is
opaque otherwise. The $R^2 = 0.881$ is itself a finding --- a large
effect for the simplest alternative explanation, reportable
regardless of how the geometric diagnostic performs.
```

**Proposed (extend by one sentence at the end):**
```
Task-family-pair identity explains 88.1\% of outcome variance on
its own; $S_{\mathrm{H1}}$ contributes $+0.003$. Without the
framework's confound-decomposition discipline, the raw correlation
(also negative, smaller magnitude) would have been reported without
context, and the reader would have had no way to assess what
fraction of any predictive claim was actually family identity
operating under a different name. The informativeness of the null
is legible because the decomposition is pre-registered; it is
opaque otherwise. The $R^2 = 0.881$ is itself a finding --- a large
effect for the simplest alternative explanation, reportable
regardless of how the geometric diagnostic performs.
% EDIT: 2026-04-23 — editorial (second round) #7: acknowledge
% EDIT: FAMILY_B's capacity explicitly to defuse "baseline eats the
% EDIT: dataset" attack vector. Purpose of decomposition is
% EDIT: incremental-information test, not deployable predictor.
Because \textsc{family\_b} uses 28 family-pair cells relative to
$N = 45$, it is a high-capacity baseline by design: its role in the
decomposition is to test whether $S_{\mathrm{H1}}$ carries
predictive information beyond the pre-registered family-pair
structure that the diagnostic would need to exceed in order to
support the claimed triage use. The $\Delta R^2$ of $+0.003$ is
informative about that specific incremental question; it is not an
estimate of what a deployable merge-risk predictor would achieve,
and \textsc{family\_b} itself is not offered as one.
```

### EDIT-17b — §7 limitations (add to "Limitations proper" subsection)

**Anchor:** the limitations section is likely in §7 (Objections, Alternatives, Limitations). Grep for a "Limitations proper" or similar heading; insert the following item.

**Proposed addition to limitations:**
```
% EDIT: 2026-04-23 — editorial (second round) #7: limitations-side
% EDIT: counterpart to §5.2's FAMILY_B clarification. Defends the
% EDIT: decomposition choice as appropriate-for-purpose.
The family-pair residualization used in the H1 test is intentionally
conservative and high-capacity relative to the sample size. It is
appropriate for testing whether the diagnostic carries incremental
information beyond a pre-registered confound structure; it is not
appropriate for estimating a general predictive model, and no such
model is offered. A lower-capacity baseline (e.g., dummy-coded
task-family main effects only, without the full family-pair
interaction) would yield a different decomposition and a different
$\Delta R^2$; we report the pre-registered specification rather than
select among post-hoc alternatives.
```

Placement: add as a new bullet or paragraph within the existing limitations list, near the other "design-choice" limitations.

---

# Tier 1.5-B (should-do)

## EDIT-18 — "No evidential weight" → "not confirmatory"

**Editorial suggestion:** second-round item 8.

**Rationale.** The claim that post-hoc analyses have "no evidential weight" is too absolute — exploratory findings have hypothesis-generating value, just not confirmatory-decision weight. The stronger and more defensible formulation distinguishes registers rather than negating evidential status entirely.

### EDIT-18 — §7.1, lines 914–916

**Current:**
```
  \item Post-hoc analyses labeled as such and assigned no
    evidential weight. The C4 pre-registration discipline is
    minimum-viable methodological hygiene.
```

**Proposed:**
```
% EDIT: 2026-04-23 — editorial (second round) #8: distinguish
% EDIT: evidential registers rather than negate evidential weight
% EDIT: entirely. Exploratory findings are hypothesis-generating.
  \item Post-hoc analyses labeled as such and assigned
    hypothesis-generating rather than confirmatory evidential
    status, consistent with the C4 pre-registration discipline as
    minimum-viable methodological hygiene.
```

---

## EDIT-19 — Conclusion ending phrase

**Editorial suggestion:** second-round item 9.

**Rationale.** "Exploratory rather than quantitative" is a register mismatch — exploratory analyses can still be quantitative; the distinction being drawn is about licensed inferential status, not about whether the analysis produces numbers. "Exploratory rather than confirmatory or decision-supporting" captures the actual distinction.

### EDIT-19 — §9 conclusion, lines 1041–1043

**Current:**
```
diagnostic that cannot yet supply these elements may still be useful,
but its evidential status should be described as exploratory rather
than quantitative. The purpose of the framework is not to slow ML
```

**Proposed:**
```
% EDIT: 2026-04-23 — editorial (second round) #9: the distinction
% EDIT: being drawn is about licensed inferential status, not about
% EDIT: whether numbers are produced. Exploratory analyses can be
% EDIT: quantitative; the issue is confirmatory vs. decision-supporting
% EDIT: status.
diagnostic that cannot yet supply these elements may still be useful,
but its evidential status should be described as exploratory rather
than confirmatory or decision-supporting. The purpose of the
framework is not to slow ML
```

---

## EDIT-20 — Soften "framework prescription" (self-correction)

**Editorial suggestion:** second-round note on "slightly branded" phrasing.

**Rationale.** The sentence inserted in the last pass (EDIT-05) reads: *"Naming what the coefficient does not cover is itself a framework prescription."* The editor's second-round note flags this as still slightly in-register. The main text benefits from the more neutral phrasing I earlier deferred to Appendix D alone.

### EDIT-20 — §4.2, lines 552–553

**Current:**
```
is moderately stable across tasks. Naming what the coefficient does
not cover is itself a framework prescription.
```

**Proposed:**
```
% EDIT: 2026-04-23 — editorial (second round): soften "framework
% EDIT: prescription" phrasing EDIT-05 introduced. The neutral
% EDIT: variant better matches the register used elsewhere in §4.
is moderately stable across tasks. Explicitly naming what the
coefficient does not cover is part of the proposed reporting
discipline.
```

---

## EDIT-22 — Parallel-development related-work section

**Trigger.** Two daily research reviews (`research_review/2026-04-25.md` inaugural pass; second pass 2026-04-26) surfaced five parallel-development items in a roughly six-month window: Messing (2026) TEE framework, NIST AI 800-2 voluntary practices, NIST AI 800-3 GLMM endorsement, the NeurIPS 2025 construct-validity systematic review, and Camuffo et al. (2026) variance-aware LLM annotation. The N134 paper's current framing — implicitly "first to apply measurement discipline to ML evaluation" — is no longer defensible against this corpus. EDIT-22 was originally drafted as a single-paragraph positioning relative to Messing; that scope is now insufficient.

**Scope expansion rationale.** The combined picture across the second-pass review is that the measurement-discipline thesis is a co-developed register, not a single-author proposal. A reviewer of N134 familiar with NIST AI 800-2/800-3 will expect to see the NIST documents engaged; not citing them would look like an oversight much sharper than the Messing-only oversight would have been. The NeurIPS 2025 review, with its 445-paper survey and 29-reviewer construct-validity taxonomy, is the canonical recent reference and similarly cannot be omitted. Camuffo et al. uses essentially identical methodological vocabulary (generalizability theory, "auditable measurement infrastructure"), and the convergence is striking enough that engagement is the honest move.

**The contribution-claim shift.** The framing should retire "first to propose" entirely (not merely soften it) and adopt "co-developed register, distinctive prescriptive contribution." The load-bearing differentiator the manuscript still owns is the **decimal-place precision tolerance schedule** as the prescriptive output — none of the parallel works produces a tolerance schedule that licenses decimal-place precision in reported scores. That is now what the paper is contributing, not "first to apply X to Y."

This remains a framing revision, not a contribution-claim revision in the load-bearing sense — the precision-pathology observation, the disciplined null on H1, and the §1.1 reporting-gap diagnosis all stand as N134's empirical and prescriptive contributions. What changes is the manuscript's positioning relative to the field.

### EDIT-22a — End of §1.2 (after line 198, after `\citep{messick1989validity,messick1995validity}` paragraph, before `\subsection{Contribution claims}`)

**Proposed insert (two paragraphs, ~290 words):**

```
% EDIT: 2026-04-26 — daily research reviews 2026-04-25 (inaugural) and
% EDIT: 2026-04-26 (second pass) surfaced five parallel-development
% EDIT: works applying measurement-discipline registers to LLM
% EDIT: evaluation. This expanded positioning paragraph names the
% EDIT: convergence and locates the paper's distinctive contribution
% EDIT: without revising the load-bearing empirical or prescriptive
% EDIT: claims.
A measurement-discipline register has emerged contemporaneously
across multiple voices. \citet{messing2026hidden} develops a Total
Evaluation Error framework for LLM evaluation pipelines that
decomposes pipeline uncertainty into design-choice variance and
shrinking-with-N variance, demonstrating on MMLU benchmarking that
optimized budget allocation halves estimation error at equivalent
cost. The U.S.\ National Institute of Standards and Technology's
voluntary-practices document
\citep{nist2026benchmarkpractices} structures benchmark evaluation
into define-target, run-evaluation, and analyze-and-report stages;
its companion statistical-models document \citep{nist2026statmodels}
formally endorses generalized linear mixed models for variance
decomposition on AI benchmarks, demonstrated on twenty-two frontier
LLMs. \citet{reuel2025measuring} survey 445 LLM benchmarks and
develop a construct-validity failure taxonomy with eight
recommendations and an operational checklist. \citet{camuffo2026variance}
identify five variance sources in LLM annotation for strategy
research and demonstrate twelve to eighty-five percentage-point
swings from minor design choices, grounding their protocol in
generalizability theory.

The shared diagnosis across this corpus is that ML evaluation
pipelines carry hidden measurement variance that ordinary reporting
does not surface; the convergence across substrates (benchmarking,
annotation, regulatory practice) is itself evidence that the
diagnosis is real. The prescriptive registers differ: Messing
optimizes evaluation-budget allocation; NIST endorses GLMM
methodology; the construct-validity survey provides taxonomic
guidance; Camuffo et al.\ develop variance-aware annotation
protocols. The present paper's contribution within this co-developed
register is the decimal-place precision tolerance schedule —
licensing what numerical precision a reported score actually
supports under a declared measurement universe. Generalizability
theory, variance decomposition, and construct-validity reasoning
are now shared methodological apparatus across the field; the
tolerance-schedule prescription is what the present paper still
distinctively contributes.
```

### EDIT-22b — `references.bib` entries

Five new entries (one per parallel work). All use the existing `% ANON:` marker convention for camera-ready restoration of the v2-anonymized-supplementary tag's anonymization commitments:

```bibtex
% ANON: full author and affiliation restored at camera-ready; the
% ANON: v2-anonymized supplement does not name these works because the
% ANON: cites were added at v2.1 (post-anonymization-tag).
@misc{messing2026hidden,
  title         = {Hidden Measurement Error in {LLM} Pipelines Distorts Annotation, Evaluation, and Benchmarking},
  author        = {Messing, Solomon},
  year          = {2026},
  eprint        = {2604.11581},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  doi           = {10.48550/arXiv.2604.11581},
  note          = {Last revised 2026-04-22, three days before benchmark-reliability v1.1.1 lock.}
}

@techreport{nist2026benchmarkpractices,
  title       = {{NIST AI 800-2}: Practices for Automated Benchmark Evaluations of Language Models},
  author      = {{National Institute of Standards and Technology, Center for AI Standards and Innovation}},
  institution = {U.S.\ Department of Commerce, NIST},
  year        = {2026},
  month       = {jan},
  type        = {Draft NIST Special Publication},
  note        = {60-day public comment period closed 2026-03-31.}
}

@techreport{nist2026statmodels,
  title       = {{NIST AI 800-3}: Expanding the {AI} Evaluation Toolbox with Statistical Models},
  author      = {{National Institute of Standards and Technology}},
  institution = {U.S.\ Department of Commerce, NIST},
  year        = {2026},
  month       = {feb},
  type        = {NIST Special Publication}
}

@inproceedings{reuel2025measuring,
  title     = {Measuring what Matters: Construct Validity in Large Language Model Benchmarks},
  author    = {Reuel, Anka and others},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2025},
  eprint    = {2511.04703},
  archivePrefix = {arXiv},
  doi       = {10.48550/arXiv.2511.04703},
  note      = {445-benchmark systematic review with 29 expert reviewers; eight key recommendations + operational checklist.}
}

@misc{camuffo2026variance,
  title         = {Variance-Aware {LLM} Annotation for Strategy Research},
  author        = {Camuffo and Gambardella and Kazemi and Malachowski and Pandey},
  year          = {2026},
  eprint        = {2601.02370},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  doi           = {10.48550/arXiv.2601.02370},
  note          = {41-pp main + 53-pp appendix; generalizability-theory grounding; five-source variance taxonomy; 12-85 pp swings from minor design choices.}
}
```

Author-name formatting for `reuel2025measuring` and `camuffo2026variance` should be verified against the actual papers when the cite is staged; the entries above use placeholder formatting that the BibTeX `tmlr.bst` will tolerate. Verify against arXiv pages and update the `author` fields before commit.

### EDIT-22c — Track v1.1.1 → v1.1.2 dependency

EDIT-22 also coordinates with the v1.1.2 amendment for D-09 (LPM-vs-GLMM hybrid; recorded in the benchmark reliability study's `LOCK_NOTES.md` and `IMPLEMENTATION_DEVIATIONS.md`). Specifically: the §1.2 paragraph cites NIST AI 800-3 as the GLMM endorsement that motivated the D-09 resolution. The two papers are now linked at the citation level — the benchmark study's methodological appendix on LPM-vs-GLMM agreement should reference NIST 800-3 as the formal endorsement that D-09 is responding to. EDIT-22's reference to NIST 800-3 is part of the same intellectual move.

### Acceptance criteria

- §1.2 ends with the two-paragraph parallel-development section; §1.3 (Contribution claims) follows immediately as before.
- `references.bib` includes all five new entries (`messing2026hidden`, `nist2026benchmarkpractices`, `nist2026statmodels`, `reuel2025measuring`, `camuffo2026variance`).
- All five entries' author fields verified against actual papers before commit.
- The paragraph compiles without warnings; no other section is modified.
- Page count delta: ~+0.45 pp (two paragraphs at 11pt). Cumulative Tier 1.5 delta should now be tracked at compile checkpoint; if the cumulative delta pushes the paper over 17 pp under microtype, consider whether one of the framework table or this expanded section can be tightened.

---

# Tier 1.5-C (conditional)

## EDIT-21 — Compact four-component framework table in §2

**Editorial suggestion:** second-round item 1 (now elevated to highest-value remaining improvement by the editor).

**Rationale.** The editor has flagged this in two consecutive rounds. Consilience across two independent readings suggests TMLR reviewers will similarly have difficulty holding the four-component framework in one place without a summary structure. Deferring to the revision round costs the same ~2 hours, but after reviewers have already complained; doing it pre-submission probably improves the Action Editor's first read. Execute only if Tier 1.5-A and -B are complete and bandwidth remains before the April 28 decision gate.

### EDIT-21 — §2, inserted before §2.1 (after line 341)

**Placement:** immediately after the closing paragraph of §2's opening (the epistemic-ordering paragraph ending at line 341), before `\subsection{What the score indicates}` at line 343.

**Proposed LaTeX:**
```
% EDIT: 2026-04-23 — editorial (second round) #1: compact framework
% EDIT: table inserted before §2.1 as an anchor reviewers can point
% EDIT: at. Summary of the four-component framework plus case-study
% EDIT: instantiation, with explicit section cross-references.

\begin{table}[h]
  \centering
  \small
  \begin{tabular}{p{0.20\textwidth}p{0.22\textwidth}p{0.20\textwidth}p{0.30\textwidth}}
    \toprule
    Measurement question & Default ML failure mode & Required reporting discipline & Case-study instantiation \\
    \midrule
    What does the score indicate?
      & Operationalization treated as construct
      & Construct articulation
      & $S_{\mathrm{H1}}$ as O-module depth-weighted spectral alignment (§\ref{sec:framework-construct}, §\ref{sec:applying-construct}) \\
    How stable is the indication?
      & Point estimate treated as reproducible
      & Cross-seed reliability (ICC, SEM)
      & Same-task $\hat{\rho}_{\mathrm{ICC}} = 0.566$, SEM $= 0.014$; regime-bounded to same-task (§\ref{sec:framework-reliability}, App.~\ref{app:reliability}) \\
    What precision is licensed?
      & Third/fourth decimals overreported
      & Tolerance schedule tied to operative sample size
      & Partial Spearman reported as $\approx -0.53 \pm 0.01$ (§\ref{sec:framework-precision}, §\ref{sec:rank-observation}) \\
    What else explains the signal?
      & Confound absorbed into metric claim
      & Pre-registered confound decomposition
      & \textsc{family\_b} $R^2 = 0.881$; $S_{\mathrm{H1}}$ $\Delta R^2 = 0.003$ (§\ref{sec:framework-confound}, §\ref{sec:applying-confound}) \\
    \bottomrule
  \end{tabular}
  \caption{The four-component measurement-discipline framework:
    the measurement question behind each component, the default
    ML reporting pattern it addresses, the required discipline,
    and how the worked example instantiates each. Section and
    appendix references locate the full development of each row.}
  \label{tab:framework}
\end{table}
```

**Label targets.** The `§\ref{...}` calls in the table assume the following `\label{}` strings already exist in the source:
- `sec:framework-construct` — §2.1
- `sec:framework-reliability` — §2.2
- `sec:framework-precision` — §2.3
- `sec:framework-confound` — §2.4
- `sec:applying-construct` — §3.1 or §4.1 (the construct-articulation subsection under "Applying the Framework")
- `sec:applying-reliability` — §4.2 (the reliability-considerations subsection; already referenced elsewhere)
- `sec:applying-confound` — §4.3 (the confound-decomposition subsection)
- `app:reliability` — Appendix D

Verify each label is defined before the table compiles; if any is missing, add the label at the appropriate section/appendix heading. The table will compile with undefined references (LaTeX renders `??`); clean compile requires the labels.

**Post-table prose connector.** Add one short paragraph immediately after the table, before §2.1, to introduce the subsection flow:

```
The remainder of this section develops each row in turn, in the
order a careful reader's credence would consult them.
```

**Page-count estimate.** A compact four-row table with caption renders to approximately 0.4–0.6 of a page at 10pt. The total paper is currently 16 pp; after Tier 1.5-A and -B edits (which net approximately zero page count — some inserts, some compressions), the paper is likely still 16 pp. Adding the table may push to 17 pp under microtype. 17 pp remains well within TMLR's flexible format; not a blocking issue.

---

# Process

## Order of operations

1. **EDIT-13 first, before anything else.** Technical-correctness edit; must verify against reference before committing the replacement text. If the Spearman-formula verification takes longer than expected, pause the rest of the pass.
2. **Tier 1.5-A in sequence** (EDIT-14 → EDIT-15 → EDIT-16 → EDIT-17). EDIT-14 touches five locations and should be completed as one coordinated substitution; others are independent.
3. **Compile checkpoint.** Full four-pass MacTeX compile with microtype. Verify: page count, zero undefined citations, zero undefined references (especially if EDIT-21 is executed — new `\ref{}` calls).
4. **Tier 1.5-B** (EDIT-18, EDIT-19, EDIT-20). Independent word-level edits; order does not matter.
5. **Tier 1.5-C (EDIT-21, conditional).** Execute only if bandwidth allows and if the compile after Tier 1.5-A+B is clean. The table introduces new cross-references; a failed compile is cheapest to diagnose before this edit is added, not after.
6. **Compile checkpoint #2.** Same checks.
7. **Commit.** Single commit for all Tier 1.5 edits unless EDIT-21 is held back, in which case two commits. Commit message: `papers/n134_workshop: pre-submission editorial pass (Tier 1.5 reviewer-proofing)`.

## Verification after the full pass

```bash
cd /Users/john/code/gradience/papers/n134_workshop/

# Enumerate edits across all three passes
grep -n '% EDIT:' draft_v2_thesis_b.tex | wc -l
# Should be roughly: prior-pass count + 8 (Tier 1.5-A + B) + 1 (if EDIT-21)

# No more "intrinsic" in the precision contexts
grep -n 'intrinsic' draft_v2_thesis_b.tex
# Remaining occurrences should only be in unrelated contexts (e.g., philosophical
# references to "intrinsic" as a term of art in measurement theory); verify each
# survival against EDIT-14's substitution plan

# Standard compile
rm -f *.aux *.bbl *.blg *.log *.out *.toc
pdflatex -interaction=nonstopmode draft_v2_thesis_b.tex
bibtex draft_v2_thesis_b
pdflatex -interaction=nonstopmode draft_v2_thesis_b.tex
pdflatex -interaction=nonstopmode draft_v2_thesis_b.tex
pdfinfo draft_v2_thesis_b.pdf | grep Pages
grep -E "(undefined|Undefined|^\!)" draft_v2_thesis_b.log

# If EDIT-21 executed, verify the table's \ref calls resolved
grep -E "Reference|reference.*undefined" draft_v2_thesis_b.log
```

---

# Notes on second-round suggestions not adopted

For completeness, suggestions from the second editorial review that do not become edits in this spec:

- **"Forcing refusals" softening in abstract** — keep original. The sentence is doing register work the neutral variant loses, and "forcing refusals" is distinctive without being aggressive. The editor acknowledges the tradeoff in-text.
- **"Third decimal would have been an unstated promise the data could not keep"** — keep original. Editor explicitly notes "I still love this" and the khaki variant drops the moral-epistemic weight that the paper's register rests on. The sentence appears once; its voice cost is minimal and its rhetorical return is substantial.
- **Minor wording edits ("not a methods upgrade" → "not merely a methods upgrade"; "framework's internal coherence is the argument" softening; various)** — individually low-value. If a global polish pass is undertaken separately, bundle them; otherwise skip.

---

# Time budget summary

| Phase | Cost | Cumulative |
|-------|------|------------|
| EDIT-13 (Spearman formula verification + edit) | 20–45 min | 20–45 min |
| EDIT-14 (intrinsic, 5 locations) | 20 min | 40–65 min |
| EDIT-15, 16 (n=200, discovery) | 20 min total | 60–85 min |
| EDIT-17 (FAMILY_B, two sites) | 30 min | 90–115 min |
| Compile checkpoint #1 | 10 min | 100–125 min |
| EDIT-18, 19, 20 (language polish) | 30 min total | 130–155 min |
| EDIT-21 (framework table + labels + compile fixes) | 60–120 min | 190–275 min |
| Compile checkpoint #2 | 10 min | 200–285 min |
| Commit | 5 min | 205–290 min |

**Minimum pass (Tier 1.5-A only):** ~1.5 hrs.
**Recommended pass (Tier 1.5-A + B):** ~2.5 hrs.
**Full pass including framework table:** ~4–5 hrs.

Decision gate remains 2026-04-28. Tier 1.5-A alone is safely inside the window under any realistic A4 timeline; Tier 1.5-A + B is comfortable; full pass with EDIT-21 requires committing a working day to editorial work and is defensible only if A4 (MacTeX verification + tarball + OpenReview form) has not slipped.
