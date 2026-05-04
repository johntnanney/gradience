# First-Revision Staged Edits

Tracking file for engagement passages prepared during pre-submission review
but appropriate at first-revision rather than v1 submission. The v1
submission ships a 9-voice convergent parallel-development register
accurate-at-time-of-submission; the contrarian-voice engagement below
strengthens the corpus claim once pre-registered (i.e., once the v1 thesis
has survived the named dissent).

These drafts engage Hua et al. 2025 ("Flaw or Artifact? Rethinking Prompt
Sensitivity in Evaluating LLMs," EMNLP 2025, arXiv:2509.01790), which
takes a contrarian position relative to the parallel-development register:
Hua et al. argue that LLM prompt sensitivity is largely an artifact of
heuristic evaluation methods (log-likelihood scoring, rigid answer
matching) rather than an inherent model property, and that LLM-as-a-Judge
evaluation substantially reduces both performance variance and
cross-prompt ranking instability.

Status as of 2026-05-03: drafts written, not yet applied to manuscript.
Apply at first-revision on a dedicated branch off whatever the
benchmark-reliability paper's submission HEAD is at that time. Suggested
branch name: `papers/benchmark_reliability_study/first-revision-hua-engagement`.

---

## 1. Bibtex entry for `references.bib`

Insert in the appropriate section of `manuscript/references.bib`. Following
the existing convention (URL comment, optional surfacing comment, biblatex
fields, verbose `note`).

```bibtex
% https://arxiv.org/abs/2509.01790
% Surfaced via daily research review 2026-05-03 (second pass).
@inproceedings{hua2025flaw,
  author        = {Hua, Andong and Tang, Kenan and Gu, Chenhe and
                   Gu, Jindong and Wong, Eric and Qin, Yao},
  title         = {Flaw or Artifact? {R}ethinking Prompt Sensitivity in
                   Evaluating {LLM}s},
  booktitle     = {Proceedings of the 2025 Conference on Empirical
                   Methods in Natural Language Processing},
  pages         = {19889--19899},
  year          = {2025},
  address       = {Suzhou, China},
  publisher     = {Association for Computational Linguistics},
  eprint        = {2509.01790},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  note          = {Argues that LLM prompt sensitivity is largely an
                   artifact of heuristic evaluation methods (log-likelihood
                   scoring, rigid answer matching) rather than an inherent
                   model property; LLM-as-a-Judge evaluation reduces
                   performance variance and increases cross-prompt
                   ranking correlation. Contrarian voice in the
                   parallel-development register
                   (§\ref{sec:parallel-dev-related-work}); engaged
                   directly in §\ref{sec:cite-discipline} and
                   §\ref{sec:ranking-stability-results}.}
}
```

---

## 2. §2 (Prompt-sensitivity literature) — register the contrarian voice

Currently §2 introduces three convergent voices (Sclar 2023, Polo 2024,
Lunardi 2025) as the "empirical baseline this study works from." Adding
Hua et al. registers the contrarian position without engaging it
substantively at this section — the engagement happens at §4.3 and §7.4.

### Where to apply

Append a new sentence to the end of the paragraph at `manuscript/draft_v1.tex`
line ~101 (the paragraph beginning "The empirical literature on prompt
sensitivity is the closest neighbor..."). The new sentence sits between
"This literature is the empirical baseline this study works from." and
the next paragraph at line ~103.

### Replacement target (line ~101)

The existing paragraph ends with: "This literature is the empirical
baseline this study works from."

Append:

```latex
\citet{hua2025flaw} take a contrarian position within this same
literature: the apparent prompt sensitivity reported across the
empirical baseline is, on their evidence, largely an artifact of
heuristic evaluation methods (log-likelihood scoring, rigid answer
matching) rather than an inherent model property, and substantially
dissolves under LLM-as-a-Judge evaluation. The present paper engages
this dissent directly at §\ref{sec:cite-discipline} (parallel-development
positioning) and §\ref{sec:ranking-stability-results} (empirical contact
point on H3); §\ref{sec:prompt-sensitivity} registers both readings of
the phenomenon as part of the empirical baseline whose precise
interpretation is itself the methodological question.
```

---

## 3. §4.3 (What is cited and what is not) — load-bearing engagement

The 9-voice corpus table at §4.1 line ~525 stays as 9 convergent voices.
Hua et al. is not added to that table because the table's epistemic
content is convergence, not engagement. Instead, three new paragraphs
land in §4.3 after the existing scope-discipline note, transforming the
register from "9 convergent voices" to "9 convergent + 1 named-and-engaged
dissent."

### Where to apply

Insert three new paragraphs at `manuscript/draft_v1.tex` after the
existing §4.3 paragraph (around line ~561, after the sentence ending
"...only a scope discipline.") and before the section break at line ~563.

### New paragraphs

```latex
One adjacent work merits explicit engagement rather than scope-disciplined
exclusion: \citet{hua2025flaw}'s contrarian reframing of the
prompt-sensitivity phenomenon. Hua et al.\ argue that the prompt
sensitivity reported across the empirical baseline
(§\ref{sec:prompt-sensitivity}) and the parallel-development register's
nine voices is largely an artifact of heuristic evaluation methods rather
than a genuine measurement-instability finding: log-likelihood scoring
and rigid answer matching, on their evidence, fail to credit semantically
correct responses, producing apparent variance that LLM-as-a-Judge
evaluation substantially dissolves. Their conclusion is that modern LLMs
are more robust to prompt templates than the literature suggests, and that
prompt sensitivity is more an artifact of evaluation than a flaw in the
models. This is a real disagreement with the parallel-development
register's diagnosis, not a vocabulary mismatch.

The disagreement turns out to be productive rather than dissolving. The
present paper's regime split (§\ref{sec:design-regime-split}) is itself a
partial concession to exactly the kind of point Hua et al.\ press:
parse-failure-dominated cells, where the generate-and-parse scoring rule
produces variance dominated by parse-failure rather than measurement-design
facets, are routed to a sample-SD-based tolerance precisely because the
variance-components decomposition is uninterpretable on those cells under
that scoring rule. The mechanism Hua et al.\ flag --- heuristic scoring
producing variance qualitatively different from real measurement-design
variance --- is the same mechanism the regime split addresses, with a
different methodological response: the regime split records the
parse-failure variance as evidence of measurement-instrument fragility
rather than evaluating it away with an alternative scoring rule. On the
twenty-three $g$-theory cells where parsing is not the issue, the
variance-components decomposition still attributes substantial variance
to the scoring-rule facet (§\ref{sec:variance-components-results}), which
the present paper treats as part of the admissible measurement universe
rather than as artifact to be evaluated away. The two readings of
scoring-rule variance --- Hua et al.'s artifact reading, the
parallel-development register's measurement-universe reading --- both
have evidentiary content; the present paper's position is that scoring
rule belongs in the universe rather than outside it, and that the
rule-conditioned schedule (Table~\ref{tab:tolerance-rule-conditioned}) is
the appropriate prescriptive response when reports name a scoring rule,
with the across-rule schedule (Table~\ref{tab:tolerance-across-rule}) the
appropriate prescriptive response when reports do not.

The convergence claim §\ref{sec:codev-register} rests on is strengthened,
not weakened, by named engagement with this dissent. A nine-voice
convergent corpus that has been pressure-tested against a contrarian
reading and survives the test is a stronger evidentiary base than a
nine-voice register that has not encountered its dissent. The reading
discipline the parallel-development register supports --- that benchmark
scores carry hidden measurement variance that ordinary single-occasion
reporting does not surface --- is one the regime split, the
rule-conditioned schedule, and the across-rule schedule preserve under
Hua et al.'s critique by being scoring-rule-aware rather than
scoring-rule-blind.
```

---

## 4. §7.4 (Ranking stability, H3) — empirical contact point

Hua et al.\ specifically claim that LLM-as-a-Judge evaluation produces
"consistently higher correlation in model rankings across prompts." This
is in direct empirical contact with H3's close-skill ranking-instability
finding. The current §7.4 already engages Brittlebench's 63% reversal as
an external comparison anchor; a parallel paragraph engaging Hua et al.
fits the existing structure naturally.

### Where to apply

Insert a new paragraph at `manuscript/draft_v1.tex` after the existing
Brittlebench paragraph (around line ~1103, after the sentence ending
"...regime-bounded instance of the same phenomenon Brittlebench's $63\%$
finding registers at frontier scale.") and before the
`\begin{table}[h]` for the ranking-reversals table.

### New paragraph

```latex
\citet{hua2025flaw} report a different empirical pattern at frontier
scale: LLM-as-a-Judge evaluation produces, on their seven-LLM panel and
six-benchmark substrate, a substantially higher correlation in model
rankings across prompts than the heuristic evaluation methods
(log-likelihood scoring, rigid answer matching) on which the present
panel and the broader prompt-sensitivity literature rely. Their finding
is, on its face, in tension with the H3 confirmation: if scoring-rule
choice is doing most of the apparent ranking-instability work, a panel
scored under LLM-as-a-Judge would presumably show fewer condition-reversals
than the present panel reports. Two limits on that inference apply.
First, the H3 instability is conditional rather than aggregate: it
concentrates on close-skill (within-family) pairs across all five
benchmarks, and the cross-skill (cross-lineage) ranking is stable on the
four benchmarks where the cross-lineage gap is large. Hua et al.'s
``higher correlation'' is not ``near-perfect correlation''; their finding
does not entail that the close-skill instability specifically would
dissolve under their scoring rule. Second, the present panel's scoring
rules (length-normalized log-likelihood and generate-and-parse) were
pre-registered at v1.1 lock as the panel's measurement universe, before
any data collection; switching scoring rules post-hoc to evaluate
Hua et al.'s reframing on the present substrate would be exactly the
kind of post-hoc analysis the pre-registration discipline is meant to
prevent. The cleanest test of Hua et al.'s reframing on the H3 question
is a frontier-scale panel under LLM-as-a-Judge scoring, which is named in
§\ref{sec:future-work} as a regime-bounded extension.
```

---

## 5. Application notes

**Branching.** Apply on a first-revision branch off whatever the
manuscript's submission HEAD is at first-revision time. Suggested branch:
`papers/benchmark_reliability_study/first-revision-hua-engagement`. Do not
land these passages on the current submission branch lineage —
they are first-revision content, not pre-submission content.

**Order of edits.** Apply in this order: (1) bibtex entry into
`references.bib` first so the `\citet{hua2025flaw}` references resolve;
(2) §2 sentence; (3) §4.3 three paragraphs; (4) §7.4 paragraph. Compile
after each step is fine, or compile at the end. Both work cleanly because
the edits are additive only.

**Compile-verify expectation.** Page count growth approximately 1–2
pages, mostly from the §4.3 three-paragraph block. 0 expected
overfull/underfull beyond pre-existing baseline. New citation key
`hua2025flaw` should resolve on bibtex pass.

**Tone considerations.** The drafts are not defensive. The framing is
"the convergence claim is strengthened by engaging this dissent" rather
than "we acknowledge that some have argued..." If the author prefers a
more concessive tone, the §4.3 third paragraph is the right place to
soften — the empirical engagement at §7.4 should remain rigorous on the
inference-bounding claims (close-skill conditional pattern, pre-registered
scoring rule, regime-bounded scope).

**Anonymity check at first-revision tarball assembly.** If the
first-revision tarball is for TMLR, run a paper-wide recursive grep for
anonymity leaks per the discipline established at the N134 v3 tarball
rebuild. The new bibtex entry is a published EMNLP 2025 paper and does
not introduce author-identifying content; the engagement passages do not
make first-person references to identifiable institutional or
collaborative context. Standard scope.

**Section-number drift.** The handoff that originated these drafts
referenced §2, §4, §7.4. The current numbering in
`manuscript/draft_v1.tex` matches: §2 is `sec:prompt-sensitivity`, §4 is
`sec:parallel-dev-related-work`, §7.4 is `sec:ranking-stability-results`.
If section numbering shifts before first-revision (e.g., from new
intervening subsections), the section labels are stable; locate insertion
points by `\label` rather than by section number.
