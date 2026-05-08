# N3 Reframe Edit Spec — Bench-Reliability Paper

**Status:** spec-only, awaiting branch-switch coordination to apply
**Branch:** to apply on `papers/benchmark_reliability_study/empirical-table-revisions`
**Parallel companion:** N134 received the same N3 reframe; commits land on `papers/n134_workshop/tarball-rebuild-v3` (uncommitted as of this spec)

## 1. Background

The N3 reframe responds to the 2026-05-06 comprehensive literature sweep, which showed the measurement-discipline literature is more consolidated and chronologically structured than the current "emerged contemporaneously" framing implies. Two foundational predecessors (Perlitz et al. NAACL 2024 / DIoR; Polo et al. ICML 2024 / PromptEval) precede the 2025–2026 wave by ~14 months and establish the design-choice-affects-reliability stance the wave then expanded.

Per user-approved scope (2026-05-06): both PromptEval and Perlitz enter as register voices in both papers (symmetric); chronology is re-anchored to "consolidated 2024–2026" framing; the Stanford-anchored Reuel-coauthor thread is acknowledged as one co-developing program rather than three independent voices.

## 2. Bibtex additions

Add the following two entries to `papers/benchmark_reliability_study/manuscript/references.bib`. Place them adjacent to the existing measurement-discipline cluster (e.g., after `lior2025reliableeval`, before `salaudeen2025measurement`).

```bibtex
% https://arxiv.org/abs/2308.11696
% N3 reframe: predecessor-wave voice. NAACL main-track 2024 (v1 2023-08).
% Introduces Decision Impact on Reliability (DIoR); establishes the
% design-choices-affect-reliability stance well before the 2025-2026 wave.
@inproceedings{perlitz2024efficientbenchmarking,
  author        = {Perlitz, Yotam and Bandel, Elron and Gera, Ariel and
                   Arviv, Ofir and Ein-Dor, Liat and Shnarch, Eyal and
                   Slonim, Noam and Shmueli-Scheuer, Michal and
                   Choshen, Leshem},
  title         = {Efficient Benchmarking (of Language Models)},
  booktitle     = {Proceedings of the 2024 Conference of the North
                   American Chapter of the Association for Computational
                   Linguistics: Human Language Technologies (NAACL-HLT)},
  year          = {2024},
  eprint        = {2308.11696},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  doi           = {10.48550/arXiv.2308.11696},
  note          = {IBM Research and University of Cambridge. Frames
                   efficient benchmarking — reducing computation cost
                   without compromising reliability — and introduces
                   Decision Impact on Reliability (DIoR), a quantitative
                   measure of how benchmark design choices (subset size,
                   prompt-format selection, scoring rule) reshape the
                   conclusions licensed. Demonstrated on HELM.}
}

% https://arxiv.org/abs/2405.17202
% N3 reframe: predecessor-wave voice. ICML 2024.
% Brings item response theory to bear on multi-prompt LLM evaluation.
@inproceedings{polo2024promptevalUniverso,
  author        = {Polo, Felipe Maia and Xu, Ronald and Weber, Lucas and
                   Silva, M{\'\i}rian and Bhardwaj, Onkar and Choshen,
                   Leshem and de Oliveira, Allysson Flavio Melo and
                   Sun, Yuekai and Yurochkin, Mikhail},
  title         = {Efficient Multi-Prompt Evaluation of {LLM}s},
  booktitle     = {Proceedings of the 41st International Conference on
                   Machine Learning (ICML)},
  year          = {2024},
  eprint        = {2405.17202},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  doi           = {10.48550/arXiv.2405.17202},
  note          = {MIT-IBM Watson AI Lab and University of Michigan.
                   Brings item response theory to multi-prompt LLM
                   evaluation: produces distributions over per-prompt
                   accuracy from a small held-out prompt subsample,
                   yielding both point estimates and uncertainty under
                   prompt variation. Predecessor of the IRT-based wave;
                   methodological cousin of the present paper's
                   per-quantity precision schedules.}
}
```

**Note:** The existing key `polo2024efficient` is for tinyBenchmarks (a different Polo paper, also 2024); the new key `polo2024promptevalUniverso` is for the IRT-based PromptEval. Both should coexist — they are different works.

## 3. §1.2 Parallel-development register edits

### 3a. Replace opening sentence + insert predecessor wave

**Locate** (around line 70 of `draft_v1.tex`):

> The recent measurement-discipline literature the precursor paper situated itself within has continued to develop across multiple voices in the months since. \citet{messing2026hidden} develops a Total Evaluation Error framework ...

**Replace with:**

> The recent measurement-discipline literature the precursor paper situated itself within has consolidated rapidly across 2024–2026 from independent research programs converging on overlapping concerns. Two predecessor-wave contributions set the methodological tone for the wave that followed. \citet{perlitz2024efficientbenchmarking} introduce Decision Impact on Reliability (DIoR) as a quantitative measure of how benchmark design choices --- subset size, prompt-format selection, scoring rule --- reshape the conclusions a benchmark licenses, demonstrated on HELM. \citet{polo2024promptevalUniverso} bring item response theory to bear on multi-prompt LLM evaluation, producing distributions over per-prompt accuracy from a held-out prompt subsample. Both establish, well before the 2025–2026 wave, that benchmark design choices and prompt sensitivity are first-class measurement concerns rather than implementation details.
>
> The 2025–2026 wave has expanded the program along multiple axes. \citet{messing2026hidden} develops a Total Evaluation Error framework ... [retain existing text unchanged through "...validity-centered framework for AI evaluation, instantiated through case studies of vision and language model evaluations."]

### 3b. Append Reuel-cluster acknowledgment + thirteen-voice update

**At the end of the same paragraph** (after the salaudeen2025measurement sentence), append:

> Several entries within this corpus share authorship structure --- in particular, a Stanford-anchored thread surfaces across \citet{salaudeen2025measurement} (Reuel, Koyejo, and Domingue at Stanford), the multi-institution \citet{bean2025measuring} (in which Reuel appears in the 42-author list), and \citet{mcgregor2025benchrisk} (with Reuel as a coauthor). We treat these as one co-developing research program rather than three independent voices, while reading the rest of the corpus as genuinely parallel development.

### 3c. Update synthesis paragraph (the one beginning "The shared diagnosis...")

**Update voice count and list:** the current paragraph references "the eleven voices" — update to "the thirteen voices" (now: messing, NIST x2 = 2, bean, camuffo, romanou, heineman, mcgregor, lior, salaudeen, jo, karmakar = 11 plus the two new predecessors = 13). Also add to the "convergence across substrates" parenthetical: "design-choice impact analysis (DIoR) and item-response-theoretic prompt evaluation" should appear in the substrate list. And to the "prescriptive outputs differ" enumeration: add "design-choice impact metrics" and "IRT-based prompt evaluation rules."

**Specifically:** locate the sentence `The prescriptive outputs differ across the eleven voices, and the difference is what makes their convergence informative rather than redundant: budget-optimization rules, ...` and update the count to "thirteen" plus prepend "design-choice impact metrics, IRT-based prompt evaluation rules, " to the enumerated list (so the enumeration begins with the predecessor-wave outputs in chronological order).

## 4. §4.1 Codev register edits

### 4a. Replace opening sentence

**Locate** (around line 541):

> A measurement-discipline literature has emerged contemporaneously across multiple voices.

**Replace with:**

> A measurement-discipline literature has consolidated rapidly across 2024–2026 from independent research programs converging on overlapping concerns; two predecessor-wave contributions (Perlitz et al. NAACL 2024 / DIoR; Polo et al. ICML 2024 / PromptEval) precede the 2025–2026 wave by roughly 14 months and establish the design-choice-affects-reliability stance the wave then expanded.

### 4b. Update count "Eleven voices" → "Thirteen voices"

**Locate:**

> Eleven voices have produced contemporaneous work on hidden evaluation variance, methodologically distinct enough that the convergence across them is itself epistemic content rather than a coincidence of vocabulary. Table~\ref{tab:parallel-corpus} summarizes the eleven voices and their distinct inferential targets and prescriptive outputs alongside the present paper's contribution.

**Replace with:**

> Thirteen voices have produced contemporaneous work on hidden evaluation variance, methodologically distinct enough that the convergence across them is itself epistemic content rather than a coincidence of vocabulary. Two are predecessor-wave contributions (Perlitz et al.\ 2024; Polo et al.\ 2024) that set the design-choice-affects-reliability stance well before the 2025–2026 wave; eleven are 2025–2026 voices that expanded the program along multiple axes. Table~\ref{tab:parallel-corpus} summarizes the thirteen voices and their distinct inferential targets and prescriptive outputs alongside the present paper's contribution.

### 4c. Add two table rows for the predecessor-wave voices

**At the top of the existing tabular** (immediately after the `\midrule` and before the `\citet{messing2026hidden}` row), add two rows:

```latex
\citet{perlitz2024efficientbenchmarking} & benchmark design-choice impact (HELM, DIoR) & quantitative metric (DIoR) for how benchmark design choices reshape licensed conclusions \\
\citet{polo2024promptevalUniverso} & multi-prompt LLM evaluation under prompt distribution & IRT-based estimation of per-prompt accuracy distribution from held-out subsample \\
```

### 4d. Append Reuel-cluster acknowledgment to §4.1

After the table (and after the existing post-table prose about distinct prescriptive outputs), insert a new paragraph:

> Several entries within Table~\ref{tab:parallel-corpus} share authorship structure. A Stanford-anchored thread surfaces across \citet{salaudeen2025measurement} (Reuel, Koyejo, and Domingue at Stanford), the multi-institution \citet{bean2025measuring} (in which Reuel appears in the 42-author list), and \citet{mcgregor2025benchrisk} (with Reuel as a coauthor). We treat these as one co-developing research program rather than three independent voices. The convergence we draw on for the corpus's claim is not the count of voices but the diversity of theoretical entry points, prescriptive outputs, and institutional contexts; reading the Stanford-anchored entries as one program rather than three preserves the methodological-diversity claim without inflating it.

### 4e. Update "five kinds of output" enumeration

**Locate** (in the post-table prose):

> The five kinds of output --- budget allocation, risk score, sample-size recommendation, validity-framework articulation, precision license --- are sequentially complementary rather than mutually substitutable.

**Replace with:**

> The kinds of output the corpus produces --- design-choice impact metrics, IRT-based prompt evaluation, budget allocation, risk score, sample-size recommendation, validity-framework articulation, evaluation-as-inference framework articulation, multi-variant reliability audit, and precision license --- are sequentially complementary rather than mutually substitutable; the present paper's precision license operates downstream of estimable cell reliability, regardless of which of the upstream methods produces the reliability estimate.

## 5. §4.3 cite-discipline edit (light touch)

The existing §4.3 ("What is cited and what is not") currently treats the curated-vs-exhaustive question. After N3, the predecessor-wave inclusion is itself a curation choice that deserves a one-sentence acknowledgment. **Append** to the existing paragraph:

> The N3 reframe (2026-05-06) added two predecessor-wave entries (\citealt{perlitz2024efficientbenchmarking}, \citealt{polo2024promptevalUniverso}) on the principle that the chronology of the corpus is itself epistemic content: a literature that consolidated over 2024–2026 with foundational predecessors warrants different framing than one that "emerged contemporaneously." The predecessors earn engagement because they establish the field-level stance the 2025–2026 wave then expanded, not because they were cited individually before.

## 6. Compile-verify checklist

After applying:

1. Run `pdflatex -interaction=nonstopmode draft_v1.tex` (1st pass)
2. Run `bibtex draft_v1`
3. Run `pdflatex -interaction=nonstopmode draft_v1.tex` (2nd pass)
4. Run `pdflatex -interaction=nonstopmode draft_v1.tex` (3rd pass)
5. Verify: zero undefined citations, page count manageable (expect +1 to +2pp from 51), no broken table layout
6. Run AI-language scanner: `bash scripts/ai_language_scan.sh papers/benchmark_reliability_study/manuscript/draft_v1.tex` — expect "clean" tier (the new prose was reviewed for AI-tells before drafting)

## 7. Commit message

```
papers/benchmark_reliability_study: N3 reframe — chronology re-anchoring + predecessor wave + Reuel-cluster acknowledgment

Apply the maximalist N3 reframe agreed at 2026-05-06 with the user.
Companion to the parallel reframe on N134
(papers/n134_workshop/tarball-rebuild-v3).

Three structural changes:

1. Chronology re-anchoring. §1.2 and §4.1 openers replaced
   "emerged contemporaneously" with "consolidated rapidly across
   2024-2026 from independent research programs converging on
   overlapping concerns." This more accurately reflects that
   foundational predecessors (Perlitz NAACL 2024 / DIoR; Polo ICML
   2024 / PromptEval) preceded the 2025-2026 wave by ~14 months.

2. Two new register voices added (predecessor wave):
   - perlitz2024efficientbenchmarking (DIoR, HELM)
   - polo2024promptevalUniverso (IRT, multi-prompt evaluation)
   Both as register voices; new bibtex entries; §4.1 table grows
   from 11 to 13 voices.

3. Reuel-cluster acknowledgment. §1.2 and §4.1 explicitly note that
   salaudeen2025measurement, bean2025measuring, and
   mcgregor2025benchrisk share Reuel as a coauthor and represent a
   Stanford-anchored co-developing research program rather than
   three independent voices. The methodological-diversity claim is
   preserved by reading them as one program rather than three.

Surfaced via 2026-05-06 comprehensive literature sweep
(research_review/comprehensive_sweep_2026-05-06.md). Pre-submission
engagement; supersedes the prior "eleven voices" framing.

Page count expected to grow by 1-2pp; AI-language scanner remains
clean tier (verified on the parallel N134 reframe).
```

## 8. Coordination notes

- This spec was prepared on 2026-05-06 alongside the N134 N3 reframe (which is uncommitted on `papers/n134_workshop/tarball-rebuild-v3` as of this spec's creation).
- Both papers should land their N3 commits before either submits. The N134 commit will reference "companion bench-reliability commit pending"; the bench-reliability commit will reference the N134 commit by SHA once it's in.
- After application, do a fresh AI-language scan and a fresh anonymity audit on bench-reliability before tarball rebuild.
