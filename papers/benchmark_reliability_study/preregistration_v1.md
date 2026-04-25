# Pre-Registration: Benchmark Accuracy as Measurement

**Working title:** Benchmark Accuracy as Measurement: Reliability and Tolerance Schedules for LLM Evaluation

**Version:** v1 (draft for internal review)

**Date:** 2026-04-24

**Status:** Pre-data-collection draft. No data has been collected under this protocol. Not yet finalized; v1 is an internal-review version. The final lock, with exact prompt text appendices and software-version pinning, will be v1.1 (or higher) after internal review and before any data collection begins.

**Authors:** [author list redacted pre-finalization]

**Relationship to prior work:** Successor study to the N134 / Thesis B paper ("Measurement Discipline for ML Diagnostics: A Psychometric Framework with a LoRA-Merging Case Study"). This protocol applies the same four-component framework (construct articulation, reliability estimation, precision/tolerance, confound decomposition) to a second substrate — benchmark evaluation scores — that the N134 paper's §1.1 second example named without pursuing. The present study is intended as the second public demonstration of the framework and as an independent test of the framework's substrate portability.

**Repository path:** `papers/benchmark_reliability_study/`

---

## Version history

- **v1 (2026-04-24):** initial draft protocol following synthesis of the low-GPU research planning note and second-round editorial input on study design. Two-tier design (constrained-choice primary, open-generation secondary). Instruction-tuned model included. MMLU operationalized as a pre-registered subject panel rather than as the full benchmark. Tolerance schedule formulated with explicit decimal-place precision logic.

- **v1.1-draft (2026-04-24):** research decisions resolved following internal review. (1) MMLU subject panel finalized: world_religions, elementary_mathematics, high_school_psychology, professional_accounting, global_facts — STEM and professional slots substituted to reduce small-model floor risk; psychology subject name specified. (2) Instruction-tuned model locked to Qwen2.5-1.5B-Instruct; alternatives (TinyLlama-Chat, SmolLM2-Instruct, Phi-family) not adopted. (3) GSM8K extraction variants remain two (strict + permissive); chain-of-thought scoring flagged as out-of-scope for this study, deferred to future work. (4) H3 ranking-reversal threshold raised from 10% to 20% to strengthen the decision rule against the reviewer objection of low-bar confirmation. Remaining open items for the v1.1-LOCKED promotion: prompt sourcing (24 files), model HF revision pins, dataset version hashes, prompt content hashes. This draft is not yet lock-eligible; `00_validate_config.py` will report warnings for the placeholder fields until they are filled.

---

## 1. Purpose and Scope

### 1.1 What this study does

This study estimates the measurement properties of headline LLM benchmark accuracy scores under admissible variation in prompt template, few-shot exemplar selection, and scoring rule. It produces:

1. A variance-components decomposition of benchmark accuracy into model, condition, and item sources.
2. Generalizability coefficients for benchmark scores under single-occasion and averaged-design reporting conventions.
3. A tolerance schedule specifying the decimal-place precision licensed by the observed measurement tolerance for each (benchmark, model) cell in the primary panel.
4. A ranking-stability analysis showing how often model rankings reverse across admissible measurement conditions.
5. A prescriptive reporting standard: how benchmark accuracy should be reported to reflect the measurement process that produced it.

### 1.2 What this study does not do

It does not claim that LLM benchmarks are invalid, that MMLU does not measure knowledge, or that leaderboards are meaningless. It does not claim that any specific reported accuracy in the literature is incorrect. It does not establish universal tolerances applicable to all models or all benchmarks; the tolerances derived here are specific to the tested small-to-mid-scale open-model regime under the pre-registered measurement universe. It does not propose a new benchmark, a new evaluation metric, or a new model-training protocol.

### 1.3 Relation to the measurement-discipline framework

The four framework components each receive an explicit operational implementation:

- **Construct articulation** (§2.2): distinguish latent capability, operational benchmark score, measurement occasion, and generalized score.
- **Reliability estimation** (§7.1–7.2): variance-components decomposition via mixed-effects logistic regression at item level, with generalizability coefficients under multiple averaging schemes.
- **Precision / tolerance** (§7.3): single-occasion and averaged-design tolerance schedules derived from the SEM implied by the variance decomposition.
- **Confound decomposition** (§8): enumerated methodological risks with pre-registered mitigations or explicit acknowledgments.

---

## 2. Theoretical Framing

### 2.1 Thesis

Current LLM benchmark reporting treats single-prompt accuracy as a stable point estimate of model capability. This study estimates the measurement properties that such point estimates actually support. The one-sentence thesis: **benchmark accuracy is better understood as a score sampled from a measurement design — with prompt format, few-shot exemplar selection, scoring rule, and item composition each contributing variance — than as a fixed property of a model-benchmark pair.**

Prior work has established that benchmark scores move under prompt-format variation (Sclar et al. 2023 and subsequent prompt-sensitivity literature), that multi-prompt evaluation is better understood as estimating a performance distribution than as selecting among prompts (Polo et al. 2024), and that reproducible evaluation infrastructure matters (lm-evaluation-harness, HELM). The contribution of the present study is not empirical-first on benchmark variance. It is methodological: the integrated measurement argument that ties reliability, precision, and confound structure to the substantive interpretation of the reported number. Prior work flags the problem; this study systematizes the response.

### 2.2 Construct hierarchy

The framework distinguishes four levels that benchmark-score reporting routinely conflates:

1. **Latent capability.** The cognitive or knowledge-like property the benchmark is taken to measure — "the model's knowledge of introductory biology," "the model's commonsense reasoning ability." Latent constructs are theoretical and are what substantive claims about model capability reference.
2. **Operational benchmark score.** The numeric output of a specific evaluation implementation: a specific codebase, prompt template, extraction rule, item set, evaluation harness.
3. **Measurement occasion.** One realized combination of (prompt template, few-shot exemplar draw, scoring rule) applied to a specific (model, benchmark) pair. A measurement occasion produces one aggregate score plus per-item correctness data.
4. **Generalized score.** The expected score of a model on a benchmark, generalized over the universe of admissible measurement occasions. This is the quantity a field-level claim about "the model's MMLU accuracy" implicitly references but that single-occasion reporting does not estimate.

The paper's central methodological move is to treat the generalized score as the inferential target and to estimate its reliability and error bounds under a declared measurement universe. Single-occasion reporting of a benchmark score is, in the framework's terms, a measurement without a stated measurement universe — an implicit point estimate of a generalized score with no accompanying account of what generalization is claimed.

---

## 3. Primary Research Question and Hypotheses

### 3.1 Primary research question

Under admissible variation in prompt template, few-shot exemplar selection, and scoring rule, how reliable are benchmark accuracy scores for small-to-mid-scale open language models on a fixed panel of constrained-choice benchmarks, and what decimal-place precision do they support in reporting?

### 3.2 Primary hypothesis (H1)

For at least three of the five benchmarks in the primary panel, the median single-occasion tolerance across the three primary models exceeds ±0.005 accuracy (i.e., ±0.5 percentage points), making second-decimal-in-decimal reporting questionable and third-decimal reporting unsupported under the decision rule in §3.5.

### 3.3 Secondary hypotheses

- **H2 (generalizability):** For at least three of the five primary benchmarks, the generalizability coefficient under single-occasion measurement (single prompt × single seed × single scoring rule) is less than 0.80, indicating that single-occasion evaluation is an unreliable estimator of the generalized score.
- **H3 (ranking stability):** For at least two of the five primary benchmarks, the fraction of condition-pair ranking reversals between any two of the three tested models exceeds 20%, indicating that leaderboard-style ranking claims between models can flip under admissible measurement-condition resampling. (Threshold raised from 10% to 20% at v1.1-draft per §14.5 resolution: 20% is harder to dismiss as a trivial-confirmation bar; if the actual data shows reversal rates in the 12–18% range, H3 fails at 20% and the paper reports an honest near-miss rather than a soft confirmation.)
- **H4 (MMLU subject interaction):** The MMLU subject-panel model × subject interaction explains ≥ 10% of total item-level response variance (after controlling for model main effect and subject main effect), indicating that collapsed "MMLU accuracy" masks meaningful model-specific subject variation.

### 3.4 Exploratory analyses (no hypothesis test)

- Scale sensitivity across Pythia-410M → Pythia-1.4B: does measurement-condition tolerance narrow with scale within the same model family?
- Base-vs-instruction-tuned contrast at approximately matched scale (Pythia-1.4B vs Qwen2.5-1.5B-Instruct): does tolerance differ between base and instruction-tuned regimes?
- Extension model (Mistral-7B-v0.3 or equivalent, budget permitting): are tolerance and ranking-stability numbers stable at 7B scale on a reference subset?

### 3.5 Decision rule (tolerance schedule)

A reported decimal place at precision _p_ is licensed under these data only if the estimated measurement tolerance (2 × single-occasion SEM, treated as a two-sided 95%-approximation bound) is less than half a unit at that place. Specifically:

| Reported form | Precision _p_ | Tolerance licensing condition |
|---|---|---|
| 0.742 (three decimals) | ±0.001 | tolerance < ±0.0005 |
| 0.74 (two decimals) | ±0.01 | tolerance < ±0.005 |
| 74.2% (one decimal in percentage form) | ±0.1 pp | tolerance < ±0.05 pp |
| 74% (integer percent) | ±1 pp | tolerance < ±0.5 pp |

The tolerance schedule produced by this study is the empirical instantiation of this rule for each (benchmark, model) cell in the primary panel. The schedule is reported at two averaging levels: single-occasion (what a leaderboard-style single-prompt report licenses) and full-design-averaged (what a multi-prompt multi-seed averaged report licenses).

---

## 4. Measurement Universe and Admissibility Rules

### 4.1 Admissibility of measurement conditions

A measurement condition (prompt template, few-shot exemplar draw, scoring rule) is admissible for inclusion in the measurement universe of a benchmark if and only if it satisfies all of the following:

1. **Task preservation.** The condition's prompt preserves the benchmark's intended task, answer space, and instruction semantics. A template that rephrases "answer the question" as "which of the following is true" is admissible. A template that asks the model to debate, refuse, or elaborate on the question rather than answer it is not admissible.
2. **Community provenance.** The condition is traceable to a community source: the benchmark authors' original prompt (from the benchmark's GitHub or publication), a standard evaluation-harness default (e.g., `lm-evaluation-harness`), a published variant in a peer-reviewed evaluation paper, or the HELM reference prompt. Novel prompts constructed de novo for this study are not admissible.
3. **Scoring-rule fidelity.** The scoring rule correctly identifies the benchmark's intended answer under its operationalization. Log-likelihood scoring over answer choices and generate-and-parse scoring with regex extraction are both admissible for constrained-choice benchmarks, and both are documented in community evaluation practice. A scoring rule that treats an open-ended generation as correct on semantic grounds without explicit community protocol (e.g., ad-hoc LLM-as-judge) is not admissible in the primary study.

The admissibility rule is designed to forestall the reviewer objection that observed variance reflects the authors' choice of pathological prompts or scoring rules. Every prompt and scoring rule in the study's measurement universe is documented with its source in the pre-registration lock appendix.

### 4.2 The measurement universe (primary study)

For each benchmark in the primary panel, the measurement universe is:

- **Four admissible prompt templates**, each with a documented community source (see §5.3).
- **Three few-shot exemplar draws** (random seeds over the benchmark's published development or training split; see §5.4).
- **Two scoring rules** for constrained-choice benchmarks: normalized log-likelihood over answer choices, and generate-and-parse with regex extraction (see §5.5).

Total admissible conditions per (benchmark × model) cell in the nominal case: 4 × 3 × 2 = 24. Reductions occur where a facet naturally collapses (e.g., 0-shot-standard benchmarks collapse the exemplar-seed facet to a single level).

### 4.3 The generalized score

The generalized score for a (model, benchmark) pair is the expected accuracy when a measurement condition is drawn uniformly from the benchmark's admissible condition space. This study estimates the generalized score and its standard error of measurement under two averaging levels: single-occasion (the expected accuracy of one randomly drawn condition) and full-design-averaged (the expected accuracy of the mean of all 24 conditions). The tolerance schedule derives from the SEM at each averaging level.

---

## 5. Materials

### 5.1 Models

Three models constitute the primary sample:

1. **Pythia-410M** (base): smallest same-family reference. Choice rationale: fits comfortably on CPU or a single consumer GPU; part of the Pythia scaling-ladder which enables the exploratory scale-sensitivity analysis; widely used in evaluation-methodology literature.
2. **Pythia-1.4B** (base): larger same-family reference. Same family as #1; scale difference isolates scale-from-lineage effects in the exploratory analysis.
3. **Qwen2.5-1.5B-Instruct** (instruction-tuned): cross-lineage and cross-training-paradigm reference, sized to approximately match Pythia-1.4B. Rationale: many published benchmark evaluations are conducted on instruction-tuned chat models whose prompt sensitivity is known to differ from base models; omitting instruction-tuned regimes would leave the paper's claim scope thinner than the field's actual reporting practice.

**Extension model** (optional, budget permitting): Mistral-7B-v0.3 or equivalent 7B-scale open model, evaluated on a reference subset of the primary panel. Pre-registered as exploratory and not part of the primary-hypothesis tests.

All models are loaded at their original public checkpoints without fine-tuning, quantization, or further adaptation. Inference is performed in full precision (or bfloat16 where standard) using a pinned transformers version recorded in the pre-registration lock.

### 5.2 Benchmarks

**Tier 1 — Primary panel (constrained-choice, five benchmarks):**

1. **ARC-Challenge.** Multiple-choice science questions. Standard 25-shot or 0-shot; this study uses 5-shot as the primary few-shot setting for uniformity across the panel.
2. **HellaSwag.** Constrained commonsense completion. 10-shot is common in the literature; this study uses 5-shot for uniformity.
3. **TruthfulQA-MC.** Multiple-choice truthfulness. Standard 0-shot; the exemplar-seed facet collapses to a single level for this benchmark.
4. **MMLU subject panel.** Five pre-registered subjects, one from each of the MMLU subject groupings: **world_religions** (humanities), **elementary_mathematics** (STEM), **high_school_psychology** (social sciences), **professional_accounting** (professional), **global_facts** (cross-domain). Full MMLU is not used; the subject panel is locked at pre-registration. STEM and professional slots were chosen over high_school_mathematics and professional_medicine respectively to reduce small-model floor risk (at 3 models × 25% chance, a floor-pinned subject produces no measurable model × subject interaction). Psychology slot specified as high_school_psychology (largest item count among MMLU psychology variants) to maximize item-level decomposition power.
5. **Winogrande.** Constrained-choice commonsense reasoning. Standard 5-shot.

Benchmark choice rationale: the five are all constrained-choice, enabling uniform application of both log-likelihood and generate-and-parse scoring rules. GSM8K and other open-generation benchmarks are treated as secondary (Tier 2) because the scoring-rule facet is qualitatively different (answer parsing becomes a primary measurement-instrument feature rather than a scoring-rule choice within a stable operationalization).

**Tier 2 — Secondary study (open-generation stress case):**

6. **GSM8K.** Open-generation arithmetic reasoning. Analyzed as a separate case study demonstrating that scoring-rule sensitivity is more severe for open-generation benchmarks, where the extraction protocol is itself part of the measurement instrument. Not included in the primary-hypothesis tests; reported in a separate section with its own analysis structure (§7.6).

### 5.3 Prompt templates

For each of the six benchmarks, four admissible prompt templates are locked at pre-registration, each with a documented source. The canonical source set per benchmark:

- **P1: original benchmark authors' prompt** (from the benchmark's GitHub repository or publication).
- **P2: `lm-evaluation-harness` default.**
- **P3: HELM reference prompt** where available; otherwise a widely cited community variant from a peer-reviewed evaluation paper.
- **P4: minimal direct-instruction prompt** — the shortest admissible variant that preserves the task construct, intended to represent an evaluator making the simplest possible prompt choice.

Exact prompt text for each benchmark × template combination is documented in Appendix A of the finalized pre-registration (the lock appendix), with commit hashes of the originating sources.

### 5.4 Few-shot exemplar protocol

Three seeds — 42, 123, 2024 — are used for random selection of 5-shot exemplars from the benchmark's published development split (or training split if no development split is available). Within a (benchmark, model, seed) cell, the same seed draws the same exemplars across all four prompt templates, so that prompt-format variance is isolated from exemplar-selection variance.

Benchmarks with no natural few-shot protocol (TruthfulQA-MC) are evaluated at 0-shot; the exemplar-seed facet collapses to a single level, and the variance decomposition omits the exemplar facet for that benchmark.

### 5.5 Scoring rules

Two scoring rules are used for each constrained-choice benchmark in the primary panel:

- **Rule A — normalized log-likelihood (LL).** For each item, compute the sum of log-probabilities of each candidate answer's tokens conditioned on the prompt. Normalize by token count (length-normalized LL is the standard in `lm-evaluation-harness` for several benchmarks; use the length-normalized variant where the harness does). Select the argmax. Accuracy is the fraction of items where the selected answer matches the labeled correct answer.
- **Rule B — generate-and-parse (G&P).** Prompt the model to generate an answer, extract the answer via benchmark-specific regex, and compare to the labeled correct answer after standard normalization (case, whitespace, punctuation).

The two rules are different operationalizations of the benchmark task. Log-likelihood scoring measures which candidate the model assigns highest probability, conditioning on the prompt. Generate-and-parse measures whether the model can follow the prompt's instruction format to produce a parseable answer. Both are legitimate, widely used, and admissible under §4.1.

For GSM8K (Tier 2), only Rule B applies. Within Rule B for GSM8K, two extraction variants are tested: strict exact-match on the numerical answer, and a more permissive regex accepting common formatting variants (trailing units, commas in large numbers, etc.).

---

## 6. Design and Data Hierarchy

### 6.1 Full factorial for primary study

The primary study is a fully crossed design over:

- 3 models × 5 benchmarks × 4 prompt templates × 3 few-shot seeds × 2 scoring rules.

Nominal total of condition-level scores: 3 × 5 × 4 × 3 × 2 = 360. Actual total is lower where a benchmark collapses a facet (TruthfulQA-MC collapses the seed facet from 3 to 1, reducing its contribution from 24 to 8 conditions per model).

Per (benchmark × model) cell, the nominal count of admissible conditions is 24; the reduced count ranges from 8 (TruthfulQA-MC) to 24 (all other primary benchmarks). This is the condition-level replicate count; aggregate-score G-theory operates at this scale.

### 6.2 Data hierarchy

Item-level data are the primary analysis unit:

- Item _i_, for _i_ ∈ {1, ..., N_benchmark}, where N_benchmark is the benchmark's evaluation set size.
- Nested within measurement condition (prompt, seed, scoring rule).
- Nested within (benchmark, model) cell.

Each (item × condition) yields one binary correctness outcome. Per (benchmark × model) cell, the item-level sample size is N_benchmark × 24 (nominal; fewer where facets collapse). Across benchmarks the item-level sample ranges from roughly 800 × 24 = 19,200 (Winogrande dev) to roughly 1,500 × 8 = 12,000 (TruthfulQA-MC) to 100 × 24 × 5 = 12,000 (MMLU subject panel with 5 subjects of ~100 items each).

Item-level mixed-effects logistic regression is the primary analysis vehicle (§7.1). Aggregate-score G-theory and condition-level ANOVA are the companion analyses for tolerance-schedule derivation (§7.3).

### 6.3 Replicate-structure clarification

The design yields 24 condition-level scores per nominal (benchmark × model) cell. Aggregate-score G-theory, condition-level ANOVA, and tolerance-schedule derivation operate at this scale. The per-cell condition-level sample (24 or fewer) is small; statistical power for variance-components estimation derives primarily from item-level data, which are an order of magnitude more numerous. Tolerance-schedule SEM is derived from aggregate-score variance components but is reported with a companion figure showing the item-level binomial uncertainty contribution as a check on interpretation.

---

## 7. Pre-Registered Analyses

### 7.1 Analysis 1: Variance components

**Method.** For each benchmark, fit a mixed-effects logistic regression at the item level. Fixed effects: model (3 levels). Random effects: prompt template, few-shot exemplar seed, scoring rule, item (nested within benchmark), model × prompt interaction, model × scoring rule interaction. The regression models the log-odds of item correctness.

Variance components are extracted as proportions of total item-level response variance. Intraclass correlations (ICCs) for each random-effect facet are reported.

**Companion.** For interpretability, an aggregate-score ANOVA / G-theory decomposition is also reported: for each (benchmark, model), the variance in condition-level accuracy (24 scores) is decomposed into prompt, seed, scoring-rule, and residual components via a fully-crossed random-effects model.

**Reporting.** A per-benchmark variance-components table, with both item-level mixed-effects estimates and aggregate-score G-theory estimates. Interpretation focuses on proportions: how much of the observed score variance is attributable to each measurement-design facet vs. to stable model differences and to item sampling.

### 7.2 Analysis 2: Generalizability coefficients

**Method.** For each benchmark, estimate the generalizability coefficient (ratio of systematic model variance to total score variance) under multiple averaging schemes:

- Single-occasion: single prompt × single seed × single scoring rule.
- Prompt-averaged: average accuracy over 4 prompts (seed and scoring rule held).
- Seed-averaged: average accuracy over 3 seeds (prompt and scoring rule held).
- Scoring-rule-averaged: average accuracy over 2 scoring rules (prompt and seed held).
- Full-design-averaged: average accuracy over all 24 conditions.

**Reporting.** A per-benchmark generalizability-coefficient table with the five averaging schemes as columns and models as rows. The comparison between single-occasion and full-design-averaged coefficients is the paper's prescriptive lever: if single-occasion coefficients are low but full-design coefficients are high, the implication is that benchmark reports should declare an averaging design rather than quote a single-condition score.

### 7.3 Analysis 3: Tolerance schedules

**Method.** For each (benchmark, model) cell, derive from the variance-components decomposition:

- Single-occasion SEM: standard error of the aggregate accuracy for one randomly drawn condition.
- Single-occasion tolerance: 2 × single-occasion SEM, reported both in decimal-accuracy and percentage-point form.
- 4-prompt-averaged tolerance: 2 × SEM of the mean of 4 prompt-varying conditions (scoring-rule and seed held).
- Full-design-averaged tolerance: 2 × SEM of the mean over all 24 conditions.

Under the §3.5 decision rule, each tolerance value licenses a maximum decimal-place reporting precision. The tolerance-schedule table reports all three tolerances per cell and the corresponding licensed reporting precision.

**Cross-model median summary.** For each benchmark, the median of the three models' single-occasion tolerances is computed. This is the summary statistic for the H1 primary hypothesis test (§3.2).

### 7.4 Analysis 4: Ranking stability

**Method.** For each benchmark, compute:

- Kendall's τ between model rankings across all pairs of admissible conditions.
- For each pair of models (three pairs among the three primary models), the fraction of conditions in which the pair's ranking differs from the overall-mean ranking. This is the "ranking reversal fraction" for the model pair.
- Bootstrap-resampled probability that each model's accuracy exceeds each other model's accuracy under measurement-condition resampling.

**Reporting.** A ranking-stability figure showing, for each benchmark, the distribution of model pairs' ranking reversals across conditions. The operational interpretation: if ranking-reversal fractions are non-trivial (H3 threshold: ≥ 10% for at least two benchmarks), single-occasion comparisons between models are not stable, and field-level claims of the form "model A outperforms model B on benchmark X" are under-specified without a measurement universe.

### 7.5 Analysis 5: MMLU subject decomposition

**Method.** Within the MMLU subject panel (five subjects), decompose item-level response variance into:

- Model main effect (3 levels).
- Subject main effect (5 levels).
- Prompt main effect (4 levels).
- Model × subject interaction.
- Prompt × subject interaction.
- Residual / item-within-subject.

**Reporting.** A variance-components table; a model × subject interaction heat map; the estimated proportion of model × subject interaction variance (H4 primary check). Interpretation: if the model × subject interaction is substantial, "collapsed MMLU accuracy" averages over heterogeneous subject-specific model behavior, and the generalized-score interpretation of a single MMLU number is weaker than the single-number reporting implies.

### 7.6 Analysis 6: GSM8K open-generation case (Tier 2)

**Method.** On GSM8K only, across the 3 models and 4 prompts, evaluate both strict-exact-match and permissive-regex scoring. The factorial is 3 × 4 × 3 × 2 = 72 condition-level scores. Estimate the same variance-components decomposition as §7.1, with scoring-rule replaced by extraction-variant.

**Reporting.** GSM8K is reported as a standalone section with a narrower framing: "open-generation benchmarks introduce a further source of measurement variance — the answer-extraction rule — which is not merely a scoring choice but is part of the measurement instrument itself. The GSM8K case demonstrates this in the direction suggested by the primary study's scoring-rule facet."

GSM8K is not included in the primary-hypothesis tests. Its tolerance schedule is reported separately.

---

## 8. Confound Decomposition and Methodological Risks

The paper's confound structure is enumerated pre-registered, each with an explicit mitigation (or explicit acknowledgment where mitigation is not offered).

### 8.1 Risk: "Bad prompts are driving the observed variance"

**Mitigation.** The admissibility rule in §4.1. Every prompt in the measurement universe is traceable to a community source (benchmark authors, standard harness, published variant, HELM reference, or minimal direct variant). No prompts are constructed de novo by the authors. The variance observed is the variance a reasonable evaluator following community practice could encounter when choosing among existing prompts. Reviewers skeptical of the finding can audit the four-prompt source list for each benchmark.

### 8.2 Risk: "Small models are prompt-sensitive because they are weak"

**Mitigation.** The paper's object is the measurement process, not frontier model ranking. The claim scope is pre-registered (§9.1) as small-to-mid-scale open models. One instruction-tuned small model is included to check whether the measurement-condition variance is reduced for instruction-tuned regimes (which many published evaluations use). Within-family scaling across Pythia-410M → Pythia-1.4B provides an exploratory scale check. The optional 7B extension, if budget permits, provides a reference point for whether tolerance narrows at the frontier-open-model scale.

### 8.3 Risk: "Item-sampling variance is conflated with condition variance"

**Mitigation.** The variance-components decomposition (§7.1) explicitly separates item-level variance from condition-level variance. The tolerance schedule is derived from condition-level SEM; item-level binomial uncertainty is reported as a companion figure. The universe-of-generalization for the tolerance schedule is admissible measurement conditions, not benchmark items (items are fixed; the benchmark's item set is treated as given).

### 8.4 Risk: "Scoring rules are incommensurable across log-likelihood and generate-and-parse"

**Acknowledgment, not mitigation.** This is a deliberate feature of the design. The two scoring rules represent different operationalizations of the benchmark task, and observing divergence between them is itself a measurement-discipline finding. The paper's construct-validity argument treats the scoring-rule choice as a facet of the measurement universe rather than a technical nuisance to be averaged away. If the two rules produce systematically different accuracies for the same (model, benchmark) pair, the conclusion is that "the benchmark score" is under-specified without declaring the scoring rule.

### 8.5 Risk: "The variance is dominated by a single prompt outlier"

**Mitigation.** For each benchmark's variance-components decomposition, the per-prompt marginal mean and per-prompt item-level accuracy are reported as supplementary figures. If any single prompt is a clear outlier driving most of the observed variance, this is surfaced rather than hidden. A leave-one-prompt-out sensitivity analysis is run for each benchmark as a robustness check.

### 8.6 Risk: "Instruction-tuned and base models are not comparable"

**Acknowledgment with design feature.** The base-vs-instruction-tuned contrast is an exploratory analysis, not a primary hypothesis test. The two base models (Pythia-410M, Pythia-1.4B) enable same-family tolerance comparisons without this confound. The instruction-tuned model (Qwen2.5-1.5B-Instruct) provides coverage of the regime in which most field reporting occurs but is not treated as commensurable with the base Pythia models for scale-sensitivity purposes.

---

## 9. Scope Claims and Non-Claims

### 9.1 Scope of claims

This study's tolerance schedules, reliability coefficients, and ranking-stability numbers apply to:

- The three tested models: Pythia-410M, Pythia-1.4B, Qwen2.5-1.5B-Instruct (plus the optional 7B extension if executed).
- The five primary benchmarks: ARC-Challenge, HellaSwag, TruthfulQA-MC, MMLU subject panel, Winogrande.
- The declared measurement universe of admissible prompts (four per benchmark, community-sourced), few-shot seeds (three per non-collapsed facet), and scoring rules (log-likelihood and generate-and-parse).

The tolerance schedule derived here is an instantiation of the §3.5 decision rule on this specific regime. The rule itself is general and applicable to other benchmark-model combinations, but the numerical tolerances are not transferable without re-estimation.

### 9.2 What the study does not claim

- That any specific reported accuracy in the literature is "wrong."
- That the observed tolerance schedule generalizes to frontier closed models (GPT, Claude, Gemini frontier checkpoints, etc.).
- That MMLU, ARC, HellaSwag, TruthfulQA, or Winogrande fails to measure anything of substance.
- That current benchmark reporting should be abandoned.
- That the measurement universe defined here is exhaustive of admissible evaluation practice. Many other admissible prompts, scoring rules, and exemplar protocols exist; the universe used here is a principled sample, not a census.
- That reliability alone tells the full story of benchmark validity; construct validity in the full Messick sense requires additional argumentation (convergent, discriminant, predictive), not attempted in the present study.

### 9.3 What the study prescribes

An actionable reporting recommendation, structured as the paper's concrete contribution to field practice. A measurement-disciplined benchmark report, in the sense this study argues for, includes:

1. A declaration of the measurement design under which the reported score was obtained (prompt template, few-shot protocol, scoring rule).
2. Either (i) the score reported at a decimal-place precision justified by the single-occasion tolerance for that benchmark and model scale, or (ii) a generalized score averaged over a declared design with an explicit tolerance interval.
3. If model comparisons are made, an explicit statement of whether the ranking is stable under measurement-condition resampling, or a tolerance interval on the difference.

Single-occasion point estimates to three decimals, without accompanying tolerance declaration or averaging design, overclaim precision relative to what the measurement process supports. This is the study's bottom-line normative claim, and the tolerance-schedule table is its empirical instantiation.

---

## 10. Timeline and Budget

### 10.1 Timeline

| Phase | Duration | Artifacts |
|---|---|---|
| Pre-registration lock | 1 week | v1.1 locked pre-reg with prompt-appendix; committed to repo; timestamped |
| Infrastructure setup | 1 week | harness code, evaluation pipeline, per-benchmark prompt templates verified |
| Primary data collection | 3–4 weeks | 360 condition-level runs across primary panel; per-item correctness data |
| Analysis | 2 weeks | all five primary analyses executed; tolerance-schedule table; figures |
| Extension data collection (optional, budget-permitting) | 1–2 weeks | 7B reference extension on subset of primary panel |
| Writing | 4–5 weeks | full manuscript + supplementary materials |
| Internal review | 1–2 weeks | editorial pass analogous to N134 pre-submission passes |
| Submission | — | TMLR or venue to be determined at writing phase |

Total: 12–15 weeks from pre-registration lock to submission-ready manuscript.

### 10.2 Budget

The primary study is designed to be GPU-inexpensive. Small-model inference on Pythia-410M, Pythia-1.4B, and Qwen2.5-1.5B-Instruct across the full primary factorial is estimated at 20–40 GPU-hours on a consumer-tier GPU (RTX 4090 class or similar), or substantially more on CPU (100–200 CPU-hours for the smallest benchmarks is feasible). Dollar cost on Colab or RunPod tier: $5–$30 for the full primary study.

The optional 7B extension (Mistral-7B-v0.3) on a reference subset (one or two benchmarks × all conditions, or all benchmarks × a reduced condition set) is estimated at 40–80 GPU-hours on an A100, roughly $60–$150 depending on provider. The extension is not required for the primary-hypothesis tests.

### 10.3 Budget tier framing

The study is deliberately designed to be executable at the lowest budget tier:

- **$0–$50 tier:** primary study on the three small models, executed on a single consumer GPU or aggressively on CPU. Produces the complete primary-panel tolerance schedule, all five primary analyses, and the GSM8K stress case. This tier is sufficient for a standalone paper.
- **$50–$200 tier:** primary study plus the 7B reference extension. Adds scale-sensitivity context without changing the primary claim structure.
- **$200+ tier:** not envisioned; the study is constructed so that additional budget does not materially strengthen the contribution.

---

## 11. Deviations Protocol

Any deviation from this pre-registration discovered during execution must be documented in a separate `deviations.md` file committed to the repository at the time of the deviation, and cited in the paper's "Deviations from Pre-Registration" section. Deviations include (but are not limited to):

- Changes to the benchmark set or MMLU subject panel.
- Changes to the prompt template set or the §4.1 admissibility rule.
- Substitution of models.
- Changes to the decision rule (§3.5) or hypothesis thresholds (§3.2–3.3).
- Additional analyses not listed in §7.

The authors commit to reporting all pre-registered analyses regardless of outcome. Post-hoc analyses are permitted but must be labeled as such and assigned hypothesis-generating rather than confirmatory evidential status (cf. the N134 paper's §7.1 post-hoc framing).

---

## 12. Deliverables

### 12.1 Primary artifacts produced

- Pre-registration lock document (this file, at v1.1 finalized form) with prompt-appendix.
- Per-benchmark variance-components tables.
- Per-benchmark generalizability-coefficient tables.
- Per-benchmark tolerance schedules (single-occasion and averaged).
- Ranking-stability figures and numbers.
- MMLU subject-decomposition results.
- GSM8K secondary case results.
- Prescriptive reporting standard (the concrete norm-change close).

### 12.2 Replication package

- Full prompt text for all 4 × 6 = 24 benchmark × prompt combinations.
- Few-shot exemplar draws for all 3 × 6 = 18 (benchmark, seed) cells.
- Scoring-rule implementation code, pinned to a specific `lm-evaluation-harness` (or equivalent) version.
- Item-level correctness data for all primary conditions (CSV or JSONL).
- Condition-level aggregate scores (CSV).
- Analysis scripts reproducing all five primary analyses from the item-level data.
- Environment pin: Python version, library versions, CUDA/CPU details.

### 12.3 Not attached

- Model weights (publicly available; linked by canonical Hugging Face path).
- Benchmark item sets (publicly available; linked by canonical source).
- Raw model-generation logs (potentially multi-gigabyte; held in offline storage, available on request).

---

## 13. Relationship to the N134 Paper

This study is the second worked demonstration of the measurement-discipline framework articulated in the N134 / Thesis B paper. The explicit continuities:

- The four-component framework (construct articulation, reliability, precision, confound decomposition) is instantiated here on a different substrate, supporting the N134 claim that the framework is substrate-portable.
- The pre-registration-and-tolerance-schedule discipline used in N134's §4 is applied here, including explicit decision rules with numeric thresholds committed to before data collection.
- The "two kinds of findings" framing from N134's §6.4 is available for organizing the paper's results: the tolerance-schedule and ranking-stability analyses are calibration-and-modesty findings (the field's current reporting exceeds what the data supports); the construct-hierarchy articulation is closer to a discovery-like-in-the-narrower-reporting-sense finding.

The paper's positioning explicitly cites the N134 paper as the methodological precedent and as the source of the framework being applied. If the N134 paper is accepted at TMLR, this paper is a natural submission to the same venue; if it is rejected, this paper is submittable standalone with the framework introduced in-paper at the length the N134 paper used for its framework section.

The two papers together constitute a two-paper sequence demonstrating the measurement-discipline thesis across two substrates (LoRA diagnostics at decoder scale; benchmark-evaluation accuracy at small-to-mid-open-model scale). Neither paper depends on the other for its primary contribution, but the pair demonstrates the portability claim more strongly than either alone.

---

## 14. Open Questions for Internal Review

Items flagged for resolution before pre-registration is locked at v1.1. Resolution status as of 2026-04-24 marked per item.

1. **MMLU subject panel selection.** *Resolved at v1.1-draft (2026-04-24):* panel locked as **world_religions, elementary_mathematics, high_school_psychology, professional_accounting, global_facts**. STEM slot moved from high_school_mathematics to elementary_mathematics to reduce small-model floor risk on abstract-algebra-heavy content; professional slot moved from professional_medicine to professional_accounting for the same reason; psychology slot specified as high_school_psychology (largest item count among psychology variants). See §5.2 for full rationale.

2. **Instruction-tuned model choice.** *Resolved at v1.1-draft (2026-04-24):* **Qwen2.5-1.5B-Instruct** confirmed. Alternatives considered: TinyLlama-1.1B-Chat (ruled out: chance-level on MMLU, below floor), SmolLM2-1.7B-Instruct (defensible backup — Apache 2.0 license cleaner for redistribution), Phi-family (Microsoft license constraints). Qwen2.5 chosen on balance of benchmark-regime signal strength, modern training recipe (distinct from Pythia lineage), and tractable research-use licensing.

3. **Optional 7B extension target.** *Deferred:* Mistral-7B-v0.3 remains the proposed choice but execution is contingent on budget availability; formal lock not required for the primary study.

4. **Scoring rule B for GSM8K.** *Resolved at v1.1-draft (2026-04-24):* two variants only (strict exact-match + permissive regex). Chain-of-thought scoring not added. Rationale: CoT is coupled to prompt choice (CoT prompts emit reasoning chains; scoring requires chain-of-thought separation), so adding CoT as a third extraction variant would collapse the prompt × extraction factorial in ways that make the decomposition harder to interpret. The manuscript will note CoT as a further source of measurement variance not included in the present tolerance schedule and flag it as a direction for future work.

5. **Ranking-stability threshold for H3.** *Resolved at v1.1-draft (2026-04-24):* **20%**. Rationale: 10% is close enough to the measurement-noise floor that a skeptical reviewer could argue the threshold is set too low; 20% is unambiguously substantial and cannot be dismissed as trivial. If actual reversal rates fall in the 12–18% range (a plausible outcome), H3 fails at 20% and the paper reports that as an honest near-miss — consistent with the measurement-discipline register of preferring hard thresholds with honest reporting over soft thresholds that confirm easily.

6. **Exemplar-seed protocol.** *Deferred:* default (unbiased random draw, 3 seeds) locked. No reason to stratify has emerged.

**Remaining for v1.1-LOCKED promotion (operational):** prompt sourcing (24 files), model HF revision pins, dataset version hashes, prompt content hashes. These are not research decisions but operational tasks that must complete before `00_validate_config.py` passes with no placeholder warnings.

---

*End of v1 draft. No data has been collected under this protocol.*
