# Reliability and Validity of Feature-Level Interpretability: A Psychometric Framework

**Working outline, v0.** Flag decision points with `[DECIDE]`. Flag scope-risk areas with `[SCOPE]`. Flag things that need John's specific judgment with `[JUDGMENT]`.

**Target length:** 25–30 pages plus appendices.
**Target venue options:** `[DECIDE]` NeurIPS Datasets & Benchmarks track (good fit, short lead time), ICLR methods track, JMLR methods paper (longer, higher prestige, fits audience), BlackboxNLP / NeurIPS interpretability workshop (faster turnaround, narrower audience), or bridge venue like *Psychological Methods* / *Behavior Research Methods* (unusual, could be strategic for discipline-crossing credibility).
**Timeline:** 4–6 months single-author.
**Compute budget:** modest — open-model SAE analysis on Gemma-2-2B or similar, multiple seeds, activation collection and statistical analysis on CPU after GPU feature extraction. Estimated $500–1500 in RunPod costs.

---

## Thesis in one paragraph

Interpretability research increasingly makes claims of the form: *feature F represents concept C, and intervening on F causally affects behavior B.* This is structurally identical to the claim psychometric measurement theory has refined over roughly a century: *instrument I measures construct C, and C predicts outcome O.* Psychometrics developed specific machinery — reliability coefficients, standard error of measurement, minimal detectable change, convergent and discriminant validity, differential item functioning, structural validity via factor analysis — precisely for adjudicating such claims. Interpretability has adopted some of this machinery implicitly (reproducibility across seeds, qualitative construct validation via examples) but lacks systematic psychometric discipline. We translate classical test theory and modern validity theory into the interpretability setting, demonstrate the framework empirically on emotion features in a small open model, and identify specific gaps in current practice that adopting the framework would close.

---

## §1. Introduction (2 pages)

**Goals:** establish the problem, preview the contribution, situate in the moment.

**Opening move.** Lead with Lindsey et al. 2026 as the motivating example. The paper reports that emotion concept representations in Claude Sonnet 4.5 "causally influence the LLM's outputs, including Claude's preferences and its rate of exhibiting misaligned behaviors such as reward hacking, blackmail, and sycophancy." This is an important finding, and taking it seriously *as a measurement claim* opens a set of questions that the paper does not address because they are outside its scope: How reliable is the identification of the "anger" feature across training runs of the feature extractor? How much of reported activation variance is true-score vs. measurement error? Does activating the "anger" feature correlate with related features (convergent validity) and dissociate from "frustration" (discriminant validity)? Do contextual factors (prompt language, first vs. third person) produce activation patterns for reasons unrelated to the emotion itself?

**Framing move.** Interpretability researchers may initially find the framing foreign — psychometric theory was developed for measurement of human psychological constructs. We argue that the relevant equivalence is not between models and humans but between measurement-claim structures. Any claim of the form "observable X reflects underlying construct Y and predicts outcome Z" is a measurement-validity claim regardless of the substrate. The machinery that handles such claims well for human psychology applies with appropriate translation to model interpretability.

**Contributions list:**
1. A translation of classical test theory and validity theory concepts into interpretability-native definitions.
2. Formal estimators for reliability, SEM, MDC, discriminant validity, and differential context functioning as applied to feature-level measurements.
3. Empirical demonstration on emotion-adjacent features in [`[DECIDE]` Gemma-2-2B with Gemma Scope SAEs / Llama-3.2-3B with open SAEs / alternative open model].
4. Identification of specific gaps in current interpretability practice, with concrete reporting-standard recommendations.
5. Released code for computing the proposed psychometric statistics on any SAE-style feature set.

**What the paper is not.** Not a critique of Lindsey et al. or any specific interpretability paper. Not a claim about whether LLMs have subjective emotions. Not a proof that current features are wrong. A methodological contribution that makes future feature-level claims more rigorously defensible.

---

## §2. Background (4 pages)

### §2.1 Feature-level interpretability: a brief history (1 page)

- Superposition hypothesis (Elhage et al. 2022)
- Sparse dictionary learning / SAEs (Cunningham et al. 2023, Bricken et al. 2023)
- Scaling SAEs (Templeton et al. 2024, Gao et al. 2024)
- Feature circuits and causal methods (Marks et al. 2024, Syed et al. 2024)
- Lindsey et al. 2026 as current state of the art for concept-level features with causal claims

Keep this tight and citation-dense. The interpretability audience doesn't need a tutorial; the psychometrics audience needs enough orientation to read the rest.

### §2.2 Classical test theory and generalizability theory (1.5 pages)

- **CTT basics.** Observed score = true score + error; reliability = var(true) / var(observed); reliability ceiling on validity.
- **The move to generalizability theory.** Cronbach, Gleser, Nanda, Rajaratnam 1972. Rather than one undifferentiated "error," partition variance across facets: persons, occasions, items, raters. The G-study / D-study distinction.
- **Why generalizability theory matters for interpretability specifically.** Interpretability has multiple distinct sources of measurement variance: random seed of extraction method, choice of extraction method (SAE vs. probe vs. gradient attribution), prompt context, token position, layer choice. These are *facets*, and generalizability theory is built precisely for multi-faceted measurement.

### §2.3 Validity theory: Messick, Borsboom, and modern approaches (1 page)

- Messick 1989: the unified view. Construct validity subsumes content, criterion, convergent, discriminant.
- Borsboom 2005: validity as attribute-existence + causal-contribution. "The test is valid if variation in the attribute causally produces variation in the test score."
- Note the match to interpretability's causal intervention methods: activation patching and ablation are precisely attempts to establish Borsboom-style validity.
- `[JUDGMENT]` Decide whether to include a brief discussion of construct-representation vs. nomothetic-network approaches — probably yes for the interpretability audience who may not know this terminology.

### §2.4 What interpretability already does, implicitly (0.5 pages)

- Feature reproducibility studies (Bricken et al. 2023 noted feature stability across training runs)
- Qualitative construct validation via top-activating examples
- Causal intervention as validity evidence
- Activation-based feature clustering as implicit factor analysis
- But: no systematic reporting of reliability coefficients, no SEM estimates, no DIF, no formal factor analysis of feature structures

---

## §3. The Framework (6 pages)

This is the core methodological contribution. Each subsection: formal psychometric concept → interpretability-native definition → estimator → practical example.

### §3.1 Reliability (1.5 pages)

Three subtypes, each with a distinct interpretability translation:

- **Test-retest reliability.** Same feature, same extraction method, different training run / random seed. Estimator: ICC(2,1) on feature activations across matched examples.
- **Internal consistency.** For claimed multi-feature constructs (e.g., "the emotion of anger is represented by features F1, F2, F3"), Cronbach's alpha on activations across examples. If alpha < 0.7, the "anger" cluster is not a consistent measurement of a single construct.
- **Inter-method reliability.** Same claimed concept via different extraction methods (SAE vs. supervised probe vs. activation steering vector). Estimator: correlation or generalized reliability coefficient across methods on a matched example set.

Flag: **each of these requires infrastructure interpretability researchers typically don't build.** Test-retest reliability needs matched feature identification across training runs — a non-trivial problem given SAE feature permutation ambiguity. Address this head-on: matching via activation similarity, the problem of features that "split" or "merge" across runs, and how to report results when matching is imperfect.

### §3.2 Standard Error of Measurement and Minimal Detectable Change (1 page)

SEM = SD × √(1 − reliability). Practical translation: given an observed feature activation, the 95% CI on the "true" activation is roughly ±1.96 × SEM.

MDC = SEM × √2 × 1.96. The smallest activation change between two measurements that represents real change rather than noise.

**Why this matters for causal intervention claims.** If an ablation reduces the "anger" feature from activation 3.2 to 1.8 and produces a behavior change, but MDC for that feature is 1.5, the intervention is within the noise band of measurement. Reporting MDC alongside intervention effect sizes is a concrete discipline improvement.

### §3.3 Construct validity: convergent and discriminant (1.5 pages)

**Convergent validity.** Features claimed to measure the same construct should correlate more with each other than with features measuring different constructs. Estimator: correlation matrix of feature activations on a balanced example set, tested against Heterotrait-Monomethod Matrix predictions (Campbell & Fiske 1959).

**Discriminant validity.** Features claimed to measure different constructs should not correlate at activation level more than chance. This is where current interpretability practice is weakest — related-but-distinct constructs (anger / frustration, sadness / disappointment, fear / anxiety) are rarely tested.

**The monotrait-multimethod approach.** For the emotion of "anger": extract the putative anger feature via SAE, via linear probe on labeled anger examples, via gradient attribution. If all three correlate strongly on a held-out set, convergent validity is established. If not, one or more extraction methods is not actually finding "anger."

Include worked example with synthetic features where we know the ground truth, then the empirical application in §4.

### §3.4 Differential Context Functioning (1 page)

DIF in classical psychometrics: the same item functions differently for different groups (e.g., men vs. women) for reasons unrelated to the measured construct.

Translation: **Differential Context Functioning (DCF).** The same feature activates differently in different contexts (first-person vs. third-person, different languages, different system prompts) for reasons unrelated to the measured concept.

Estimator: conditional activation analysis — regress feature activation on (concept present? yes/no) × (context). The interaction term estimates DCF.

**Why this matters.** A feature that activates for "anger in English" but not "anger in Japanese" may not be measuring anger; it may be measuring an English-specific emotional lexical field. Without DCF analysis, you cannot tell.

### §3.5 Structural validity: factor analysis of feature structures (1 page)

For emotion specifically, there are multiple competing theoretical structures:
- Ekman 1992: six basic emotions (anger, disgust, fear, happiness, sadness, surprise)
- Plutchik 1980: eight primary emotions arranged dimensionally
- Russell 1980: two-dimensional valence-arousal space
- OCC model (Ortony, Clore, Collins 1988): cognitive appraisal structure

If an interpretability researcher identifies N emotion features, exploratory factor analysis on their activations across an emotion-balanced stimulus set can test which theoretical structure the feature set best matches. If the factor structure matches none of the theories, that's interpretable data. If it matches dimensional valence-arousal better than discrete basic emotions, that's a substantive finding about the model's emotion representation.

**This is where the framework moves beyond method-correction into substantive contribution.** Psychometric factor analysis is well-developed (exploratory factor analysis, confirmatory factor analysis, ESEM). Applying it to feature structures produces genuinely new knowledge.

### §3.6 What doesn't translate cleanly (0.5 pages)

Honest about disanalogies:

- **No sampling from a population in the CTT sense.** We're measuring a specific model, not a sample of models. Generalizability-theory framing handles this: the facets of generalization are seeds, prompts, contexts — not persons.
- **No "true score" in the philosophical sense.** Psychometrics assumes an underlying true value exists. For model features, it's debatable whether there's a "true" concept representation or only the operational definition provided by the extraction method.
- **Construct irrelevant variance is harder to identify.** In psychology, you know a priori that math anxiety shouldn't affect reading comprehension scores. In interpretability, the space of potentially confounding constructs is vast and not fully enumerable.

These are real limitations. The framework still applies where it applies; we should be explicit about where it doesn't.

---

## §4. Empirical Demonstration (8–10 pages)

**Core question:** does the framework, applied to a real open-model feature set analogous to Lindsey et al.'s emotion features, produce substantive methodological findings?

### §4.1 Methods (1.5 pages)

**Model choice.** `[DECIDE]` Primary: Gemma-2-2B or Gemma-2-9B with Gemma Scope pretrained SAEs (Lieberum et al. 2024). Rationale: (a) open model with public SAE features, (b) SAE features have descriptions enabling emotion-feature identification, (c) 2B / 9B scale keeps compute manageable. Alternative: Llama-3.2-3B-Instruct with community-trained SAEs, but SAE coverage is less complete.

**Target features.** Three emotion concepts: **anger**, **sadness**, **fear**. Classical basic emotions with clear behavioral correlates, likely to appear as discrete features in SAE decompositions. Complementary choice: one *related-but-distinct* distractor per primary (frustration for anger, disappointment for sadness, anxiety for fear) for discriminant validity testing.

**Feature extraction.** For each emotion, extract the putative feature via three methods:
1. Gemma Scope SAE (the pretrained features from Lieberum et al.)
2. Supervised linear probe trained on labeled emotion examples
3. Activation-steering vector (mean activation on emotion-positive examples minus mean on emotion-neutral, classical steering approach)

Run each method across **five random seeds** where seeding is meaningful (the probe; the steering vector on different example subsamples; the SAE if multiple trained checkpoints are available).

**Stimulus set.** Curate roughly 500 example texts balanced across: 6 emotions (3 primary × 3 distractor, with distractor as control), 3 contexts (first-person narrative, third-person narrative, dialogue), 2 languages (English, `[DECIDE]` Spanish or Japanese or drop and make monolingual). Total: 500 texts, roughly 40–45 per cell, balanced by cell rather than balanced across cells.

`[SCOPE]` The stimulus set is the biggest scope risk — curating 500 balanced emotion texts across contexts and languages is substantial work. If this becomes infeasible, drop the language dimension and use 300 English-only texts across contexts. The DCF analysis loses the cross-linguistic test but retains first/third/dialogue.

### §4.2 Reliability results (1.5 pages)

Expected structure of results:

- **Test-retest reliability.** Report ICC(2,1) for each extraction method × emotion. Hypothesis: SAE features will have highest test-retest (deterministic once trained), probes will have moderate (depends on training examples), steering vectors will vary most.
- **Internal consistency.** For each emotion, alpha across the three extraction methods treating them as "items" measuring the same construct. If alpha < 0.7, the three methods are not measuring the same thing.
- **Inter-method reliability.** Pairwise correlations: SAE-probe, SAE-steering, probe-steering. Report as a generalizability coefficient summing over methods.

Anticipate that at least one emotion will show poor reliability on one or more dimensions. This is expected and useful — it demonstrates that the framework identifies real problems.

### §4.3 SEM and MDC (1 page)

For each emotion × extraction method: compute SEM from the cross-seed reliability. Report MDC for intervention studies. Compare to typical intervention effect sizes reported in the interpretability literature (ablation effects, steering magnitudes) to determine how many such interventions fall below MDC.

This is the section that will be cited most heavily in practice. The headline result should be something like: "Of 47 interpretability papers surveyed reporting intervention effects, 12 report effects below MDC; another 22 do not report sufficient statistics to compute MDC." `[JUDGMENT]` Decide whether to include a mini-survey of existing papers or keep the comparison at the anecdotal level.

### §4.4 Convergent and discriminant validity (2 pages)

Monotrait-multimethod matrix for anger / sadness / fear × SAE / probe / steering.

- Diagonal (same trait, same method): irrelevant — correlation with self is 1.
- Monotrait-heteromethod (same trait, different method): hopefully high. Convergent validity.
- Heterotrait-monomethod (different trait, same method): hopefully moderate — different emotions are correlated but not identical.
- Heterotrait-heteromethod (different trait, different method): hopefully lowest. Discriminant validity.

The expected test for discriminant validity: anger-SAE vs. frustration-SAE correlation should be *lower* than anger-SAE vs. anger-probe correlation. If not, the "anger feature" is not specifically about anger.

This section will have the strongest novel-findings potential. Predicted result: anger and frustration will be hard to distinguish, fear and anxiety will be hard to distinguish, sadness and disappointment may be moderately distinguishable. If so: interpretability claims about "the anger feature" should be more cautious than current practice.

### §4.5 Differential Context Functioning (1.5 pages)

For each emotion feature (primary only, not distractors, to limit scope): ANOVA of feature activation across concept-presence × context. The interaction term estimates DCF.

Expected findings: some features will show large DCF (activate more strongly in first-person even when content is matched), others will not. Report the distribution. Highlight specific features with large DCF as cases where interpretability claims about "the anger feature" need to be contextually qualified.

### §4.6 Factor structure of emotion features (1.5 pages)

Once we have ~20–30 emotion-related features across the three primary × three distractor × three extraction methods (some overlap, some unique), run exploratory factor analysis on their activation patterns across the 500-text stimulus set.

Compare fit of:
- Six-factor model (Ekman)
- Two-factor model (valence × arousal, Russell)
- Three-factor model (the three primaries)
- Eight-factor model (Plutchik)

Report fit indices (CFI, RMSEA, TLI). The winner tells us something about the model's representational geometry for emotion. This is the section most likely to produce a genuinely substantive finding beyond the methodological framework.

`[SCOPE]` Factor analysis with this many variables and this sample size will be underpowered. Be realistic: report findings as exploratory and effect-size-oriented, not confirmatory.

### §4.7 What the framework reveals: case studies (1–1.5 pages)

Three or four specific worked examples where applying the framework produces a finding that would not be visible through current interpretability practice. For each: the original claim from the interpretability paper, the psychometric analysis, the refined claim.

`[JUDGMENT]` Use our own empirical results for case studies rather than critiquing others' papers. Keeps tone collaborative, demonstrates the framework's value via our own data.

---

## §5. Implications for Interpretability Practice (3 pages)

### §5.1 Recommended reporting standards (1 page)

A compact recommendations list that can be cited by future papers:

- Report cross-seed test-retest reliability for all feature-based claims (ICC(2,1) with 95% CI).
- Report SEM and MDC for any feature used in causal intervention studies. State whether observed intervention effects exceed MDC.
- For multi-feature concept claims, report internal consistency (Cronbach's alpha or McDonald's omega).
- For causal claims, report discriminant validity against at least one related-but-distinct construct.
- Report Differential Context Functioning across at least two prompt contexts if the feature is claimed to be context-general.
- Pre-register construct definitions and validity predictions before extracting features from data used for confirmatory claims.

`[JUDGMENT]` The pre-registration recommendation is the most controversial for interpretability audiences who aren't accustomed to pre-registration as a norm. Either (a) lead with it as a flagship recommendation or (b) soft-pedal it as "consider pre-registration where confirmatory claims are made" depending on how combative we want to be.

### §5.2 Design implications for feature extraction methods (1 page)

Some classes of feature extraction method will fare better than others under the framework. Initial hypotheses:
- SAE features will have strong test-retest but weak discriminant validity for related concepts.
- Probe-based features will have strong discriminant validity within labeled datasets but weak reliability across probe training runs.
- Steering vectors will have weak reliability and moderate validity.

Design suggestions: multi-method feature identification (features identified via two methods are more trustworthy), explicit discriminant validity training (regularize against activating on distractor examples), contextual stratification in training data to minimize DCF.

### §5.3 Design implications for causal intervention claims (1 page)

How to strengthen causal claims under the framework:
- Establish measurement validity before claiming causation.
- Report intervention effect sizes with MDC context.
- Use discriminant interventions: ablate the "anger" feature, ablate the "frustration" feature, show behavioral effects are distinct. This is a direct application of discriminant validity to interventions.
- Consider dose-response designs: if feature F causes behavior B, parametric variation in F activation should produce monotonic variation in B.

---

## §6. Limitations and Extensions (2 pages)

### §6.1 Limitations

- Framework applies to measurement claims; does not resolve the underlying interpretability question of whether features are "real" computational objects.
- Computation cost of full framework is non-trivial; for large models and many features, careful subsetting will be needed.
- Factor analysis requires adequate sample sizes, which may not always be feasible.
- Pre-registration norm is unfamiliar in interpretability and will face resistance.

### §6.2 Extensions

- **Item Response Theory for features.** Beyond CTT, IRT models (Rasch, 2PL, 3PL) could model feature-activation probability as a function of construct strength. Potentially powerful for graded feature activations.
- **Measurement invariance testing.** Are features measuring the same construct across model sizes / model families? Structural equation modeling can test this.
- **Hierarchical measurement.** Concepts nested within higher-order factors (specific emotions nested within valence-arousal space). Multilevel factor models apply.
- **Network psychometrics.** Treating features as nodes in a partial correlation network, using tools from Borsboom et al.'s network psychometrics. May be the best match for the SAE setting where feature interactions matter.

### §6.3 Computational cost considerations

Running the framework in full adds several thousand forward passes for activation collection, plus substantial CPU for statistical analysis. For scaling to many features, strategies:
- Report full framework for a curated subset of high-interest features.
- Release code as a standalone library so others can apply incrementally.
- Develop cheaper proxies (e.g., approximate reliability from single-seed activation variance).

---

## §7. Conclusion (1 page)

Summary of the framework. The moment-in-interpretability argument: feature-based claims are becoming central to AI safety decisions; the measurement-validity infrastructure that psychometrics developed exists for precisely this kind of claim; adopting it rigorously is a low-cost high-return contribution that strengthens interpretability as a discipline. Close with a note on the appropriate humility: measurement validity is necessary for feature claims but not sufficient for claims about what the model is "really doing." Features that pass all validity checks are reliable indicators of a measured pattern; whether that pattern is the computational object the model uses is a separate, substantive question.

---

## Appendices

### Appendix A: Formal definitions and estimator equations (3–4 pages)
Mathematical definitions for each framework concept and reference implementations.

### Appendix B: Open-model replication full results (5–6 pages)
Complete statistical tables, figures, and supplementary analyses from §4.

### Appendix C: Code and data release
- GitHub repository with Python implementation of all estimators.
- Pre-registered analysis plan (committed before empirical analysis).
- Stimulus set with emotion and context annotations.
- Trained probes, steering vectors, and reliability estimates.

### Appendix D: A note on correspondence with interpretability researchers
`[JUDGMENT]` Decide whether to include a brief appendix documenting (with permission) reactions from 2–3 interpretability researchers we ran the framework past during development. Strengthens the paper's credibility but adds coordination overhead.

---

## Decision points to resolve before drafting begins

1. `[DECIDE]` **Venue**: NeurIPS D&B, ICLR methods track, JMLR, or bridge venue. Affects target length and review timeline.
2. `[DECIDE]` **Model**: Gemma-2-2B, Gemma-2-9B, or Llama-3.2-3B. Affects compute and SAE availability.
3. `[DECIDE]` **Language dimension**: include cross-linguistic DCF test or stay monolingual. Affects stimulus-curation work.
4. `[DECIDE]` **Survey of existing papers** for the MDC headline result: include as mini-meta-analysis or keep anecdotal. Affects scope.
5. `[DECIDE]` **Pre-registration recommendation framing**: flagship recommendation or softer suggestion. Affects tone.
6. `[DECIDE]` **Whether to seek early feedback from Lindsey et al. or other interpretability researchers** before submission. Tradeoff: stronger paper vs. scooping risk vs. coordination overhead.
7. `[JUDGMENT]` **Name of the framework**. "Psychometric Framework for Interpretability" is descriptive but bland. "Measurement Validity for Feature-Level Interpretability" is more precise. Consider a shorter name for abbreviation (MV-FLI? PVI? Just "psychometric interpretability"?).

---

## What's needed to start

1. Resolve decision points 1–3 (venue, model, language dimension).
2. Literature review in two directions: recent interpretability (2024–2026) and psychometric frameworks (Borsboom, Messick, Cronbach, Embretson & Reise).
3. Stimulus set design and pilot IRB-equivalent review (not formal IRB, but peer review of stimulus appropriateness).
4. Compute allocation: estimated $500–1500 RunPod budget for feature extraction and analysis.
5. Timeline commitment: 4–6 months assumes 60%+ of research time on this. Compatible with N134/N135 consolidation path only if this is strictly second-priority and opportunistic.

---

## One meta-note on scope

This outline proposes a single paper with a narrow empirical demonstration, a clear methodological contribution, and specific recommendations. The biggest risk is scope creep into "let's also do IRT" or "let's also test on a bigger model" — resist. If the paper works, the IRT extension and the scale-up become follow-on papers. Measurement validity for feature-level interpretability is the flagship contribution; everything else is second-album material.

---

*Prepared April 19 2026. Working outline v0. Revise freely.*
