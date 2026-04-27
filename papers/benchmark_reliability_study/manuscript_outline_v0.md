# Benchmark Reliability Study — Manuscript Outline v0

**Status:** outline + early prose for sections that do not depend on Phase 5 analysis output.

**Working title:** *Benchmark Accuracy as Measurement: Reliability and Tolerance Schedules for LLM Evaluation* (per `prereg_v1_1_LOCKED.md` frontmatter; final wording open).

**Target venue:** open. Candidates ordered by fit:

1. **TMLR** — continues the methodological track N134 / Thesis B is on; framework-as-coordinated-methodology fits TMLR's "no novelty bar" position. Same reviewer pool likely to read both papers, which makes the cross-paper "second worked demonstration" framing work.
2. **NeurIPS 2026 Datasets and Benchmarks Track** — venue match for measurement-of-evaluation papers; Bean et al. 2025 ("the NeurIPS 2025 construct-validity systematic review") established this track as a home for measurement-discipline-applied-to-LLM-benchmarks work.
3. **ICLR 2027 Position Track** (or equivalent) — if the prescriptive register is foregrounded over the empirical findings, the position track is a fit. Probably wrong venue for this paper specifically; the paper's load-bearing content is a pre-registered empirical study, not a position argument.

Decision deferrable until Phase 5 results are in hand and the actual contribution shape stabilizes.

**Page-count target:** 16–20pp main + appendix (TMLR style) or 9pp main + appendix (NeurIPS D&B). Body sections sized to TMLR; if NeurIPS D&B becomes the venue, §3, §4, §5 collapse into a single "Methods" section per their style.

**Key dependencies on Phase 5 (cannot be drafted now):**

- Variance-components results (§7.1) — depends on `06_variance_components.py` output across 36+ cells.
- Generalizability coefficients (§7.2).
- Tolerance schedule (§7.3) — depends on `07_tolerance_schedule.py` regime-aware output.
- Ranking-stability findings (§7.4) — depends on `08_ranking_stability.py`.
- MMLU subject-decomposition case (§7.5).
- GSM8K case (§7.6).
- All discussion sections that engage results (§8).

**Sections draftable now:**

- §1 (Introduction): the reporting-gap motivation, the construct-hierarchy reframe, parallel-development positioning, contribution claims, and a "what this paper does not do" boundary statement. Significant prose can be lifted from `prereg_v1_1_LOCKED.md` §1.1, §1.3 and adapted.
- §2 (Prompt-sensitivity baseline): brief; one to two paragraphs establishing the empirical literature this paper systematizes the response to.
- §3 (Framework setup): partly portable from N134 §2 with substrate substitution. The four-component framework is identical; the construct-articulation (§3.2–3.3) and confound-list (§3.6) need substrate-specific instantiation.
- §4 (Parallel-development register): full draft possible; structure parallels N134 EDIT-22 but adds Brittlebench as the sixth voice and includes an explicit "what is cited and what is not" methodological note.
- §5 (Pre-registered design): substantially portable from `prereg_v1_1_LOCKED.md` §1, §2, §4–§5; needs editorial pass to convert from protocol register to manuscript register.
- §6 (Pipeline implementation summary): portable from `SPEC_CPU_v0_2.md` and `SPEC_GPU_v0_1.md`; brief, anchors reproducibility appendix.

---

## Abstract sketch (~200 words; placeholder, refine after results)

Single-occasion benchmark accuracy reports treat the score as a fixed property of a (model, benchmark) pair, but admissible variation in prompt template, few-shot exemplar selection, and scoring rule contributes substantial variance to the observed score. This study applies a four-component measurement-discipline framework — construct articulation, reliability estimation, precision/tolerance, and pre-registered confound decomposition — to small-to-mid-scale open LLMs evaluated on six standard benchmarks under a declared measurement universe. We pre-register a mixed-effects variance-components decomposition, a decimal-place precision tolerance schedule, and a ranking-stability test. **[Empirical findings TBD]** A regime split between cells where variance components are inferentially meaningful and cells where parse-failure dominates the observed score is identified and motivated; we report tolerance schedules under each regime separately. The study is the second public demonstration of a measurement-discipline framework, on a substrate (LLM benchmark evaluation) deliberately chosen to be distinct from the framework's first demonstration on LoRA-merging diagnostics, and is intended as a portability test of the framework as much as an empirical study of benchmark reliability.

---

## Section structure

### §1. Introduction (draftable now)

#### §1.1 The reporting gap

**Status:** prose-ready. Mirrors N134 §1.1 in structure but with the LLM-benchmark substrate. Key beats:

- Single-prompt accuracy reports are routine; tolerance bounds calibrated to the actual measurement design are rare.
- Three specific gaps: (a) reliability coefficients in the classical-test-theory sense (ICC, SEM, generalizability coefficients) are nearly absent from LLM-benchmark reports; (b) point estimates quoted to two decimal places without uncertainty bounds calibrated to the actual dependence structure; (c) confound decomposition (item-level vs. condition-level vs. model-level variance attribution) is rare and usually post-hoc.
- The reader who wants to know how much credence to place in a reported MMLU number has no reliable way to calibrate.

**Source material:**
- `prereg_v1_1_LOCKED.md` §1.1, §1.2.
- N134 §1.1 paragraph structure (prose register portable; substrate substituted).

**Draft prose:** see *Section 1 prose sketch* below.

#### §1.2 Parallel-development register

**Status:** prose-ready. Mirrors N134 EDIT-22 paragraph but with the LLM-evaluation substrate substituted in. Same five voices (Messing, NIST 800-2, NIST 800-3, Bean et al. 2025, Camuffo et al.) plus Brittlebench (Romanou et al. 2026) as the sixth voice that earns its keep here specifically (where it does not in N134 — see `RESEARCH_INVENTORY.md` Section 4 HIGH commentary).

**Distinctive contribution claim:** the decimal-place precision tolerance schedule remains the load-bearing differentiator, just as in N134.

**Draft prose:** see *Section 1 prose sketch* below.

#### §1.3 Contribution claims

**Status:** prose-ready. Adapted from `prereg_v1_1_LOCKED.md` §1.1 ("This study... produces...").

Three contribution claims:

1. The four-component measurement-discipline framework, applied to LLM benchmark evaluation as a second worked demonstration following the N134 / Thesis B paper's LoRA-merging instantiation. The framework's substrate-portability is itself a contribution; it is one thing to propose a framework on a single substrate, another to demonstrate it survives application to a structurally different second substrate.

2. A pre-registered variance-components decomposition, generalizability-coefficient table, and decimal-place precision tolerance schedule for benchmark accuracy on the small-to-mid-scale open-model regime under a declared measurement universe (six benchmarks, three models, six prompt templates per benchmark, fixed scoring rules).

3. A regime-split methodological resolution distinguishing cells where variance-components decomposition is inferentially meaningful (g_theory regime) from cells where parse-failure dominance makes the variance-components SEM uninterpretable (parse_failure_dominated regime). The regime split is itself a measurement-discipline finding the framework's reproducibility-and-pre-registration discipline surfaced; it is offered in the same epistemic register as N134's rank-on-residuals observation — discovery-like in the narrower reporting sense (cf. N134 §6.4 EDIT-16).

**Draft prose:** see *Section 1 prose sketch* below.

---

### §2. Prompt-sensitivity literature: the empirical baseline (draftable now)

**Status:** prose-ready. Brief; one to two paragraphs.

**Content:** Sclar et al. 2023; Polo et al. 2024; Lunardi et al. 2025 — the empirical literature this paper systematizes the response to. Brief by design; the paper positions its contribution as methodological, not as the first to observe prompt sensitivity. This section establishes that the substrate-level phenomenon (benchmark scores moving under admissible variation) is independently established in the literature, freeing the paper to spend its space on the measurement-theoretic response rather than re-establishing the empirical premise.

**Reorder rationale.** This was previously §2.1 within a combined "Related work" section. The split (prompt-sensitivity baseline now, parallel-development register after framework setup) reflects that §3's framework setup is what licenses the comparison register §4 develops — we can characterize other measurement-discipline work *as* measurement-discipline work because we have a measurement-discipline framework to compare with. Putting all of "related work" before §3 would force §4's positioning to land before the framework that anchors it.

**Draft prose:** see *Section 2 prose sketch* below (the prompt-sensitivity-baseline subsection there is the §2 content; the parallel-development-register subsection there is the §4 content, post-reorder).

---

### §3. Framework setup (mostly portable from N134 §2)

**Status:** structure portable; substrate-specific content needs writing.

**Sub-structure:**

#### §3.1 The four components

Identical to N134 §2. One-sentence summary of each:

- Construct articulation (what the score indicates)
- Reliability estimation (how stable the indication is)
- Precision and tolerance (what numerical precision is licensed)
- Confound decomposition (what else could explain the signal)

**Action:** can be lifted near-verbatim from N134 §2 opening with minimal editing.

#### §3.2 Construct hierarchy for benchmark accuracy

**Substrate-specific.** Four levels per `prereg_v1_1_LOCKED.md` §2.2: latent capability → operational benchmark score → measurement occasion → generalized score.

**Action:** fully portable from prereg §2.2 prose; minor editorial smoothing for manuscript register.

#### §3.3 Construct articulation in this study

What this study takes the constructs to be — what claims about model capability the benchmark accuracy is offered as evidence for, and what claims it is not. Names what is being articulated and what is being deferred (chain-of-thought scoring, contamination, agentic evaluation are out of scope).

**Action:** new prose. ~0.5pp.

#### §3.4 Reliability machinery

Variance-components decomposition via mixed-effects models on item-level binary correctness data. Generalizability coefficients under single-occasion and averaged-design reporting conventions. Bootstrap CIs on the SEM and the generalizability coefficients.

**Action:** new prose; technical content largely standard.

#### §3.5 Tolerance schedule construction

The load-bearing methodological move. Decimal-place precision logic: a score reported to k decimal places carries an implicit claim that the (k+1)th digit is below the SEM. The schedule estimates SEM under the declared measurement universe and licenses decimal-place precision accordingly.

**Action:** new prose; this is the distinctive-contribution paragraph the manuscript's positioning rests on.

#### §3.6 Confound decomposition

Pre-registered confound list. Same epistemic discipline as N134 §2.4: distinguishing what the diagnostic predicts from what is predictable by simpler alternatives, with confounds enumerated *before* data collection.

**Action:** new prose; can mirror N134 structure.

---

### §4. Parallel-development register (draftable now)

**Status:** prose-ready. Two-paragraph positioning paralleling N134 EDIT-22, plus a short principled-selection note.

**Sub-structure:**

#### §4.1 The co-developed register

Six voices that have arrived at structurally similar measurement-discipline conclusions about LLM evaluation: Messing 2026 (TEE / budget allocation); NIST AI 800-2 (Jan 2026, voluntary practices); NIST AI 800-3 (Feb 2026, GLMM endorsement); Bean et al. NeurIPS 2025 (construct-validity systematic review on 445 benchmarks with 29 expert reviewers); Camuffo et al. 2026 (variance-aware LLM annotation, generalizability-theory grounding); Romanou et al. 2026 (Brittlebench, prompt-brittleness decomposition with 63% pairwise-ranking-reversal under any-single-perturbation).

**The shared diagnosis** is that LLM evaluation pipelines carry hidden measurement variance that ordinary reporting does not surface; the convergence across substrates (benchmarking, annotation, regulatory practice, brittleness diagnostics) is itself evidence that the diagnosis is real. The prescriptive registers differ across the corpus; the present paper's distinctive contribution within the shared register is the **decimal-place precision tolerance schedule** developed in §3.5 — licensing what numerical precision a reported score actually supports under a declared measurement universe. Generalizability theory, variance decomposition, and construct-validity reasoning are now shared methodological apparatus across the field; the tolerance-schedule prescription is what this paper distinctively contributes.

#### §4.2 Distinguishing inferential targets

Several of the parallel works are close enough in vocabulary to risk substitution. Brittlebench targets model-level brittleness — a property of the model under adversarial-flavored semantics-preserving perturbations — and produces a model-comparison diagnostic. The present paper targets measurement tolerance — a property of the (model, benchmark) cell under a community-provenance measurement universe — and produces a per-cell precision-licensing schedule. The two are complementary, not redundant. Sui et al. (2025) and Camuffo et al. (2026) apply G-theory and Rasch modeling to LLM-as-rater on human-produced text; the present paper applies G-theory to benchmark items as evaluation instruments for LLMs. Bean et al. (2025) provide a construct-validity failure taxonomy with eight key recommendations; the present paper is one possible operational instantiation of items in that taxonomy on a specific substrate. The taxonomic-vs.-operational distinction is what allows both contributions to coexist.

#### §4.3 A note on what is cited and what is not

The parallel-development register is curated, not exhaustive. Parallel work earns engagement here when its empirical or framing register is load-bearing for *this* paper's argument — e.g., Brittlebench's 63% ranking-reversal headline serves as the comparison anchor for §6.4's H3 discussion (cf. §6.4 below); NIST 800-3's GLMM endorsement is what the §3.5 decision rule explicitly responds to (cf. §3.4 below and the v1.1.2 amendment described in §5.4). Parallel works methodologically adjacent without inferential-target match (e.g., Benchmark$^2$, Judge Reliability Harness, the broader IRT-for-LLM-benchmarks literature) are tracked by the program but not engaged here; that does not entail a quality judgment, only a scope discipline. A reviewer who would expect a specific parallel-development item engaged in the related-work register and finds it absent is encouraged to ask whether that work's framing is load-bearing for the present paper's argument as opposed to neighboring it; the present register is constructed against the former criterion.

**Draft prose:** see *Section 2 prose sketch* below for the §4.1–§4.2 content (the prose there is structured to land in the new §4, post-reorder; §4.3 is a short note rather than load-bearing prose).

---

### §5. Pre-registered design (draftable now from prereg)

**Status:** substantially portable from `prereg_v1_1_LOCKED.md` §3–§5.

**Sub-structure:**

#### §5.1 Models, benchmarks, measurement universe

- Three models: pythia-410m, pythia-1.4b, Qwen2.5-1.5B-Instruct (revisions pinned).
- Six benchmarks: ARC-Challenge, HellaSwag, TruthfulQA-MC, MMLU (subject panel), Winogrande, GSM8K (secondary tier).
- Measurement universe: six prompt templates per benchmark × few-shot exemplar selection × scoring rule × seed.
- Specific facets sourced from canonical community repositories at pinned commits.

**Action:** portable from prereg §3, §5.

#### §5.2 The mixed-effects cascade

Pre-registered random-effects structure: prompt, seed, scoring_rule, item, model_prompt_interaction, model_scoring_rule_interaction. Cascade levels per `SPEC_CPU_v0_2.md` §10.1.

**Action:** portable from prereg §7.1, SPEC_CPU §10.1.

#### §5.3 Decision rules

H1 (tolerance schedule), H2 (variance components), H3 (ranking stability — 20% threshold). Pre-registered decision criteria with rationale.

**Action:** portable from prereg §3.

#### §5.4 The regime split (D-09 → v1.1.2)

The amendment. Rationale: NIST AI 800-3 endorsed GLMM, but parse-failure-dominated cells sit outside the LPM-GLMM agreement range; cells with median parseability < 0.30 route to sample-SD-based tolerance instead of variance-components SEM.

**Action:** new prose, but contents are recorded in `LOCK_NOTES.md` v1.1.2 amendment, `IMPLEMENTATION_DEVIATIONS.md` D-09 v1.1.2 resolution, and `analysis_config.yaml` `tolerance.parse_failure_threshold`. Faithfully report the chronology — early Phase 4 GPU output revealed the parseability problem; NIST 800-3 endorsement constrained the resolution; regime split was the result.

---

### §6. Pipeline implementation (draftable now, brief)

**Status:** brief; anchors reproducibility appendix.

**Content:** one-paragraph summary of the 13-script pipeline (`00_validate_config` through `99_make_report`, plus `98_reproducibility_trace`). Refers to SPEC_CPU + SPEC_GPU for full detail. Notes the test suite (181 passing as of v1.1.2 lock).

**Action:** new prose, ~0.5pp.

---

### §7. Empirical results (waits for Phase 5)

**Status:** waits. Sub-section structural specifications below are deliberate skeletons — table columns, expected figure types, and inferential moves are pre-specified so that drafting after Phase 5 is fast and so that the writing does not silently shift inferential targets when results land.

#### §7.1 Variance components

Per-cell and per-benchmark variance decomposition. Identifies which facets dominate variance (expected: prompt + scoring rule for parse-failure-dominated cells; item-level for stable cells).

**Expected tables/figures:**
- *Table 1.* Per-cell variance components (prompt, seed, scoring_rule, item, model_prompt_interaction, model_scoring_rule_interaction, residual) reported as percent of total variance, with bootstrap CIs.
- *Table 2.* Per-benchmark aggregated variance components (averaged across models within benchmark).
- *Figure 1.* Variance-component composition by cell, stacked bar, color-coded by facet, ordered by parseability rate (cells in g_theory regime grouped left, parse_failure_dominated grouped right).

**Depends on:** `06_variance_components.py` output.

#### §7.2 Generalizability coefficients

Single-occasion vs. averaged-design generalizability. Likely beat: averaged designs across multiple prompts substantially outperform single-occasion reporting at the population level, but this story breaks down at the per-cell level for parse-failure-dominated cells.

**Expected tables/figures:**
- *Table 3.* Generalizability coefficients per (model, benchmark) cell under three averaging schemes: single-occasion (k=1), three-prompt average (k=3), all-prompts average (k=6). Bootstrap CIs.
- *Figure 2.* Generalizability gain curve (g vs. k) per benchmark, with regime-split line styles.

**Depends on:** `06_variance_components.py` output.

#### §7.3 Tolerance schedule

The load-bearing empirical product of the paper. Per-cell decimal-place precision recommendation under each regime (g_theory, parse_failure_dominated). Headline finding likely: typical cells license 1 decimal place; some cells license 2; parse-failure-dominated cells license 0–1 with a register caveat that the SEM under that regime estimates parse-failure variance, not measurement-design variance.

**Expected tables/figures:**
- *Table 4.* Per-cell tolerance schedule with columns: cell ID, regime, parseability rate, observed accuracy point estimate, SEM, recommended decimal-place precision, source of SEM (variance-components or sample-SD).
- *Figure 3.* Calibration plot: reported decimal places vs. licensed decimal places under the schedule, separated by regime.

**Depends on:** `07_tolerance_schedule.py` regime-aware output.

#### §7.4 Ranking stability (H3)

Fraction of cross-condition pairwise rankings reversed under admissible measurement conditions. Pre-registered threshold 20%. Comparison anchor: Brittlebench's 63% reversal under any-single-perturbation on frontier models — different perturbation universe and model regime, but the order-of-magnitude framing supports H3's hypothesis.

**Expected tables/figures:**
- *Table 5.* Pairwise ranking-reversal fractions per benchmark, per condition-resampling scheme, with H3 decision (PASS / FAIL against 20% threshold).
- *Figure 4.* Ranking-reversal rate distribution across benchmarks; horizontal line at 20% threshold; secondary horizontal line at 63% (Brittlebench reference) for comparison context.

**Depends on:** `08_ranking_stability.py`.

#### §7.5 MMLU subject-decomposition case

Subject-by-subject breakdown of MMLU panel: world_religions, elementary_mathematics, high_school_psychology, professional_accounting, global_facts. Useful as a within-benchmark variance check; expected: some subjects much more brittle than others.

**Expected tables/figures:**
- *Table 6.* Per-subject variance components within MMLU panel, with regime classification.
- *Figure 5.* Subject-level forest plot: point estimate ± SEM per (model, subject), showing where within-MMLU subject variance dominates between-subject signal.

**Depends on:** `09_mmlu_decomposition.py` output.

#### §7.6 GSM8K case

Free-form generation + extraction; the paper's only G&P-only benchmark. Demonstrates the regime split most starkly. Strict vs. permissive extraction comparison.

> **Scope note (2026-04-26, post-cut):** The GSM8K case is now a **single-model case study** (pythia_1_4b only, 24 conditions: 4 prompts × 3 seeds × 2 extraction variants). Pythia_410m's and qwen2_5_1_5b's 48 GSM8K conditions were dropped via the pre-committed cost-projection tripwire (see `IMPLEMENTATION_DEVIATIONS.md` D-18 and the budget-driven scope amendment in `LOCK_NOTES.md`). The case-study claim — *that the parse-failure-dominated regime is most starkly visible on open-generation benchmarks* — is preserved by single-model demonstration (pre-reg §11.4 frames Tier 2 as a case study with self-contained scope). What is given up: the base-vs-instruct comparison originally planned for Table 7. The §7.6 prose-draft must reflect the 1-model scope: drop the "instructive comparison is base vs. instruct" framing, keep the regime-split demonstration on pythia_1_4b alone. The §8.3 cross-paper discussion must absorb this same reduction.

**Expected tables/figures (revised under post-cut scope):**
- *Table 7.* GSM8K outcomes for pythia_1_4b across (prompt template, extraction-rule, seed) cells, with parseability rate, accuracy, and regime classification. The within-model parseability variation (across prompts and seeds) carries the regime-split point even without cross-model comparison.
- *Figure 6.* Parseability rate distribution per (prompt × seed) on pythia_1_4b — argues empirically for the 0.30 threshold or against it on the single-model evidence base.

**Depends on:** `10_gsm8k_case.py` output (run on pythia_1_4b only).

---

### §8. Discussion (waits)

**Sub-structure:**

#### §8.1 Methodological implications

What this study licenses readers to do differently when reading benchmark numbers. Concrete recommendation: a tolerance-schedule-aware reading discipline that asks "under what measurement universe was this number generated, and what decimal-place precision does that universe support?"

#### §8.2 Limitations

- Small-to-mid-scale open-model regime — does not extrapolate to frontier models.
- Six benchmarks — does not exhaust the LLM benchmark landscape.
- Constrained measurement universe (no contamination assessment, no agentic evaluation).
- Three models, two of which are same-family (Pythia 410m, Pythia 1.4b) — interaction with model family confound is acknowledged but not exhaustively studied.
- Mixed-effects cascade is intentionally high-capacity relative to per-cell N (cf. N134 §5.2's FAMILY_B-equivalent caveat). The cascade is appropriate for testing whether the framework's variance-decomposition machinery surfaces measurement structure on this substrate; it is not appropriate as a deployable accuracy predictor, and no such predictor is offered.

#### §8.3 Relationship to N134

The cross-paper move: two independent applications of the same framework to structurally different substrates. The empirical findings are paper-specific; the framework's portability is the meta-claim. This is the manuscript's strongest cross-paper position.

#### §8.4 Future work

- Frontier-scale extension.
- Chain-of-thought and reasoning-trace scoring (deferred from this study).
- IRT-style item-level decomposition (per `RESEARCH_INVENTORY.md` Section 8 note 3).
- Adversarial-perturbation universe (Brittlebench-style) as orthogonal complement.

#### §8.5 Anticipated objections

Four reviewer objections are predictable; this subsection defuses each in two-to-three sentences. Mirrors the register of N134 §7.1 (anticipated-objections, distinct from limitations).

- **"Two demonstrations is not a substrate-portability test."** The defense is recorded in §1.3 contribution claim 1: the precursor's substrate (LoRA-merging diagnostic correlations on $n=45$ cross-task pairs, partial Spearman with ICC reliability and family-pair confound decomposition) and the present paper's substrate (item-level binary correctness across thousands of items per cell, mixed-effects variance-components GLMM with bootstrap CIs) differ on inferential target, scoring object, sample-size regime, and error structure. Two demonstrations on substrates this structurally different is non-trivial evidence of portability; we do not claim $k=\infty$.

- **"The small-to-mid-scale open-model regime does not extrapolate to frontier."** The pre-registered scope is exactly the regime tested. We do not claim the tolerance schedule transfers without re-estimation. The framework is what's offered for re-application; the specific schedule is regime-bounded.

- **"The regime split (D-09 → v1.1.2) was added after data collection began and looks ad-hoc."** The regime split was added before any analysis was run, in response to observed parseability rates from early Phase 4 GPU output and the formal NIST AI 800-3 GLMM endorsement (February 2026). The amendment is recorded in `LOCK_NOTES.md` v1.1.2 with full audit; the parseability threshold (0.30) is a pre-specified parameter, not a tuned one. The split is explicitly justified as a response to a methodological tension (NIST endorsing GLMM on cells where LPM and GLMM substantially diverge), not as a fit-the-data move.

- **"No contamination assessment."** Correct. Benchmark contamination is a separate (and well-studied, e.g., Hasan et al. 2025) measurement question. This paper's measurement universe is admissible-prompt-and-scoring-rule variation conditional on a fixed item set; whether the item set itself is contaminated is orthogonal to the variance decomposition this paper estimates. Contamination-aware extension is named in §8.4 future work.

---

### §9. Conclusion (waits)

Brief; reiterates the contribution-claim hierarchy. Connects to the broader "measurement discipline" thesis: not as the first to apply, but as one of several converging voices applying it to LLM evaluation, with the tolerance-schedule prescription as the distinctive prescriptive contribution.

---

### Appendices

- **App. A:** Full pre-registration document (`prereg_v1_1_LOCKED.md`) reproduced.
- **App. B:** Pipeline implementation detail and reproducibility trace.
- **App. C:** LPM-vs-GLMM appendix (per D-09 v1.1.2 resolution, the methodological side-by-side that the cross-paper N134 → benchmark-reliability link requires).
- **App. D:** Per-cell tolerance-schedule tables (full).
- **App. E:** Ranking-stability detail per benchmark.
- **App. F:** MMLU subject-decomposition detail.

---

## Section 1 prose sketch (early-draft, ~600 words; refine)

### §1.1 The reporting gap

A typical recent LLM evaluation paper reports benchmark accuracy as a single number, often quoted to two decimal places: "model M achieves 0.74 on MMLU." A reader who wants to know how much credence to place in this number — how stable it would be under a re-run with a different prompt template, a different few-shot exemplar draw, a different scoring rule — has no systematic way to find out. The number is presented as if it were a stable property of the (model, benchmark) pair; the measurement design that produced it is rarely declared, and the variance attributable to that design is rarely estimated.

The literature on prompt sensitivity makes the underlying problem visible. Sclar et al. (2023) demonstrated that small format changes shift accuracy by tens of percentage points; Polo et al. (2024) reframed multi-prompt evaluation as estimating a performance distribution rather than selecting among prompts; the Sclar / Polo register is now the established baseline for the empirical phenomenon. What the literature does not yet provide is a systematic measurement-theoretic response: a framework that ties reliability, precision, and confound structure to the substantive interpretation of the reported number.

This study estimates the measurement properties that single-prompt accuracy reports actually support. It does not claim that LLM benchmarks are invalid, that MMLU does not measure knowledge, or that leaderboards are meaningless. It claims that benchmark accuracy is better understood as a score sampled from a measurement design — with prompt format, few-shot exemplar selection, scoring rule, and item composition each contributing variance — than as a fixed property of a (model, benchmark) pair, and that field-level claims about "the model's MMLU accuracy" implicitly reference a generalized score the field's reporting practice does not estimate.

Three specific gaps follow from this pattern. Reliability coefficients in the classical-test-theory sense — intraclass correlation coefficients (ICC), standard errors of measurement (SEM), generalizability coefficients — are nearly absent from LLM benchmark reports. Point estimates are quoted to two or three decimal places without uncertainty bounds calibrated to the actual dependence structure of the data. Confound decomposition (attributing variance to prompt vs. scoring rule vs. item-level difficulty vs. model) is rare and usually post-hoc. Each of these gaps on its own is repairable by a careful reviewer; collectively, they produce a literature in which a reader who wants to know how much credence to place in a reported value has no reliable way to calibrate.

The conceptual move the present paper rests on follows from these gaps directly. A reported benchmark score is not, on inspection, a fixed property of a (model, benchmark) pair; it is a score sampled from a measurement design — a specific prompt template, few-shot exemplar draw, scoring rule, and item composition — applied to that pair. The score that field-level claims about *the model's MMLU accuracy* implicitly reference is the *generalized* score: the expected score under the universe of admissible measurement designs. Single-occasion reporting is, in measurement-theoretic terms, a measurement without a stated measurement universe — an implicit point estimate of a generalized score with no accompanying account of what generalization is claimed. Naming the measurement universe, and estimating its variance, is what licenses the rest of the apparatus this paper develops.

The diagnosis is not a private one. Across roughly six months, multiple voices have arrived at structurally similar conclusions about LLM evaluation pipelines from different starting positions. The next subsection situates the present paper within that co-developed register.

### §1.2 Parallel-development register

The measurement-discipline register the precursor paper situated itself within has continued to develop across multiple voices in the months since. Messing (2026) develops a Total Evaluation Error framework decomposing pipeline uncertainty into design-choice variance and shrinking-with-N variance, demonstrating on MMLU benchmarking that optimized budget allocation halves estimation error at equivalent cost. The U.S. National Institute of Standards and Technology's voluntary-practices document (NIST AI 800-2, January 2026) structures benchmark evaluation into define-target, run-evaluation, and analyze-and-report stages; its companion statistical-models document (NIST AI 800-3, February 2026) formally endorses generalized linear mixed models for variance decomposition on AI benchmarks, demonstrated on twenty-two frontier LLMs. Bean et al. (NeurIPS 2025) survey 445 LLM benchmarks with 29 expert reviewers and develop a construct-validity failure taxonomy with eight key recommendations and an operational checklist. Camuffo et al. (2026) identify five variance sources in LLM annotation for strategy research and demonstrate twelve to eighty-five percentage-point swings from minor design choices, grounding their protocol in generalizability theory. Romanou et al. (2026) [Brittlebench] decompose total performance variance under semantics-preserving prompt perturbations and report that perturbations account for up to half of total variance and 63% of model-pair rankings flip under any single perturbation.

The shared diagnosis across this corpus is that LLM evaluation pipelines carry hidden measurement variance that ordinary reporting does not surface; the convergence across substrates (benchmarking, annotation, regulatory practice, brittleness diagnostics) is itself evidence that the diagnosis is real. The prescriptive registers differ: Messing optimizes evaluation-budget allocation; NIST endorses GLMM methodology; the construct-validity survey provides taxonomic guidance; Camuffo et al. develop variance-aware annotation protocols; Brittlebench frames the same phenomenon as model-level brittleness. The present paper's contribution within this co-developed register is the **decimal-place precision tolerance schedule** — licensing what numerical precision a reported score actually supports under a declared measurement universe. Generalizability theory, variance decomposition, and construct-validity reasoning are now shared methodological apparatus across the field; the tolerance-schedule prescription is what the present paper distinctively contributes.

### §1.3 Contribution claims

This paper makes three claims, in priority order:

1. **A second worked demonstration of the four-component measurement-discipline framework**, applied to LLM benchmark evaluation. The framework's substrate-portability is itself a contribution claim, and one we owe the reader an explicit defense of. The precursor paper's substrate (LoRA-merging diagnostic correlations on $n=45$ cross-task pairs, scored via partial Spearman with cross-seed ICC reliability and family-pair confound decomposition) and the present paper's substrate (benchmark accuracy on item-level binary correctness across thousands of items per cell, scored via mixed-effects variance-components GLMM with bootstrap confidence intervals and pre-registered random-effects structure) differ on inferential target, scoring object, sample-size regime, and error structure. Surviving application to both, with pre-registered findings on each, is non-trivial evidence that the framework is portable across the kinds of structural variation the field's substrates actually exhibit. Two demonstrations is not $k=\infty$ portability, and we do not claim it; the claim is that two demonstrations on appropriately diverse substrates is enough evidence to establish that the framework is not silently overfit to the substrate it was developed on.

2. **Empirical results**: a pre-registered variance-components decomposition, generalizability-coefficient table, and decimal-place precision tolerance schedule for benchmark accuracy on the small-to-mid-scale open-model regime under a declared measurement universe (six benchmarks, three models, six prompt templates per benchmark, fixed scoring rules). Plus pre-registered ranking-stability findings against a 20% reversal threshold (cf. H3).

3. **A regime-split methodological resolution**: distinguishing cells where variance-components decomposition is inferentially meaningful (g_theory regime) from cells where parse-failure dominance makes the variance-components SEM uninterpretable (parse_failure_dominated regime). The split is offered in the same epistemic register as the precursor paper's rank-on-residuals observation — discovery-like in the narrower reporting sense: a measurement constraint that was not visible in the precursor study's substrate and would not be visible in unstructured benchmark-score reports, surfaced here by the framework's reproducibility and pre-registration discipline.

### §1.4 What this paper does not do

This paper does not claim that LLM benchmarks are invalid, that any specific recent leaderboard entry is incorrect, or that single-occasion accuracy reporting is irrational under the field's existing norms. The norms produce point estimates because point estimates have been treated as adequate; the norms are not violated by a paper following them. Our claim is that the norms should change, that the apparatus this paper develops licenses the change, and that we apply the apparatus ourselves to the regime we test. We do not propose a new benchmark, a new evaluation metric, a new model-training protocol, or a new contamination-detection technique. We do not extrapolate beyond the small-to-mid-scale open-model regime our protocol tests; frontier-scale extension, contamination-aware extension, and chain-of-thought scoring are named in the future-work section, not adopted here. We do not retroactively re-adjudicate published benchmark numbers; readers should calibrate their credence to the measurement-discipline practices of the reports they consult, not to point estimates produced under norms the report was following.

---

## §2 and §4 prose sketch (early-draft, ~350 words combined; post-reorder mapping)

### §2. Prompt-sensitivity literature: the empirical baseline

The empirical literature on prompt sensitivity is the closest neighbor to this study's substrate. Sclar et al. (2023) quantified language model sensitivity to spurious features in prompt design; Polo et al. (2024) reframed multi-prompt evaluation as estimating a performance distribution rather than selecting among prompts; Lunardi et al. (2025) extended the analysis across 34 LLMs and 6 benchmarks under paraphrase-sensitivity perturbations. This literature is the empirical baseline this study works from. Where it ends — having established that benchmark scores move under prompt-format variation — this study begins, by systematizing the methodological response: a measurement-theoretic framework, developed in §3, that ties reliability, precision, and confound structure to the substantive interpretation of the reported number. The parallel-development register §4 turns to once the framework is in hand is what licenses the comparison the §4 prose draws.

### §4.1 The co-developed register (post-§3, post-reorder)

A co-developed measurement-discipline register has emerged contemporaneously across multiple voices: Messing (2026) on Total Evaluation Error decomposition; NIST AI 800-2 / 800-3 on voluntary practices and GLMM endorsement; Bean et al. (NeurIPS 2025) on construct-validity systematic review; Camuffo et al. (2026) on variance-aware LLM annotation; Romanou et al. (2026) on prompt-brittleness decomposition; Sui et al. (2025) on G-theory and many-facet Rasch modeling for LLM-as-rater. The convergence across substrates (benchmarking, annotation, regulatory practice, brittleness, rater behavior) is itself evidence that the underlying diagnosis is real.

The prescriptive registers differ across these works. Messing's TEE optimizes evaluation-budget allocation; NIST AI 800-3 endorses GLMM as variance-decomposition methodology; the construct-validity survey provides taxonomic guidance; Camuffo et al. develop a variance-aware annotation protocol; Brittlebench frames the same variance phenomenon as a model-level brittleness diagnostic; Sui et al. apply MFRM to a different substrate (LLM-as-rater of human writing). The present paper's distinctive contribution within this register is the decimal-place precision tolerance schedule (§3.5), licensing what numerical precision a reported score supports under a declared measurement universe. Generalizability theory, variance decomposition, and construct-validity reasoning are now shared methodological apparatus across the field; the tolerance-schedule prescription is what this paper's measurement-discipline framework distinctively contributes.

### §4.2 Distinguishing inferential targets

Several of the parallel works are close enough in vocabulary to risk substitution. Brittlebench targets model-level brittleness — a property of the model under adversarial-flavored semantics-preserving perturbations — and produces a model-comparison diagnostic. This study targets measurement-tolerance — a property of the (model, benchmark) cell under a community-provenance measurement universe — and produces a per-cell precision-licensing schedule. The two are complementary; neither subsumes the other. Sui et al. and Camuffo et al. apply G-theory and Rasch modeling to LLM-as-rater on human-produced text; this study applies G-theory to benchmark items as evaluation instruments for LLMs. Bean et al. provide a construct-validity failure taxonomy; this study is one possible operational instantiation of items in that taxonomy on a specific substrate. The taxonomic-vs.-operational distinction is what allows both contributions to coexist.

### §4.3 What is cited and what is not (short methodological note)

The parallel-development register here is curated, not exhaustive. Parallel work earns engagement when its empirical or framing register is load-bearing for *this* paper's argument: Brittlebench's 63% ranking-reversal headline is the comparison anchor for the §7.4 H3 discussion; NIST AI 800-3's GLMM endorsement is what the regime-split decision rule (§5.4) explicitly responds to; Bean et al.'s eight recommendations and operational checklist are what this paper's confound-decomposition list (§3.6) is one operational instantiation of. Methodologically adjacent works without inferential-target match (e.g., Benchmark$^2$, Judge Reliability Harness, the broader IRT-for-LLM-benchmarks literature) are tracked by the program but not engaged here; that does not entail a quality judgment, only a scope discipline. A reviewer who would expect a specific parallel item engaged here and finds it absent is encouraged to ask whether that work is load-bearing for the present paper's argument as opposed to neighboring it; the present register is constructed against the former criterion.

---

## Citation staging

Cites needed in the manuscript, with their target sections. (Assumes the Reuel→Bean rename in `references.bib` has landed; if not, replace `bean2025measuring` with whatever key is in use.)

| Cite | Target section(s) | Role |
|---|---|---|
| Cronbach 1951 (alpha) | §3.1, §3.4 | Reliability foundations. |
| Shrout & Fleiss 1979 (ICC) | §3.4 | ICC machinery. |
| Cronbach & Meehl 1955 (construct validity) | §3.1 | Construct-validity foundations. |
| Messick 1989 / 1995 (validity) | §3.1, §3.3 | Unified validity framework. |
| Brennan 2001 (G-theory) | §3.4, §7.1 | G-theory machinery. |
| Sclar et al. 2023 (prompt sensitivity) | §2 | Empirical baseline. |
| Polo et al. 2024 (multi-prompt eval) | §2, §3.5 | Multi-prompt distribution framing. |
| Lunardi et al. 2025 | §2 | Extended prompt-sensitivity literature. |
| Messing 2026 (TEE) | §4.1 | Parallel-development; budget-allocation register. |
| NIST AI 800-2 (Jan 2026) | §4.1, §3.6, §8.1 | Parallel-development; voluntary practices register. |
| NIST AI 800-3 (Feb 2026) | §4.1, §5.4, App. C | Parallel-development; GLMM endorsement. **Load-bearing for D-09 / v1.1.2 amendment justification.** |
| Bean et al. NeurIPS 2025 (construct validity systematic review) | §4.1, §3.6, §8.1 | Parallel-development; construct-validity taxonomy. **NB: cite key in `references.bib` is `reuel2025measuring`; this is a misattribution caught at pre-submission verification — first author is Andrew M. Bean, 42 authors total. Should be renamed to `bean2025measuring` before manuscript citation use.** |
| Camuffo et al. 2026 (variance-aware annotation) | §4.1 | Parallel-development; methodological cousin (G-theory grounding identical). |
| Romanou et al. 2026 (Brittlebench) | §4.1, §4.2, §7.4, §8.4 | **H3 ranking-stability comparison anchor: 63% reversal under any-single-perturbation vs. this study's 20% threshold under condition resampling.** |
| Sui et al. 2025 (LLM-as-rater MFRM) | §4.1, §4.2, §8.4 | Parallel-development on different substrate (LLM-as-rater); MFRM as candidate strengthening tool for future-work IRT extension. |
| Hasan et al. 2025 (open-benchmark contamination) | §8.5 | Anticipated-objections defense for "no contamination assessment" — the contamination-detection literature is a separate measurement question. |
| N134 / Thesis A (the precursor) | §1.3, §3.1, §8.3, App. C | Cross-paper anchor; the framework's first worked demonstration. |
| ARC, HellaSwag, TruthfulQA, MMLU, Winogrande, GSM8K (data sources) | §5.1, App. B | Pinned by SHA in `configs/benchmarks.yaml`. |
| HELM (Liang et al. 2022) | §5.1 | P3 prompt source. |
| lm-evaluation-harness (Gao et al.) | §5.1 | P2 prompt source. |
| GPT-3 (Brown et al. 2020) | §5.1, App. B.7 | Winogrande P3 prompt source. |

---

## Cross-paper coordination with N134

**Mutually load-bearing:**

- **§8.3 cross-paper section** must be coordinated with N134 §9 conclusion's "Three directions follow" paragraph. The benchmark-reliability paper is the second worked demonstration N134 §9 implicitly references (without naming, since both papers are anonymized for review).

- **The "discovery-like in the narrower reporting sense" register.** N134 §6.4 (post-Tier 1.5 EDIT-16) frames the rank-on-residuals observation as discovery-like in the narrower reporting sense. This paper's parse-failure-regime finding (developed in §5.4 design and §7.3/§7.6 results) should use the same register — same epistemic discipline, same retreat from claiming priority over numerical-statistics literature. Verbatim phrasing convergence is appropriate; cross-paper consistency is part of the program's coherence.

- **Post-hoc analysis register.** N134 §7.1 (post-Tier 1.5 EDIT-18) substitutes "no evidential weight" with "hypothesis-generating rather than confirmatory evidential status." This paper's post-hoc handling (e.g., §7.3 narrative around any unexpected tolerance-schedule finding) should use the same vocabulary. `prereg_v1_1_LOCKED.md` §11 already commits to this register.

- **FAMILY_B-equivalent capacity caveat.** N134 §5.2 + §7 limitations (post-Tier 1.5 EDIT-17) acknowledge that the family-pair residualization is a high-capacity baseline by design. This paper's mixed-effects cascade has an analogous caveat — the random-effects structure is high-capacity relative to the per-cell N — already named in §8.2 limitations and §8.5 anticipated objections. Same epistemic move; substrate-specific instantiation.

**Independent (no coordination needed):**

- Empirical content of each paper.
- Specific statistical machinery (ICC vs. variance-components GLMM in different roles).
- Venue choice.

---

## Open decisions

Decisions that should be settled before §7 (results) is drafted, but can wait for Phase 5 to inform them:

1. **Venue.** TMLR vs. NeurIPS D&B. Decision when Phase 5 results are in hand.
2. **Whether to foreground the regime split or treat it as a methodological-appendix item.** Phase 5 results will tell us how many cells fall into each regime. If most cells are parse_failure_dominated, the regime split *is* the headline; if most cells are g_theory, the regime split is an important methodological footnote.
3. **Whether to fold §3 (framework setup) into §1 (introduction) or keep separate.** TMLR-style permits separate; NeurIPS D&B style prefers folded.
4. **Title final wording.** Working title is from prereg; final wording should be set after results stabilize.
5. **Adversarial-perturbation universe extension** — defer, name in §8.4 future work, or attempt as supplementary appendix? Bandwidth-dependent.

Decisions that can be settled now:

1. **Bib key correction `reuel2025measuring` → `bean2025measuring`** — flagged at pre-submission verification 2026-04-26; cascades into N134 manuscript and `RESEARCH_INVENTORY.md`. **Status: pending user decision** (Claude reported the misattribution; user has not yet authorized the rename).

2. **Adopt the same `% EDIT:` marker convention** used in the N134 paper for tracking editorial passes. Helpful when this paper enters its own editorial cycle.

3. **Working file:** `papers/benchmark_reliability_study/manuscript/draft_v1.tex`. Create when §1 prose is ready to commit; until then, the early prose lives in this outline document.

---

## Status summary (2026-04-26 drafting checkpoint)

- **Pre-registration:** v1.1.2-LOCKED. Config hash `fbc4a5dd`. Test suite 181/183 passing.
- **GPU run:** mid-flight; ~50 hours remaining; ~$27–31 projected (within $30 cap, tripwire at ~$29).
- **Outline status:** §1, §2, §4 prose-ready (drafted in this document). §3, §5, §6 portable-with-edits from existing program documents. §7, §8, §9 wait for Phase 5; §8.5 anticipated-objections is prose-drafted in this outline.
- **Citation staging:** complete; one rename pending (`reuel2025measuring` → `bean2025measuring`).
- **Cross-paper coordination notes:** §8.3, "discovery-like" register, post-hoc framing, FAMILY_B-equivalent caveat — all flagged for verbatim or near-verbatim convergence with N134.
- **Next dependency:** Phase 5 analysis. After GPU completes, run pipeline scripts 04 → 10 + 98 + 99, then return to this outline to draft §7 + §8 (results + discussion).
