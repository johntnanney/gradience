# Research Inventory

**Status:** living document seeded 2026-04-25. Tracks external work the program has either cited, plans to cite, or is watching for relevance to the current research lines.

**How this is maintained:** the daily research review (`research_review/daily_review_prompt.md`) flags new candidates in dated reports under `research_review/`. Promotion of a candidate from a daily report into this inventory is a human decision — the daily reviewer recommends; this file records what the program has accepted as worth tracking.

**How to read criticality tiers:**

- **MUST-CITE** — load-bearing for at least one current or in-prep manuscript. Cannot be omitted from related work without an explicit reason.
- **HIGH** — parallel-development work or methodologically adjacent. Must be engaged in related-work / discussion sections of in-prep manuscripts.
- **MEDIUM** — adjacent literature consolidating in a register the program engages with. Track-and-cite-if-natural; not action-forcing.
- **LOW** — methodologically related but on a substrate the program has deemphasized, or confirmatory rather than novel. Watch-only.
- **OUTSTANDING** — flagged but not yet retrievable or not yet adequately assessed. Re-evaluate on a future daily review pass.

---

## Section 1 — Foundational psychometric and measurement-theoretic citations

These anchor the N134 paper's framework and the benchmark reliability study's analysis vocabulary. **MUST-CITE** for both manuscripts.

| Citation | Role |
|---|---|
| Cronbach, L. J. (1951). *Coefficient alpha and the internal structure of tests.* Psychometrika, 16(3), 297–334. | Reliability coefficient foundations; cited in N134 §1.2 and used implicitly throughout the benchmark study. |
| Shrout, P. E. & Fleiss, J. L. (1979). *Intraclass correlations: uses in assessing rater reliability.* Psychological Bulletin, 86(2), 420. | ICC(2,1) framework; locked at N134's §4.2 and cited explicitly. |
| Cronbach, L. J. & Meehl, P. E. (1955). *Construct validity in psychological tests.* Psychological Bulletin, 52(4), 281–302. | Construct-validity foundational; cited at N134 §1.2 and §2 opening. |
| Messick, S. (1989). *Validity.* In R. L. Linn (Ed.), Educational measurement (3rd ed.). | Unified validity framework; the construct-articulation register the framework adopts. |
| Messick, S. (1995). *Validity of psychological assessment: Validation of inferences from persons' responses and performances as scientific inquiry into score meaning.* American Psychologist, 50(9), 741. | Modern Messick formulation; specifically cited at N134 §2. |
| Brennan, R. L. (2001). *Generalizability Theory.* Springer. | G-theory machinery the benchmark reliability study uses for variance-components decomposition (Analyses 1–2). |

---

## Section 2 — Active citation list (in-prep manuscripts)

### N134 / Thesis A — *Measurement Discipline for ML Diagnostics*

Currently submitted (or imminent) at TMLR. Citation list locked at v2-anonymized-supplementary tag.

| Citation | Tier | Role in manuscript |
|---|---|---|
| Zhou et al. (2026). *Demystifying LoRA merging.* | MUST-CITE | §1.1 second example (the focal critique-as-illustration). Already in `references.bib`. |
| Akbar et al. (2025). *Cross-term interference in LoRA merging.* | MUST-CITE | §1.1 / §3 / §6 — formalizes the interference-term decomposition the program builds on. |
| Xu (2026). *The Spectral Edge Thesis.* | MUST-CITE | §1.1 / §3 — proposes spectral gap as controlling variable; tracked under "tracked parallel work" status. |
| Burnell et al. (2023). *Revisiting reporting practices in ML evaluation.* | HIGH | §1.1 prior-work on reporting reliability. |
| Sclar et al. (2023). *Quantifying language models' sensitivity to spurious features in prompt design.* | MUST-CITE for benchmark study; HIGH for N134 | Prompt-format sensitivity — the empirical literature that establishes the problem the benchmark study reframes. |
| Polo et al. (2024). *Efficient multi-prompt evaluation of LLMs.* | MUST-CITE for benchmark study | Multi-prompt-evaluation framing; benchmark study positions against this. |

### N135 / Benchmark Reliability Study — *Benchmark Accuracy as Measurement*

Currently at v1.1.1-LOCKED (pre-data-collection). Citation list will firm up after Phase 5 analysis.

| Citation | Tier | Role |
|---|---|---|
| Messing (2026). *Hidden Measurement Error in LLM Pipelines.* arXiv:2604.11581 | HIGH | Parallel-development; flagged 2026-04-25. Add positioning paragraph to §2 or §3.1 distinguishing G-theory + tolerance schedule from TEE + design-study projection. See `research_review/2026-04-25.md`. |
| Ye et al. (2026 v3). *Large Language Model Psychometrics: A Systematic Review.* arXiv:2505.08245 | MEDIUM | Adjacent-literature consolidation; flagged 2026-04-25. Cite to acknowledge existence of "LLM psychometrics" as a recognized field name; differentiate ML-diagnostic-as-instrument vs LLM-as-subject. |
| NIST CAISI (Jan 2026). *NIST AI 800-2: Practices for Automated Benchmark Evaluations of Language Models* (draft, 60-day public-comment period closed 2026-03-31). | **MUST-CITE** | Surfaced 2026-04-26 second-pass review. NIST voluntary practices document structured around three stages (define target, run, analyze). Overlaps directly with §9.3 prescriptive contribution. Distinctive contribution preserved (NIST does not formulate the decimal-place tolerance schedule), but framing must shift from "no measurement-discipline standards exist" to "complements emerging NIST standards by adding numerical-precision licensing." |
| NIST (Feb 2026). *NIST AI 800-3: Expanding the AI Evaluation Toolbox with Statistical Models.* | **MUST-CITE** | Surfaced 2026-04-26 second-pass review. Formal NIST endorsement of generalized linear mixed models (GLMMs) for variance decomposition on AI benchmarks; demonstrated on 22 frontier LLMs across GPQA-Diamond, BIG-Bench Hard, Global-MMLU Lite. **In tension with D-09** (LPM substitution); resolved at v1.1.2 amendment via hybrid regime-split per parseability rate. The manuscript needs a side-by-side LPM-vs-GLMM appendix. |
| Reuel et al. (NeurIPS 2025). *Measuring what Matters: Construct Validity in Large Language Model Benchmarks.* arXiv:2511.04703 | **MUST-CITE** | Surfaced 2026-04-26 second-pass review. Systematic review by 29 expert reviewers covering 445 LLM benchmarks; taxonomy of construct-validity failures; eight key recommendations + operational checklist. Engages Messick directly. The §8 confound decomposition should engage the failure taxonomy point-by-point during manuscript writing. Will be canonical recent reference for both papers. |
| Sclar et al. (2023). *Quantifying language models' sensitivity to spurious features in prompt design.* | MUST-CITE | Benchmark study §2 thesis statement directly engages this literature. |
| Polo et al. (2024). *Efficient multi-prompt evaluation of LLMs.* | MUST-CITE | Same. |

---

## Section 3 — Benchmark and evaluation-infrastructure sources

These are cited as data sources / evaluation-infrastructure references; provenance pinned in `configs/benchmarks.yaml` and `configs/prompts.yaml` at v1_1_1_LOCKED.

| Resource | Citation | Role |
|---|---|---|
| ARC-Challenge | Clark et al. (2018). *Think you have Solved Question Answering? Try ARC, the AI2 Reasoning Challenge.* arXiv:1803.05457 | Primary benchmark; HF dataset `allenai/ai2_arc` at SHA `210d026f`. |
| HellaSwag | Zellers et al. (2019). *HellaSwag: Can a Machine Really Finish Your Sentence?* arXiv:1905.07830 | Primary benchmark; HF `Rowan/hellaswag` at SHA `218ec52e`. |
| TruthfulQA | Lin et al. (2022). *TruthfulQA: Measuring How Models Mimic Human Falsehoods.* arXiv:2109.07958 | Primary benchmark; HF `truthful_qa` at SHA `741b8276`. |
| MMLU | Hendrycks et al. (2021). *Measuring Massive Multitask Language Understanding.* arXiv:2009.03300 | Primary benchmark (subject panel); HF `cais/mmlu` at SHA `c30699e8`. |
| Winogrande | Sakaguchi et al. (2020). *WinoGrande: An Adversarial Winograd Schema Challenge at Scale.* arXiv:1907.10641 | Primary benchmark; HF `winogrande` at SHA `01e74176`. |
| GSM8K | Cobbe et al. (2021). *Training Verifiers to Solve Math Word Problems.* arXiv:2110.14168 | Secondary-tier benchmark; HF `gsm8k` at SHA `740312ad`. |
| HELM | Liang et al. (2022). *Holistic Evaluation of Language Models.* arXiv:2211.09110 | P3 prompt source for 5 of 6 benchmarks; commit `11937097`. |
| lm-evaluation-harness | Gao et al. *EleutherAI lm-evaluation-harness.* | P2 prompt source for 6 of 6 benchmarks; commit `c1c4bea3`. |
| GPT-3 (Winogrande P3) | Brown et al. (2020). *Language Models are Few-Shot Learners.* arXiv:2005.14165 | Winogrande P3 prompt source (HELM has no English Winogrande scenario); App. G.7. |

---

## Section 4 — Tracked parallel work (HIGH/MEDIUM)

External work that the program is monitoring for cross-reference, replication targets, or framework-applied scrutiny. Not on the must-cite list of any current manuscript, but prominent enough to flag.

### HIGH

| Paper | Date flagged | Notes |
|---|---|---|
| Tang et al. (2026). *Crowded in B-Space: Calibrating Shared Directions for LoRA Merging.* arXiv:2604.16826 | 2026-04-24 | A/B asymmetry decomposition complements Gradience's spectral partition finding. If we ever revisit option (a) of the Gradience program, **Pico-under-the-framework is a candidate replication target** — what's its cross-seed reliability, what's its tolerance schedule for the claimed 3.4–8.3 point gains, what confound decomposition does it survive? |
| Camuffo, Gambardella, Kazemi, Malachowski, Pandey (2026 v3). *Variance-Aware LLM Annotation for Strategy Research.* arXiv:2601.02370 | 2026-04-26 | Closest methodological cousin surfaced in second-pass review. 41-pp main + 53-pp appendix; generalizability-theory grounding identical to ours; same vocabulary ("auditable measurement infrastructure"); five-source variance taxonomy (construct specification, interface effects, model preferences, output extraction, system-level aggregation); 12–85 pp swings demonstrated. Substrate (LLM-as-text-annotator in strategy research) distinct from ours; methodological vocabulary essentially identical. Must-cite for benchmark study; the convergence is striking. |

### MEDIUM

| Paper | Date flagged | Notes |
|---|---|---|
| Chowdhury et al. (EMNLP 2025). *Spectral Scaling Laws in Language Models.* arXiv:2510.00537 | 2026-04-24 | Asymmetric soft-rank vs hard-rank scaling in FFN weights. Methodology-relevant if the spectral substrate is revived; LOW for current execution. Do NOT conflate with Gradience's compression-line findings (different scoring objects). |
| Wang (April 2026). *Grokking as Dimensional Phase Transition in Neural Networks.* arXiv:2604.04655 | 2026-04-24 | Effective cascade dimension D(t) as alternative phase-transition detector. **Framework-applied note**: D(t) is itself a measurement instrument; if the program ever revisits training-dynamics phase transitions, it could be a third demonstration substrate for the framework. |
| Wang (April 2026). *Dimensional Criticality at Grokking Across MLPs and Transformers.* arXiv:2604.16431 | 2026-04-24 | Companion to above; TDU-OFC offline probe is computationally tractable for reanalysis of existing telemetry. |

---

## Section 5 — Substrate-deemphasized (LOW, watch-only)

Methodologically related to areas the program has actively decided to deemphasize. Track for completeness; do not action-force.

These all fall under "merge-substrate work" (the line negatively settled at N134 decoder scale) or "training-dynamics work" (deferred per Gradience options memo). New entries here should be rare and require explicit justification for inclusion.

| Paper | Date flagged | Notes |
|---|---|---|
| Lunardi et al. (2025, ECAI 2025). *On Robustness and Reliability of Benchmark-Based Evaluation of LLMs.* arXiv:2509.04013 | 2026-04-25 | Paraphrase-sensitivity across 34 LLMs / 6 benchmarks. Within the established Sclar / Polo prompt-sensitivity literature. LOW additive value. |
| Hasan et al. (2025/2026). *Pitfalls of Evaluating Language Models with Open Benchmarks.* arXiv:2507.00460 | 2026-04-25 | Data-leakage / static-benchmark gaming; different scope (contamination, not measurement-condition variance). |
| Various LoRA-merging method papers (LoRI, DO-Merging, RobustMerge, TT-LoRA MoE) | 2026-04-25 | Multi-task LoRA merge methods. Substrate negatively settled for Gradience; informs follow-up rather than current execution. |

---

## Section 6 — Outstanding retrieval items

Papers flagged as potentially relevant but not adequately assessed yet. Most are paywall-blocked or otherwise need a manual retrieval step.

| Paper | Date flagged | Status |
|---|---|---|
| Sui et al. (2025/2026). *Evaluating large language models as raters in large-scale writing assessments: A psychometric framework for reliability and validity.* Computers and Education: Artificial Intelligence (Elsevier) | 2026-04-25 | **Potentially CRITICAL** — applies G-theory and many-facet Rasch modeling to LLM raters. Direct overlap with benchmark study Analyses 1–2. Elsevier paywall blocked the daily reviewer's fetch. **Action:** retrieve via institutional access, authors' personal pages, or preprint search. Promote to Section 4 once verified. |

---

## Section 7 — Daily review trajectory

Brief one-liner per dated daily report under `research_review/`. Inventory updates flow from these reports.

| Date | Reviewer summary | New flagged items |
|---|---|---|
| 2026-04-25 (inaugural) | 14 candidates, 2 flagged | Messing 2026 (HIGH); Ye et al. 2026 (MEDIUM); Sui et al. (OUTSTANDING) |
| 2026-04-26 (second pass) | 4 HIGH-importance items missed by inaugural pass | NIST AI 800-2 (MUST-CITE); NIST AI 800-3 (MUST-CITE, affects D-09); NeurIPS 2025 construct-validity review (MUST-CITE); Camuffo et al. 2026 (HIGH) |

(Append future days as they arrive.)

---

## Section 8 — Notes for manuscript writing

These observations have accumulated across daily reviews and are worth surfacing when manuscript-revision time arrives. Treat as candidate-content for the §2 or §3.1 framing of either paper.

1. **Parallel-development register.** The program is no longer the only one applying measurement-decomposition to LLM evaluation. Across roughly six months, the field has produced: Messing (2026) TEE framework for LLM evaluation pipelines; NIST AI 800-2/800-3 voluntary practices and GLMM endorsement; the NeurIPS 2025 construct-validity review with 445-paper systematic taxonomy; Camuffo et al. (2026) variance-aware LLM annotation in strategy research; Ye et al. (2026) LLM-psychometrics review. The N134 manuscript's framing as "first to propose" must be retired entirely (not merely softened) in favor of "co-developed register, distinctive prescriptive contribution." The load-bearing differentiator is the **decimal-place precision tolerance schedule** as the prescriptive output — Messing's TEE optimizes budget allocation, NIST 800-3 endorses GLMM methodology, the construct-validity review provides a taxonomy, Camuffo et al. develop variance-aware annotation protocols, and Ye et al. survey psychometric instruments applied to LLMs-as-subjects, but none formulates a tolerance schedule that licenses decimal-place precision in reported scores. That is the contribution we still own.

2. **Recursive framework application.** The program's framework can be applied as a critical instrument to the papers in this inventory. Specifically: Messing's "halves estimation error" claim is reported as a percentage gain, not a bounded interval — a framework-applied scrutiny would request reliability evidence on the small-sample pilot itself. This is the "framework-as-meta-tool" angle a future paper could exploit (analyzing recent ML-eval methods under measurement-discipline standards).

3. **Substrate-deemphasis defensibility.** The program's decision to deemphasize the merge substrate is increasingly visible in the inventory: most LoRA-merge work (Pico, LoRI, DO-Merging, etc.) sits in Section 4-5 territory rather than Section 2's must-cite. The N134 paper's discussion should name this shift explicitly — the merge-line is not "abandoned"; it's "negatively settled at decoder scale, with the framework now applied to a different substrate."
