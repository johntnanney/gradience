# Thesis B Paper — Draft Outline v1

**Working title:** *Measurement Discipline for ML Diagnostics: A Psychometric Framework with a LoRA-Merging Case Study*

**Status:** outline, not prose. Destined for `papers/n134_workshop/draft_v2_thesis_b_outline.md`. Supersedes the v1 findings-paper skeleton for the purpose of revision planning; v1 is retained as `draft_v1.md` per consolidation convention.

**Thesis (one sentence).** ML diagnostic metrics are routinely reported as point estimates without the measurement-theoretic infrastructure — reliability coefficients, standard error of measurement, bounded precision claims, confound decomposition — that psychological assessment has treated as baseline methodology for decades, and the cost of this gap is that published claims about diagnostic performance systematically overclaim their precision and generality in ways that a psychometric perspective would catch.

**What this paper is not.** It is not the N134 findings paper. It is not a merge-methods paper. It is not a Gradience tool paper. N134's findings function as the extended worked example that demonstrates the thesis; the Gradience program's existence motivates the case-study choice; but the paper's argument is general and its target audience is methodology-adjacent ML researchers rather than the LoRA-merging subfield.

**Target venue candidates (in rough order of fit).** ICML Position Paper track. ML Reproducibility workshop (NeurIPS or ICML). NeurIPS Datasets & Benchmarks track (as a measurement-methods adjacent submission). ICLR blog post track if the argument is tight enough to land in blog format. The venue decision is downstream of the first internal review pass, not of the outline.

**Target length.** 8–10 pages main body plus appendices, positioning it as a substantial position-paper / methods-paper submission rather than a workshop-length short paper. This is longer than the v1 draft's workshop-paper target. The length is justified because Thesis B requires framing, theoretical development, and worked example in sequence, none of which can be compressed to 4–6 pages without losing the argument.

---

## Section-by-Section Outline

### Abstract (~200 words)

Leads with the methodological claim, not the empirical finding. The empirical result (H1 null, three-architecture replication of task-boundary detection, regime-null on four-method comparison) appears in the abstract only as the worked example, not as the primary contribution.

Draft sketch: "ML diagnostic metrics — scores claimed to predict model behavior, merging outcomes, training dynamics, or capability — are routinely reported as point estimates to three or four decimal places without the measurement-theoretic infrastructure that psychological assessment has treated as baseline methodology for decades. We argue that this practice systematically overclaims precision and generality, and we introduce a framework for applying classical measurement theory (reliability coefficients, standard error of measurement, confound decomposition, pre-registered decision rules with explicit tolerance schedules) to ML diagnostic reporting. We demonstrate the framework through a pre-registered decoder-scale study of spectral LoRA-merging diagnostics (N = 45 cross-task adapter pairs on Mistral-7B-v0.3), which produces a null on per-pair merge prediction under family-confound control, replicates task-boundary detection across three architectures and two metric families, and — in the course of being subjected to measurement-discipline scrutiny — surfaces a previously unnamed property of the primary test statistic: rank-based correlations on small-N residuals have intrinsic floating-point precision of order 10⁻², comparable to their sampling variability. We argue this last observation is not an incidental finding but an instance of the paper's central claim: measurement discipline applied to ML diagnostics produces findings that unstructured reporting would not surface."

### 1. Introduction (~1 page)

**1.1 The reporting gap.** Opens with two concrete examples from the current ML literature of diagnostic metrics reported without measurement-theoretic context — one from the LoRA-merging subfield, one from an unrelated subfield (interpretability, training dynamics, or alignment evaluation). Shows that the pattern is not field-specific. Names the gap explicitly: reliability coefficients are nearly absent from reported ML diagnostic results; point estimates are quoted to three or four decimals without uncertainty bounds respecting the actual dependence structure in the data; confound decomposition is rare and, when present, usually conducted post-hoc without pre-registration.

**1.2 The psychometric analog.** Three paragraphs on how psychological assessment handles the same class of problem. Reliability as a first-class property of an instrument. Standard error of measurement as the decimal-place precision claim. Construct validity as the methodological discipline for claiming an instrument measures what its name says it measures. The psychometric tradition is not presented as the only source of the relevant methodology, but as a mature and well-developed one that ML has drawn from less than it could.

**1.3 Contribution claims.** Three, in order:

1. A four-component framework for applying measurement discipline to ML diagnostics: (i) construct validity, (ii) reliability coefficients, (iii) bounded precision claims with explicit tolerance schedules, (iv) confound decomposition with pre-registered decision rules.
2. A detailed worked example applying the framework to spectral LoRA-merging diagnostics, including a pre-registered decoder-scale null on per-pair merge prediction under family-confound control, a three-architecture replication of task-boundary detection, and a pre-registered four-method comparison.
3. A specific previously-unnamed measurement property of partial Spearman correlations on small-N residuals, surfaced by the measurement-discipline scrutiny the framework prescribes, which has implications for any rank-based diagnostic metric reported at small sample sizes.

**1.4 Road map.** §2 develops the framework. §3 introduces the worked example's setting. §4–§6 present the worked example. §7 returns to the framework and generalizes from the case study to the broader reporting practice. §8 addresses limitations, alternative framings, and common objections to the thesis.

### 2. A Measurement-Discipline Framework for ML Diagnostics (~1.5 pages)

The framework section. This is new material relative to v1, and it is the section that most determines whether the paper reads as a position paper with teeth or as a findings paper with methodological framing.

**2.1 Construct validity.** What is the diagnostic claiming to measure? For an ML diagnostic, this requires stating the theoretical object (merge compatibility, capability presence, training-dynamics regime, etc.) and then explicitly defending the operationalization (why this particular computation from model weights or activations or outputs constitutes a measurement of the theoretical object). The ML literature's typical failure mode is conflating operationalization with construct, reporting results about the former as if they were about the latter.

**2.2 Reliability.** For a diagnostic intended to support predictive claims, what is its test-retest behavior across seeds, calibration data samples, or implementation details? An ML diagnostic reported as "the score is ρ = 0.573" without an accompanying cross-seed ICC or SEM is making an implicit claim that the score is perfectly reliable, which is almost never true. The paper's recommendation is that ICC or a distribution-based equivalent be reported for any diagnostic used to support quantitative predictive claims.

**2.3 Bounded precision.** The decimal-place precision of a reported diagnostic value implicitly communicates confidence. A value reported as "−0.533" claims precision the data may not support. Bounded precision is the discipline of stating explicitly what the uncertainty is on the reported decimal places — from sampling variability, from implementation sensitivity, from cross-environment reproduction. The rank-on-residuals observation from the N134 worked example is an instance of this: the paper's committed −0.533 has intrinsic precision of roughly ±0.01, which means the third decimal is not a real claim.

**2.4 Confound decomposition with pre-registered decision rules.** Distinguishing what a diagnostic predicts from what is predictable by simpler alternatives (task identity, input length, surface features). The N134 worked example makes this concrete: 88% of merge-outcome variance is captured by task-family-pair identity alone, leaving 12% within which a geometric diagnostic must operate to clear ΔR² ≥ 0.10. Pre-registration of the decomposition is what distinguishes the framework from post-hoc confound analysis.

### 3. Case Study: Spectral Diagnostics for LoRA Adapter Merging (~0.5 page)

Brief section. Not the paper's center of gravity. Introduces the domain just enough for a non-merging-subfield reader to follow. Explains LoRA, adapter merging, the decision a merging practitioner has to make (which adapters to combine), and the role of spectral diagnostics in supporting that decision. Cites KnOTS, TSV, SVC, OSRM briefly as representative of the spectral-methods literature. Names the Gradience program as the source of the pre-registered study that constitutes the worked example.

**Important framing note.** This section treats the LoRA-merging setting as incidental to the paper's argument. The paper's argument would hold equally for interpretability scores, capability evaluations, or any other ML diagnostic class. LoRA merging is the case the authors happened to study rigorously; it is not the paper's subject.

### 4. Applying the Framework: Pre-Registration and Design (~1 page)

**4.1 Construct articulation for the spectral triage claim.** What does "spectral alignment predicts per-pair merge outcome" assert as its theoretical object? We state it explicitly: a geometric property of two adapters' weight-space updates should predict the degradation observed when the two adapters are linearly combined and evaluated on their source tasks. The operationalization — O-module depth-weighted SV-weighted alignment — is one specific computation among many that could plausibly measure this object.

**4.2 Reliability considerations at pre-registration time.** The 2-seeds-per-task design of the precursor study (N133) estimated same-task alignment variance from 6 observations across 6 tasks. The N134 design commits to 3 seeds per task specifically to enable cross-seed reliability estimation. We report cross-seed ICC for the primary score at the audit stage, before merge evaluation, as part of the instrument validation. [Actual ICC from the data to be inserted at revision time.]

**4.3 Confound decomposition pre-registration.** Four confounds identified from the precursor study (C1 source-metric dynamic range, C2 task-family partition, C3 within-task variance, C4 post-hoc fitting). Each maps to a specific design choice. The decision rule is pre-registered as a partial correlation on family-residualized data with ΔR² threshold, not a raw correlation, specifically to force the diagnostic to produce information beyond what task-family identity alone provides.

**4.4 Decision rule with explicit precision.** The pre-registered thresholds (partial ρ ≥ 0.50, ΔR² ≥ 0.10) are not arbitrary — they are calibrated to represent practically meaningful effect sizes for the triage decision. We report both the point estimate and a bootstrap confidence interval on the partial correlation, and we commit pre-experimentally to interpreting a significant but wrong-signed result as a null under the rule, not as a reversed confirmation.

### 5. Worked Example: Empirical Results (~1.5 pages)

This section contains the N134 empirical results but subordinates them to the framework's demonstration.

**5.1 H1 outcome under the pre-registered rule.** The primary hypothesis was not confirmed. Partial ρ = −0.533 (p = 1.6 × 10⁻⁴), wrong-signed under the pre-registered directional test; ΔR² = +0.003, far below the 0.10 threshold. Under the pre-registered decision rule, H1 is a null. Figure 1 (the H1 decision figure from v1) appears here.

**5.2 Why this is an informative null, framework-wise.** Task-family-pair identity explains 88.1% of outcome variance; the geometric score contributes +0.003. The framework's confound-decomposition discipline is what makes this informativeness legible — without pre-registered residualization, the raw correlation (also negative, smaller magnitude) would have been reported without context, and the reader would have had no way to assess what fraction of any predictive claim was actually family identity operating under a different name.

**5.3 Three-architecture replication of task-boundary detection.** B-P1, B-P2, B-P4 all pass on N134 with the usual numbers. Combined with prior results on DistilBERT and DeBERTa (two architecture classes, two metric families), task-boundary detection replicates cleanly. Figure 3 (layer-depth trend from v1) appears here, supporting the depth-dependence observation. The three-architecture claim is framed honestly — two metric families, overlapping research teams — and the framework's construct-validity discipline is what enforces that honesty rather than letting the claim overreach.

**5.4 Four-method comparison as regime-scope test.** Gradience, KnOTS, TSV, SVC all fail to clear significance on the 45-pair sample under family-confound control. Three of four produce wrong-signed correlations in [−0.275, −0.180]. Figure 2 (four-method forest plot from v1) appears here. The framework's reading: this is evidence that the measurement substrate (weight-space) rather than any specific algorithmic choice within it is the binding constraint on per-pair prediction at this scale. Honestly flagged as one plausible interpretation rather than a definitive claim.

### 6. A Previously-Unnamed Measurement Property (~0.75 page)

New material, the paper's most distinctive methodological contribution, surfaced by the consolidation-pass reproducibility check.

**6.1 Observation.** Partial Spearman ρ computed on OLS residuals at n = 45 has intrinsic floating-point precision of approximately ±0.01. Reproduction in a different numerical environment produced ρ = −0.545 versus committed ρ = −0.533, with every other scalar (raw ρ, R², ΔR², bootstrap statistics) reproducing to within 10⁻³ or bit-identical. Localized to the numerical locus: rank-based statistics on small-N residuals amplify floating-point path differences in the residualization step that aggregate quantities absorb.

**6.2 Implication.** The headline statistic for the pre-registered H1 decision has intrinsic precision comparable in magnitude to its sampling-variability precision from n = 45. A responsible reporting convention would present the value as ρ ≈ −0.53 ± 0.01 rather than as −0.533; the third decimal is not a real claim. At n = 200 or larger, the rank-statistic precision would stabilize substantially; at n = 45, it does not.

**6.3 Generalization.** Rank-based statistics are common in ML diagnostic reporting — Spearman correlations, Kendall's tau, rank-based robustness metrics, ranked model comparisons. Whenever these are computed on residuals (i.e., after controlling for another variable via regression), the residualization step is a numerical-precision amplifier. Any paper reporting such statistics at small sample sizes without explicitly bounding the precision is making an implicit claim the data doesn't support. This is not a one-off observation about N134; it is a general measurement property the framework's reproducibility discipline surfaced.

**6.4 What this demonstrates about the framework.** The paper's central claim is that measurement discipline applied to ML diagnostic reporting surfaces findings that unstructured reporting would not. The rank-on-residuals observation is an instance: it was not in the original N134 analysis; it was not in the committed report; it was surfaced specifically because the framework's reproducibility check ran the analysis in a different environment and compared tier-by-tier with a tolerance schedule calibrated to quantity class. If the framework had not been applied, the N134 paper would have reported −0.533 to three decimals, and no one — including the authors — would have known the third decimal was not reliable.

### 7. Generalizing from the Case Study (~1.25 pages)

**7.1 The reporting practices the framework prescribes.** Pulled together as a short list suitable for direct adoption by ML methods writers. Cross-seed ICC or equivalent reliability metric reported alongside any diagnostic claim. Bounded precision with explicit tolerance schedule, calibrated to quantity class and sample size. Confound decomposition with pre-registered decision rules, reporting the family-residualized quantity rather than the raw one. Floating-point reproducibility check with tiered verification, not binary pass/fail. Post-hoc analyses labeled as such with no evidential weight; the C4 discipline as minimum-viable pre-registration hygiene.

**7.2 What this changes about how existing claims should be read.** The paper does not re-adjudicate specific published claims, but it does argue that readers of the ML-methods literature should calibrate their credence to the measurement-discipline practices of reports, not to the reported point estimates. A diagnostic reported as "Spearman ρ = 0.572" without a cross-seed distribution, without a family-residualized partial correlation, and without a reproducibility-check trace is making a weaker evidential claim than the reported value suggests. Readers should discount accordingly.

**7.3 Why this is not a critique of any specific paper.** The practices the framework prescribes are not currently normative in the ML-methods literature. Papers not following them are not violating field norms; they are following the norms that exist. The paper's claim is that the norms should change, and that individual papers should begin applying the framework voluntarily ahead of any field-wide shift. We apply it ourselves in N134; we don't ask anyone else to retroactively apply it to work they've already published.

**7.4 The psychometric-tradition reference.** Explicit acknowledgment that the framework is drawn from classical test theory and its modern descendants (IRT, generalizability theory, measurement invariance), with citations to canonical references (Cronbach 1951 on alpha; Shrout & Fleiss 1979 on ICC; Messick 1995 on construct validity; Cronbach & Meehl 1955 on nomological networks). The paper is not claiming originality for the measurement-theoretic concepts; it is claiming that their systematic application to ML diagnostics is underdeveloped and proposing a specific framework for that application.

### 8. Objections, Alternatives, Limitations (~1 page)

Handling the predictable pushback explicitly rather than leaving it to reviewers.

**8.1 "This is just good statistics; nothing psychometric-specific is needed."** Reply: classical test theory contains specific machinery (ICC, SEM, construct validity, convergent and discriminant validity) that is not covered by statistics coursework typical in ML training. The framework is "just good statistics" in the sense that it is not mathematically novel, but it is a specific coordinated application of methods that are not currently coordinated in ML practice.

**8.2 "Pre-registration is high-overhead and impractical for most ML research."** Reply: the N134 experiment ran end-to-end in 36 hours for $40 on commercial cloud compute with a pre-registration written in one document committed before data collection. The overhead is lower than typical reviewer-response cycles. The practicality objection reflects current norms, not actual cost.

**8.3 "The worked example is a null; the framework would be more compelling with a positive result."** Reply: the framework's value is demonstrated by what it rules out as much as by what it confirms. A framework that would have licensed a false-positive claim (as N133's original B-P5 analysis almost did) is a framework worth adopting. The null is a feature of the worked example, not a weakness of the framework.

**8.4 "Rank-on-residuals floating-point sensitivity is a niche observation; generalizing from it is overreach."** Reply: the observation is not the paper's center of gravity; it is a demonstration instance of the framework producing findings that unstructured reporting doesn't. The generalization is to the class of rank-based statistics on small-N residuals, which is not niche — it covers a meaningful fraction of diagnostic reporting in the ML-methods literature.

**8.5 Limitations proper.** Single backbone (Mistral-7B). Single LoRA rank (r = 16). Single merge operation (0.5/0.5 linear). Single class of diagnostic metric (spectral). The framework's claims generalize beyond these specifics, but the worked example's empirical claims do not; a reader interested in the framework's application to their own diagnostic class will need to carry out their own application rather than directly importing N134's results.

### 9. Conclusion (~0.5 page)

Short. Restates the thesis. Restates the three contributions. Names the direction the program the paper emerges from is taking next (capability-expansion via activation-informed follow-up, v2.0 of the tool with validated expanded scope). Acknowledges that the framework is a starting point rather than a complete methodology and that its specific prescriptions will evolve with application experience.

### References (~0.75 page)

Covers: the LoRA-merging literature (Hu, Stoica, Gargiulo, Li, Zhang & Zhou, Rahamim, Zhou), the classical measurement-theoretic literature (Cronbach, Shrout & Fleiss, Messick, Cronbach & Meehl), relevant ML reproducibility and methodology writing (reproducibility-crisis papers, position papers on measurement in ML if any exist, NeurIPS reproducibility checklist), the Gradience program's prior publications (self-citations, anonymized for submission).

### Appendices (~2–3 pages)

**A. Precise specification of the H1 score and decision rule.** As in v1 Appendix A.

**B. Pairwise triage adaptations of KnOTS, TSV, SVC.** As in v1 Appendix B, with expanded caveat language reflecting the paper's honesty-about-adaptations discipline.

**C. Cross-seed ICC for the primary diagnostic score.** New appendix. Reports the reliability coefficient computed on the 24 adapters (3 seeds × 8 tasks), with a short explanation of what the coefficient means and how to interpret it for readers not fluent in the psychometric tradition. Data source: N134 audit JSONs.

**D. Full reproducibility-check trace.** New appendix, summarizing `sidecar/notes/n134_reproducibility_check.md` for the paper's audience. Includes the tier-by-tier table, the rank-on-residuals locus analysis, and the tolerance-schedule justification.

**E. Deviations from pre-registration.** As in v1 Appendix D.

**F. Compute and environment.** As in v1 Appendix E, with the environment gap between committed and reproducing environments documented.

---

## What Changes from v1

**Sections 1 and 2 are new.** The v1 draft opened with a findings-paper introduction; Thesis B opens with the methodological argument. This is the biggest structural change in the paper and the one that most determines whether the revision succeeds or fails.

**Sections 5 and 6 replace v1's §3–§4 results sections.** The N134 empirical findings are preserved but restructured so each subsection has an explicit "what this demonstrates framework-wise" beat. Numbers and figures are unchanged; framing is restructured throughout.

**Section 6 is partially new.** The rank-on-residuals observation exists in v1 only as a reproducibility-statement item. In Thesis B, it expands into a dedicated section because it is the paper's cleanest demonstration of the framework producing findings unstructured reporting wouldn't.

**Section 7 is new.** The v1 discussion focused on what N134 establishes and does not establish empirically. The Thesis B §7 generalizes from the case study to broader reporting practice, which is a different intellectual operation.

**Section 8 is new.** The v1 draft did not explicitly engage likely reviewer objections. Thesis B papers live or die on whether they preempt the objections that position-paper reviewers inevitably raise; §8 exists to do that preemption work explicitly.

**v1's §6 limitations section is absorbed into §8.5.** Treating limitations as part of objection-handling rather than as a standalone section makes the paper's defensive posture tighter.

**v1's four contribution claims become three.** The old "pre-registration infrastructure held under stress" observation (v1 §6.5) becomes part of the framework demonstration in §4 rather than a standalone contribution. This tightens the paper's thesis to three coordinated claims (framework, worked example, specific measurement property) rather than four co-equal ones.

## What Gets Cut or Demoted

**The "regime null, not a Gradience null" framing** becomes one line in §5.4 rather than a headline claim. Thesis B doesn't need it to be a headline; the four-method comparison's role is to demonstrate the framework's four-method-comparison discipline, not to license a claim about the class of weight-space methods.

**The four candidate interpretations of the wrong-signed partial ρ** (v1 §6.3) shrink to a single paragraph in §5.1 or §8.5 acknowledging the directional information without pursuing it. Thesis B isn't making a claim that needs this material; the candidate interpretations are for the N135-alt follow-up paper.

**The explicit next-research-directions discussion** (v1 §6.2, §7) shrinks to a single sentence in §9. Thesis B isn't a program-overview paper; its forward-looking content is about framework adoption, not about the specific follow-up experiments the authors will run.

**The "methodological remark" currently at v1 §6.5** was the Thesis B argument in embryo. It is now distributed throughout the paper rather than concentrated in a single remark. Specifically: the framework is §2, the framework-applied is §4, the framework-demonstrated is §6, the framework-generalized is §7. The entire paper is the expanded version of what was one paragraph in v1.

## What Stays Unchanged

All empirical numbers. All figures (F1/F2/F3). The pre-registration document referenced throughout. The deviations catalogue. The appendices A, B, E, F (renamed but substantively the same).

The N134 experiment's scientific claims are not re-adjudicated in Thesis B; they are preserved and reframed. A reader interested purely in the LoRA-merging findings can read §§3–5 and get substantially what the v1 draft offered. A reader interested in the methodological argument reads §§1–2 and §§6–8 and gets what only Thesis B provides.

## Immediate Next Actions

1. Draft the thesis memo (one page) articulating the three contribution claims precisely and the framework's four components precisely. This is the first piece of revision-period writing; it should happen before any prose revision of the draft itself.

2. Pull the measurement-theoretic references (Cronbach, Shrout & Fleiss, Messick, Cronbach & Meehl) into `references.bib` with full canonical citations. These are not on arXiv; verification comes from published-journal records.

3. Compute the cross-seed ICC on existing N134 audit data for the primary diagnostic score. This is a one-day analysis on committed data; the result populates Appendix C and grounds §4.2's reliability discussion.

4. Audit `draft_v1.md` for material that should be preserved verbatim in the Thesis B version versus material that should be rewritten from the new framing. The section structure here is the guide; any v1 prose that fits a Thesis B section's purpose can be reused, and anything that doesn't gets cut or rewritten.

5. Begin Section 2 prose. This is the section most differentiated from v1 and the section that most determines whether the paper works. Starting here — rather than at the introduction or the worked example — forces the framework's articulation to be substantive before any of the material around it takes shape.

## One Structural Decision to Re-Confirm Before Drafting

The outline above assumes Thesis B is a standalone paper that does not depend on N135-alt's outcome. That was the decision from the prior planning conversation (N135-alt as a separate paper; Thesis B paper with just N134 as worked example). Confirming this here because the outline's scope and length budget are calibrated to that decision. If N135-alt's results were to be folded into the paper, §5 would expand substantially and the paper would approach 12–14 pages rather than 8–10, which changes the venue options.

Decision is preserved as made: standalone Thesis B paper, N134 material only, 8–10 pages, submission before N135-alt runs.

---

*Outline v1. Prepared as the first revision-period artifact following consolidation closure. Destined for `papers/n134_workshop/draft_v2_thesis_b_outline.md`. To be revised against the thesis memo before any prose drafting begins.*
