# Internal Summary — Measurement Discipline Paper (N134 / Thesis B)

**Purpose.** Honest reference document for the author. Not for reviewers. Names what the paper actually claims and does not claim, which limits are load-bearing (including ones not foregrounded in the paper itself), and where the framework points next — for the Gradience research program and for the measurement-discipline thesis considered independently.

**Date.** 2026-04-23, after A3 template port; before A4 submission.

---

## What the paper actually claims

The paper is a *normative-methodological* paper with a worked empirical example, not an empirical-methods paper with a methodological preface. This matters because the two framings generate different reviewer expectations and different truth conditions for whether the paper succeeds.

**The normative core.** ML diagnostics — scores claimed to predict model behavior, training dynamics, merging outcomes, or capability — should adopt the measurement-theoretic infrastructure that psychological assessment treats as baseline methodology. Concretely: reliability coefficients, standard errors of measurement, construct-validity decomposition in roughly Messick's register, pre-registered decision rules with explicit tolerance schedules, and confound decomposition that distinguishes the diagnostic's explanatory share from simpler alternatives. The current ML practice of reporting point estimates to three or four decimal places without this infrastructure systematically overclaims precision and generality. The paper's force here is argumentative rather than empirical: it asks readers to treat measurement as the phase that precedes claims, not the post-hoc bookkeeping for claims already made.

**The worked empirical example.** At decoder scale (Mistral-7B-v0.3), the paper tests a pre-registered spectral triage score ($S_\mathrm{H1}$) against a sharp three-criterion decision rule under deliberate confound-defeat design (source-metric dynamic range, task-family non-partition, within-task variance, no post-hoc fitting). The rule returns a clean null: $\rho_\mathrm{partial} = -0.533$ against a committed threshold of $+0.5$, $\Delta R^2 = 0.003$ against a committed threshold of $0.1$, sign-incorrect. The coarser task-boundary-detection claim replicates across three architectures and two metric families. Cross-seed reliability on the $S_\mathrm{H1}$ instrument is moderate (ICC(2,1) = 0.566, SEM = 0.014). In the course of subjecting these results to measurement-discipline scrutiny, the paper surfaces an intrinsic-precision observation: rank-based correlations on small-$N$ residuals have floating-point precision of order $10^{-2}$, comparable to sampling variability — meaning the third and fourth decimal of such a number is not a real claim.

**The meta-claim that binds the two.** Measurement discipline applied to ML diagnostics produces findings that unstructured reporting would not surface — and the rank-on-residuals precision observation is itself an instance of the paper's central thesis, not an incidental finding. Three findings emerge from the framework's application: the pre-registered null, the moderate-reliability bound, and the intrinsic-precision observation. These split into two epistemic kinds — *discovery* (the precision observation, unnamed prior) and *calibration-and-modesty* (the reliability bound and the null). Both are framework products; they differ in epistemic shape. A framework yielding only one of the two kinds would either overreach or be merely defensive. The two kinds together are what measurement discipline is *for*.

## What the paper does not claim

The paper is careful about this, but worth naming the absences explicitly for internal clarity.

It does *not* claim that spectral geometry is useless for merge decisions. It claims that this specific pre-registered instantiation at decoder scale under this confound regime does not clear the gate. Activation-based methods (Zhou et al. 2026 report $|r| = 0.572$ on TSV-relevant activation metrics at vision-classifier scale) remain live; the mergeability-prediction literature broadly is unaffected by this null.

It does *not* claim that psychometric methods port wholesale to ML. The analogy is staged carefully. Psychometric construct validity has a referent — the construct — with prior theoretical substance independent of the measurement. Many ML "constructs" (mergeability, task similarity, capability) are operationally defined by the diagnostic under test, which raises a circularity concern the paper flags in §7 without fully resolving.

It does *not* claim to present a new method. The $S_\mathrm{H1}$ score exists to be an instrument the measurement framework can be tested against; its instantiation is purposeful but not novel in the way a merge-execution-method paper would be.

It does *not* claim pre-registration is sufficient. The paper argues pre-registration is a necessary condition for the class of discipline it advocates; the sufficient conditions include the reliability, construct-validity, and tolerance-schedule components that are co-load-bearing with pre-registration.

## The limits that are load-bearing

The paper's Appendix F (deviations from pre-registration) and §8 (objections) foreground some limits. Others are worth naming internally because they affect the paper's forward trajectory even though they are not reviewer-critical.

*Sample size and scale.* $N = 45$ cross-task pairs is what the pre-registration specifies, which is enough to reject $\rho_\mathrm{partial} \geq 0.5$ cleanly but not enough to narrow the rank-ordering of methods in the four-method comparison (Appendix C). Larger $N$ would tighten the CIs but would not change the magnitude of the committed point estimate. The cross-architecture replication on task-boundary detection uses smaller encoder backbones (the precursor studies), not multiple decoder-scale models — so the decoder-scale finding is single-base-model.

*Instrument reliability at the floor.* ICC = 0.566 is moderate by psychometric convention, meaning the paper's own headline instrument has less-than-ideal reliability. The paper flags this honestly in Appendix D and argues that *reporting* moderate reliability is exactly the framework's contribution — an unreliable-but-transparent instrument is better than an apparently-reliable instrument whose reliability was never tested. That argument is correct but points at a tension the paper does not resolve: the framework's normative force is strongest when applied to high-stakes diagnostics whose reliability bounds matter decisively; applied to a moderate-reliability instrument already returning a null, the framework is contributing the bounds but the diagnostic itself is at the edge of the regime where those bounds are load-bearing.

*The SEM transferability caveat.* The ICC design targets same-task pairs where alignments are in the 0.044–0.112 range; transferring SEM to cross-task precision claims (where $S_\mathrm{H1} \in [0.015, 0.025]$) is a broader inferential step than the estimate directly licenses. Appendix D names this; internally, it's worth remembering that the SEM we report is a same-task SEM used to bound cross-task precision claims, and the implicit stationarity assumption is not trivially warranted.

*Confound-defeat as a package.* The four confounds (C1–C4) are pre-registered as a joint constraint. If the null is robust to relaxing any one of them, we don't know it; the design doesn't include ablations. A determined reviewer could argue that C2 (task-family non-partition) is doing much of the work — that confound was what the precursor study revealed as the dominant failure mode, and forcing the distribution to span five task families may have independently destabilized whatever weak signal existed.

*Adaptation of baseline methods.* The four-method comparison (Appendix C) adapts KnOTS, TSV, and SVC to the pairwise-triage setting. The original methods were not designed for pairwise triage — they were designed for multi-adapter merge execution. A method author can reasonably object that the adapted version is not the method. The paper's honest response is: the adapted version is the best available pairwise-triage instantiation of the geometric assumptions those methods rest on, and its failure is evidence about those assumptions, not necessarily about the methods as originally deployed. But this is a weak point.

*Methodological register is CTT-flavored.* What the paper calls "measurement discipline" is closer to classical test theory (CTT) than to modern psychometrics. IRT-style item-level decomposition, multilevel modeling of seed/task/architecture variance components, and generalizability theory's multifaceted reliability frameworks all offer finer-grained tools the paper does not use. CTT was chosen partly because it's simpler to port and partly because the critiques CTT raises (reliability, SEM, construct validity) are the ones ML needs first. A follow-up engaging IRT-style decomposition would be a richer paper, and arguably a truer one.

*No institutional argument.* Pre-registration's normative force in psychology and medicine comes partly from ecosystem infrastructure — registered-reports venues, trial registries, ethics boards — that ML does not have. The paper argues for a practice without arguing for the institutions that would make the practice enforceable. This is a real gap but not one a single workshop/journal paper can close.

*Construct validity is under-resolved.* The paper gestures at Messick's construct-validity framework but does not fully exploit its six-aspect structure (content, substantive, structural, generalizability, external, consequential). In particular, *consequential* validity — the claim that a diagnostic's impact on downstream decisions is itself part of its validity — maps cleanly onto ML's "deployment decisions driven by diagnostic scores" setting and deserves its own treatment.

## Next implications

Three threads, separable but related.

### For Gradience as a research program

The spectral-triage-for-merging hypothesis at this specific instantiation is negatively settled at decoder scale under controlled confounds. The forward options are:

*(a) Retire triage-via-spectral-geometry; redirect to substrate generality.* The routing pilot recorded in the Gradience architecture assessment (2026-03-29) shows the four-layer substrate model (measurement → diagnosis → aggregation → policy) carries across scenarios with zero module modifications. The measurement framework this paper argues for maps cleanly onto the substrate's measurement layer. The substrate generalization is a more interesting forward direction than retrying spectral triage — it positions Gradience as a compatibility engine parameterizable by scenario rather than as a merge-specific tool.

*(b) Return to triage via activation-based methods.* Zhou et al. 2026's $r = 0.572$ on activation-dot-product metrics (vision scale) is informative enough to be worth testing at decoder scale under analogous confound control. This would be a direct successor experiment, inheriting the N134 pre-registration machinery and swapping the instrument. Attractive because the pre-registration infrastructure is already built; less attractive because it retries a class of hypotheses the current paper's framing would read as already retired.

*(c) Pursue the measurement framework independent of the triage claim.* Treat the framework as the product. The worked example stays but the next papers apply the framework to unrelated diagnostic contexts — capability evaluations, reliability audits of released models, pre-registered robustness claims. This is the strongest thesis extension and the weakest product extension.

My (the author's) current internal preference, unstated in the paper: (a) + (c). The substrate extraction is the empirical program; the framework extension is the philosophical program; the two co-evolve with the triage hypothesis as a cautionary example rather than as a live research bet.

### For the measurement-discipline thesis, considered independently of Gradience

One worked example is an existence proof for the framework's productivity. The thesis becomes load-bearing when a second unrelated diagnostic context surfaces findings via the framework. Candidate contexts worth next-year attention:

- Capability benchmark score variance. The current practice of reporting single-run benchmark numbers to four-decimal precision is directly in the paper's crosshairs, and the construct-validity critique would bite hard there — what construct does "MMLU score" measure, and what reliability does the instrument have?
- Evaluation leakage as a construct-validity failure. The paper frames this in §7 briefly; a standalone treatment would engage the Messick consequential-validity aspect directly and would be philosophically richer than the merging case.
- Fine-tuning diagnostic stability over training. Pre-registering reliability bounds on training-dynamics diagnostics (Fisher-trace, Hessian spectra, gradient-norm trajectories) before running production fine-tunes. The framework's tolerance-schedule component is directly useful here.

Any of these, done under the framework, would convert the measurement-discipline thesis from "one demonstration" to "systematic practice."

### Philosophical

The paper is methodologically argued but philosophically implicit. Two threads worth developing as separable work in a philosophy-of-ML venue rather than an ML venue:

*The transcendental argument.* The paper treats measurement as a precondition for claims. There is a stronger, Kantian-flavored argument available: what makes something a diagnostic rather than an opinion is precisely the availability of reliability, validity, and precision bounds; a diagnostic report without these is not a weak diagnostic report but a category error (an opinion disguised as a measurement). The current paper stops short of this argument because it's too philosophical for an ML venue. Worth developing as a companion piece in a philosophy-of-science journal.

*The two-kinds-of-findings structure.* The §6.4 distinction between discovery and calibration-and-modesty is philosophically rich. It connects to debates in philosophy of measurement about whether the role of measurement is to *read off* a pre-existing quantity (the realist/substantivist reading) or to *bound and regulate* subsequent inference (the constructivist/operationalist reading, closer to Hasok Chang and Eran Tal). The paper's implicit position is that both kinds of findings are measurement-discipline products, which is ecumenical but under-argued. A standalone treatment could position measurement discipline as a stance that is indifferent between the two philosophical readings — it takes bounding work seriously whether or not there's an underlying quantity to read off — and argue that this is the right stance for ML precisely because ML constructs are often operationally defined.

## What this paper is load-bearing for

Both for Gradience as a program and for the author's broader research trajectory:

- Establishes that the measurement register can be sustained at full paper length with a non-trivial worked example. This is a capacity demonstration as much as an argument.
- Produces one unambiguous technical result (the rank-on-residuals precision observation) that is reusable independent of the framework — anyone running rank-based correlations on small-$N$ residuals should be cautioning the third decimal.
- Retires the spectral-merge-triage hypothesis at decoder scale cleanly, which frees research attention without leaving the question ambiguous.
- Positions a research program where the framework itself is the product and the applications are instantiations. The Gradience substrate generalization then becomes the first of those applications rather than the primary product.

What the paper is *not* load-bearing for: any claim about whether Gradience-the-tool should continue as a merge-specific product. That's a product decision informed by the paper but not determined by it.
