# Executive Internal Memo — Measurement Discipline Paper (N134 / Thesis B)

**Purpose.** Internal reference for collaborators after A3 template port and before A4 submission. This memo states what the paper actually establishes, what it does not establish, the limits that matter for interpretation, and the most plausible next directions for both Gradience and the broader measurement-discipline program.

## Bottom line

This paper is best understood as a normative-methodological paper with a worked empirical example, not as an empirical merge-method paper with a methodological preface. That distinction is crucial. Its success does not depend on showing that a spectral merge score works. It depends on showing that ML diagnostics should be treated as fallible measurement instruments whose evidentiary status must be established before they are used to support claims or decisions.

The paper succeeds on that ground. It demonstrates that applying a measurement framework to an ML diagnostic can generate three kinds of useful output that ordinary reporting would likely miss: a clean preregistered null, an explicit reliability bound, and a precision limit that invalidates spurious decimal-level claims.

## What the paper actually claims

The normative claim is straightforward: ML diagnostics should adopt baseline measurement infrastructure that psychology would treat as standard methodological discipline. In this paper that means reliability estimates, standard errors of measurement, construct-validity reasoning in a Messick-like register, preregistered decision rules with tolerance schedules, and confound decomposition that distinguishes the diagnostic's contribution from simpler alternatives. The paper argues that current ML practice often overclaims precision and generality by reporting point estimates without this infrastructure.

The worked example is a preregistered test of a spectral triage score, $S_{\mathrm{H1}}$, at decoder scale on Mistral-7B-v0.3 under a deliberately confound-defeating design. Under a sharp three-criterion decision rule, the score fails cleanly: partial correlation is negative and sign-incorrect, incremental $R^2$ is negligible, and the committed thresholds are not met. At the same time, a coarser task-boundary-detection claim does replicate across architectures and metric families. Cross-seed reliability for the instrument is moderate rather than strong.

The paper's most reusable technical contribution is the precision observation surfaced by the framework itself: for rank-based correlations on small-$N$ residuals, floating-point precision is on the order of $10^{-2}$, comparable to sampling variability. In practical terms, the third and fourth decimal places are not meaningful claims. That is not an incidental footnote. It is an instance of the paper's central thesis: disciplined measurement can surface findings that unstructured reporting conceals.

## What the paper does not claim

The paper does not claim that spectral geometry is useless for merge decisions. It claims that this specific preregistered instantiation fails at decoder scale under controlled confounds. Activation-based approaches and other mergeability-prediction lines remain open.

It does not claim that psychometric methods transfer wholesale into ML. The analogy is careful and limited. Psychological constructs often have prior theoretical substance independent of the measure; many ML constructs are more operationally circular, and the paper flags that tension without resolving it.

It does not claim to introduce a new merge method. The spectral score is an instrument used to test the framework, not the paper's primary innovation.

It does not claim that preregistration alone is enough. The paper's view is that preregistration is necessary but only meaningful when paired with reliability, validity, tolerance schedules, and confound analysis.

## Load-bearing limits

Several limits matter for interpretation and for what comes next.

The sample size is sufficient to reject the preregistered success criterion cleanly, but not to finely rank competing methods. The decoder-scale finding is also single-base-model rather than multi-decoder replication.

Instrument reliability is only moderate. That does not weaken the methodological argument; in some sense it strengthens it, because the framework exposes this explicitly. But it also means the headline instrument is operating near the edge of where its own bounds matter most.

The reported SEM is derived from same-task settings and then used to help interpret cross-task precision claims. That transfer is reasonable but not trivial, and the stationarity assumption should not be forgotten internally.

The confound-defeat design is load-bearing as a package. Because the paper does not ablate the individual confounds, we do not know which constraints are doing the most work in suppressing signal.

Finally, the framework is closer to classical test theory than to the full richness of modern psychometrics. That is appropriate for a first paper, but it leaves open more sophisticated future work involving multilevel variance decomposition, generalizability theory, or IRT-like approaches.

## Strategic implications

For Gradience, the cleanest conclusion is that spectral triage for merging in this exact form is negatively settled at decoder scale under controlled confounds. The strongest forward options are: retire merge-specific spectral triage and move toward substrate generality; pursue activation-based triage as a successor experiment; or decouple the measurement framework from the merge setting and apply it elsewhere. The best current internal reading is substrate generalization plus framework extension, with the present triage hypothesis treated as a cautionary example rather than an ongoing central bet.

For the measurement-discipline thesis independently, this paper should be treated as a first demonstration, not an endpoint. The next serious test is a second unrelated domain where the framework surfaces a finding ordinary ML reporting would have missed. The best candidate remains benchmark-evaluation reliability, with training-dynamics diagnostics and evaluation leakage as other strong possibilities.

## What this paper is load-bearing for

This paper establishes that the measurement register can sustain a full ML paper with a nontrivial empirical example. It produces one reusable technical result — the precision warning for small-$N$ rank-based residual correlations. It cleanly retires one specific spectral-merge-triage hypothesis at decoder scale. And it opens a broader program in which the framework is the durable intellectual core and the application domains are successive demonstrations.

What it does not determine is whether Gradience should continue as a merge-specific product. That remains a separate strategic decision. The paper informs that decision, but does not settle it.

## Standing guidance

Collaborators should treat this paper as the first demonstration of inferential measurement in ML diagnostics, not as a merge paper that happened to produce a null. The key next step is not to defend it harder, but to apply the framework in a second domain and show that it continues to generate findings that ordinary reporting would have missed.
