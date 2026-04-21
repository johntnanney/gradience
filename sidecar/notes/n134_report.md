# N134 — Decoder-Scale Controlled Merge Triage (Final Report)

**Status:** complete. Data collected, analyses run, H1 decision emitted. Pod decommissioned.

**Timeline.** Spec v3.1 committed 2026-04-19. Data collection began 2026-04-19. End-to-end experiment (Phase 0 pilot through Phase 5 comparison) completed in approximately 30 hours on a single RTX 6000 Ada 48GB node (RunPod Secure Cloud). Total compute spend: approximately $40.

**Related documents.**
- `sidecar/notes/n134_spec.md` — pre-registration (v3.1, this document's evidential frame)
- `sidecar/notes/n134_incident_log.md` — the CPU-contention incident log (Phase 0)
- `sidecar/results/n134/` — all artifacts (audit summaries, pair alignments, merge evaluations, analysis outputs, figures). Large per-adapter SVD binary sidecars (`.npz`) were produced on the pod but not committed to git; they were the source data for Phase 5's `LazyPairV21` synthesizer.
- commit range: `643c192` (spec + scripts) through `3f6a7c7` (final figures), with all data in `0a827e2`, `4677dde`, `1527d72`, `3f6a7c7`.

---

## 1. Abstract

We report the outcome of N134, a pre-registered decoder-scale test of Gradience's spectral triage hypothesis on Mistral-7B-v0.3. The primary hypothesis — that an O-module depth-weighted SV-weighted alignment score ($S_{\text{H1}}$) predicts per-pair merge degradation, with Spearman partial $\rho \geq 0.50$ and $\Delta R^2 \geq 0.10$ over a task-family-pair baseline — **was not confirmed**. The observed partial correlation was $\rho = -0.533$ ($p = 1.6 \times 10^{-4}$), statistically significant but opposite in sign to the pre-registered prediction; $\Delta R^2$ was $+0.003$. Under the pre-registered decision rule, this constitutes a null outcome (sign constraint not met), not a reversed confirmation. Three confirmatory replications of task-boundary detection (B-P1 zero-overlap, B-P2 same/cross ratio, B-P4 erank-by-task ANOVA) passed cleanly, adding Mistral-7B as a third architecture for this finding alongside DistilBERT (N127, N130) and DeBERTa (N132). A pre-registered four-method comparison (Gradience, KnOTS, TSV, SVC) applied to the same 45 cross-task pairs produced no statistically significant rank correlation with merge degradation for any method; three of four methods produced wrong-signed correlations in the range $\rho \in [-0.275, -0.180]$. Task-family-pair identity alone explains 88% of max-degradation variance (OLS $R^2 = 0.881$ under FAMILY_B). We read the result as a regime null at $N = 45$ cross-task pairs under confound control: at this budget and on this task set, no off-the-shelf weight-space spectral triage method we tested clears its pre-registered threshold. Gradience's validated decoder-scale operating surface narrows to task-boundary triage. Per-pair risk regression within-family is outside the current measurement envelope and is a candidate subject for activation-informed follow-up work (N135-alt).

---

## 2. Methods

This section summarizes the pre-registered design. The canonical specification is `sidecar/notes/n134_spec.md`; readers are referred to §§2–6 of that document for the full rationale, confound analysis, and statistical protocol. What follows is the minimum needed to evaluate the findings.

**Model and adapters.** Mistral-7B-v0.3 base, LoRA at rank 16 targeting `q_proj`, `k_proj`, `v_proj`, `o_proj` with $\alpha = 32$. AdamW, lr $= 2 \times 10^{-4}$, 6% warmup, bf16 precision. Three seeds per task (42, 123, 456); eight pre-registered tasks yielding 24 adapters. Training duration per task was scaled to target the pre-registered [0.70, 0.90] validation-accuracy band rather than a fixed step count. All 24 adapters landed in-band at final checkpoint; per-adapter accuracies are in `sidecar/results/n134/audit/adapter_profiles.json` and summarized in Appendix C.

**Task set and families.** The pre-registered eight-task set (see spec §3.2) spans five task families: science QA, commonsense completion, coreference/commonsense, knowledge QA, commonsense QA (two tasks), physical commonsense, social commonsense, and reading comprehension, with no family exceeding two tasks. The family labels are the pre-committed FAMILY_B partition used for residualization in the H1 decision rule. Pilot-stage task replacements (if any) are documented in §8.

**Audit.** Schema version 2.1. Per-layer U and V orthonormal factors, singular values, stable rank, energy rank at 90%, entropy effective rank, and Frobenius norm persisted for every LoRA layer (32 layers × 4 modules × 24 adapters = 3,072 layer-module records). This schema change from the N133 audit was the hard precondition for the Phase 5 four-method comparison; storing the factors adds approximately 50 MB per adapter in `.npz` binary sidecars (approximately 1.2 GB in total for 24 adapters). These sidecars live on the pod and were read by Phase 5 via a lazy per-pair synthesizer; the per-adapter JSON summaries committed to the repository contain the spectral statistics but not the full factors.

**Pair sampling.** All 276 adapter pairs were audited. Merge evaluation sampled 69 pairs: all 24 same-task pairs plus 45 cross-task pairs drawn using the committed RNG (`N134_PAIR_SEED = 134`) from the C(8, 2) = 28 cross-task-type cells, with a minimum of one observation per cell. The pair sample (`pair_sample.json`) was git-committed before any merge evaluation began, locking the sample against post-hoc reselection.

**Merge and outcome.** 0.5/0.5 linear merge of adapter weight deltas. Each merged adapter was evaluated on both source tasks' held-out validation sets. The outcome variable is `max_degradation` $= \max(s_A - m_A, s_B - m_B)$, where $s_A, s_B$ are source accuracies and $m_A, m_B$ are the merged model's accuracies on task A and task B respectively.

**Statistical protocol.** Pre-registered block bootstrap for all confidence intervals, with blocks defined by task-family-pair cell and 5,000 resamples. Significance threshold $\alpha = 0.05$ for individual tests. No multiple-comparison correction applied to the single pre-registered H1 test. Tied-pair handling: H1 computed at float64 precision; pairs with $|S_A - S_B| < 10^{-6}$ treated as genuinely tied and ranked by bootstrap-median tiebreak.

**Hardware and compute.** Single RTX 6000 Ada 48GB on RunPod Secure Cloud. Approximately 30 hours end-to-end across all phases (pilot + training + audit + merge evaluation + analysis + comparison). Cost approximately $40. Full environment specification in Appendix D.

---

## 3. Primary Results: H1

### 3.1 H1 score and decision rule

The pre-registered primary score is O-module depth-weighted SV-weighted alignment:

$$
S_{\text{H1}}(\text{pair}) \;=\; \frac{\sum_{\ell=1}^{L} w_\ell \cdot \alpha^{(O)}_\ell}{\sum_{\ell=1}^{L} w_\ell}
$$

with $\alpha^{(O)}_\ell$ the SV-weighted alignment on the O-projection of layer $\ell$, $L = 32$, and $w_\ell = \ell / L$. The pre-registered decision rule requires **both** of the following to hold, with sign correct (higher $S_{\text{H1}}$ predicting higher `max_degradation`):

1. Spearman partial $\rho(S_{\text{H1}}, \text{max\_degradation})$ residualized on FAMILY_B task-family-pair dummies $\geq 0.50$ with $p < 0.05$; and
2. OLS $\Delta R^2$ of $S_{\text{H1}}$ over the FAMILY_B-only baseline $\geq 0.10$.

### 3.2 H1 outcome

**H1 is not confirmed.**

| Quantity | Observed | Required for H1 | Status |
|---|---|---|---|
| Raw Spearman $\rho$ | $-0.180$ | sign $\geq 0$ | fails |
| Partial $\rho$ (FAMILY_B) | $-0.533$ ($p = 1.6 \times 10^{-4}$) | $\geq +0.50$, right sign | fails |
| $\Delta R^2$ (FAMILY_B) | $+0.003$ | $\geq 0.10$ | fails |
| FAMILY_B-only $R^2$ | $0.881$ | (reference) | — |

Block-bootstrap (5,000 resamples, family-pair blocked) 95% confidence intervals: partial $\rho \in [-0.825, -0.131]$; $\Delta R^2 \in [2 \times 10^{-5}, 0.023]$.

### 3.3 Reading the wrong-signed significant partial correlation

The partial correlation is both statistically significant and sizable in magnitude. It is also opposite in sign to the pre-registered prediction. Under the pre-registered decision rule this outcome constitutes a null — the sign constraint is not met — **not a reversed confirmation**. The pattern is of potential hypothesis-generating interest for subsequent pre-registered work, but it licenses no conclusion from N134's data.

Stating this cleanly matters because the failure mode N134 was designed to rule out is exactly the kind of interpretive slide that treats a significant-but-unexpected result as evidence "for something." The C4 constraint in the spec (no post-hoc fitting, no retrospective score revision) applies with equal force to the sign of the primary score. If the hypothesis space had included "higher $S_{\text{H1}}$ predicts lower degradation" as an equally-motivated alternative, the spec would have registered a two-sided test. It did not; the N133 Phase-2 signal-to-noise argument (O-projection gives 7.23× same/cross separation; deeper layers separate same/cross more sharply) straightforwardly predicts positive $\rho$. The observed negative $\rho$ is a hypothesis-generating residual, not a partial win.

### 3.4 The informative content of the null

What the null does license is the observation that **task-family-pair identity alone accounts for 88% of the variance in max_degradation** on this evaluated sample. Under FAMILY_B, $R^2 = 0.881$ for a pure task-family-pair dummy regression with no geometric features at all. The geometric score $S_{\text{H1}}$ contributes $+0.003$ additional $R^2$ — less than a tenth of one percent of additional explained variance once family membership is known. This is the payload of the null: at this budget on this task set, knowing which families a cross-task pair spans is a near-complete account of how damaging the merge will be. The residual room within which a spectral score must operate is narrow enough that no score we tested could occupy it.

This is substantively different from the N133 null. In N133, $R^2$ for FAMILY_B was $0.97$ on a 12-pair evaluation — a ceiling so high that no score *could* clear $\Delta R^2 = 0.10$ from the residual. In N134, $R^2$ drops to $0.88$ on a 45-pair evaluation, which opens approximately 12 points of $R^2$ room for a geometric score to inhabit. The tested scores do not inhabit it.

---

## 4. Confirmatory Replications

Three replications of N133's architecture-general findings were specified in the N134 spec as non-gating but reported alongside H1. All three passed.

**B-P1: Task-boundary detection — zero overlap.** On Mistral-7B with the N134 task set, 0 of 24 same-task pairs fell below the same/cross midpoint threshold, and 0 of 252 cross-task pairs rose above it. The alignment signal perfectly discriminates the task boundary at this budget. Combined with the DistilBERT (N130) and DeBERTa (N132) results, this is the third architecture — and the second decoder — on which task-boundary detection replicates with zero false negatives and zero false positives at the tested budget.

**B-P2: Same/cross spectral separation.** Ratio of mean same-task alignment to mean cross-task alignment was **2.28×**, Welch's $t = 15.9$, $p = 6 \times 10^{-14}$. The ratio is smaller than N133's 3.06× on Mistral-7B with the six-task N133 set but remains well above the pre-registered 2.0× threshold. The smaller ratio under N134's tighter task-family diversity is consistent with the intuition that more closely-spaced cross-task pairs yield higher cross-task alignment, compressing the separation.

**B-P4 first-half: Erank varies systematically by task.** Between-adapter ANOVA on per-adapter mean erank: $F = 165.6$, $p = 1 \times 10^{-13}$. Erank continues to partition reliably by task, replicating N130, N132, and N133.

**What this combined passes mean.** The N134 spec flagged failure of any of these replications as itself a substantial finding, most likely pointing to task-set structural similarity caught by neither the pilot nor the dip-test for erank continuity. All three passing indicates that the task-set design is not pathological in ways that would invalidate the H1 test. The null on H1 cannot be attributed to a degenerate task set.

Task-boundary detection at decoder scale is now a three-architecture finding. It is the strongest and most robust empirical result in the Gradience program, and it is what the N134 null leaves intact as Gradience's validated decoder-scale contribution.

---

## 5. Secondary Exploratory Results (Non-Evidential)

The measures in this section were specified in the N134 spec as non-evidential and are logged here as hypothesis generators for subsequent work. None bears on the H1 decision. The analyses reported below were conducted before any post-hoc re-examination of scoring choices, and were pre-specified in the spec.

**V+O vs. Q+K alignment ratio.** Ratio of mean same-task V+O alignment to mean same-task Q+K alignment: 0.56. On Mistral-7B with the N134 task set, V- and O-projections are *less* aligned across seeds than Q- and K-projections, opposite the N133 pattern on a different task set and opposite the intuition (from earlier Gradience work on DistilBERT) that V/O carries the task-specific signal most cleanly. This observation is one of the residuals of the N134 experiment that most warrants careful replication before being treated as a claim about Mistral-specific module geometry.

**Depth-trend in same/cross ratio.** Correlation between layer depth and layer-level same/cross ratio: $r = 0.919$ on the 32 layers. Deeper layers separate same-task from cross-task pairs more sharply, replicating the N133 layer-depth trend. This observation is what originally motivated $S_{\text{H1}}$'s linear depth weighting. The N134 null means the motivated score does not predict per-pair risk; it does not mean the depth trend itself is false. The trend is robust and architecture-general now across N130, N133, and N134.

**Post-hoc composite score sweep.** Ten composite risk scores drawn from the N133 post-hoc search were recomputed on N134 data for comparison purposes only. These are logged under the explicit stipulation that they carry no evidential weight — the C4 constraint on scores not pre-registered in the spec applies with full force. Maximum observed values:

- $|\rho_{\text{partial}}|$ (FAMILY_B) = 0.479, for `O_deep_mean` (wrong sign).
- $\Delta R^2$ (FAMILY_B) = 0.013, for `erank_ratio` (only right-signed variant).

None of the ten scores clears the pre-registered decision rule by either criterion. `erank_ratio` is the single variant with a right-signed partial correlation, but its $\Delta R^2$ is roughly an order of magnitude below the threshold. These numbers are published in full in `sidecar/results/n134/analysis_secondary.json` and are discussed here only for the record.

---

## 6. Four-Method Scheduled Comparison

The spec's §6 pre-registered a four-method head-to-head on the same 45 cross-task pairs used for H1: Gradience $S_{\text{H1}}$, KnOTS (Stoica et al. 2024), TSV (Gargiulo et al. 2025), and SVC (Li et al. 2026). Each method is applied as a triage: rank the 45 pairs by the method's risk score, select the safe lowest-half (N = 22 pairs by float-precision midpoint), measure mean `max_degradation` in the retained set. Bootstrap confidence intervals are family-pair-blocked, 5,000 resamples.

Random baseline: mean `max_degradation` across all 45 cross-task pairs = 3.14%.

The KnOTS, TSV, and SVC scores here are **adaptations** of each method's core measurement quantity to the pairwise triage setting, not claims of operational equivalence to the published methods. We did not import the published reference implementations; instead, each of the three score functions in `scripts/n134/08_compare_methods.py` computes the quantity that the corresponding paper identifies as its central interference/inflation signal, aggregated at the pair level as a triage score. See the faithfulness and adaptation notes in the docstrings of `knots_score_from_v21`, `tsv_score_from_v21`, and `svc_score_from_v21` for the per-method documentation of what is preserved and what is departed from. A reviewer who reads "we used KnOTS" in this report should read it as shorthand for "we adapted KnOTS's shared-subspace interference-norm quantity to the pairwise triage setting, as documented in the code" — not as a claim that we executed the paper's full pipeline.

### 6.1 Results

| Method | Spearman $\rho$ | $p$ | Retained mean deg. | $\Delta$ vs. random | Bootstrap 95% CI (retained deg.) |
|---|---|---|---|---|---|
| Gradience $S_{\text{H1}}$ | $-0.180$ | $0.236$ | 4.27% | $-1.13$ pp | [0.84%, 8.61%] |
| KnOTS | $+0.183$ | $0.230$ | 2.18% | $+0.96$ pp | [$-1.18$%, 4.75%] |
| TSV | $-0.214$ | $0.159$ | 4.59% | $-1.45$ pp | [1.09%, 8.68%] |
| SVC | $-0.275$ | $0.068$ | 4.89% | $-1.74$ pp | [1.34%, 9.07%] |

### 6.2 Reading the comparison

**No method is statistically significant.** All four rank correlations have $p > 0.05$. SVC is closest to the threshold at $p = 0.068$, with the largest magnitude but the wrong sign. KnOTS is the only method with a right-signed correlation and positive improvement over random baseline, but its confidence interval for retained-set degradation crosses zero.

**The null is a regime null, not a Gradience null.** At $N = 45$ cross-task pairs under the confound-defeating task design, with task-family-pair identity carrying $R^2 = 0.881$ of the outcome variance, no off-the-shelf weight-space spectral triage method we tested clears pre-registered thresholds. This is a claim about what weight-space spectral methods can do at this scale and budget with family confound controlled — not a claim specific to Gradience's score. The measurement regime itself appears to be the binding constraint.

**Convergent wrong-sign pattern (hypothesis-generating only).** Three of four methods produce wrong-signed rank correlations in the range $\rho \in [-0.275, -0.180]$, while KnOTS produces a right-signed but small-magnitude correlation ($\rho = +0.183$). The convergence of three independently-motivated spectral scores on the same wrong-signed pattern is a residual of the experiment worth naming. It is not evidence from N134's data that the alternative hypothesis ("higher spectral alignment within a cross-task pair indicates *preserved* shared routing, not shared interference") is correct; that interpretation is outside the pre-registered hypothesis space and requires independent pre-registered testing. What the pattern does suggest is that an N135-style follow-up should pre-register an explicitly bidirectional hypothesis on per-pair weight-space alignment at decoder scale, because the N134 data show systematic directional information in the opposite direction from the Gradience prior.

**SVC-specific note.** SVC was designed for portfolio-scale merging (k ≥ 3 adapters). The pairwise triage adaptation used here computes SVC's SV-inflation index on each 2-adapter pair and uses it as a ranking signal; this is not the paper's intended use and the adaptation is documented in `scripts/n134/08_compare_methods.py`. The finding that SVC reaches the wrong-signed near-threshold most closely among the four methods should not be read as a claim about SVC's portfolio-scale behavior. It is consistent with the general pattern that weight-space SV-magnitude interaction carries some systematic per-pair signal on this data, but that signal is in the opposite direction from what any of the four methods predicts as "safe merge."

### 6.3 What the comparison does and does not establish

Three things the comparison does establish, within its descriptive pre-registration:

1. Gradience's pre-registered score does not clear a significance threshold at $N = 45$ with family confound controlled.
2. Three alternative spectral methods from the 2024–2026 LoRA-merging literature also do not clear a significance threshold under the same conditions.
3. The wrong-sign convergence across three methods is consistent enough to motivate a bidirectional N135 hypothesis, but is not itself a claim that can be licensed from this data.

Three things it does not establish:

1. That any of the four methods fails in general. The methods were designed for different tasks (triage, alignment-and-merge, whitening, SV calibration) across different regimes (pairwise, portfolio, encoder, decoder). N134 compares them on one specific triage objective at one scale on one task set. Additionally, the KnOTS/TSV/SVC scores here are pairwise-triage adaptations of each method's core measurement quantity; a reviewer insisting on literal-paper operational fidelity at k ≥ 3 should read the comparison as descriptive of the adaptation only.
2. That activation-informed spectral methods would fail. All four tested methods are static weight analysis only; none use calibration data. The activation-informed direction (OSRM-style, Zhang & Zhou ACL 2025) is explicitly out of scope for N134 and remains a live candidate.
3. That larger $N$ would not recover a signal. With 88% of variance going to family identity, detecting a $\Delta R^2 \geq 0.10$ residual at $N = 45$ requires the geometric score to carry a non-negligible effect on within-family residuals. $N = 100$ or $N = 200$ would provide more statistical room for modest effects; whether that would change the picture is empirical and not decidable from N134's data alone.

---

## 7. Discussion

### 7.1 What N134 establishes

Two things, clearly:

**First, task-boundary triage on LoRA adapters is architecture-general at decoder scale.** The B-P1/B-P2 results now replicate on three architectures (DistilBERT, DeBERTa, Mistral-7B) and at two scales (encoder-class and 7B-decoder-class). The same/cross alignment separation — zero overlap at the evaluated budget, with same/cross ratios of 5× (DistilBERT), 2.3× (DeBERTa), 3.06× (N133 Mistral), and 2.28× (N134 Mistral) — appears to be a reliable property of LoRA as a fitting procedure under task-conditional data distributions, not an artifact of any one backbone or task set. This is the Gradience program's strongest architecture-general empirical finding, and N134 is the third independent replication at decoder scale.

**Second, within-family per-pair risk regression using weight-space spectral triage is outside the current measurement envelope.** At $N = 45$ under confound control, neither Gradience's pre-registered score nor three alternative spectral methods can occupy the 12% of residual variance left after family-pair identity is accounted for. The hypothesis that weight-space spectral geometry carries per-pair risk information beyond task-family identity — a hypothesis the Gradience program has pursued since N128 — is empirically constrained to "if true, then too subtle to detect at this budget with these methods."

### 7.2 What N134 does not establish

It does not establish that decoder-scale per-pair risk triage is impossible. Specifically, four distinct follow-up programs remain live after N134:

**Larger-$N$ follow-up.** A budget-scaled replication (e.g., $N = 200$ evaluated cross-task pairs) would distinguish "effect too small to detect at $N = 45$" from "no effect present." This is the most conservative follow-up, and the one most directly licensed by N134's data. The tradeoff is that the GPU cost scales roughly linearly with evaluated pairs.

**Activation-informed follow-up (N135-alt).** OSRM (Zhang & Zhou, ACL 2025) demonstrates that activation-weighted overlap carries information that static weight overlap does not. Zhou et al.'s activation-dot-product result ($r = 0.572$) is within a factor of roughly two of the $\rho$ magnitude Gradience would need to clear H1. An N135-alt that pre-registers an activation-informed analog of $S_{\text{H1}}$ would test whether the measurement substrate, not the score or the $N$, is the binding constraint. This is the scientifically most consequential follow-up because it would move Gradience's measurement program from static-only to calibration-based, which is a substantive change in what Gradience *is* as an instrument.

**Intrinsic-mergeability follow-up.** Rahamim et al. (2026) propose that some adapters are intrinsically mergeable and others are not, in a way that is partner-independent. This framing would predict that per-pair prediction is the wrong object of study: the right object is per-adapter intrinsic property, with pairwise outcome determined primarily by whether both partners have it. N134 does not directly test this hypothesis, but the 88% family-$R^2$ result is consistent with a world in which most of the explainable variance is adapter-level rather than pair-level, and the residual pair-level variance is dominated by noise. A per-adapter-intrinsic-property analysis on N134 data (using Rahamim-style features computed from the U/V factors N134 persisted) would be a next-quarter exercise.

**Training-time intervention.** If weight-space per-pair prediction is genuinely outside the measurement envelope at this scale, the prescriptive implication is that merge-readiness should be cultivated during training rather than diagnosed afterward. The training-time intervention program (spec track 5) becomes more central post-N134, not less.

### 7.3 Four candidate interpretations of the wrong-signed partial $\rho$

The partial $\rho = -0.533$ result is statistically significant and systematic; it is not a null, it is directional residual information that the pre-registered decision rule correctly excludes from licensing any conclusion. Four candidate interpretations are available for pre-registered testing in subsequent work:

*Over-commitment within family.* Inside a tight family band like N134's [0.70, 0.90] accuracy window, adapters for genuinely similar tasks may share enough routing that high spectral alignment indicates *preserved* shared computation — a merge that preserves both source behaviors because the relevant directions are compatible. Low alignment within the same family may indicate adapters that over-committed to task-specific idiosyncrasies, which collide at merge time. Under this interpretation, the Gradience prior had the sign right for between-family pairs (where high alignment indicates collision) and the sign wrong for within-family pairs (where high alignment indicates compatibility). N134's confound-defeating design deliberately enriched within-family-adjacent pairs, which would flip the observed sign.

*Source-accuracy residual.* The [0.70, 0.90] band is narrow by design but not trivially narrow. Adapters at the 0.90 end have less room to degrade than adapters at the 0.70 end, and the `max_degradation` outcome may be picking up residual source-accuracy variance that correlates inversely with spectral alignment for reasons having nothing to do with merge geometry per se. A partial correlation that residualizes on both family and source-accuracy-range would test this; the analysis is runnable on N134 data as a subsequent exploratory exercise (flagged as non-evidential).

*Inverted-substrate measurement.* Weight-space spectral alignment may be measuring something systematically different from what merge outcomes depend on. The OSRM result — that activation-weighted overlap carries information weight-space overlap does not — suggests that the substrate matters. If the relevant substrate is activations-and-their-covariance rather than weight-deltas-and-their-subspaces, then weight-space scores may systematically *anti-correlate* with the true geometric risk variable in regimes where the activation-weighted overlap and the weight-space overlap diverge. This is a testable prediction: if Gradience's score on N134 is measuring the wrong substrate, an activation-informed score on the same data should both clear H1 *and* recover the predicted positive sign.

*Rahamim-style intrinsic mergeability.* Merge outcome may be primarily a property of each adapter individually (intrinsic mergeability), with pairwise alignment carrying only residual information. Under this interpretation, the 88% family-$R^2$ is not a confound to be controlled but rather a reflection of the fact that merge outcome is mostly task-family-level (via adapter-level properties that correlate with task family) rather than pair-level. The wrong-signed residual might reflect that high-alignment pairs tend to be pairs where *at least one* adapter has intrinsic over-fitting properties, which produces both the high alignment and the high degradation — a common-cause structure, not a causal pathway through alignment itself.

These four are offered as pre-registration targets for follow-up work. None is supported by N134's data. Naming them is the honest way to expose the residual information the null contains without treating that information as license for a claim N134 did not test.

### 7.4 Epistemological remark

N134's most consequential finding may be methodological rather than substantive. The pre-registration infrastructure held under the first real stress test in the Gradience program: a primary hypothesis failed in a specific, sign-reversing, statistically significant way that would have been very easy to rescue post-hoc, and it was not rescued. Three of four scheduled comparisons produced wrong-signed correlations that converge on a pattern consistent with a potentially interesting alternative hypothesis, and that alternative hypothesis is being held out for subsequent pre-registered testing rather than licensed from the current data. Ten post-hoc composite scores — ported directly from the N133 diagnostic that motivated N134's existence — were logged as non-evidential and stopped there.

This is not how most ML research reports failure, and it is worth naming. The program's distinctive commitment — that measurement instruments must be psychometrically disciplined, with pre-registered decision rules, confound controls, and honest accounting of what the data can and cannot support — passed the test. The hypothesis did not, but the epistemology did. Whether this commitment makes the work adoptable by the broader field is a separate question; that it survives its own first serious stress test is a prerequisite for that question being askable.

---

## 8. Deviations

All deviations below were caught before any analysis was committed to its output (i.e., before the H1 decision was computable from the committed data). None altered the pre-registered protocol, the task set, the training design, the H1 score definition, or the decision rule. All are code-level or infrastructure-level.

**D1 — `transformers` API change (Phase 0 pilot, day 1).** The `from_pretrained` signature in the installed version of `transformers` had migrated from `dtype=` to `torch_dtype=` between the spec-commit date and pilot training. Caught on the first pilot training attempt before any pilot adapter was trained. Fixed in commit `a4032e9`. No scientific consequence.

**D2 — Pair-key ordering mismatch (Phase 4 analysis, pre-H1).** The pair keys in the committed cross-task pair sample (`pair_sample.json`) were stored in the order the pairs were generated during stratified sampling, while the downstream analysis script `06_analysis_h1.py` initially constructed pair keys in alphabetical order. The mismatch produced a silent misalignment between `max_degradation` values and their corresponding $S_{\text{H1}}$ values, which would have corrupted the H1 correlation. Detected by a sanity check on H1 output values before the committed run. Fixed by introducing a `lookup_pair` helper at module level that tolerates either key ordering and maps to a canonical internal form. All Phase 4 and Phase 5 analyses use the corrected helper. The raw committed sample file is unmodified.

**D3 — Layer indexing and module-string bugs (Phase 2 audit and Phase 4 analysis).** Two related schema inconsistencies: the audit JSON persisted layer index under the key `layer_idx` while `compute_s_h1` initially read `layer["layer"]`; and the module field was persisted as `"o_proj"` while the score function initially checked `module == "O"`. Both bugs caused the H1 score to be computed over an empty set of layers on the first dry run. Detected before any H1 value was committed. Fixed together in the same commit that fixed D2. No scientific consequence.

**D4 — JSON bool serialization (Phase 4 output).** The Phase 4 analysis output JSON initially failed to serialize `numpy.bool_` values (decision flags) using the default JSON encoder. Fixed with a custom `default=` that handles `numpy.bool_`, `numpy.integer`, `numpy.floating`, and `numpy.ndarray`. No scientific consequence.

**D5 — CPU-contention incident (Phase 0 pilot, training phase).** During the first pilot training run, background audit processes from an earlier phase were still running on the pod and competing for CPU resources, causing `hellaswag` training to slow from approximately 1 s/step to approximately 18–21 s/step during a 13-minute window. Four rogue PIDs were identified (`pgrep -u $USER python`) and killed; training recovered to approximately 2.14 s/step for the remainder of the task, which post-incident investigation established as the honest steady-state for the workload rather than residual contention. The incident and its diagnostic analysis (loss-trajectory continuity check; absence of time-leaked control flow in `00_pilot_train.py`; step-based scheduler) are documented in `sidecar/notes/n134_incident_log.md` for the record. No scientific consequence for the trained adapters, whose final validation accuracies landed in-band and whose spectral audits passed schema validation.

**D6 — Per-pair v2.1 audit file shape (Phase 5 plumbing only; does not affect the Phase 4 H1 test).** `08_compare_methods.py` (the Phase 5 four-method comparison script) was written against a hypothetical per-pair v2.1 directory (`audit/v2.1/*.json`) that was never populated — `03_spectral_audit.py` writes per-adapter `.npz` sidecars instead, and writing the per-adapter v2.1 JSONs in parallel would have exhausted the pod's MooseFS quota. Resolved by writing a lazy per-pair synthesizer (`LazyPairV21` in `08_compare_methods.py`) that reconstructs the per-pair dict shape on demand from the per-adapter SVD factors. No method math changed; only the I/O path differed from the original script's expectation. Fixed in commit `4677dde`, and the first and only set of method-comparison numbers committed to the repository (`method_comparison.json`, committed in `1527d72`) was produced under the fixed plumbing — no pre-fix numbers were ever committed or reported.

This deviation is evidentially separate from the Phase 4 H1 test. Phase 4 (the pre-registered primary hypothesis test) reads from `pair_alignment_full.json`, not from the per-pair v2.1 path; its result was computed and committed before Phase 5 was touched. The H1 decision in §3 is therefore unaffected by D6. The scope of D6 is confined to Phase 5, which is descriptive pre-registration under the spec, not primary confirmatory inference.

No deviations occurred after H1 was computable from committed data. The primary decision was made under the pre-registered protocol without modification.

---

## 9. Conclusion

N134 delivers a decision. The pre-registered primary hypothesis — that O-module depth-weighted spectral alignment predicts per-pair decoder-scale merge degradation — is not confirmed, with a wrong-signed significant partial correlation that constitutes a null under the pre-registered decision rule rather than a reversed result. Three confirmatory replications of task-boundary detection pass cleanly, establishing that finding on three architectures. A four-method pre-registered comparison finds no weight-space spectral triage method clearing significance at $N = 45$ cross-task pairs with family confound controlled.

Gradience's operationally validated decoder-scale surface narrows to task-boundary triage. Per-pair risk regression within-family is outside the current measurement envelope. This is a specific boundary on the approach, not a falsification of the measurement framework: the instrument reads the task boundary reliably across three architectures, and the B-P1/B-P2 result is publishable on its own terms. What N134 rules out is a class of predictive claims about per-pair outcomes that weight-space-only spectral scores can make at this scale on this task design.

Three next directions are licensed by the finding. First, writing the N134 paper — the null itself is a substantive contribution to a literature (Zhou et al., Rahamim et al., Cocchieri et al., and the 2025–2026 spectral-merging wave) that has not yet produced a clean pre-registered decoder-scale null under confound control. Second, drafting a psychometric-methods paper with N134 as a worked example; the program's distinctive commitments (pre-registration discipline, confound decomposition, honest accounting of what data licenses) translate more cleanly into a methodological contribution now that there is a non-toy case study of those commitments held under stress. Third, the activation-informed follow-up (N135-alt); the wrong-signed significant partial correlation is informative enough, and the OSRM / activation-dot-product literature close enough, that an explicitly pre-registered bidirectional activation-informed test becomes the most consequential next experiment.

The experiment cost approximately $40 and 30 hours end-to-end. That it produced a publishable result at that budget is a data point about the protocol itself: a disciplined pre-registered decoder-scale experiment is within reach of an independent researcher on RunPod Secure Cloud. The protocol infrastructure is what makes the work durable beyond any single experiment's outcome, and it is the thing worth consolidating into whatever shape the subsequent publications take.

---

## Appendix A — Exact H1 Score Definition and Decision Rule

**Score.** For a pair of adapters $(a, b)$ trained on tasks $T_a, T_b$ (with $T_a \neq T_b$ for cross-task pairs), let $\alpha^{(O)}_\ell(a, b)$ denote the SV-weighted alignment on the O-projection of layer $\ell$, computed from the U and V factors persisted in the v2.1 audit schema. Specifically,

$$
\alpha^{(O)}_\ell(a, b) \;=\; \frac{\sum_{i, j} \sigma_i^{(a)} \sigma_j^{(b)} |\langle u_i^{(a)}, u_j^{(b)} \rangle| \cdot |\langle v_i^{(a)}, v_j^{(b)} \rangle|}{\big(\sum_i (\sigma_i^{(a)})^2\big)^{1/2} \big(\sum_j (\sigma_j^{(b)})^2\big)^{1/2}}
$$

restricted to the O-projection of layer $\ell$. The primary score aggregates these layer-level alignments with a linear depth weight:

$$
S_{\text{H1}}(a, b) \;=\; \frac{\sum_{\ell=1}^{32} (\ell / 32) \cdot \alpha^{(O)}_\ell(a, b)}{\sum_{\ell=1}^{32} (\ell / 32)} \;=\; \frac{\sum_\ell \ell \cdot \alpha^{(O)}_\ell(a, b)}{\sum_\ell \ell}.
$$

**Decision rule.** Let $F(a, b) \in \mathcal{F}$ be the unordered family-pair label for the pair $(a, b)$, where $\mathcal{F}$ is the FAMILY_B partition from spec §3.2. Let $D(a, b) = \text{max\_degradation}(a, b)$. Compute:

1. $\rho_{\text{raw}} = \text{Spearman}(S_{\text{H1}}, D)$ over the 45 evaluated cross-task pairs.
2. $\rho_{\text{partial}} = \text{Spearman}(S_{\text{H1}}^{\perp F}, D^{\perp F})$, where $\cdot^{\perp F}$ denotes OLS residuals after regressing on FAMILY_B dummies.
3. $R^2_{\text{family}}$ from OLS $D \sim F$; $R^2_{\text{full}}$ from OLS $D \sim F + S_{\text{H1}}$; $\Delta R^2 = R^2_{\text{full}} - R^2_{\text{family}}$.

H1 clears iff: $\rho_{\text{partial}} \geq +0.50$, $p(\rho_{\text{partial}}) < 0.05$, and $\Delta R^2 \geq 0.10$. Sign constraint on $\rho$ is absolute; wrong sign fails regardless of magnitude.

---

## Appendix B — Bootstrap Protocol

All confidence intervals reported in §§3–6 use block-bootstrap resampling with blocks defined by task-family-pair cell (under FAMILY_B). Each bootstrap replicate resamples whole family-pair cells with replacement, rather than individual pairs, to respect the dependence structure that produced N133's Simpson's paradox.

Number of resamples: 5,000 for all reported CIs. Confidence level: 95%, computed as empirical 2.5th and 97.5th percentiles of the bootstrap distribution.

For the tied-pair tiebreak in the four-method comparison (§6), pairs with $|S_A - S_B| < 10^{-6}$ on any method's score are ranked by the median of the bootstrap distribution of the score differences, computed once per resample and then aggregated across resamples.

Full implementation: inline in `scripts/n134/06_analysis_h1.py` and `scripts/n134/08_compare_methods.py`.

---

## Appendix C — Per-Adapter Metadata

All 24 adapters trained to validation accuracy within the pre-registered [0.70, 0.90] band at final checkpoint. Observed range: [0.745, 0.870]. No retry-ladder invocation was triggered (no task required retraining at r=8 or r=32). Full per-layer spectral metadata is in `sidecar/results/n134/audit/adapter_profiles.json`; the accuracy summary below is derived from the source scores embedded in `sidecar/results/n134/merges/merge_eval_summary.json`.

**Pilot-stage task replacements:** none required. All eight pre-registered tasks (arc_challenge, hellaswag, winogrande, openbookqa, commonsenseqa, piqa, siqa, boolq) landed inside the [0.70, 0.90] band on first attempt at pilot (seed 42) and at full-sweep (seeds 123 and 456). The reserve task list committed in the spec was not touched.

| Adapter | Task | Seed | Val accuracy | Mean erank |
|---|---|---|---|---|
| `arc_challenge_s42` | arc_challenge | 42 | 0.770 | 8.990 |
| `arc_challenge_s123` | arc_challenge | 123 | 0.820 | 8.986 |
| `arc_challenge_s456` | arc_challenge | 456 | 0.810 | 9.040 |
| `boolq_s42` | boolq | 42 | 0.870 | 11.450 |
| `boolq_s123` | boolq | 123 | 0.825 | 11.178 |
| `boolq_s456` | boolq | 456 | 0.820 | 11.288 |
| `commonsenseqa_s42` | commonsenseqa | 42 | 0.855 | 7.372 |
| `commonsenseqa_s123` | commonsenseqa | 123 | 0.835 | 7.442 |
| `commonsenseqa_s456` | commonsenseqa | 456 | 0.840 | 7.428 |
| `hellaswag_s42` | hellaswag | 42 | 0.745 | 8.845 |
| `hellaswag_s123` | hellaswag | 123 | 0.795 | 8.026 |
| `hellaswag_s456` | hellaswag | 456 | 0.780 | 8.055 |
| `openbookqa_s42` | openbookqa | 42 | 0.815 | 8.179 |
| `openbookqa_s123` | openbookqa | 123 | 0.830 | 8.254 |
| `openbookqa_s456` | openbookqa | 456 | 0.850 | 8.310 |
| `piqa_s42` | piqa | 42 | 0.850 | 8.908 |
| `piqa_s123` | piqa | 123 | 0.805 | 8.912 |
| `piqa_s456` | piqa | 456 | 0.850 | 8.947 |
| `siqa_s42` | siqa | 42 | 0.790 | 7.567 |
| `siqa_s123` | siqa | 123 | 0.810 | 7.629 |
| `siqa_s456` | siqa | 456 | 0.810 | 7.667 |
| `winogrande_s42` | winogrande | 42 | 0.800 | 7.207 |
| `winogrande_s123` | winogrande | 123 | 0.865 | 7.061 |
| `winogrande_s456` | winogrande | 456 | 0.825 | 7.212 |

Rank used: 16 for all 24 adapters.

---

## Appendix D — Hardware and Environment

**Hardware.** Single RTX 6000 Ada 48GB on RunPod Secure Cloud. No multi-GPU training; no gradient checkpointing.

**Software environment (pod, at time of experiment).** Python 3.11. Key library versions, inferred from the compatibility fixes required during D1/D2/D3 and from the pod's pre-installed stack at the commit dates of the N134 scripts:

- `torch` 2.3.x
- `transformers` 4.44.x (post-`dtype=`/`torch_dtype=` migration)
- `peft` 0.12.x
- `datasets` 2.x
- `gradience` 0.11.x
- `numpy` 1.26.x
- `scipy` 1.13.x

A `pip freeze` snapshot of the pod environment was not captured before the pod was decommissioned. A best-effort development-environment snapshot, captured post-hoc from the machine used to author and debug the analysis scripts, is provided at `sidecar/results/n134/environment_dev.txt` with a header comment that states explicitly what it is and is not. It shares major versions of the core libraries with the pod environment (`torch` 2.x, `transformers` 4.x, `peft` 0.x) but may differ in transitive dependencies and in the newer minor/patch versions the dev environment runs (Python 3.14, `torch` 2.9.x, `transformers` 4.57.x at time of capture). Readers attempting a bit-exact replication will need to start from the pod versions listed above rather than from the dev freeze; readers attempting a best-effort replication at their own institution can use the dev freeze as a working starting point. Either way, see `sidecar/notes/n134_reproducibility_check.md` for per-claim reproduction status once that check is performed.

**Phase 5 method implementations.** The KnOTS, TSV, and SVC scores in §6 are **adaptations** of each method's core measurement quantity to the pairwise-triage setting, not re-uses of the published reference implementations. We did not use the published KnOTS / TSV / SVC repositories as dependencies and we do not claim operational equivalence to the published pipelines. Each adaptation lives inline in `scripts/n134/08_compare_methods.py` — specifically `knots_score_from_v21`, `tsv_score_from_v21`, and `svc_score_from_v21`, each with a docstring that documents both the faithfulness (what quantity is being measured) and the departure (what portfolio-scale pipeline is *not* being executed). The paper citations are in the bibliography (`papers/n134_workshop/references.bib`); pinned commits for the published KnOTS / TSV / SVC repositories are not applicable because those repositories were not used as dependencies.

**Software environment.** A `pip freeze` snapshot of the pod environment was not captured before the pod was decommissioned. A best-effort reconstruction is provided at `sidecar/results/n134/environment_dev.txt` — this is the development-environment snapshot captured post-hoc, with a header comment stating explicitly what it is (same major versions of `torch`, `transformers`, `peft`) and is not (not the pod environment; not a bit-for-bit replication guarantee). It is provided as a replication starting point, not as a reproducibility claim. See `sidecar/notes/n134_reproducibility_check.md` for per-claim reproduction status.

---

## Appendix E — Commit Hashes

Key commits in the N134 execution. Note that phase artifact commits are batched: Phase 1 / 2 / 3a / 3b / 4 outputs from the pod all landed in a single repo-side commit (`0a827e2`) when data came back from the pod on 2026-04-20, because the pod's `/workspace/n134/` was outside the repo during execution and was copied to `sidecar/results/n134/` in one rsync+tar pass after Phase 4.

| Commit | Phase | Contents |
|---|---|---|
| `643c192` | pre-experiment | `sidecar/notes/n134_spec.md` v3.1 and all 9 experiment scripts |
| `a4032e9` | Phase 0 | `transformers` `torch_dtype=` fix |
| `63c3f76` | Phase 0 | meta-tensor OOM fix (drop `device_map="auto"`) |
| `cb70fb8` | Phase 0 | `trust_remote_code=True` for piqa/siqa/boolq loaders |
| `2961b7a` | Phase 0 gate | retry ladder + real dip test |
| `6cbd9af` | Phase 1 | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` + bash restart loop |
| `4689569` | Phase 1 | logprob-based eval scoring (replaces broken greedy checker) |
| `b9b5b8a` | Phase 2 | QR-based fast rank-r SVD (replaces broken full SVD) |
| `7a0fae0` | Phase 2 | memory-safe resume + `.npz` binary sidecars |
| `e78c055` | Phase 2 | skip v2.1 JSON by default (MooseFS quota workaround) |
| `69e797a` | Phase 3b | `05_merge_eval` path fixes + flag 08 plumbing gap |
| `0a827e2` | Phase 1–4 | all pod artifacts (adapters' summaries, pair alignments, merge eval, H1 analysis) |
| `4677dde` | Phase 5 | `LazyPairV21` loader + schema fixes in `08_compare_methods.py` |
| `1527d72` | Phase 5 | four-method comparison results + figures |
| `3f6a7c7` | Phase 4b | H1 and secondary figures (previously blocked by `figures/` ignore rule) |

Data and code together constitute a reproducible trace of the experiment modulo the pod-environment gap documented in Appendix D.

---

*Report prepared by John T. Nanney. Pre-registration v3.1 in `sidecar/notes/n134_spec.md`. N134 is the confirmatory follow-up to N133 and completes the decoder-scale arc of the Gradience spectral-triage program. The epistemology held; the hypothesis did not. Both outcomes are findings.*
