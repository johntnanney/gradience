# Reproducibility Check — Tiered Verification Convention

**Status:** convention, first version (2026-04-21).
**Origin:** the N134 consolidation pass's T6 reproducibility check, documented in `sidecar/notes/n134_reproducibility_check.md`. The four-tier structure and the rank-on-residuals exception were developed while the pod that ran N134 was already decommissioned, which forced an honest reckoning with what a reproducibility check can and cannot claim when the production environment is not recoverable. This document generalizes that reckoning.

**Scope:** reproducibility checks for sidecar studies where the analysis scripts are committed to the repo but the production environment (hardware, OS, exact Python/library versions) is not guaranteed to match the reproduction environment. This is most sidecar studies. The convention is a reusable *structure*, not a rigid protocol; future consolidations should adapt the tiers' granularity and tolerances to their own statistics.

---

## 1 — Why tiered verification, not binary pass/fail

A reproducibility check is a claim about what a future replicator can expect to see. The simplest framing — "the analysis reproduces" or "it doesn't" — is adequate when the reproduction environment is a controlled copy of production (same Dockerfile, same locked dependency set, same hardware architecture). In that setting, "reproduces" means "bit-identical outputs modulo known float-precision noise," and the claim has a single failure mode.

When the reproduction environment is *not* a controlled copy of production, which is the common case for sidecar studies — pods are ephemeral, dev environments drift, Python and numpy majors change — binary pass/fail collapses two distinct questions into one:

- Did the *analytical logic* reproduce? (i.e. does the committed code, given the committed data, produce outputs whose qualitative content and structural shape match what the report claims?)
- Did the *numerical output* reproduce? (i.e. do the specific scalar values agree to some tolerance?)

These have different failure modes and different fixes. Analytical-logic failure means the code is wrong or the report misquotes it, and the fix is code or report correction. Numerical-output failure can mean several things: a library version changed an algorithm's default, a BLAS implementation handles ill-conditioning differently, a rank-based statistic is sensitive to residual precision at the N of interest. Some of these require code changes; some require only documentation; some reveal that the reported value has inherent numerical precision that should be stated explicitly in the paper.

Binary pass/fail forces all of these into one channel. A tiered check separates them. If tier 1 passes and tier 3 shows a localized drift, the check reports "analytical logic reproduces; one quantity exhibits precision-limited drift of magnitude X" rather than just "reproduction failed." The former is useful; the latter is misleading.

The N134 experience made the distinction load-bearing: tier 1 claims reproduced completely, tier 2 schema matched exactly, and tier 3 surfaced one quantity (partial Spearman on 45 OLS residuals) that drifted by 1.18e-2 — outside the standard ±0.01 correlation tolerance, inside a relaxed ±0.02 tolerance for the specific class of rank-based-on-residuals statistics that have intrinsic precision at small N. A binary protocol would have had to choose between calling this a failure (which would be wrong — the finding is robust) and relaxing the tolerance to accommodate it (which would feel like moving the goalposts post-hoc). The tiered protocol made the principled distinction: the statistic's precision is a property of the statistic on this data, not a bug, and the tolerance was calibrated to that property in advance during the pre-T6 scoping discussion.

**The generalization beyond N134:** any reproducibility check that might encounter version-sensitive statistics, data-availability limitations, or localized numerical drift benefits from the same structural separation. The cost of the tiered framing is minimal (a half-page of protocol specification); the benefit is that the check can report what it found rather than passing or failing.

---

## 2 — The four-tier template

Each tier answers a different question and has a different success criterion. Tier order is deliberate: lower-numbered tiers are more important and should be verified first, because their failure has larger implications.

### Tier 1 — Qualitative claims

**Question:** do the paper's stated qualitative findings reproduce?

**Criterion:** for each pre-registered or headline qualitative claim, produce the corresponding reproduced-environment output and verify the claim holds. Examples:

- "H1 not confirmed under the pre-registered rule" → re-run the decision logic, check the resulting `h1_confirmed` flag.
- "Method X is the only right-signed correlator" → re-run the comparison, check which methods produce positive correlations.
- "Three of four replications pass" → re-run the replication logic, count how many pass.

Qualitative claims should be formulated in a way that's robust to small numerical drift. "Partial ρ is large in magnitude but wrong sign" is a tier-1 claim; "partial ρ = −0.533" is not. If the paper's headline findings cannot be re-stated as qualitative claims, they are probably overclaiming the precision of the underlying numerics.

**Failure mode:** a qualitative claim from the paper does not reproduce. This indicates either a code bug, a report error, or a genuine environment-dependent finding. All three require investigation before the paper can stand.

**Sidecar practice:** formulate the tier-1 claim list from the paper's abstract and conclusion — these are the claims that most need to survive reproduction. If the paper claims something in a subsection but the abstract doesn't mention it, it's usually a tier-3 quantitative claim rather than a tier-1 qualitative claim.

### Tier 2 — Structural agreement

**Question:** does the reproduced analysis produce outputs with the same shape as the committed outputs?

**Criterion:** JSON schemas match, file paths align, required keys are present, array lengths are equal. Access-pattern verification: no `KeyError` on any field referenced in downstream consumption (paper figures, follow-on analyses, other scripts that read the output).

**Failure mode:** a structural mismatch indicates code drift between when the committed outputs were produced and when the reproduction attempts them. Either the committed code has been edited without regenerating outputs, or the committed outputs were produced from a different version of the code than what's currently in the repo.

**Sidecar practice:** tier 2 is usually fast and almost always passes. When it fails, the fix is straightforward (regenerate outputs from current code, or revert code to match outputs). When it *surprises* — passing when you thought it would fail or failing when you thought it would pass — that's diagnostic information worth investigating.

### Tier 3 — Quantitative agreement under a tolerance schedule

**Question:** do the specific scalar values in the reproduced outputs match the committed values to within the appropriate tolerance for each quantity class?

**Criterion:** scalar-by-scalar comparison, with tolerance chosen per quantity class rather than per quantity. A default tolerance schedule suitable for most sidecar studies:

| Quantity class | Default tolerance |
|---|---|
| Raw correlations (Pearson, Spearman, Kendall) on unresidualized data | ±0.01 absolute |
| Percentage-scale quantities (accuracy, degradation, ratio of means) | ±0.1 percentage point |
| R² and derived (ΔR², η², partial R²) | ±0.005 absolute |
| p-values of order 10⁻² or larger | ±0.01 absolute |
| p-values of smaller order | proportional; two-orders-of-magnitude window |
| Rank-based statistics on residuals or other derived values at small N | **±0.02 absolute** (see callout) |

> **Example callout (from N134 T6).** Rank-based statistics (Spearman, Kendall) computed on OLS residuals or other numerically-derived values are sensitive to floating-point paths in ways that aggregate quantities (sum-of-squares, R², mean, variance) are not. Small perturbations in individual residuals too small to shift the sum-of-squares past its 15th decimal place can still flip the rank order of near-tied observations, and rank-based statistics depend on ordering, not magnitude. The sensitivity is a property of the statistic on the data at the given N, not a bug in any library. Tolerance for this class of statistic should be calibrated to observation count rather than to library precision: at n ≈ 45, the statistic's intrinsic precision is roughly ±0.01; at n ≈ 200, closer to ±0.002; at n ≈ 1000, effectively sub-noise.

**Failure mode:** a scalar exceeds its tolerance. Three sub-cases, each with different implications:

1. *Tolerance just missed, class-appropriate statistic.* Expected drift for the statistic class; document and widen the tolerance for that class. The N134 partial ρ at 1.18e-2 drift is this sub-case.
2. *Tolerance significantly missed, unexpected class.* A quantity that should be numerically stable is not. This indicates a library default change, an algorithm change, or genuine environment sensitivity; localize the cause before deciding whether to accept or escalate.
3. *Tolerance missed across many quantities simultaneously.* Something deeper is wrong — wrong data, wrong code version, algorithm semantics changed. Escalate to tier-4 investigation; the tier-3 numbers are symptoms of a larger issue.

**Sidecar practice:** the tolerance schedule should be fixed before reproduction begins, not after. If reproduction reveals that a specific quantity class needs a wider tolerance than the default, name the class (not just the value) and state the reason — "rank-correlation-on-residuals needs ±0.02 because the statistic has intrinsic floating-point precision at small N" is principled; "partial ρ needs ±0.02 because we got −0.5448 and committed was −0.5330" is post-hoc tolerance-widening.

### Tier 4 — Gap documentation

**Question:** what is the reproduction environment, what gaps exist relative to production, and what can a future replicator infer from the check's results?

Tier 4 is not a pass/fail tier; it is a documentation tier. The failure mode is not "tier 4 fails" but "tier 4 is absent," which makes the check uninformative to future replicators.

There are (at least) **two distinct gap classes** that tier 4 should document separately because they have different implications.

**Environment gap.** The reproduction environment differs from the production environment in version or configuration. Document: Python version on both ends, major/minor versions of load-bearing libraries, BLAS implementation if relevant, hardware architecture if relevant. Name the specific gap, not just its existence. A future replicator needs to know whether to expect bit-identical reproduction (small gap) or precision-sensitive reproduction (large gap), and what verification protocol is appropriate for their own environment.

**Data-availability gap.** Some of the data required for reproduction is not in the committed repository. Document: which data, where it lived in production, whether and how a future replicator could regenerate or obtain it, and what conclusions of the original analysis cannot be re-verified from the committed state alone. This is a fundamentally different concern from environment drift: it's about what *can* be reproduced, not how closely it *does* reproduce.

N134 hit both. The environment gap was the Python 3.11 → 3.14 and numpy 1.26 → 2.4 version bump; the data-availability gap was the Phase 5 per-adapter `.npz` factors that were pod-only and never committed. The first required the tolerance-schedule calibration that produced the rank-on-residuals exception; the second required explicit acknowledgment that Phase 5's quantitative claims cannot be independently re-verified from the tagged repo alone. These are different kinds of limitations and were documented separately in the check.

**Sidecar practice:** most consolidations will hit only one gap type (environment gaps are the default; data-availability gaps happen when production artifacts are too large to commit). The convention should prepare for either. When a study hits both, name them separately; do not collapse them into a single "known limitations" paragraph.

---

## 3 — Escalation versus documentation per tier

A tiered protocol's value depends on what the tiers *do* when they fail. The following decision tree was used in N134 and is proposed as the default.

### Tier 1 failure → escalation

A qualitative claim from the paper does not reproduce. This is always an escalation, never documentation-only. The paper's core findings are at stake. Options:

1. Investigate whether the code has drifted between the committed outputs and the committed scripts. Regenerate outputs under the current scripts; compare to committed. If the regeneration produces the original qualitative claim, the issue is code-output drift and should be fixed by regenerating and re-tagging.
2. Investigate whether the paper mis-quotes the analysis. Compare the paper's claim language to the analysis output directly. Correct the paper.
3. If neither applies — if the code is current, the outputs are current, and the paper correctly summarizes them, but the reproduction environment genuinely produces a different qualitative result — this is a finding about environment-dependence and should be escalated to a scoping discussion with the project lead before any further action.

Tier 1 failures should not be accommodated by tolerance adjustment. If the claim is sensitive to reproduction environment, the claim is overclaiming its robustness.

### Tier 2 failure → escalation, but usually cheap

A structural mismatch almost always means code drift between the committed state and the committed outputs. The fix is usually straightforward: regenerate, retag, commit. The cost of ignoring it is high — downstream consumers (figures, other scripts) will break on the mismatched schema — so fix rather than document.

One exception: if the structural change is additive (e.g., a new optional field was added to the output schema between tag time and check time), document rather than fix. Additive changes don't break downstream consumers.

### Tier 3 failure → depends on sub-case

See the three sub-cases in the tier-3 section. Summary:

- Tolerance-just-missed in an expected-drift class → document and calibrate the tolerance for that class. No retag.
- Tolerance-significantly-missed in an unexpected class → localize the cause. If the cause is localized and explicable, document (usually as an environment-sensitivity note in the paper). If the cause is not localizable, escalate to tier-4 environment investigation.
- Tolerance-missed-across-many-quantities → this is almost always a data or code correctness issue masquerading as a numerical-precision issue. Escalate.

### Tier 4 failure → documentation only, by definition

Tier 4 does not fail in the pass/fail sense. The failure mode is absence of documentation, not a failed verification. If the gap is named and characterized honestly, tier 4 is complete.

### Cross-tier: the "worth-escalating-to-rebuilt-environment" threshold

One escalation path deserves explicit mention because it is expensive. When tier 3 reveals a drift that cannot be explained by a known library-version sensitivity or a small-N statistical property, the next-best verification is to rebuild the production environment in a container and verify that the committed numbers reproduce there. This is expensive (pod rebuild, Dockerfile construction, possibly re-auditing data) and should not be the default.

The N134 case explicitly declined this escalation because the tier-3 drift was in a single quantity with a clean diagnostic explanation (rank-on-residuals precision sensitivity), and a pinned-environment rebuild would have verified that the committed number reproduces in its original environment — which we already knew, because that environment produced the committed number. The rebuild would not have changed the finding that the statistic has inherent precision at this N.

The decision threshold is: rebuild the environment only when the tier-3 drift is *unexplained* and affects *qualitative claims*. If the drift is explained and affects only point-estimate precision, document rather than rebuild. If the drift is unexplained but affects only point-estimate precision, localize the cause before escalating. If the drift affects qualitative claims, escalate regardless.

---

## Adaptation

The four-tier template is a starting structure, not a fixed protocol. Future sidecar consolidations may find that their statistics need different tolerance classes, that their data-availability gaps need sub-tiers, or that their qualitative claims benefit from being structured as a formal checklist with each item verified independently. All of these are welcome adaptations. What the convention commits to is the *separation of concerns* — qualitative versus structural versus quantitative versus gap-documentation — and the principle that tolerances should be class-based rather than per-value, calibrated in advance rather than post-hoc.

If a consolidation substantially revises the tier structure, log the revision as an updated version of this document. The N134 experience is this document's worked example; future experiences should expand it rather than replace it.

---

*Derived from the N134 T6 reproducibility check (2026-04-21). The concrete numerical diagnostic (rank-on-residuals sensitivity at n = 45, partial-ρ drift of 1.18e-2 across Python 3.11 → 3.14 and numpy 1.26 → 2.4) is in `sidecar/notes/n134_reproducibility_check.md` §Rank-on-residuals observation. This convention generalizes that experience for reuse in future consolidation passes.*
