# Engine Notes — Study 16 Integration

**Date:** 2026-03-08
**Scope:** `gradience.vnext.merge` — recommendation engine, eligibility screening, reporting, CLI

---

## What changed

### Architecture

The merge recommendation engine was split into two explicit stages:

- **Stage A — Diagnosis** (`diagnose_layer`, `diagnose_pair`): Extracts structural facts from spectral metrics. No policy decisions. Outputs `LayerDiagnosis` and `PairDiagnosis`. Reusable and testable independently.
- **Stage B — Policy** (`_apply_layer_policy`, `_apply_policy`): Translates diagnosis into strategy choices and coefficients. This is where eligibility context modulates decisions.

Previously these were tangled in a single `_recommend_layer` function.

### Eligibility screening

New module `eligibility.py` with:

- `EligibilityStatus` enum: `ELIGIBLE`, `UNCERTAIN`, `FLAGGED_WEAK`, `UNKNOWN`
- `AdapterQAResult`: Lightweight QA summary (status + metric comparison + evidence)
- `classify_eligibility()`: Heuristic classifier from adapter-vs-base metric comparison
- `screen_adapters()`: Generates merge-audit warnings from QA results

This is the hook where behavioral gating lives. The heuristics are intentionally simple. The architecture matters more than the decision logic right now.

### Typed containers

Replaced `Dict[str, Any]` fields in `MergeAuditReport` with frozen dataclasses:

- `AdapterMetadata`, `MatchingSummary`, `AggregateResult` — structural data
- `WarningCode` enum + `MergeWarning` — machine-readable warnings
- `PairAuditResult`, `RecommendationResult` — clean public-facing result types

All containers have dict-compat layer (`__getitem__`, `.get()`, `__contains__`, `__iter__`) so existing code like `agg["n_safe"]` still works during transition.

### Reporting

QA report output restructured into four explicit sections:

1. **Structural Result** — spectral compatibility verdict, score, layer distribution
2. **Behavioral Status** — source adapter eligibility, confidence note
3. **Eligibility Warning** — data gaps, weak adapters, structural risk
4. **Recommended Action** — concrete action and strategy

Previously these were collapsed into one summary that mixed structural diagnosis with behavioral assessment.

### CLI flags

Added to `merge-audit`:

- `--source-a-qa` / `--source-b-qa`: Ingest prior QA JSON files (AdapterQAResult format)
- `--strategy`: Highlight a user-specified merge strategy alongside the auto-recommendation
- `--emit-report`: Write structured JSON to a specific path
- `--strict-qa`: Gate that refuses recommendations when QA data is missing or shows weak adapters

### Norm-equalized merge

Promoted to first-class strategy:

- `norm_equalized_coefficients()` in recommend.py computes geometric-mean rescaling
- `plan_norm_equalized()` in plan.py generates full merge plans
- Always appears in fallback strategies
- CLI output includes ready-to-run norm-equalized command

---

## Why it changed

Study 16 showed that structural compatibility (spectral analysis) is necessary but not sufficient for merge quality. The core problem: a structurally fair merge can still produce behaviorally disappointing results when one or both source adapters are weak.

The previous system had no concept of "is this adapter worth preserving?" It would happily rebalance coefficients to give equal weight to an adapter that underperforms the base model. This is structurally correct but practically useless.

---

## What Study 16 forced us to learn

1. **Structural diagnosis and behavioral assessment are different questions.** Subspace overlap tells you whether two adapters *can* merge cleanly. It does not tell you whether they *should*. These must be reported separately.

2. **Norm-equalized merge is a surprisingly strong baseline.** Rescaling both adapters to geometric-mean Frobenius norm before linear averaging removes scale imbalance as a confound. It often matches or beats more sophisticated per-layer strategies. It should always be offered as an alternative.

3. **Weak sources must never be silently preserved.** When source QA flags an adapter as FLAGGED_WEAK, every code path that touches that adapter must emit a user-visible warning. The system should never quietly rebalance coefficients to preserve a signal that isn't worth preserving.

4. **Missing QA data is itself a finding.** When no behavioral evaluation is available, the system must explicitly say "this recommendation is structural only and cannot predict downstream task performance." Silence implies confidence.

5. **The recommendation is not one thing.** "Merge with caution" conflates structural risk, behavioral status, eligibility concerns, and the actual recommended action. Users need to see each of these separately to make informed decisions.

---

## What is now validated

Test coverage for the policy layer (396 merge tests total, 23 specifically encoding Study 16 conclusions):

- **Norm-equalized coefficient correctness**: Geometric-mean rescaling produces expected coefficients for equal norms, imbalanced norms (1:100, 1:1000), custom weights, and zero-norm edge cases.
- **Weak-source warning (Pair 06 scenario)**: Single weak adapter produces "weak adapter" warning. Both weak produces "both underperform" warning and "reconsider" recommendation. QA report caveats always mention weak adapters.
- **Balanced pair no-op**: All-safe layers with both eligible adapters produce low risk, linear (0.5, 0.5), no warnings.
- **Missing QA structural-only disclaimer**: No source_qa and empty source_qa both produce "structural balance only" warning. QA report confidence note mentions lack of behavioral data.
- **Flagged-weak never silent invariant**: Tested across `recommend_merge`, `_eligibility_warnings`, `build_qa_report`, and `PairAuditResult.from_audit_report`. Every path that sees FLAGGED_WEAK emits at least one user-visible warning.

---

## What is still unvalidated

1. **End-to-end CLI flag testing.** The new `--source-a-qa`, `--source-b-qa`, `--strict-qa`, `--emit-report`, and `--strategy` flags are wired up but not tested via subprocess CLI invocation. They're tested indirectly through the functions they call, but no integration test runs `gradience merge-audit --source-a-qa qa.json ...` against real adapter directories.

2. **AdapterQAResult JSON schema stability.** The `from_dict` / `to_dict` roundtrip works, but there's no formal schema version for QA JSON files. If the format changes, old QA files loaded via `--source-a-qa` could silently produce wrong results.

3. **Eligibility classification thresholds.** `classify_eligibility()` uses a simple delta-vs-margin heuristic. No study has validated what margin values are appropriate for different metrics (perplexity vs accuracy vs F1). The current default margin is 0.0, which means any adapter that beats base by any amount is ELIGIBLE.

4. **Policy modulation by eligibility.** The Stage B policy currently does NOT change per-layer strategy based on eligibility context. The `eligibility` parameter is passed to `_apply_layer_policy` but unused — it's a hook for future work. For example: should a FLAGGED_WEAK adapter get smaller coefficients? Should conflicting layers with a weak source be dropped entirely? These policy questions are architecturally supported but not implemented.

5. **Multi-adapter merging.** Everything assumes exactly two adapters. The containers, diagnosis, and policy all take pairs. N-way merge would require rethinking the pairwise comparison model.

6. **Norm-equalized vs audit-aware empirical comparison.** We claim norm-equalized is "often competitive." This is based on Study 16 observations but not validated by automated benchmarks in this codebase. The bench suite doesn't yet include merge quality evaluation.

7. **`--strict-qa` UX.** The flag exits with error code 1 when QA is missing or weak. This is correct for CI pipelines but potentially confusing for interactive use. No `--strict-qa` + `--force` escape hatch exists.
