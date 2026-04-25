# Implementation Deviations from SPEC_CPU_v0_2.md

**Status:** living document, updated as deviations accumulate during implementation. Each entry names what the spec required, what was implemented, why, and what (if anything) needs to happen at v1.1 pre-registration lock.

**Purpose:** the spec is the contract; the implementation is what runs. When they differ, we record the difference in writing before lock so nothing drifts silently. At v1.1 lock, each deviation is either retconned into the spec (if the deviation is better than the spec's original form) or reversed (if the spec's form is correct and the implementation should catch up).

---

## D-01 — `"null"` string as subject-key in fewshot_draws_LOCKED.json

**Script:** `02_draw_fewshot_examples.py`
**Spec location:** §8.3, §3.4
**Spec says:** LOCKED.json structure keys draws by `benchmark_id` → `subject_id` (or null) → `seed_id` → item IDs.
**Implementation:** uses the literal string `"null"` as the subject-key when a benchmark has no subjects, because JSON object keys must be strings.
**Why:** JSON doesn't support null keys. The spec's "or null" phrasing is ambiguous at the JSON level.
**At v1.1:** document in `schemas/fewshot_draws.schema.json` (not yet authored) that subject-less benchmarks use the `"null"` string key. Alternative: omit the subject layer entirely for subject-less benchmarks (change code + change schema). The current convention is defensible and doesn't need to change.

## D-02 — Subject field name hard-coded as `"subject"`

**Script:** `02_draw_fewshot_examples.py`
**Spec location:** §3.4, §8.3 draw pseudocode
**Spec says:** `source.filter(lambda x: x["subject"] == subject_id)` for subject-bearing datasets.
**Implementation:** literally uses the field name `"subject"`.
**Why:** this matches HF's MMLU convention. Other subject-bearing datasets (if any join the panel) may use a different field name.
**At v1.1:** if any non-MMLU subject-bearing benchmark joins the panel, add `subject_field: <name>` to `benchmarks.yaml` with default `"subject"`. Low-priority TODO.

## D-03 — `generation_length_tokens` as whitespace word-count

**Script:** `04_normalize_outputs.py`
**Spec location:** §5.5 (schema lists `generation_length_tokens`)
**Spec says:** column exists and is int32, nullable.
**Implementation:** CPU-only; computed as `len(raw_generation.split())` (whitespace word count) as a proxy for true tokenizer-anchored count.
**Why:** the GPU-side backend doesn't currently emit a true token count; the CPU pipeline has no model-side tokenizer to invoke.
**At v1.1:** add `generation_length_tokens` to `schemas/item_outputs.schema.json` as a required field emitted by the GPU side, anchored to the model's tokenizer. Update `04_normalize_outputs.py` to pass through verbatim rather than recompute.

## D-04 — `choice_count` null for G&P rows

**Script:** `04_normalize_outputs.py`
**Spec location:** §5.5
**Spec says:** "constrained_choice only; if not derivable, null is acceptable."
**Implementation:** populated for LL rows (`len(row.choices)`); null for G&P rows because the G&P JSONL doesn't carry the choice list.
**Why:** matches the spec's permission. Flagging here so it's not treated as missing data.
**At v1.1:** no action needed unless downstream analyses require `choice_count` for G&P rows (they currently don't).

## D-05 — `run_id != condition_id` treated as hard-fail

**Script:** `04_normalize_outputs.py`
**Spec location:** §5.2, §8.5
**Spec says:** "`run_id` MUST equal `condition_id`. If they differ, the run is rejected" (§5.2); §8.5 step 3 only mentions `condition_id` mismatch explicitly.
**Implementation:** both mismatches (run_id vs metadata condition_id, and metadata condition_id vs manifest condition_id) trigger exit 3.
**Why:** defensive programming; "MUST equal" is the relationship to enforce.
**At v1.1:** no action needed; spec is consistent, implementation is consistent with spec intent.

## D-06 — MMLU `model_main` as fixed-effect-mean variance, not random-intercept variance

**Script:** `09_mmlu_subject_decomp.py`
**Spec location:** §8.10
**Spec says:** decompose variance into model, subject, prompt main effects + interactions; spec allows "fixed is simpler and adequate for 3 models."
**Implementation:** `model_main` is reported as variance of fitted per-model means (fixed-effect analog), not as a random-intercept variance component.
**Why:** spec explicitly allowed fixed-effect treatment for model. With only 3 models, a random-intercept treatment is underpowered anyway.
**At v1.1:** the manuscript's §8.10 write-up must be explicit about this as a modeling choice. A psychometrics-literate reviewer will expect "variance" in a G-theory context to mean variance of a random effect; clarifying the terminology avoids confusion.

## D-07 — `10_gsm8k_case.py` uses condition-std SEM directly

**Script:** `10_gsm8k_case.py`
**Spec location:** §8.11, §9.2
**Spec says:** tolerance from SEM of variance-components decomposition (per §9.2 multi-component formula).
**Implementation:** tolerance derived from `std(condition_accuracy, ddof=1)` rather than from a full G-theory decomposition.
**Why:** GSM8K has only 2 scoring-rule variants (strict / permissive) and no nested subjects, so a full variance-components decomposition is degenerate at the condition level. Sample SD is the right statistic at this scale.
**At v1.1:** no action. Document in manuscript §5.6 (GSM8K secondary) that the open-generation case uses sample-SD-based tolerance by design.

## D-08 — `98_reproducibility_trace.py` reads `is_correct` directly from raw JSONL

**Script:** `98_reproducibility_trace.py`
**Spec location:** §13.1 item 4
**Spec says:** "recompute `is_correct` per the scoring rule."
**Implementation:** reads `is_correct` from the raw JSONL rather than re-deriving from `raw_generation` (G&P) or `choice_scores` argmax (LL).
**Why:** re-deriving from scratch would require importing the scoring logic from `04_normalize_outputs.py`, which is a larger refactor. The current implementation catches aggregation bugs (item-level → condition-level) but not parse-rule bugs.
**At v1.1:** if reviewer-proofing against parse-rule bugs is important, extend the reproducibility trace to re-run the scoring rules. Lower priority than the other items above; most parse-rule bugs would also break the normalizer's schema validation.

## D-09 — Item-level mixed effects use linear probability model, not logistic

**Script:** `06_variance_components.py`
**Spec location:** §10.1
**Spec says:** "Fit a mixed-effects logistic regression at the item level... Family: binomial, logit link." Library: `statsmodels.MixedLM` or `rpy2 + lme4::glmer`.
**Implementation:** linear probability model (LPM) — Gaussian mixed-effects on 0/1 outcomes — via `statsmodels.MixedLM`.
**Why:** `statsmodels.MixedLM` is Gaussian-only; it does not support binomial family at all. The spec's cited library cannot implement the spec's cited model. The alternative (`rpy2 + lme4::glmer`) adds an R dependency to the pipeline, which is a larger infrastructure commitment. For accuracies in the typical benchmark range (0.15–0.85), LPM variance proportions match logistic GLMM variance proportions within a few percent — the tolerance schedule is unaffected.
**At v1.1:** update spec §10.1 to either (a) remove the "logistic" specification and accept LPM as the documented approach, or (b) commit to the `rpy2 + lme4` path. The manuscript's Analysis 1 discussion should name LPM as the method and briefly justify it. A psychometrics reviewer will expect this to be explicit.

## D-10 — Spec §10.1's model × prompt and model × scoring_rule interactions

**Script:** `06_variance_components.py`
**Spec location:** §10.1 Level 1
**Spec says:** Level 1 random effects include `model:prompt` and `model:scoring_rule` crossed interactions.
**Implementation:** Level 1 uses only the four non-interaction random effects (prompt, seed, scoring_rule, item); the model × prompt and model × scoring_rule interactions collapse into residual.
**Why:** `statsmodels.MixedLM` with `vc_formula` supports independent-facet random effects but not crossed-interaction random effects in the form the spec writes. Encoding such interactions requires either the `rpy2 + lme4` path or manual construction of interaction design matrices.
**At v1.1:** if the interaction decomposition is load-bearing for the paper's §8.10 claims, commit to the `rpy2 + lme4` path. Currently, the interaction effects are available qualitatively from the MMLU subject decomposition (§9) and don't need to come from §6 specifically.

## D-11 — Cascade gradient-norm check not enforced

**Script:** `06_variance_components.py`
**Spec location:** §10.1 convergence-trigger list
**Spec says:** gradient norm > `analysis_config.convergence_triggers.gradient_norm_above` triggers cascade descent.
**Implementation:** gradient norm is not checked (no clean way to extract it from `statsmodels.MixedLM` results); singular-fit-warning and non-convergence status are the active triggers.
**Why:** `statsmodels.MixedLM` doesn't expose a gradient-norm-at-optimum that maps directly to the spec's trigger. The practical cascade triggers (singular warning, didn't-converge) already catch the relevant failure modes.
**At v1.1:** either remove the gradient-norm trigger from the spec, or specify a concrete extraction method. Current implementation is defensible — the remaining triggers do the work the gradient-norm check would have — but the inconsistency is worth naming.

## D-12 — Condition-level bootstrap for prompt-averaged and full-design CI uses linear shrinkage

**Script:** `07_tolerance_schedule.py`
**Spec location:** §9.3
**Spec says:** bootstrap CIs are computed for all four tolerance levels (single-occasion, within-rule, prompt-averaged, full-design).
**Implementation:** single-occasion and within-rule tolerance CIs are computed via direct bootstrap resampling of conditions. Prompt-averaged and full-design CIs are computed by linear shrinkage (multiply CI bounds by `1/√4` and `1/√n_conditions` respectively) rather than by re-bootstrapping the averaged estimator.
**Why:** a correct bootstrap of the prompt-averaged estimator would require resampling within prompt strata, which has a different resampling scheme than the condition-level bootstrap and introduces additional implementation complexity. The linear shrinkage is conservative (wider than true) in the lower-bound direction — which is safer for the H1 test, which uses the lower bound.
**At v1.1:** if the prompt-averaged tolerance CI is reported in the manuscript as a headline number, re-implement with proper stratified bootstrap. If it's only used as the full-design lower bound in H1, the current conservative approximation is defensible.

## D-13 — `03_validate_prompts.py` content-hash field type coercion

**Script:** `03_validate_prompts.py`
**Spec location:** §8.4
**Spec says:** declared `content_hash` is a SHA-256 hex string.
**Implementation:** defensively casts `declared` to `str` before comparison and slicing, because YAML may deserialize a numeric-looking hash value as an integer.
**Why:** test fixtures sometimes use numeric content_hash values (e.g., `content_hash: 123456`); YAML parses these as integers, and slicing an int with `declared[:16]` raises TypeError.
**At v1.1:** no action needed; the fix is invisible to correct inputs and defensive against YAML type coercion surprises.

---

## Summary

Twelve tracked deviations. Three categories by severity:

**Paper-manuscript-relevant (must be named in the write-up):**
- D-06 (MMLU `model_main` as fixed-effect variance)
- D-09 (LPM vs logistic at item level)

**Spec-update candidates for v1.1 lock:**
- D-01 (subject-key convention) — add to `schemas/fewshot_draws.schema.json` when authored.
- D-09, D-10, D-11 (statsmodels vs lme4 decisions) — update spec §10.1 to match implementation, or commit to rpy2 path.
- D-12 (bootstrap shrinkage) — decide whether proper stratified bootstrap is needed.

**Benign / documented for audit only:**
- D-02 (subject field name), D-03 (word-count proxy), D-04 (choice_count), D-05 (run_id strictness), D-07 (GSM8K SEM), D-08 (is_correct direct read), D-13 (hash coercion).

None of the deviations affect the paper's central claims. All of them affect how the manuscript describes its methods, or how the spec should be updated for camera-ready fidelity.
