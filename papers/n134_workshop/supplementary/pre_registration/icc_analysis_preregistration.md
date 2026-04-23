<!-- ANON: title renamed from "N134 Cross-Seed ICC Specification" to drop project number. -->
# Cross-Seed ICC Analysis — Pre-Registration Supplement

**Status:** spec, pending implementation.
<!-- ANON: "Purpose" line rewritten: working-repo path ("papers/n134_workshop/draft_v2_thesis_b.tex") replaced with descriptive paper-section reference. -->
**Purpose:** close the `\emph{[TODO...]}` in the Appendix-D reliability
section of the main manuscript.
<!-- ANON: companion-artefacts block rewritten: bundle-relative paths replace the working-repo "sidecar/results/n134/" and "scripts/n134/" prefixes per checklist §2.2. -->
**Companion artefacts at completion time:**
`raw_analytical_artifacts/analysis_icc.json`,
`analysis_scripts/compute_icc.py`, short Appendix-D paragraph and §2.2 /
§4.2 in-text insertions.

This spec is deliberately explicit about design choices. The paper's own
argument is that measurement-theoretic commitments an author makes in
computing a reliability coefficient should be reviewable from prose before
being encoded in code; a spec rather than a script is the natural place to
make them. The choices below each carry a short defense, so a reader who
disagrees can locate the choice and the defense, rather than
reverse-engineering both from the reported number.

---

## 1. What is being estimated

<!-- ANON: "scripts/n134/06_analysis_h1.py" path → bundle-relative "analysis_scripts/compute_s_h1.py". -->
Cross-seed intraclass correlation for $S_{\mathrm{H1}}$ as an instrument,
where "the instrument" is the `compute_s_h1(pair)` function in
`analysis_scripts/compute_s_h1.py` applied to a same-task adapter pair.
The estimate is an **instrument-validation** step rather than an
exploratory measurement: it quantifies how much the score for a same-task
pair would shift if one of the two adapters were resampled (i.e.,
retrained under a different seed), holding the task fixed.

The target value goes into:

- **Appendix D (`app:reliability`)**: closes the `[TODO: insert cross-seed
  ICC computation…]` block. Expands to a short paragraph reporting ICC, CI,
  SEM, and a one-line interpretation.
- **§2.2 (`How stable the indication is`)**: currently states the general
  principle. Add a clause naming the specific ICC and SEM, with a citation
  to the appendix.
- **§4.2 (`Reliability considerations at pre-registration time`)**:
  already forward-references `app:reliability`. Extend with the specific
  number once available.

---

## 2. Data inputs

<!-- ANON: "sidecar/results/n134/audit/" path → bundle-relative "raw_analytical_artifacts/". -->
- **Source file.** `raw_analytical_artifacts/pair_alignment_full.json`.
  Verified top-level schema: dictionary keyed by pair identifier
  `{task_a}_s{seed_a}_vs_{task_b}_s{seed_b}`, with per-pair per-layer
  alignment data required by `compute_s_h1`.
- **Filtering.** Retain the 24 pairs where `is_same_task == true` (the
  three within-task seed-pairs $\{(42,123), (42,456), (123,456)\}$ for
  each of the 8 tasks $\{$arc\_challenge, boolq, commonsenseqa,
  hellaswag, openbookqa, piqa, siqa, winogrande$\}$).
- **Panel.** $8 \times 3$: row = task, column = within-task seed-pair index
  (deterministic ordering: $(42,123)$, $(42,456)$, $(123,456)$; the
  within-task ordering is arbitrary for an ICC(2,1) estimate but should be
  fixed for reproducibility).
<!-- ANON: "the N134 pre-registration's" → "this study's pre-registration". -->
- **Completeness.** 24/24 by design (this study's pre-registration's
  three-seeds-per-task commitment guarantees no missing cells). The
  implementer should assert this explicitly rather than silently proceed on
  a partial panel.

---

## 3. Design choices

### 3.1 ICC form: ICC(2,1), absolute agreement, single measurement

- **Two-way random effects.** Both tasks and within-task seed-pairs are
  treated as random samples from larger populations. Tasks are a sample
  from the population of plausible NLU tasks one could have designed the
  study around; within-task seed-pairs are a sample from the population of
  seed-pair draws one could have used to estimate same-task alignment.
  Neither is a fixed set of "conditions" whose specific identity the
  reliability estimate should treat as structural.
- **Absolute agreement, not consistency.** We care whether two
  independently-drawn seed-pair realizations on the same task produce the
  *same numerical value* for $S_{\mathrm{H1}}$, not whether they
  rank-order the tasks identically. The paper's bounded-precision claims
  couple to absolute-agreement SEM; consistency ICC would answer a
  different (weaker) question.
<!-- ANON: "the N134 design happened to have available" → "this study's design happened to have available". -->
- **Single measurement, not average.** The paper's primary H1 test applies
  $S_{\mathrm{H1}}$ to a single seed-pair per cross-task pair. The
  relevant reliability is therefore of a single measurement, not of a
  three-pair average this study's design happened to have available for
  the same-task subset.

**Excluded: ICC(3,1)** (two-way mixed, consistency) — appropriate when the
raters are a fixed set whose biases we want to partition. The three
seed-pairs are not raters with stable biases; they are interchangeable
draws.

**Excluded: ICC(1,1)** (one-way random) — appropriate when there is no
replicate structure or no rater identity. We have three replicate draws
per task, so the two-way form is the more informative estimator.

### 3.2 Confidence intervals: Shrout–Fleiss F-distribution primary, block-bootstrap secondary

- **Shrout–Fleiss F-distribution CI** is the standard parametric CI for
  ICC(2,1), closed-form, valid under normality of the panel. Reported as
  primary.
- **Block-bootstrap over tasks** (resample the 8 tasks with replacement,
  recomputing ICC within each bootstrap draw) is reported as a secondary
  robustness check, because it answers a subtly different question: the
  sampling variability under the specific 8-task realization. 5000
  resamples, seed 134 for determinism.
- **Comparison rule.** If the two CIs' lower bounds agree to within 0.05,
  report the F-distribution CI in the appendix with a sentence noting
  bootstrap agreement. If they disagree by more than 0.05, report both
  explicitly and note the divergence — it is itself informative about
  whether the parametric assumptions hold at $N = 8$ tasks.

### 3.3 SEM (standard error of measurement) reported alongside

$\mathrm{SEM} = \mathrm{SD}_{\mathrm{pooled}} \cdot \sqrt{1 - \mathrm{ICC}}$,
where $\mathrm{SD}_{\mathrm{pooled}}$ is the SD of the 24 same-task
$S_{\mathrm{H1}}$ values. SEM is the measurement-theoretic quantity that
couples to §2.3's bounded-precision claims: it states how much the
observed $S_{\mathrm{H1}}$ on a given pair could differ from the "true"
same-task alignment that the instrument targets. Report to two significant
figures.

### 3.4 Instrument-output definition: raw `compute_s_h1(pair)`

<!-- ANON: "scripts/n134/06_analysis_h1.py" → "analysis_scripts/compute_s_h1.py". -->
No normalization, no standardization, no family-residualization (same-task
pairs have no family-identity confound structure — they are the
reliability design itself). Compute $S_{\mathrm{H1}}$ by calling
`compute_s_h1` from `analysis_scripts/compute_s_h1.py` directly; do not
reimplement the O-module depth-weighting in the ICC script. This
preserves the identity "instrument under reliability estimation = instrument
under H1 test."

---

## 4. Computation procedure

<!-- ANON: all working-repo paths in steps 1/3/7 → bundle-relative. -->
1. Load `raw_analytical_artifacts/pair_alignment_full.json`.
2. Filter to same-task pairs (`is_same_task == true`); assert the
   resulting count is exactly 24.
3. For each pair, compute $S_{\mathrm{H1}}$ by calling
   `compute_s_h1()` imported from `analysis_scripts/compute_s_h1.py`.
4. Assemble the $8 \times 3$ panel with deterministic row/column ordering.
5. Fit the two-way random-effects ANOVA model: between-tasks MS
   ($\mathrm{MS}_R$), between-seed-pairs MS ($\mathrm{MS}_C$), residual
   MS ($\mathrm{MS}_E$). Implementation: `pingouin.intraclass_corr` with
   `type='ICC2'` is the canonical Python tool; alternatively a direct
   Shrout–Fleiss implementation is trivial on a complete $n \times k$
   panel.
6. Compute ICC(2,1), F-distribution 95 % CI, and block-bootstrap 95 % CI
   (5000 resamples, seed 134). Compute SEM.
7. Emit `raw_analytical_artifacts/analysis_icc.json` per the schema in §5.
8. Print a one-paragraph prose summary for copy-paste into the paper.

---

## 5. Output schema

<!-- ANON: "instrument_source" field value → bundle-relative path. -->
```json
{
  "instrument": "S_H1",
  "instrument_source": "analysis_scripts/compute_s_h1.py:compute_s_h1",
  "design": {
    "icc_form": "ICC(2,1)",
    "agreement_type": "absolute",
    "measurement_type": "single",
    "n_tasks": 8,
    "n_seed_pairs_per_task": 3,
    "n_total_observations": 24,
    "task_ordering": [
      "arc_challenge", "boolq", "commonsenseqa", "hellaswag",
      "openbookqa", "piqa", "siqa", "winogrande"
    ],
    "seed_pair_ordering": [[42, 123], [42, 456], [123, 456]]
  },
  "results": {
    "icc": <float>,
    "ci_shrout_fleiss_95": [<lower>, <upper>],
    "ci_bootstrap_95": [<lower>, <upper>],
    "bootstrap_n_resamples": 5000,
    "bootstrap_seed": 134,
    "sem": <float>,
    "sd_pooled": <float>,
    "ms_between_tasks": <float>,
    "ms_between_seed_pairs": <float>,
    "ms_residual": <float>
  },
  "panel": {
    "<task_name>": {
      "seed_pair_42_123": <float>,
      "seed_pair_42_456": <float>,
      "seed_pair_123_456": <float>
    },
    ...
  },
  "sanity_checks": {
    "panel_complete": true,
    "icc_in_range": true,
    "sem_less_than_sd_pooled": true,
    "ci_agreement": "within_0.05" | "divergent"
  }
}
```

---

## 6. Sanity checks

- **Panel completeness.** 24/24 same-task cells; abort otherwise.
- **Range check.** $\mathrm{ICC} \in [0, 1]$. Negative ICC indicates
  computational error or degenerate between-task variance (all tasks
  producing indistinguishable $S_{\mathrm{H1}}$ distributions); investigate
  before reporting.
<!-- ANON: "Prior Gradience program experience with" → "Prior experience in this research program with". -->
- **Magnitude expectation.** Prior experience in this research program
  with spectral scores on seed-replicated adapters in the same task
  suggests ICC > 0.6 is the expected regime. $\mathrm{ICC} < 0.5$ is not a
  bug *per se* but is unexpected enough to warrant a second-look at the
  panel before it is paper-reported.
- **SEM sanity.** $\mathrm{SEM} < \mathrm{SD}_{\mathrm{pooled}}$ by
  construction (since ICC ∈ [0, 1]). Violation indicates a sign or MS
  swap in the formula.
- **MS relations.** $\mathrm{MS}_R, \mathrm{MS}_C, \mathrm{MS}_E$ all
  non-negative; $\mathrm{MS}_R \geq \mathrm{MS}_E$ when ICC > 0 (otherwise
  ICC is negative by the Shrout–Fleiss formula).

---

## 7. Failure modes and escalation

- If `pair_alignment_full.json` is missing fields `compute_s_h1` requires
  (layer-indexed O-module alignment components), halt and report rather
  than silently substituting a mean-alignment shortcut from
  `pair_alignment_summary.json`. Mean alignment is *not* $S_{\mathrm{H1}}$.
- If $\mathrm{MS}_E$ is effectively zero (perfect within-task reliability
  to floating-point), report $\mathrm{ICC} = 1.0$ with a note. A spectral
  score identically reproducing across three distinct seed-pair
  calibrations would be itself worth examining in the appendix rather
  than reported without comment.
- If the Shrout–Fleiss and bootstrap CIs disagree by more than 0.05 on
  the lower bound, report both in the appendix with a one-sentence
  acknowledgement that parametric assumptions may be borderline at
  $N = 8$ tasks.
<!-- ANON: internal revision-note identifier "RN-NNN" stripped per checklist §1 "paper-private working-session labels". -->
- If the implementer has a principled reason to depart from a design
  choice in §3, do not depart silently: document the alternative in a
  supplementary revision note and update this spec before proceeding.

---

## 8. Paper destinations — prose templates

**Appendix D expansion** (replacing the current `[TODO…]` placeholder):

> Cross-seed ICC(2,1) for $S_{\mathrm{H1}}$, estimated from the 24
> same-task adapter pairs (3 within-task seed-pairs × 8 tasks, absolute
> agreement, single measurement), is $\hat{\rho}_{\mathrm{ICC}} = [VAL]$
> (95 % CI $[[LOW], [HIGH]]$; Shrout–Fleiss F-distribution method, with
> block-bootstrap over tasks in [agreement / divergent] at [VAL]).
> The corresponding SEM is $[VAL]$, which is the standard error on a
> single-seed-pair $S_{\mathrm{H1}}$ estimate and is the decimal-place
> precision the score's reports actually support independent of sampling
> variability. [Interpretive line: ICC of [VAL] places the instrument at
> [excellent / good / moderate / poor] cross-seed reliability by the
> conventional descriptive thresholds $(>0.9, >0.75, >0.50, <0.50)$; we
> report the conventional description only as orientation and not as a
> binary gate, since the paper's framework argues against thresholding
> continuous reliability estimates into verdicts.]

**§2.2 insertion** (one clause in the paragraph's second sentence):

> Cross-seed intraclass correlation (ICC) is the natural form for a
> per-pair diagnostic with replicate structure; on our design this
> estimate is $\hat{\rho}_{\mathrm{ICC}} = [VAL]$ with SEM $= [VAL]$
> (Appendix~\ref{app:reliability}).

**§4.2 insertion** (one sentence at the end of the existing paragraph):

> The resulting estimate, $\hat{\rho}_{\mathrm{ICC}} = [VAL]$ with SEM
> $= [VAL]$, is reported at Appendix~\ref{app:reliability}.

---

## 9. Promotion to convention

<!-- ANON: §9 rewritten to strip project-brand ("Gradience"), follow-up study identifier ("N135-alt"), and sidecar/internal path reference ("sidecar/conventions/cross_seed_icc.md") and "reproducibility_check_tiers.md" sibling-convention reference per checklist §1 patterns. -->
If a subsequent study in this research program uses the same procedure,
promote §3's design choices to a standalone convention document,
following the study-specific-spec-first, convention-abstracted-on-second-use
pattern this program already uses for its reproducibility-check tier
structure. For the present study this spec is scoped to the study.
