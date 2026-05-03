# Reviewer-Deferred Tasks (Empirical Tables)

Tracking file for two reviewer-comment tasks that require empirical Phase-4
pipeline data and are not landed in the framing/figure commit. Both tasks
have data-source resolution noted here; both should ship in a separate
empirical-table commit/branch from the framing/figure edits.

Status as of 2026-05-03: data sources verified and tractable; not yet
applied to manuscript.

---

## Task #12 — Reviewer #3: split tolerance schedule into within-rule and across-rule

### Reviewer concern (estimand clarity)

The current Table 4 in §sec:tolerance-schedule-results visually implies
rule-specific tolerances everywhere, but in g-theory cells the displayed
tolerances are actually rule-invariant per (benchmark, model) — i.e., the
SEM is computed at the (benchmark, model) level with rule treated as a
random facet, and reported once per row. In parse-failure cells, tolerances
are rule-specific (sample SD of cell scores). The schedule is therefore
mixing two estimands (within-rule for `pf`, across-rule for `g`) under one
table layout. Reviewer #3 wants this distinction made explicit.

### Data source

`analysis/tolerance_schedules/tolerance_by_cell.csv`

Relevant columns:

- `sem_single` / `tolerance_single` — single-occasion (k=1)
- `sem_within_rule` / `tolerance_within_rule` — averaging across prompts and seeds, scoring rule fixed
- `sem_prompt_avg` / `tolerance_prompt_avg` — averaging across prompt and rule and seed (across-rule)
- `sem_full_design` / `tolerance_full_design` — full-design averaging
- Plus 95% CI bounds for each tolerance estimate

The current Table 4 uses columns 1 + 4 (single + full). The within-rule and
across-rule schedules are recoverable directly from columns 2 + 3 — the
numbers already exist in the Phase-4 output.

### Revision goal

Replace the current Table 4 with two tables (or one restructured table with
clearly labeled estimand columns):

- **Table A — within-rule schedule.** Rows: benchmark × model × rule.
  Values: per-rule SEM/tolerance. The schedule a report gets when it
  commits to one scoring rule.
- **Table B — across-rule schedule.** Rows: benchmark × model. Values:
  rule-marginalized SEM/tolerance. The schedule a report gets when it
  treats scoring rule as part of the admissible measurement universe
  (the implicit position of "MMLU score" claims that don't name a rule).

Add a short interpretation paragraph distinguishing the two estimands:

> The within-rule and across-rule schedules answer different measurement
> questions. A within-rule tolerance licenses precision conditional on a
> specified scoring rule. An across-rule tolerance licenses precision for a
> benchmark–model claim that treats scoring rule as part of the admissible
> measurement universe. The latter is therefore generally more conservative
> when scoring-rule variance is non-negligible. Reporting a single benchmark
> score without naming the scoring rule implicitly makes the across-rule
> claim.

### Framing note

This is not just a formatting complaint. It is an estimand-clarity
complaint. Reviewer #3 is asking, in effect, "Tell me what uncertainty
universe this tolerance belongs to." That request is completely aligned
with the paper's thesis, so it is worth fixing cleanly rather than patching
around.

### Footnote at line 930

The footnote currently says: "A complementary across-rule schedule,
treating scoring rule as an additional random facet rather than a fixed
condition, answers a different question and is discussed in
§\ref{sec:future-work}." This should be removed once the across-rule
schedule lands in the body text — the footnote was the deferral marker the
revision now resolves.

---

## Task #13 — Reviewer #4: regime-split audit table

### Reviewer concern (evidentiary table for regime classification)

Table 4 reports a `Regime` column (`g` or `pf`) per cell, but the manuscript
does not surface the per-cell parseability evidence that drove each
classification. A reader cannot verify, on the manuscript alone, that any
specific cell's regime assignment matches its empirical parseability rate.
Reviewer #4 wants an audit table that closes that gap.

### Data source

`runs/normalized/condition_level_primary.csv`

Relevant field: `parseability_rate` (per condition, per cell)

Aggregation: median `parseability_rate` by (benchmark, model) for `scoring_rule_id == "generate_parse"` only.

Verified per-cell medians (15 cells, all match Table 4's `Regime` column):

| Benchmark        | Model                  | Median parseability | Threshold (0.30) |
|------------------|------------------------|---------------------|------------------|
| ARC-Challenge    | Pythia-1.4B            | 0.4010              | g_theory         |
| ARC-Challenge    | Pythia-410M            | 0.3225              | g_theory         |
| ARC-Challenge    | Qwen2.5-1.5B-Instruct  | 0.5107              | g_theory         |
| HellaSwag        | Pythia-1.4B            | 0.0604              | parse_failure_dominated |
| HellaSwag        | Pythia-410M            | 0.1877              | parse_failure_dominated |
| HellaSwag        | Qwen2.5-1.5B-Instruct  | 0.9669              | g_theory         |
| MMLU panel       | Pythia-1.4B            | 0.5003              | g_theory         |
| MMLU panel       | Pythia-410M            | 0.2835              | parse_failure_dominated |
| MMLU panel       | Qwen2.5-1.5B-Instruct  | 0.3881              | g_theory         |
| TruthfulQA-MC    | Pythia-1.4B            | 0.0251              | parse_failure_dominated |
| TruthfulQA-MC    | Pythia-410M            | 0.2075              | parse_failure_dominated |
| TruthfulQA-MC    | Qwen2.5-1.5B-Instruct  | 0.7411              | g_theory         |
| Winogrande       | Pythia-1.4B            | 0.0600              | parse_failure_dominated |
| Winogrande       | Pythia-410M            | 0.1310              | parse_failure_dominated |
| Winogrande       | Qwen2.5-1.5B-Instruct  | 0.8461              | g_theory         |

### Revision goal

Add a 15-row G&P-only audit table to §sec:results (between
§sec:variance-components-results and §sec:tolerance-schedule-results, or as
a subsection inside the latter), with the structure above plus the
following caption:

> Parseability audit for the generate-and-parse regime classification.
> Median parseability is reported for each benchmark–model cell and
> compared with the pre-registered 0.30 threshold. Cells below threshold
> are assigned to the parse-failure regime. Log-likelihood cells are not
> separately parseability-audited because no parsing step is involved.

Plus a crosswalk sentence near the table explaining how G&P parseability
determines the pf designation and why LL rows are not separately audited.

### Adjacent prose correction (§sec:design-regime-split bimodality claim)

The current §sec:design-regime-split prose at line 793-794 of draft_v1.tex
says:

> The threshold of $0.30$ is locked as a parameter of the
> pre-confirmatory-analysis amendment, not tuned post-hoc on outcome data.
> Its value was chosen at the time of the amendment to align with the
> empirical distribution of cell-level parseability observed in the early
> GPU-run data (a clear bimodality between cells well above $0.5$ and
> cells well below $0.10$, with few cells in the $0.10$--$0.50$ band that
> the threshold separates).

The actual audit data has **six** cells in the 0.10–0.50 band (all four
Pythia-410M G&P cells are between 0.13 and 0.33; MMLU Pythia-1.4B is at
0.5003; MMLU Qwen at 0.3881). Several of these are very close to the 0.30
threshold (Pythia-410M MMLU at 0.2835 just below; Pythia-410M ARC at 0.3225
just above). The bimodality claim is only loosely accurate; an audit table
exposes this.

The §sec:design-regime-split prose should be revised in the same commit as
the audit table. Suggested adjustment:

- Soften "a clear bimodality" to something like "an empirical distribution
  with most cells either above 0.5 or below 0.20, plus a transition band of
  cells near the threshold that the audit table makes visible."
- Add a brief threshold-sensitivity note, or at minimum a sentence that the
  0.30 cutoff is pre-registered and locked, and that near-threshold cells
  are surfaced in the audit table rather than hidden.

The revised prose preserves the substantive defense (the threshold is
pre-registered, not tuned post-hoc) while honestly representing the
distribution's shape.

---

## When to apply

Both Task #12 and Task #13 alter the evidence presentation more
substantially than the framing/figure edits (Tasks #14, #21, #20 — landed
in the framing-and-figure commit). They should be a separate empirical-table
commit on a follow-up branch. Suggested branch name:
`papers/benchmark_reliability_study/empirical-table-revisions`.

Compile-verify after each table addition; expect page-count growth (Table
A + Table B for #12 may be 2–3 pages; Task #13 audit table is ~1 page plus
a paragraph of prose correction).

The §sec:tolerance-schedule construction prose at line 359-364 may need
small adjustment to match the new estimand-explicit framing — the "tolerance
schedule estimates the SEM per cell under the declared measurement universe"
sentence currently elides the within-rule/across-rule distinction the new
tables will make explicit.
