# ADAPTIVE_RANK_COMPARISON_EXTERNAL_VALIDATION_TARGET_STUDY_IMPLEMENTATION_SPEC
## Repo-Facing CPU-Only Gradience Validation Plan

## Purpose

This document defines the next bounded Gradience research and validation study:

> **Adaptive-Rank Comparison / External Validation-Target Study**

The central question is:

> **Can Gradience's cheap spectral rank suggestions recover meaningful layerwise allocation structure and produce competitive fixed-budget compression in the currently validated bounded regime?**

This study is designed to validate one of Gradience's stronger medium-term claims:

- that a cheap pilot-measure-derive workflow can provide useful rank guidance
- without becoming an adaptive training method itself

It is **not**:
- a full reproduction of ARD-LoRA, LaRA, IGU-LoRA, or BladeLoRA
- a new training-method paper
- a claim that Gradience matches state-of-the-art adaptive rank allocation
- a broad cross-architecture benchmark
- a decoder-model extension study

It is a bounded **external validation-target** study using:
- Gradience's existing spectral rank machinery
- cleaned internal proxy comparators
- fixed-budget compression outcomes
- and carefully scoped interpretation

---

# 1. Why this study now

Gradience now has a much clearer bounded comparison stack than before.

Current bounded results support that:

- compression comparisons are only informative in **compressible families**
- in the present run, that means primarily **SST-2** and **IMDB**
- **OHT** is the strongest current spectral policy
- **gradient proxy** is the strongest current operational comparator
- **attenuate ablation** is the best current explanatory companion signal
- **rank-reduction ablation** is paused in this regime due degeneracy

That means the project is now in a good position to ask a sharper validation question:

> **Can Gradience's spectral rank policies function as credible cheap allocation advisors when compared against external-style importance targets and matched-budget compression outcomes?**

This is the right next step because it:
- directly tests a real Gradience claim
- stays within the validated bounded regime
- uses outside methods as validation targets, not product pivots
- and avoids unnecessary scope expansion

---

# 2. Core study questions

This study should answer:

## RQ1
Do Gradience's spectral rank policies produce competitive fixed-budget compression relative to simple baselines in the compressible subset?

## RQ2
Which spectral policy is the strongest lead candidate in the current bounded regime?

## RQ3
How do Gradience's spectral allocations relate to external-style importance targets:
- gradient-style
- ablation-style
- and simple matched-budget baselines

## RQ4
When spectral and proxy methods disagree, what kind of disagreement is it?
- ranking disagreement
- top-k agreement but budget redistribution
- task-family-specific difference
- source-quality-conditioned difference

## RQ5
What bounded claim does the evidence support about Gradience as a cheap rank advisor?

---

# 3. Core hypotheses

## H1 - Spectral policies are competitive in the compressible subset
At least one spectral policy should outperform uniform/random matched-budget compression on average in the compressible encoder regime.

## H2 - OHT is the lead spectral policy
Based on prior bounded results, OHT is expected to remain the strongest spectral policy in this study.

## H3 - Spectral policies align more with structural / ablation-style importance than with gradient-style importance
Even if gradient remains the stronger operational comparator, spectral policies may reflect a different but meaningful notion of layer importance.

## H4 - Gradient remains the operational default comparator
The current bounded evidence suggests gradient proxy is more stable and operationally stronger than ablation under the CPU protocol.

## H5 - The study should support a bounded advisor claim, not a state-of-the-art equivalence claim
Even strong results here should justify only a bounded "competitive cheap advisor" claim.

---

# 4. Scope constraints

## Hard limits
This study must remain:
- CPU-only
- bounded to the currently validated regime
- small-model only
- classification-only
- focused on compressible families
- fixed-budget compression only
- comparison-oriented rather than method-reproduction-oriented

## In-scope regime
Primary regime:
- shared-base small encoders
- LoRA / low-rank PEFT adapters already used in current Gradience validation
- compressible families only
- likely SST-2 and IMDB as primary informative set

## Explicitly non-primary / non-informative context
Families where realized compression saturates at ~1.0 should be:
- retained only as secondary context
- excluded from main allocation-quality claims

Examples from current bounded work:
- tweet_eval
- ag_news

## Out of scope
Do not:
- reproduce external adaptive methods end-to-end
- add decoder-only models
- expand to new artifact classes
- turn this into a full leaderboard study
- reopen rank-reduction ablation in this regime
- use saturated families in the primary conclusions

---

# 5. Study philosophy

The purpose of this study is not to prove that Gradience is "best."

The purpose is to determine whether:

> **cheap spectral pilot measurements recover a meaningful rank-allocation structure that is competitive under fixed-budget compression and interpretable relative to external importance targets.**

This means the study should emphasize:

- bounded, interpretable comparisons
- matched-budget evaluation
- task-family-aware reporting
- disagreement analysis
- honest separation between:
  - operational comparator
  - explanatory companion
  - and structural signal

---

# 6. Program architecture

This study should be executed in **six stages**:

## Stage A - Cohort freeze
Freeze the bounded informative cohort and define primary vs secondary analysis sets.

## Stage B - Allocation generation
Generate rank allocations from:
- spectral policies
- gradient proxy
- attenuate ablation
- simple baselines

## Stage C - Allocation comparison
Compare allocation patterns directly.

## Stage D - Compression evaluation
Compare fixed-budget compressed outcomes under matched evaluation conditions.

## Stage E - Disagreement anatomy
Analyze where and why spectral policies agree or disagree with proxies.

## Stage F - Bounded validation memo
Produce the bounded strategy-facing interpretation of what the line now supports.

---

# 7. Stage A - Cohort Freeze

## Objective

Create a fixed bounded cohort for the study.

## Why this stage matters

The main claims should not be diluted by non-informative or saturated families.

---

## Required cohort split

### Primary informative cohort
Families where compression is actually active.

Current expectation:
- SST-2
- IMDB

### Secondary non-informative context
Families retained for completeness but excluded from the main interpretation.

Current expectation:
- tweet_eval
- ag_news

---

## Required cohort metadata

For each adapter:
- adapter_id
- base model
- task family
- dataset
- source quality indicators
- compressible vs saturated label
- current inclusion status:
  - `primary_informative`
  - `secondary_context`

---

## Stage A deliverables

Create:

- `field_trials/rank_proxy_validation_v2/cohort_definition.md`
- `field_trials/rank_proxy_validation_v2/cohort_definition.json`

---

## Stage A success criteria

Stage A is successful if:
- the main informative subset is explicit
- saturated families are clearly separated
- source-quality metadata is retained for later control slices

---

# 8. Stage B - Allocation Generation

## Objective

Generate all candidate allocations under the same total budget conditions.

## Why this stage matters

This stage defines the objects that will be compared.

---

## Required allocation families

### Spectral policies
At minimum:
- `oht`
- `knee`
- `energy_90`
- optionally:
  - `erank`
  - `stable_rank_ceil`

### Operational comparator
- `proxy_gradient`

### Explanatory companion
- `proxy_ablation_attenuate`

### Baselines
- `uniform`
- `random_matched_budget`

---

## Budget settings

Use the already established bounded compression budgets, unless a strong reason emerges to change them:

- `0.35`
- `0.50`
- `0.65`

If changed, document why and keep the new set small.

---

## Required output fields

For each adapter x budget x method:
- layerwise allocation vector
- realized budget
- top-k layers
- allocation concentration statistics
- attention-vs-MLP split if applicable

---

## Stage B deliverables

Create:

- `field_trials/rank_proxy_validation_v2/allocation_table.json`
- `field_trials/rank_proxy_validation_v2/allocation_table.md`

Implementation may reuse or extend:
- existing CPU rank-proxy study runner

---

## Stage B success criteria

Stage B is successful if:
- all methods produce comparable matched-budget allocation outputs
- realized budgets are tracked
- primary informative cohort is complete

---

# 9. Stage C - Allocation Comparison

## Objective

Compare allocation structure across methods.

## Central question

> **Do Gradience's spectral policies allocate budget in a way that meaningfully resembles external importance targets or reveals a distinct structural notion of importance?**

---

## Why this stage matters

Compression outcomes alone do not tell you what kind of importance Gradience is capturing.

---

## Required metrics

At minimum include:

### Rank-order agreement
- Spearman
- Kendall tau-b where appropriate

### Top-k overlap
- q25
- q50
- or another clearly documented top-k definition

### Allocation concentration comparison
- attention-vs-MLP budget split
- top-k budget mass
- concentration summaries

### Disagreement summaries
- rank disagreement
- top-k agreement with budget redistribution
- proxy divergence flags

---

## Required reporting split

All main comparison tables must be:
- **informative-subset only**

Secondary context may be reported separately but must not drive the main interpretation.

---

## Stage C deliverables

Create:

- `field_trials/rank_proxy_validation_v2/allocation_comparison_table.md`
- `field_trials/rank_proxy_validation_v2/allocation_comparison_table.json`
- `field_trials/rank_proxy_validation_v2/task_family_stratified_readout.md`
- `field_trials/rank_proxy_validation_v2/task_family_stratified_readout.json`

Optional figures:
- `allocation_agreement_heatmap.svg`
- `topk_overlap_comparison.svg`

---

## Stage C success criteria

Stage C is successful if:
- spectral vs proxy relationships are explicit
- task-family differences are visible
- saturated families do not contaminate the main analysis

---

# 10. Stage D - Compression Evaluation

## Objective

Compare compressed adapter outcomes under matched budgets and dataset-matched evaluation.

## Central question

> **Do Gradience's spectral policies produce competitive fixed-budget compression in the primary informative subset?**

---

## Why this stage matters

This is the main practical validation stage.

---

## Required evaluation setup

For each adapter x budget x method:
- evaluate compressed adapter on the exact dataset-matched evaluation set
- compare against:
  - full adapter
  - uniform
  - random matched-budget
  - proxy comparators where appropriate

---

## Required outcome metrics

At minimum:
- `delta_vs_full_adapter`
- `delta_vs_uniform`
- `delta_vs_random`
- optionally:
  - `delta_vs_best_proxy`
  - `delta_vs_gradient`
  - `delta_vs_ablation`

The main outcome should remain interpretable and consistent across methods.

---

## Required reporting split

Primary tables:
- informative-subset only

Secondary context:
- clearly labeled as non-informative due saturation or realized-budget collapse

---

## Stage D deliverables

Create:

- `field_trials/rank_proxy_validation_v2/compression_evaluation_table.md`
- `field_trials/rank_proxy_validation_v2/compression_evaluation_table.json`

Optional figures:
- `compression_delta_by_method.svg`
- `compression_delta_by_task_family.svg`

---

## Stage D success criteria

Stage D is successful if:
- the compressible subset yields a clean primary comparison
- the strongest spectral policy is identifiable
- the relative position of gradient and attenuate is explicit

---

# 11. Stage E - Disagreement Anatomy

## Objective

Explain where and why methods diverge.

## Central question

> **When spectral, gradient, and ablation-style signals disagree, what kind of disagreement is it?**

---

## Why this stage matters

The disagreement pattern is one of the most scientifically useful outputs of this line.

---

## Required analyses

### Analysis 1 - Gradient vs spectral
- where does gradient win behaviorally over OHT?
- is this concentrated in particular task families or source-quality bands?

### Analysis 2 - Spectral vs attenuate
- where do spectral and attenuate align structurally?
- where do they diverge?
- is top-k agreement stronger than full ranking agreement?

### Analysis 3 - Source-quality-gap control slice
Use dataset-matched source-quality bands:
- `near_top`
- `mid_gap`
- `large_gap`
- `single_source_dataset`

This should remain primary in the bounded interpretation.

### Analysis 4 - Task-family-specific interpretation
Especially:
- SST-2
- IMDB

These should be separated, not collapsed into one average if they behave differently.

---

## Stage E deliverables

Create:

- `field_trials/rank_proxy_validation_v2/disagreement_memo.md`
- `field_trials/rank_proxy_validation_v2/source_quality_gap_control_slice.md`
- `field_trials/rank_proxy_validation_v2/source_quality_gap_control_slice.json`
- `field_trials/rank_proxy_validation_v2/compressible_family_summary.md`

Optional figures:
- `gradient_vs_oht_gap_by_quality_band.svg`
- `spectral_vs_ablation_alignment_by_family.svg`

---

## Stage E success criteria

Stage E is successful if:
- the disagreement pattern is no longer mysterious
- the project can distinguish structural similarity from operational superiority
- the bounded strategy conclusion becomes clearer

---

# 12. Stage F - Bounded Validation Memo

## Objective

Write the final bounded interpretation of what this study supports.

## Central question

> **What does this line now justify saying about Gradience as a cheap rank advisor?**

---

## Why this stage matters

This is the strategy-facing payoff.

The study is only useful if it ends in a clear bounded conclusion.

---

## Required memo sections

### Section 1 - What was tested
Briefly describe:
- informative subset
- methods compared
- budgets
- dataset-matched evaluation
- source-quality control

### Section 2 - What the strongest positive result is
Examples:
- OHT is the lead spectral policy
- spectral policies are competitive in the compressible subset
- spectral aligns more with attenuate-style structure than gradient-style structure

### Section 3 - What remains bounded
Examples:
- encoder-only
- classification-only
- compressible families only
- no broad adaptive-method equivalence claim

### Section 4 - Current policy interpretation
Expected current bounded interpretation:
- gradient = operational default proxy
- attenuate = explanatory companion
- OHT = lead spectral policy
- no broader escalation yet

### Section 5 - What would strengthen the line next
Examples:
- external recovered allocation targets from published methods
- decoder-side fingerprinting once compute returns
- broader but still bounded family replication

---

## Stage F deliverables

Create:

- `field_trials/rank_proxy_validation_v2/bounded_validation_memo.md`
- `docs/strategy/rank_proxy_bounded_validation_summary.md`

Optional JSON:
- `field_trials/rank_proxy_validation_v2/bounded_validation_summary.json`

---

## Stage F success criteria

Stage F is successful if:
- the project has a clear bounded interpretation
- the memo is usable in strategy, packaging, and external writing
- the result does not overclaim beyond the current evidence

---

# 13. Relationship to existing work

This study should explicitly build on:
- the current CPU rank-proxy study
- task-family stratified readout
- source-quality-gap control slice
- ablation reliability follow-up
- bounded memo updates
- current claims ladder and bounded strategy docs

It should **not** pretend to start from zero.
It is the cleanly packaged and canonicalized version of this line.

---

# 14. Relationship to product strategy

This study should inform:
- Gradience's "cheap rank advisor" story
- what rank-policy language is safe in public/internal docs
- which spectral policies deserve emphasis
- what should be presented as operational vs explanatory

It should **not** be used to claim:
- equivalence to adaptive-rank training methods
- cross-architecture generality
- decoder-side validation
- universal ranking truth

---

# 15. Guardrails

Do not:
- reintroduce saturated families into the primary interpretation
- treat gradient superiority as disproving the spectral line
- treat spectral-ablation alignment as operational dominance
- elevate every spectral policy equally
- broaden the story beyond the informative subset

This is a bounded validation study.

---

# 16. Suggested execution order

1. Stage A - cohort freeze
2. Stage B - allocation generation
3. Stage C - allocation comparison
4. Stage D - compression evaluation
5. Stage E - disagreement anatomy
6. Stage F - bounded validation memo

If the existing outputs already cover some of these stages, canonicalize them rather than rerun unnecessarily.

---

# 17. Deliverables checklist

By the end of this study, the repo should contain:

## Cohort
- [ ] cohort definition MD/JSON

## Allocations
- [ ] allocation table MD/JSON

## Comparisons
- [ ] allocation comparison table MD/JSON
- [ ] task-family stratified readout MD/JSON

## Evaluation
- [ ] compression evaluation table MD/JSON

## Interpretation
- [ ] disagreement memo
- [ ] source-quality-gap control slice MD/JSON
- [ ] compressible family summary MD

## Summary
- [ ] bounded validation memo
- [ ] strategy summary doc
- [ ] optional summary JSON

---

# 18. Definition of done

This study is complete when one of the following is true:

## Success condition
The project can make a clear bounded claim that Gradience's spectral rank policies are competitive fixed-budget allocation guides in the compressible encoder subset, with OHT as the lead current spectral policy and with a clear distinction between operational and explanatory comparison targets.

## Partial success condition
The study clarifies the bounded regime and comparison structure, but the evidence remains too mixed to support a clean advisor story.

## Negative completion condition
The bounded informative subset does not support a meaningful spectral advisor claim once all controls are applied.

All three outcomes are useful.

---

# 19. Bottom line

This study asks one focused and strategically valuable question:

> **In the subset where compression actually matters, does Gradience recover a meaningful and competitive notion of layerwise rank allocation?**

Answering that is the strongest remaining CPU-only Gradience validation line before decoder-side compute returns.
