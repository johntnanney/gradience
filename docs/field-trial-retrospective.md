# Field Trial Retrospective

**Date:** April 2026
**Scope:** 5 inventories, 3 backbones, 4 task families, 16 evaluated merges, 2 micro-campaigns
**Companion to:** [Technical Report §5](technical-report.md#5-field-trial-validation) (results), [Product Validation](product/product-validation.md) (summary)

This document tells the story of the field trials — not just what we found, but what we expected, what surprised us, and what changed in the product as a result. The technical report covers the validated conclusions; this retrospective covers the judgment calls.

---

## Phase 1: The Evidence Gate Discovery

### What we expected

We expected the first three pilots to test whether spectral analysis plus behavioral evidence could narrow a candidate pool effectively. The plan was simple: give Gradience a set of adapters, run the pipeline, see if the retained set made sense.

### What actually happened

**Pilot 1** (same-task control, 3 adapters, 3 pairs) returned an empty retained set. Zero pairs. This looked like failure — a same-task inventory should have retained most pairs. But the cause was instructive: one adapter (jmeneu) had no behavioral evaluation data, and the other two had evaluation artifacts that made their scores unreliable. With no eligible adapters, the pipeline had nothing to compare.

The lesson was immediate and unambiguous: **without behavioral evidence, Gradience produces nothing useful.** This was not a design flaw — it was the correct answer. An adapter with beautiful spectral geometry and zero demonstrated task performance is still a bad merge candidate. The evidence gate, which we had implemented as one feature among several, turned out to be the single most impactful component in the system.

**Pilot 2** (mixed-task, RoBERTa, 5 adapters, 10 pairs) was the first real success. From 10 candidate pairs, Gradience retained 1: the same-task AG News pair. That is a 90% reduction, and it was the right call — the only pair where both adapters were eligible and targeted the same task. Cross-task pairs received the advisory. The irony adapter was correctly classified as `uncertain` (delta exactly 0.000 against base). A field note from this pilot reads: "This is Gradience's best result so far."

**Pilot 3** (large mixed-task, DistilBERT, 8 adapters, 28 pairs) scaled the test to a more realistic inventory. Two adapters were correctly excluded as weak. Two same-task pairs were retained (AG News and SST-2), a 93% reduction. But something important happened that we didn't anticipate: a hate-speech pair (jaesun × Aureliano) was structurally clean and same-task, but got excluded entirely because one source adapter (Aureliano) was flagged as weak. The pair was invisible in the output — not flagged, not mentioned, just absent.

This was technically correct (one source didn't meet the evidence threshold) but operationally wrong. A practitioner would want to know that a structurally plausible same-task pair exists but is blocked by thin evidence. The pair wasn't a bad merge candidate; it was an *under-evidenced* one. The distinction matters.

### What changed

Nothing yet — we deferred the fix to Phase 2b. But the observation was logged, and it shaped everything that followed.

---

## Phase 2: Does the Narrowing Actually Work?

### What we expected

Phase 1 showed that Gradience could *recommend* merges. Phase 2 asked the harder question: do the recommendations produce better outcomes than the alternatives? We merged the retained pairs and several controls, and evaluated them on held-out data.

The hypothesis was straightforward: retained pairs should degrade less (or improve more) than cross-task controls.

### What actually happened

**Pilot 2 evaluation:** The retained AG News pair improved to 0.944 accuracy — beating both source adapters (best source 0.938, delta +0.006). A modest gain, but real. This was the first evidence that Gradience's top recommendation produced an actually useful merge.

One control surprised us. The cross-task hate × irony merge substantially improved over the hate adapter alone (0.602 vs 0.486, delta +0.116), despite Gradience flagging it as cross-task. This was our first encounter with a pattern that would recur: **cross-task structural signals are cautionary, not prohibitive.** Some cross-task merges transfer useful features. The task-boundary advisory is correct to flag them for caution, but wrong to imply they always fail.

**Pilot 3 evaluation:** Both retained pairs degraded somewhat (AG News −0.018, SST-2 −0.066), but both degraded less than the cross-task controls (−0.042 and −0.048). The prioritization was correct: retained pairs were the better first choices.

But the real story was the near-miss hate pair. jaesun × Aureliano — the pair Gradience had excluded in Phase 1 because Aureliano was flagged weak — improved by +0.078 over the best source. This was a structurally clean, same-task merge that worked well, and the system had hidden it from the practitioner entirely.

### The judgment call

The evidence gate was doing its job — it correctly identified that Aureliano had thin evidence. But the *consequence* of the gate (complete invisibility of the pair) was disproportionate to the *reason* (marginal weakness in one source). The right response wasn't to lower the gate threshold — that would let genuinely weak adapters contaminate the retained set. It was to create a new category: **near-miss**.

Near-miss pairs are same-task, structurally plausible, blocked only by evidence constraints. They belong in the output — visible, explained, ranked — not in the retained set. The practitioner sees them, understands why they're not retained, and can choose to fix the evidence gap (run a better evaluation on the weak source) or accept the risk.

This feature was implemented immediately: added to the action plan, the preflight summary, and the HTML report. It is one of the most consequential design changes to come out of the field trials.

---

## Phase 2b: Is the Near-Miss Pattern Real?

### What we expected

Phase 2 found one near-miss pair that worked well. One data point is an anecdote, not a pattern. Phase 2b was designed specifically to answer: **is the near-miss category genuinely useful, or was the hate-speech result an exception?**

We built two new inventories engineered to produce near-miss pairs through evidence-gate variance — one irony cluster on DistilBERT (8 adapters) and one hate+emotion cluster on BERT-base (8 adapters). We merged 11 pairs across retained, near-miss, and cross-task control categories.

### What actually happened

The pattern held. Across three backbones and three task families:

| Category | Pairs | Avg Δ vs best source | Improvers |
|----------|-------|---------------------|-----------|
| Retained | 4 | −0.018 | 1 (25%) |
| Near-miss | 7 | −0.006 | 1 (14%) |
| Cross-task control | 1 | −0.096 | 0 |

Near-miss pairs degraded 5× less than the cross-task control and slightly *less* than retained pairs. The near-miss category was not just real — it was arguably the safest category in the data.

### The deeper finding: severity gradation

The more interesting result was *within* the near-miss category. How weak the excluded source was predicted how well the merge would work:

- **Barely weak** (delta −0.002 to −0.010 from threshold): average merge degradation −0.007. Functionally indistinguishable from retained pairs.
- **Deeply weak** (delta < −0.050): average degradation −0.045. Better than cross-task controls but genuinely risky.

The evidence gate boundary (delta < 0) was correct — it just needed a graduated response on the other side. A barely-missed source is almost certainly fine for merging. A deeply-missed source probably isn't.

### What changed

Near-miss severity ranking was added to the action plan. Pairs are now classified as marginal (small evidence gap), moderate (fixable with better evaluation), or substantial (far from eligible), and the action plan presents them in that order. This is a small implementation change — mostly rendering — but it meaningfully improves the practitioner's ability to prioritize their next steps.

---

## Campaign A: Same Family, Different Dataset

### What we expected

Gradience's task-boundary detection is metadata-driven: it compares evaluation dataset labels. If Adapter A was trained on SST-2 and Adapter B on IMDB, the system flags them as cross-task — different datasets, different labels. But SST-2 and IMDB are both binary sentiment classification. They're the same *task family* with different surface data. We expected the cross-task flag to be overprotective here.

### What actually happened

We built a 4-adapter inventory (2 SST-2, 2 IMDB on DistilBERT) and evaluated 7 merges:

| Category | Avg Δ vs best source |
|----------|---------------------|
| Same-task (SST-2 × SST-2) | −0.017 |
| Same-family (SST-2 × IMDB) | −0.022 |
| Cross-task control | −0.047 |

The gap between same-task and same-family was 0.005 — within noise. Same-family pairs behaved like same-task pairs, not like cross-task controls. The boundary was indeed overprotective for this task family.

A secondary finding: Gradience's structural recommendations (which merge strategy to use) varied correctly based on spectral properties (rank mismatch, norm ratio) even when the task advisory was misleading. The audit_aware strategy recommendation outperformed uniform_linear, confirming that structural signals carry information independent of task metadata.

### What changed

A task-family registry was added — a static mapping from known datasets to validated task families. Currently, the only validated family is binary sentiment (SST-2, IMDB, Yelp Polarity, Amazon Polarity). When both adapters in a pair belong to the same family, the task advisory downgrades from "cross-task caution" to "same-family informational." This is conservative — the registry is additive and manually curated — but it eliminated a known class of false positives.

---

## Route 2: Can the Workflow Broaden?

### What we expected

The core pipeline was designed for LoRA adapter merging. Route 2 asked: can the same workflow shape — evidence bootstrap, eligibility classification, pairwise analysis, action plan — apply to full fine-tuned checkpoints? The underlying question was whether Gradience was a merge-triage tool or something more general.

### What actually happened

Two checkpoint inventory trials (T01 and T02) confirmed that the workflow transfers. The representation path is different (summary statistics on checkpoint deltas rather than factor extraction on LoRA matrices), but the pipeline stages, evidence gate, and action plan all apply without modification. Both trials produced empty retained sets — the checkpoint pools didn't contain strong same-task pairs — but the action plans were still useful: they explained *why* each pair was excluded and surfaced near-miss candidates for follow-up.

Trial T02 introduced the same-family dimension to checkpoints. An SST-2 checkpoint and a Yelp checkpoint, both binary sentiment, were correctly identified as same-family and routed to the informational advisory rather than cross-task caution.

### The judgment call

Checkpoint triage was shipped as an alpha feature with an explicit scope contract: shared base model, small encoders, classification only, evidence required. The workflow generalizes; the merge execution does not (checkpoint merging is a different problem). We chose to ship the triage capability while being explicit about what it doesn't do, rather than waiting until the full stack was validated.

---

## What We Learned

Five lessons emerged from the field trials that shaped the product's design philosophy:

**1. Evidence comes first, always.** The evidence gate is not a preliminary filter — it is the foundational feature. Without it, spectral analysis produces elegant but useless results. Pilot 1 taught this in the starkest possible way: zero output from zero evidence. Every subsequent design decision was made with this constraint in mind.

**2. Conservatism needs a visible middle.** The evidence gate is correctly calibrated — it should not be lowered. But conservatism creates invisible exclusions, and invisible exclusions are the worst kind of false negative. The near-miss feature exists because a conservative gate needs a graduated response: not "admitted" or "excluded" but "excluded, and here's why, and here's what you can do about it."

**3. Task boundaries are real but coarser than metadata suggests.** Dataset labels create false boundaries within task families. Binary sentiment is binary sentiment whether the data comes from movie reviews or product reviews. But the boundary between sentiment and entailment is real. The task-family registry threads this needle: it relaxes known false boundaries while preserving genuine ones.

**4. Structural signals and behavioral outcomes are correlated but not identical.** Retained pairs are better *candidates* than cross-task controls, not guaranteed successes. The 29% improvement rate (2 of 7 retained pairs improved over best source) is honest: most merges degrade somewhat even when structurally sound. The value is in *narrowing and prioritizing*, not in predicting success.

**5. Each surprise became a feature.** The empty Pilot 1 output led to the evidence gate's prominent role in documentation and error messages. The invisible hate-speech pair led to near-miss. The near-miss severity gradient led to marginal/moderate/substantial ranking. The same-family false positive led to the task-family registry. The product matured through its failures, not despite them.

---

## Source Material

The raw field trial data, after-action notes, and evaluation logs are in `field_trials/`:

| Document | What it covers |
|----------|---------------|
| `field_trials/README.md` | Phase index and status |
| `field_trials/phase2_evaluation_report.md` | Merge follow-through results |
| `field_trials/phase2b_confirmation_memo.md` | Near-miss confirmation data |
| `field_trials/near_miss_validation.md` | Cross-backbone near-miss validation |
| `field_trials/product_validation_memo.md` | Aggregate product assessment |
| `field_trials/task_family_equivalence_memo.md` | Campaign A results |
| `field_trials/marginal_adapter_behavior_memo.md` | Campaign B results |
| `field_trials/checkpoint_inventory_summary.md` | Route 2 checkpoint triage |
