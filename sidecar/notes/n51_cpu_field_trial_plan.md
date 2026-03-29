# CPU-Only Field Trial Plan for Gradience

## Small Public Encoder Adapters

### Purpose

This plan is for getting real-world experience with Gradience as a preflight decision system using small public encoder adapters on CPU only.

The goal is not to brute-force a giant merge benchmark. The goal is to answer practical questions:

* Does Gradience narrow real inventories usefully?
* Does it help pick plausible evaluation subsets?
* Does it save wasted merge/eval effort?
* Where do the summaries feel helpful, vague, or annoying?
* What does it get right, and what does it miss?

This plan is designed to be:

* realistic on CPU
* operationally informative
* small enough to run now
* structured enough to produce reusable evidence

---

## 1. Core idea

Use Gradience in field-trial mode:

1. collect real public small-model adapter inventories
2. run Gradience preflight
3. evaluate only:
   * retained / evaluate-first pairs
   * a few excluded controls
   * optionally one tempting-but-risky cross-task pair
4. write short after-action notes
5. compare Gradience's practical guidance to actual outcomes

This is not a leaderboard exercise. It is a workflow validation program.

---

## 2. Why small public encoder adapters

This plan should stay within CPU-friendly limits.

### In-scope model classes

Prefer:

* DistilBERT
* RoBERTa-base
* BERT-base or similar compact encoder families

### Preferred task types

Prefer:

* sentence classification
* binary classification
* pair classification
* moderate-size validation sets

Examples:

* SST-2
* MRPC
* RTE
* QNLI
* domain sentiment tasks like Yelp / Amazon reviews

### Why these are good

They are:

* cheap to inspect
* cheap to merge
* cheap to evaluate on slices
* already aligned with the evidence base Gradience was built around

---

## 3. What this plan is trying to validate

The field trials should test practical Gradience claims, not just merge behavior.

### Claim A — Search-space reduction is useful

Does Gradience reduce the candidate space enough to matter?

### Claim B — Retained subsets are better than naive search

Are retained / evaluate-first pairs more often useful than excluded controls?

### Claim C — Policy summary and action plan help real decisions

Do the reports actually make it easier to decide what to evaluate?

### Claim D — Trust/provenance matters in practice

Does evidence quality change which inventories feel worth exploring?

### Claim E — Larger-inventory ergonomics work

Do region summaries, candidate-space maps, and review packets help on real 8–20 adapter inventories?

---

## 4. Trial structure

### Number of inventories

Start with:

* 8 to 10 inventories

That is enough variety without becoming a slog.

### Composition of the inventory set

Try to include:

#### Type 1 — Same-task control inventories

2 inventories

Purpose:

* confirm that same-task safe zones feel useful and non-annoying
* make sure Gradience is not overcomplicating clean pools

#### Type 2 — Standard mixed-task inventories

3 inventories

Purpose:

* test the core use case
* see whether Gradience narrows mixed-task space sensibly

#### Type 3 — Messy mixed-quality inventories

2 inventories

Purpose:

* test trust/provenance handling
* test whether weak-evidence messaging is useful

#### Type 4 — Larger inventories

2 to 3 inventories with:

* 8+ adapters or 20+ pairs

Purpose:

* test large-inventory mode
* test region summaries / candidate-space maps / HTML reports

---

## 5. Inventory selection rules

Each inventory should come from public adapters and meet these conditions:

### Required

* same backbone family within an inventory
* adapters are loadable
* tasks are identifiable
* enough metadata exists to construct a usable inventory

### Preferred

* at least some inventories should include:
   * same-task seeds
   * related tasks
   * distant tasks
   * one or more weaker or suspicious sources

### Avoid

Avoid inventories that require:

* large decoder models
* generation-heavy evals
* full custom training pipelines
* giant validation sets

---

## 6. Per-inventory workflow

For each inventory, follow the same trial steps.

### Step 1 — Build the inventory

Collect:

* adapter names
* task labels
* backbone
* any available evidence / eval metadata

Create a clean inventory manifest.

### Step 2 — Run Gradience preflight

Generate:

* preflight summary
* review packet
* HTML report
* policy summary
* action plan
* region summary if large inventory
* JSON artifacts

Archive all outputs.

### Step 3 — Record the Gradience stance

Before running any merges, record:

* retained candidates
* evaluate-first subset
* excluded / deprioritized sources
* same-task safe zones
* cross-task caution zones
* policy summary
* trust/evidence summary

This is important. Lock in what Gradience said before seeing outcomes.

### Step 4 — Define a tiny evaluation plan

For each inventory, choose:

#### A. Retained subset

Evaluate:

* all retained pairs if the set is small

or

* the top 2–3 evaluate-first pairs

#### B. Excluded controls

Evaluate:

* 1–2 excluded pairs

These are crucial. Without them, you cannot tell whether Gradience is actually saving search.

#### C. Optional "tempting" control

If relevant, add:

* one pair that looks intuitively plausible but that Gradience deprioritized

This is especially good for mixed-task inventories.

### Step 5 — Merge and evaluate

Run the selected merges and evaluate on:

* modest validation slices
* task-appropriate metrics
* a standardized small evaluation protocol where possible

Keep it small. The point is comparative evidence, not exhaustive benchmarking.

### Step 6 — Write a field note

For each inventory, write a short note:

* what Gradience predicted / implied
* what was evaluated
* what happened
* what Gradience got right
* what it got wrong
* whether the product surfaces helped
* whether the narrowing was worth it

This is one of the most important outputs.

---

## 7. Evaluation budget per inventory

Keep each inventory small enough to stay CPU-friendly.

### Recommended budget

Per inventory:

* 2–5 merged evaluations total

Suggested split:

* 2–3 retained pairs
* 1–2 excluded controls

That is enough to learn something without turning the plan into a compute marathon.

---

## 8. What to measure

This plan should track both merge outcomes and product usefulness.

### 8.1 Merge outcome measures

For each evaluated pair:

* did it outperform the excluded controls?
* did it degrade badly?
* did the retained subset contain the clearly better candidates?
* did excluded controls mostly deserve exclusion?

Do not overcomplicate the metrics. The question is comparative practical usefulness.

### 8.2 Workflow usefulness measures

For each inventory, rate:

#### Search reduction usefulness

* Did Gradience narrow the search space enough to matter?

#### Interpretive clarity

* Did the policy summary/action plan feel clear?
* Did the report make the inventory easier to reason about?

#### Trust usefulness

* Did provenance / evidence language change the evaluation plan?

#### Report usefulness

* Did you actually open the HTML report first?
* Was the review packet enough for handoff?

#### Large-inventory usefulness

* Did region summary and candidate-space map help?

Use a simple qualitative rating:

* high
* medium
* low

You do not need a formal scale unless you want one.

---

## 9. Trial outputs

By the end of the field trial plan, produce these artifacts.

### Per inventory

* inventory manifest
* Gradience preflight outputs
* tiny evaluation plan
* evaluation results
* short field note

### Across all inventories

* one comparison table
* one summary memo
* one list of product pain points
* one list of what Gradience gets consistently right
* one list of what still feels weak or confusing

---

## 10. Suggested directory structure

```
field_trials/
  inventory_01_same_task_control/
    manifest.json
    preflight/
    eval_plan.md
    eval_results.json
    field_note.md
  inventory_02_mixed_task/
    ...
  summary/
    trial_comparison_table.md
    trial_summary_memo.md
    product_pain_points.md
    strengths.md
```

This keeps the work clean and reusable.

---

## 11. Trial success criteria

This plan is successful if, across the inventory set, you can answer:

### Success condition A

Gradience usually reduces search enough to matter.

### Success condition B

Retained / evaluate-first pairs usually outperform or are more plausible than excluded controls.

### Success condition C

The reports actually help real decisions.

### Success condition D

You identify concrete product weaknesses from use, not from speculation.

Even mixed results are useful if they are concrete.

---

## 12. What to look for specifically

As you run the trials, pay attention to these questions:

* Does Gradience ever exclude something surprisingly good?
* Does it keep too many bad-but-plausible pairs alive?
* Does the action plan feel actionable or generic?
* Does the policy summary clarify the run or just repeat the obvious?
* Do trust/provenance notes actually affect behavior?
* Does large-inventory mode genuinely help?
* Does the HTML report become the default thing to open?
* Does the portfolio view help prioritize across inventories?

These are the questions that define the product's next practical phase.

---

## 13. What not to do

Do not:

* evaluate every pair
* turn this into a benchmark paper
* add new core logic mid-trial
* chase one-off surprising results into immediate feature work
* broaden to large decoder adapters
* rebuild the methodology while still in the field-trial phase

This phase is about learning from use, not inventing new machinery.

---

## 14. Suggested rollout

### Phase 1 — Pilot

Run:

* 3 inventories

Goal:

* confirm the workflow is manageable
* make sure the eval budget is realistic
* fix obvious operational annoyances

### Phase 2 — Main field trial

Run:

* remaining 5–7 inventories

Goal:

* gather enough variety to see patterns

### Phase 3 — Synthesis

Produce:

* one summary memo
* one product feedback memo
* one "what we learned from field use" note

---

## 15. Best first inventory set

If I were choosing the pilot three, I'd do:

### Pilot 1

Same-task control pool

Purpose: confirm the tool stays appropriately conservative

### Pilot 2

Standard mixed-task inventory

Purpose: test the core use case

### Pilot 3

Large mixed-task inventory

Purpose: test region summaries / candidate-space maps / HTML report

That will tell you a lot very quickly.

---

## 16. Bottom line

The point of this CPU-only field trial plan is not to do more abstract thinking about Gradience.

It is to answer, in practice:

> Now that Gradience is built, does it behave like a useful preflight decision system on real small-model adapter inventories, and where does it still need refinement?

That is the highest-value way to use your CPU time right now.
