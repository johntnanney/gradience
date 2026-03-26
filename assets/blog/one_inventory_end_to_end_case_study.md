# One Inventory, End to End  
### From adapter pool to merge neighborhoods with a Gradience preflight pass

A flat inventory of four adapters and six plausible pair combinations looked like it needed broad exploration.

After one preflight pass, that action space collapsed to:

- one source excluded early
- one pair moved to a caution track
- one local neighborhood prioritized as the best next exploration target

This post walks through that pass end to end.

## The starting pool

This inventory used four QNLI-oriented adapters on `distilbert-base-uncased`:

- `final_uniform_median_r16`
- `qnli_probe_elig`
- `qnli_uniform_weak`
- `qnli_per_layer_elig`

At first glance, all four looked plausible enough to keep in play. The easy move would have been to treat the pool as a flat menu of pairings and start trying merges.

The point of the workflow is to make that decision surface smaller before evaluation gets expensive.

## Step 1: source QA narrows the pool

The first pass is single-adapter QA.

```bash
gradience audit-adapter --adapter ./adapters/qnli_uniform_weak --out ./qa/qnli_uniform_weak.json
```

Here are the key QA outcomes for the full pool:

- `final_uniform_median_r16` — `eligible`
- `qnli_probe_elig` — `eligible`
- `qnli_per_layer_elig` — `eligible`
- `qnli_uniform_weak` — `flagged_weak`

A compact version of the weak-source result looked like this:

```json
{
  "adapter": {"name": "qnli_uniform_weak"},
  "eligibility": {
    "status": "flagged_weak",
    "confidence": "high"
  },
  "behavioral_summary": {
    "eval_available": true,
    "metric_name": "accuracy"
  }
}
```

That changed the candidate set immediately.

Before QA, all four adapters sat in the pool as plausible merge sources. After QA, `qnli_uniform_weak` was no longer part of the real candidate set.

That is the first practical lesson of the example:

**source QA changes the merge problem before pairwise logic even begins.**

## Step 2: pairwise audit makes the inventory less flat

Once the weakest source is removed, the next pass is pairwise structural risk.

```bash
gradience merge-audit \
  --adapter-a ./adapters/qnli_probe_elig \
  --adapter-b ./adapters/qnli_per_layer_elig \
  --source-a-qa ./qa/qnli_probe_elig.json \
  --source-b-qa ./qa/qnli_per_layer_elig.json \
  --emit-report ./reports/qnli_probe_x_per_layer.json
```

A compact view of the pair set looked like this:

| Pair | Pair risk | Dominant issue | Strategy |
|---|---|---|---|
| `final_uniform_median_r16 × qnli_probe_elig` | `low` | `none` | `linear` |
| `final_uniform_median_r16 × qnli_per_layer_elig` | `low` | `none` | `linear` |
| `qnli_probe_elig × qnli_per_layer_elig` | `low` | `none` | `linear` |
| `final_uniform_median_r16 × qnli_uniform_weak` | `medium` | `partial_redundancy` | `norm_equalized` |
| `qnli_probe_elig × qnli_uniform_weak` | `low` | `none` | `linear` |
| `qnli_uniform_weak × qnli_per_layer_elig` | `medium` | `partial_redundancy` | `audit_aware` |

Two things are already clear here:

- the inventory is no longer flat
- `qnli_uniform_weak` is dragging several pairs into less trustworthy territory simply by remaining in the pool

Even before any advanced diagnostic is used, the pairwise pass is already separating:
- more natural local combinations
- weaker, less trustworthy combinations
- one pair that still looks ordinary enough to deserve closer inspection

## Step 3: one pair remains genuinely ambiguous

Among the remaining pairs, one stood out because it did **not** look especially alarming under the ordinary pair-risk report:

- `qnli_probe_elig × qnli_per_layer_elig`

Under the standard report, it looked like a plausible default candidate:

- pair risk: `low`
- dominant issue: `none`
- recommended strategy: `linear`

That is exactly the kind of pair that matters.

Not because it is dramatic, but because it is uncertain. It still looks like something one might reasonably try, which means it is the right place to ask whether deeper inspection changes anything real.

## Step 4: deeper inspection changes the local judgment

For that one ambiguous pair, I ran the deeper structural audit.

```bash
gradience merge-audit \
  --adapter-a ./adapters/qnli_probe_elig \
  --adapter-b ./adapters/qnli_per_layer_elig \
  --source-a-qa ./qa/qnli_probe_elig.json \
  --source-b-qa ./qa/qnli_per_layer_elig.json \
  --compute-core-space \
  --emit-report ./reports/qnli_probe_x_per_layer_core_space.json
```

The ordinary pair-risk report said “low risk / linear.” The deeper structural result looked like this:

```json
{
  "core_space": {
    "shared_basis_score": 0.9078,
    "basis_distortion": 0.00346,
    "effective_shared_rank": 22,
    "status": "incompatible"
  }
}
```

That changed the local judgment.

The pair did not move from “good” to “catastrophic.” It moved from:

- **plausible default candidate**

to:

- **caution / inspect further / not part of the easy path**

That is exactly the role this advanced diagnostic should play.

Not replacing the default path.  
Not firing on every pair.  
Just changing a narrower class of decisions that would otherwise remain too permissive.

That gives us the second practical lesson of the example:

**a deeper diagnostic is useful when it changes a real judgment in a narrower class of ambiguous cases.**

In this case, it did.

## Step 5: the inventory resolves into neighborhoods

At this point, the inventory has already changed shape.

- one weak adapter is out
- one ambiguous pair has moved to a caution track
- the remaining candidate set is no longer flat

Now the neighborhood pass becomes useful.

```bash
gradience suggest-neighborhoods \
  --qa-dir ./qa \
  --report-dir ./reports \
  --emit-report ./inventory/neighborhoods.json
```

The resulting neighborhood output was simple:

```json
{
  "groups": [
    {
      "members": [
        "final_uniform_median_r16",
        "qnli_probe_elig",
        "qnli_per_layer_elig"
      ],
      "characterization": "likely-safe neighborhood",
      "common_strategy": "linear"
    }
  ],
  "excluded": [
    {
      "adapter": "qnli_uniform_weak",
      "reason": "flagged_weak"
    }
  ],
  "boundary_warnings": []
}
```

That is enough.

The inventory no longer looks like one undifferentiated merge pool. It now has:

- one likely-safe local region worth exploring first
- one weak source explicitly removed from the active plan

That is the third practical lesson of the example:

**the value of neighborhoods is turning pairwise clutter into an intelligible local decision surface.**

## What changed

Before the workflow, the inventory looked like:

- four plausible adapters
- six pair combinations worth trying
- no strong reason not to explore the pool broadly

After the workflow, the picture was narrower:

- `qnli_uniform_weak` was excluded early
- pairs containing that weak source dropped out of priority exploration
- `qnli_probe_elig × qnli_per_layer_elig` moved from “plausible” to “caution / inspect further”
- one local neighborhood emerged as the best next place to explore

That is the point of the case study.

The value of the workflow was not that it generated more reports. It was that it **reduced the action space**.

In this case, the workflow reduced the action space from a flat 6-pair exploration plan to a focused neighborhood-first plan with one source excluded early and one structurally ambiguous pair explicitly cautioned.

> ### Compact workflow view
>
> **Starting pool**  
> 4 plausible adapters
>
> **After QA**  
> `qnli_uniform_weak` excluded
>
> **After pair audit**  
> Most pairs look benign, but `qnli_probe_elig × qnli_per_layer_elig` remains ambiguous
>
> **After deeper inspection**  
> That pair moves to a caution track
>
> **After neighborhoods**  
> One local neighborhood becomes the best next exploration target
>
> **Net effect**  
> The workflow reduced the action space from a flat set of plausible combinations to a narrower, more defensible plan.

## Why this example matters

This is one worked example, not a general benchmark claim.

What it does show clearly is:

- source QA matters before pairwise merging
- pairwise structural risk is useful but not always sufficient
- one ambiguous pair may justify deeper inspection
- the inventory can have local structure that a flat merge view misses
- the final practical outcome can be a smaller, more defensible next-step set

## What this does not show

This is still a small, controlled, same-family inventory. It shows the workflow clearly, but it does not establish that every larger or messier adapter pool will resolve this cleanly. The point of the example is narrower: to show what changes when the workflow is actually run end to end.

## The workflow I trust most right now

If I had to compress the whole example into one sentence, it would be this:

> **screen sources first, audit pairs second, inspect one ambiguous pair more deeply if needed, then organize the inventory into conservative neighborhoods before spending more effort downstream.**

That is the value of the preflight pass in this case: not more scores, but a better next decision.
