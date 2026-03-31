# Checkpoint Triage Alpha

**A turnkey workflow for triaging fine-tuned checkpoint compatibility.**

Given a set of fine-tuned checkpoints sharing a base model, this workflow tells you which pairs are worth exploring, which are blocked by weak evidence, and which are cross-task noise — before you spend compute on merging or evaluation.

---

## What this workflow is for

You have several checkpoints fine-tuned from the same base model. You want to know:

- **Which checkpoints actually learned something?** (Evidence bootstrap gates out checkpoints that don't meaningfully beat the base model.)
- **Which pairs might be compatible?** (Pairwise structural comparison ranks candidates by risk.)
- **What should I do first?** (The action plan tells you: retain these, investigate these near-misses, skip these.)

This is a **triage** workflow, not a merge workflow. It narrows your candidate space (typically by 90%+) so you spend evaluation time on the pairs most likely to matter. It does not execute merges.

### Scope contract

This alpha workflow is validated under these conditions:

| Constraint | Value |
|-----------|-------|
| Base model | Shared (all checkpoints from the same pretrained model) |
| Model size | Small encoders (distilbert-class, ~66M parameters) |
| Task type | Classification only |
| Evidence | Bootstrap evaluation required before triage decisions |

If any constraint is not met, treat results as experimental.

---

## How to run it

### Prerequisites

```bash
pip install gradience[hf]    # Needs HuggingFace integration
```

### Quick start (canonical example)

```bash
# Step 1: Run the full trial (evidence bootstrap + QA + pairwise + action plan)
python3 field_trials/checkpoint_inventory_t02/run_trial.py

# Step 2: Build the polished report bundle
python3 field_trials/checkpoint_inventory_t02/build_alpha_bundle.py

# Step 3: Open the report
open field_trials/checkpoint_inventory_t02/preflight/alpha_bundle/report.html
```

Step 1 takes 2–5 minutes on CPU. Step 2 takes <1 second.

### What Step 1 does (in order)

1. **Evidence bootstrap** — Evaluates each checkpoint on 200 sample examples from its training task. Compares to the base model. Classifies as `eligible`, `uncertain`, or `flagged_weak`. This is a first-class gate: weak checkpoints are excluded from merge consideration.

2. **QA artifact generation** — Builds a structured eligibility record for each checkpoint, combining evidence status with structural flags.

3. **Pairwise compatibility** — Computes structural similarity between every checkpoint pair using layer-level spectral features. Assigns `pair_risk` (low / medium / high) and a recommended merge strategy.

4. **Inventory action plan** — Aggregates all QA and pairwise data into a single triage decision: which pairs to retain, which are near-misses worth revisiting, and which to exclude.

5. **Follow-through evaluation** — Re-evaluates a small probe set on a different random sample (seed 123, 300 examples) to confirm that evidence bootstrap results are stable.

### Command-line options

```bash
python3 run_trial.py \
    --base-model distilbert-base-uncased \
    --evidence-sample-size 200 \
    --evidence-seed 42 \
    --follow-through-sample-size 300 \
    --follow-through-seed 123 \
    --margin 0.01
```

---

## What the outputs mean

### The HTML report (`preflight/alpha_bundle/report.html`)

A single-page report with five sections, one per workflow step:

| Section | What it shows |
|---------|---------------|
| **Hero** | Inventory type, dominant driver, checkpoint/pair counts |
| **Evidence bootstrap** | Per-checkpoint scores, base comparison, eligibility status |
| **QA snapshot** | Eligibility status distribution across the inventory |
| **Pairwise compatibility** | Per-pair task relationship, compatibility score, risk level |
| **Action plan** | Retained candidates, near-miss candidates, exclusions |
| **Follow-through** | Probe results confirming triage stability |

The scope contract (alpha constraints) is displayed prominently as pill badges.

### The action plan

The action plan is the operational output. It routes every pair into one of these buckets:

| Bucket | Meaning | What to do |
|--------|---------|------------|
| **Retained** | Passed both QA and structural triage | Safe to evaluate further |
| **Near-miss** | Structurally plausible, but evidence is weak | Fix the evidence gap first (re-evaluate the source checkpoint) |
| **Same-task priority** | Same task, compatible — highest-value candidates | Evaluate first |
| **Cross-task caution** | Different tasks, proceed with care | Only if you have a specific use case |
| **Exclude** | Weak evidence or high structural risk | Skip |

### Key terms in the output

| Term | Meaning |
|------|---------|
| `inventory_type` | `mixed_quality` = some eligible, some weak. `high_quality` = all eligible. |
| `dominant_driver` | What controls triage. `source_qa` = evidence quality is the bottleneck. `structural` = compatibility scores matter more. |
| `exploration_posture` | `narrow` = conservative, few candidates. `moderate` = some exploration room. |
| `pair_risk` | `low` = high compatibility + high cosine. `medium` = decent compatibility. `high` = structural concern. |
| `near_miss_severity` | How close a near-miss is to eligibility: `marginal` (almost there), `moderate` (fixable), `substantial` (far). |
| `candidate_reduction` | Fraction of pairs filtered out. 1.0 = all filtered (evidence-dominated). 0.9 = 90% filtered. |

### Same-family detection

Checkpoints on different datasets from the same task family (e.g., SST-2 and Yelp Polarity are both binary sentiment) are automatically detected and routed to the `same_task_priority` bucket rather than `cross_task_caution`. The task-family registry is static and conservative — only empirically validated families are included.

---

## Example inventory (canonical T02 instance)

This trial uses 5 distilbert-base-uncased checkpoints:

| Checkpoint | Task | Dataset | Seed | Role |
|-----------|------|---------|------|------|
| `sst2_s42` | Sentiment | SST-2 | 42 | Primary sentiment |
| `sst2_s123` | Sentiment | SST-2 | 123 | Same-task seed variation |
| `yelp_s42` | Sentiment | Yelp Polarity | 42 | Same-family probe |
| `mrpc_s42` | Paraphrase | MRPC | 42 | Cross-task |
| `qnli_s42` | NLI | QNLI | 42 | Cross-task |

### What happened

- **Evidence bootstrap**: 2 eligible (yelp_s42, mrpc_s42), 3 flagged_weak (sst2_s42, sst2_s123, qnli_s42). The SST-2 checkpoints performed near base level on the 200-example sample.
- **Dominant driver**: `source_qa` — evidence quality is the binding constraint, not structural compatibility.
- **Retained pairs**: 0 — no pairs passed both QA and structural gates.
- **Near-miss**: 1 pair (sst2_s42 x sst2_s123) — structurally plausible same-task pair, but both sources are evidence-weak.
- **Candidate reduction**: 100% — all pairs filtered. The right answer for this inventory: don't merge until you fix the evidence.

### What this demonstrates

The workflow correctly identified that this inventory is evidence-dominated. Despite structural compatibility in several pairs, the lack of reliable evidence made triage conservative. The near-miss identification correctly flagged the one pair worth revisiting if the evidence improves. Same-family routing correctly grouped SST-2 x Yelp as sentiment rather than cross-task.

---

## Output file reference

All outputs live under `field_trials/checkpoint_inventory_t02/`:

```
.
├── run_trial.py                       # Run this first
├── build_alpha_bundle.py              # Then this
├── manifest.json                      # Checkpoint catalog
│
├── evidence/
│   └── bootstrap_results.json         # Evidence gate (per-checkpoint scores)
│
├── qa_artifacts/
│   └── qa_summary.json                # QA eligibility rollup
│
├── pairwise/
│   └── pairwise_results.json          # Structural compatibility (all pairs)
│
├── preflight/
│   ├── inventory/
│   │   ├── inventory_summary.json     # Policy + structural detail
│   │   ├── inventory_summary.md       # Human-readable version
│   │   ├── inventory_action_plan.json # The operational triage output
│   │   └── inventory_action_plan.md   # Human-readable version
│   │
│   ├── qa/                            # Individual QA artifacts (per checkpoint)
│   ├── pair_reports/                  # Merge QA reports (per pair)
│   │
│   ├── run_001/                       # Preflight run bundle
│   │   ├── preflight_summary.json
│   │   ├── preflight_summary.md
│   │   ├── run_manifest.json
│   │   └── review_packet.md
│   │
│   └── alpha_bundle/                  # Polished demo package
│       ├── report.html                # <-- Open this
│       ├── alpha_summary.json         # Compact metadata
│       └── bundle_manifest.json       # File manifest
│
├── eval_results.json                  # Follow-through evaluation
├── trial_memo.md                      # Design decisions
└── field_note.md                      # Product summary
```

---

## Adapting for your own checkpoints

To triage your own checkpoint inventory:

1. **Edit the checkpoint specs** at the top of `run_trial.py`. Each spec needs:
   - `checkpoint_id`: unique label
   - `path`: path to the saved checkpoint directory
   - `task` and `dataset`: for evidence bootstrap evaluation
   - `metric_name`: what to measure (e.g., `"accuracy"`)

2. **Ensure all checkpoints share a base model.** Set `--base-model` accordingly.

3. **Run the trial.** The evidence bootstrap will evaluate each checkpoint against the base model on its declared dataset.

4. **Read the action plan**, not just the compatibility scores. The action plan integrates evidence quality, structural compatibility, and task relationships into a single triage decision.

### What to expect

- If most checkpoints are well-trained: the action plan will retain several pairs and the dominant driver will be `structural`.
- If evidence is mixed: the action plan will be conservative and the dominant driver will be `source_qa`. This is correct behavior — don't merge when you can't trust the sources.
- Cross-task pairs will almost always land in `cross_task_caution` or `exclude`. This is by design.

---

## Further reading

- [Checkpoint triage alpha workflow](../../docs/examples/checkpoint-triage-alpha-workflow.md) — Concise workflow shape and scope contract
- [Route 2 packet](../../sidecar/packet/route2/00_route2_packet_index.md) — Research context for why this workflow exists
- [Project map](../../docs/project-map.md) — Where this fits in the overall Gradience project
