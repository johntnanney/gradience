# CASE_STUDY_INVENTORY_PROTOCOL

## Purpose

This protocol defines how to select adapters and run a single end-to-end worked example for Gradience.

The goal is not broad benchmarking.
The goal is to produce one concrete, practitioner-legible inventory walkthrough that shows:
- single-adapter QA
- pairwise merge-risk reporting
- one ambiguous pair with optional core-space audit
- merge neighborhoods
- final inventory summary
- one explicit "what decision changed" conclusion

This protocol is for case study generation, not threshold calibration or new feature development.

---

## Objective

Produce one worked example that demonstrates the practical value of the Gradience workflow on a small real adapter inventory.

The final case should show, at minimum:
1. one adapter that is excluded or downgraded by QA
2. one pair that looks straightforward under ordinary pair-risk logic
3. one pair where deeper inspection (core-space) adds meaningful caution or clarification
4. one neighborhood structure that is more informative than a flat merge pool
5. one concise final decision that is narrower than the naive "try everything" approach

---

## Constraints

Do not do:
- no threshold changes
- no merge logic changes
- no new features
- no synthetic/random-only inventories
- no giant benchmark
- no graph UI
- no more than 5 adapters in the final case inventory
- no more than 10 pair reports in the final write-up artifact set

Keep fixed:
- current default workflow
- strict-QA semantics
- current advanced feature behavior
- corpus append rules

---

## Inventory selection requirements

### Target inventory size

Select 4 or 5 adapters total.

This is the best range because it is:
- large enough to show inventory structure
- small enough to explain clearly
- small enough to run and inspect manually

### Required inventory composition

The selected inventory must include adapters that plausibly occupy the following roles:

#### Role A - strong anchor adapter

One adapter that appears behaviorally credible and structurally sane enough to serve as an anchor.

Desired characteristics:
- likely eligible
- not obviously weak
- not trivially incompatible with everything
- likely to survive into the final candidate set

#### Role B - related secondary adapter

One adapter that is plausibly mergeable with the anchor.

Desired characteristics:
- same or adjacent task family, or otherwise structurally plausible
- not obviously dominated by extreme norm imbalance
- likely candidate for same-neighborhood membership

#### Role C - weak or exclusion-worthy adapter

One adapter that has a realistic chance of being:
- flagged_weak
- unknown_no_behavioral_eval under strict interpretation
- or otherwise plausibly excluded

This role is important because the case study should show that early exclusion changes the pool.

#### Role D - ambiguous or structurally non-obvious adapter

One adapter that creates at least one interesting ambiguous pair.

Desired characteristics:
- not trivially bad
- not trivially same-task redundant
- not purely random
- capable of generating a pair that looks plausible under ordinary pair-risk but worth deeper inspection via core-space

#### Optional Role E - boundary / cross-neighborhood adapter

If using 5 adapters, include one adapter likely to:
- remain a singleton
- form a caution neighborhood
- or generate a meaningful boundary warning

---

## Selection heuristics

Coding agents should prefer adapters that satisfy all of the following:
- real adapters, not synthetic fixtures
- stable local paths or reproducible sources
- interpretable names and provenance
- mix of behavioral quality and structural relationships
- at least one pair likely to be low/medium risk but nontrivial
- at least one pair likely to produce an interesting neighborhood boundary

Avoid inventories dominated by:
- repeated checkpoint aliases with no identity clarity
- adapters that are all obviously weak
- adapters that are all near-duplicates
- inventories where every pair is trivially blocked
- inventories where every pair is equally benign and uninteresting

---

## Pre-screening step

Before finalizing the inventory, agents should perform a light pre-screen.

### Required pre-screen outputs

For each candidate adapter, record:
- adapter name
- source path
- base model
- nominal rank
- any already-known evaluation context
- reason for inclusion in the candidate pool

### Deliverable

Create a short candidate note at:
- `docs/internal/case_study_inventory_candidates.md`

Include:
- shortlisted adapters
- intended role assignment (A/B/C/D/E)
- why each is promising for the case study

This note can be short. It is just to prevent arbitrary selection.

Template:
- `docs/internal/templates/case-study-inventory-candidates-template.md`

---

## Execution workflow

Once the final 4-5 adapters are selected, run the following pipeline.

### Step 1 - Single-adapter QA

Run QA for every selected adapter.

Required command shape:

```bash
gradience audit-adapter --peft-dir <adapter_path> --out <qa_output>
```

Required outputs:

Save all QA artifacts under:
- `results/case_study/<inventory_id>/qa/`

Acceptance condition:

Every selected adapter must have:
- a valid QA artifact
- clear eligibility status
- a one-line interpretation note for later narrative use

---

### Step 2 - Pairwise merge-risk reports

Run pairwise reports for all non-excluded candidate pairs.

If inventory has 4 adapters:
- maximum 6 pairs

If inventory has 5 adapters:
- maximum 10 pairs

Required command shape:

```bash
gradience merge-audit \
  --adapter-a <a> \
  --adapter-b <b> \
  --source-a-qa <qa_a> \
  --source-b-qa <qa_b> \
  --emit-report <report_out>
```

Required outputs:

Save under:
- `results/case_study/<inventory_id>/pair_reports/`

Acceptance condition:

Each pair report must have:
- valid schema
- clear pair risk
- dominant issue
- recommended strategy

---

### Step 3 - Identify the ambiguous pair

After pair reports are generated, agents must choose exactly one pair for deeper inspection.

Selection criteria:

Choose the pair that best satisfies:
- ordinary pair-risk report is not already obviously decisive
- pair remains practically plausible
- there is genuine uncertainty about whether it belongs in the candidate set
- deeper structural inspection could change judgment

Do not choose:
- the most obviously bad pair
- a purely synthetic mismatch
- the most trivial same-neighborhood pair

Deliverable:

Record the chosen pair and why in:
- `docs/internal/case_study_ambiguous_pair_note.md`

Keep it short:
- pair name
- why it is ambiguous
- why core-space is justified here

Template:
- `docs/internal/templates/case-study-ambiguous-pair-note-template.md`

---

### Step 4 - Run core-space on the ambiguous pair only

Run the deeper structural diagnostic only on the selected ambiguous pair.

Required command shape:

```bash
gradience merge-audit \
  --adapter-a <a> \
  --adapter-b <b> \
  --source-a-qa <qa_a> \
  --source-b-qa <qa_b> \
  --compute-core-space \
  --emit-report <report_out>
```

Required outputs:

Save under:
- `results/case_study/<inventory_id>/core_space/`

Acceptance condition:

The ambiguous pair should now have:
- ordinary pair-risk interpretation
- core-space interpretation
- a documented statement of whether judgment changed

If core-space adds nothing useful, that must be stated explicitly.

---

### Step 5 - Inventory summary

Run inventory summary across the selected inventory.

Required command:

```bash
gradience summarize-inventory \
  --qa-dir <qa_dir> \
  --report-dir <report_dir> \
  --emit-report <inventory_summary_out>
```

Output:

Save under:
- `results/case_study/<inventory_id>/inventory/`

---

### Step 6 - Neighborhood suggestion

Run neighborhoods on the full inventory.

Required command:

```bash
gradience suggest-neighborhoods \
  --qa-dir <qa_dir> \
  --report-dir <report_dir> \
  --emit-report <neighborhood_report_out>
```

Output:

Save under:
- `results/case_study/<inventory_id>/neighborhoods/`

Acceptance condition:

The neighborhood output should ideally show at least one of:
- one excluded adapter
- one nontrivial group of size > 1
- one meaningful boundary warning
- one singleton/caution case that helps clarify the inventory

If neighborhoods are totally trivial, agents should note that the inventory may not be strong enough for publication as the worked example.

---

### Step 7 - Corpus append

Append the final case inventory to the corpus.

Required command:

Use the existing append script:

```bash
python3 scripts/append_corpus_entry.py ...
```

Output:

Save manifest under:
- `results/corpus/manifests/<case_study_manifest>.json`

This keeps the case study inside the same review ecosystem as the rest of the project.

---

## Output bundle structure

Use this structure:

```text
results/case_study/<inventory_id>/
  qa/
  pair_reports/
  core_space/
  inventory/
  neighborhoods/
  notes/
```

Required note files under `notes/`:
- `inventory_selection_note.md`
- `ambiguous_pair_note.md`
- `final_decision_note.md`

Reference template:
- `docs/internal/templates/case-study-final-decision-note-template.md`

---

## Final decision note requirement

Agents must produce one short note explaining:

What decision changed because of the workflow?

File:
- `results/case_study/<inventory_id>/notes/final_decision_note.md`

This note must answer:
- which adapter was excluded or downgraded?
- which pair was narrowed, deferred, or treated with extra caution?
- what neighborhood structure mattered?
- how is the final next step narrower than the naive "try everything" approach?

This is the most important output in the whole protocol.

---

## Selection success criteria

An inventory is suitable as the published worked example only if all of the following are true:
- 4-5 real adapters selected
- at least one adapter is excluded or clearly downgraded by QA
- at least one ambiguous pair justifies core-space
- core-space either changes judgment or clearly clarifies why it does not
- neighborhood output is nontrivial and interpretable
- final decision note shows a narrower, better-justified action set than the naive baseline

If these are not met, agents should reject the inventory and choose another candidate set.

---

## Preferred final deliverables

Agents should produce:

1. Execution bundle

Under:
- `results/case_study/<inventory_id>/...`

2. Selection note
- `docs/internal/case_study_inventory_candidates.md`

3. Ambiguous-pair note
- `docs/internal/case_study_ambiguous_pair_note.md`

4. Publication-ready summary note
- `docs/internal/case_study_publication_summary.md`

Use template:
- `docs/internal/templates/case-study-publication-summary-template.md`

This final summary should be 1-2 pages and include:
- inventory composition
- QA outcome
- pairwise outcome
- core-space outcome
- neighborhood outcome
- final decision changed by the workflow

That note will make it much easier to turn the result into:
- a HF case study post
- a README advanced example
- a paper supplement or appendix example

---

## Guardrails

Agents should not:
- retune thresholds to make the case study cleaner
- choose synthetic cases just because they "look good"
- overload the study with too many adapters
- write a benchmark instead of a walkthrough
- treat core-space as mandatory on every pair
- treat neighborhoods as a graph-discovery exercise

The purpose is:
- one small real inventory
- one worked decision process
- one clear practical lesson

---

## Bottom line

The case study should demonstrate:
- why source QA matters first
- why pairwise structural risk is not the whole story
- why one ambiguous pair may deserve deeper inspection
- why the inventory stops being a flat merge pool
- what concrete decision became narrower and better because the workflow was run

That is the standard the published walkthrough should meet.
