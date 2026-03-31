# CPU-Only Field Research Protocol

**Targeted Micro-Campaigns for Unresolved Product Questions in Gradience**

---

## Purpose

This protocol is designed to extend field testing without repeating broad validation work that is already close to saturated.

Its goal is to answer the remaining practical product questions that still matter after the first field-trial phases:

- Is exact task identity too strict a boundary in practice?
- How should Gradience treat marginal adapters that are barely weak?
- Does large-inventory mode remain usable at somewhat higher density?
- How robust is the workflow to messy public adapter ecosystem conditions?

This protocol is CPU-only and assumes: small encoder adapters, classification tasks, modest evaluation budgets, existing Gradience workflow and reports.

---

## 1. Research goals

This protocol is meant to answer four product-facing questions.

**Goal A — Task-family equivalence.** Test whether Gradience's current strict task-identity logic is appropriately conservative or too restrictive for practically similar task families. Examples: IMDB vs SST-2, SST-2 vs Yelp/Amazon, related classification tasks with similar decision structure.

**Goal B — Marginal-adapter behavior.** Test whether adapters with small but non-zero weakness behave more like valid near-miss candidates or genuinely weak sources that should remain excluded.

**Goal C — Larger-inventory stress.** Test whether current large-inventory ergonomics remain readable and useful on somewhat denser pools: region summaries, candidate-space maps, action plans, review packets, HTML rendering, portfolio view.

**Goal D — Public-ecosystem robustness.** Test whether Gradience continues to behave sensibly under real-world public-adapter messiness: incomplete metadata, unusual LoRA configs, weak evidence, partial compatibility issues, transfer-chain base models.

---

## 2. High-level design

This is not a new broad field-trial phase. It is a set of four targeted micro-campaigns, each with a narrow question, 2–4 inventories, a small evaluation budget, and one short memo at the end. Each micro-campaign can be run independently, but they should use the same artifact conventions and reporting structure.

---

## 3. Shared inclusion criteria

All inventories in this protocol should satisfy:

- Small encoder models only
- Classification tasks only
- Adapters are loadable or at least diagnosable
- Evaluation dataset is available
- Behavioral evidence can be bootstrapped on CPU
- Inventories are small enough to evaluate a few merges without GPU

Preferred backbones: DistilBERT, RoBERTa-base, BERT-base.

Avoid: decoder models, generation-heavy tasks, large-scale retraining, inventories requiring large benchmark sweeps.

---

## 4. Shared workflow per inventory

Every inventory in every micro-campaign should follow the same sequence.

**Step 1 — Build manifest.** Record: adapter IDs, task labels, backbone, LoRA config, known metadata issues, expected inventory type.

**Step 2 — Evidence bootstrap.** Run a small CPU eval for each adapter: 100–500 examples, one primary metric, record adapter score, base score, delta, evidence status.

**Step 3 — Run Gradience preflight.** Generate: JSON summary, markdown summary, HTML report, review packet, portfolio entry, drift output if repeated.

**Step 4 — Lock Gradience stance.** Before evaluating merges, record: retained pairs, near-miss pairs, excluded pairs, action plan, policy summary, trust snapshot, region summary if applicable.

**Step 5 — Evaluate a small pair subset.** Evaluate: retained pairs, near-miss pairs where relevant, excluded controls, any specially targeted contrast pair for the micro-campaign.

**Step 6 — Write field note.** Each inventory gets a short note: question being tested, Gradience stance, evaluation outcomes, what this inventory suggests about the micro-campaign question.

---

## 5. Shared evaluation rules

Keep the merge budget small.

**Default per-inventory budget:** 2–5 merges total. Suggested split: 1–2 retained pairs, 1–2 near-miss or excluded controls, 1 targeted contrast if needed.

**Output metrics.** For each evaluated merge: merged score, best-source score, delta vs best source, delta vs average source, category (retained / near-miss / excluded control / targeted contrast).

---

## 6. Micro-Campaign A — Task-Family Equivalence

**Central question:** Is exact task identity too strict, or is it the right conservative boundary for practical preflight use?

**Inventory types.** Use 2–4 inventories built around: same-label but different-domain sentiment tasks, related classification tasks with similar surface structure, exact same-task controls for comparison. Examples: SST-2 + Yelp + Amazon, IMDB + SST-2, related NLI-like or pair-classification tasks if available.

**Required pair classes.** Per inventory, try to include: exact same-task retained pair, same-family but non-identical-task pair, clearly cross-task control.

**Key measures.** Do same-family non-identical-task pairs behave more like retained pairs or excluded controls? Does Gradience's strict task boundary look too conservative? Does "family similarity" help or mostly create false hope?

**Success criteria.** This micro-campaign is informative if it can answer: exact task identity is the right practical boundary, or a small controlled notion of task-family equivalence may be worth considering.

**Deliverables:** `field_trials/task_family_equivalence/` and one memo: `task_family_equivalence_memo.md`.

---

## 7. Micro-Campaign B — Marginal-Adapter Behavior

**Central question:** Do adapters that are only barely weak behave more like near-miss candidates or like truly weak/excluded sources?

**Inventory types.** Use 2–4 inventories containing: one or more adapters with small negative deltas, at least one structurally plausible same-task pair involving a marginal source.

**Pair classes to emphasize.** Retained same-task pair, near-miss with barely weak source, near-miss with deeply weak source, excluded control. Separate near-miss cases into barely weak and deeply weak — this split is important.

**Key measures.** Average delta vs best source by weak-source band, variance of merge outcome by weak-source band, whether barely weak near-miss cases behave like retained pairs.

**Success criteria.** This micro-campaign is informative if it can answer: whether the current near-miss treatment is enough, whether "barely weak" deserves distinct handling, whether deeply weak sources still belong in the same near-miss bucket.

**Deliverables:** `field_trials/marginal_adapter_behavior/` and one memo: `marginal_adapter_behavior_memo.md`.

---

## 8. Micro-Campaign C — Large-Inventory Stress Test

**Central question:** Do Gradience's large-inventory ergonomics remain useful as inventory density increases modestly beyond the first validated example?

**Inventory types.** Use 2–3 inventories with 10–14 adapters or 40–90 pairs. Still CPU-friendly, still classification-only.

**What to test.** Focus less on merge outcomes and more on usability: region summary clarity, candidate-space map readability, action plan usefulness, HTML report scanability, review packet usefulness, portfolio prioritization value.

**Light evaluation component.** Do a very small merge follow-through: 1–2 retained pairs, 1 excluded control. Just enough to keep the outputs grounded.

**Key measures.** Qualitative: was the region summary useful, was the candidate-space map actually helpful, did the report still feel legible. Quantitative: candidate reduction, number of regions, retained count, time-to-understand inventory.

**Success criteria.** This micro-campaign is informative if it can answer: whether current large-inventory mode scales gracefully, where presentation starts to strain, what table/report elements need simplification at higher density.

**Deliverables:** `field_trials/large_inventory_stress/` and one memo: `large_inventory_stress_memo.md`.

---

## 9. Micro-Campaign D — Public-Ecosystem Robustness

**Central question:** How robust is Gradience to the real-world messiness of public adapters?

**Inventory types.** Use 2–4 inventories chosen specifically for ecosystem variety: unusual target modules, transfer-chain bases, sparse metadata, rank-1 LoRA, odd PEFT config behavior, partially loadable or brittle adapters.

**What to test.** Manifest construction effort, evidence bootstrap friction, loadability issues, how often structural analysis still runs, how gracefully Gradience reports uncertainty.

**Merge evaluation.** Minimal. Only evaluate if the adapters are clean enough after bootstrap. The real target here is ecosystem robustness, not merge quality.

**Key measures.** Load success rate, evidence bootstrap success rate, number of adapters requiring manual intervention, whether reporting remains honest and readable, whether the workflow fails gracefully.

**Success criteria.** This micro-campaign is informative if it can answer: whether Gradience is robust enough for public-adapter use, what kinds of adapter weirdness need better handling, whether ecosystem friction is now mostly operational rather than conceptual.

**Deliverables:** `field_trials/public_ecosystem_robustness/` and one memo: `public_ecosystem_robustness_memo.md`.

---

## 10. Shared measurement framework

Across all micro-campaigns, record the following:

**Product-behavior measures:** candidate reduction, retained count, near-miss count, excluded count, evidence profile, dominant driver, exploration posture.

**Decision-quality measures:** retained average delta vs best source, near-miss average delta vs best source, excluded control average delta vs best source.

**Workflow-usability measures.** Qualitative rating (high / medium / low) for: HTML report usefulness, review packet usefulness, policy summary usefulness, action plan usefulness, large-inventory region summary usefulness, portfolio usefulness.

**Robustness measures:** adapter load failures, bootstrap failures, manual intervention count, metadata ambiguity count.

---

## 11. Shared output structure

Use a common layout:

```
field_trials/
  task_family_equivalence/
    inventory_01/
    inventory_02/
    memo.md
  marginal_adapter_behavior/
    inventory_01/
    inventory_02/
    memo.md
  large_inventory_stress/
    inventory_01/
    inventory_02/
    memo.md
  public_ecosystem_robustness/
    inventory_01/
    inventory_02/
    memo.md
  synthesis/
    cross_campaign_summary.md
    product_implications.md
```

---

## 12. Final synthesis step

After all micro-campaigns, write two summary documents.

**A. Cross-campaign summary.** Should answer: what was confirmed, what remains ambiguous, which product questions are now settled, which are still open.

**B. Product implications memo.** Should rank: immediate product changes worth making, things that are confirmed good enough already, things that should wait for more evidence.

---

## 13. Suggested sequencing

Do not run all four at once. Recommended order:

**First — Task-family equivalence.** Most interesting unresolved product question.

**Second — Marginal-adapter behavior.** Most actionable if task-family equivalence stays ambiguous.

**Third — Public-ecosystem robustness.** Good operational hardening step.

**Fourth — Large-inventory stress.** Only after the first three, unless you have a particularly good inventory ready.

---

## 14. Stop conditions

You do not need to complete all four if the first two already answer the most important product questions. Stop early if: task-family equivalence clearly does not help, marginal-adapter behavior is already sufficiently resolved, robustness issues are mostly known and repetitive, additional inventories are adding volume but not insight.

---

## 15. Bottom line

This protocol is meant to extend CPU-only field testing without repeating broad validation work. Its purpose is to answer the remaining practical questions: Is exact task identity too strict? How should marginal adapters be treated? Does large-inventory mode keep working as scale rises? How robust is Gradience to the public adapter ecosystem?

The aim is not more field testing for its own sake. The aim is to close the remaining product-facing unknowns with small, targeted, CPU-friendly campaigns.
