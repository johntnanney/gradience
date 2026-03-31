# Demo Paths

**Last updated:** 2026-03-31

Three guided tours through the project, each for a different reader. Pick the one that matches what you want to understand. Each path takes 30–60 minutes of reading.

For orientation before starting any path, see the [project map](project-map.md). For what didn't work and what not to claim, see [boundaries and non-generalizations](boundaries-and-non-generalizations.md) — it complements every path.

---

## Path A — The Stable Product

**For:** Practitioners, potential users, workflow owners. "What does this tool do and should I use it?"

### Stops

**1. Product validation memo** (10 min)
[`docs/product-validation.md`](product-validation.md)

Start here. This establishes what Gradience actually does — not what it could do or what the research suggests, but what was tested in the field. The core finding: evidence bootstrap is a prerequisite, not optional enrichment. Without behavioral evidence, Gradience produces nothing useful. With it, you get zero false positives and 90–93% candidate reduction.

**2. Field trial index** (5 min)
[`field_trials/README.md`](../field_trials/README.md)

Skim the three-phase structure: Pilot (candidate reduction), Merge Evaluation (retained-vs-control outcomes), Near-Miss Confirmation (near-miss is behaviorally safe). Each phase built on the previous one. This is the empirical backbone.

**3. Example QA artifacts** (5 min)
[`examples/qa/`](../examples/qa/)

Open two or three of these JSON files. They are the atomic unit of the product: a single-adapter eligibility record. Look at how `eligible`, `uncertain`, `flagged_weak`, and `unknown_no_behavioral_eval` are used. Notice which fields are null when no eval data exists.

- `eligible_adapter_qa.json` — what a passing adapter looks like
- `uncertain_adapter_qa.json` — what uncertainty looks like
- `structural_only_qa.json` — what happens with no behavioral evidence

**4. Example merge reports** (5 min)
[`examples/reports/`](../examples/reports/)

The pairwise output. Open at least:

- `safe_merge_report.json` — low risk, linear strategy
- `high_risk_warn_report.json` — high risk, what triggers a warning
- `strict_blocked_report.json` — what --strict-qa blocks and why

**5. Example inventory summary** (5 min)
[`examples/inventory/realistic_inventory_summary.json`](../examples/inventory/realistic_inventory_summary.json)

The inventory-level aggregate. Notice how individual adapter QA and pairwise reports roll up into status counts, strategy recommendations, and strict-QA block candidates. This is the action-plan input.

**6. Retained-vs-control outcomes** (10 min)
[`field_trials/product_validation_memo.md`](../field_trials/product_validation_memo.md)

The product verdict. Retained same-task pairs average -0.024 degradation vs best source. Cross-task controls average -0.047. The evidence gate is the most impactful single feature. Near-miss pairs are indistinguishable from safe.

**7. Preflight example** (5 min)
[`examples/inventory_preflight_same_task_control/`](../examples/inventory_preflight_same_task_control/)

Walk through a concrete preflight bundle — QA artifacts, pair reports, action plan, summary. This is what a user gets at the end of the workflow.

### What you should know after this path

- How the evidence gate works and why it is the binding constraint
- What the QA, merge report, and inventory schemas look like
- What "zero false positives" means concretely (retained pairs are safe, near-misses are safe, cross-task is excluded)
- Whether this tool is useful for your workflow

---

## Path B — The Broadened Route 2 Workflow

**For:** ML engineers interested in extending beyond LoRA merge. "Can this work for my checkpoints / my routing problem / my PEFT variant?"

### Stops

**1. Broadened substrate scope** (10 min)
[`docs/strategy/broadened_substrate_scope.md`](strategy/broadened_substrate_scope.md)

The strategic framing. What has already been generalized (scenario, artifact class, representation path), what is bounded (scope contract), and what is still experimental. This sets expectations.

**2. Route 2 packet orientation** (10 min)
[`sidecar/packet/route2/01_route2_orientation.md`](../sidecar/packet/route2/01_route2_orientation.md)

The three generalization axes explained with concrete experiments. For each: what was tested, how much new code was needed, what generalized, what didn't. The table at the end is the clearest summary.

**3. Checkpoint triage alpha README** (10 min)
[`field_trials/checkpoint_inventory_t02/README.md`](../field_trials/checkpoint_inventory_t02/README.md)

The first working broadened workflow, packaged as a mini product. Read the "What this workflow is for" and "Example inventory" sections. If you want to run it yourself, follow the quickstart.

**4. Example HTML report** (5 min)
[`field_trials/checkpoint_inventory_t02/preflight/alpha_bundle/report.html`](../field_trials/checkpoint_inventory_t02/preflight/alpha_bundle/report.html)

Open in a browser. This is the output a checkpoint triage user would see. Five steps, one page. Notice how the evidence bootstrap dominates the T02 inventory — 3 of 5 checkpoints are flagged_weak, so no pairs are retained. This is the conservative correct answer.

**5. Trial memo** (5 min)
[`field_trials/checkpoint_inventory_t02/trial_memo.md`](../field_trials/checkpoint_inventory_t02/trial_memo.md)

Post-trial reflections: what transferred from adapter workflows, what was checkpoint-specific, how same-family detection worked on non-identical tasks. The honest account of where the alpha boundary is.

**6. Routing pilot results** (10 min)
[`docs/routing-pilot-results.md`](routing-pilot-results.md)

The substrate working in a completely different scenario. ~370 lines of new code, zero core changes, same spectral functions called identically. The seam between merge and routing is at the policy layer, not the measurement layer.

**7. Behavioral Route 2 summary** (10 min)
[`docs/strategy/behavioral_route2_summary.md`](strategy/behavioral_route2_summary.md)

The behavioral grounding. Route 2 compatibility profiles are not just structural categories — four of five have distinct behavioral signatures. The collapse/contamination mode split is the most operationally important finding: same failure rate, opposite channels.

### What you should know after this path

- Which parts of Gradience are LoRA-specific and which are general
- What the checkpoint triage workflow looks like end-to-end
- What the scope contract is and what happens outside it
- Whether your use case (different PEFT method, routing, triage) is within the validated envelope

---

## Path C — The Research Program

**For:** Collaborators, reviewers, theoretically curious readers. "What did you find, how solid is it, and what is still open?"

### Stops

**1. Settled / open / next dashboard** (10 min)
[`sidecar/notes/n69_settled_open_next.md`](../sidecar/notes/n69_settled_open_next.md)

Start here, not with the synthesis. This is the state-of-project dashboard: fourteen settled claims, ranked open questions, pending tests. It tells you what is established and what is not before you read the arguments.

**2. Mechanism-ladder synthesis** (20 min)
[`sidecar/packet/01_where_the_research_stands.md`](../sidecar/packet/01_where_the_research_stands.md)

The core theoretical account. Commensurability as the organizing concept, V-module pathology as the strongest signal (d=3.36 separation, zero overlap), head-level modulation, readout attractors, conjunctive failure. This is the mechanism ladder — each rung depends on the ones below.

**3. Ruled-out mechanisms** (15 min)
[`sidecar/packet/03_ruled_out.md`](../sidecar/packet/03_ruled_out.md)

Ten hypotheses tested and rejected. Portable severity, task-pair lookup, readout-as-risk, feature plurality as universal origin. Read this to understand the epistemic discipline: the eliminations are as important as the findings.

**4. Evidence register** (5 min)
[`sidecar/packet/04_evidence_table.md`](../sidecar/packet/04_evidence_table.md)

Compact table: every settled claim with its evidence citation and source. Cross-reference this against anything in the synthesis you want to verify.

**5. Route 2 synthesis** (15 min)
[`sidecar/notes/n93_route2_synthesis.md`](../sidecar/notes/n93_route2_synthesis.md)

The broadened-compatibility account. Four layers: decision-dependent (same structure, different meaning under different decisions), cross-artifact (portable signals are workflow-level), aggregation-sensitive (aggregation is computational, not presentational), behavioral bridge (profiles have behavioral reality). Each layer answers a question the previous one raised.

**6. Cross-artifact product relevance** (5 min)
[`docs/strategy/cross_artifact_product_relevance_summary.md`](strategy/cross_artifact_product_relevance_summary.md)

The product translation: what crosses artifact boundaries (evidence gating, narrowing), what doesn't (structural metrics, merge strategies), and the three-layer framework.

**7. Aggregation-sensitive summary** (5 min)
[`docs/strategy/aggregation_sensitive_route2_summary.md`](strategy/aggregation_sensitive_route2_summary.md)

Five aggregation-sensitive patterns. The central finding: only 2/12 cases are aggregation-invariant. Decision-context-dependent family selection is not a preference — it follows from what each decision context optimizes for.

**8. Behavioral Route 2 bridge** (10 min)
[`docs/strategy/behavioral_route2_summary.md`](strategy/behavioral_route2_summary.md)

The behavioral grounding. Three tiers: no pathology, localized pathology (with the collapse/contamination mode split), stasis. Same-family optional is safe-like. Routing-confusability has no merge-visible signature.

**9. GPU re-entry note** (5 min)
[`sidecar/packet/05_gpu_reentry.md`](../sidecar/packet/05_gpu_reentry.md)

What happens next: DeBERTa adjudication. Five pre-registered predictions, a decision tree for every outcome, ~3 hours of compute. The single most important open question.

### What you should know after this path

- The full mechanism ladder from commensurability to conjunctive failure
- What was eliminated and why the eliminations matter
- How the Route 2 programs extend the core account across decisions, artifacts, and aggregation strategies
- What is settled (fourteen claims), what is thin, and what is blocked on GPU
- Whether the research program is solid enough to build on

---

## Choosing your path

| You are... | Start with... |
|-----------|---------------|
| Evaluating whether to use Gradience | Path A |
| Exploring whether Gradience works for non-LoRA / non-merge | Path B |
| Considering collaboration or reviewing the research | Path C |
| Returning after time away | Path A (skim), then the [project map](project-map.md) |
| A new contributor | [Project map](project-map.md), then Path A, then the [CLAUDE.md](../CLAUDE.md) |
