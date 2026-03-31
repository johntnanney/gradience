# Product Validation

**What Gradience gets right, where the limits are, and what the field trials proved.**

This memo gives core Gradience its own empirical identity, separate from the sidecar research program. The sidecar investigates *why* merges fail at the geometric level. This document is about whether the product — as a practitioner-facing tool — makes correct recommendations on real adapters from public hubs.

Evidence base: 5 inventories + Campaign A/B micro-campaigns + targeted confirmation runs T01/T02, 3 backbones, 4 task families, 53+ pairs screened, 30 merges evaluated. Full data in `field_trials/`.

---

## 1. The evidence bootstrap lesson

The single most important finding from Phase 1 is a negative result: without behavioral evidence, Gradience produces nothing useful.

Pilot 1 ran 4 adapters (IMDB + emotion on DistilBERT) through the full pipeline without evaluation scores. Every adapter was classified as `unknown_no_behavioral_eval`. The pipeline could compute spectral profiles, pairwise alignments, and structural risk — but it could not populate the evidence gate, so the action plan excluded all pairs. The structural analysis was technically correct and operationally useless.

Adding behavioral evidence — even lightweight 500-sample CPU evaluations — transformed the output. With scores, the three-way classification (eligible / uncertain / flagged_weak) correctly handled every case we tested: genuine failures (Aureliano hate, delta -0.150 vs base), misleading evals (jmeneu IMDB, flagged for base-model artifact), marginal passes (hate adapters with delta +0.01 to +0.05), ambiguous ties (irony adapter at exactly 0.000 delta), and strong performers (AG News adapters at delta +0.65). The gate is well-calibrated across the full range.

The practical lesson: evidence generation is not an optional enrichment step. It is a prerequisite. The product should make behavioral evaluation as frictionless as possible — the current `evidence_bootstrap.py` script works, but a built-in `gradience eval` command would lower the barrier. The 500-sample budget (~30s per adapter on CPU) is sufficient to separate eligible from flagged in every case we tested.

## 2. Retained vs control: the narrowing logic works

Across all three Phase 2 pilots, Gradience reduces the candidate merge space by 90–93% and the retained pairs are the correct first choices.

| Pilot | Pairs | Retained | Reduction |
|-------|-------|----------|-----------|
| 1 (same-task control) | 3 | 0 | 100% (degenerate — all evidence-gated) |
| 2 (mixed-task, RoBERTa) | 10 | 1 | 90% |
| 3 (large mixed-task, DistilBERT) | 28 | 2 | 93% |

Phase 2 closed the loop by actually merging and evaluating retained pairs, near-miss pairs, and cross-task controls.

| Category | Pairs evaluated | Avg Δ vs best source | Improvers |
|----------|----------------|----------------------|-----------|
| Retained same-task | 7 | -0.024 | 2/7 (29%) |
| Near-miss | 7 | -0.006 | 1/7 (14%) |
| Cross-task control | 4 | -0.047 | 0/4 (0%) |

The ordering is correct. A practitioner following Gradience's evaluate-first list spends time on the most promising candidates. Two retained pairs actually improved over their best source (+0.028 on BERT hate, +0.006 on RoBERTa AG News). No cross-task control improved. The retained-vs-control gap is consistent and meaningful — not an artifact of a single inventory.

Task-boundary detection — the advisory that fires when a pair crosses tasks — achieved zero false positives across all 5 inventories and 53+ evaluated same-task pairs. Every genuine task-boundary crossing received the advisory. Every same-task pair was left clean. This is the product's most reliable signal.

## 3. Near-miss confirmation

The near-miss category is the product's most interesting validated finding. These are same-task, structurally plausible pairs excluded from the evaluate-first list only because one source is evidence-constrained (flagged_weak or uncertain).

The original case was a surprise: a hate-speech pair (jaesun × Aureliano) excluded because Aureliano's accuracy was below base. When merged and evaluated, it scored 0.598 — outperforming the best source by +0.078. The evidence gate's binary exclusion overrode a correct structural signal.

Phase 2b tested whether this was repeatable. It was. Across 7 near-miss pairs, 3 backbones, and 3 task families:

- Near-miss avg Δ vs best source: **-0.006** (practically indistinguishable from retained at -0.024)
- Cross-task control avg Δ: **-0.047** (5× worse)
- Weak-source severity modulates the outcome: sources that barely miss the gate (delta -0.002 to -0.004 vs base) produce merges indistinguishable from retained; deeply weak sources (delta -0.150) introduce more variance but still outperform cross-task controls

The near-miss action-plan section is now implemented — it sits between the same-task safe zone and the cross-task caution zone, explicitly labels which source is evidence-constrained, and lets practitioners decide whether to invest evaluation budget based on how close the weak source is to the threshold.

The fix was not to weaken the evidence gate. It was to make the exclusion visible and graduated rather than silent and binary. The gate correctly identifies risk; the near-miss layer correctly tells practitioners how much risk.

**Near-miss severity ordering (added 2026-03-29).** Near-miss pairs are now ordered by weak-source severity: marginal (delta > −0.010), moderate (−0.010 to −0.050), or substantial (< −0.050). The ordering puts best prospects first. Targeted confirmation T02 verified the mechanism works correctly as a presentation layer — labels are clear, the section is legible, and a user with limited evaluation budget can immediately identify the best optional candidate. Outcome discrimination between severity levels remains lightly underconstrained in minimal-rank (r=1) public-adapter conditions, where merged outputs are too similar to differentiate. This is an inherent limitation of r=1 adapters, not a feature defect.

## 4. Large-inventory ergonomics

Pilot 3 (9 adapters, 28 pairs, DistilBERT) was the scale test. The pipeline handled it without special configuration, but the output exposed ergonomic lessons.

At 28 pairs, the action plan's value shifts from "which pairs to merge" to "what to ignore." The 93% reduction (28 → 2) is sharp, but a flat pair table with 28 rows is still hard to scan. Neighborhoods — which partition adapters into same-task clusters — become genuinely useful at this scale. Below 6 adapters, the pair table is readable enough that neighborhoods add overhead without insight. Above 6, they are the primary navigation aid.

The HTML report scales well. Structural detail (per-layer risk, norm mass profiles) can be expanded per pair without overwhelming the summary view. The preflight bundle (`preflight_summary.json`, `inventory_action_plan.md`, `review_packet.md`) provides machine-readable and human-readable outputs at every scale tested.

Norm imbalance dominated structural analysis across all pilots — 75–100% of pairs showed it as the dominant issue, driven by the rank heterogeneity in public adapters (r=1 TransferGraph adapters merging with r=4 to r=16 community adapters). When the majority of pairs share the same dominant issue, the label stops being informative. A severity ranking (magnitude ratio already computed internally) would add triage value at scale.

## 5. Current product strengths

These are empirically confirmed, not aspirational:

**Task-boundary detection with family awareness.** The most reliable single signal. Zero false positives, 53+ pairs, 5 inventories, 3 backbones. Now includes a static task-family taxonomy so that same-family pairs (e.g. SST-2 × IMDB) receive informational routing rather than overprotective caution. Catches the single most common practitioner mistake — merging adapters from different tasks — while correctly permitting known equivalent task families.

**Evidence gate calibration.** The three-way classification handles the full range from strong to failed adapters. The 500-sample CPU eval budget is sufficient. The gate is the highest-impact single feature — without it, the pipeline produces nothing useful; with it, every recommendation is grounded in measured behavior.

**Candidate reduction at useful rates.** 90–93% reduction with correct prioritization. The retained set includes the right pairs in every inventory tested. A practitioner following the evaluate-first list does not miss the best candidates.

**Near-miss as a structured second tier.** Validated across 3 backbones and 3 task families. Fills the gap between retained pairs and full exclusion. Now includes severity-based ordering (marginal / moderate / substantial) so practitioners see best prospects first. Gives practitioners an informed option when the retained set is small.

**Preflight bundle as communication artifact.** The JSON + markdown + HTML triple output serves different audiences (automation, reviewers, practitioners) from a single pipeline run. The bundle is self-contained — you can hand it to a colleague without additional context.

**PEFT-general audit substrate.** Ring 1 confirmed that the measurement layer, pairwise comparison, and inventory triage generalize to LoHa (Low-Rank Hadamard Product) adapters via a ~160-line extraction shim. Zero core code was modified. The spectral math, QA eligibility logic, and report vocabulary are artifact-agnostic. Any PEFT type that exposes low-rank factor pairs can use the existing pipeline. Details: `docs/strategy/ring1_peft_generalization_results.md`. Design: `docs/design/peft_generalization_audit.md`.

**Full-checkpoint delta triage path (bounded).** Ring 2 confirmed that full fine-tuned checkpoint deltas can be audited and triaged on CPU by switching from factor export to layer-summary representation (Representation C). The workflow (single-artifact audit, pairwise comparison, inventory triage) survives; the representation path differs. Evidence bootstrap and source QA remain the binding constraints. Merge execution was intentionally out of scope. Details: `docs/design/ring2_stage_d_assessment_memo.md`.

## 6. Remaining product limitations

These are known boundaries, not bugs:

**Adapter ecosystem bias — partially addressed.** All validation uses classification tasks on small encoders (DistilBERT, BERT, RoBERTa) with LoRA ranks 1–16. Generation tasks, large language models, high-rank adapters (r≥32), and non-accuracy metrics (F1, BLEU, perplexity) are untested. Ring 1 extended the validated PEFT scope to include LoHa (CPU-only, SST-2, distilbert, ranks 4–16). Ring 2 added bounded full-checkpoint-delta triage (shared-base distilbert panel, CPU-only) via summary-based representation. Non-low-rank methods (IA3), decoder LLMs, and broad checkpoint-delta inventory sweeps remain uncharted. The product's validated scope is broader than before but still narrower than its aspirational scope.

**Spectral analysis underexercised at r=1.** Most field trial adapters are r=1 TransferGraph models. At rank 1, a LoRA adapter is a single direction in weight space — energy-rank profiles, utilization patterns, and multi-rank interactions are invisible. The spectral layer's distinctive contribution is clearest on same-task pairs with rank ≥8. A richer-adapter validation would test whether spectral analysis adds value beyond what the evidence gate and task-boundary detection already provide.

**Marginal-adapter problem.** Adapters that barely beat base (delta +0.01 to +0.05 on binary tasks) pass the evidence gate as eligible but contribute little to merges. The gate is binary — any positive delta passes. A graduated confidence score or minimum-delta threshold would help practitioners avoid investing evaluation budget in merges involving marginal sources.

**Task-family blindness — resolved.** Task-boundary detection was previously metadata-based only (comparing eval_dataset strings), which was overprotective for same-family tasks. Campaign A confirmed the problem; a static task-family taxonomy (`vnext/merge/task_families.py`) was implemented and confirmed in targeted run T01. Same-family pairs (e.g. SST-2 × IMDB) now route to the same-task safe zone with an informational "TASK-FAMILY NOTE" advisory instead of a cross-task caution warning. T01 confirmed the routing is correct: the same-family merge (0.878) performed identically to the retained same-task merge (0.876), while the cross-task control (0.842) was meaningfully worse. The taxonomy is static and narrow (currently: `sentiment_binary` for sst2, imdb, yelp_polarity, amazon_polarity only). See `field_trials/targeted_confirmation_summary.md`.

**Scale ceiling untested.** The largest validated inventory has 28 pairs (9 adapters). Inventories with 50+ pairs, heterogeneous backbones, or multi-adapter-per-model configurations are uncharted. The pipeline runs, but the output quality is unvalidated.

**No GPU-accelerated evaluation integration.** The evidence bootstrap uses CPU-only 500-sample evals. This is sufficient for the evidence gate but too coarse to distinguish marginal adapters from strong ones. GPU-accelerated evaluation on full validation sets would tighten the gate and reduce the marginal-adapter problem.

---

**Broader-utility summary (bounded).** The substrate now generalizes across artifact classes and across downstream decisions, with aggregation and policy as the main scenario-specific seams. In practical bounded scope, Gradience supports evidence-aware triage for both adapter inventories and full checkpoint inventories on shared-base small encoder models, including same-task, same-family, and cross-task distinctions. Adapter evidence comes from field inventories plus routing/near-miss validation; checkpoint evidence comes from Ring 2 representation + guardrail stages and checkpoint field trials (`field_trials/checkpoint_inventory_t01/`, `field_trials/checkpoint_inventory_t02/`).

**Substrate generality — validated on three orthogonal axes.** (1) A routing confusability pilot consumed the spectral analysis pipeline without modifying any existing module, producing meaningfully different operational guidance from the same geometric data (scenario axis: merge vs. routing). Details: `docs/routing-pilot-results.md`. (2) Ring 1 PEFT generalization confirmed the pipeline operates on LoHa adapters via a thin extraction shim — measurement, pairwise comparison, and inventory triage all ran unmodified on non-LoRA artifacts (artifact-class axis within low-rank PEFT: LoRA vs. LoHa). Details: `docs/strategy/ring1_peft_generalization_results.md`. (3) Ring 2 showed that full checkpoint deltas can be handled through summary-based reuse, preserving audit and triage workflows while changing representation path (representation-path axis: factor-based vs summary-based). Details: `docs/design/ring2_stage_d_assessment_memo.md`. Together these results support broader-opportunity confidence while still requiring narrow-scope claims: evidence bootstrap and QA remain central, and merge execution is still out of scope for Ring 2 checkpoint-delta work. Architectural model: `docs/architecture-assessment.md`.

---

*Full field trial data: `field_trials/`. Phase comparison: `field_trials/pilot_phase1_comparison.md`. Evaluation results: `field_trials/phase2_eval_130608/phase2_results.json`. Near-miss validation: `field_trials/near_miss_validation.md`. Targeted confirmation: `field_trials/targeted_confirmation_summary.md`. Next-phase protocol for targeted product questions: `field_trials/cpu_field_research_protocol.md`. Architecture assessment: `docs/architecture-assessment.md`. Routing pilot results: `docs/routing-pilot-results.md`. Ring 1 PEFT generalization: `docs/strategy/ring1_peft_generalization_results.md`. Ring 1 design: `docs/design/peft_generalization_audit.md`. Ring 1 experiments: `experiments/peft_ring1/`. Ring 2 Stage D memo: `docs/design/ring2_stage_d_assessment_memo.md`. Ring 2 experiments: `experiments/ring2_checkpoint_delta/`.*
