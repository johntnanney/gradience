# n69 — Settled / Open / Next

**Type:** index
**Date:** 2026-03-31
**Depends on:** n67 (mechanism-ladder synthesis), n93 (Route 2 synthesis), n68 (ruled-out mechanisms), n07 (DeBERTa protocol)
**Purpose:** Standalone state-of-the-project dashboard. Updated independently of the synthesis memo so the repo always has one current-truth file about what is established, what is unresolved, and what comes next.

---

## Settled

These claims are supported by multiple independent lines of evidence across the existing two-backbone evidence base. Reversing any of them would require new data that contradicts several converging results simultaneously.

**Core Gradience use case.** Task-boundary detection is reliable: zero false positives across 5 inventories and 53+ evaluated pairs. The advisory layer (same-task safe, cross-task caution) is a stable interpretive signal that Gradience can surface without deep geometric analysis.

**Same-task safe zone on small encoders.** All same-task seed pairs merge safely on DistilBERT and RoBERTa, including pairs with orthogonal readout axes. Same-task merging is not where catastrophe lives. This is the cleanest negative result in the sidecar — the risk is specifically at the task boundary.

**Task-boundary advisory as stable interpretive layer.** The gap between same-task and cross-task merge outcomes is consistent across backbones, seeds, and tasks. It is the first thing Gradience reports and the most reliable thing it can say.

**Evidence gate importance.** The conjunctive model — catastrophe requires V-module pathology AND readout incompatibility, either alone insufficient — is confirmed by four independent evidence lines: (1) readout-orthogonal same-task pairs are safe, (2) V-module pathology without readout incompatibility produces absorption not catastrophe, (3) readout incompatibility without V-module pathology produces mild degradation, (4) example-level double dissociation between failure modes (n64). The evidence gate matters: neither condition alone produces catastrophic failure.

**Near-miss as validated middle category.** Seven pairs across three backbones. Average delta versus best source = −0.006, comparable to retained merges. Behaviorally indistinguishable from safe on all discriminating metrics (n65). Not a fragile precursor — there is a sharp threshold, not a gradient. Campaign B (2026-03-29) confirmed: barely-weak near-miss pairs (source delta −0.002 to −0.004) show avg merge delta −0.007, better than retained (−0.018). Deeply-weak sources (delta −0.150) show avg −0.045, approaching cross-task controls.

**Task-family equivalence within binary sentiment.** Campaign A (2026-03-29) showed that SST-2 × IMDB merges are indistinguishable from same-task retained merges: avg delta −0.022 (family test) vs −0.017 (retained), gap 0.005. Gradience's strict task-identity boundary is overprotective for practically similar sentiment tasks. Tested on DistilBERT with 4 adapters (2 SST-2, 2 IMDB), 7 merge evaluations.

**Orthogonal readout not a risk marker.** Five of fourteen same-task pairs have orthogonal readout axes. All are safe. The SC-QMRB falsifier case is orthogonal and safe. Multi-attractor readout structure (10 families mapped) is decoupled from fragility. Orthogonality is a structural property of the solution landscape, not a warning sign.

**Fragile and cross-task failure are qualitatively distinct.** Double dissociation confirmed at example level (n64): fragile merges produce confidence collapse (low confidence, spread predictions), cross-task control produces high-confidence wrong predictions. These are different failure channels — V-module pathology transmitted through an open gate versus readout contamination — not different severities of the same thing.

**Neither-source behavior as catastrophe discriminator.** The D-category (neither-source predictions) rate jumps from <2% in safe/near-miss to 12–14% in fragile/control, with nothing in between. This is a threshold signature, not a graded signal, and it is the cleanest behavioral marker of the conjunction at work (n63, n66).

**Decision-dependent compatibility framing.** The substrate now has bounded evidence for reuse across both artifact classes and downstream decisions (merge, routing, triage). The main scenario-specific seams are aggregation and policy, not measurement (n70-n74, routing pilot, Ring 1, Ring 2).

**Cross-artifact compatibility is workflow-level, not metric-level.** A 9-case panel across LoRA, LoHa, and checkpoint delta (n76-n80) confirms: evidence gating and conservative narrowing are strong cross-artifact invariants; task-relation ordering is a moderate invariant. No structural metric is fully portable -- the V-module dim ratio is representation-locked to factorized artifacts. Triage is the only cross-artifact decision scenario; merge and routing remain LoRA-specific. Product language can reference the workflow shape across artifact classes but not structural measurements.

**Aggregation is a computational seam, not a presentation layer.** The base aggregation program (n81-n85) showed that only 2/12 cases are aggregation-invariant and that the remaining 10 change operational label under aggregation-family choice. The stability pass (n98-n102) strengthened this result under local perturbation: aggregation-as-seam remained stable, QA-dominant remained a distinct operational family, and worst-case collapse of routing gradation remained stable. The hybrid (QA-gated distributional) remains the richest family. Aggregation family selection should be decision-context-dependent: merge → worst-case, routing → distributional, triage → QA-dominant, general-purpose → hybrid.

**Mixed-evidence triage middle is structured with guardrails.** A targeted triage-weighted perturbation pass (n103-n107) tested whether the soft middle remains interpretable when review-like and same-family optional cases are overrepresented. It held: QA-dominant remained coherent, same-family optional cases stayed review/clear-leaning rather than collapse-like, and coarse review prioritization remained usable. What stays guarded: exact review thresholds, exact taxonomy boundary cuts, and fine-grained intra-review ordering claims.

**Route 2 compatibility profiles have behavioral reality.** An 8-case, 4,000-example behavioral bridge study (n86-n92) confirms: four of five broadened Route 2 profiles have distinct example-level behavioral signatures, grouping into three tiers -- no pathology (safe/optional, neither-source <2%), localized pathology (collapse/cross-task, neither-source ~14%), and stasis (QA review, shared failure 65%). The Route 2 framework is grounded in model behavior, not only structural measurement.

**Same-family optional is behaviorally safe-like.** Near-miss / same-family optional cases are indistinguishable from aggregation-invariant safe on all discriminating metrics: <2% neither-source, zero confidence collapse, zero joint breakage (n88, n89). The structural near-miss classification does not predict behavioral pathology. QA constraints on these cases are about evidence gaps, not behavioral risk. This extends the earlier near-miss finding and grounds it in the Route 2 profile system.

**Collapse and contamination are distinct operational failure channels.** Worst-case collapse (fragile merges) and cross-task contamination produce the same ~14% neither-source rate but through opposite confidence patterns: collapse has 28-30 confidence collapses and 0 high-confidence wrong; contamination has 3 confidence collapses and 23 high-confidence wrong (n88, n90). The model either knows it doesn't know (collapse, recoverable) or doesn't know it doesn't know (contamination, dangerous). This behaviorally justifies decision-context-dependent aggregation — different aggregation families track different failure channels.

**Decision-context-dependent behavioral signatures.** Aggregation family selection is not a preference — it is behaviorally grounded (n92, n93). Worst-case aggregation detects concentrated pathology (relevant for merge, tracks collapse channel). Distributional aggregation detects confusability gradients (relevant for routing). QA-dominant aggregation detects evidence absence, which corresponds to behavioral stasis (relevant for triage). The hybrid (QA-gated distributional) preserves both evidence constraints and structural gradation. Each family is appropriate for its decision context because it is sensitive to the failure channel that matters for that context.

---

## Open

These are genuine unresolved questions where the existing evidence is insufficient or where new data could change the theoretical picture.

**DeBERTa adjudication of the multiscale ladder.** The single most important open question. Five pre-registered predictions (A–E in n07) test whether instability, V-module pathology, and head-level cancellation transfer to disentangled attention. Blocked on GPU compute (~3 hours). The outcome determines whether the mechanism ladder is architecture-general or architecture-specific.

**Backbone confound in attractor mechanism classes.** All rotational degeneracy cases are on DistilBERT; all feature-set switching cases are on RoBERTa. Mechanism and backbone are perfectly confounded (n49). A third backbone either breaks or confirms this confound. If DeBERTa shows a novel mechanism class, the two-class taxonomy needs revision.

**Exact threshold location for catastrophic conjunction.** The conjunctive model says V-module pathology plus readout incompatibility produces catastrophe. But where exactly does the V-module dim ratio cross from safe to pathological? The current evidence gives a separating gap (0.74–0.79) but not a precise threshold. Locating this in geometric terms would convert the model from explanatory to predictive.

**Cross-backbone survival of output-space descriptors.** The failure taxonomy, the neither-source rate as discriminator, and the double dissociation are all confirmed on two backbones. Whether these behavioral signatures are architecture-general is untested. The taxonomy structure might be stable even if the specific thresholds shift.

**Cross-task failure replication.** The readout contamination pathway (high-confidence wrong) is confirmed on one case only (CT-01). Whether this generalizes to other cross-task pairs with different task combinations is open. The mechanism is predicted by the model, but the single-case evidence base is thin.

**Example-level spectral correlation.** The behavior–mechanism bridge (n66) is interpretive: examples classified as D (neither-source) are predicted to come from layers where V-module dim ratio is lowest. This correlation has not been directly measured. A CPU-feasible analysis could convert the bridge from interpretive to quantitative.

---

## Next

Priority-ordered. The GPU step is decisive; everything else is either preparation for it or CPU-feasible deepening.

**1. GPU: Execute the DeBERTa adjudication protocol (n07).** Train 8 adapters, merge 28 pairs, evaluate 56 conditions. ~3 hours on a single consumer GPU. Pre-registered, pre-checked, executable as-is. This is the rate-limiting step for every major open question above. The re-entry note in the research packet (`packet/05_gpu_reentry.md`) has the condensed checklist and decision tree.

**2. GPU: Per-module and head-level analysis on DeBERTa results.** Conditional on the merge evaluation. Produces the data for Predictions D and E. Requires adapting `per_module_geometry.py` and `v_head_geometry.py` for DeBERTa module names (disentangled attention content/position projections). Module correspondence must be established at training time.

**3. CPU: Example-level spectral correlation.** Correlate per-example failure categories with per-layer V-module dim ratio scores. Tests whether D-category examples cluster at layers with the lowest ratios. This is the natural extension of the example-level program and does not require GPU.

**4. CPU: Cross-task replication.** If field trial data produces additional cross-task catastrophic cases, apply the failure taxonomy and check whether the readout contamination signature (high-confidence wrong) recurs. Strengthens or weakens the double dissociation.

**5. Decision: V-module promotion assessment.** If DeBERTa Prediction D passes, write a formal assessment for promoting the V-module dim ratio to a computable warning signal in core Gradience. This is the bridge from sidecar research to product feature.

**6. Decision: O-module escalation design.** If Predictions D and E both pass, the next mechanistic step is clear: extract per-head output weights from the O-module LoRA product to test whether catastrophic head configurations are selectively amplified. Requires a new protocol document.

---

## Freeze rule for this line

Decision-Dependent Compatibility is complete enough for this phase and should stay frozen unless at least one trigger occurs:

1. a new external use case appears,
2. GPU availability enables materially stronger validation,
3. a contradiction appears in real use data.

---

## CPU-feasible product validation (non-GPU)

Four targeted micro-campaigns are designed to close remaining product-facing unknowns without GPU. These are independent of the DeBERTa adjudication and can proceed immediately. Full protocol: `../field_trials/cpu_field_research_protocol.md`.

**A. Task-family equivalence — SETTLED (2026-03-29).** Same-family cross-dataset pairs behave like retained. Static task-family taxonomy implemented (`vnext/merge/task_families.py`) and confirmed in targeted run T01: same-family routing correct, report clarity confirmed (one headline fix applied). Feature area closed. See `../field_trials/targeted_confirmation_summary.md`.

**B. Marginal-adapter behavior / near-miss severity — SETTLED (2026-03-29).** Barely-weak ≈ retained, deeply-weak ≈ intermediate. Severity ordering implemented in action plan and confirmed as usable presentation layer in targeted run T02. Outcome discrimination remains lightly underconstrained in minimal-rank (r=1) public-adapter conditions. Feature area closed — no further poking unless a real-world use case breaks it. See `../field_trials/targeted_confirmation_summary.md`.

**C. Large-inventory stress — DEFERRED.** Low priority; current validated ceiling (28 pairs / 8 adapters) covers most practical use cases.

**D. Public-ecosystem robustness — DEFERRED.** Worth running when convenient; TransferGraph adapters already exercise some edge cases.

Cross-campaign synthesis: `../field_trials/synthesis/cross_campaign_summary.md` and `../field_trials/synthesis/product_implications.md`.

---

## How to use this index

This file is the project's state-of-play dashboard. It should be updated whenever a settled claim is revised, an open question is resolved, or a next step is completed or deprioritized. The synthesis memo (n67) provides the full theoretical narrative; this index provides the quick-reference version. The ruled-out mechanisms packet (n68) documents what the project eliminated en route to the current picture.

If you are returning to this project after a break, read this file first, then `packet/00_packet_index.md` for the full research packet, then n07 for the GPU protocol. For CPU-feasible product testing, see `../field_trials/cpu_field_research_protocol.md`.

---

*Last updated: 2026-04-01. Authoritative for project state as of this date. Campaigns A and B complete. Targeted confirmation T01/T02 complete. Same-family routing and near-miss severity ordering settled. Decision-dependent compatibility (n70-n74) integrated and frozen under trigger-based reopening. Cross-artifact portability (n76-n80) integrated. Aggregation stability check (n98-n102) integrated. Mixed-evidence triage perturbation pass (n103-n107) integrated with guarded soft-middle claims.*
