# Regime Map — After Phase 2

## Central finding

On small encoder models, **task identity is the key regime boundary**. Same-task pairs are broadly safe across all tested stressors (training style, domain shift, source-strength asymmetry — 45 pairs, 0 material degradations). Cross-task pairs are where meaningful merge failure modes live. The task-relationship advisory is therefore part of the stable interpretive layer, and same-task regimes no longer justify active blind-spot hunting.

## Evidence base summary

This regime map is backed by: 29 adjudication pairs across 3 studies on 2 backbones, 132+ advisory checks (0 false positives), 5 observation inventories, 3 same-task blind-spot studies (45 pairs, 0 material degradations), and 1 cross-task subtype study (28 pairs across 4 tasks showing distinct behavioral subtypes).

## Regime map

| Regime | Main driver | Task advisory | Behavioral pattern | Workflow value |
|--------|------------|---------------|--------------------|----------------|
| Messy mixed-quality | Source QA | Redundant — pair-risk already high | N/A (QA-dominated) | High — early exclusion |
| Same-task, all eligible | None needed | Silent ✓ | Safe (0 material degradations in 49 pairs) | Low — confirmatory |
| Cross-task: mild | Task advisory | **Caution-raising** | Both tasks degrade 1.5-4pp (33% of cross-task pairs) | High — advisory warns |
| Cross-task: asymmetric dilution | Task advisory | **Caution-raising** | Stronger task preserved; weaker degrades 5-13pp (38%) | High — advisory is the main discriminator |
| Cross-task: broad degradation | Task advisory + pair-risk | **Caution-raising** | Both tasks degrade >4pp (21%) | High — advisory + pair-risk converge |
| Cross-task: catastrophic | Task advisory | **Caution-raising** | Task-specific incompatibility, up to 42pp (13%, all QNLI×MRPC) | High — no current signal grades severity within cross-task |
| Large mixed pool (6+) | QA + neighborhoods + advisory | **Highest value** — collapses candidate space | Mixed subtypes across pair matrix | High — compression + partitioning |

### Cross-task severity structure

Cross-task merges decompose into 4 severity levels (24 pairs on DistilBERT): mild (29%), asymmetric dilution (38%), broad degradation (21%), and catastrophic (13%). **Zero cross-task pairs were near-safe.**

**Boundary:** detected by task identity (advisory fires uniformly on all cross-task pairs on both backbones). **This generalizes.**

**Severity:** On a single backbone, exact task-pair identity is the strongest severity predictor (stable across seeds). **But exact task-pair severity does NOT replicate across backbones.** QNLI × MRPC is catastrophic on DistilBERT but mild on RoBERTa. Most families shift ~1 severity zone. Zero families showed high cross-backbone stability. **Do not featureize task-pair identity.**

**Structural severity signal:** core-space shared-basis does NOT replicate as a severity signal across backbones. On DistilBERT, lower basis correlated with worse outcomes (r=-0.614). On RoBERTa, the correlation sign flips (r=+0.273). The same shared-basis value corresponds to different severity outcomes on different backbones. **Core-space should not be promoted for severity grading.**

**What does NOT predict severity across backbones:** exact task-pair identity (backbone-dependent), core-space shared-basis (correlation sign flips), format similarity (misleading), source-strength gap (confounded), pair-risk (too coarse). **No current signal reliably grades severity within cross-task pairs across backbones.**

**What DOES generalize:** the cross-task boundary itself. All cross-task pairs degrade on both backbones. The advisory catches this boundary cleanly (0 false positives). Same-task pairs remain safe. The boundary is real and stable; the severity gradient within it is not.

## Key observations

1. **Source QA anchors everything.** In messy pools it does most of the narrowing. In credible pools it establishes the baseline that makes later signals interpretable.

2. **Task advisory is an established part of the stable interpretive layer.** It is most valuable not as a pairwise override, but as an inventory-level partitioning signal. In larger mixed-task inventories, it cleanly separates same-task safe zones from different-task caution zones and can dramatically reduce the candidate space left alive by structural pair-risk alone. In the observation round's 6-adapter/15-pair inventory, the advisory collapsed 11 medium-risk candidates to 2.

3. **Advisory value concentrates on medium-risk cross-task pairs.** When pair-risk is already high, the advisory is redundant. When pair-risk is medium, the advisory is the only signal distinguishing safe same-task pairs from unsafe cross-task pairs. This is where it earns its keep.

4. **Neighborhoods scale with pool size.** At 3-4 adapters they are confirmatory. At 6+ they provide genuine compression. Their grouping follows QA status and task structure, not latent spectral clusters.

5. **Core-space is narrow.** Verified adjudication showed it overwarns on same-task pairs and is noisy in the adjacent-task middle. Its remaining supported role is in genuinely ambiguous cases where pair-risk is permissive and task relationship does not already settle the question.

6. **The workflow adds most value in the middle.** Messy pools are solved by QA. Clean same-task pools need little analysis. The workflow earns its keep in partially credible, mixed-task, or structurally ambiguous pools — and the advisory is the sharpest single signal in that middle regime.

7. **Same-task regime is closed on small encoders.** Three blind-spot studies (45 pairs total) tested training-style variation, domain shift, and source-strength asymmetry. All found 0 materially degraded merges. The subtype study confirmed this: 4 same-task control pairs showed mean degradation of 0.5pp. No further same-task rescue logic is justified in this regime.

8. **Cross-task merges decompose into behavioral subtypes.** The cross-task subtype study (28 pairs, 4 tasks) found two distinct failure modes: (a) asymmetric dilution where the stronger task dominates and the weaker degrades 5-13pp, and (b) mutual degradation where both tasks suffer, sometimes catastrophically. Crucially, "related" NLI-family pairs (e.g., QNLI × MRPC) were NOT safer than distant pairs (e.g., QNLI × SST-2) — related pairs were actually worse on average. Semantic closeness does not predict merge safety. See `docs/internal/cross_task_boundary_evidence_card.md`.

## Backbone coverage

Tested on:
- distilbert-base-uncased (66M params)
- roberta-base (125M params)

The cross-task boundary (advisory selectivity) and same-task safety both replicate across backbones. **Exact task-pair severity does not.** The regime map applies to the boundary level for small encoder models generally, but severity claims are backbone-specific.

## Evidence base

- 29 adjudication pairs across 3 studies and 2 backbones
- 132+ total advisory checks: 0/37 same-task false positives, 99/99 different-task correct fires
- 5-inventory observation round confirming operational behavior across all regime categories
- 15-pair same-task training-style study: blind spot not found, 0 materially degraded merges
- 15-pair same-task domain-shift study (sentiment x 3 domains): blind spot not found, 0 materially degraded merges
- 15-pair source-strength asymmetry study (QNLI, 19.8pp range): blind spot not found, r=0.174
- 28-pair cross-task subtype study on DistilBERT (4 tasks): subtypes confirmed, severity decomposition established
- 28-pair cross-task generalization study on RoBERTa (4 tasks): boundary generalizes, exact severity does NOT
- 48-pair core-space severity replication (both backbones): shared-basis correlation sign flips (r=-0.614 → r=+0.273), does NOT replicate
- **56 total cross-task pairs across 2 backbones: boundary generalizes, severity grading does not (neither task-pair identity nor core-space)**

## Provenance

- `docs/internal/verified_adjudication_implications.md`
- `docs/internal/adjacent_task_adjudication_implications.md`
- `docs/internal/task_relationship_advisory_round_01_synthesis.md`
- `docs/internal/task_advisory_replication_implications.md`
- `docs/internal/roberta_replication_results.md`
- `docs/internal/phase2_rq_synthesis.md`
- `docs/internal/training_style_blind_spot_results.md`
- `docs/internal/domain_shift_blind_spot_results.md`
- `docs/internal/source_strength_blind_spot_results.md`
- `docs/internal/cross_task_boundary_evidence_card.md`
- `docs/internal/cross_task_severity_grading_implications.md`
- `docs/internal/task_pair_severity_generalization_implications.md`
- `docs/internal/core_space_severity_replication_results.md`
- `results/cross_task_subtype_study_01/`
- `results/task_pair_severity_generalization_study_01/`
