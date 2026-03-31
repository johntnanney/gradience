# n67 — Where the Research Stands

**Type:** synthesis
**Date:** 2026-03-31
**Supersedes:** n51 (research synthesis memo), n25 (multiscale mechanism synthesis)
**Depends on:** the mechanism-ladder evidence base (n01-n66, n68). Route 2 programs (n70-n92) are now consolidated in n93.
**Status:** The single current-best account of the sidecar's theoretical picture.

---

## 1. Commensurability as the umbrella concept

The sidecar started from a simple observation — good adapters are not automatically good merge partners — and spent the better part of a research program discovering what "good merge partner" actually means at the level of learned representations. The answer has converged on a single organizing concept: **commensurability**.

Commensurability is not similarity. Two adapters can be very similar (same task, same backbone, comparable benchmark scores) and still produce catastrophic merges under specific geometric conditions. Conversely, two adapters can differ substantially in internal organization — orthogonal decision axes, different feature sets, different attractor structures — and merge safely. The distinction that matters is not how alike the adapters look, but whether the specific ways in which they differ are compatible with linear averaging.

The concept has been refined three times as the evidence base expanded:

**Version 1** (n33): Commensurability = upstream V-module compatibility AND readout compatibility. Two binary conditions, both necessary, either alone insufficient. This was the first conjunctive formulation.

**Version 2** (n37, n42): The readout condition decomposes by attractor topology. Single-attractor families satisfy it automatically; multi-attractor families may or may not, depending on which attractor basin each adapter landed in. This introduced the notion that readout compatibility is not a monolithic property but varies with the structure of the solution space.

**Version 3** (n49, current): The readout condition further decomposes by mechanism class. Readout incompatibility arising from rotational degeneracy (same features, different angular combinations) has different failure semantics from readout incompatibility arising from feature-set switching (genuinely different feature sets). The gate opens in both cases, but what flows through it differs — incoherent confidence versus systematic misclassification.

Each version preserved the conjunctive structure and added resolution to the readout side. The upstream condition (V-module compatibility) has remained stable since its identification — it is the load-bearing wall. The readout condition has become richer and more differentiated, but its role in the model has not changed: it is the gate that determines whether upstream pathology reaches the output.

The concept's power lies in what it excludes. Commensurability is not "general compatibility." It is a precise conjunction of two independently measurable conditions. It does not predict whether a merge will be *good* — it predicts whether a merge is *safe from catastrophic failure*. The distinction matters. Many safe merges produce modest degradation; commensurability explains why that degradation does not escalate into catastrophe.

---

## 2. Instability as the best descriptive organizer

The sidecar's empirical starting point was the observation that severity — the magnitude of degradation from a merge — reverses across backbones. QNLI×MRPC degrades 41.7% on DistilBERT and 1.7% on RoBERTa. QNLI×SST-2 degrades 30.3% on RoBERTa and 0.3% on DistilBERT. A severity ranking constructed on one backbone is worse than useless on another.

Instability — the variability of severity across seeds and backbones — behaves differently. The two pairs that are unstable on DistilBERT (instability > 0.7) are also unstable on RoBERTa. The four stable pairs (instability < 0.3) are stable on both backbones. The gap between the clusters (0.30–0.74) is empty on both backbones. Instability is the first candidate portable descriptor: a property of the task pair that transfers across architectures.

This is a conceptual shift, not just a metric change. Severity asks "how bad is this merge?" — a question whose answer depends on which backbone you happen to be using. Instability asks "how fragile is this regime?" — a question about the *structure* of the merge landscape rather than a particular point in it. The unstable pairs are the ones where small changes in geometric conditions (different seeds, different backbones) can tip the merge from safe to catastrophic. The stable pairs are robust to these variations.

Instability remains the single best pre-merge descriptor that generalizes across conditions, but it is confirmed on only two backbones. The DeBERTa adjudication (n07) is the decisive test. If the instability ranking survives on a third backbone with disentangled attention, it becomes a strong candidate for promotion to core Gradience. If it fails, the concept needs revision — perhaps instability is a property of specific attention architectures rather than of task pairs per se.

---

## 3. V-module pathology as the strongest upstream risk signal

The mechanism ladder's first rung is its most robust finding. The V-module dimensionality ratio — the ratio of effective rank in the value-projection module between the two adapters — separates catastrophic from safe collision with Cohen's d = 3.36 and zero range overlap (catastrophic: 0.64–0.74; safe: 0.79–0.89). No other single metric in the sidecar program approaches this discriminative power.

The V-module signal was discovered through a specific methodological insight: the per-module decomposition. Earlier analyses examined the concatenated Q/K/V/O product matrix and found nothing — backbone confounds dominated. Separating the four attention modules revealed that the V-module carries the catastrophe-discriminating signal, the K-module carries a weaker secondary signal (d = 1.39), and Q and O carry none. The concatenation was diluting the signal by averaging a strong V-module effect with noise from Q and O.

The V-module matters because it controls what information each attention head contributes to the residual stream. When two adapters' V-modules encode discriminative directions of very different effective rank, linear averaging produces a compromise subspace whose dimensionality is incoherent — too high-rank for the lower-rank adapter's readout to interpret, too low-rank for the higher-rank adapter's readout to exploit. The result is a merged V-module that neither source's downstream machinery can use effectively.

Critically, V-module pathology is necessary but not sufficient. The conjunctive model requires that pathology also survive through the readout gate (§6). But the V-module dim ratio is the best *predictor* because the upstream condition is harder to satisfy than the downstream one — readout incompatibility is common (40% of same-task pairs) and mostly harmless, while V-module pathology is rare and always consequential when the gate is open.

---

## 4. Head-level modulation

The second rung of the mechanism ladder explains something the first cannot: seed sensitivity. Within the catastrophic pair CA-01 (QNLI×MRPC on DistilBERT), different random seeds produce severity ranging from 12.1% to 41.7% — a 29-point gap. This gap is invisible at the module level (V-module dim ratio deltas < 0.07 between seeds) and invisible in every aggregate metric the sidecar tested.

It is visible at the head level. Seven attention heads across 6 layers show |Δ_DR| ≥ 0.15 between the catastrophic and mild seed configurations (maximum 0.229 at layer 3, head 6). The module-level aggregate was near zero because individual heads show deltas of *opposite sign* — some heads become more compatible under the catastrophic seed while others become less compatible. The net outcome depends on which heads' incompatibilities dominate the downstream prediction.

This is the cancellation mechanism. The module-level dim ratio is a mean over 12 heads. Two seed configurations can produce the same mean but very different distributions. The catastrophic configuration concentrates incompatibility at heads that matter for the downstream output; the mild configuration distributes it across heads that are less consequential. The module level identifies *which pairs are at risk*; the head level explains *which specific instantiations of those pairs become catastrophic*.

Head-level modulation is confirmed for CA-01. CA-02 (QNLI×SST-2 on RoBERTa) shows a different pattern: the "toxic" adapter (qnli_s42, which appears in both anchors' worst variants) has a large V-module signal already visible at the module level, concentrated at layer 4 with individual heads showing Δ_DR up to -0.459. The two cases illustrate different pathways to the same outcome: CA-01 achieves catastrophe through distributed head-level modulation of a marginal module-level signal; CA-02 achieves it through a concentrated module-level signal amplified at specific heads.

---

## 5. Readout attractors and benign orthogonality

The sidecar's most counterintuitive finding is that readout orthogonality — two adapters using completely opposite decision directions — is common, benign, and structurally uninformative about merge risk. Five of 14 same-task seed pairs show decision-axis cosine near zero, yet all merge safely. A readout-cosine metric used as a risk signal would false-alarm on approximately 40% of same-task merges.

The explanation lies in the attractor structure of the readout solution space. Different tasks admit different numbers of stable readout attractors. QNLI consistently produces orthogonal readout across seeds (multi-attractor). RTE and SST-2 consistently produce aligned readout (single-attractor). MRPC is backbone-contingent — orthogonal on DistilBERT, aligned on RoBERTa. The distribution is sharply bimodal: decision-axis cosine clusters at approximately 0 or approximately 1, with no intermediate values.

The attractor landscape has been mapped across 10 task families (n41): 6 single-attractor, 3 multi-attractor, 1 backbone-contingent. All families merge safely regardless of attractor type. Multi-attractor structure is not a risk factor; it is a property of the task's solution space.

Two distinct mechanisms generate multi-attractor structure:

**Rotational degeneracy** (all 4 observed instances on DistilBERT): The readout solution space has a continuous degenerate manifold. Different seeds settle at different orientations within a shared low-rank subspace. The adapters use the same features but combine them in different angular proportions. Under conjunctive failure (if V-module pathology is also present), the predicted consequence is incoherent confidence — the right features with a wrong decision boundary.

**Feature-set switching** (1 observed instance, QNLI on RoBERTa): The readout solution space has multiple discrete basins occupying different principal component subspaces. Different seeds lock onto genuinely different feature sets. One QNLI/RoBERTa seed uses a decision direction aligned with RTE (cos = 0.86), confirming cross-task feature exploitation. Under conjunctive failure, the predicted consequence is systematic misclassification — a novel failure mode arising from averaging incompatible feature spaces.

The mechanism that is expressed follows a structured determinant hierarchy: task identity (primary — determines whether multi-attractor structure is possible) → backbone architecture (secondary — selects which mechanism realizes it) → training convergence (tertiary — gates attractor count but not mechanism) → domain structure (weak). Mechanism and backbone are currently perfectly confounded: all observed degeneracy is on DistilBERT, all observed switching is on RoBERTa. Whether this confound is intrinsic or an artifact of the two-backbone evidence base is the sharpest open question for the DeBERTa adjudication.

---

## 6. Conjunctive failure

The mechanism model's central claim is that catastrophic merge failure is conjunctive: it requires the co-occurrence of upstream V-module pathology and downstream readout incompatibility. Either alone is insufficient.

The evidence for the conjunctive structure comes from four independent lines:

**The SC-QMRB falsifier** (n32): The same pair (QNLI×MRPC) on RoBERTa has nearly identical readout geometry to the catastrophic instance on DistilBERT — both show ~89° orthogonal decision axes, both show ~0.70 margin proxy — yet produces only 1.7% degradation. Readout incompatibility without upstream pathology is harmless.

**The same-task seed evidence** (n36): Five of 14 same-task seed pairs show orthogonal readout yet all merge safely. Readout incompatibility is common in the absence of V-module pathology.

**The V-module dim ratio** (n21): Catastrophic pairs show V-module dim ratios of 0.64–0.74; safe collision pairs show 0.79–0.89. The upstream condition is rare and sharp-edged. When it is present, the readout gate determines whether pathology reaches the output. When it is absent, the readout gate is irrelevant.

**The seed contrast within CA-01** (n32): The catastrophic and mild seed configurations have virtually identical readout geometry (same decision axes) but differ by 29 points of severity. The readout gate is open in both cases; the difference is upstream — head-level V-module modulation determines whether the pathology that flows through the gate is severe enough to be catastrophic.

The conjunctive model explains three observations that no single-factor model can:

First, *why catastrophe is seed-dependent*: different seeds produce different readout attractor selections (affecting the gate condition) and different head-level V-module configurations (affecting the upstream condition). Both must be satisfied simultaneously.

Second, *why catastrophe is backbone-dependent*: different backbones produce different V-module geometry (affecting the upstream condition) and different attractor mechanisms (affecting the gate condition). The conjunction is independently determined at each level.

Third, *why most cross-task merges are not catastrophic*: cross-task merges often have readout incompatibility (the gate is open), but only a minority have V-module pathology severe enough to matter. The gate condition is easily satisfied; the upstream condition is the bottleneck.

---

## 7. Behavioral signatures of fragile versus cross-task failure

The Output Example Semantics program (n59–n66) extended the analysis from geometric measurement to behavioral observation: what do safe, fragile, and catastrophic merges actually do to individual predictions? The findings provide the first downstream confirmation that the mechanism ladder describes real, separable pathology channels.

An 8-case panel spanning safe retained, fragile, cross-task control, near-miss, and anchor classes was evaluated at per-example level. Five metrics and a 5-category failure taxonomy were constructed empirically:

- **A (preserved consensus):** Both sources correct, merge correct. The baseline of safety.
- **C (better-source loss):** One source correct, merge follows the wrong source. The averaging cost.
- **D (neither-source):** The merge's prediction matches neither source's prediction. The signature of representational compromise.
- **E (benign absorption):** Sources disagree, merge lands on the correct answer. Positive averaging.
- **X (shared failure, excluded):** Neither source correct, merge wrong. Pre-existing failure, not merge-caused.

(A sixth category, B — consensus breakage — was proposed but absorbed into D because pure consensus breakage without neither-source behavior was too rare to sustain as a separate category.)

### The clean discriminators

Two metrics cleanly separate merge quality classes:

**Neither-source rate** (category D): <2% in safe and near-miss cases; 12–15% in fragile and cross-task control. Nothing in between. This is a threshold, not a gradient — the system either produces novel predictions that neither source learned, or it does not.

**Joint-source breakage rate**: <1% in safe retained; 4.2% in near-miss (moderate); 6.4–34.2% in fragile (scaling with weak-source severity); 12.6% in cross-task control. The threshold between "normal averaging cost" and "structural pathology" falls at approximately 5%.

### The double dissociation

The most theoretically significant finding is a double dissociation between two failure modes:

**Fragile merges** (FR-01, FR-02) show **confidence collapse without high-confidence wrong predictions**. The merge encounters examples where source A is confident and correct, and produces a prediction at near-chance confidence — the softmax distribution flattens to near-uniform. The merge has lost the discriminative signal. It knows it doesn't know. (30 confidence collapse events in FR-01, zero high-confidence wrong.)

**Cross-task control** (CT-01) shows **high-confidence wrong predictions without confidence collapse**. The merge encounters examples where source A is confident and correct, and produces a *different* prediction at moderate-to-high confidence. The merge has not lost signal — it has acquired the wrong signal. It doesn't know it doesn't know. (23 high-confidence wrong in CT-01, only 3 confidence collapse events.)

This double dissociation maps directly onto the mechanism ladder:

Fragile failure = **V-module pathology, faithfully reported through a compatible readout**. The merge of two same-task adapters with incompatible V-module geometry destroys the discriminative directions in the value subspace. The readout gate is open (the adapters share a task, so the readout is compatible), and it faithfully transmits the upstream incoherence as near-uniform logits. The merge fails with uncertainty because there is no coherent signal to be confident about.

Cross-task failure = **readout contamination**. The merge of a cross-task pair inherits a foreign task's readout rule. The merged readout is internally coherent — it classifies decisively — but it is applying the wrong task's decision function. The merge fails with confidence because the injected rule is consistent, just wrong.

This is not a post-hoc rationalization. The mechanism ladder predicted these distinct failure modes before the example-level program was designed. The behavioral data confirm the prediction at a level of specificity (individual predictions with associated confidence values) that the geometric measurements could not reach.

### Near-miss is not a fragile precursor

The near-miss cases (NM-01, NM-02) fall within the safe-retained envelope on every discriminating metric: neither-source rate <2%, zero confidence collapse, zero or near-zero joint breakage, stable confidence. The elevated better-source loss in near-miss (28–39%) reflects higher source disagreement rates, not structural merge pathology. The rare neither-source predictions in NM-01 (9 total) are all boundary noise at vanishing margins (0.018–0.077 on a binary problem).

This confirms a threshold model rather than a gradient model. The boundary between safe and fragile is not a continuum where near-miss occupies an intermediate position. It is a discontinuity: the system is either above the V-module pathology threshold (and produces systematic neither-source behavior) or below it (and does not). Near-miss merges are below it.

---

## 8. What is settled, what is open, what is next

### Settled

| Claim | Evidence base |
|-------|--------------|
| Task-boundary detection is reliable | 0 false positives, 5 inventories, 53+ pairs |
| Severity is not portable across backbones | Rankings reverse completely (S01) |
| Readout orthogonality is benign in isolation | 5/14 same-task pairs orthogonal, all safe |
| Multi-attractor readout ≠ fragile | 10 families mapped, all safe regardless of attractor type |
| Catastrophe requires the conjunction of V-module pathology and readout incompatibility | Four independent lines of evidence (§6) |
| Fragile and cross-task failure are qualitatively distinct at example level | Double dissociation: confidence collapse vs high-confidence wrong (n64) |
| Near-miss is behaviorally safe, not a fragile precursor | Threshold confirmed on all discriminating metrics (n65) |
| Near-miss is a useful product category | 7 pairs, 3 backbones, avg Δ = -0.006 |
| Cross-artifact compatibility is workflow-level, not metric-level | 9-case panel, 3 artifact classes, 5 signal families audited (n76-n80) |

### Confirmed but thin (two backbones only)

| Claim | What would strengthen it |
|-------|------------------------|
| Instability is a portable descriptor | DeBERTa adjudication (n07) |
| V-module dim ratio separates catastrophic from safe (d = 3.36) | Third backbone, ideally with disentangled attention |
| Mechanism and backbone are confounded | Third backbone breaks or confirms the confound |

### Open

| Question | Status |
|----------|--------|
| Does the instability ranking survive on DeBERTa? | Blocked on GPU (n07) |
| Does the V-module signal survive on disentangled attention? | Blocked on GPU (n07) |
| Does the backbone–mechanism confound dissolve? | Blocked on GPU |
| Is head-level cancellation backbone-specific or general? | Blocked on GPU (Prediction E in n07) |
| Can the V-module pathology threshold be located in geometric terms? | Example-level program observes consequences but not coordinates (n66) |
| Does the cross-task failure mode (readout contamination) replicate beyond CT-01? | One case only |
| Can the conjunctive model be tested with a third condition (V-module pathology + compatible readout on a cross-task pair)? | Requires finding or constructing such a case |

### What should happen next

The DeBERTa adjudication (n07) remains the single most important next step. It now tests five predictions simultaneously: instability portability (A–C), V-module signal portability (D), and head-level modulation portability (E). The outcome determines whether the mechanism ladder is architecture-general or architecture-specific.

The natural extension of the example-level program would correlate per-example failure categories with per-layer spectral pathology scores — converting the interpretive bridge (n66) into a measured connection. This is CPU-feasible and would directly test whether examples classified as D (neither-source) come from layers where the V-module dim ratio is lowest.

---

## 9. The ruled-out path

The theoretical picture gains credibility from what the program eliminated. Eight hypotheses tested and rejected (n38, n52):

1. **Portable severity score** — killed by ranking reversal across backbones.
2. **Task-pair catastrophe lookup** — killed by backbone-dependence of which pair is catastrophic.
3. **Aggregate within-layer threshold** — killed by backbone confound; signal was in the V-module, not the Q/K/V/O concatenation.
4. **Readout orthogonality as risk marker** — killed by SC-QMRB falsifier and 5/14 same-task orthogonal pairs.
5. **Readout-alone explanation** — killed by conjunctive evidence; readout is a gate, not a cause.
6. **Simple feature plurality** — partially falsified; two distinct mechanisms instead.
7. **Training depth as primary determinant** — subordinate to task and backbone in the hierarchy.
8. **Domain structure as primary determinant** — weakest factor in the hierarchy.

Each elimination constrained the search space and pushed the program toward the conjunctive model. The theoretical structure is not the first thing the program tried — it is what survived after everything simpler was falsified.

---

## 10. How the pieces fit together

The sidecar's theoretical picture is now a layered account with four levels:

**Descriptive level — instability.** The best observable summary of merge risk. It tells you which pairs occupy fragile regimes without requiring geometric measurement. It is the first thing a practitioner would want to know.

**Upstream mechanism — V-module pathology.** The proximate cause of catastrophic degradation. Measurable through the dim ratio, confirmed at d = 3.36 with zero overlap. It tells you *whether* pathology exists. Head-level modulation within the V-module explains seed sensitivity — *which specific instantiations* of a pathological pair become catastrophic.

**Downstream gating — readout attractors.** The condition that determines whether upstream pathology reaches the output. Measurable through decision-axis cosine, but interpretable only in the context of attractor topology and mechanism class. It tells you whether the gate is open, and if so, what kind of failure to expect (incoherent confidence for degeneracy, systematic misclassification for switching).

**Behavioral manifestation — failure taxonomy.** The per-example consequences at the output. Measurable through the 5-category taxonomy. Neither-source behavior (D) is the cleanest downstream signature of the conjunction. The double dissociation between confidence collapse (V-module pathology through open gate) and high-confidence wrong (readout contamination) confirms that the mechanism ladder describes real, separable channels.

These four levels are not competing explanations. They are nested: instability describes the landscape, V-module pathology identifies the upstream cause, readout gating determines transmission, and the failure taxonomy reveals what reaches the output. Each level explains what the others cannot, and they are consistent with each other across every case in the evidence base.

A fifth consideration cuts across the four levels: **cross-artifact portability and decision-dependent broadening**. The mechanism ladder was developed on LoRA adapters in a merge-only context. Route 2 extended this to multiple artifact classes, decision contexts, aggregation strategies, and behavioral validation. That work is now substantial enough to have its own synthesis — see **n93 (Route 2 Synthesis)** for the complete account. The key conclusions:

- The workflow (evidence gating, conservative narrowing, task-relation ordering) transfers across artifact classes; the mechanism-level signals (V-module dim ratio, subspace overlap) remain representation-locked to factorized artifacts.
- Aggregation is a computational step, not a presentation layer — different rules produce genuinely different operational judgments from the same structural evidence.
- Four of five Route 2 compatibility profiles have distinct behavioral signatures, grouping into three tiers: no pathology, localized pathology (with a collapse/contamination mode split), and stasis.
- Decision-context-dependent aggregation is both structurally and behaviorally justified.

The mechanism ladder (§§1-9 of this note) describes real structure within the factorized regime. The Route 2 synthesis (n93) describes what happens when that structure is projected through different decisions, representations, and aggregation strategies. Together they constitute the project's theoretical account.

The program's intellectual trajectory has been from observation (something goes wrong in some merges) through phenomenology (instability) through mechanism (V-module pathology × readout gating) to behavioral confirmation (example-level signatures), and most recently through Route 2 broadening (decision-dependent, cross-artifact, aggregation-sensitive, behavioral bridge). That trajectory is not complete — the DeBERTa adjudication will test whether the mechanisms generalize across backbones — but the internal consistency of the picture, combined with the breadth of what has been ruled out and the clarity about what does and does not transfer, makes it the strongest account of LoRA compatibility dynamics currently available.

---

## 11. Route 2 (n70-n92)

The Route 2 research programs are now synthesized in a standalone companion note: **[n93 — Route 2 Synthesis: Decision-Dependent Compatibility Science](n93_route2_synthesis.md)**.

n93 consolidates four completed programs:

| Program | Notes | Central finding |
|---------|-------|-----------------|
| Decision-dependent compatibility | n70-n74 | Same structure means different things under merge, routing, triage. Aggregation is the first practical divergence. |
| Cross-artifact compatibility | n76-n80 | Portable signals are workflow-level, not metric-level. No structural metric is fully portable. |
| Aggregation-sensitive compatibility | n81-n85 | Aggregation is computational, not presentational. Five stable patterns. Decision-context-dependent family selection. |
| Behavioral bridge | n86-n92 | Four of five profiles have distinct behavioral signatures. Three-tier model: no pathology / localized pathology / stasis. |

The settled Route 2 claims, boundaries, and relationship to this synthesis are documented in n93 §§6-8. The per-program details, evidence, and deliverables are in n93 §§1-4 and §9.

---

## 12. Deliverables

| Deliverable | Path |
|------------|------|
| This synthesis (mechanism ladder) | `sidecar/notes/n67_where_the_research_stands.md` |
| Route 2 synthesis | `sidecar/notes/n93_route2_synthesis.md` |
| Route 2 programs | n70-n74, n76-n80, n81-n85, n86-n92 |
| Route 2 structured outputs | `results/decision_dependent_compatibility/`, `results/cross_artifact_portability/`, `results/aggregation_sensitive_compatibility/`, `results/behavioral_route2_bridge/` |
| Route 2 product summaries | `docs/strategy/{cross_artifact,aggregation_sensitive,behavioral}_route2_summary.md` |
| Output Example Semantics program | n59–n66 |
| Mechanism bridge table | `sidecar/results/example_semantics/mechanism_bridge_table.json` |
| Ruled-out mechanisms | `sidecar/notes/n68_ruled_out_mechanisms.md` |
| DeBERTa adjudication protocol | `sidecar/notes/n07_deberta_adjudication_protocol.md` |
| Executive summary | `sidecar/notes/n50_executive_research_summary.md` |
