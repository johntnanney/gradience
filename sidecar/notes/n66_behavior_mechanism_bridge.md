# n66 — Behavior–Mechanism Bridge

**Type:** synthesis
**Date:** 2026-03-28
**Depends on:** n51 (research synthesis), n61 (behavior findings), n63 (taxonomy), n64–n65 (dossiers)
**Status:** Complete.

---

## Purpose

This note bridges the example-level behavioral findings (n61–n65) to the geometric mechanism ladder (n41–n51). The goal is to establish which mechanism-ladder rungs are *visible* in the per-example behavioral data, which are not, and what the bridge teaches us about the relationship between representation-space geometry and downstream prediction behavior.

The bridge is not a proof — the example-level program does not measure V-module geometry or readout projections directly. It is an interpretive alignment: the behavioral patterns are *consistent with* specific mechanism-ladder predictions, and inconsistent with the available alternatives.

---

## The mechanism ladder (summary)

The multiscale mechanism ladder (n41, n51) proposes three rungs of analysis for merge quality:

**Rung 1 — V-module geometry.** The dominant singular values and directions of the two adapters' value-projection modules. When these are compatible (similar rank, overlapping subspaces), linear averaging preserves discriminative directions. When they are incompatible (divergent rank, orthogonal subspaces), averaging produces a compromise direction that corresponds to neither source's learned rule.

**Rung 2 — Head-level cancellation.** Within multi-head attention, individual heads may cancel during averaging even when aggregate statistics look acceptable. This is a fine-grained pathology that the current program does not measure directly.

**Rung 3 — Readout gating.** The classifier head's projection from the merged representation to output logits. When the readout is compatible with the merged representation, it faithfully transmits the merged signal. When it is incompatible (e.g., cross-task merges where the readout inherited a foreign task's decision rule), it may transmit a coherent but wrong signal, or amplify V-module pathology.

The **conjunctive model** (n51) holds that catastrophic merge failure requires the conjunction of V-module pathology (Rung 1) and readout incompatibility (Rung 3). Either alone is recoverable.

---

## Bridge table

| Behavioral pattern | Taxonomy category | Mechanism-ladder interpretation | Evidence strength |
|---|---|---|---|
| High consensus preservation, stable confidence | A (preserved consensus) | V-module compatibility preserves discriminative directions through averaging. Readout gate transmits faithfully. | **Strong.** Directly observed in SR-01 (97.5% preservation, confidence tracks source). |
| Neither-source predictions at 14–15% rate, near-chance confidence | D (neither-source) | V-module pathology: averaging incompatible discriminative directions produces a compromise representation that corresponds to no learned rule. Readout gate transmits this incoherent signal as near-uniform logits. | **Strong.** Cleanly separates fragile (14–15%) from safe (<2%). Confidence collapse (mean ~0.47) is consistent with flattened softmax from a directionless representation. |
| Confidence collapse without high-confidence wrong | (Fragile cases globally) | V-module pathology transmitted through an open readout gate. The readout is task-compatible (same-task merge) so it faithfully reports the upstream uncertainty. | **Moderate.** Consistent with the model. The absence of high-confidence wrong in fragile cases supports the interpretation that the readout is not injecting its own errors — it is faithfully transmitting the V-module's incoherence. |
| High-confidence wrong predictions | (CT-01 control) | Readout contamination: the merged readout has inherited a foreign task's decision rule and applies it confidently. V-module pathology may or may not be present, but the readout dominates the failure mode. | **Moderate.** CT-01 shows 23 high-confidence wrong predictions, 0 confidence collapse. This is the predicted signature of readout-dominated failure. But we have only one cross-task case, so the pattern is observed, not replicated. |
| Better-source loss at stable confidence | C (better-source loss) | Imperfect averaging of two learned rules within a compatible V-module subspace. The merge lands on a weighted combination that sometimes follows the weaker source. Readout faithfully transmits the wrong-but-coherent signal. | **Moderate.** Dominates near-miss (28–39%) and safe (18–34% in SR-02). Confidence stability (no collapse) is consistent with a coherent but suboptimal merged rule. |
| Near-miss ≈ safe on all discriminating metrics | (NM-01, NM-02 globally) | Below the V-module pathology threshold. Source incompatibility is insufficient to produce representational compromise that reaches the output. The conjunctive condition is not met. | **Strong.** Neither-source rate <2%, zero confidence collapse, zero or near-zero joint breakage. The threshold is sharp, not gradual. |
| Anchor dominated by shared failure (65.2%) | AN-01 (X, excluded) | No discriminative geometry to damage. V-module pathology cannot arise when neither source has developed meaningful discriminative directions. | **Moderate.** Consistent, but the anchor is a single case. The logic is sound: you cannot corrupt what was never learned. |
| Weak-source severity modulates breakage | FR-01 vs FR-02 | More severe V-module incompatibility (weaker partner → larger geometric divergence) produces more consensus breakage. The same mechanism, stronger dosage. | **Moderate.** FR-02 (partner at 0.136) breaks 34.2% of consensus examples; FR-01 (partner at 0.204) breaks 6.4%. Consistent with V-module pathology scaling, but confounded by the different source B adapters. |

---

## What the bridge reveals

### 1. Two qualitatively distinct failure modes are visible at example level

The behavioral data cleanly separate two failure modes that the mechanism ladder predicted:

**Fragile failure = V-module pathology, faithfully reported.** The merge averages incompatible discriminative directions, producing a representation that corresponds to no learned rule. The readout gate is open (same-task merge), so this incoherence reaches the output as near-chance confidence on wrong predictions. Neither-source behavior is the signature.

**Control failure = readout contamination.** The cross-task merge inherits a foreign decision rule. The merged representation may be somewhat coherent, but the readout maps it to the wrong task's labels. This produces confident wrong predictions. High-confidence wrong is the signature.

The example-level program cannot observe the geometric mechanisms directly — we do not compute SVD of the merged V-module or project through the readout. But the behavioral signatures are exactly what the mechanism ladder predicts, and they are *distinct*: fragile merges show confidence collapse without high-confidence wrong; control merges show high-confidence wrong without confidence collapse. This double dissociation is strong evidence that the mechanism ladder identifies real, separable pathology channels.

### 2. The neither-source threshold is sharp

The most striking finding is the discontinuity in neither-source behavior. The rate jumps from <2% (safe, near-miss) to >14% (fragile, control) with nothing in between. The anchor case (5.2%) does not bridge this gap — it reflects a different regime (no discriminative geometry) rather than an intermediate pathology level.

This sharpness is consistent with a phase-transition interpretation of the V-module compatibility boundary. Below the threshold, representational averaging preserves enough discriminative structure that the output remains source-consistent. Above the threshold, averaging destroys discriminative structure and the output enters the neither-source regime. The transition is not gradual.

### 3. The conjunctive model's predictions hold at example level

The conjunctive model predicts that catastrophic failure requires both V-module pathology and readout incompatibility. The example-level data are consistent:

- **Safe merges** (SR-01, SR-02): V-module compatible, readout compatible → preserved consensus dominates.
- **Near-miss merges** (NM-01, NM-02): Mild V-module stress, readout compatible → better-source loss is the main cost, but no structural pathology. The readout gate does not amplify the mild upstream imperfection.
- **Fragile merges** (FR-01, FR-02): V-module pathology, readout compatible → neither-source behavior at 12–15%, confidence collapse, but failure is *uncertain* (the merge knows something is wrong). The readout faithfully transmits pathology.
- **Control merge** (CT-01): V-module unknown, readout incompatible → confident wrong predictions. The foreign readout produces a qualitatively different failure mode.

The conjunction is most visible in the fragile cases: V-module pathology alone produces uncertainty (confidence collapse), not confident wrong answers. Adding readout incompatibility (as in the cross-task control) converts uncertainty into confident misdirection. This is the interaction effect the conjunctive model describes.

### 4. What the bridge cannot resolve

**Head-level cancellation (Rung 2)** is not visible in the example-level data. Per-example predictions aggregate across all attention heads, so cancellation within individual heads is invisible at this measurement level. The Rung 2 pathway predicts that some fragile failures occur because a single attention head's discriminative direction was cancelled by averaging, even though the aggregate V-module statistics looked acceptable. Detecting this would require head-level probing, which is outside the scope of this program.

**The V-module pathology threshold's location.** We know the threshold exists (the discontinuity is clear), but the example-level program does not measure where it falls in geometric terms. The threshold is defined in V-module singular-value space; the example-level program only observes its downstream consequences. Connecting the two requires the spectral measurements from the mechanism ladder (n41), which measure V-module geometry directly but have not been linked to per-example predictions. This is the most natural next bridge to build.

**Single cross-task case.** The control failure mode (readout contamination → high-confidence wrong) is observed in CT-01 but not replicated. The double dissociation between confidence collapse (fragile) and high-confidence wrong (control) is suggestive but rests on a single control case.

---

## Synthesis

The Output Example Semantics program set out to determine whether example-level behavioral differences between safe and fragile merges are real, structured, and interpretable. The answer is affirmative on all three counts.

The behavioral differences are **real**: they replicate across cases within each class and separate classes cleanly on multiple metrics. They are **structured**: five taxonomy categories capture all observed patterns without forcing, and the categories concentrate in specific merge-quality classes. They are **interpretable**: the behavioral signatures map onto the mechanism ladder's predictions about V-module pathology and readout gating with notable precision, including a double dissociation between failure modes that the mechanism ladder predicted but that had not previously been observed at example level.

The bridge from behavior to mechanism is interpretive, not causal. The example-level program measures downstream consequences; the mechanism ladder measures upstream geometry. The two levels of description are consistent, and the consistency strengthens confidence in both. The natural next step — correlating per-example failure categories with per-layer spectral pathology scores — would convert this interpretive bridge into a measured connection.

---

## Deliverables

| Deliverable | Path |
|------------|------|
| This synthesis note | `sidecar/notes/n66_behavior_mechanism_bridge.md` |
| Mechanism bridge table (JSON) | `sidecar/results/example_semantics/mechanism_bridge_table.json` |
| Taxonomy findings (input) | `sidecar/notes/n63_failure_taxonomy_findings.md` |
| Research synthesis (input) | `sidecar/notes/n51_research_synthesis.md` |
