# Note: DeBERTa-v3 Adjudication Protocol

## Metadata

- **Type:** protocol
- **Date:** 2026-03-26
- **Related studies:** S01
- **Related panels:** P01
- **Related notes:** n21 (V-module dimensionality mismatch finding), n24 (head-level findings), n25 (multiscale synthesis), n19 (per-module subset), glossary (frozen V-module language, multiscale mechanism ladder)
- **Project:** Phase 3, Project F — Instability Program Consolidation
- **Revision:** Updated March 2026 to incorporate V-module adjudication target (Prediction D), head-level modulation target (Prediction E), O-module escalation target (Prediction F), and correct LoRA configuration to match actual training parameters (r=16, all four attention modules)

---

## Purpose

This document is the complete pre-registered protocol for the DeBERTa-v3 adjudication leg of Study S01. It specifies exactly what to run, what to measure, how to interpret the outcomes, and what decisions follow from each outcome pattern. It is intended to be executable as-is once GPU compute is available, with no further design decisions needed.

**Adjudication now has three targets, corresponding to the three rungs of the multiscale mechanism ladder (glossary, n25).** The original target — does the instability ranking survive? — remains (Predictions A–C). But the mechanistic picture has sharpened substantially, and each rung is now independently testable:

1. **Does the module-level V signal survive?** (Prediction D, Rung 1.) V-module dimensionality ratio separates catastrophic from safe collision with d=3.36 on RoBERTa. DeBERTa-v3's disentangled attention restructures the value projection's role. If the signal survives, it is architecturally generic.

2. **Does head-level cancellation / modulation recur?** (Prediction E, Rung 2.) On DistilBERT, CA-01's 29-point seed gap localizes to 7 heads with opposite-sign dim ratio deltas that cancel at module level. If DeBERTa shows any seed-sensitive catastrophic case, the same head-level cancellation pattern should appear — a module-level dim ratio that does not distinguish seed variants, with individual heads showing |Δ_DR| ≥ 0.10.

3. **Is O-module / output interaction the right next escalation?** (Prediction F, Rung 3.) If DeBERTa reproduces the module-level signal (D) and the head-level modulation (E), the remaining causal gap is downstream amplification. Prediction F is not a pass/fail test but a decision criterion: if Rungs 1–2 survive, Rung 3 (O-module analysis) is the confirmed next step. If Rung 1 survives but Rung 2 does not, the head-level story may be backbone-specific and the escalation should target DeBERTa's disentangled attention structure instead.

This is a much sharper adjudication frame than the original "does instability survive?" question. Each prediction tests a different claim at a different scale.

---

## 1. Training Protocol

### Backbone

**Model:** microsoft/deberta-v3-base
**Architecture:** 12 transformer layers, disentangled attention (content + position), replaced token detection pretraining
**Why this backbone:** Architecturally distinct from both DistilBERT (6 layers, standard attention) and RoBERTa (12 layers, standard attention). Matches RoBERTa's depth but not its attention mechanism, isolating the architecture variable from depth.

### Adapters

8 LoRA adapters, matching the existing study design:

| ID | Task | Seed | Expected training time |
|----|------|------|----------------------|
| deberta_qnli_s42 | QNLI | 42 | ~15 min |
| deberta_qnli_s7 | QNLI | 7 | ~15 min |
| deberta_rte_s42 | RTE | 42 | ~5 min |
| deberta_rte_s7 | RTE | 7 | ~5 min |
| deberta_mrpc_s42 | MRPC | 42 | ~5 min |
| deberta_mrpc_s7 | MRPC | 7 | ~5 min |
| deberta_sst2_s42 | SST-2 | 42 | ~10 min |
| deberta_sst2_s7 | SST-2 | 7 | ~10 min |

### LoRA configuration

Must match the actual configuration of the existing adapters (not the original S01 spec, which said r=8 — see n13 for the discrepancy): **rank 16, alpha 16, dropout 0.1, target modules = all four attention projections** (query, key, value, output). 3 epochs, learning rate 2e-4, batch size 16, warmup ratio 0.06, weight decay 0.01.

**Critical: target all four modules.** The per-module analysis (n21) depends on having Q/K/V/O LoRA weights for every adapter. Targeting only query + value (as the original S01 spec stated) would make the V-module dimensionality-ratio test possible but would eliminate the K and O modules from the comparison, losing the module-contrast structure that makes the V-module finding interpretable. DeBERTa-v3 module names will differ from both DistilBERT and RoBERTa; the canonical correspondence must be established at training time and recorded in the per-module subset table.

### Validation

After training, record each adapter's single-task accuracy on its evaluation set. These are the source scores. If any adapter fails to reach reasonable accuracy (within ~5% of published DeBERTa-v3-base fine-tuning results for that task), investigate before proceeding — a poorly trained adapter would contaminate the merge analysis.

---

## 2. Merge Protocol

### Pairs

All 28 pairs from 8 adapters: 6 cross-task × 4 seed combinations + 4 same-task controls.

**Cross-task pairs (24):**

| Pair | Seed combos |
|------|-------------|
| QNLI × MRPC | s42×s42, s42×s7, s7×s42, s7×s7 |
| QNLI × SST-2 | s42×s42, s42×s7, s7×s42, s7×s7 |
| QNLI × RTE | s42×s42, s42×s7, s7×s42, s7×s7 |
| MRPC × SST-2 | s42×s42, s42×s7, s7×s42, s7×s7 |
| RTE × SST-2 | s42×s42, s42×s7, s7×s42, s7×s7 |
| RTE × MRPC | s42×s42, s42×s7, s7×s42, s7×s7 |

**Same-task controls (4):**

| Pair | Description |
|------|-------------|
| QNLI s42 × QNLI s7 | Same-task seed contrast |
| MRPC s42 × MRPC s7 | Same-task seed contrast |
| SST-2 s42 × SST-2 s7 | Same-task seed contrast |
| RTE s42 × RTE s7 | Same-task seed contrast |

### Merge method

Linear merge, equal weights (α = 0.5 / 0.5). This matches the existing protocol. Do not use TIES, DARE, or other methods — the point is to test the same merge method across backbones, not to find the best merge method.

### Evaluation

For each merged adapter, evaluate on both source tasks. Record accuracy (or F1 for MRPC). Compute delta = source_score - merged_score for each task. The pair's worst-case delta is max(delta_A, delta_B).

---

## 3. Analysis Protocol

### Step 1 — DeBERTa severity classification

Apply P01 thresholds to each pair:

| Class | Threshold |
|-------|-----------|
| Catastrophic | worst Δ > 15% |
| Severe | 10% < worst Δ ≤ 15% |
| Broad | 5% < worst Δ ≤ 10% |
| Mild | worst Δ ≤ 5% |

### Step 2 — Per-pair seed statistics

For each of the 6 cross-task task pairs, compute across the 4 seed combinations:

- Mean worst-case delta
- Standard deviation
- Coefficient of variation (CV)
- Seed range (max - min of worst-case delta)
- Worst seed variant identity
- Best seed variant identity

### Step 3 — Three-backbone instability update

Recompute the composite instability score using the three-backbone formula. The two-backbone formula was:

```
instability = 0.4 × norm_seed_range + 0.3 × norm_backbone_shift + 0.3 × norm_cv
```

For three backbones, extend naturally:

```
seed_range = max(seed_range_distilbert, seed_range_roberta, seed_range_deberta)
backbone_shift = max pairwise |worst_delta_i - worst_delta_j| across backbones
cv = max(cv_distilbert, cv_roberta, cv_deberta)
```

Then normalize each to [0, 1] across the six cross-task pairs and apply the same 0.4/0.3/0.3 weights. This preserves comparability with the two-backbone scores while incorporating the new data.

### Step 4 — Prediction assessment

Evaluate each pre-registered prediction:

**Prediction A (instability ranking preserved):**
- PASS if: QNLI×MRPC and QNLI×SST-2 have the two highest DeBERTa seed ranges, each ≥ 2× the median of the other four pairs' DeBERTa seed ranges.
- FAIL if: any other pair has a higher DeBERTa seed range than both QNLI×MRPC and QNLI×SST-2.
- PARTIAL if: one of the two has the highest seed range but the other doesn't, or the 2× criterion is not met.

**Prediction B (stable cluster preserved):**
- PASS if: all four stable-asymmetric pairs have DeBERTa seed ranges below 10% and none reverses severity class relative to its two-backbone profile.
- FAIL if: any stable-asymmetric pair has DeBERTa seed range > 15% or shows backbone-reversal behavior.
- PARTIAL if: seed ranges are 10–15% or one pair's severity class changes but not dramatically.

**Prediction C (gap preserved):**
- PASS if: no pair's three-backbone instability score falls in the [0.30, 0.70] range.
- FAIL if: two or more pairs occupy the [0.30, 0.70] range.
- PARTIAL if: exactly one pair falls in the gap.

**Prediction D (V-module dimensionality mismatch survives):**

This is the sharpest prediction in the protocol, added after the per-module geometry program (n21) identified V-module dimensionality ratio as the strongest signal in the sidecar evidence base.

*Setup.* After DeBERTa merge evaluation, identify which DeBERTa cross-task pairs are catastrophic collision cases (worst Δ > 15% and layer-level alignment ≥ 0.70). Identify safe collision controls (alignment ≥ 0.70 but worst Δ ≤ 10%). Compute V-module dimensionality ratio at critical layers for each pair, following the per-module protocol (n20).

*Note on DeBERTa module mapping.* DeBERTa-v3 uses disentangled attention with separate content and position query/key projections. The canonical V-module correspondence is the content-side value projection. The exact module names must be established during adapter inspection and recorded in the subset table. If DeBERTa's disentangled architecture means the V-module plays a structurally different role, this is precisely what makes the test informative — it tests whether the signal is architecturally generic or specific to standard attention.

- PASS if: DeBERTa catastrophic collision cases have V-module dimensionality ratio < 0.75 at critical layers AND DeBERTa safe collision controls have V-module dimensionality ratio > 0.78. (These thresholds are derived from the RoBERTa result: catastrophic range 0.64–0.74, safe range 0.79–0.89.)
- FAIL if: DeBERTa catastrophic collision and safe collision cases have overlapping V-module dimensionality ratio ranges. Or if no DeBERTa cross-task pair crosses the catastrophic threshold in the collision regime (test cannot be conducted — see contingency below).
- PARTIAL if: separation exists but is weaker than on RoBERTa (Cohen's d < 1.5 or range overlap > 0.20), or if only one of the two conditions (catastrophic low, safe high) holds.

*Contingency:* If no DeBERTa pair is both catastrophic (worst Δ > 15%) and collision-prone (alignment ≥ 0.70), Prediction D cannot be directly tested. In this case, fall back to testing whether V-module dimensionality ratio correlates with severity within the collision subset (Spearman ρ between V-dim-ratio and worst Δ, expecting ρ < −0.5) and report this as a weaker test.

**Prediction E (head-level modulation recurs):**

*Added after the head-level V analysis (n24) resolved CA-01 seed sensitivity through a cancellation mechanism. This tests Rung 2 of the multiscale mechanism ladder.*

*Setup.* Requires Prediction D to PASS or PARTIAL (module-level signal must exist for head-level analysis to be meaningful). If DeBERTa produces any seed-sensitive catastrophic or near-catastrophic case (seed range > 15 percentage points within a single task pair), decompose the V module into per-head (head_dim × hidden_size) blocks at critical layers for the worst and mildest seed variants.

- PASS if: the seed-sensitive case shows head-level |Δ_DR| ≥ 0.10 at ≥ 3 head×layer positions, with opposite-sign deltas at different heads (evidence of cancellation). Module-level V dim ratio Δ between the same seed variants is < 0.05 (confirming the cancellation).
- FAIL if: seed sensitivity is visible at module level (no cancellation), or head-level deltas are uniformly small (< 0.08 everywhere), or no DeBERTa case shows sufficient seed sensitivity to test.
- PARTIAL if: head-level deltas are present but smaller than on DistilBERT, or the cancellation pattern is partial (some layers cancel, others do not).

*Contingency:* If no DeBERTa case has a seed range > 15pp, Prediction E cannot be tested. Record as UNTESTABLE — this does not falsify the mechanism, only limits the evidence to two backbones. In this case, check whether any DeBERTa case with seed range > 8pp shows the same qualitative pattern (head-level deltas larger than module-level) as a weaker signal.

**Prediction F (O-module / output escalation confirmed as next step):**

*This is a decision prediction, not a pass/fail empirical test. It defines what the DeBERTa results imply for Rung 3.*

The joint outcome of D and E determines the priority of O-module analysis:

| D outcome | E outcome | Rung 3 implication |
|:---------:|:---------:|:-------------------|
| PASS | PASS | **Confirmed.** All three rungs of the ladder transfer. O-module analysis is the clear next step: extract per-head output weights from the O-module LoRA product to test whether catastrophic head configurations are selectively amplified. |
| PASS | FAIL or UNTESTABLE | **Likely but unconfirmed.** Module-level risk transfers; head-level modulation may be backbone-specific. O-module analysis should be attempted on DeBERTa but the prediction about head-selective amplification is weaker. Consider whether DeBERTa's disentangled attention changes the head-weighting mechanism. |
| FAIL | — | **Deferred.** Module-level signal does not transfer. The escalation target should be the disentangled attention structure itself, not the O module. Investigate whether DeBERTa's content/position separation moves the discriminative signal to a different module. |

**Prediction G (spectral partitioning remains task-discriminating):**

*Added after the April 2026 literature integration (CHG-004). This tests whether Gradience's energy-weighted compatibility metrics survive DeBERTa's non-standard pretraining objective (replaced token detection rather than masked language modeling).*

*Setup.* Using the Gavish-Donoho optimal hard threshold as the partition point (the same method used in N127), compute SV-weighted alignment separately in the high-SV and low-SV bands for all 28 DeBERTa adapter pairs.

- PASS if: DeBERTa same-task adapter pairs show SV-weighted alignment in the high-SV band at least 2.5× higher than cross-task pairs (consistent with the Mistral-7B result; N127 found 7.8× on DistilBERT-base), with the difference significant at p < 0.01.
  Formally: $\text{SV-weighted alignment}_{high, same} / \text{SV-weighted alignment}_{high, cross} \geq 2.5$.
- FAIL if: Ratio < 2.0 and/or p > 0.01 for the same-task vs. cross-task comparison in the high-SV band. If falsified, the energy-weighted interaction bound may be insufficient for architectures with non-standard pretraining objectives, and tail-aware interference detection becomes the next priority experiment.
- PARTIAL if: Ratio is between 2.0 and 2.5, or significance is borderline (0.01 < p < 0.05).

*Pre-training status (N128, April 2026).* N128 found zero false-negative candidates in the encoder validation corpus. H0 confirmed — tail-band interference is not operationally urgent at encoder scale. Prediction G stays as written; no sharpening or module-specific targeting needed per CHG-004 Decision A.

*Note.* The prediction is specifically that the task-discriminating partition survives even if DeBERTa's pretraining objective shifts the absolute location of the Marchenko-Pastur bulk edge. The N127 result (7.8× on DistilBERT) used MLM-pretrained weights; DeBERTa-v3 uses replaced token detection, which may produce a different spectral structure in $W_0$. The test is whether the *relative* separation (same-task vs. cross-task) persists, not whether the absolute alignment magnitudes match.

### Step 5 — Per-module geometry analysis (DeBERTa)

This step is new. It produces the data needed for Prediction D and extends the per-module evidence base to three backbones.

**5a.** Save all 8 DeBERTa adapter weights in safetensors format to `sidecar/results/s01/deberta/sources/`.

**5b.** Run the per-layer analysis (compute_metrics.py) on DeBERTa adapters. Record norm mass distributions and layer-level alignment for all cross-task pairs. Classify each pair into the collision subset scheme (n16).

**5c.** Establish the DeBERTa module correspondence. DeBERTa-v3-base uses disentangled attention; identify the four target modules and their canonical Q/K/V/O mappings. Record in an updated per-module subset table.

**5d.** Run the per-module geometry analysis (per_module_geometry.py, adapted for DeBERTa) on all cross-task pairs. Compute all four metrics per module at critical layers.

**5e.** Evaluate Prediction D using the results from 5d.

**5f.** Compare DeBERTa per-module profiles to the existing DistilBERT and RoBERTa profiles. Does the V-module signal pattern generalize? Does any other module emerge as a discriminator on DeBERTa?

### Step 5b — Head-level V analysis (DeBERTa)

*This step is conditional on Prediction D achieving PASS or PARTIAL.* It produces the data needed for Prediction E and extends the head-level evidence base to three backbones.

**5b-i.** Identify DeBERTa's head dimension. DeBERTa-v3-base uses 12 heads with hidden size 768, yielding 64 dimensions per head (matching DistilBERT and RoBERTa). Confirm this from the model config and record.

**5b-ii.** If any DeBERTa cross-task pair shows seed range > 8pp, designate it as the head-level test case. Identify worst and mildest seed variants.

**5b-iii.** Run the head-level V analysis (v_head_geometry.py, adapted for DeBERTa) on the test case and at least one safe collision control. Compute per-head dim ratio, top overlap, and directional conflict at critical V layers.

**5b-iv.** For the seed-sensitive test case, compare worst vs. mild variant per head. Record head-level Δ_DR and Δ_OV. Test for cancellation: do opposite-sign deltas appear at different heads?

**5b-v.** Evaluate Prediction E using the results from 5b-iv.

**5b-vi.** If both D and E PASS, record the DeBERTa head-level pattern and compare to the DistilBERT/RoBERTa patterns. Assess whether the cancellation mechanism is backbone-generic or backbone-specific, and whether DeBERTa's disentangled attention produces any qualitatively different head-level structure.

### Step 6 — Taxonomy update

Based on the three-backbone data, update the taxonomy:

| Current class | Two-backbone criteria | Three-backbone update rule |
|--------------|----------------------|---------------------------|
| Backbone reversal | instability > 0.7, classes differ across backbones | Maintain if instability > 0.7 on three-backbone composite; relabel to "unstable severe" if classes don't reverse but instability remains high |
| Unstable severe | instability > 0.5, high variance but no full reversal | Populate if DeBERTa data shows high instability without full class reversal |
| Stable asymmetric | instability < 0.3, consistent class across backbones | Maintain if instability < 0.3 on three-backbone composite |
| Stable mild | instability < 0.1, minimal degradation | Populate if any cross-task pair shows < 5% worst delta on all three backbones |

---

## 4. Output Specification

All outputs go to `sidecar/results/s01/deberta/`:

| File | Contents |
|------|----------|
| `adjudication_results.json` | Raw per-pair results (same schema as existing) |
| `severity_classification.json` | Per-pair DeBERTa severity class |
| `seed_statistics.json` | Per-pair seed range, CV, mean, std |
| `source_scores.json` | Per-adapter single-task accuracy |
| `sources/` | All 8 adapter weights in safetensors format |

Per-module outputs go to `sidecar/results/per_module_geometry/deberta/`:

| File | Contents |
|------|----------|
| `module_metrics.json` | Per-variant, per-module metrics (same schema as RoBERTa/DistilBERT) |
| `group_module_comparison.json` | Group-level per-module summary |
| `module_discrimination.json` | Cohen's d and overlap for Prediction D evaluation |
| `collision_subset_deberta.json` | DeBERTa collision classification |

Head-level outputs go to `sidecar/results/head_level_v/deberta/` (conditional on D PASS/PARTIAL):

| File | Contents |
|------|----------|
| `head_metrics.json` | Per-variant, per-head V metrics (same schema as existing) |
| `seed_sensitivity_per_head.json` | Per-head seed variant deltas for Prediction E evaluation |
| `head_discrimination.json` | Head-level discrimination (if applicable) |

Updated cross-backbone outputs go to `sidecar/results/s01/`:

| File | Contents |
|------|----------|
| `three_backbone_comparison.json` | Updated with DeBERTa column |
| `instability_profiles.json` | Updated with three-backbone composite |
| `instability_case_table.md` | Updated with DeBERTa data |
| `taxonomy.json` | Updated with any new classifications |

New figures:

| File | Contents |
|------|----------|
| `figures/s01_three_backbone_comparison.svg` | Three-column severity comparison |
| `figures/s01_instability_three_backbone.svg` | Updated instability scatter with DeBERTa |

---

## 5. Decision Tree

After completing the analysis, the decision depends on the joint outcome of the instability predictions (A–C), the module-level structural prediction (D), and the head-level modulation prediction (E). These are logically independent: A–C test a retrospective descriptor (requires merge evaluation), D tests a structural predictor at module scale (Rung 1), and E tests a distributional mechanism at head scale (Rung 2). Prediction F (Rung 3 escalation) is a decision criterion, not a pass/fail test.

### Strongest outcome: A–C PASS + D PASS + E PASS

**Conclusion:** The full multiscale mechanism ladder transfers to a third backbone with disentangled attention. Instability is a portable descriptor. V-module dim ratio is a portable structural predictor. Head-level cancellation is an architecturally generic modulation mechanism. The O-module amplification hypothesis (Rung 3) is the confirmed next step.

**Actions:**
1. Update n06 (program statement) to record the confirmation of all three rungs
2. Write a promotion assessment for V-module dimensionality ratio as a computable warning signal
3. Begin prototype implementation in core Gradience
4. Design the O-module head-weight analysis (Rung 3 test) using DeBERTa adapters
5. Write up the multiscale mechanism as a standalone finding

### D PASSES, E FAILS or UNTESTABLE

**Conclusion:** Module-level risk transfers but head-level modulation may be backbone-specific. The V-module dim ratio is still a portable predictor of group-level risk. Head-level cancellation may depend on standard attention architecture (DistilBERT/RoBERTa) and not apply to disentangled attention.

**Actions:**
1. Proceed with V-module promotion assessment (Rung 1 is sufficient for a useful warning signal)
2. Investigate whether DeBERTa's disentangled structure produces a different modulation pattern — the head-level story may need revision for non-standard architectures
3. O-module analysis is still plausible but the prediction about head-selective amplification is weaker; consider targeting DeBERTa's position/content separation as an alternative Rung 2

### D PASSES, A–C mixed

**Conclusion:** V-module dimensionality mismatch is portable even though the composite instability formula needs recalibration. The structural signal is more robust than the retrospective descriptor.

**Actions:**
1. Prioritize V-module signal over instability ranking as the sidecar's primary finding
2. Diagnose which instability component failed and whether recalibration recovers the ranking
3. Write promotion assessment focused on V-module dimensionality ratio
4. The V-module signal does not depend on instability — it is computable from a single pair of adapters on a single backbone

### A–C PASS, D FAILS

**Conclusion:** Instability rankings are portable, but the V-module signal does not generalize to disentangled attention. The mechanistic hypothesis needs revision; the descriptor stands.

**Actions:**
1. Record instability confirmation
2. Investigate whether DeBERTa's disentangled attention moves the discriminator from V to another module (content query? position key?)
3. Run the full per-module discrimination analysis on DeBERTa to identify whether any module replaces V
4. Write promotion assessment for instability as a descriptor; defer structural predictor work
5. E is moot if D fails — do not run head-level analysis

### G FAILS (partition not task-discriminating on DeBERTa)

**Conclusion:** The energy-weighted compatibility metrics do not generalize to architectures with non-standard pretraining objectives. The spectral partition found on DistilBERT (N127) and partially echoed on Mistral-7B may depend on MLM-style pretraining producing a specific spectral structure in $W_0$.

**Actions:**
1. Do not generalize Gradience's energy-weighted compatibility metrics to DeBERTa-architecture adapters until the source of the failure is understood.
2. Escalate `scripts/tail_interference_probe.py` (CHG-005) to the next GPU session: run it on DeBERTa adapters specifically rather than the existing encoder corpus.
3. Add "tail-aware compatibility metric" to the v0.12.0 spec as a high-priority research item, cross-referencing THEORY.md §7.2 ("Tail-band interference as an independent compatibility signal").
4. The DeBERTa study result (G failed) should be reported in the technical report under §7.1 "DeBERTa Adjudication" with the decision tree branch taken, rather than treating the failure as a study-terminating event.
5. Investigate whether DeBERTa's replaced-token-detection pretraining produces a qualitatively different $W_0$ spectral structure (e.g., less concentrated energy in the top-k subspace), which would explain the partition failure.

### A FAILS (regardless of D and E)

**Conclusion:** Instability rankings are not portable. The sidecar's working concept needs fundamental revision.

**Actions:**
1. Write a closure note documenting the failure
2. If D PASSED: V-module dimensionality mismatch is still a valid structural signal even without instability portability — it may identify catastrophe-prone pairs on a per-backbone basis
3. If D also FAILED: the sidecar's mechanistic picture is substantially weakened; reassess whether the evidence base supports continued investigation at this resolution
4. Do not abandon the sidecar — the research question remains important even if the current framework is wrong

---

## 6. Estimated Compute Budget

| Step | GPU hours (estimated) |
|------|----------------------|
| Train 8 adapters | ~1.5 hours |
| Merge 28 pairs | ~0.5 hours |
| Evaluate 56 conditions (28 pairs × 2 tasks) | ~1 hour |
| **Total** | **~3 hours on a single consumer GPU** |

This is a modest compute investment for a decisive test. The protocol should be executed in a single session to avoid configuration drift.

---

## 7. Pre-Run Checklist

Before executing:

- [ ] Confirm DeBERTa-v3-base is available via HuggingFace
- [ ] Verify LoRA configuration: rank 16, all four attention modules targeted. Identify DeBERTa-v3's module names for Q/K/V/O and record the canonical correspondence
- [ ] Confirm LoRA is compatible with DeBERTa's disentangled attention (content and position projections are separate — verify which projections are targeted and document)
- [ ] Confirm GLUE datasets are accessible
- [ ] Set up output directory structure: `sidecar/results/s01/deberta/` and `sidecar/results/per_module_geometry/deberta/`
- [ ] Review this protocol and confirm no design changes are needed
- [ ] Record the exact model revision hash for reproducibility
- [ ] Verify that per_module_geometry.py can be adapted for DeBERTa module names (may need a `DEBERTA_CONFIG` entry in `BACKBONE_CONFIG`)
- [ ] Verify that v_head_geometry.py can be adapted for DeBERTa (confirm head dimension = 64, add `DEBERTA_V_KEY` template)

---

## 8. What This Protocol Does Not Cover

- **O-module head-weight analysis (Rung 3 test):** This protocol identifies *whether* O-module analysis is the right next step (Prediction F). It does not execute the analysis itself. If D+E PASS, a separate protocol for extracting per-head output weights from the O-module LoRA product is needed.
- **Promotion decision:** This protocol produces the data; the promotion assessment is a separate document.
- **Alternative merge methods:** This protocol uses linear merge only. Testing other methods is a future study.
- **Rank variation:** All adapters use LoRA rank 16 (matching the existing evidence base). Varying rank is a future study.
- **Head-level decomposition of modules other than V:** The protocol tests head-level V only. If D FAILS and the discriminator shifts to a different module on DeBERTa, that module's head-level analysis would be a follow-on study.
