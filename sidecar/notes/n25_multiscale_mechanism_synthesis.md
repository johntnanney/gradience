# Note: Multiscale Mechanism Synthesis

## Metadata

- **Type:** synthesis
- **Date:** 2026-03-26
- **Related notes:** n18 (within-layer findings), n21 (per-module findings), n24 (head-level findings), n07 (DeBERTa adjudication protocol)
- **Project:** Mechanistic picture integration
- **Status:** This is the single best "where the sidecar stands" note as of March 2026.

---

## Purpose

This note integrates the three resolution levels at which the sidecar has now examined within-layer geometry — aggregate (n18), per-module (n21), and per-head (n24) — into a single mechanistic picture. Each level explains something the others cannot. The note states what is resolved at each scale, what remains open, and where the causal chain breaks.

---

## 1. The Three Scales

### 1.1 Aggregate within-layer (n18): MIXED/NEGATIVE

The concatenated Q/K/V/O matrix per layer was the first attempt to find within-layer geometric conditions that distinguish catastrophic from safe collision. It failed. When backbone is controlled, catastrophic cases are indistinguishable from safe controls on all four metrics (principal angles, top overlap, dimensionality ratio, directional conflict). The separation visible in raw data was a backbone confound: DistilBERT's 6 layers force tighter subspace alignment than RoBERTa's 12 layers.

**What this level explains:** Nothing about the catastrophic threshold. It explains only that the signal is not in aggregate within-layer geometry.

**What it rules out:** The "global subspace misalignment" hypothesis — the idea that catastrophic pairs would show systematically different overall geometry from safe pairs when backbone is held constant.

### 1.2 Per-module (n21): POSITIVE

Decomposing the same layers into separate Q, K, V, and O modules recovered a clean signal that the aggregate analysis had diluted. The V-module dimensionality ratio separates catastrophic from safe collision with d = 3.36 and zero range overlap (catastrophic 0.64–0.74, safe 0.79–0.89). The K module shows a secondary signal (d = 1.39). Q and O are non-discriminating.

**What this level explains:**

- **Group discrimination.** Which pairs are catastrophic vs. safe. The V-module dim ratio is the strongest correlate of the catastrophic threshold in the entire sidecar evidence base.
- **CA-02 seed sensitivity (partially).** The toxic adapter (qnli_s42) shows lower V-module dim ratio and lower O-module alignment than the benign adapter (qnli_s7). The module-level signals are directionally consistent and large enough to detect.

**What it does not explain:**

- **CA-01 seed sensitivity.** The 29-point severity gap produces module-level V deltas below 0.07 — indistinguishable from noise at this resolution.
- **Why the threshold exists.** The dim ratio is a correlate, not a mechanism. Why does a particular ratio value trigger catastrophe?

### 1.3 Per-head (n24): MIXED-POSITIVE

Slicing the V-module's (768, 768) product matrix into 12 per-head (64, 768) blocks reveals the structure hidden within the module aggregate.

**What this level explains:**

- **CA-01 seed sensitivity.** The mystery that survived two prior resolution levels is now resolved. Seven head×layer positions show |Δ_DR| ≥ 0.15 between the worst and mild seed variants (maximum 0.229 at layer 3 head 6). The module-level aggregate was near zero because individual heads show deltas of *opposite sign* — some heads become more compatible under the catastrophic seed while others become less compatible. The net catastrophic outcome depends on which heads' incompatibilities dominate the downstream output.
- **The cancellation mechanism.** The module-level dim ratio is a mean over 12 heads. Two different seed configurations can produce the same mean but very different distributions. The catastrophic configuration concentrates incompatibility at heads that matter for the downstream prediction; the mild configuration distributes it across heads that do not.
- **CA-02 head-level amplification.** The toxic adapter's V-module signal, already visible at module level, concentrates at layer 4 with individual heads showing Δ_DR up to -0.459, 4.5× the module-level delta.

**What it does not explain:**

- **Group discrimination.** Head-level resolution *weakens* discrimination (d drops from 3.36 to 1.25). The module-level signal gains its power from cross-head consistency — all 12 heads show the same directional pattern (catastrophic < safe). Averaging reinforces the shared signal and suppresses head-specific noise.
- **Which heads matter for the output.** The cancellation mechanism identifies *where* the seed sensitivity lives but not *why* certain head configurations produce catastrophe. That depends on the O module and classification head, which weight heads' contributions to the final prediction.

---

## 2. The Integrated Picture

The three scales form a coherent mechanistic hierarchy:

```
Scale               | What it measures          | What it explains         | d (dim ratio)
--------------------|---------------------------|--------------------------|-------------
Aggregate (n18)     | Concatenated Q/K/V/O      | Nothing (confounded)     | n/a
Per-module (n21)    | Separate Q, K, V, O       | Group discrimination     | 3.36
Per-head (n24)      | 12 heads within V         | Seed sensitivity         | 1.25
```

The hierarchy has a clear scaling pattern: each decomposition trades discrimination power for explanatory depth. The module level is the *right* granularity for classification — it provides the cleanest separation with the least noise. The head level is the *right* granularity for understanding variability — it reveals the distributional mechanisms that module-level statistics compress.

This is not a failure of the head-level analysis. It is a structural feature of how averaging interacts with signal and noise at different scales. The module-level dim ratio aggregates a consistent directional signal (all heads point the same way for group differences) while averaging out inconsistent distributional noise (heads point different ways for seed differences). Head-level analysis recovers the distributional information at the cost of amplifying the noise.

---

## 3. The Multiscale Mechanism Ladder (Glossary Term)

The sidecar's evidence now supports a three-rung mechanism ladder (see glossary for the frozen definition). Collision loading (n15–n16) is a necessary precondition that gates entry to the ladder. The three rungs are:

**Rung 1 — Module-level risk (n21).** V-module dimensionality ratio separates catastrophic from safe collision with d = 3.36 and zero range overlap. This is a *group discriminator*: it identifies which pairs are at risk. Aggregation across 12 heads improves the signal by averaging out head-level noise while preserving a consistent directional pattern (all heads show catastrophic < safe). Detectable from adapter structure alone.

**Rung 2 — Head-level modulation (n24).** The same module-level dim ratio can produce catastrophe or not, depending on how mismatch distributes across attention heads. Different training seeds produce different head-level configurations that average to the same module value. Opposite-sign deltas at different heads cancel at module level, explaining why CA-01's 29-point severity gap is invisible at Rung 1 but visible at Rung 2. Head-level geometry is a *modulator of severity within a risk class*, not a discriminator between classes.

**Rung 3 — Downstream amplification (hypothesized).** The O module and classification head determine which heads' incompatibilities manifest as classification errors. This rung explains *why* certain head-level configurations are catastrophic while others with the same module average are mild: the output pathway selectively amplifies or attenuates specific heads' contributions. Not yet tested empirically.

The ladder is nested: Rung 1 sets the precondition, Rung 2 modulates the outcome, Rung 3 determines whether the outcome is catastrophic. Rungs 1–2 are established (empirical, two backbones). Rung 3 is hypothesized.

---

## 4. What Remains Open

### 4.1 The amplification mechanism (Link 4)

The highest-priority open question. The O module's per-head weighting could be extracted from the O-module LoRA product matrix. If the O module amplifies heads where V-module mismatch is concentrated (in the catastrophic seed configuration) but not where it is dispersed (in the mild configuration), that would close the causal chain. This analysis is CPU-feasible with existing adapters but has not yet been attempted.

### 4.2 DeBERTa adjudication of the mechanism ladder

The entire ladder (Rungs 1–2) is established on two backbones. DeBERTa-v3 has structurally different attention (disentangled content and position), making it the sharpest adjudication test. The DeBERTa protocol (n07) now tests each rung independently:

- **Prediction D (Rung 1):** Does the module-level V dim ratio signal survive on disentangled attention?
- **Prediction E (Rung 2):** Does head-level cancellation / modulation recur on any seed-sensitive DeBERTa case?
- **Prediction F (Rung 3):** The joint D×E outcome determines whether O-module analysis is the confirmed next escalation, or whether DeBERTa's architecture demands a different approach.

This is the decisive test. If D+E pass, the mechanism ladder is architecturally generic. If D passes but E fails, Rung 2 may be backbone-specific. If D fails, the signal may shift to a different module under disentangled attention.

### 4.3 Predictive use

The V-module dim ratio is a correlate, not yet a validated predictor. Turning it into a usable signal requires: (a) confirming the threshold on DeBERTa, (b) understanding whether the dim ratio alone suffices for risk classification or whether the head-level distribution also carries predictive weight, and (c) calibrating against a wider range of task pairs.

### 4.4 The Q and K modules

The K module shows a secondary discrimination signal (d = 1.39 at module level). The Q module shows weak signals. Neither has been decomposed to head level. If the V-module story proves insufficient on DeBERTa, Q and K head-level analysis would be the next step.

### 4.5 Whether head-level descriptors add predictive value

The module-level dim ratio cannot distinguish CA-01's catastrophic seed from its mild seed. If a predictive system needs to flag *which seed configurations* are dangerous (not just which task pairs), it would need a head-level descriptor — perhaps max |Δ_DR| across heads, or the variance of head-level dim ratios, or the entropy of the head-level dim ratio distribution. Whether such a descriptor generalizes beyond CA-01 is unknown.

---

## 5. Relationship to Canonical Concepts

### V-module dimensionality mismatch (glossary)

Unchanged by the head-level analysis. The canonical finding and interpretation remain valid at the module level. The head-level refinement adds explanatory depth (the cancellation mechanism, the seed sensitivity localization) without displacing the module-level result.

### Thresholded subspace interference (glossary)

Refined. The "threshold" is now understood as the multiscale mechanism ladder: Rung 1 (module-level dim ratio) provides the aggregate risk precondition, Rung 2 (head-level dim ratio distribution) modulates whether a given module-level value actually triggers catastrophe, and Rung 3 (downstream amplification) determines the functional outcome. The threshold is not a single number but a function of the interaction across all three rungs.

### Instability (n05)

The mechanism ladder gives instability a geometric interpretation: an unstable pair is one where Rung 1 is satisfied (module-level mismatch puts the pair at risk) but the Rung 2 outcome is seed-dependent (small parameter changes reconfigure head-level geometry, crossing or uncrossing the catastrophic threshold at specific heads). The module average is unchanged, but the functional outcome depends on how Rung 3 weights the reconfigured heads — which is exactly what "instability" measures at the behavioral level.

### Multiscale mechanism ladder (glossary)

This note is the primary reference for the ladder concept. The glossary entry provides the frozen definition; this note provides the evidence, the reasoning, and the open questions.

---

## 6. Summary Table

| Question | Scale | Status | Evidence |
|:---------|:------|:-------|:---------|
| Which pairs are catastrophic? | Module | **Resolved** | V dim ratio d=3.36, zero overlap (n21) |
| Why does CA-02 have a toxic adapter? | Module | **Partly resolved** | V Δ_DR=-0.10, O Δcos=-0.31 (n21) |
| Why does CA-01 show 29pp seed gap? | Head | **Resolved** | 7 hot heads, cancellation mechanism (n24) |
| Why do specific head configs cause catastrophe? | Downstream | **Open** | Requires O-module / task head analysis |
| Does the signal survive DeBERTa? | Module + Head | **Blocked on GPU** | Protocol ready (n07) |
| Is V dim ratio a usable predictor? | Module | **Open** | Requires DeBERTa + calibration |
| Do head-level descriptors add predictive value? | Head | **Open** | Requires wider task pair sample |
