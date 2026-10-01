# Gradience Research Scan — October 1, 2026

**Scope**: New empirical research and technical documentation highly relevant to Gradience's spectral analysis framework, identified via systematic search across LoRA spectral analysis, phase transitions in training dynamics, Hessian/curvature telemetry, and subspace merge compatibility.

**Exclusions**: Papers already tracked in `docs/RESEARCH_INVENTORY.md` §9 (Xu 2026, Roi Paul 2026, Spectral Surgery, KnOTS, TSV, Akbar et al., Badirli et al., Tian et al., Medina & Sørensen, Hitit et al.) were filtered out.

---

## 1. The Intruder Threshold: A Spectral Law for LoRA Fine-Tuning

**Citation**: Peng Xie (TU Munich). arXiv:2607.23711, July 2026.

**Summary**: Derives a parameter-free, per-layer critical update strength s\* = θ̄/(γσ₁(BA)) from the pretrained weight spectrum alone, via the rectangular spiked-deformation transform, that predicts when LoRA fine-tuning creates "intruder dimensions" — new leading singular vectors of W+BA nearly orthogonal to all pretrained singular vectors. These intruder dimensions drive catastrophic forgetting. Validated on 4 dense Transformer families + a state-space model + a mixture-of-experts model + an encoder-decoder (18 adapters, 9,840 layer scans): the law localizes the empirical threshold within 2× on 82% of layers, achieves mean AUC 0.89 for classifying intruder-bearing vs. intruder-free layers, generalizes to 6 third-party adapters, and predicts WikiText-2 perplexity degradation onset.

**Relevance to Gradience**:

- **Core audit pipeline**: The per-layer critical threshold s\* is a formal, parameter-free analog of Gradience's UDR metric (Update Dominance Ratio = ‖ΔW‖₂/‖W_base‖₂). Where UDR is a heuristic magnitude comparison, s\* is derived from the spiked random matrix model and predicts the qualitative transition point (intruder emergence) rather than just relative magnitude. This could augment or replace UDR as a per-layer diagnostic.

- **Marchenko-Pastur partition refinement**: The spiked-deformation transform generalizes the Gavish-Donoho optimal hard threshold currently used in Gradience's spectral partition (N127). Where the GD threshold separates signal from noise in a static matrix, the spiked-deformation transform characterizes signal emergence in the perturbed matrix W+BA — the operationally relevant object for merge compatibility.

- **Tail-band interference (THEORY.md §7.2)**: Intruder dimensions are the formal mechanism by which low-SV directions become operationally dominant — a new leading SV orthogonal to pretrained structure is, by definition, a direction that was in the "noise band" of W₀ and has become the dominant band of W+BA. This provides theoretical grounding for the open question about whether low-SV interference can produce merge degradation.

- **Structural-behavioral bridge (Study 16 gap)**: The paper connects spectral structure (intruder threshold) to behavioral outcome (perplexity degradation), partially addressing the structural ≠ behavioral gap that Study 16 identified. An intruder-aware audit could predict *which* structurally compatible pairs will fail behaviorally.

- **Practical integration path**: s\* is computable from the pretrained weight spectrum and the LoRA update's dominant singular value — both quantities Gradience's audit pipeline already extracts. Adding an intruder-threshold check to the existing per-layer audit is a modest engineering task with potentially high diagnostic value.

**Integration priority**: **HIGH — replicate on Gradience's validation corpus, then integrate**.

The natural first step is to compute s\* for every adapter in the existing validation corpus (DistilBERT, DeBERTa, Mistral-7B) and verify the AUC claim against known merge outcomes. If it replicates, add s\* as a sixth rank policy / diagnostic quantity in the audit pipeline.

**Caveats**: Single author, July 2026, no peer review yet. The rectangular spiked-deformation transform derivation should be verified against established random matrix theory results before building infrastructure.

---

## 2. Spectral Phase Transitions and Trainability in Neural Network Learning Dynamics

**Citation**: Chanju Park, Dario Bocchi, Francesco D'Amico, Biagio Lucini, Gert Aarts. arXiv:2606.28486, June 2026.

**Summary**: Formulates neural network training as the stochastic evolution of an initially random matrix ensemble driven by SGD updates, identifying a Baik-Ben Arous-Péché (BBP) transition during training where isolated eigenvalues detach from the random bulk distribution. Demonstrates this in a solvable linear teacher-student model where spectral evolution is analytically tractable, obtains a phase diagram of trainability governed by learning rate and initial weight variance, and extends the formalism to nonlinear and stochastic settings. Numerical simulations in realistic settings show robust emergence of spectral alignment during training.

**Relevance to Gradience**:

- **Phase transition framework (THEORY.md §4)**: The BBP transition is precisely the formal dynamical mechanism for the spectral phase transitions Gradience's `research/phase_transitions.py` module detects heuristically (autocorrelation time divergence, variance ratio spikes). The paper provides the first analytically tractable model of *why* isolated eigenvalues emerge from noise during training — the phenomenon Gradience measures empirically as energy concentration sharpening (56% → 86% in N127, Finding §14).

- **DFA regime classification (Finding §18)**: The paper's phase diagram (learning rate × initial weight variance → trainability) maps directly onto Gradience's DFA exponent findings: high learning rate produces α ≈ 1.57 vs. ~1.90–2.07 for other regimes (F = 116.86, p ≈ 10⁻²³). The BBP framework predicts that high learning rate should place training in a qualitatively different spectral regime — consistent with the DFA result showing high-LR disrupts temporal persistence.

- **Curvature-partition correspondence (THEORY.md §7.2)**: The BBP transition offers a unified mechanism: eigenvalue detachment from the bulk governs both when Hessian eigenvalues reorganize (curvature change) and when the SVD partition sharpens (signal separation). This addresses the open question about whether curvature collapse and spectral partition sharpening are "the same event observed through different instruments" — in the BBP framework, they literally are.

- **Xu's Spectral Edge Thesis complement**: Park et al. work from spiked random matrix models (statistical physics tradition), while Xu works from spectral gap dynamics of Gram matrices (optimization theory tradition). Both arrive at structurally compatible conclusions about low-dimensional spectral structure governing training dynamics. Having two independent theoretical frameworks converging on the same phenomenology strengthens both and provides cross-validation opportunities.

- **Marchenko-Pastur partition grounding**: The BBP transition formally characterizes when and why the Marchenko-Pastur edge (used in Gradience's `optimal_hard_threshold` policy) separates meaningful from noise eigenvalues during training. Gradience applies this partition post-hoc; the BBP framework explains the dynamical origin of why it works.

**Integration priority**: **HIGH — track and test predictions against existing data**.

The natural first step is to test the phase diagram predictions (LR × init variance → trainability class) against Gradience's Study 12 data (49 runs, 5 regimes × 10 seeds, NanoGPT) where DFA exponents are already computed. If the BBP phase boundaries align with the DFA regime boundaries, the paper provides the formal framework THEORY.md §4 identifies as needed. A more ambitious step would be to reconstruct the spectral evolution dynamics from N07 Experiment B's checkpointed parameter trajectory and look for the BBP transition explicitly.

**Caveats**: Multi-author group (lattice QCD / statistical physics background), preprint June 2026, no peer review yet. The extension from solvable linear models to nonlinear settings is numerical, not analytical — the formal results are strongest in the linear regime.

---

## 3. Subspace Geometry Governs Catastrophic Forgetting in Low-Rank Adaptation

**Citation**: Brady Steele (Georgia Institute of Technology). arXiv:2603.02224, March 2026.

**Summary**: Presents a geometric theory of catastrophic forgetting in LoRA via gradient subspace interactions, deriving the law F = α(1 − cos²θ_min) + β, where θ_min is the minimum principal angle between task gradient subspaces. Reveals an approximate rank-invariance property: at high subspace angles, forgetting is largely independent of adapter rank. Validated on synthetic tasks (r = 0.994 correlation), Split-CIFAR100 with ViT-LoRA, and sequential GLUE with RoBERTa-LoRA.

**Relevance to Gradience**:

- **Principal angle → outcome formalization**: Gradience's entire merge triage pipeline rests on the empirical observation that principal angles between adapter subspaces predict merge compatibility (Finding §7: r = 0.846 on Mistral-7B; Finding §8: high overlap → safe merge). Steele's paper formalizes the exact functional form: forgetting scales as (1 − cos²θ_min). This is the first closed-form law connecting principal angles (Gradience's core measured quantity) to behavioral outcomes, applicable to sequential fine-tuning and, by extension, to the interference dynamics relevant in merge contexts.

- **Rank-invariance at high angles**: The finding that forgetting becomes approximately rank-independent when task subspaces are sufficiently different has a direct implication for Gradience's triage: above a principal-angle threshold, the nominal rank of the adapter (which Gradience knows is poorly utilized — Finding §3: mean utilization 0.172) becomes irrelevant for predicting interference. This could simplify triage rules for cross-task pairs where angles are naturally high.

- **Concentration-weighted convergence bound (THEORY.md §7.2)**: Steele's geometric law is stated in terms of cos²θ_min, not spectral mass concentration. But the relationship between the two is well-defined: spectral mass concentration in W₀ controls θ_min (Finding §13: C_k predicts alignment for QNLI). Composing Steele's forgetting law with a concentration → θ_min bound would yield the full chain: W₀ spectral concentration → adapter alignment → merge/forgetting outcome. This brings the open convergence bound problem closer to closure.

- **Empirical testability on existing data**: Gradience already computes principal angles for every merge-audit pair. Computing θ_min per pair and testing the F = α(1 − cos²θ_min) + β prediction against known merge outcomes (Study 16's 5 Llama-2-7B pairs, N07's 52 DeBERTa merge evaluations) is immediate.

**Integration priority**: **MEDIUM-HIGH — replicate on existing merge corpus, cite in THEORY.md**.

The natural first step is to extract θ_min from existing merge-audit data (N07 DeBERTa: 28 pairs with 52 evaluations, Study 16: 5 Llama-2-7B pairs with known outcomes) and test whether F = α(1 − cos²θ_min) + β holds for merge quality degradation, not just sequential forgetting. If the functional form transfers from forgetting to merging, it provides the quantitative prediction Gradience's merge triage currently lacks.

**Caveats**: Single author, March 2026, not yet peer-reviewed. The forgetting→merging transfer is not demonstrated in the paper; merge interference and sequential forgetting share geometric structure (principal angles) but differ mechanistically (simultaneous combination vs. sequential overwriting). The r = 0.994 synthetic result is strong; the GLUE results on RoBERTa are less detailed.

---

## Summary Table

| # | Paper | arXiv | Integration Priority | Key Connection |
|---|-------|-------|---------------------|----------------|
| 1 | The Intruder Threshold (Xie) | 2607.23711 | **HIGH — replicate, then integrate** | Per-layer spectral law for LoRA; formal basis for UDR/MP partition; addresses tail-band interference open question |
| 2 | BBP Phase Transitions (Park et al.) | 2606.28486 | **HIGH — track, test against Study 12 data** | Formal dynamical framework for spectral phase transitions; explains DFA regime dependence; addresses curvature-partition correspondence |
| 3 | Subspace Geometry → Forgetting (Steele) | 2603.02224 | **MEDIUM-HIGH — replicate on merge corpus** | Closed-form principal-angle → outcome law; rank-invariance at high angles; advances concentration-weighted convergence bound |

---

## Recommended Next Steps

1. **Intruder Threshold**: Compute s\* for existing adapter corpus (DistilBERT N127, DeBERTa N07, Mistral-7B Study 16). Check whether intruder-bearing layers correlate with known merge pathologies. If AUC replicates, open a design doc for integration as a per-layer diagnostic.

2. **BBP Phase Transitions**: Test phase diagram predictions against Study 12's 49-run × 5-regime DFA data. Specifically: does the BBP critical learning rate match the DFA exponent discontinuity at high-LR?

3. **Subspace Geometry Law**: Extract θ_min from N07's 28 merge-audit pairs, fit F = α(1 − cos²θ_min) + β against merge quality degradation. Report whether the forgetting law transfers to the merge setting.

All three papers should be added to `docs/RESEARCH_INVENTORY.md` §9 (External Reference Literature) with appropriate integration priorities and honest caveats about replication status.
