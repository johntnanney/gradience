# Research Briefing: Spectral LoRA Literature Scan

**Date:** 2026-11-11
**Scope:** New empirical research and technical documentation relevant to Gradience's spectral analysis framework for LoRA fine-tuning dynamics.

---

## 1. The Intruder Threshold: A Spectral Law for LoRA Fine-Tuning

**Authors:** Peng Xie, Amr Alanwar
**Source:** [arXiv:2607.23711](https://arxiv.org/abs/2607.23711) (July 2026)

### Summary

This paper derives a closed-form, per-layer critical update strength from the pretrained weight spectrum alone. When a LoRA update exceeds this threshold in a given layer, it creates "intruder dimensions" — new leading singular vectors in the updated weight matrix that displace the pretrained model's top directions. These intruder dimensions are identified as the primary mechanism driving catastrophic forgetting in LoRA.

The threshold is computed via the rectangular spiked-deformation transform applied to the measured spectrum of W_base, with no fitted parameters, no training curves, and no tunable constants. The authors validate this on real 7-8B parameter LLMs.

### Practical contribution

A per-layer alpha/rank budget rule: cap each layer's LoRA scale so the end-of-training spectral spike stays at or below the intruder threshold. The authors report a 62% reduction in forgetting on the most fragile model at no task-performance cost. They also use the threshold to explain the LoRA-vs-full-fine-tuning asymmetry (full FT disperses its update below the threshold of every layer).

### Relevance to Gradience

**Critical — directly extends the theoretical framework.**

- **Rank suggestion policies (vnext/policy/):** The intruder threshold provides a principled, spectrum-derived upper bound on per-layer rank that Gradience's rank suggestion policies currently lack. The existing `energy_threshold`, `knee_elbow`, and `optimal_hard_threshold` policies determine how much rank is *needed*; the intruder threshold determines how much is *safe*. These are complementary constraints.
- **Phase transition detection (research/phase_transitions.py):** The intruder threshold gives an exact critical point prediction for the norm-driven phase transition hypothesized in THEORY.md §4. Gradience's current detection relies on empirical heuristics (autocorrelation time, variance ratio); this paper offers a derived threshold.
- **UDR and compression safety (THEORY.md §2):** The Update Dominance Ratio already measures adapter "loudness" relative to base weights. The intruder threshold refines this: UDR is a global scalar, while the intruder threshold is per-layer and spectrum-aware. This is the formal version of what UDR approximates.
- **Merge triage (vnext/merge/):** Knowing which layers are near or past their intruder threshold before merging could improve the IMBALANCED verdict (Branch 0/4) and inform per-layer merge strategy selection.

### Integration priority: HIGH

The intruder threshold could become a new rank policy (`intruder_ceiling`) or a safety check layered on top of existing policies. Implementation requires: (1) computing the spiked-deformation transform from W_base spectra (which Gradience already measures), and (2) comparing LoRA update norms against the derived threshold. The theoretical alignment with Gradience's existing framework is strong — this paper essentially derives what Gradience has been measuring empirically.

---

## 2. Spectral and Subspace Dynamics of LoRA in Supervised Fine-Tuning and Reinforcement Learning

**Authors:** Chenruo Liu et al.
**Source:** [arXiv:2611.08414](https://arxiv.org/abs/2611.08414) (November 5, 2026)

### Summary

This very recent paper tracks the singular value spectrum and singular subspaces of LoRA updates over the course of training, comparing supervised fine-tuning (SFT) and reinforcement learning (RL) on math reasoning tasks. Key finding: RL consistently narrows the spectrum (concentrating energy into fewer directions) across all learning rates tested, while SFT broadens the spectrum at moderate learning rates. RL also rotates the singular subspaces faster than SFT at small learning rates. The authors attribute the difference to group-mean baseline subtraction in RL.

### Practical contribution

The authors propose a stage-aware spectral correction: for SFT, flatten the full update spectrum; for RL, flatten only the component outside the current LoRA increment span (to encourage subspace rotation). This yields improved performance on math reasoning benchmarks.

### Relevance to Gradience

**High — validates and extends Gradience's monitoring approach.**

- **Telemetry and spectral trajectory tracking (vnext/telemetry.py):** Gradience already tracks per-layer singular spectra over training steps via the GradienceCallback. This paper provides the first systematic evidence that SFT and RL produce qualitatively different spectral trajectories — a distinction Gradience's monitoring does not currently flag or exploit.
- **Phase transition detection:** The SFT spectral broadening → narrowing pattern maps onto the "explore then compress" dynamics described in THEORY.md §4 (early vs. late training dynamics). RL's consistent narrowing suggests a different phase structure that Gradience's phase detection could be extended to recognize.
- **Rank suggestion policies:** If RL concentrates energy more aggressively, then energy-threshold rank suggestions may need to be training-regime-aware. A rank-8 RL adapter might have effective rank 2, while a rank-8 SFT adapter has effective rank 5 — same nominal rank, very different compression opportunities.
- **Three-act gradient alignment (FINDINGS.md §17):** The SFT broadening at moderate LR followed by eventual compression is consistent with the explore→lock-on→destabilize three-act structure observed in Gradience's Mistral-7B study. This paper adds the comparison arm (RL) that Gradience's studies have not yet included.

### Integration priority: MEDIUM-HIGH

No immediate code changes needed, but two actions are warranted: (1) Add SFT-vs-RL spectral trajectory comparison to the next telemetry study design (if/when RL fine-tuning enters the experimental pipeline). (2) Consider whether the GradienceCallback should detect and label the training regime (SFT vs. RL) to contextualize spectral trajectory interpretation.

---

## 3. Spectral Surgery: Training-Free Refinement of LoRA via Gradient-Guided Singular Value Reweighting

**Authors:** Zailong Tian, Yanzhe Chen, Zhuoheng Han, Lizi Liao
**Source:** [arXiv:2603.03995](https://arxiv.org/abs/2603.03995) (March 2026)

### Summary

Spectral Surgery decomposes a trained LoRA adapter via SVD, estimates per-singular-component sensitivity using gradients on a small calibration set, and reweights the singular values under a magnitude constraint — all while preserving the learned singular directions. The method adjusts only ~1,000 scalar coefficients per adapter. Reported gains: up to +4.4 points on CommonsenseQA and +2.4 pass@1 on HumanEval (Llama-3.1-8B, Qwen3-8B).

### Practical contribution

A post-training refinement step that sits between training and deployment. It is training-free but not data-free (requires a small calibration set for gradient estimation). The key insight: not all singular directions contribute equally to task performance, and the magnitudes assigned by training are suboptimal — gradient-guided reweighting can improve them without changing the directions.

### Relevance to Gradience

**Medium-High — directly complements the SVD truncation pipeline.**

- **SVD truncation (vnext/svd_truncate.py):** Gradience's current compression pipeline truncates low-energy singular components entirely (binary keep/discard). Spectral Surgery suggests a softer alternative: reweight rather than truncate. This could improve compression outcomes in cases where Gradience's energy threshold policies identify borderline components.
- **Audit → action gap:** Gradience currently audits spectral structure and suggests rank, but the action is limited to truncation. Spectral Surgery provides a principled post-audit action that uses the same SVD decomposition Gradience already computes. It would extend the audit-to-compression pipeline from "measure and cut" to "measure, reweight, and cut."
- **Bench validation:** The gradience-bench framework could evaluate reweighting as a compression strategy alongside truncation, measuring whether gradient-guided reweighting preserves more downstream accuracy per parameter removed.
- **Calibration data requirement:** Unlike Gradience's current data-free audit, Spectral Surgery needs a calibration set for gradient computation. This is a practical constraint but aligns with the bench suite's existing evaluation infrastructure.

### Integration priority: MEDIUM

This is an enhancement to the compression pipeline rather than a theoretical advance. Worth prototyping as an alternative to hard truncation in svd_truncate, particularly for adapters where the spectral gap is ambiguous (smooth decay, no clear knee). Lower urgency than the Intruder Threshold because it refines an existing capability rather than filling a theoretical gap.

---

## Honorable Mentions

Several other 2026 papers are relevant but less directly aligned:

- **"Can Spectral-Clipping Enable Better Learning While Forgetting Less for Low-Rank Adaptation?"** ([arXiv:2608.12332](https://arxiv.org/abs/2608.12332)) — Proposes clamping singular value growth during training to control forgetting. Relevant to Gradience's monitoring but focuses on a training-time intervention rather than post-hoc analysis.

- **"IGU-LoRA: Adaptive Rank Allocation via Information Gain Uncertainty"** ([ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/file/407106f4b56040b2e8dcad75a6e461e5-Paper-Conference.pdf)) — Uncertainty-aware per-layer rank allocation. Relevant to rank suggestion policies but uses a different allocation signal (information gain vs. spectral energy).

- **"Phantom Transitions in Language Model Fine-Tuning: A Density-Matrix Analysis"** ([arXiv:2606.07559](https://arxiv.org/abs/2606.07559)) — Reports sharp jumps in an order parameter during fine-tuning that persist under LoRA. Relevant to phase transition detection but uses a density-matrix formalism rather than SVD-based analysis.

---

## Summary Table

| Paper | Relevance to Gradience | Integration Priority | Primary Touchpoint |
|-------|----------------------|---------------------|-------------------|
| Intruder Threshold | Theoretical framework + rank policies | **HIGH** | vnext/policy/, research/phase_transitions.py |
| Spectral & Subspace Dynamics | Telemetry validation + SFT/RL distinction | **MEDIUM-HIGH** | vnext/telemetry.py, GradienceCallback |
| Spectral Surgery | Compression pipeline enhancement | **MEDIUM** | vnext/svd_truncate.py, bench/ |
