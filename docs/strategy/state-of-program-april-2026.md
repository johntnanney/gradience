# State of the Program

John T. Nanney  
April 2026  
Status: CPU phase substantially complete; next proving ground requires GPU

**Audience:** maintainer, collaborator, strategy reviewer  
**Status:** stable (program snapshot)  
**Purpose:** decision aid for what is validated, bounded, exploratory, paused, and GPU-blocked  
**Canonical for:** April 2026 program state and next-phase gating  
**Supersedes:** ad-hoc status spread across multiple memos  
**See also:** [`../technical-report.md`](../technical-report.md), [`../THEORY.md`](../THEORY.md), [`../FINDINGS.md`](../FINDINGS.md), [`../RESEARCH_INVENTORY.md`](../RESEARCH_INVENTORY.md), [`../00_start_here/project-map.md`](../00_start_here/project-map.md), [`../00_start_here/stable-vs-experimental.md`](../00_start_here/stable-vs-experimental.md), [`../plans/2026-04-03-decoder-only-spectral-fingerprinting-gpu-return-plan.md`](../plans/2026-04-03-decoder-only-spectral-fingerprinting-gpu-return-plan.md)

## Purpose

This memo summarizes what Gradience has established, what it has not, and what determines the next phase of work. It is written as a decision aid, not a public report.

The current technical report, theory document, findings compendium, and research inventory all point to the same conclusion at altitude: Gradience is no longer an exploratory collection of spectral observations. It now has a validated encoder-side workflow core, a stronger theory spine than before, a bounded but real decoder-side ecological precursor, and a growing archive that needs deliberate consolidation rather than more opportunistic branching.

The next major advances will not come from opening more CPU branches. They will come from controlled GPU adjudication.

## 1. Validated

These claims are supported by operational evidence strong enough for product reliance. "Validated" here means: tested across enough conditions that a practitioner can act on them without needing extra interpretive caveats.

### Spectral triage eliminates 90-93 percent of merge candidates without losing winners

Across five field-trial inventories on three small-encoder backbones (DistilBERT, BERT-base, RoBERTa-base), 53+ adapter pairs, and 16 fully evaluated merges, the Gradience workflow (adapter QA, evidence gating, pairwise merge-risk audit, and inventory preflight) dramatically narrows the candidate set while preserving the correct first choices. Task-boundary detection produced zero false positives across the tested set.

Important distinction: Gradience currently validates candidate narrowing much more strongly than merge success prediction. The system reliably separates promising pairs from unpromising ones; retained pairs still require behavioral evaluation.

Also validated within this encoder regime:

- evidence gating as a necessary first filter before pairwise analysis,
- task-boundary detection as the highest-confidence workflow feature,
- near-miss as a real middle category rather than a silent exclusion bucket,
- and the practical distinction between candidate narrowing and merge success prediction.

### Spectral auditing is portable across architectures

Across 86 public adapters spanning 22 architectures and 12 task categories, mean utilization remains about 0.172 and median compression potential remains 50 percent. The core audit works without loading base-model weights and reads nontrivial structure across architectures from Gemma-2B to Mistral-7B.

### Cross-seed spectral stability

Stable rank and related coarse spectral metrics are reproducible properties of the task-architecture-hyperparameter setting rather than arbitrary run noise. In the Mistral-7B/GSM8K multi-seed work, coefficient of variation for stable rank stays below 0.1 across independent seeds.

### The conjunctive failure model (on two backbones)

The best-supported mechanistic account is that catastrophic merge failure requires both V-module dimensionality mismatch and readout incompatibility, with head-level cancellation modulating severity. Neither condition alone is sufficient.

This account was reached by eliminating five simpler hypotheses and is the mechanistic heart of the program. It is validated on DistilBERT and RoBERTa, strong enough to matter, but not strong enough yet for an unrestricted generality claim.

### Spectral compression at Mistral-7B scale

Rank-64 adapters use only about 4-8 effective dimensions per layer. Fifty percent parameter reduction was validated with less than 2.5 percent accuracy degradation across three seeds, guided by the `energy_threshold(0.90)` policy.

### Eligibility gating

Study 16 made the point sharply: structural compatibility is necessary but not sufficient for merge quality. Behaviorally weak adapters produce bad merges even when structural metrics look favorable. This directly motivated the QA artifact and eligibility screening in the product pipeline.

### Structural-behavioral separation

Gradience's strongest validated stance is not "spectral analysis predicts everything." It is: spectral analysis pre-filters and narrows; behavioral evaluation adjudicates. The two are complementary, not substitutive.

## 2. Bounded but Reusable

These findings are real and reproducible but live inside explicit bounds. They are reusable within those bounds and useful for careful public writing, internal design, and future hypotheses.

### Spectral partitioning

The optimal hard threshold now has more than theoretical elegance: it appears to mark a meaningful split between shared high-SV structure and lower-SV task-specific or noise structure. In independently trained small-encoder adapters, same-task pairs show a 7.8x high/low alignment ratio versus 2.5x for cross-task pairs. High-SV alignment increases during checkpoint progression while low-SV alignment barely moves, plateauing around step 150. W0 energy concentration predicts alignment (r = 0.53-0.58 for QNLI).

This upgrades the mtLoRA connection from adjacent literature support to a converging-operations result: two very different methodological pipelines land on the same partition. The formal convergence bound is still open. This is bounded to the small-encoder regime with independently trained adapters.

### Decoder-only ecosystem census

The completed census (`n=36`, 3 architecture families) found non-random spectral structure in public decoder adapters: task eta-squared exceeded architecture eta-squared in the augmented cohort, architecture retained strong local cluster coherence, and the encoder-era module-type asymmetry did not replicate cleanly.

The best bounded interpretation is not "task wins" or "architecture wins," but that both matter differently: architecture shapes local precision or cluster form, while broader task representation increases larger-scale dispersion in fingerprint space.

This should not be read as a universal task-dominance claim; the result remains observational and confound-aware, with substantial nominal-rank pressure.

### Decoder-scale merge evidence

Subspace overlap predicts merge dominance at `r = 0.846` on Mistral-7B across 27 cross-task pairs, with same-task/cross-task separation of 2.4x. End-to-end merge ablation on Llama-2-7B confirms that structural compatibility is necessary but not sufficient.

This is the strongest decoder-scale merge evidence in the program, but pair counts are still small and architecture coverage remains narrow.

### Training telemetry

On NanoGPT-scale telemetry, spectral complexity alone classifies training regimes at 83.7 percent accuracy, DFA exponents separate regimes strongly, and some geometry signals lead loss by about 300 steps.

The three-act gradient alignment pattern observed on Mistral-7B fine-tuning is conceptually rich and probably real, but remains a single-run, single-scale, single-task result. This cluster is reusable as a hypothesis set and explanatory motif, not yet as a stable program-level claim.

### Encoder module-type asymmetry does not transfer

The encoder-era attention-lower-than-MLP utilization pattern holds in only a minority of decoder adapters. This is a negative result, but a reusable one with direct design consequences: heuristics calibrated on encoder structure cannot simply be carried over to decoders.

### Verdict thresholds

The April 2026 stress test and threshold sweep produced a cleaner threshold configuration. Branch-5 catch-all volume dropped while preserving the desired same-task/cross-task behavior.

This is a bounded engineering result rather than a general theorem, but it is stable enough to rely on for current product logic. The residual Branch-5 population remains spectrally heterogeneous, which is why verdict-confidence stratification remains live.

### Secondary probes and bounded companions

Edge-gap, HTSR alpha, direction-aware compatibility metrics, and merge-aware monitoring all survived in bounded form. None displaced the core workflow signals, but each found a legitimate companion role:

- edge-gap as a stronger secondary research observable,
- HTSR alpha as a conditional tail-shape probe,
- direction-aware metrics as bounded interpretive companions,
- merge-aware monitoring as an internal diagnostic, especially with same-task references.

These are reusable, but they are not primary decision engines. Their value is mostly explanatory, diagnostic, or hypothesis-generating.

## 3. Exploratory

These are active research directions with preliminary signal but not enough evidence for product commitments.

### Verdict-confidence stratification

Branch-5 still conflates spectrally distinct populations. The natural integration point for partition findings is verdict confidence: high-confidence vs low-confidence SAFE-like cases. This is well motivated, but still exploratory until validated against merge outcomes.

### Over-accumulation

The architectural idea was worth integrating and sharpened the merge-audit picture, but empirical validation did not justify promotion into strong policy or execution-side strategy changes. It remains a credible exploratory signal, not a settled recommendation-engine input.

### Concentration-weighted convergence bound

This is probably the lead theoretical problem now. Empirical evidence suggests top-subspace stability is governed more by concentration of spectral mass than by naive adjacent-SV gaps. If formalized, this would give Gradience and training-side spectral methods a shared foundation. Right now it is a compelling open move, not a result.

### Broader theory ambitions

Stronger Fisher/Hessian correspondence claims, phase-transition/order-parameter claims beyond currently bounded telemetry evidence, and spectrum universality all remain open. The 86-adapter audit is consistent with universality but cannot cleanly distinguish deep regularity from current training convention.

### Merge vs route vs reject

Strategically promising, especially in light of routing literature, but Gradience has not yet developed or validated a routing decision layer. It is probably the right future framing for composition decisions, but remains conceptual rather than operational.

### Block-level vs component-level adaptation

If block-level LoRA becomes standard, some component-specific findings, especially V-module specificity, may need revisiting. This is a forward-looking pressure point, not a current result.

## 4. Paused

These are work streams that have been deliberately stopped, with reasons recorded.

### Decoder-only ecosystem census (closed April 2026)

Closed as an ecological precursor. It succeeded in showing decoder spectral structure is real and non-random, and that architecture and task both matter under different geometric views of the data. Remaining questions require controlled training, not more public-artifact expansion.

### Pre-merge spectral compression (Study 17)

Clean negative result: conservative pre-merge compression is not a meaningful improvement step in the recommended merge workflow. Compression remains a feature in the library but not part of the endorsed triage pipeline.

### CPU-phase threshold exploration

The threshold sweep exhausted the useful space available on current profile data. Further tuning now is likely pseudo-progress unless new outcome data are added.

### Rank-policy validation

This line has reached a stable bounded state: spectral policies are competitive, OHT is the lead spectral policy, proxy-gradient remains the stronger operational comparator, and attenuate-style ablation is a useful explanatory companion. It does not need further widening without new model regimes or external targets.

### Direction-aware compatibility as a primary upgrade path

This was worth testing and improved interpretive coverage slightly, but it did not beat coarse summaries in the intended encoder slice. The line is paused in a healthy way: retained as a bounded companion, not pursued as a main redesign.

### Merge-aware training monitor

The prototype works technically and same-task references are the preferred usage pattern, but evidence does not justify turning it into training control or broader product logic. It remains a bounded internal diagnostic.

### Philosophy program (separated April 2026)

The Deleuzean research program that originally motivated the spectral approach has been cleanly separated from the engineering program. The conceptual transfer is complete enough that Gradience no longer needs philosophical scaffolding to justify empirical workflow claims.

## 5. What Specifically Needs GPU

At this point, every remaining question that would materially change product scope or mechanism generality is GPU-blocked. Items are ordered by expected information value per compute-hour.

### 5a. DeBERTa adjudication (~3 hours)

This is the single highest-value next experiment. It decides whether the conjunctive failure model is backbone-general, backbone-contingent, or something in between.

Planned setup: train 8 DeBERTa-v3 adapters (4 GLUE tasks x 2 seeds), evaluate 28 merge pairs. Five pre-registered predictions govern the test:

1. task-boundary detection maintains zero false positives on the third backbone,
2. V-module dimensionality ratio separates catastrophic from safe merges,
3. instability transfers as a portable descriptor,
4. the mechanism-backbone confound either dissolves or solidifies,
5. head-level modulation explains seed-to-seed severity variation.

Everything else about encoder-side mechanism generality is downstream of this answer.

### 5b. Controlled decoder fingerprinting / merge triage (~8-12 hours)

The public ecosystem census established decoder spectral structure exists but cannot cleanly separate architecture, task, training setup, and public-artifact confounds. A matched decoder study (12-16 adapters under controlled conditions and all pairwise merges) would either extend the triage pipeline into the commercially relevant regime or show where it breaks.

### 5c. Verdict-confidence validation (~2-4 hours)

Verdict confidence is one of the clearest ways to turn partition findings into operational value. It can be implemented on CPU, but only merge outcomes will show whether it improves recommendations.

### 5d. Cross-scale spectral partitioning (~4-6 hours)

The independent-adapter partition result is strong on DistilBERT and partially echoed in Mistral work, but a clean scale-up on 7B-class controlled adapters is needed to show whether the phenomenon transports across scale.

### 5e. Generation-task merge triage (~6-10 hours)

The cleanest merge-validation work still lives in classification and narrowly bounded decoder settings. Whether triage logic works on true generation tasks remains open. GPU is also required for serious high-rank work beyond the currently validated range and for stronger causal decoder claims.

## Summary

| Category | Count | Interpretation |
|---|---:|---|
| Validated | 7 claims | Operationally reliable core |
| Bounded but reusable | 7 finding groups | Real findings, strong within scope |
| Exploratory | 6 directions | Worth keeping in play, not promotable yet |
| Paused | 7 work streams | Deliberate closure or bounded resting state |
| GPU-blocked | 5 experiments | ~25-35 hours total; DeBERTa is the gate |

The strategic position is good. The validated core is strong enough to ship and defend. The bounded findings are strong enough to guide research choices and careful external writing. The exploratory directions are now much better posed than they were a few months ago.

The next decisive step, DeBERTa adjudication, is small enough to be unblockable with modest GPU access.

## Bottom line

Gradience has moved out of the phase where the main danger is lack of evidence. The main danger now is diffusion: too many adjacent summaries, too many partly overlapping documents, and too much temptation to keep squeezing CPU for marginal gains instead of moving to the next decisive test.

The right next move is to treat the CPU phase as substantially complete, consolidate the current corpus into a cleaner canonical structure, and prepare the GPU-return adjudication packet.
