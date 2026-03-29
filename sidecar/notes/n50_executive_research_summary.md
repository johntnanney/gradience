# Gradience: Executive Research Summary

**Date:** March 2026
**Status:** Current as of the Attractor Mechanism Determinants program (n46–n49)

---

## The problem Gradience addresses

Gradience starts from a simple but underappreciated problem: good adapters are not automatically good merge partners.

In practice, teams often decide what to merge using superficial signals — task labels, benchmark scores, architecture match, rough intuition about domain similarity. Those signals help, but they do not get at the deeper question: did these adapters learn internally commensurable solutions, or are they only superficially similar?

That is the core idea behind Gradience.

---

## The central claim

Merge success depends less on whether two adapters look similar on paper and more on whether they learned internally commensurable solutions.

This is the point that now anchors the project. Gradience exists to make that hidden compatibility structure more legible before practitioners spend time and compute evaluating implausible merges.

---

## What Gradience is today

Gradience is best understood as a mixed-task inventory preflight system. Its current practical role is to help users reduce the candidate space before evaluation, expose task-boundary risk, distinguish same-task safe zones from cross-task caution zones, summarize source quality and evidence strength, and turn a messy adapter inventory into a smaller, more defensible evaluation plan.

That is the core product identity.

It is deliberately conservative. It does not claim to solve merge prediction with a single score. It does not claim that every risky merge can be ranked perfectly in advance. Its job is narrower and more useful: surface the most important structure early enough to save wasted search.

---

## What the research sidecars have established

The sidecar projects have progressively sharpened what "compatibility" really means. They have shown that many attractive simplifications do not hold up: there is no reliable universal severity score; exact task-pair identity is not a stable catastrophe lookup; readout orthogonality is not itself a risk marker; some internal differences are completely benign; catastrophic outcomes are not explained by one level of structure alone.

Instead, the research now supports a more mature picture.

### 1. Instability is often a better descriptor than raw severity

The important question is not always "how bad is this merge on average?" but often "how fragile and contingent is this regime?" Severity rankings reverse across backbones — the most severe pair on DistilBERT is among the mildest on RoBERTa. Instability — the variability of severity across seeds and backbones — is consistent across both tested backbones, with a clean gap separating two unstable pairs (instability > 0.7) from four stable ones (< 0.3). Instability is the first candidate portable descriptor, pending DeBERTa confirmation.

### 2. Merge failure is multiscale

The strongest current evidence suggests that incompatibility can appear at different levels. The V-module dimensionality ratio separates catastrophic from safe collision with Cohen's d = 3.36 and zero range overlap (Rung 1). Head-level geometry explains seed sensitivity — the 29-point severity gap within CA-01 that was invisible at every other resolution (Rung 2). The classifier readout layer functions as a gate that transmits or absorbs upstream pathology, not an amplifier (Rung 3). These rungs are nested: module-level sets the precondition, head-level modulates severity, readout gating determines whether upstream pathology manifests as catastrophic classification errors.

### 3. Benign diversity is common

Two runs can differ internally, even strongly, and still merge safely. This is crucial. Orthogonal readout, multiple attractors, or different internal organization are not automatically pathologies. Five of 14 same-task seed pairs show orthogonal decision axes, yet all merge safely (max Δ = 2.2%). Six of 10 task families are single-attractor; three are multi-attractor; one is backbone-contingent. All merge safely regardless of attractor structure.

The sidecar now distinguishes three qualitatively different kinds of benign diversity. Single-attractor stability: the readout has one dominant basin and seeds converge. Rotational degeneracy: seeds find orthogonal orientations within a shared low-rank subspace — same features, different combinations. Feature-set switching: seeds lock onto genuinely different principal components of the pretrained representation — different features entirely. All three are benign in themselves; they differ in their failure profile *if* upstream pathology is also present.

### 4. Catastrophic failure is conjunctive

The best current model is not "one bad signal causes collapse" but rather: risky upstream structure (V-module pathology) plus the wrong downstream conditions (readout incompatibility) plus sensitivity to how the solution is internally organized (head-level configuration). Catastrophic behavior emerges when the wrong kinds of internal differences line up across scales. Either condition alone is insufficient — readout incompatibility without upstream pathology is harmless; upstream pathology with compatible readout is absorbed by the gate.

### 5. Commensurability is the right umbrella concept

The sidecars increasingly suggest that the practical problem is not generic similarity, but commensurability: are two solutions different in a harmless way, or different in a way that becomes incompatible under linear composition?

Commensurability has been refined through three versions. Version 1: upstream AND readout (two binary conditions). Version 2: readout decomposes by attractor topology — single-attractor families satisfy it automatically; multi-attractor families may or may not. Version 3 (current): readout further decomposes by mechanism class. Rotational degeneracy and feature-set switching produce different kinds of openness in the readout gate, with different failure semantics (incoherent confidence vs systematic misclassification) and different computational checks required (angular separation within shared subspace vs subspace overlap).

### 6. Mechanism choice is structured

What determines whether a multi-attractor family expresses its multiplicity through rotational degeneracy versus feature-set switching? A structured hierarchy: task identity (primary — determines whether multi-attractor structure is possible at all) → backbone architecture (secondary — selects which mechanism realizes it) → training convergence (tertiary — gates attractor count but not mechanism) → domain structure (weak). The critical finding: mechanism and backbone are perfectly confounded in the current panel — all degeneracy on DistilBERT, all switching on RoBERTa — which is the main open question for the DeBERTa adjudication.

---

## Why this matters for practitioners

The practical value of Gradience is not that it promises perfect merge prediction. The practical value is that it gives ML practitioners a better way to think about compatibility. Task similarity is not enough. Eval strength is not enough. Internal diversity is not always bad. Risky merges usually have more specific structure than "these models are different."

The result is a more disciplined workflow: narrow first, inspect the likely safe region first, treat cross-task space carefully, do not overreact to harmless diversity, focus evaluation effort where internal commensurability is most plausible.

---

## Gradience north star

The long-term north star for Gradience is to become the system practitioners run before merge experiments to determine whether a set of adapters learned internally commensurable solutions — and therefore which combinations are actually worth evaluating.

### Core north star

Build the best practical preflight workflow for adapter inventories: trustworthy, conservative, workflow-oriented, repeatable, and useful under real constraints.

### Research north star

Develop a multiscale theory of merge commensurability that explains when diversity is benign, when it becomes fragile, where incompatibility first appears, and how it propagates into downstream failure.

The core product should stay narrow and useful. The research program should stay ambitious and explanatory. Together, they support the same big idea: the future of merge tooling is not better superficial matching, but better visibility into whether learned solutions are internally compatible.

That is the point Gradience is establishing, and it is the standard the project should keep building toward.
