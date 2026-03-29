# n51 — Research Synthesis Memo

**Type:** synthesis
**Date:** 2026-03-28
**Status:** Current as of the near-miss confirmation (Phase 2b) and the mechanism determinant program (n46–n49).

---

## The picture as it stands

The sidecar program started from the observation that good adapters are not automatically good merge partners, and asked what "compatibility" actually means at the level of learned representations. After eight ruled-out hypotheses, two anchoring studies, and a sequence of increasingly focused measurement programs, the picture is this:

**Commensurability is the umbrella concept.** The question is not whether two adapters are similar in some generic sense, but whether their learned internal solutions — upstream representations and downstream readout — are compatible under linear composition. Commensurability is a conjunction of two independently measurable conditions, not a scalar.

**The sidecars have moved from severity to multiscale incompatibility.** The early framing — "some pairs are more severe than others" — collapsed when severity rankings reversed across backbones. The replacement is a three-rung mechanism ladder: V-module dimensionality mismatch (module-level, d=3.36), head-level cancellation (explains seed sensitivity), and readout gating (transmits or absorbs upstream pathology). Each rung explains what the others cannot. The rungs are nested, not competing.

**Readout orthogonality is benign.** Five of 14 same-task seed pairs show orthogonal decision axes, yet all merge safely. A stand-alone readout-cosine metric would false-alarm on 40% of same-task merges. Orthogonal readout is common, bimodally distributed, and decoupled from upstream V-module geometry. It is the gate condition in the conjunctive model, not a risk signal.

**Attractor multiplicity has two mechanisms.** Multi-attractor readout structure arises through rotational degeneracy (same features, different angular combinations; all on DistilBERT) or feature-set switching (genuinely different feature sets; QNLI/RoBERTa only). Both are benign in themselves. They differ in their failure profile under conjunctive pathology: incoherent confidence vs systematic misclassification. A structured determinant hierarchy — task identity → backbone architecture → training convergence → domain structure — governs which mechanism is expressed, but mechanism and backbone are currently confounded.

**Catastrophe is conjunctive.** The best current model: V-module pathology AND readout incompatibility, together. Either alone is insufficient. Readout incompatibility without upstream pathology is harmless. Upstream pathology with compatible readout is absorbed by the gate. The conjunction explains why catastrophe is seed-dependent (different seeds produce different readout attractor selections) and backbone-dependent (different backbones produce different V-module geometry).

---

## What this means for the product

The research program has progressively narrowed which signals matter and which do not. Core Gradience already captures the outermost layer (task-boundary detection) and the evidence gate (behavioral eligibility). The sidecar findings inform what a future commensurability assessment would need to measure — V-module dimensionality ratios and readout attractor topology — but these require GPU inference and a third backbone before promotion.

The field trials validated the product's practical workflow on exactly the use case the research motivates: mixed inventories of structurally heterogeneous adapters where task labels and eval scores alone are not enough to determine which merges are worth evaluating.

---

## What is settled, what is open

| Claim | Status | Evidence |
|-------|--------|----------|
| Task-boundary detection is reliable | Settled | 0 false positives, 5 inventories, 53+ same-task pairs |
| Severity is not portable | Settled | Rankings reverse across backbones (S01) |
| Instability is the first candidate portable descriptor | Promising | Consistent ranking on 2 backbones; awaiting DeBERTa |
| V-module dim ratio separates catastrophic from safe | Strongest signal | d=3.36, zero overlap; 2 backbones only |
| Readout orthogonality is benign alone | Settled | 5/14 same-task pairs orthogonal, all safe |
| Catastrophe is conjunctive | Current best model | Two independent conditions, both necessary |
| Mechanism and backbone are confounded | Open | All degeneracy on DistilBERT, all switching on RoBERTa |
| Near-miss is a useful product category | Confirmed | 7 pairs, 3 backbones, 3 task families |

---

## Deliverables

| Deliverable | Path |
|------------|------|
| This synthesis | `sidecar/notes/n51_research_synthesis_memo.md` |
| Product validation memo | `field_trials/product_validation_memo.md` |
| Near-miss validation | `field_trials/near_miss_validation.md` |
| Ruled-out mechanisms | `sidecar/notes/n52_ruled_out_mechanisms_packet.md` |
| Executive research summary | `sidecar/notes/n50_executive_research_summary.md` |
