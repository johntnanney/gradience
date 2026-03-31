# What Gradience Knows How to Do

**Last updated:** 2026-03-31

Plain-language capabilities. No architecture, no theory. What it does, how well it does it, and where it stops.

---

## The short version

Gradience looks at a collection of fine-tuned model adapters and tells you which ones are worth merging, which ones need more evidence before you decide, and which ones you should skip. It gets this right — zero false positives across five real inventories — by being conservative: it would rather tell you "not enough evidence" than guess.

---

## What it does today (stable, shipped, tested)

### Adapter inventory preflight

You have a pile of LoRA adapters. Some are yours, some are from the Hub, some are experiments you half-remember. Gradience triages the pile:

- **Evidence bootstrap.** Before looking at any structural compatibility, it checks: does each adapter actually do something? It evaluates each one against the base model on a small sample. Adapters that don't meaningfully beat base are flagged immediately. This is not optional — without it, the tool produces nothing useful.

- **QA classification.** Each adapter gets an eligibility status: `eligible` (clear evidence it works), `uncertain` (ambiguous), `flagged_weak` (below threshold), or `unknown` (no evaluation data). Weak and unknown adapters are excluded from merge consideration.

- **Pairwise compatibility.** For every pair of eligible adapters, Gradience computes structural compatibility using spectral analysis (SVD-based measurement of how the adapters' weight spaces relate). Each pair gets a risk level, a dominant structural issue, and a recommended merge strategy.

- **Action plan.** All of the above rolls up into a single triage output: which pairs to pursue, which are near-misses worth revisiting, and which to skip. Typically filters out 90–93% of candidates.

### Task-boundary detection

The most reliable thing Gradience does. It separates:

- **Same-task pairs** (both adapters trained on the same task) — almost always safe to merge.
- **Same-family pairs** (different datasets, same task type — e.g., SST-2 and IMDB are both binary sentiment) — treated like same-task for triage purposes. Behaviorally confirmed as safe.
- **Cross-task pairs** (different task types) — flagged for caution. Not automatically excluded, but the action plan routes them differently.

Zero false positives on this classification across 53+ evaluated pairs and 3 backbones.

### Near-miss prioritization

Some pairs are structurally plausible but blocked by weak evidence (one or both source adapters didn't clearly beat base). Instead of silently dropping these, Gradience identifies them as near-misses and ranks them by severity:

- **Marginal** — almost eligible, small evidence gap
- **Moderate** — fixable with better evaluation
- **Substantial** — far from eligible

Near-miss merges are behaviorally indistinguishable from safe merges (avg degradation -0.006 vs best source). The structural concern is real; the behavioral concern is not. The right action is to fix the evidence, not to avoid the merge.

### Merge execution

For LoRA adapters specifically, Gradience can execute the merge itself — not just triage. It supports linear, TIES, DARE, and norm-equalized strategies. The strategy recommendation comes from the triage step; execution is a separate command.

### Reports and artifacts

Everything produces structured output:

- **QA artifacts** — per-adapter eligibility records (JSON, frozen v1 schema)
- **Merge reports** — per-pair risk assessment (JSON, frozen v1 schema)
- **Inventory summaries** — aggregate triage (JSON, frozen v1 schema)
- **Action plans** — what to do next (JSON + Markdown)
- **Preflight bundles** — packaged run with manifest, review packet, and optional HTML report
- **Batch summaries** — cross-run comparison tables

All schemas are frozen and additive-only. Old outputs remain valid.

---

## What it does in alpha (working, scoped, not yet promoted)

### Checkpoint triage

The same triage workflow — evidence bootstrap, QA, pairwise comparison, action plan — applied to full fine-tuned checkpoints instead of LoRA adapters. Uses a different representation path (summary statistics instead of factor geometry) but the same workflow shape.

**Scope contract:** shared base model, small encoders only, classification only, evidence bootstrap required. Outside these bounds, treat results as experimental.

**What it does:** Triages checkpoint compatibility. **What it doesn't do:** Merge checkpoints. Triage generalizes; merge execution does not.

One canonical instance exists (`field_trials/checkpoint_inventory_t02/`). See the [checkpoint triage README](../field_trials/checkpoint_inventory_t02/README.md).

---

## What is validated but not packaged

These experiments confirmed that the underlying analysis generalizes, but the results are not part of the installed tool.

### Routing confusability

The same spectral functions that assess merge compatibility can assess routing confusability — whether two adapters are similar enough that a router might confuse them. Validated with ~370 lines of new code and zero changes to existing code. The analysis substrate is shared; the policy layer (what you do with the measurements) is different.

### LoHa adapter support

The full pipeline (audit, pairwise comparison, inventory) works on LoHa adapters via a ~160-line extraction shim. Zero core code changes. Factor extraction is different; everything downstream is identical.

### Broader PEFT and checkpoint analysis

The workflow shape — evidence first, then structural comparison, then triage — has been validated across three generalization axes (different scenarios, different PEFT methods, different representation paths). The structural metrics are not portable across these axes; the workflow is.

---

## What is research-only

The sidecar research program has established why Gradience's signals work (the mechanism ladder) and how they extend beyond merge (the Route 2 framework). These findings inform the tool's design but are not part of the product:

- **Why merges fail catastrophically.** V-module pathology in the value-projection attention heads, combined with readout incompatibility, produces catastrophe. Either alone is benign. Ten alternative explanations were tested and eliminated.

- **Why severity is unpredictable.** The same task pair can be catastrophic on one backbone and safe on another. Severity is not portable; instability (the variability of severity) is.

- **Why different decisions need different aggregation.** The same structural evidence produces different operational judgments depending on whether you're merging, routing, or triaging. This is not a presentation choice — it reflects genuine differences in what each decision optimizes for.

- **Behavioral grounding.** Four of five compatibility profiles have distinct behavioral signatures at the example level. The most important distinction: some failures involve the model knowing it's confused (recoverable) vs the model being confidently wrong (dangerous).

For the full research story, see the [research packet](../sidecar/packet/00_packet_index.md) and [Route 2 packet](../sidecar/packet/route2/00_route2_packet_index.md).

---

## What it does not do

- **It does not merge for you automatically.** It triages. Merge execution exists but is a separate, explicit step.
- **It does not work without evidence.** An adapter with no evaluation data gets no useful output. This is by design.
- **It does not predict severity.** It identifies risk (same-task vs cross-task, structural compatibility) but cannot tell you how bad a failed merge will be. Severity depends on backbone, seed, and conditions that are not measurable without running the merge.
- **It does not support decoder models.** All validation is on small encoder models (DistilBERT, RoBERTa, BERT-class). Larger models and decoder architectures are untested.
- **It does not support non-classification tasks.** Generation, extraction, retrieval, and other task types are outside the validated envelope.
- **It does not detect all failure modes.** The tool is good at task-boundary detection and evidence gating. Subtle within-task incompatibilities at high rank may not be caught.
- **It is not a universal compatibility platform.** The substrate generalizes further than the product packaging. But the product is a LoRA merge triage tool. The broader capabilities are validated experiments and research findings, not shipped features.

---

## Where to go from here

| You want to... | Start with... |
|----------------|---------------|
| Try it | `pip install gradience[hf]` and `gradience verify` |
| See the full CLI | [CLAUDE.md](../CLAUDE.md) (key commands table) |
| Understand the product evidence | [Product validation memo](product-validation.md) |
| Try the checkpoint triage alpha | [Checkpoint triage README](../field_trials/checkpoint_inventory_t02/README.md) |
| Understand what didn't generalize | [Boundaries and non-generalizations](boundaries-and-non-generalizations.md) |
| See the research | [Demo path C](demo-paths.md#path-c--the-research-program) |
| Orient to the full project | [Project map](project-map.md) |
