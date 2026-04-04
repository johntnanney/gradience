# Capabilities Reference

**Last updated:** 2026-04-04

What each Gradience feature does, what it covers, and where it stops. No theory or architecture — for the full argument, see the [Technical Report](technical-report.md).

---

## Stable capabilities (shipped, tested)

### Adapter inventory preflight

| Stage | What it does | Key output |
|-------|-------------|------------|
| Evidence bootstrap | Evaluates each adapter against base model on a small sample; flags adapters that don't meaningfully improve over base | Per-adapter evaluation delta |
| QA classification | Assigns eligibility: `eligible`, `uncertain`, `flagged_weak`, or `unknown_no_behavioral_eval` | QA artifact (`gradience.adapter_qa/v1`) |
| Pairwise compatibility | Per-layer spectral analysis of every eligible pair; assigns risk level, dominant issue, and strategy recommendation | Merge report (`gradience.merge_qa_report/v1`) |
| Task-boundary detection | Classifies pairs as same-task, same-family, or cross-task using evaluation dataset metadata | Task-relationship advisory on each merge report |
| Near-miss identification | Identifies structurally plausible pairs blocked only by weak source evidence; ranks by severity (marginal / moderate / substantial) | Near-miss section in action plan |
| Action plan | Aggregates all pair results into retained / near-miss / skip categories | Inventory summary (`gradience.inventory_summary/v1`) + action plan |
| Preflight bundle | Packages a complete run: JSON summary, markdown review, optional HTML report, run manifest | `--emit-bundle` output directory |
| Batch summary | Cross-run comparison table from multiple preflight bundles | `gradience batch-summary` output |

### Merge execution (LoRA only)

| Strategy | When recommended |
|----------|-----------------|
| `linear` | Low-risk same-task pairs |
| `ties` | Pairs with sign disagreement across layers |
| `dare_ties` / `dare_linear` | Pairs where dropout may reduce interference |
| `norm_equalized` | *Not* recommended for imbalanced pairs (contraindicated); used when norm ratio is moderate |

Strategy recommendation comes from the triage step. Execution is a separate, explicit command.

### Spectral measurements (per adapter)

| Metric | What it measures |
|--------|-----------------|
| Stable rank | Frobenius / spectral norm ratio — effective dimensionality |
| Energy rank @ 90% | Minimal rank capturing 90% of Frobenius energy |
| Entropy effective rank | Information-theoretic dimensionality measure |
| Utilization ratio | Stable rank / nominal rank — how much of the rank budget is used |

### Pairwise compatibility metrics (per layer)

| Metric | What it measures |
|--------|-----------------|
| Principal angles | Subspace overlap between the two adapters' column/row spaces |
| Directional agreement | Whether shared dimensions are used cooperatively or antagonistically |
| Magnitude balance | Norm ratio between adapters |
| Subspace overlap | Combined overlap score |

Per-layer verdicts: `SAFE`, `REDUNDANT`, `CONFLICTING`, `IMBALANCED`.

### Task-boundary detection

| Classification | Meaning | Validated coverage |
|---------------|---------|-------------------|
| Same-task | Both adapters trained on the same dataset | 53+ pairs, 3 backbones, 0 false positives |
| Same-family | Different datasets, same task type (e.g., SST-2 and IMDB) | Behaviorally confirmed as safe-like |
| Cross-task | Different task types | Flagged for caution; routed separately in action plan |

Currently validated task family: binary sentiment (SST-2, IMDB, Yelp Polarity, Amazon Polarity).

### Machine-readable artifacts

| Schema | Content | Contract |
|--------|---------|----------|
| `gradience.adapter_qa/v1` | Per-adapter eligibility, spectral summary, behavioral evidence | Frozen, additive-only |
| `gradience.merge_qa_report/v1` | Per-pair risk, dominant issue, strategy, task advisory | Frozen, additive-only |
| `gradience.inventory_summary/v1` | Aggregate counts, risk distribution, strategy recommendations | Frozen, additive-only |

All schemas are frozen. Old outputs remain valid indefinitely. New fields may be added; existing fields will not change meaning.

---

## Alpha capabilities (working, scoped)

### Checkpoint triage

The same triage workflow applied to full fine-tuned checkpoints instead of LoRA adapters. Uses summary representation (not factor geometry) but the same workflow shape.

| Aspect | Scope |
|--------|-------|
| Base model | Shared base required |
| Architecture | Small encoders only |
| Task type | Classification only |
| Evidence | Bootstrap required |
| Merge execution | **Not supported** — triage generalizes, execution does not |

One canonical instance: `field_trials/checkpoint_inventory_t02/`.

---

## Validated but not packaged

These extensions confirmed that the analysis substrate generalizes. They are not part of the installed tool.

| Extension | What was validated | Effort to ship |
|-----------|-------------------|----------------|
| Routing confusability | Same spectral functions assess whether a router might confuse two adapters | ~370 LOC, zero core changes |
| LoHa adapter support | Full pipeline works via extraction shim | ~160 LOC, zero core changes |
| Workflow portability | Evidence → structure → triage pattern works across PEFT methods and representation paths | Structural metrics not portable; workflow is |

---

## What Gradience does not do

| Limitation | Detail |
|-----------|--------|
| Automatic merging | Triages only. Merge execution exists but is a separate, explicit step |
| Work without evidence | No evaluation data → no useful output. By design |
| Predict severity | Identifies risk, not magnitude. Severity depends on backbone, seed, and head-level geometry |
| Decoder models | All validation is on small encoders (DistilBERT, BERT-base, RoBERTa). Decoder spectral structure exists (census confirmed) but merge triage is unvalidated |
| Non-classification tasks | Generation, extraction, retrieval are outside the validated envelope |
| High-rank adapters | Tested at rank ≤ 16. Higher ranks are untested |
| Large inventories | Largest validated: 9 adapters, 28 pairs |
| Detect all failure modes | Good at task-boundary detection and evidence gating. Subtle within-task incompatibilities at high rank may not be caught |
| Universal compatibility | The substrate generalizes further than the product. But the product is a bounded LoRA merge triage tool |

---

## Quick reference

| You want to... | Go to |
|----------------|-------|
| Understand why this approach works | [Technical Report](technical-report.md) |
| Run your first inventory | [Playbook](playbook.md) |
| See example scenarios | [Example Gallery](example-gallery.md) |
| Look up a CLI command | [CLI Reference](cli.md) |
| Check the validation evidence | [Product Validation](product/product-validation.md) |
| See what's bounded vs experimental | [Stable vs Experimental](00_start_here/stable-vs-experimental.md) |
