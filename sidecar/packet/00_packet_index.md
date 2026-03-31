# Gradience Sidecar — Research Packet

**Date:** March 2026
**Status:** Current. The conceptual structure is stable; DeBERTa adjudication is the next empirical step.

---

## What this is

A self-contained packet summarizing the Gradience sidecar research program — a structured investigation into why some LoRA adapter merges fail catastrophically while others succeed. The packet is designed to be handed to a collaborator, a future version of this project, or anyone who needs the complete picture in an afternoon's reading.

The sidecar operates alongside the Gradience library (a Python tool for spectral analysis of LoRA fine-tuning dynamics). Core Gradience detects same-task/cross-task boundaries and gates merge candidates by behavioral evidence. The sidecar asks a deeper question: within the cross-task regime, what determines whether a merge is catastrophic or merely degraded?

---

## Packet contents

Read in this order:

| # | Document | What it is | Length |
|---|----------|------------|--------|
| 1 | `01_where_the_research_stands.md` | **Anchor memo.** The complete theoretical picture: commensurability, instability, V-module pathology, head-level modulation, readout attractors, conjunctive failure, behavioral signatures. Start here. | ~3500 words |
| 2 | `02_product_validation.md` | **Product memo.** What the field trials proved works: 90%+ candidate reduction, correct prioritization, near-miss confirmation. | ~600 words |
| 3 | `03_ruled_out.md` | **Negative results.** Ten hypotheses tested and rejected. Portable severity, task-pair lookup, readout-as-risk, feature plurality as universal origin. What was eliminated and why. | ~2500 words |
| 4 | `04_evidence_table.md` | **Evidence register.** Every settled claim, every open question, every pending test — in one table. | ~400 words |
| 5 | `05_gpu_reentry.md` | **GPU re-entry note.** What to do when compute returns: the DeBERTa adjudication protocol, what it tests, and the decision tree for outcomes. | ~500 words |

| 6 | [`../docs/strategy/ring1_peft_generalization_results.md`](../../docs/strategy/ring1_peft_generalization_results.md) | **Ring 1 results.** PEFT artifact-class generality: LoHa adapters through the full pipeline via thin shim. What generalized, what stayed LoRA-specific. | ~800 words |
| 7 | [`../docs/design/ring2_stage_d_assessment_memo.md`](../../docs/design/ring2_stage_d_assessment_memo.md) | **Ring 2 results.** Checkpoint-delta representation-path generality: full fine-tuned deltas via summary-based reuse. Workflow survives; representation path differs; merge execution out of scope. | ~400 words |
| 8 | [`../../docs/strategy/cross_artifact_product_relevance_summary.md`](../../docs/strategy/cross_artifact_product_relevance_summary.md) | **Cross-artifact portability.** What transfers across artifact classes (evidence gating, narrowing, task ordering) and what doesn't (structural metrics, merge strategies, numeric scores). Three-layer framework. | ~500 words |
| 9 | [`../../docs/strategy/aggregation_sensitive_route2_summary.md`](../../docs/strategy/aggregation_sensitive_route2_summary.md) | **Aggregation-sensitive compatibility.** Different aggregation rules produce different operational judgments from the same structural evidence. Five patterns, decision-context-dependent family selection. | ~500 words |
| 10 | [`../../docs/strategy/behavioral_route2_summary.md`](../../docs/strategy/behavioral_route2_summary.md) | **Behavioral Route 2 bridge.** Route 2 profiles have behavioral reality. Three-tier model: no pathology, localized pathology (collapse vs contamination), stasis. | ~500 words |

### Figures

| Figure | What it shows |
|--------|--------------|
| `figures/s01_summary_panel.png` | **The founding observation.** Instability across two backbones. Severity reverses; instability does not. |
| `figures/per_module_v_spotlight.png` | **The strongest signal.** V-module dimensionality ratio separates catastrophic from safe (d=3.36, zero overlap). |
| `figures/output_space_readout_alignment.png` | **The key falsifier.** Readout orthogonality is common and benign. 5/14 same-task pairs are orthogonal, all safe. |
| `figures/example_semantics_preservation_breakage.png` | **Behavioral signatures.** Preservation and breakage rates across merge quality classes. |
| `figures/example_semantics_taxonomy_composition.png` | **Failure taxonomy.** Five-category composition by class. Neither-source behavior (D) is the clean discriminator. |

For a curated 8-figure visual packet including Route 2 figures, presentation guidance, and usage notes, see [`../../docs/visual-packet.md`](../../docs/visual-packet.md).

---

## Route 2 sub-packet

The Route 2 research programs (decision-dependent, cross-artifact, aggregation-sensitive, behavioral bridge) have their own self-contained packet:

- **[`route2/00_route2_packet_index.md`](route2/00_route2_packet_index.md)** — Broadened compatibility science: substrate generalization, four research programs, checkpoint triage alpha workflow, key figures.

The Route 2 packet is designed to be read independently. It references the synthesis memo (n93), the product-facing summaries in `docs/strategy/`, and the substrate evidence (Ring 1, Ring 2, routing pilot). Start there if your question is about extending Gradience beyond LoRA merge.

---

## Where the full evidence base lives

This packet is a curated summary. The complete sidecar contains 62 notes, 16 analysis scripts, 72 structured data outputs, and 39 figures in the parent directory (`sidecar/`). Key entry points:

- `sidecar/README.md` — full document index
- `sidecar/STATUS.md` — current status with asset counts
- `sidecar/glossary.md` — frozen definitions
- `sidecar/results/` — all JSON data outputs
- `sidecar/scripts/` — all reproducible analysis scripts

---

## The one-paragraph version

Catastrophic LoRA merge failure is conjunctive: it requires both upstream V-module pathology (incompatible dimensionality in the value-projection modules, d=3.36 separation) and downstream readout incompatibility (orthogonal classifier decision axes). Either alone is harmless — readout orthogonality is common and benign in 40% of same-task pairs. The conjunction is seed- and backbone-dependent, which is why severity reverses across architectures but instability (the variability of severity) does not. At the example level, the two pathology channels produce qualitatively distinct failure modes: fragile same-task merges fail with confidence collapse (the model knows it doesn't know), while cross-task merges fail with high-confidence wrong predictions (the model doesn't know it doesn't know). Near-miss merges are behaviorally indistinguishable from safe merges — the boundary is a threshold, not a gradient. Ten alternative hypotheses were tested and eliminated. The DeBERTa adjudication on a third backbone is the next decisive test.
