# Terminology

**Last updated:** 2026-03-31

Canonical terms used across documentation, packets, and workflow docs. When writing or editing, prefer these forms. The sidecar has its own frozen glossary (`sidecar/glossary.md`) for research-internal terms; this document covers project-wide terminology.

---

## Product concepts

| Term | Meaning | Notes |
|------|---------|-------|
| **evidence bootstrap** | The evaluation procedure that compares each adapter/checkpoint to the base model on a held-out sample. Produces per-adapter scores and eligibility classifications. | The procedure, not the filter. |
| **evidence gate** | The binary pass/fail filter that excludes adapters without sufficient evidence from merge consideration. | The filter, not the procedure. Fed by the bootstrap. |
| **evidence gating** | The overall workflow pattern: run the bootstrap, apply the gate, proceed only with gated adapters. | The workflow, encompassing both bootstrap and gate. |
| **task boundary** | The same-task / same-family / cross-task classification of a pair. | Unhyphenated in prose. Use "task-boundary" only as a compound modifier before a noun (e.g., "task-boundary detection"). |
| **task relationship** | The specific three-way field on a pair: `same_task`, `same_family`, `cross_task`. | "Task relationship" in prose; `task_relationship` in code. Not "task-relation." |
| **near-miss** | A pair that is structurally plausible but blocked by weak source evidence. | Always hyphenated in prose. `near_miss` in code. |
| **same-family** | Two adapters trained on different datasets from the same task family (e.g., SST-2 and Yelp Polarity are both binary sentiment). | Hyphenated in prose. `same_family` in code. |
| **checkpoint triage** | The alpha workflow for triaging full fine-tuned checkpoint compatibility. | Canonical short name. Use "checkpoint inventory triage" only when distinguishing from adapter inventory triage. |
| **action plan** | The operational triage output: which pairs to retain, which are near-misses, which to skip. | Not "action-plan." Two words. |
| **candidate reduction** | The fraction of pairs filtered out by the triage workflow (typically 90–93%). | A rate, not a count. |
| **near-miss severity** | How far a near-miss is from eligibility: marginal, moderate, substantial. | Three levels. Best prospects (marginal) first. |
| **dominant driver** | What controls the inventory's triage outcome: `source_qa` (evidence quality) or `structural` (compatibility scores). | A field on the inventory summary. |

## Research concepts

| Term | Meaning | Notes |
|------|---------|-------|
| **instability** | The variability of a pair's merge severity across seeds and backbones. The stable descriptor — severity reverses, instability does not. | Not "seed sensitivity" (which describes the phenomenon, not the metric). |
| **mechanism ladder** | The theoretical account of catastrophic merge failure: commensurability → V-module pathology → head-level cancellation → readout gating → conjunctive failure. | Unhyphenated. Use "mechanism-ladder" only as a compound modifier (e.g., "mechanism-ladder synthesis"). |
| **V-module** | The value-projection attention module. Carries the catastrophe-discriminating signal (d=3.36). | Capital V, hyphenated. Not "V module" or "v-module." |
| **conjunctive failure** | The failure mode: catastrophe requires V-module pathology AND readout incompatibility; either alone is benign. | The phenomenon. |
| **conjunctive model** | The theoretical framework that explains conjunctive failure. | The theory. |
| **spectral analysis** | The general discipline of SVD-based measurement of adapter weight spaces. | Broadest term. |
| **spectral audit** | A specific operation: running spectral analysis on one adapter or one pair. | The CLI/API operation. |
| **spectral measurement** | The data produced by a spectral audit: singular values, energy rank, stable rank, etc. | The output. |

## Route 2 concepts

| Term | Meaning | Notes |
|------|---------|-------|
| **decision-dependent compatibility** | The finding that the same structural relation means different things under merge, routing, and triage. | Hyphenated. Prefer over "decision-context-dependent" (longer form acceptable for emphasis but not default). |
| **aggregation-sensitive compatibility** | The finding that different aggregation rules produce genuinely different operational judgments from the same evidence. | Hyphenated. |
| **aggregation-invariant** | A case where all aggregation families produce the same judgment. Only 2/12 in the current panel. | Hyphenated. |
| **compatibility profile** | One of the five Route 2 structural/behavioral categories: aggregation-invariant safe, same-family optional, worst-case collapse, cross-task separable, QA-dominant review. | Canonical term for the categories. |
| **aggregation-sensitive pattern** | One of the five patterns describing how aggregation families diverge: invariant exclusion, distributional gradient, QA dominance override, QA-gated enrichment, mixed evidence nuance. | "Pattern" for the aggregation taxonomy; "profile" for the compatibility categories. |
| **QA-dominant** | The aggregation family that prioritizes QA evidence over structural compatibility. | Hyphenated adjective. Not "QA dominant" or "QA dominance" when used as modifier. |
| **behavioral tier** | One of three tiers in the Route 2 behavioral model: no pathology, localized pathology, stasis. | Not "behavioral level" or "behavioral class." |
| **broadened substrate** | The validated finding that Gradience's analysis substrate extends beyond LoRA merge. | The architectural claim. |
| **broadened workflow** | The process-level generalization: the same workflow shape (evidence → QA → pairwise → action plan) works across scenarios. | The process claim. |

## Formatting conventions

- **Hyphenation:** Compound modifiers before nouns are hyphenated (task-boundary detection, mechanism-ladder synthesis, decision-dependent aggregation). Standalone nouns are not (task boundary, mechanism ladder).
- **Code vs prose:** `snake_case` in code (`same_family`, `near_miss`, `task_relationship`). Hyphenated in prose (same-family, near-miss, task relationship).
- **Capitalization:** V-module (capital V). QA (all caps). Route 2 (capital R). LoRA, LoHa, PEFT (standard casing).
