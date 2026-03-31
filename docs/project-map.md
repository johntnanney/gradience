# Gradience Project Map

**Last updated:** 2026-03-31

This document is the canonical orientation guide for Gradience. It answers: what is stable, what is alpha, what is experimental, what is research-only, and what is blocked.

If you are new to the project, read this first. For plain-language capabilities, see [what Gradience does](what-gradience-does.md). Then pick a [demo path](demo-paths.md) for a guided tour. For the visual story, see the [visual packet](visual-packet.md) (8 curated figures).

---

## Layer 1 — Stable Product (shipped, tested, versioned)

Gradience v0.11.0 on PyPI. Python 3.10+. CI on 3.10/3.11/3.12.

### CLI commands (`gradience`)

| Command | Purpose |
|---------|---------|
| `verify` | Installation health check |
| `check` | Config validation (LoRA/training args) |
| `monitor` | Training telemetry analysis (JSONL) |
| `audit` | Spectral measurement of a single LoRA adapter |
| `audit-adapter` | Single-adapter QA eligibility artifact (v1 schema) |
| `merge-audit` | Pairwise geometric compatibility between two adapters |
| `merge-plan` | Multi-adapter merge strategy planner |
| `merge` | Execute a merge |
| `explain` | Natural-language interpretation of audit/merge results |
| `truncate` | SVD-based rank truncation |
| `summarize-inventory` | Aggregate QA + merge reports into inventory summary |
| `suggest-neighborhoods` | Conservative merge neighborhoods from existing reports |
| `batch-summary` | Cross-run comparison table from multiple preflight bundles |
| `portfolio` | Cross-inventory landing view |
| `preflight-report` | HTML preflight report generation |

### Stable Python API (`gradience.api`)

| Function | Purpose |
|----------|---------|
| `run_bench()` | Run compression benchmark suite |
| `aggregate_bench_runs()` | Aggregate multiple bench runs |
| `audit()` | Programmatic spectral audit |
| `monitor()` | Programmatic telemetry analysis |
| `audit_adapter()` | Build QA eligibility artifact |
| `merge_risk_report()` | Build merge QA report |
| `summarize_inventory()` | Aggregate into inventory summary |
| `suggest_neighborhoods()` | Merge neighborhood suggestions |
| `scan_portfolio()` | Cross-inventory portfolio scan |
| `compute_core_space_diagnostic()` | Core-space overlap diagnostic |

### Stable schemas (frozen, additive-only)

- `gradience.adapter_qa/v1` — single-adapter QA artifact
- `gradience.merge_qa_report/v1` — pairwise merge risk report
- `gradience.inventory_summary/v1` — inventory-level summary

### Validated capabilities (field-trial confirmed)

- **Zero false positives** across 5 inventories and 53+ evaluated pairs
- **90–93% candidate reduction** (inventory action plan filters noise)
- **Near-miss as validated middle category** — behaviorally indistinguishable from safe
- **Same-family routing** — cross-dataset same-family pairs (e.g., SST-2 x IMDB) route to safe bucket
- **Task-family registry** — static taxonomy in `vnext/merge/task_families.py`
- **Near-miss severity ordering** — marginal / moderate / substantial classification

---

## Layer 2 — Alpha Workflow (functional, scoped, not yet promoted)

### Checkpoint Triage (Route 2 Alpha)

**What it is:** An end-to-end workflow for triaging full fine-tune checkpoint compatibility — not just LoRA adapters.

**Status:** Working. Mini-product README at [`field_trials/checkpoint_inventory_t02/README.md`](../field_trials/checkpoint_inventory_t02/README.md). HTML report at `preflight/alpha_bundle/report.html`.

**Scope contract (alpha boundaries):**
- Shared base model only (no cross-base comparison)
- Small encoder checkpoints only (distilbert-class)
- Classification tasks only
- Evidence bootstrap required before triage decisions

**What it reuses from stable:** QA artifact schema, inventory summary, action plan, evidence bootstrap, report vocabulary. All stable CLI and API paths work on the shimmed inputs.

**What it does not do:** Merge execution for checkpoint deltas is out of scope. The alpha workflow triages — it does not merge.

---

## Layer 3 — Validated Experiments (results confirmed, not in main package)

These live in `experiments/` or `field_trials/`. They validated specific generalization claims but are not part of the installed package.

### Routing Pilot (`experiments/routing_pilot/`)

~370 lines, 3 files. Reuses 5 functions from `vnext/merge/` to assess adapter routing confusability instead of merge compatibility. Zero core modifications. Result: same-task/same-family/cross-task discrimination works for routing, not just merge. **Seam identified:** policy vocabulary and aggregation strategy are the two extraction points where routing diverges from merge.

**Validated claim:** The spectral substrate generalizes from merge to routing scenarios.

### Ring 1 — LoHa PEFT Generalization

~160-line extraction shim (`loha_shim.py`). Full Gradience workflow (audit, merge-audit, inventory) ran on LoHa adapters with zero core code changes. Factor extraction differs; everything downstream is identical. Results: `docs/strategy/ring1_peft_generalization_results.md`.

**Validated claim:** The substrate generalizes from LoRA to at least one additional low-rank PEFT method.

### Ring 2 — Checkpoint Delta Representation (`experiments/ring2_checkpoint_delta/`)

8 scripts across 4 stages. Tested three candidate representations for full-checkpoint deltas. Selected Representation C (layerwise summary statistics) for the alpha workflow. Low-rank SVD approximation was rejected at CPU-feasible ranks. Assessment: `docs/design/ring2_stage_d_assessment_memo.md`.

**Validated claim:** Checkpoint deltas can be analyzed with the same workflow shape, but through a different representation path (summary-based, not factor-based).

### Broadened Substrate Scope (summary)

Three validated generalization axes, each proven by one experiment:

| Axis | Experiment | Representation path |
|------|-----------|---------------------|
| Scenario (merge → routing) | Routing pilot | Same (factor-based) |
| Artifact class (LoRA → LoHa) | Ring 1 | Same (factor-based, via shim) |
| Artifact class (LoRA → checkpoint delta) | Ring 2 | Different (summary-based) |

---

## Layer 4 — Sidecar Research (theory, mechanisms, evidence base)

Everything in `sidecar/`. Not part of the product. Not importable from core. Governed by `sidecar/CLAUDE.md`.

### Entry points

| Document | Purpose |
|----------|---------|
| `sidecar/notes/n69_settled_open_next.md` | Dashboard — what is established, open, next |
| `sidecar/notes/n67_where_the_research_stands.md` | Mechanism-ladder synthesis (commensurability → V-module → conjunction) |
| `sidecar/notes/n93_route2_synthesis.md` | Route 2 synthesis (decision-dependent, cross-artifact, aggregation, behavioral) |
| `sidecar/README.md` | Full index of all sidecar artifacts |
| `sidecar/packet/00_packet_index.md` | Research packet (mechanism ladder) |
| `sidecar/packet/route2/00_route2_packet_index.md` | Route 2 packet (broadened compatibility) |

### What the sidecar has established

Fourteen settled claims (see n69 for the full list). The most product-relevant:

- **Task-boundary advisory is the stable interpretive signal.** Same-task = safe, cross-task = caution. Zero false positives.
- **Conjunctive failure model.** Catastrophe requires V-module pathology AND readout incompatibility; neither alone is sufficient.
- **Near-miss is safe.** Behaviorally indistinguishable from retained merges.
- **Same-family optional is safe.** SST-2 x IMDB merges behave like same-task.
- **Collapse and contamination are distinct.** Two different failure channels with different operational implications.
- **Decision-context-dependent aggregation.** Different aggregation rules suit different downstream decisions (merge, routing, triage).

### Research programs (complete)

| Program | Notes | Key output |
|---------|-------|------------|
| Decision-dependent compatibility | n70–n74 | 9-case panel, scenario-specific seams |
| Cross-artifact portability | n76–n80 | 3-layer framework (invariant / representation-family / decision-dependent) |
| Aggregation-sensitive compatibility | n81–n85 | 5 stable patterns, decision-dependent family selection |
| Behavioral Route 2 bridge | n86–n92 | 3-tier behavioral model, collapse/contamination mode split |

### Evidence base

86 notes, ~100 structured JSON artifacts, ~64 figures. Two backbones (DistilBERT, RoBERTa). 53+ evaluated pairs. 8 cases with full 500-example behavioral data.

---

## Layer 5 — GPU-Blocked

### DeBERTa Adjudication Protocol (`sidecar/notes/n07_deberta_adjudication_protocol.md`)

**What it tests:** Whether instability, V-module pathology, and head-level cancellation transfer to disentangled attention. Five pre-registered predictions (A–E).

**Why it matters:** This is the single most important open question. It determines whether the mechanism ladder is architecture-general or architecture-specific.

**What it requires:** ~3 hours on a single consumer GPU. Train 8 adapters, merge 28 pairs, evaluate 56 conditions. Protocol is pre-registered and executable as-is.

**What is blocked on it:**
- Backbone confound resolution (all rotational degeneracy on DistilBERT, all feature-set switching on RoBERTa)
- V-module dim ratio promotion to computable warning signal
- O-module escalation design

**Re-entry checklist:** `sidecar/packet/05_gpu_reentry.md`

---

## How the layers relate

```
Layer 1 (Stable Product)
  │
  ├── Layer 2 (Alpha Workflow) — uses stable schemas + CLI, adds checkpoint delta path
  │
  ├── Layer 3 (Experiments) — validated that the substrate generalizes;
  │     results inform Layer 2 design and Layer 4 theory
  │
  └── Layer 4 (Sidecar Research) — uses Layer 1 as dependency (never reverse);
        provides theoretical grounding for why Layer 1 signals work
        │
        └── Layer 5 (GPU-Blocked) — the decisive next step for Layer 4
```

**Promotion path:** Sidecar → Experiment → Alpha → Stable. Each step requires meeting the five promotion criteria in `sidecar/strategy_memo.md` §11.

---

## Stability and evidence table

| Area | Status | Evidence level | User-facing? | Notes |
|------|--------|---------------|--------------|-------|
| **Single-adapter spectral audit** | Stable | Field-trial validated (5 inventories, 53+ pairs) | Yes — CLI + API | Core product. Zero false positives. |
| **Pairwise merge compatibility** | Stable | Field-trial validated | Yes — CLI + API | Verdicts, risk labels, strategy recommendations. |
| **Inventory preflight / action plan** | Stable | Field-trial validated (90–93% candidate reduction) | Yes — CLI + API | Evidence bootstrap, action plan zones, severity ordering. |
| **QA eligibility artifact (v1 schema)** | Stable | Field-trial validated | Yes — CLI + API | Frozen schema. Four eligibility statuses. |
| **Merge QA report (v1 schema)** | Stable | Field-trial validated | Yes — CLI + API | Frozen schema. Dominant issue, strategy, confidence. |
| **Evidence bootstrap** | Stable | Field-trial validated (the founding lesson) | Yes — workflow gate | First-class precondition. Promoted from T01 finding. |
| **Same-family routing** | Stable | Validated (SST-2 × IMDB, behavioral confirmation) | Yes — task-family registry | Static registry in `task_families.py`. Safe-like behavior confirmed. |
| **Near-miss severity** | Stable | Validated (behaviorally indistinguishable from safe) | Yes — action plan | Marginal / moderate / substantial. Ordering validated. |
| **Checkpoint triage alpha** | Alpha | Single canonical instance (T02) | Partially — workflow doc, not packaged | Scope contract: shared base, small encoder, classification only. |
| **Routing pilot** | Experiment | Single validated run (4 adapters, 6 pairs) | No — `experiments/` only | Substrate generalizes. Policy extraction seams identified. Not packaged. |
| **Ring 1 (LoHa generalization)** | Experiment | Single validated run (3 adapters, 3 pairs) | No — shim only | ~160-line shim. Full pipeline ran. Zero core changes. |
| **Ring 2 (checkpoint delta)** | Experiment | Four-stage validation (Repr C selected) | No — `experiments/` only | Summary-based path. Triage works; merge out of scope. |
| **Decision-dependent compatibility** | Research | 9-case panel, complete program (n70–n74) | No — sidecar only | Settled: aggregation and policy are the scenario-specific seams. |
| **Cross-artifact portability** | Research | 9-case panel, complete program (n76–n80) | No — sidecar only | Settled: portable signals are workflow-level, not metric-level. |
| **Aggregation-sensitive compatibility** | Research | 12-case panel, complete program (n81–n85) | No — sidecar only | Settled: five patterns, decision-dependent family selection. |
| **Behavioral Route 2 bridge** | Research | 8 cases, 4000 examples, complete program (n86–n92) | No — sidecar only | Settled: three-tier behavioral model, collapse/contamination split. |
| **Mechanism ladder (conjunctive failure)** | Research | 53+ pairs, two backbones, 8 behavioral cases | No — sidecar only | Settled: V-module × readout → catastrophe. Ten alternatives eliminated. |
| **DeBERTa adjudication** | GPU-blocked | Protocol pre-registered, zero data | No | ~3h on one GPU. Tests architecture generality. The decisive next step. |
| **V-module dim ratio as warning signal** | GPU-blocked | Descriptive (d=3.36 separation on 2 backbones) | No | Needs DeBERTa confirmation before promotion to computable signal. |

**Reading the table:**
- **Stable** = shipped, tested, versioned, field-trial-validated. Safe to depend on.
- **Alpha** = functional with explicit scope contract. May promote after broader validation.
- **Experiment** = validated result, not packaged. Informs design but not importable.
- **Research** = theoretical finding. Grounds understanding but does not enter the product.
- **GPU-blocked** = protocol ready, compute needed. No claims until data exists.

---

## Quick disambiguation

| If you see... | It belongs to... |
|---------------|------------------|
| `gradience audit`, `gradience merge-audit` | Layer 1 — stable CLI |
| `gradience.api.merge_risk_report()` | Layer 1 — stable API |
| Checkpoint triage, alpha bundle, evidence bootstrap | Layer 2 — alpha workflow |
| `experiments/routing_pilot/`, Ring 1, Ring 2 | Layer 3 — validated experiments |
| n67, n69, n93, mechanism ladder, V-module | Layer 4 — sidecar research |
| DeBERTa, Predictions A–E, n07 | Layer 5 — GPU-blocked |

**See also:** [Boundaries and non-generalizations](boundaries-and-non-generalizations.md) — what didn't work, what doesn't generalize, what not to claim. [Next leaps](next-leaps.md) — what GPU, users, and collaborators would unlock. [Terminology](terminology.md) — canonical terms and formatting conventions.
