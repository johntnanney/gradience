# CLAUDE.md — Gradience Sidecar

## Identity

This is **not** part of core Gradience. It is a structured research sidecar for questions too exploratory, mechanistic, or unstable to belong in core yet.

**Mission:** Investigate catastrophic cross-task interference and the mechanisms that distinguish it from ordinary cross-task degradation.

**Core Gradience** is operational, conservative, workflow-oriented, and useful today.
**This sidecar** is empirical, exploratory, mechanism-oriented, and optimizes for durable understanding.

## Relationship to Core

Nothing in the sidecar should be imported by, depended on by, or shipped with core Gradience. Promotion from sidecar to core requires meeting all five promotion criteria (see `strategy_memo.md` §11).

The sidecar may use core Gradience as a dependency (e.g. `gradience.api`, CLI commands, telemetry readers) but never the reverse.

## Directory Structure

```
sidecar/
├── README.md              # Index page — start here for re-entry
├── CLAUDE.md              # This file — conventions and identity
├── glossary.md            # Frozen canonical term definitions
├── strategy_memo.md       # Founding strategy document
├── Makefile               # One-command regeneration and validation
├── studies/               # Individual research studies
│   └── TEMPLATE.md        # Study template
├── panels/                # Reusable experimental panel definitions
│   └── TEMPLATE.md        # Panel template
├── results/               # Empirical outputs (tables, metrics, artifacts)
├── notes/                 # Short interpretation and decision documents
│   └── TEMPLATE.md        # Note template
├── figures/               # Publication-quality visual outputs
└── benchmarks/            # Reusable scripts for running canonical panels
```

## Conventions

### Studies

Each study lives in `studies/` as either a single `.md` file or a subdirectory with a `README.md` plus supporting files. Every study must contain:

- **Question** — one crisp question
- **Design** — what is being compared, on what data, using what method
- **Outputs** — what was measured or produced
- **Conclusion** — what was learned
- **Implication for core** — whether anything is promotable, and if not, why not

### Panels

A panel is a reusable definition of an experimental comparison. Panels live in `panels/` and define:

- **Anchors** — which adapter pairs or task pairs
- **Conditions** — backbone, seed, training config
- **Metrics** — what is measured
- **Rerun protocol** — how to repeat

Panels are referenced by studies. A study uses one or more panels.

### Results

Raw and processed empirical outputs. Naming convention: `{study_id}_{description}.{ext}` (e.g. `s01_anchor_rerun_table.json`).

### Notes

Short (1–3 page) interpretation documents. Types:

- **Implication note** — what a result means for core
- **Closure note** — why a line of inquiry is being paused or abandoned
- **Promotion decision** — formal assessment against the five promotion criteria
- **"Not stable enough" note** — documenting a negative result for the record

### Figures

Publication or post-quality outputs. Should be self-contained and captioned.

### Benchmarks

Runnable scripts. Each benchmark must:

- be executable with `python benchmarks/{name}.py`
- document its dependencies and expected runtime
- produce output into `results/`

## Research Programs

The sidecar is organized around three programs (see `strategy_memo.md` §7):

- **Program A** — Cross-task severity lab (severity subtypes, catastrophic anchors, adjudication panels)
- **Program B** — Representation interference lab (layerwise conflict, module-level competition, output-space incompatibility)
- **Program C** — Evidence-pack / adjudication benchmark lab (controlled panels, reproducible bundles, case dossiers)

## Current Priority

**Catastrophic Cross-Task Interference Program** (§8 of strategy memo), with three initial workstreams:

1. **Workstream A** — Catastrophic anchor replication
2. **Workstream B** — Layerwise conflict contrast
3. **Workstream C** — Output-space / task-head incompatibility probe

## Promotion Rules (summary)

A finding may be promoted into core Gradience only when it:

1. Solves a real workflow problem
2. Replicates across backbone or clearly defined regime
3. Improves preflight decisions beyond existing core signals
4. Can be expressed conservatively and simply
5. Does not add more confusion than value

Most sidecar findings will remain research-only. That is acceptable.

## Anti-Patterns

Do **not**:

- Import sidecar code from core Gradience
- Treat the sidecar as a feature staging area
- Prioritize UX polish over empirical clarity
- Run broad sweeps before sharp questions are defined
- Build infrastructure without a driving research question
