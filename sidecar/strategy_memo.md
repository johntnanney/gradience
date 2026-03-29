# Research Strategy Memo: Gradience Sidecar Direction

## Purpose

This memo defines the medium-term research strategy for work that should **not** live inside core Gradience.

Core Gradience now has a clear and useful center: mixed-task inventory preflight, search-space reduction before merge evaluation, task-boundary risk exposure, same-task safe zone vs cross-task caution zone partitioning, and action-oriented inventory summaries.

That clarity is valuable and should be protected.

At the same time, several unresolved research questions remain scientifically important: why some cross-task pairs degrade mildly while others fail catastrophically, whether any structural or representational signal can grade cross-task severity robustly, and whether catastrophic failures have mechanistic signatures that are more local or architecture-dependent than current stable signals capture.

This memo argues that those questions should be pursued in a **sidecar research track**, not folded back into core Gradience prematurely.


## 1. Strategic Split

### Core Gradience

Core Gradience should remain operational, conservative, workflow-oriented, and useful today. Its job is to help users reduce the candidate space, exclude weak or low-value sources, partition same-task and cross-task regions, and produce a smaller, more defensible evaluation subset.

Core Gradience is **not** the place to continue unresolved severity-grading research unless a signal has already proven stable enough to help real workflow decisions.

### Sidecar Research Track

The sidecar should be the place where open mechanistic and explanatory questions can be pursued aggressively, without destabilizing the main tool. Its job is to explore cross-task severity, catastrophic interference, local conflict mechanisms, representation-level explanations, backbone dependence, and signals that are suggestive but not yet promotable.

The sidecar should optimize for empirical iteration, controlled adjudication studies, mechanistic exploration, evidence accumulation, and future promotion decisions.

It should **not** optimize for stable UX, polished end-user workflows, or conservative product messaging.


## 2. Why a Sidecar Is Needed

### What Core Gradience Now Knows with High Confidence

**Same-task regime.** On small encoder models, same-task pairs are broadly safe across training-style variation, domain shift in a high-transfer task, and source-strength asymmetry. No actionable same-task blind spot was found.

**Cross-task boundary.** The same-task vs cross-task boundary is robust. The task-relationship advisory generalizes across backbones, has 0 false positives in the current evidence base, and is correctly positioned as a boundary detector rather than a severity grader.

**Utility.** The current stable stack is already useful as a mixed-task inventory preflight system. The utility round supports the practical claim that Gradience can substantially reduce candidate space before evaluation in mixed-task inventories.

### What Remains Unresolved

**Cross-task severity grading.** Severity inside the already-flagged cross-task regime remains open. The project has already tested the strongest obvious candidates: exact task-pair identity, core-space shared-basis, pair-risk, format similarity, and source-strength gap. None has proven stable enough across backbones to justify featureization as a general cross-task severity signal.

This is the main reason a sidecar is needed: the severity problem is still scientifically live, but not ready for core.


## 3. Sidecar Mission

> **A structured merge-interference research lab for questions that are too exploratory, too mechanistic, or too unstable to belong in core Gradience yet.**

Its mission is to investigate why catastrophic failures happen, whether cross-task severity has deeper mechanistic structure, and whether any signals survive rerun and backbone replication strongly enough to deserve later promotion into core.


## 4. Recommended Priority: Catastrophic Cross-Task Interference

The highest-priority sidecar program should focus on **catastrophic cross-task interference**. Not cross-task severity in the abstract. Not a broad benchmark first. Not another generic structural-signal sweep.

The sidecar should begin with the specific question:

> **What distinguishes catastrophic cross-task interference from ordinary cross-task degradation, and are those distinguishing features more local, mechanistic, or architecture-dependent than current Gradience signals capture?**


## 5. Why Catastrophic Interference Is the Right Focus

Catastrophic cases are more useful than merely mild or moderate ones because they are more likely to reveal stronger underlying incompatibilities, localized conflict, output-space mismatch, backbone-sensitive interference mechanisms, and sharper distinctions between plausible and implausible explanatory theories.

A mild degradation case may reflect many weak causes. A catastrophic case is more likely to expose a real mechanism.


## 6. Core Research Hypotheses

### Hypothesis 1 — Catastrophic failures are not just the tail of a smooth severity curve

Some catastrophic pairs may reflect qualitatively different interference, not simply "more degradation." This would mean catastrophic cases deserve separate study and severity should not be treated as one smooth scalar without mechanistic differentiation.

### Hypothesis 2 — Catastrophic pairs may show localized conflict signatures

Even if global signals like shared-basis do not generalize cleanly across backbones, catastrophic pairs may still show layerwise spikes, module-level conflict concentration, output-head incompatibility, or local competition patterns. This is the strongest mechanistic hypothesis.

### Hypothesis 3 — Backbone dependence may be part of the phenomenon, not just noise

Negative results on general severity grading do **not** prove that there is no underlying structure. They may instead imply that the structure is backbone-local, the relevant signal is local rather than global, or current summary-level features are too coarse.


## 7. Research Programs

### Program A — Cross-Task Severity Lab

Studies severity subtype replication, catastrophic anchors, cross-task adjudication panels, task-pair interaction stability, and candidate explanatory variables for severity. The most direct continuation of the current empirical work.

### Program B — Representation Interference Lab

Studies layerwise conflict, module-level competition, representational overlap vs functional incompatibility, output-space or task-head incompatibility, and whether similar input format intensifies conflict rather than reducing it. The most ambitious and potentially most original line.

### Program C — Evidence-Pack / Adjudication Benchmark Lab

Provides controlled panels, benchmark scripts, reproducible adjudication bundles, figures and tables, structured result summaries, and case dossiers for catastrophic anchors. The infrastructure and memory layer that keeps the research durable.


## 8. First Sidecar Project: Catastrophic Cross-Task Interference Program

### Workstream A — Catastrophic Anchor Replication

Use known catastrophic or near-catastrophic task pairs as anchors. Questions: Do these cases recur across reruns? Do they recur across backbones? If not, what changes? What features travel with the catastrophic cases when they do recur?

Example anchor class: QNLI × MRPC in the current evidence base.

**Deliverables:** anchor panel definition, rerun table, backbone comparison note, short case dossiers.

### Workstream B — Layerwise Conflict Contrast

Compare catastrophic pairs, broad-degradation pairs, asymmetric pairs, and mild pairs. Questions: Are catastrophic failures more localized? Do they show conflict concentration in specific modules or layers? Do same-format catastrophic pairs differ from cross-format asymmetric pairs in where the conflict appears?

**Deliverables:** layerwise conflict summaries, conflict heatmaps or ranked plots, category-level comparison note.

### Workstream C — Output-Space / Task-Head Incompatibility Probe

Investigate whether catastrophic failures reflect incompatibility at the output or decision-boundary level more than in broad-body geometry. Questions: Are some catastrophic failures driven by incompatible task heads or decision surfaces? Do "same format" tasks fail because they recruit similar representations for incompatible objectives?

**Deliverables:** hypothesis note, simple probes, comparison against non-catastrophic pairs.


## 9. Sidecar Structure

See `CLAUDE.md` for directory layout and conventions.


## 10. What Should Not Be Prioritized First

**Broad task-pair atlas** — interesting but premature. Exact task-pair severity did not generalize strongly enough across backbones.

**Another generalized core-space repo** — too close to a weakened prior hypothesis.

**Benchmark repo without a sharp question** — useful later, but infrastructure without a research spine dilutes momentum.

**Experimental severity feature module** — too early. The sidecar should explain first, not productize.


## 11. Promotion Rules Back Into Core

A finding may be promoted into core Gradience only when it:

1. Solves a real workflow problem
2. Replicates across backbone or clearly defined regime
3. Improves preflight decisions beyond existing core signals
4. Can be expressed conservatively and simply
5. Does not add more confusion than value

This is a deliberately high bar. Most sidecar findings may remain research-only. That is acceptable.


## 12. Decision Rule: Sidecar vs Core

**Sidecar** if it tries to explain cross-task severity, depends on exploratory structural or representational signals, is not yet stable across backbones or reruns, needs controlled adjudication before it can be trusted, or is more useful for research understanding than immediate workflow support.

**Core** only if it improves preflight usability now, is already empirically stable enough, does not blur the current simple workflow, and directly improves search-space reduction, trust, or actionability.


## 13. Practical Payoff

The sidecar's immediate payoff is **not** a new product feature. Its payoff is better understanding of catastrophic failures, better criteria for what should never be promoted into core, stronger scientific explanations for why core Gradience stops at boundary detection today, a durable evidence base for future promotion decisions, and paper/post-quality mechanistic insight.


## 14. First-Phase Deliverables

The first sidecar phase should produce:

1. One catastrophic anchor replication note
2. One layerwise conflict contrast study
3. One output-space incompatibility note
4. One benchmark-ready catastrophic anchor panel
5. One synthesis memo on what appears stable, local, and not yet promotable


## 15. Bottom Line

The sidecar with the highest research value is not a general severity sandbox, a benchmark warehouse, or a second version of Gradience.

It should be:

> **A focused interference lab built around catastrophic cross-task failures, because those are the cases most likely to reveal mechanisms that current stable Gradience cannot yet see.**

Core Gradience should remain narrow, operational, high-trust, and workflow-oriented.

The sidecar should explore catastrophic interference, local conflict, output-space incompatibility, and backbone-sensitive severity mechanisms.

That is the healthiest medium-term split for the project.
