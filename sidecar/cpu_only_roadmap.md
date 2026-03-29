# Medium-Term CPU-Only Roadmap Memo
## Purpose
This memo defines a medium-term roadmap for Gradience work that can be completed **without GPU resources**.
The goal is to use CPU-only time productively in a way that:
- increases the immediate practical value of core Gradience
- sharpens the sidecar's scientific direction
- improves reproducibility, handoff, and reuse
- prepares the project to move quickly once GPU resources return
This roadmap assumes:
- core Gradience is now positioned as a **mixed-task inventory preflight system**
- same-task safety is well supported on small encoder models
- cross-task boundary detection is stable and useful
- cross-task severity grading remains open
- the sidecar's strongest current research concept is **instability**, not severity
The roadmap is designed to avoid drift into speculative feature work while GPU resources are unavailable.
---
# 1. Strategic framing
The CPU-only period should be used to strengthen two lanes in parallel:
## Lane A — Core Gradience hardening
Focus on making core Gradience:
- easier to run repeatedly
- easier to interpret
- easier to hand off
- easier to trust
- more workflow-oriented
This is the highest immediate-payoff lane.
## Lane B — Sidecar instability research
Focus on turning the sidecar's current evidence into a cleaner, more coherent scientific program, especially around:
- instability
- catastrophic anchors
- local/mechanistic interpretation
- evidence organization
- DeBERTa readiness for later adjudication
This is the highest scientific-payoff CPU-only lane.
---
# 2. Roadmap structure
This roadmap is organized into:
1. **Core Gradience projects**
2. **Sidecar research projects**
3. **Shared evidence/communication projects**
4. **Sequencing**
5. **Decision points**
---
# 3. Core Gradience CPU-only projects
These projects should be treated as the highest-priority practical work.
## Project A — Inventory Action Summary Completion
### Goal
Finish and stabilize the `INVENTORY ACTION PLAN` concept so that every major preflight summary explicitly answers:
- what to exclude
- what to prioritize
- what to treat as cross-task caution
- what to evaluate first
### Why
This is the most direct path from "signals" to "usable plan."
### Deliverables
- action summary design doc
- implemented action summary block
- tests across same-task, mixed-task, and messy inventories
- walkthrough updates
### Completion signal
A user can open a summary and see a clear plan without mentally re-synthesizing all the underlying outputs.
---
## Project B — Preflight Run Bundle Hardening
### Goal
Make repeated preflight use feel like a workflow rather than a pile of outputs.
### Core elements
- stable per-run bundle layout
- `preflight_summary.md`
- `preflight_summary.json`
- `run_manifest.json`
- stable "current result" artifact
- `compare_to_previous.md`
### Why
This is what makes Gradience repeatable and handoff-friendly.
### Deliverables
- run-bundle design doc
- canonical file layout
- current-result pointer
- previous-run comparison artifact
- tests
- workflow doc updates
### Completion signal
A user can:
1. run preflight
2. open one summary
3. compare it to the previous run
4. know which result is current without hunting through files
---
## Project C — Provenance / Trust Language Completion
### Goal
Make source credibility more explicit by distinguishing:
- verified behavioral evidence
- claimed / user-provided evidence
- missing evidence
- confidence in source credibility
### Why
This is tightly aligned with Gradience's identity as a high-trust preflight tool.
### Deliverables
- provenance/trust language design doc
- source QA rendering updates
- inventory summary trust snapshot
- trust language in action plans where appropriate
- tests
- docs/walkthrough updates
### Completion signal
A user can tell, quickly and explicitly, what kind of evidence supports each source.
---
## Project D — Continued Light Summary UX Pass
### Goal
Make inventory summaries cleaner, more modular, and more reusable.
### Focus
- cleaner summary block structure
- stronger reduced-candidate-set presentation
- more consistent action wording
- tighter alignment between human-readable and machine-readable summary artifacts
### Why
This improves usability without changing logic.
### Deliverables
- block design doc
- wording style guide
- refined summary renderers
- tests
- doc alignment pass
### Completion signal
The summary becomes the obvious first artifact to read in every workflow.
---
## Project E — Batch / Repeated Preflight Ergonomics
### Goal
Make it easier to run Gradience on multiple inventories or repeated versions of the same inventory.
### Possible scope
- batch preflight runner
- summary-of-summaries table
- multi-inventory comparison output
- canonical aggregation script
### Why
This increases real-world practical payoff without new science.
### Deliverables
- design note
- one stable batch entry point
- one aggregated markdown or JSON output
- tests
### Completion signal
Running Gradience across multiple inventories or snapshots no longer feels ad hoc.
---
# 4. Sidecar CPU-only projects
These projects should deepen the scientific line without needing new training.
## Project F — Instability Program Consolidation
### Goal
Promote instability from "promising finding" to "working research program."
### Questions
- Is instability more portable than severity?
- Can pairs be grouped into stable vs unstable regimes?
- Does instability explain why severity signals failed to generalize?
- Is catastrophic behavior better understood as instability than as high average severity?
### Deliverables
- instability synthesis note
- instability case table updates
- additional figures/tables
- refined classification notes
- explicit DeBERTa adjudication framing
### Completion signal
Instability becomes the default conceptual lens for sidecar interpretation.
---
## Project G — Local Artifact Mining
### Goal
Use already-saved artifacts to ask more local structural questions without retraining.
### Candidate analyses
- per-layer norm concentration
- layerwise conflict concentration
- within-pair structural dispersion
- catastrophic vs mild local structural contrasts
- instability-localization patterns
### Why
This may reveal more than coarse global summaries and is entirely CPU-feasible if artifacts exist.
### Deliverables
- one or more analysis scripts
- structured JSON outputs
- figures
- interpretation note(s)
### Completion signal
The sidecar gains at least one new local/mechanistic descriptive layer from existing artifacts.
---
## Project H — Catastrophic Anchor Dossiers
### Goal
Turn catastrophic or near-catastrophic anchors into stable reference cases.
### For each dossier
Capture:
- pair
- backbone
- severity behavior
- seed range
- instability score
- backbone shift
- what core signals said
- what failed to predict the outcome
- why this case matters
### Why
This creates durable research memory and better anchor cases for future studies.
### Deliverables
- dossier template
- completed dossier set for current anchors
- note linking dossiers to instability hypothesis
### Completion signal
Every important catastrophic case can be re-entered quickly and compared systematically.
---
## Project I — Backbone-Local Interpretation Notes
### Goal
Separate:
- what appears stable within DistilBERT
- what appears stable within RoBERTa
- what fails across them
### Why
Even if nothing is promotable to core, backbone-local regularities may still matter scientifically.
### Deliverables
- per-backbone notes
- per-backbone tables
- local-pattern comparison memo
### Completion signal
The sidecar clearly distinguishes local regularity from cross-backbone portability.
---
# 5. Shared evidence / communication projects
These projects strengthen both core and sidecar by organizing what is already known.
## Project J — Evidence Atlas 2.0
### Goal
Upgrade the evidence atlas into the durable internal source of truth.
### It should distinguish:
- stable core claims
- closed questions
- open questions
- non-promotable findings
- promotable-candidate findings
- sidecar-only findings
- current best evidence by concept
### Why
This reduces repetition and confusion later.
### Deliverables
- expanded atlas
- better sectioning
- stronger linkage to notes and studies
- promotion/non-promotion markers
### Completion signal
Future work can begin from the atlas instead of from memory.
---
## Project K — Promotion Dossier Template
### Goal
Create a standard way to evaluate whether a sidecar finding belongs in core.
### Template questions
- what workflow problem does this solve?
- what evidence supports it?
- what replication exists?
- what failed?
- what would promotion require?
- what would block promotion?
### Why
This keeps the core/sidecar split disciplined.
### Deliverables
- promotion dossier template
- at least 2 filled examples
  - one non-promotable signal
  - one maybe-promotable signal
### Completion signal
Promotion decisions become explicit rather than implicit.
---
## Project L — Publication-Ready Figure / Table Package
### Goal
Use CPU time to prepare strong reusable artifacts for later writing.
### Possible artifacts
- utility round summary table
- regime contrast figure
- instability summary panel
- same-task closure panel
- advisory evidence card
- sidecar anchor comparison figures
### Why
This raises the communication quality of both repos without needing new compute.
### Deliverables
- figures with scripts
- figure inventory note
- a single "best current visuals" index
### Completion signal
A future post, paper section, or talk can be assembled quickly from stable assets.
---
# 6. Recommended sequencing
## Phase 1 — Finish high-payoff core tightening
Priority order:
1. Inventory Action Summary Completion
2. Preflight Run Bundle Hardening
3. Provenance / Trust Language Completion
4. Continued Light Summary UX Pass
### Goal
Make core Gradience feel complete as a preflight workflow tool.
---
## Phase 2 — Add repeatability and operational scale
Priority order:
5. Batch / Repeated Preflight Ergonomics
6. Evidence Atlas 2.0
7. Promotion Dossier Template
### Goal
Make the tool easier to reuse and the project easier to steer.
---
## Phase 3 — Deepen the sidecar on CPU
Priority order:
8. Instability Program Consolidation
9. Catastrophic Anchor Dossiers
10. Backbone-Local Interpretation Notes
11. Local Artifact Mining
### Goal
Strengthen the sidecar's scientific center while waiting for GPU.
---
## Phase 4 — Communication and packaging
Priority order:
12. Publication-Ready Figure / Table Package
13. Optional outward-facing synthesis writing, if useful
### Goal
Make the current state easy to communicate when needed.
---
# 7. Recommended allocation of effort
If CPU-only time is limited, a good split is:
## 70% core Gradience
Spend most time on:
- action summaries
- run bundles
- trust/provenance clarity
- repeated-use ergonomics
- workflow polish
## 30% sidecar
Spend the remaining time on:
- instability as the main working concept
- anchor dossiers
- local structural mining where possible
- DeBERTa-ready adjudication framing
This keeps immediate payoff high while still advancing the research frontier.
---
# 8. Decision points
This roadmap should include explicit decision points so work does not drift.
## Decision Point 1 — Is core Gradience "workflow-complete enough"?
Ask after Phase 1:
### Criteria
- action summaries are stable
- run bundles work
- current-result artifacts are clear
- provenance/trust language is explicit
- summaries are readable and reusable
### If yes
Proceed to Phase 2 and keep sidecar separate.
### If no
Keep tightening core before adding more sidecar ambition.
---
## Decision Point 2 — Is instability still the strongest CPU-only sidecar concept?
Ask after Phase 3 progress.
### Criteria
- instability continues to outperform severity as a descriptive organizer
- local artifact mining does not undermine the concept
- no better competing CPU-only concept emerges
### If yes
Keep instability as the sidecar's main working concept and use DeBERTa as adjudication.
### If no
Reframe before GPU work resumes.
---
## Decision Point 3 — Is any sidecar result close to promotable?
Ask after additional sidecar analysis and atlas updates.
### Promotion criteria
A finding should move toward core only if it:
1. solves a real workflow problem
2. replicates across backbone or clearly defined regime
3. improves decisions beyond current stable signals
4. can be expressed simply and conservatively
5. does not add more conceptual overhead than value
### If no signal meets that bar
Keep the work in the sidecar and preserve core clarity.
---
## Decision Point 4 — What should happen when GPU returns?
When compute is available again, the immediate question should be:
> which CPU-only preparation work makes the GPU phase most decisive?
At that point:
- DeBERTa should act as an adjudication test for instability portability
- core should not need major structural work before GPU studies resume
- the sidecar should have clear success/failure criteria already written
---
# 9. What not to do during the CPU-only period
Do not:
- reopen same-task blind-spot hunting
- build speculative severity features into core
- create a giant schema or ontology
- broaden into a benchmark without a sharp question
- let the sidecar become a dumping ground
- confuse interpretive signals with promotable signals
- overbuild infrastructure that is not yet needed
This CPU-only period should favor:
- sharpening
- packaging
- repeatability
- disciplined research framing
---
# 10. Summary recommendation
## Highest immediate practical payoff
Keep tightening core Gradience into a polished, repeatable mixed-task inventory preflight workflow tool.
## Highest scientific payoff
Deepen the sidecar around instability, catastrophic anchors, and local/mechanistic analysis using existing artifacts.
## Best medium-term split
- core Gradience: operational, conservative, useful now
- sidecar: exploratory, mechanistic, non-promoted until proven
---
# 11. Bottom line
The CPU-only period should not be treated as dead time.
It is a chance to do two valuable things at once:
1. make core Gradience more repeatable, legible, and useful in real workflows
2. sharpen the sidecar's strongest emerging concept — **instability** — into a more rigorous research program
That is the best medium-term use of CPU-only time until GPU resources return.
