# DECODER_ONLY_SPECTRAL_FINGERPRINTING_GPU_RETURN_STUDY_SPEC
## Repo-Facing GPU-Return Execution Plan

## Purpose

Define the first GPU-return proving-ground study:

> **Decoder-Only Spectral Fingerprinting**

Central question:

> **Given that the CPU public decoder census already shows bounded decoder-side structure in found artifacts, what share of spectral variation is architecture-driven vs task-driven under controlled matched conditions?**

This is a roadmap-ready execution spec for when compute reopens.
It is now framed as a causal disambiguation study rather than a signal-existence check.

## Why This Is the Next Major Study

CPU phase is now consolidated with bounded decisions across:

- rank-proxy line (bounded positive)
- ablation proxy line (resolved in bounded regime)
- phase-probe add-on (bounded_keep)
- merge-aware monitor (bounded_keep; same-task default)
- over-accumulation (exploratory, not policy-ready)

The highest-value unanswered question now sits in decoder-only territory.

New decoder census handoff context:

- public decoder census + task-balanced extension produced bounded ecological structure
- both architecture and task signatures appeared, with strong confound pressure
- result status: `mixed_but_bounded` (useful, not causal)

So the GPU study’s job is now:

- not “does any decoder structure exist?”
- but “what portion is genuinely architecture vs task when confounds are controlled?”

## Scope

### In scope

- decoder-only architectures
- shared protocol with matched tasks per architecture
- adapter/checkpoint artifacts compatible with spectral extraction
- architecture-vs-task decomposition of spectral fingerprints

### Out of scope

- optimizer/control interventions
- broad leaderboard benchmarking
- policy rollout changes from first pass
- universal claims across all model families

## Program Questions

1. Architecture attribution under controls:
   - how much variance remains architecture-led when task and training protocol are matched?
2. Task attribution under controls:
   - how much variance remains task-led within architecture?
3. Metric-level dominance map:
   - which observables are architecture-led vs task-led after confound removal?
4. Census replication check:
   - which ecological census trends persist, weaken, or disappear under controlled training?

## Hypotheses

- H1: decoder spectral variation will retain a nontrivial architecture component under matched controls.
- H2: task effects will persist, but their magnitude will differ by observable (no universal dominance).
- H3: controlled results will split metrics into architecture-led and task-led subsets.
- H4: some census-era effects will attenuate under control, clarifying which ecological signals were confounded.

## Candidate Architectures

Minimum recommended architecture set (GPU-return first pass):

1. Llama-family decoder (baseline anchor)
2. Mistral-family decoder (attention/normalization variation anchor)
3. Qwen-family decoder (tokenization/implementation diversity anchor)

If one family is unavailable, keep at least two families with explicit substitution rationale.

## Task Set (Matched Across Architectures)

Use a small matched set with distinct behavior:

1. Instruction-following style task
2. Reasoning/math style task
3. Domain/topic adaptation task

Selection rule:

- same task definitions and evaluation recipe per architecture
- avoid architecture-specific task substitutions unless forced by data availability

## Artifact Types

Primary artifact targets:

- LoRA/low-rank adapters (preferred for continuity with existing substrate)
- bounded checkpoint deltas if needed for architecture-level contrast

Keep artifact classes fixed for first pass to minimize confounding.

## Spectral Fingerprint Metrics

Core metrics:

- stable rank / effective rank
- energy concentration profile
- top singular-value structure
- concentration and tail summaries already in current toolkit

Secondary probes:

- edge-gap (`sigma1/sigma2`) as companion
- HTSR alpha where fit quality passes thresholds

Do not promote secondary probes to primary decision metrics in first pass.

## Minimum Cohort

Suggested floor (first executable pass):

- 2 architectures minimum (3 preferred)
- 3 tasks per architecture
- 3 adapters/checkpoints per architecture-task cell when feasible

Practical minimum cell count target:

- 18 artifacts (2 arch x 3 tasks x 3 variants)

Preferred:

- 27 artifacts (3 arch x 3 tasks x 3 variants)

## Analysis Plan

1. Within-architecture task spread:
   - estimate task-conditioned shifts with fixed architecture + matched protocol.
2. Cross-architecture fixed-task spread:
   - estimate architecture-conditioned shifts with fixed task + matched protocol.
3. Controlled variance decomposition:
   - architecture, task, and interaction partitions with transparent confound controls.
4. Metric-level map:
   - report architecture-led vs task-led observables explicitly.
5. Census-to-GPU bridge:
   - direct pre/post comparison against the census handoff summary for consistency checks.

## Success / Failure Criteria

### Success condition

Controlled decoder results cleanly separate architecture-led and task-led signal components and explain which census-era signals were ecological vs causal.

### Partial success condition

Signal exists but attribution is mixed/noisy; supports only guarded extension language and targeted follow-up.

### Negative completion condition

No stable controlled structure beyond noise/confounding; census structure does not transport to controlled decoder regime.

All outcomes are useful and should be documented as such.

## Execution Readiness Deliverables

Before GPU opens:

- finalized cohort table template
- artifact manifest schema
- metric extraction script entrypoints
- analysis notebook/script skeleton
- report template for success/partial/negative outcomes

At GPU return day 1:

- lock architecture/task cohort
- begin artifact generation + extraction
- run first 20% pilot cell to validate protocol

## Deliverables

Planned outputs under a new study directory:

- `field_trials/decoder_spectral_fingerprinting/cohort_definition.{md,json}`
- `field_trials/decoder_spectral_fingerprinting/artifact_manifest.json`
- `field_trials/decoder_spectral_fingerprinting/fingerprint_table.{md,json}`
- `field_trials/decoder_spectral_fingerprinting/architecture_task_decomposition.{md,json}`
- `field_trials/decoder_spectral_fingerprinting/study_memo.md`

## Guardrails

- do not claim decoder generality from partial cohort
- do not over-interpret secondary probes
- do not convert findings into product policy without replication
- keep architecture and task matching explicit in every summary

## Bottom Line

This is the clearest next proving ground after CPU consolidation and decoder census handoff:

> **run a controlled decoder-only study to causally disambiguate architecture vs task contributions to spectral fingerprints, using the census as ecological prior evidence.**
