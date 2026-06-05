# Gradience Research Inventory

**Compiled**: April 4, 2026
**Purpose**: Comprehensive index of all Gradience-related research materials across the filesystem, to support consolidation into a single authoritative location.

---

## 1. Active Repositories

### Gradience I — Main Library (`~/code/gradience/`)
The PyPI-published Python package (v0.11.0). Contains source code, tests, CLI, bench suite, and all current documentation.

- `gradience/` — Package source (api, cli, vnext/, bench/, research/, etc.)
- `tests/` — ~60 test files
- `docs/technical-report.md` — Primary technical report (updated April 4, 2026)
- `docs/THEORY.md` — Theoretical foundations document (updated April 4, 2026)
- `docs/plans/` — Spec documents including verdict boundary stress-test spec
- `sidecar/notes/` — Research notes (N127 MP partition test, etc.)
- `sidecar/results/` — Experiment result JSON files
- `scripts/` — Analysis and experiment scripts
- `experiments/` — Research experiment scripts
- `experiments/exp01_mistral_gsm8k_multiseed/` — Mistral GSM8K multi-seed experiment artifacts

### Gradience II — GPU Studies & Blog Series (`~/Gradience II/`)
Decoder-only (Mistral-7B, Llama) GPU experiments, blog series, and broader benchmark studies. See `INDEX.md` for full structure.

- `docs/blog_series/` — Canonical published blog posts (Posts 1–5 + announcement)
  - POST_1: Introduction to spectral fine-tuning analysis
  - POST_2: Geometry vs. loss (Mistral-7B compression, 50% reduction, 3 seeds)
  - POST_3: Mistral-7B merge study (r=0.846 dominance prediction, 2.4× same/cross-task separation, t=12.985, 27 cross-task pairs)
  - POST_4: Reanalysis corrections
  - POST_5: Spectral microscope (three-act gradient alignment, expand-then-compress PR)
- `docs/implementation/merge_experiment_report.md` — MNLI×QNLI Mistral-7B merge experiment (3-condition)
- `results/study14_broader_benchmarks/` — 29 public LoRA adapters audited across 8 base models, 5 task categories
- `results/telemetry/` — Primary telemetry data (601 Hessian records, 1,200 geometry records)
- `reanalysis/` — March 2026 corrected analysis scripts and report
- `signals-telemetry/` — Curvature forecasting study (active)
- `src/` — Source code for analysis, experiments, telemetry, tests
- `data/` — Configs and word lists for embedding experiments

---

## 2. Downloads (Uncategorized Research Documents)

Location: `~/Downloads/`

### Papers & Drafts
| File | Content | Relevance |
|------|---------|-----------|
| `GRADIENCE_PAPER_DRAFT_REVISED.md` | Full academic paper draft — Llama-2-7B results, 5 adapter pairs, dominance reduced 0.758→0.235, complete methodology | **Critical** — primary paper draft |
| `DFA_WORKSHOP_PAPER.md` | Formal workshop paper on Detrended Fluctuation Analysis of spectral complexity (F=116.86) | **High** — publishable manuscript |
| `GRADIENCE_EXECUTIVE_SUMMARY.md` | Updated March 2026, 86 adapter audit, 4 empirical pillars | **High** — summary document |

### Blog Posts (Later Series)
| File | Content | Relevance |
|------|---------|-----------|
| `SERIES_POST_7_FINAL.md` | Post 7: 114 adapters, 22 architectures, broader benchmarks | **Critical** — major scaling study |
| `POST8_FINAL.md` | Post 8: Recommendation engine, structural→behavioral validation | **Critical** — system design |

### Studies
| File | Content | Relevance |
|------|---------|-----------|
| `STUDY16B_DRAFT.md` | Study 16: 5 Llama-2-7B pairs, end-to-end merge ablation (Frobenius ratios up to 19.7×) | **Critical** — merge validation |
| `STUDY17A_INTERIM_RESULTS.md` | Study 17: Clean negative result — 95% energy compression too conservative as pre-merge cleanup | **High** — important null finding |

### Specs & Plans
| File | Content | Relevance |
|------|---------|-----------|
| `spec-phase-a.md` | v0.12.0 spec: subspace overlap + margin confidence | **High** — next version roadmap |
| `LORA_FINDINGS.md` | LoRA findings compilation | **Medium** — may overlap with other docs |

---

## 3. Documents / AI Research

Location: `~/Documents/AI Research/`

### Abstracts & Summaries
- `ABSTRACTS.md` — 9 research abstracts covering the entire Gradience program

### Code
- `Code/Deleuzean_AI/` — 17 Python scripts (Hessian computation, curvature analysis, training experiments)

### Experiment Data
- `Experiment_Data/Gradience_Experiments/` — Archived experiment outputs (Studies S6–S11, telemetry data, signals analysis)

### Manuscripts
- `Manuscripts/Deleuzean_AI/` — ~35 documents including curvature telemetry manuscripts, philosophical framing
- `Manuscripts/Philosophy_and_AI/` — ~18 documents including LLM intellectual stress tests

### Findings & Notes
- `Findings_and_Notes/` — `LORA_FINDINGS.md`, Gradience recommendations, accumulated research notes

### Protocols
- `Protocol_and_Methods/` — Research protocols including merge audit protocol

### Papers
- `Papers/` — Academic papers (own and reference literature)

### Reference Texts
- `Reference_Texts/` — Background reading and theoretical sources

---

## 4. Philosophy (Separated)

**Relocated**: April 2026. All philosophical materials formerly in `archive/philosophy/`
have been moved to a top-level `~/Gradience II/philosophy/` directory, separate from
the engineering archive. See `philosophy/INDEX.md` for rationale and structure.

```
Gradience II/philosophy/
├── manuscripts/       — Published and draft philosophical papers (18 files)
├── frameworks/        — Deleuzean research program design (9 files)
├── model_evaluations/ — LLM intellectual stress tests (20 files)
├── curvature_telemetry/ — Hessian/curvature research strand (6 files)
└── notes/             — Working notes, study protocols, misc (27+ files)
```

Additional philosophy materials on the local filesystem (not yet consolidated):
- `~/Documents/Research/Deleuzean AI/` — Philosophical framework documents
- `~/Documents/Research/Philosophy Writings/` — Broader philosophical work
- `~/Documents/AI Research/Manuscripts/Philosophy_and_AI/` — ~18 manuscripts

---

## 5. Desktop

Location: `~/Desktop/`

- `Gradience_LoRA_Decision_Infrastructure.pdf` (15 MB) — Likely a comprehensive PDF report or presentation on the decision infrastructure

---

## 6. Other Locations

### PycharmProjects (`~/PycharmProjects/`)
- May contain earlier Gradience development projects or experiment scripts

### iCloud Drive Archive (`~/iCloud Drive (Archive)/`)
- Searched; no significant Gradience-specific materials found beyond what's indexed above

---

## 7. Consolidation Priority

### Tier 1 — Must consolidate immediately
These are authoritative documents not currently in either repository:

1. `~/Downloads/GRADIENCE_PAPER_DRAFT_REVISED.md` — The paper draft
2. `~/Downloads/SERIES_POST_7_FINAL.md` — Post 7 (114 adapters, 22 architectures)
3. `~/Downloads/POST8_FINAL.md` — Post 8 (recommendation engine)
4. `~/Downloads/STUDY16B_DRAFT.md` — Study 16 (Llama-2-7B merge ablation)
5. `~/Downloads/STUDY17A_INTERIM_RESULTS.md` — Study 17 (compression negative result)
6. `~/Downloads/DFA_WORKSHOP_PAPER.md` — DFA workshop paper
7. `~/Downloads/spec-phase-a.md` — v0.12.0 spec
8. `~/Desktop/Gradience_LoRA_Decision_Infrastructure.pdf` — Decision infrastructure report

### Tier 2 — Should consolidate for completeness
9. `~/Downloads/GRADIENCE_EXECUTIVE_SUMMARY.md` — Executive summary
10. `~/Documents/AI Research/ABSTRACTS.md` — Program abstracts
11. `~/Documents/AI Research/Findings_and_Notes/` — Accumulated findings
12. `~/Documents/AI Research/Protocol_and_Methods/` — Research protocols

### Tier 3 — Archive / reference
13. `~/Documents/AI Research/Code/Deleuzean_AI/` — Historical analysis scripts
14. `~/Documents/AI Research/Experiment_Data/` — Raw experiment archives
15. `~/Documents/AI Research/Manuscripts/` — Philosophical and theoretical manuscripts
16. `~/Documents/Research/Deleuzean AI/` — Philosophy framework documents

---

## 8. Recommended Consolidation Structure

The natural home for consolidated materials is `~/Gradience II/`, which already has the most organized structure. Suggested additions:

```
Gradience II/
├── docs/
│   ├── papers/           ← PAPER_DRAFT_REVISED, DFA_WORKSHOP_PAPER
│   ├── blog_series/      ← POST_7, POST_8
│   ├── blog_drafts/      ← EXECUTIVE_SUMMARY
│   └── specs/            ← spec-phase-a (create if needed)
├── results/
│   ├── study16_merge_ablation/    ← STUDY16B
│   └── study17_compression/       ← STUDY17A
├── philosophy/            ← SEPARATED (April 2026) — manuscripts, frameworks,
│   ├── manuscripts/         model evaluations, curvature telemetry, notes
│   ├── frameworks/          See philosophy/INDEX.md for full structure
│   ├── model_evaluations/
│   ├── curvature_telemetry/
│   └── notes/
└── archive/
    ├── abstracts/         ← ABSTRACTS.md
    ├── protocols/         ← Research protocols
    ├── findings/          ← Accumulated findings notes
    └── experiment_data/   ← Historical experiment archives
```

The philosophy separation reflects a deliberate boundary: the Deleuzean
research program motivated the spectral-geometry tooling, but the two now
have independent evidentiary standards and audiences.

---

## 9. External Reference Literature

External preprints and publications that intersect with Gradience's
empirical or theoretical surface. Entries are flagged by *integration
priority* — whether the work is live for citation, experimental
replication, or just tracking.

| Paper | arXiv | Integration priority | Notes |
|-------|-------|---------------------|-------|
| Xu, Yongzhong — *The Spectral Edge Thesis* | [2603.28964](https://arxiv.org/abs/2603.28964) | **Medium — replicate before integrating** | Formal framework: phase transitions in training controlled by spectral gap of rolling-window Gram matrix of *parameter updates*. Identifies $k^*$ (largest intra-signal gap) as dynamically privileged; $k^* \leq 3$ across six model families. Defines adiabatic parameter $A = \\|\\Delta G\\|_F / (\\eta g^2)$. 19/20 pre-registered predictions confirmed across thesis + companion (arXiv:2603.15678). Structurally aligned with Gradience's curvature telemetry (§16a) and N133's stable-rank concentration. Caveats: solo-authored March 2026, no peer review yet; Xu's object is the update-trajectory Gram, Gradience's is weight-space SVD — mapping requires care. Cited in THEORY.md §4 (candidate formal framework) and §7.2 (curvature-partition correspondence). Dedicated replication on N07 Experiment B's existing dual-instrument data is the natural next step. |
| Xu, Yongzhong — *Spectral Edge Dynamics of Training Trajectories* (companion) | [2603.15678](https://arxiv.org/abs/2603.15678) | **Medium — tracking** | Companion to the thesis paper, extends the framework. Cited only as the source of the remaining 12/20 predictions validated outside the primary thesis paper. Not an independent integration target. |
| Roi Paul (et al.) — Spectral Geometry of Fine-Tuning at Decoder Scale | [2604.08844](https://arxiv.org/abs/2604.08844) | **High — already integrated** | Independent validation of spectral fingerprinting at 3B-decoder scale; replicates cross-method failure pattern observed in N133. Already cited in post-N134 thinking and THEORY.md §3. No new action required — confirmatory rather than novel. |
| Anon. / preprint — *Spectral Surgery for LoRA Refinement* | [2603.03995](https://arxiv.org/abs/2603.03995) | **Low — note only** | Claims trained LoRA updates have inefficient spectra: some directions are not just low-energy but actively detrimental. Gradient-guided SV-level reweighting (subspace directions fixed) shows single-adapter gains. Confirmatory for Gradience's low-utilization finding (Study 14, energy_rank_90 distributions) but operationally orthogonal: (a) requires gradient computation on a calibration set, violating Gradience's training-free CPU-audit regime; (b) targets single-adapter refinement, whereas Gradience §10/Study 17 found that pre-merge spectral compression does not improve merge outcomes (different problem space). Worth a one-line mention in THEORY.md about "gradient-informed SV reweighting" as an adjacent method class; not worth building infrastructure to reproduce. |
| Steele, Brady — *Subspace Geometry Governs Catastrophic Forgetting in Low-Rank Adaptation* | [2603.02224](https://arxiv.org/abs/2603.02224) | **High — test on existing data** | Derives closed-form geometric law for LoRA forgetting: ℱ = α(1−cos²θ_min) + β, where θ_min is minimum principal angle between task gradient subspaces. Validated on synthetic tasks (r=0.994), Split-CIFAR100 (ViT-LoRA), sequential GLUE (RoBERTa-LoRA). Key finding: approximate rank-invariance at high subspace angles (CV ≈ 0.8% synthetic, 10–19% real). Directly relevant to Gradience's principal-angle framework (THEORY.md §6) and conjunctive failure model: the cos²θ_min functional form could formalize the V-module pathology signal. Rank-invariance explains why triage works despite ignoring nominal rank — the operative signal is geometric. Natural test: compute cos²θ_min from existing merge pair principal angles and check whether the same functional form predicts merge degradation (not just forgetting). Single-authored preprint, Feb 2026, not yet peer-reviewed. |
| Feng, Lin, Kao — *Learning in the Fisher Subspace: A Guided Initialization for LoRA Fine-Tuning (FILet)* | [2605.01046](https://arxiv.org/abs/2605.01046) | **Medium — methodology candidate for open question** | Formulates LoRA initialization via Fisher information: uses Kronecker-factored empirical covariances to estimate data-induced parameter sensitivity, demonstrating that data-aware Fisher sensitivity outperforms weight-only SVD (PiSSA) for subspace selection. Directly addresses THEORY.md §3 open question: "Does erank(ΔW) predict erank(F) restricted to the adapter subspace?" FILet's Kronecker-factored Fisher computation provides tractable methodology for per-layer Fisher spectrum estimation. Theoretical implication: post-hoc SVD metrics may underestimate task-relevant dimensionality when task data activates directions spectrally quiet in W₀ but high-Fisher. Could explain task-dependence of W₀ concentration → alignment relationship (holds for QNLI but not SST-2 in N127). GPU-blocked for testing. May 2026 preprint, not yet peer-reviewed. |
| Shuttleworth, Meister, Ren, Cotterell — *LoRA vs Full Fine-tuning: An Illusion of Equivalence* | [2410.21228](https://arxiv.org/abs/2410.21228) | **Medium — cite with integration test** | NeurIPS 2025. LoRA produces "intruder dimensions" — high-ranking singular vectors absent in pre-trained model. These are causally related to behavioral differences; forgetting concentrates in intruder dimensions; scaling them down recovers pre-training distribution. Complementary to Gradience's spectral partitioning (THEORY.md §6): intruder dimensions may be a third spectral category beyond shared high-SV and task-specific low-SV — emergent high-SV directions unique to each adapter. Could explain why cross-task high-SV alignment is low (0.133 vs 0.634 same-task in N127): cross-task adapters develop different intruder dimensions. Potential mechanism for V-module pathology in the conjunctive failure model. Testable: check whether intruder dimension counts correlate with V-module dimensionality ratio on existing adapter pairs. Peer-reviewed (NeurIPS 2025), strong causal evidence. |

### How to use this section

- **Before citing a paper in THEORY.md or technical-report.md**, check
  here first to see whether it has been assessed and what its integration
  priority is. Papers at "Low" priority should be cited sparingly;
  "Medium" papers cited with honest caveats about replication status;
  "High" papers can be cited as we would cite peer-reviewed literature.
- **Before building infrastructure around a framework from an external
  paper**, run a small dedicated replication experiment on existing
  Gradience data. The Spectral Edge Thesis is the current candidate for
  this test (N07 Experiment B's dual-instrument data is the natural
  input).
- **New external references should be added here** rather than cited
  directly in THEORY.md or technical-report.md without assessment.
  This section is the gate between "the agent found a paper" and "we
  are building on top of a paper."

