# N134 Publication Bundle — Manifest

Single-folder reference for everything N134-publication-relevant. Every entry below is a symlink to a canonical file or directory elsewhere in the repo — **not a copy**. Editing through any link edits the canonical file. This keeps the bundle navigable from one place while preventing the drift that a copy-based bundle would accumulate.

**Read order for someone coming to N134 cold:** `00_spec.md` → `01_report.md` → `paper/draft_v2_thesis_b_outline.md` → dive into `data/` and `scripts/` as needed. `02_incident_log.md`, `03_reproducibility_check.md`, and `04_repro_convention.md` are reference documents for specific questions, not sequential reading.

**Canonical-state tag:** `n134-submission-draft-v1` (commit `f462767`). The env-var amendment and this manifest are post-tag additive commits.

---

## Narrative documents

Numbered for reading order; each is a single self-contained markdown document.

| Link | Canonical path | Role |
|---|---|---|
| `00_spec.md` | `sidecar/notes/n134_spec.md` | Pre-registration v3.1. Hypothesis H1, decision rule, confound decomposition, design commitments. Committed *before* data collection began. |
| `01_report.md` | `sidecar/notes/n134_report.md` | Findings report v1.0.1. Abstract, methods, H1 outcome, four replications, four-method comparison, discussion, deviations, appendices. Tagged `n134-report-v1`. |
| `02_incident_log.md` | `sidecar/notes/n134_incident_log.md` | Phase 0 CPU-contention incident (2026-04-19). Diagnostic trace, remediation, training-trajectory verification, operational rules added. No scientific consequence for the trained adapters. |
| `03_reproducibility_check.md` | `sidecar/notes/n134_reproducibility_check.md` | T6 four-tier reproducibility-check trace. Tier-by-tier results, rank-on-residuals precision observation, environment-gap and data-availability-gap documentation, script-hygiene note. |
| `04_repro_convention.md` | `sidecar/conventions/reproducibility_check_tiers.md` | Reusable four-tier reproducibility-check convention derived from the N134 T6 experience. Grounded in N134 but generalized for future consolidations. |
| `05_icc_spec.md` | `sidecar/notes/n134_icc_spec.md` | Cross-seed ICC specification for $S_{\mathrm{H1}}$ as an instrument. Commits to ICC(2,1) absolute-agreement single-measurement with Shrout–Fleiss primary CI and block-bootstrap secondary, SEM reported alongside. Pre-implementation spec for the Appendix-D `[TODO]` in `paper/draft_v2_thesis_b.tex`; implementation (`scripts/n134/09_analysis_icc.py`, `sidecar/results/n134/analysis_icc.json`) is follow-on work. See `paper/revision_notes.md` RN-015. |

---

## Paper artifacts

Under `paper/`. Drafts, revision tracking, bibliography, figures, captions, history.

| Link | Canonical path | Role |
|---|---|---|
| `paper/draft_v1.md` | `papers/n134_workshop/draft_v1.md` | Frozen v1 skeleton. Findings-paper framing. Do not edit. |
| `paper/draft_v2_thesis_b_outline.md` | `papers/n134_workshop/draft_v2_thesis_b_outline.md` | Current revision-planning outline. Repositions paper from findings (v1) to *Measurement Discipline for ML Diagnostics*. 8–10 page target. Use the outline to see intended section purpose and length; use the `.tex` below for actual prose. Post-memo amendment at the bottom supersedes the outline's §2 four-parallel-subsection structure. |
| `paper/thesis_memo.md` | `papers/n134_workshop/thesis_memo.md` | **One-page anchor document preceding §2 revision.** Philosophical commitments the §2 rewrite will track: direct-readout-vs-inferential framing of measurement (names what the paper argues against); jointly-constitutive reading of the four framework components (supersedes the parallel-subsection structure); normative/demonstrative/productive dialectical structure of the three contributions. See CHANGELOG `v2-memo` entry and `revision_notes.md` RN-013. |
| `paper/draft_v2_thesis_b.tex` | `papers/n134_workshop/draft_v2_thesis_b.tex` | **Complete first-pass LaTeX draft** implementing the Thesis B outline as prose. ~4,500 words main body + appendices, three figures included. Expected to go through at least one revision cycle before venue submission; §2 rewrite (tracking `thesis_memo.md`) is the next revision-period work item. One TODO explicit in Appendix D (cross-seed ICC, scheduled as early-revision action). |
| `paper/BUILD.md` | `papers/n134_workshop/BUILD.md` | Compile instructions, dependency list, figure-resolution notes, venue-template-swap guide, anonymization notes for the `.tex`. |
| `paper/draft_current.md` | `papers/n134_workshop/draft_current.md` | Legacy markdown working file (from v1 skeleton era). Superseded by the `.tex` for prose drafting; retained as historical reference. |
| `paper/references.bib` | `papers/n134_workshop/references.bib` | Bibliography. 14 verified entries; one open TODO (Panahi OpenReview FSDxP3ZpAx, ICLR 2026, author list pending). Entries include foundational (Hu LoRA), comparison methods (Stoica KnOTS, Gargiulo TSV, Li SVC), mergeability-prediction program (Zhang & Zhou OSRM, Zhou et al. demystifying, Rahamim et al. mergeability, Bolton et al. SimMerge), decoder-scale evaluation (Hitit et al. 2026 — not Cocchieri), MergeBench (He et al.), and five anonymized self-cites. |
| `paper/figure_captions.md` | `papers/n134_workshop/figure_captions.md` | Caption drafts v1 for the three figures, plus dropped-original-F2 record. Captions drafted here, pasted into paper prose only when finalized. |
| `paper/revision_notes.md` | `papers/n134_workshop/revision_notes.md` | Revision log. RN-000 through RN-012: draft-v1-is-skeleton, Zhou citation resolution, MergeBench-attribution retraction, Panahi verification pending, SVC resolution, SVC-fallback-not-triggered record, Cocchieri→Hitit correction, Panahi retry with ICLR 2026 confirmation, Cocchieri provenance unknown, tag-convention append-only, F2 dropped, partial-ρ precision → paper language, Thesis B repositioning. |
| `paper/CHANGELOG.md` | `papers/n134_workshop/CHANGELOG.md` | Paper version history. v1 → v1.0.1 (attribution corrections) → v2-outline (Thesis B repositioning). |
| `paper/figures/` | `papers/n134_workshop/figures/` | Three paper figures as PDF + PNG: `h1_decision` (Figure 1 H1 scatter), `four_method_forest` (Figure 2), `layer_depth_trend` (Figure 3). |

---

## Empirical data

`data/` → `sidecar/results/n134/`. Entire committed-data directory as a single link. Key files:

- `data/README.md` — summary table of findings with replication criteria.
- `data/analysis_h1.json` — primary H1 decision output (criterion-by-criterion pass/fail, bootstrap CIs, per-pair table).
- `data/analysis_secondary.json` — exploratory per-module / depth-trend / composite-score output.
- `data/method_comparison.json` — Phase 5 four-method triage outputs with bootstrap CIs.
- `data/pair_sample.json` — 69-pair deterministic sample (seed=134; 24 same-task + 45 cross-task).
- `data/environment_dev.txt` — post-hoc dev-environment `pip freeze` with honest-disclaimer header.
- `data/audit/` — per-adapter spectral summaries (24 adapter summaries + pair alignment files + adapter profiles + w0 properties).
- `data/merges/merge_eval_summary.json` — 69-pair merge + evaluate outputs.
- `data/figures/` — six additional figures generated by the analysis scripts (h1_scatter, h1_bootstrap, h1_replications, secondary_*).

**Not in `data/` (pod-only, 1.2 GB):** per-adapter `.npz` SVD factor sidecars. These are required by `scripts/08_compare_methods.py` and are the reason Phase 5 is not reproducible from committed state alone. Documented in `03_reproducibility_check.md` §Tier 4.

---

## Analysis and figure scripts

`scripts/` → `scripts/n134/`. Entire script directory as a single link. Key files:

- `scripts/00_pilot_train.py` — Phase 0 pilot training for 8 tasks × seed 42.
- `scripts/01_pilot_gate.py` — Phase 0 gate decision with retry ladder.
- `scripts/02_train_adapters.py` — Phase 1 full training (seeds 123, 456 + copy pilot s42).
- `scripts/03_spectral_audit.py` — Phase 2 audit (QR-based rank-r fast SVD; v2.1 schema with U/V factors).
- `scripts/04_sample_pairs.py` — Phase 3a deterministic 69-pair sampling.
- `scripts/05_merge_eval.py` — Phase 3b merge and evaluate 69 pairs.
- `scripts/06_analysis_h1.py` — Phase 4 primary H1 analysis.
- `scripts/07_analysis_secondary.py` — Phase 4b exploratory analysis.
- `scripts/08_compare_methods.py` — Phase 5 four-method comparison with `LazyPairV21` synthesizer.
- `scripts/figures/` — three paper-figure generation scripts (`fig_h1_decision.py`, `fig_four_method_forest.py`, `fig_layer_depth_trend.py`) plus shared `mpl_style.py` and a regeneration README.
- `scripts/verify_v21_minimal.py` — single-layer v2.1 schema verification.
- `scripts/test_pilot_gate.py` — 14 unit tests for the pilot-gate retry ladder.
- `scripts/test_analysis_h1.py` — 5 dry-run scenarios for the H1 decision rule.

**Reproduction note.** All three analysis scripts (06/07/08) default to `WORKSPACE = /workspace/n134` (the pod path that produced committed artifacts). Override via the `N134_WORKSPACE` environment variable:

```sh
N134_WORKSPACE="$(pwd)/sidecar/results/n134" python scripts/n134/06_analysis_h1.py
```

See `03_reproducibility_check.md` §Script-hygiene for the full replicator procedure and expected drift magnitudes.

---

## Status summary

- **Pre-registration:** complete (`00_spec.md` v3.1, committed `643c192`).
- **Data collection:** complete (Mistral-7B-v0.3, 24 adapters, 276 pairs audited, 69 pairs merge-evaluated; pod decommissioned).
- **Analysis:** complete (Phase 4 H1, Phase 4b secondary, Phase 5 comparison).
- **Report:** finalized at v1.0.1 (`n134-report-v1` tag → `a13c3c3`).
- **Consolidation:** complete (`n134-submission-draft-v1` tag → `f462767`).
- **Reproducibility check:** complete (four-tier protocol, all tiers pass under amended rank-on-residuals tolerance; see `03_reproducibility_check.md`).
- **Paper draft:** v1 skeleton frozen; v2 Thesis B outline filed; prose drafting not yet begun.

---

*Bundle assembled 2026-04-22. Entries are symlinks to canonical locations, not copies; editing through a link edits the underlying file. If a symlink appears broken after a repo clone on Windows, run `git config core.symlinks true` and re-checkout. Linux and macOS replicators see symlinks resolved transparently.*
