# N134 Workshop Paper

## Files

- `draft_v1.md` — frozen v1 skeleton (findings-paper framing; structure, section plan, figure plan; no paper prose). Do not edit.
- `draft_v2_thesis_b_outline.md` — **current revision-planning outline.** Repositions the paper from findings-paper (v1) to *Measurement Discipline for ML Diagnostics*, with the framework as primary contribution and N134 as worked example. See CHANGELOG `v2-outline` entry and `revision_notes.md` RN-012 for the scope decision.
- `draft_current.md` — working revision file. Currently still mirrors `draft_v1.md`; will become the Thesis B prose draft once prose drafting begins against the v2 outline.
- `figures/` — generated figures; regenerate via `scripts/n134/figures/*.py`.
- `references.bib` — bibliography.
- `figure_captions.md` — caption staging area.
- `revision_notes.md` — revision log.
- `CHANGELOG.md` — version history.

## Source material

`sidecar/notes/n134_report.md` (tag `n134-report-v1`) is the canonical internal findings document. All numbers, tables, and interpretive framing in the paper draw from it. The report is ~350 lines with full appendices; the paper target is 4–8 pages. The revision process transforms report-prose into paper-prose.

## Regenerating figures

From repository root, after `scripts/n134/figures/` is populated (T2 in the consolidation spec):

    python scripts/n134/figures/fig_h1_decision.py
    python scripts/n134/figures/fig_three_arch_comparison.py
    python scripts/n134/figures/fig_four_method_forest.py
    python scripts/n134/figures/fig_layer_depth_trend.py

Each script reads from `sidecar/results/n134/` and writes to `papers/n134_workshop/figures/`.

## Compiling to submission format

To be specified after venue is selected and format requirements are known.

## Target venue

TBD; candidates: NeurIPS ENLSP workshop, ML for Systems workshop, ICLR workshop track.
