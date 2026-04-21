# Changelog

## v1.0.1 — 2026-04-20 (attribution corrections; findings unchanged)

Post-T4 correction pass. All changes are attribution / citation metadata only; no N134 finding, number, figure, or deviation changes. The `n134-report-v1` tag is moved forward to the post-correction commit; the commit range leading to v1.0.1 is traceable via `git log`.

- RN-001 (Zhou et al. r = 0.572): citation resolved via `sidecar/notes/n134_spec.md` trace. Paper is Zhou et al. 2026, arXiv:2601.22285 ("Demystifying Mergeability..."). Bib entry added.
- RN-002: retracted. My initial MergeBench-misattribution claim was based on a misread of positioning material.
- RN-003 (Panahi attribution): softened to citation-by-title across `docs/positioning/gradience_differentiation.md`, `docs/THEORY.md`, `docs/technical-report.md`. Attribution flagged as unverified pending OpenReview access.
- RN-004 (SVC citation): resolved via arXiv fetch. Paper is Li et al. 2026, arXiv:2602.05536 ("When Shared Knowledge Hurts..."). Bib entry updated with full author list.
- RN-005: SVC code-audit fallback documented but not triggered (RN-004 made it unnecessary).
- RN-006: **"Cocchieri et al." → "Hitit et al."** attribution correction. The paper cited throughout Gradience materials (arXiv:2511.21437) is actually by Hitit, Girrbach, Akata. Corrected in `sidecar/notes/n134_spec.md`, `sidecar/notes/n134_report.md`, `docs/THEORY.md`, `docs/positioning/gradience_differentiation.md`. Archive (`sidecar/notes/archive/n134_spec_v3_update.md`) left as-written.

## v1 — 2026-04-20

Initial skeleton committed. `draft_v1.md` contains structure, intended section lengths, and figure plan only — no paper prose. See RN-000 for the reasoning. Corresponds to report tag `n134-report-v1` (pre-correction) on commit `3ba4270`.

**Target-venue word-count sanity check (2026-04-20):** the skeleton's stipulated section lengths (Abstract 150w; Intro 1p; Methods 1p; H1 1p; Replications 0.5p + fig; Comparison 0.75p + fig; Discussion 1p + fig; Conclusion 0.25p) sum to approximately 4,100 words plus 4 figures — roughly 6.8 pages at the NeurIPS-style ~600 words/page rate. This is in range for a short-paper workshop (4–6 pages) but tight for a full 8-page workshop. Revision-time may surface real word counts materially above or below this target; if so, recalibrate the section allocation before writing rather than after. Flag for the first revision session.
