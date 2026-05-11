# ANON Audit Checklist — Benchmark Reliability Study Manuscript

**Status:** internal (do NOT include in the OpenReview upload). Persistent audit trail of the anonymization pass on the bench-reliability submission source.

**Scope:** the manuscript source that ships in the OpenReview tarball:
- `manuscript/draft_v1.tex`
- `manuscript/references.bib`

The compiled PDF that ships alongside the source is audited at the metadata + stream level (Phase 1.3.3 below).

**Convention:** identifying-content leaks are stripped, not paraphrased. Camera-ready restoration is from git history at the pre-strip commits + a deliberate de-anonymization pass.

---

## Phase 1.3.1 — Paper-wide identifying-content grep (2026-05-11)

### 1.3.1a. Author / affiliation grep

Pattern: `Cocchieri|Nanney|gradience|johntnanney|Anthropic|Stanford|MIT|Google`
on `manuscript/draft_v1.tex` and `manuscript/references.bib`.

| Hit | Location | Disposition |
|---|---|---|
| `Mitchell, Melanie` (substring "MIT") | `references.bib` lines 161, 186 | False positive (third-party author cite). KEEP. |
| `Mitigation`/`Mitigating`/`committed` (substring "MIT") | various lines | False positive (word-boundary). KEEP. |
| `MIT-IBM Watson AI Lab` | `references.bib` line 518 | False positive (third-party affiliation in cited work's note). KEEP. |
| "Stanford-anchored thread" + "Stanford" naming Reuel/Koyejo/Domingue | `draft_v1.tex` §1.2 register paragraph | KEEP. Third-party affiliations of cited authors; substantive content of the register paragraph. |

No author/affiliation leak attributable to the present authors. ✓

### 1.3.1b. Project-number grep

Pattern: `N13[0-9]|tier_1_5|prereg_v1_1|Thesis [AB]` on both files.

| Hit | Location | Disposition |
|---|---|---|
| `% Cite-key correctness verified at draft-creation time (2026-04-26)... reuel2025measuring entry that appeared in the N134 paper's bib at Tier 1.5 staging; ... documented in manuscript_outline_v0.md citation-staging table.` | `references.bib` header comment block, ~lines 11–15 of pre-strip state | **STRIPPED 2026-05-11.** Replaced with: "Cite-key correctness verified at draft-creation time via arXiv API queries." Drops project identifier "N134", staging-pass reference, and internal-project doc reference. |
| `% First author verified 2026-04-26 via arXiv API as Andrew M. Bean (Oxford); 42 authors total. Replaces the misattributed "reuel2025measuring" key from the N134 paper's earlier bib staging.` | `references.bib` bean2025measuring pre-comment, ~lines 259–261 of pre-strip state | **STRIPPED 2026-05-11.** Kept: "First author verified via arXiv API as Andrew M. Bean (Oxford); 42 authors total." Drops second sentence with project identifier. |
| `% Cross-paper anchor: the precursor (N134 / Thesis A)` + `% ANON: anonymized for review per venue policy; restore at camera-ready. % The cross-paper "second worked demonstration" framing references the precursor paper without naming it explicitly during review.` | `references.bib` divider block ~lines 712–718 of pre-strip state | **STRIPPED 2026-05-11.** Replaced with: "Anonymized cross-paper self-citation (precursor paper)". The `@misc{anonymized2026n134, ...}` entry itself is KEPT (legitimate anonymized self-cite). |

Post-strip recheck: no project-identifier leaks. ✓

### 1.3.1c. Date-stamped author signatures

Pattern: `20[0-9][0-9]-[0-9][0-9]-[0-9][0-9]` filtered by `edit|author`.

Result: 0 hits. ✓

### 1.3.1d. Email patterns

Pattern: `[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}`.

Result: 0 hits. ✓

### 1.3.1e. EDIT/ANON comment lines

Pattern: `^% EDIT:|^% ANON:|^[[:space:]]*% EDIT:|^[[:space:]]*% ANON:`.

| Hit | Location | Disposition |
|---|---|---|
| `% ANON: author block anonymized for review; restore at camera-ready` | `draft_v1.tex` line 27 of pre-strip state, immediately before `\author{Anonymous Author(s)...}` | **STRIPPED 2026-05-11.** Same class as the EDIT/ANON sweep on N134's `draft_v2_thesis_b.tex` from earlier; author-process metadata with no submission purpose. |

Post-strip recheck: 0 EDIT/ANON comment lines remain. ✓

---

## Phase 1.3.2 — Citation key audit (2026-05-11)

Pattern: `^@\w+\{(\w+)` filtered by `n13|cocchieri|nanney|gradience|tier_1_5|prereg_v1`.

| Key | Disposition |
|---|---|
| `anonymized2026n134` | KEEP. Legitimate anonymized self-cite per convention; the key encodes the anonymization fact, not the project identifier in a leak-prone way (the inverse — "anonymized2026" makes the anonymization explicit; the "n134" suffix is a self-cite identifier the reviewer cannot resolve without the de-anonymized record). |

No keys flagged for rename. ✓

---

## Phase 1.3.3 — PDF metadata audit (2026-05-11)

Audit via `strings` on the compiled `draft_v1.pdf` (pdfinfo + exiftool not installed locally; `strings`-based check covers metadata-stream fields directly).

| Field | Value | Disposition |
|---|---|---|
| `/Producer` | `pdfTeX-1.40.29` | OK. Generic TeX engine identifier. |
| `/Author` | (empty) | OK. Anonymized. |
| `/Title` | (empty) | OK. Title not leaked into metadata. |
| `/Subject` | (empty) | OK. |
| `/Creator` | `LaTeX with hyperref` | OK. Generic build-tool identifier. |
| `/Keywords` | (empty) | OK. |
| `/CreationDate` / `/ModDate` | compile-time timestamps | OK. Not identifying. |

PDF-stream grep for `cocchieri|nanney|gradience|johntnanney`: 0 matches. ✓

---

## Phase 1.3.4 — Final state

- `draft_v1.tex`: 1 ANON comment line stripped (line 27).
- `references.bib`: 3 comment regions edited (header block, bean2025measuring pre-comment, cross-paper-anchor divider). All `@entry` bibtex entries unchanged.
- Compile state: 54 pages, 0 citation undefined, 0 reference undefined, 0 errors (4 pre-existing "empty journal" bibtex warnings on `jo2025evalinference`, `karmakar2026singleprompt`, `lewis2024counterfactual` unchanged; deferred to editorial pass).
- PDF metadata: clean.

Anonymity audit pass complete. Source-shipped state ready for tarball assembly (Phase 1.6).

---

## For camera-ready de-anonymization

When the paper is accepted:

1. `draft_v1.tex` line 27 region: restore `\author{<real authors>\\ <real affiliations>}` block; the manuscript currently ships `Anonymous Author(s) \\ Anonymous Affiliation`. (The pre-strip ANON-marker comment is not restored; commit-history is the audit trail.)
2. `references.bib` header block: optionally restore the staging-provenance comment paragraph if the author wants the cite-staging history visible. (Not recommended; the provenance lives in commit history and `RESEARCH_INVENTORY.md`.)
3. `references.bib` bean2025measuring pre-comment: optionally restore the misattribution-correction record. (Same recommendation: keep stripped; the audit trail lives in commit history.)
4. `references.bib` `@misc{anonymized2026n134, ...}` entry: replace with the proper bib entry for the precursor paper once N134's OpenReview record exists. Update the citation key from `anonymized2026n134` to the real author-year-shorttitle convention; update all `\citet{anonymized2026n134}` and `\citep{anonymized2026n134}` invocations in `draft_v1.tex` accordingly.
