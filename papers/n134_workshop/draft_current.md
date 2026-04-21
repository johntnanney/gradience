# N134 Workshop Paper — draft v1 (skeleton)

**Status:** skeleton, not prose. No paper draft exists at time of this file's creation.

**Source material:** `sidecar/notes/n134_report.md` (tag `n134-report-v1`) is the canonical internal findings document and the source of all numbers, tables, and interpretive framing that this paper will draw from. The report is ~350 lines including appendices; a workshop paper is 4–8 pages. The revision process transforms report-prose into paper-prose; this skeleton file is a placeholder marking the paper's frozen-v1 state, not the actual v1 content.

**Convention:** per the consolidation spec, this file is frozen on first commit. All revision work happens in `draft_current.md`, which initializes as a copy of this skeleton. When the first real paper prose exists in `draft_current.md`, that content should be copied into this file at the appropriate revision milestone (e.g., "committed draft v1 ready for outside reader," "committed draft v1 ready for submission"), at which point this file becomes the frozen reference it was always intended to be.

---

## Intended structure (to be filled during revision)

1. **Abstract** — null result + three replicated findings + regime-null framing. Target 150 words.

2. **Introduction** — Gradience's spectral-triage program, N127 → N133 → N134 arc, pre-registration as methodological anchor. Target 1 page.

3. **Methods** — abbreviated from `sidecar/notes/n134_report.md` §2. Model, adapters, task set, FAMILY_B partition, audit schema, pair sampling, merge-and-eval protocol, statistical protocol. Target 1 page.

4. **H1 primary result** — drawn from report §3. Table of the pre-registered decision criteria and observed values; short prose on the null outcome and why the wrong-signed significant partial ρ constitutes a null rather than a reversed confirmation. Target 1 page + Figure 1.

5. **Confirmatory replications** — B-P1/B-P2/B-P4 from report §4; four-study comparison across two metric families (DistilBERT N130, DeBERTa N132, Mistral-7B N133, Mistral-7B N134); emphasis on task-boundary detection as the Gradience program's most robust architecture-general finding. The same/cross alignment ratio (5×, 2.3×, 3.06×, 2.28×) is cited in prose rather than plotted because N130/N132 and N133/N134 use different metric families (subspace principal-angle vs. SV-weighted cosine); see RN-010. Target 0.6 page (no figure).

6. **Four-method comparison** — drawn from report §6. Table of four methods with Spearman ρ, p, retained-set mean degradation, bootstrap CI. Prose emphasizing the regime null (no method is significant; 88% of variance is family-identity). Target 0.75 page + Figure 2.

7. **Discussion** — condensed from report §7. Three-architecture replication as the consolidation; per-pair risk regression outside current measurement envelope; four candidate follow-up programs (larger N, activation-informed, intrinsic-mergeability, training-time intervention); brief note on the wrong-signed ρ as hypothesis-generating residual. Target 1 page + Figure 3.

8. **Conclusion** — null decisive; epistemology held; follow-up directions. Target 0.25 page.

9. **References** — bibliography from `references.bib`; see `papers/n134_workshop/references.bib`.

Figure plan. Paper has three figures, numbered 1/2/3 by `\begin{figure}` ordering in LaTeX. Script files and PDF/PNG output filenames are descriptive (no numeric prefix), so a later drop or reorder of figures does not force a rename.

| Paper Fig | Claim | Script | Output basename |
|---|---|---|---|
| 1 | H1 decision (primary null) | `fig_h1_decision.py` | `h1_decision` |
| 2 | Four-method comparison | `fig_four_method_forest.py` | `four_method_forest` |
| 3 | Layer-depth trend (geometry) | `fig_layer_depth_trend.py` | `layer_depth_trend` |

(A fourth figure — cross-architecture same/cross distribution comparison — was considered and dropped. N130 does not persist per-pair alignment records at the granularity needed for plotting, and the three remaining studies use two different metric families. See RN-010. The cross-architecture claim is made in §5 prose instead.)

---

## Target venue

TBD. Candidates: NeurIPS ENLSP workshop, ML for Systems workshop, ICLR workshop track. Selection drives format requirements (page count, template, anonymization rules, bibliography style) — none of which are locked at time of this file's creation.

---

## Do not modify this file after first commit

All revision happens in `draft_current.md`. This skeleton is the frozen v1 reference per the consolidation spec. If subsequent revision produces meaningful paper prose, copy that prose into this file only as part of an explicit versioning milestone (recorded in `CHANGELOG.md`), not as an ongoing editing process.
