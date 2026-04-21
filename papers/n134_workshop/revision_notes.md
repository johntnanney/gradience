# Revision Notes for N134 Paper

Each entry records a specific revision decision, its trigger (self-review, outside reader, reviewer comment), and its resolution. Entries are numbered sequentially and referenced from commit messages.

---

## RN-000 — draft_v1 is a skeleton, not prose

**Trigger:** T3 consolidation (2026-04-20).
**Issue:** The consolidation spec's T3 source assumed a paper draft existed at `/mnt/user-data/outputs/n134_paper_draft_v1.md` or similar. No such draft was produced. The only prose asset is `sidecar/notes/n134_report.md`, which is an internal findings report with full appendices (~350 lines), not a workshop paper.
**Resolution:** `draft_v1.md` committed as a structure-only skeleton that (a) points at the report as canonical source, (b) lists the intended section structure and target lengths, and (c) specifies the figure plan. `draft_current.md` initialized as an identical copy. The actual paper prose is the revision work that remains to be done.
**Commit:** `d07ff84`.

## RN-001 — "Zhou et al. activation-dot-product (r = 0.572)" — RESOLVED

**Trigger:** T4 bibliography consolidation (2026-04-20).
**Issue (original):** The N134 report (§7.2, §7.3) references "Zhou et al.'s activation-dot-product result (r = 0.572)" as motivation for the activation-informed N135-alt follow-up. Initial web search on 2026-04-20 did not locate a distinct paper matching this description; the closest hit was Zhang & Zhou ACL 2025 OSRM (`zhang2025osrm`), which is a different paper.
**Resolution:** Trace through `sidecar/notes/n134_spec.md` line 23 identified the paper explicitly as Zhou et al. 2026, arXiv:2601.22285, from the GLADIA/Sapienza group (same as TSV). An arXiv fetch on 2026-04-20 confirmed the author list (Luca Zhou, Bo Zhao, Rose Yu, Emanuele Rodolà) and the paper's topic (190 task pairs, ViT-B/32 vision classifiers, interpretable pairwise metrics). The r = 0.572 statistic is body-of-paper (not abstract); accepting the spec's report of it as accurate. Bib entry `zhou2026demystifying` added. The report's citation is correct; no report amendment required.
**Commit:** `[pending]` (this commit).

## RN-002 — RETRACTED

**Trigger:** T4 bibliography consolidation (2026-04-20).
**Issue (retracted):** I initially claimed that prior Gradience positioning material attributed MergeBench to "Cocchieri et al." On re-reading `docs/positioning/gradience_differentiation.md` line 16 ("MergeBench **and** Cocchieri et al. 2026 cover more ground") and cross-checking `sidecar/notes/n134_spec.md` line 25, the positioning material has always treated MergeBench (He et al. 2025) and Cocchieri et al. (arXiv:2511.21437) as **distinct** papers. The original RN-002 claim that MergeBench was misattributed in Gradience positioning was wrong.
**Resolution:** Retracted. See RN-006 below for the actual attribution error that was surfaced during the Cocchieri investigation.
**Commit:** `[pending]`.

## RN-003 — "Panahi et al. 2026" attribution softened

**Trigger:** T4 bibliography consolidation (2026-04-20).
**Issue:** Gradience positioning material (`docs/positioning/gradience_differentiation.md`, `docs/THEORY.md`, `docs/technical-report.md`) referenced "Panahi et al. 2026" as an example of rigorous formal work on LoRA merging theory. An April-2026 OpenReview paper ("LoRA Provably Reduces Forgetting and Enables Adapter Merging in Multiclass Linear Classification," OpenReview `FSDxP3ZpAx`) matches the described character, but the first author's surname was not confirmed as "Panahi." An OpenReview page fetch on 2026-04-20 returned HTTP 403 to the reviewing agent.
**Resolution:** In-text citations softened from "Panahi et al. 2026" to citation-by-title (paper title plus OpenReview id, with an explicit note that the author attribution was not verified). Done in `docs/positioning/gradience_differentiation.md`, `docs/THEORY.md` (two instances), and `docs/technical-report.md`. A TODO entry is retained at the bottom of `references.bib` for verification via direct OpenReview login.
**Commit:** `[pending]`.

## RN-004 — SVC citation RESOLVED

**Trigger:** T4 bibliography consolidation (2026-04-20).
**Issue:** `li2026svc` was initially kept in `references.bib` with a TODO marker because the spec template's arXiv handle (2602.05536) had not been independently verified.
**Resolution:** arXiv fetch on 2026-04-20 confirmed the paper: Li, Peng, Zhang, Guo, Duan, Shi 2026, "When Shared Knowledge Hurts: Spectral Over-Accumulation in Model Merging," arXiv:2602.05536. Method is "Singular Value Calibration (SVC)," matching what `scripts/n134/08_compare_methods.py` implements as a pairwise-triage adaptation. Bib entry updated with verified author list and note confirming the match. RN-005's code-audit fallback is therefore not needed, but RN-005 is kept on file as documentation of the resolution path.
**Commit:** `[pending]`.

## RN-005 — Unresolved-SVC-citation fallback (documentary; not triggered)

**Trigger:** T4 bibliography consolidation (2026-04-20).
**Issue (hypothetical):** If `li2026svc` had remained unresolvable at paper-revision time, the paper would have needed to decide whether SVC belongs in the method comparison at all. The resolution path would have been: audit `scripts/n134/08_compare_methods.py:svc_score_from_v21` for whether the implementation is a recognizable SV-rescaling operation against shared-subspace directions. If yes, cite as "an adaptation of the singular-value calibration approach (reference unresolved; see code for specification)." If no, rename the method in the paper to something descriptive of what the code actually does, and note that a plausible SV-rescaling baseline was chosen as the fourth comparator.
**Resolution:** Not triggered; RN-004 resolved the citation. This entry is kept as documentation of the fallback reasoning for future situations where a cited method's paper cannot be verified.
**Commit:** `[pending]`.

## RN-006 — "Cocchieri et al." → "Hitit et al." attribution correction

**Trigger:** T4 bibliography consolidation + Decision 4 trace (2026-04-20).
**Issue:** While investigating Decision 4 (tracing the Zhou et al. citation), arXiv fetch on arXiv:2511.21437 revealed that the paper cited as "Cocchieri et al. 2026" throughout Gradience documentation is actually **Hitit, Girrbach, Akata 2026**, "A Systematic Study of In-the-Wild Model Merging for Large Language Models," TMLR March 2026. The paper contents match exactly what the spec describes (six merge methods × four open-weight LLMs, Task Arithmetic as the only reliably constructive method). The "Cocchieri" name was a misattribution that propagated through `sidecar/notes/n134_spec.md`, `sidecar/notes/n134_report.md` §9, `docs/THEORY.md` (four instances), and `docs/positioning/gradience_differentiation.md` (two instances).
**Resolution:** All four Gradience-maintained documents corrected in this commit: `Cocchieri et al.` → `Hitit et al.` The change does not affect any N134 finding, analysis, or figure; it is purely an attribution correction. `sidecar/notes/archive/n134_spec_v3_update.md` was left unmodified because it is in the `archive/` subdirectory (historical record preserved as-written). Bib entry `hitit2026insystematic` added.

Per the consolidation spec's T5 discussion, this correction triggers re-tagging of `n134-report-v1` to the corrected commit.
**Commit:** `[pending]`.

## RN-007 — [Title]

**Trigger:** [self-review | reader X | reviewer Y]
**Issue:** [what was identified]
**Resolution:** [what was changed]
**Commit:** [hash of the commit that implements the resolution]
