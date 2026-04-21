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

## RN-007 — Panahi attribution remains unverified; ICLR 2026 confirmed

**Trigger:** Pre-T2 closing pass (2026-04-20).
**Issue:** Second verification attempt on the `Panahi et al.` attribution for OpenReview FSDxP3ZpAx. Direct PDF URL (`/pdf?id=FSDxP3ZpAx` and `/pdf/aebdf741910422ddd41feaadc40a93d5a378d1d4.pdf`) returned HTTP 403 to the reviewing agent. Title-based web search confirmed the paper was **accepted to ICLR 2026** (previously unknown; search result titled "Published as a conference paper at ICLR 2026"). Author list remains unconfirmed.
**Resolution:** Bib entry TODO updated with the ICLR 2026 acceptance. Attribution-softening in `docs/positioning/gradience_differentiation.md`, `docs/THEORY.md`, and `docs/technical-report.md` stays as-is. Before paper submission, either (a) verify via direct OpenReview login, (b) verify via ICLR 2026 proceedings page once proceedings URLs are indexed, or (c) keep the softened citation.
**Commit:** `[pending]`.

## RN-008 — Provenance of "Cocchieri" misattribution: unknown

**Trigger:** User request at checkpoint 2 (2026-04-20).
**Issue:** The `Cocchieri et al.` misattribution for arXiv:2511.21437 propagated through four documents (N134 spec, N134 report, THEORY.md, positioning doc). User asked whether the attribution had a discoverable source so we can identify a class of error and guard against recurrence.
**Resolution:** **Provenance unknown.** The author searched the repository's `.claude/worktrees/` history and the archived `n134_spec_v3_update.md` (where "Cocchieri" first appears in the committed history) and found no earlier document, note, or agent output that would explain where "Cocchieri" came from. The most likely hypothesis is an LLM-assisted literature-review turn that produced the attribution and was not verified against the arXiv primary source before being adopted into positioning material; this is a hypothesis, not a finding. What we do know: the first committed appearance is in `sidecar/notes/archive/n134_spec_v3_update.md` (the spec update memo written shortly before N134 pre-registration v3.1). No earlier file in git history contains the string "Cocchieri."

Operational implication for future attribution audits: treat attributions introduced by LLM-assisted literature review as unverified until checked against primary sources (arXiv, venue proceedings, author page). A class-level defense would be: when adding any new citation to `docs/`, `sidecar/notes/`, or `papers/` prose, grep the repo for the author-year string; if it appears only in LLM-conversation artifacts and not in a committed reference document (bib, primary-source URL), treat as unverified.
**Commit:** `[pending]`.

## RN-009 — Tag convention going forward: append-only on future corrections

**Trigger:** User preference at checkpoint 2 (2026-04-20).
**Issue:** `n134-report-v1` was moved forward from its original commit (`3ba4270`) to the attribution-corrections commit (`a13c3c3`) rather than frozen in place with a new tag `n134-report-v1.0.1` pointing at the correction. User flagged the tradeoff and expressed a mild preference for the append-only pattern (tags as historical pointers, not mutable labels) for future corrections.
**Resolution:** Accept. The current forwarded tag (`n134-report-v1` → `a13c3c3`) is retained because forwarding was already pushed and there are no known downstream consumers of the original tag. Going forward: if further corrections to the report are needed, **create a new tag rather than moving the existing one**. Candidate convention: `n134-report-v{major}.{minor}.{patch}` where `patch` increments for attribution / typo / presentation fixes that do not alter findings. The original (pre-correction) commit remains reachable via `git log` and CHANGELOG cross-references even without a dedicated tag.

Documented here as a repo-wide convention so that subsequent consolidation passes on other studies inherit the pattern.
**Commit:** `[pending]`.

## RN-011 — Partial-ρ precision observation → paper language amendment

**Trigger:** T6 reproducibility check (2026-04-21).
**Issue:** Reproduction of `06_analysis_h1.py` in a dev environment (Python 3.14, numpy 2.4, scipy 1.17) against committed audit data produced partial Spearman ρ = −0.5448 versus committed pod-environment ρ = −0.5330, a drift of 1.18e-2. The diagnostic pattern is clean: raw Spearman ρ is bit-identical across environments; OLS fits (R² base, R² full, ΔR²) are bit-identical; only the Spearman-on-OLS-residuals drifts. The drift is a property of rank-based statistics on small-N residuals (n = 45): floating-point perturbations too small to shift aggregate quantities like sum-of-squares can still flip the rank order of near-tied residual pairs, which Spearman is sensitive to. This is not a bug in any library or an environment issue beyond what is expected at this version gap; it is an intrinsic precision property of the statistic class on this data at this observation count.
**Resolution:** Paper reproducibility-statement prose should name this observation explicitly, with language along the lines of:

> We note that partial Spearman ρ computed on 45 OLS residuals is sensitive to floating-point paths in the residualization step: reproduction in a different numerical environment produced ρ = −0.545 versus committed ρ = −0.533, with all other quantities (raw ρ, R², ΔR², bootstrap statistics) reproducing to within 10⁻³ or bit-identical. This sensitivity is a property of rank-based statistics on small-N residuals rather than a bug in any implementation; the headline decision (H1 not confirmed under the pre-registered rule) is robust to this precision, but the specific value of partial ρ should be understood as approximately −0.53 ± 0.01 rather than as a four-decimal point estimate. This observation is itself an instance of the measurement-discipline concerns that motivate the Gradience program more broadly.

When paper revision reaches the methods / reproducibility section, pull the full numerical detail from `sidecar/notes/n134_reproducibility_check.md` §Rank-on-residuals observation and the precision-vs-sampling-CI comparison from that document's closing paragraph.
**Commit:** `[pending]`.

## RN-010 — F2 dropped as a plot after equivalence check

**Trigger:** T2 pre-figure-work equivalence check (2026-04-20).
**Issue:** F2 was originally planned as a cross-architecture same/cross alignment comparison covering DistilBERT (N130), DeBERTa (N132), Mistral-7B N133, Mistral-7B N134. Data-availability recon revealed that N130 does not persist per-pair alignment records at the granularity needed for distributional plotting. A subsequent metric-equivalence check on the three studies that do have per-pair data revealed that **N132 and N133/N134 use different metric families**:

- N132 (inherited from N07, `scripts/n07_deberta/experiment_a_per_module.py:68`): subspace principal-angle metric, `mean(cos²(θ_i))` where `θ_i` are principal angles between U subspaces. No singular-value weighting.
- N133, N134 (`scripts/n133_spectral_audit.py:81` and `scripts/n134/03_spectral_audit.py:114`, character-identical): SV-weighted mean absolute cosine, `sum_ij σ_i σ_j |<u_i, u_j>| / (Σσ_a · Σσ_b)`.

These are not rescalings of each other; they are different computations on different subsets of the spectrum, and the raw distributions are not directly comparable.

**Resolution:** F2 dropped as a plot. The same/cross ratio comparison is preserved in §5 of the paper as prose naming the four studies, the two metric families, and the common direction (same > cross); the robustness-across-metric-families is itself part of the claim rather than a caveat. A bar-chart-of-ratios alternative was considered and rejected — when a visual requires a multi-sentence caption to justify the cross-study comparison, the visual is doing less work than the caption. The prose version makes the modest claim cleanly without the reviewer-objection asymmetry a mixed-metric bar chart would invite. See `figure_captions.md` for a §5 paragraph candidate. Skeleton adjusted: `draft_v1.md` figure plan reduced from four to three figures; §5 word-count budget increased by ~0.1 page to accommodate the named-studies sentence. Paper now targets three figures (F1, F3, F4), each carrying analytical weight for a distinct claim rather than a fourth-for-the-sake-of-four.
**Commit:** `[pending]`.

## RN-011 — [Title]

**Trigger:** [self-review | reader X | reviewer Y]
**Issue:** [what was identified]
**Resolution:** [what was changed]
**Commit:** [hash of the commit that implements the resolution]
