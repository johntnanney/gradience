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

## RN-012 — Paper repositioned to Thesis B; v2 outline filed

**Trigger:** Post-consolidation planning (2026-04-22).
**Issue:** The v1 skeleton (`draft_v1.md`) framed the paper as a findings paper centered on the N134 empirical results. Post-consolidation review concluded that the paper's most durable contribution is methodological — the measurement-discipline framework that the consolidation pass surfaced and exercised — rather than the N134 findings in isolation. The findings paper target (4–6 page workshop) also under-serves the framework contribution, which requires framing, theoretical development, and worked example in sequence.
**Resolution:** Paper repositioned to *Thesis B* working title *"Measurement Discipline for ML Diagnostics: A Psychometric Framework with a LoRA-Merging Case Study"*. V2 outline filed at `papers/n134_workshop/draft_v2_thesis_b_outline.md`. Three contribution claims (framework, worked example, previously-unnamed measurement property); 8–10 page target; position-paper / methods-paper venues (ICML Position track, NeurIPS Reproducibility workshop, Datasets & Benchmarks, ICLR blog). N134 empirical content preserved but subordinated to framework demonstration. The rank-on-residuals observation from RN-011 becomes its own section (§6) as the cleanest worked example of the framework producing findings that unstructured reporting would not surface. Scope decision preserved: standalone paper, N134 material only, submission before N135-alt runs (folding N135-alt would expand §5 to 12–14 pages total and change the venue options). `draft_v1.md` retained frozen per consolidation convention; `draft_current.md` not yet updated (outline is planning, not prose).
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

## RN-013 — Thesis memo filed; §2 restructuring indicated

**Trigger:** Pre-§2-revision diagnostic pass (2026-04-22).
**Issue:** The v2 Thesis B outline (RN-012) and the first-pass LaTeX draft present §2's four framework components (construct validity, reliability, bounded precision, confound decomposition) as four parallel subsections, each with its own rationale. External-reader review flagged this as under-powered for the position-paper genre: the components arrive as a coordinated toolkit rather than as aspects of a conceptual commitment, and §2 does not name the implicit view of measurement the paper is arguing against. The outline's immediate-action 1 specified a one-page thesis memo as a diagnostic before §2 prose revision.

**Resolution:** `thesis_memo.md` filed. The memo commits to three philosophical moves that supersede the outline's §2 framing going forward:

1. **Named opposition.** The paper argues against an implicit "direct-readout" view of measurement — the view that a score is a transparent report of a latent property — in favor of the psychometric tradition's inferential / fallible-indicator view, on which a score is an observable whose relationship to the theoretical object must be structurally defended. The outline did not name the argued-against view; §2 prose should.

2. **Jointly-constitutive framework components.** The four components are not four independent methodological virtues; they are four aspects of a single commitment to treating scores as indicators. Omitting any one tacitly reverts to the direct-readout view at the point of omission. The parallel-subsection structure currently in §§2.1–2.4 is incompatible with this framing; §2 should be rewritten as a continuous argument traversing the four questions an inferential defense must answer.

3. **Dialectical shape of the contributions.** The three contributions are normative (framework), demonstrative (worked example with a null outcome), and productive (rank-on-residuals observation). Removing any one collapses the argument: without 2, the framework lacks evidence of application; without 3, the application lacks epistemic advantage over alternatives; without 1, the isolated finding has no argument for general practice. §1.3 and §9 should track this structure explicitly; §8's objection-handling should answer each objection in a way that preserves the dialectical relationships to the other two contributions.

Diagnostic conclusion: the memo compressed to one page without strain once the three philosophical positions were accepted, which (per the outline's stated test) indicates §2 needs *restructuring* rather than expansion. The §2 rewrite is the next paper-revision work item; it is separated from this commit deliberately to keep the memo-as-blueprint distinct from the prose-revision work that will track it. Two operational consequences recorded here for the §2 rewrite: §2.4's current concrete example (88% family-pair variance) moves to §4 to keep §2 fully general; the parallel-subsection layout is dropped in favor of continuous prose.

**Commit:** `[pending]`.

## RN-014 — §1.1 expanded to two concrete reporting-gap examples

**Trigger:** Post-memo revision pass (2026-04-22).

**Issue:** The first-pass §1.1 stated the reporting-gap pattern abstractly and cited `zhou2026demystifying` and `rahamim2026mergeability` as illustration, but did not develop either into a concrete worked example. The v2 outline specified "two concrete examples from the current ML literature of diagnostic metrics reported without measurement-theoretic context — one from the LoRA-merging subfield, one from an unrelated subfield" so that the cross-subfield convergence carries the weight of the paragraph's final claim. Without two worked examples, the "recurs across subfields" assertion is a promissory note the §1.1 prose does not redeem.

**Resolution:** §1.1 restructured from two paragraphs to four.

1. Para 1 now states the pattern abstractly (what a typical ML diagnostic report is missing) and announces two examples.
2. Para 2 develops Zhou et al. 2026's $r = 0.572$ (TSV, 190 task pairs, ViT-B/32) as the LoRA-merging example. Each of the three measurement-theoretic gaps (reliability, tolerance, confound decomposition) is stated as a specific absence from this specific report. The paragraph explicitly distances the example from ad hominem critique: "not that $0.572$ is wrong" but that the reporting convention treats it as self-standing.
3. Para 3 describes capability-evaluation reporting (MMLU / HellaSwag / BBH to two or three decimals without cross-seed or cross-prompt reliability coefficients, without tolerance schedules, without pre-registered confound controls against prompt-format sensitivity, evaluation-metric choice, or contamination) as the cross-subfield pattern. The paragraph's closing sentence makes the argumentative payload explicit: the same gap in a subfield with no methodological contact with merging research is evidence the gap is not idiosyncratic.
4. Para 4 preserves the original three-gap synthesis as the paragraph's conclusion.

**Outstanding.** The capability-evaluation paragraph is argumentatively self-sufficient as a pattern description but would be strengthened by a specific cited claim. A LaTeX comment in the source names candidate directions (method-paper leaderboard claim reporting MMLU/BBH to two or three decimals; capability-evaluation reliability literature such as Burnell et al. 2023; probing-accuracy or faithfulness-score claim from the interpretability literature). Resolution requires the author's editorial choice of exemplar plus verification of the exact reported value and addition of a bib entry; deferred to revision rather than faked in first pass.

The `rahamim2026mergeability` citation previously in §1.1 is no longer used there; it remains cited in §8 (line 394). No orphaned citations result from this revision.

**Commit:** `[pending]`.

## RN-015 — Cross-seed ICC spec filed ahead of Appendix-D implementation

**Trigger:** Post-§1.1 revision pass (2026-04-22). Appendix D carries the only explicit `[TODO]` in the first-pass .tex (cross-seed ICC computation, flagged in the v2-outline as an early-revision action). Rather than write a script first and document-the-choices-retrospectively, this revision pass produces the spec first — matching the paper's own argument that the measurement-theoretic commitments going into a reliability estimate belong in prose before they go into Python.

**Issue:** The ICC computation carries real design choices that determine what the resulting number means. Form (ICC(1,1) / ICC(2,1) / ICC(3,1)) turns on whether seed-pairs within a task are treated as interchangeable random draws versus fixed raters. Agreement-vs-consistency turns on whether the paper's bounded-precision claims couple to absolute-agreement SEM or to rank-preserving consistency. CI method (Shrout–Fleiss parametric versus block-bootstrap over tasks) answers subtly different questions about where the sampling uncertainty lives. Reporting SEM alongside ICC is not automatic and couples tightly to §2.3's bounded-precision argument. A spec is the place these choices get defended; a script is where they get executed.

**Resolution:** `sidecar/notes/n134_icc_spec.md` committed. Contents: ~9 sections covering (§1) what is estimated and paper destinations, (§2) data inputs with verified audit-directory schema, (§3) four design choices each with a short measurement-theoretic defense — ICC form, CI method, SEM reporting, instrument-output definition, (§4) computation procedure, (§5) JSON output schema, (§6) sanity checks, (§7) failure modes and escalation, (§8) prose templates for Appendix D, §2.2, and §4.2, (§9) promotion-to-convention note. The spec commits to ICC(2,1) absolute-agreement single-measurement with Shrout–Fleiss primary CI, block-bootstrap secondary, and SEM reported to two significant figures.

Implementation target: `scripts/n134/09_analysis_icc.py`, producing `sidecar/results/n134/analysis_icc.json`. The spec is deliberately explicit enough for a coding agent to implement without further design conversation. Execution may be delegated to coding agents or run in-session; the spec is valuable as a standalone artefact regardless.

**Outstanding.** Implementation of the spec (script + JSON output + Appendix-D paragraph insertion + §2.2 / §4.2 in-text insertions) is the follow-on work.

**Commit:** `[pending]`.

## RN-016 — Cross-seed ICC computed; Appendix-D TODO closed

**Trigger:** In-session execution of the RN-015 spec (2026-04-22). The spec was tight enough to execute without further design conversation.

**Issue:** RN-015 filed the spec; `[TODO]` in `app:reliability` of `draft_v2_thesis_b.tex` remained the one explicit placeholder in the first-pass .tex prose. Spec §8 prose templates named the paper destinations (Appendix D, §2.2, §4.2) with `[VAL]` slots to fill.

**Resolution.** `scripts/n134/09_analysis_icc.py` implements the spec: direct Shrout–Fleiss ICC(2,1) absolute-agreement single-measurement on the 8×3 same-task panel (no `pingouin` dependency — `scipy.stats.f` for the F-distribution CI is sufficient and gives full control over the formula); `compute_s_h1` imported from `06_analysis_h1.py` by the spec-mandated `importlib.util` shim (filename starts with a digit); block-bootstrap over tasks (5000 resamples, seed 134) for the secondary CI.

Results, committed to `sidecar/results/n134/analysis_icc.json`:

- ICC(2,1) = **0.566** (conventional descriptor: *moderate*).
- Shrout–Fleiss 95% CI: [0.165, 0.874].
- Block-bootstrap 95% CI: [0.141, 0.965].
- **CI agreement: divergent** (max bound difference 0.091, past the spec's 0.05 threshold). Both intervals reported per spec §3.2 escalation rule. The divergence is itself informative about parametric-assumption strain at N = 8 tasks.
- SEM = **0.014**; SD_pooled = 0.021 on same-task S_H1 values in [0.044, 0.112].
- All four sanity checks pass (panel complete, ICC in range, SEM < SD_pooled, CI agreement explicitly recorded).

Three prose insertions applied to `draft_v2_thesis_b.tex`:

1. **Appendix D** (`app:reliability`): full paragraph replaces the `[TODO]` block, reporting ICC / both CIs / SEM / conventional descriptor, explicitly flagging the CI divergence as informative, and naming one transferability caveat — the reliability design targets same-task pairs (where alignments are in the 0.044–0.112 range); transferring SEM to cross-task-pair precision claims (where $S_{\mathrm{H1}} \in [0.015, 0.025]$) is a broader inferential step this estimate does not license directly. This caveat was not in the spec but is an honest framework-consistency concern worth naming.
2. **§2.2** (`sec:framework-reliability`): one clause added to the existing "N134 design commits to three seeds per task" sentence, naming $\hat{\rho}_{\mathrm{ICC}} = 0.566$, SEM = 0.014, 95% CI [0.165, 0.874], with Appendix D cross-reference.
3. **§4.2** (`Reliability considerations at pre-registration time`): one sentence appended, reporting the resulting estimate with Appendix D reference.

No explicit `[TODO]` or `[TBD]` placeholder remains in the .tex. One `% TODO(...)` LaTeX comment at line 113 is unrelated (the §1.1 second-example citation swap from RN-014, not the ICC TODO).

**Honest state observation.** Cross-seed ICC of 0.566 on a per-pair spectral diagnostic is moderate-by-convention; combined with the rank-on-residuals precision observation in §6, the paper now reports two independent measurement-property findings on $S_{\mathrm{H1}}$: cross-seed reliability is moderate and rank-on-residuals precision is $\pm 0.01$ at $n = 45$. These are not redundant; they couple to different terms of the paper's bounded-precision argument. The framework's productive-rather-than-descriptive claim (§6.4 in the current draft) now has two grounded instances rather than one.

**Commit:** `[pending]`.

## RN-017 — Revision-length pass: mild trim, three-instances-two-kinds insertion, Option B tightening; 13 pp main

**Trigger:** Post-ICC first compile reported 18 pp total / 14 pp main, too long for the 10--12 pp position/methods venue target. User endorsed venue target as position/methods paper at 10--12 pp; accepted mild-trim package + paragraph-insertion reweave + subsequent Option B tightening.

**Issue:** Paper's contributions structurally require the three-legged dialectic (normative framework + demonstrative worked example + productive unnamed-property discovery) to carry visible weight, so aggressive §5 contraction was ruled out. Needed to recover 2 pp of main text without compromising the case-study weight.

**Resolution, in execution order:**

1. **References.bib hygiene fix.** The header comment block and TODO stub for the Panahi attribution both contained `@article` / `@inproceedings` / `@misc` tokens inside `%` comments. BibTeX's parser treats `@` as structural even inside line comments, so it was silently skipping the first four entries and erroring on the TODO stub. Rewrote the header format-conventions line and the TODO stub to prose-only (no `@` tokens). The canonical repo now compiles cleanly without manual intervention.

2. **Mild-trim package** (from the coding-agent-identified targets):
   - §5.1 tail: removed the N135-alt four-candidate-interpretation digression (\~4 source lines).
   - §5.2 tail: compressed the "readers interested in the informativeness calculation" elaboration (\~2 source lines).
   - §5.3 tail: tightened the three-architecture restraint prose (\~4 source lines).
   - §5.4 closing: tightened the framework-reading interpretation (\~3 source lines).
   - §7.4 psychometric-tradition reference: compressed from 14 lines to 8 (\~6 source lines).
   - §8 objections: merged "null would be less compelling" + "rank-on-residuals niche" into a single `\paragraph{``The case study is weaker than the framework.''}` with two-form structure (\~11 source lines).

3. **Three-instances-two-kinds paragraph insertion at §6.4 close.** New final paragraph of §6.4 articulates that the paper has produced three framework findings (rank-on-residuals, ICC = 0.566, H1 null) which split into two epistemic kinds: *discovery* (the rank-on-residuals precision property, unnamed prior) and *calibration-and-modesty* (the moderate reliability bounds and the informative null). Argues that both kinds are framework products but differ in epistemic shape --- one generates new measurement claims, the other bounds old ones --- and that a framework yielding only one of the two kinds would either overreach or be merely defensive. The two kinds together are what measurement discipline is *for*. ~150 words; paragraph-insertion variant was chosen over structural-recast variant to avoid making the two-kinds distinction load-bearing for §7.1's generalization claim.

4. **Option B second-pass tightening:**
   - §5 preamble: 6 lines → 2 lines (removed procedural metatext redundant against subsection headings).
   - §7.2 + §7.3 merged into single subsection "What this changes, and what it does not" (~7 source lines saved). Argumentative improvement independent of page math: the two previously-separated subsections shared a rhetorical move (not-a-critique-of-prior-work) that reads stronger as a single continuous argument.

**Final state:** Main text §§1--9 spans 13 pp (ends \~81\% down p13, with References beginning on p13). Total PDF 18 pp (13 main + \~2 refs + 4 appendix). User accepted 13 pp main as final; further trimming beyond Option B yields diminishing returns because remaining candidate cuts are either structural (§3-into-§4 merge, figure-to-appendix moves) with framing-cost implications, or contiguous-prose cuts that would compromise the case-study weight the argument depends on.

**Deferred to v3 (not this submission cycle):** (a) §2 rewrite tracking `thesis_memo.md` (four-parallel-subsections → continuous argument tracking the four questions of the inferential view); (b) §1.1 second-example first-order citation (capability-evaluation paragraph remains argumentatively self-sufficient with a `% TODO` comment marking candidate-citation direction); (c) Panahi attribution resolution if OpenReview access succeeds.

**Commit:** `[pending]`.

## RN-018 — A2 double-blind anonymization pass with `% ANON:` markers

**Trigger:** Workstream A2 (2026-04-22). TMLR selected as primary venue per A1/venue_shortlist.md; TMLR requires double-blind submission. The canonical `draft_v2_thesis_b.tex` and `references.bib` contained multiple identity-leaking strings that had to be neutralized before submission — several of them load-bearing enough that the paper would have been trivially de-anonymizable to anyone running `grep` on the source.

**Issue.** Seven classes of leak were catalogued in the Task #20 scan:

1. **Author block.** `\author{Anonymized\thanks{De-anonymized references in the tagged repository; see \texttt{n134-submission-draft-v1}.}}` — the `\thanks{}` note literally named the git tag where the un-anonymized version lived. The "Anonymized" field was technically anonymous but the thanks defeated it.
2. **Internal project numbers in body prose.** Sixteen occurrences of `N07`, `N127`, `N130`, `N132`, `N133`, `N134`, `N135-alt` across §§2, 4, 5, 6, 7, 8 and the appendices. Reviewers googling "N134 merge triage" would have reached the Gradience sidecar.
3. **Project brand leaks.** `Gradience N133` (§4.1), `Gradience $S_{\mathrm{H1}}$` (§5.4).
4. **Companion-repository file paths in appendices.** `sidecar/results/n134/analysis\_icc.json`, `sidecar/notes/n134\_icc\_spec.md`, `08\_compare\_methods.py`, `n134\_reproducibility\_check.md`, `n134\_incident\_log.md`, `n134\_report.md`, plus several "the companion repository" phrases.
5. **Cloud-provider name.** `(RunPod Secure Cloud)` in Appendix C.
6. **Tag reference.** `\texttt{n134-submission-draft-v1}` — the git tag where the un-anonymized canonical state lived. This was the single most identifying string in the paper after the thanks note.
7. **`.bib` leaks.** Five `note = {...}` fields that said "Cited in N134 report / discussion as ..." (renders in most bibstyles including `plainnat`); anonymized-self-citation titles containing literal `N127 —`, `N130 —`, `N132 —`, `N133 —`, `N134 pre-registration (v3.1)`; section-header comments mentioning "prior Gradience work" and the "Cocchieri" misattribution history; an anonymized-self-citation `note` field pointing to `repository tag n134-report-v1 for de-anonymized reference`.

**Resolution.**

Edits were applied with `% ANON:` comment markers at each change point (TeX `%` comments are stripped at compile time, so they do not render in the PDF but remain in the source for reviewable audit and camera-ready restoration). The marker convention is one `% ANON:` comment line directly above the edited region noting what changed and why; the git diff remains the authoritative record of the precise text substitution.

Inside `.bib` entries the `% ANON:` comments had to live *above* the `@...{` opener rather than between fields, because BibTeX treats `%` inside an entry as a missing field name and silently drops the remainder of the entry. First compile attempt surfaced this (five "You're missing a field name" errors, five entries silently dropped); comments were moved above each entry's `@` line and the compile cycle re-ran clean.

**Concrete edits.**

*TeX (27 marker regions):*
- Author `\thanks{}` note removed; `\author{Anonymized for review}` substituted.
- All body-prose `N134` / `N133` / `N132` / `N130` / `N127` / `N07` references rephrased to *the present study* / *the worked example* / *the precursor studies* — the measurement content is unchanged, but the prose no longer carries internal project nomenclature.
- `N135-alt` parenthetical at §9 conclusion stripped (the activation-informed follow-up is still described; only the project-internal designator is removed).
- `Gradience N133` → removed; `Gradience $S_{\mathrm{H1}}$` → `$S_{\mathrm{H1}}$`.
- All `\texttt{sidecar/...}` and `\texttt{n134_*.md}` paths replaced with generic pointers to "supplementary materials" or "the supplementary incident log / technical report".
- `(RunPod Secure Cloud)` stripped in Appendix C; "commercial cloud" retained.
- "companion repository under tag \texttt{n134-submission-draft-v1}" phrase replaced with "the supplementary materials accompanying this submission."

*`.bib` (13 marker regions):*
- Five `note = {...}` fields had their "Cited in N134 ..." / "cited in the discussion" phrases stripped (zhang2025osrm, zhou2026demystifying, rahamim2026mergeability, hitit2026insystematic, he2025mergebench).
- Five anonymized-self-citation titles rewritten to generic descriptors (e.g., `N127 — same/cross alignment separation on DistilBERT` → `Same/cross alignment separation on a small encoder backbone`).
- Anonymized-self-citation `note` fields shortened to `Anonymized for review.` (stripped the `See repository for de-anonymized reference` and `See repository tag n134-report-v1` phrases).
- Two section-header comments ("Mergeability-prediction literature (Gradience positions against)" and "Anonymized self-citations (prior Gradience work)") neutralized.
- The Hitit-section comment block describing an internal "Cocchieri" misattribution correction and naming `sidecar/notes/*`, `docs/THEORY.md`, `docs/positioning/gradience_differentiation.md` was replaced with a neutral two-line authorship note.
- BibTeX keys (e.g., `anonymized2026n133`, `anonymized2026n134spec`) were intentionally *not* renamed, because (a) they are referenced from the body `.tex` and renaming would require coordinated edits in both files, and (b) in `plainnat` the key strings are not rendered in the bibliography — only the author, year, title, and note fields appear in the rendered entry, and all four have been neutralized. The rendered bibliography for the five self-citations shows `Anonymized. <generic title>. 2026. Anonymized for review.`

**Camera-ready restoration procedure.** Every edit is tagged with a grepable `% ANON:` comment that names what was stripped and where it came from. To restore the un-anonymized paper, the editor:

1. Checks out `v2-anonymized` tag, then branches.
2. Runs `grep -n "% ANON:" draft_v2_thesis_b.tex references.bib` to enumerate all edit sites.
3. At each site, consults the git diff at the anonymization commit to see the exact pre-anon text, and restores the original string. The `% ANON:` comment line itself is deleted in the restore edit.
4. Re-adds the `\thanks{}` note to the author block (named in the ANON comment at line ~40 of the tex).
5. Re-adds the `n134-submission-draft-v1` tag pointer in Appendix C (explicitly flagged in the ANON comment as "the single most identifying phrase in the original appendix").
6. Restores the original `note = {...}` field contents in the five `.bib` entries that had N134-referencing citation-purpose text.
7. Re-runs the full compile cycle (`pdflatex → bibtex → pdflatex → pdflatex`) and diffs the new PDF against the pre-anonymization PDF to confirm the restoration is complete.

**Compile verification.** Full cycle ran clean in the sandbox (TeX Live 2022 / Debian) with `microtype` disabled for a font-expansion sandbox incompatibility unrelated to the anonymization edits. Final PDF: 18 pages, matching the pre-anon baseline exactly (13 pp main + 5 pp refs/appendices). Zero undefined citations in the final pass. All 19 cited bibliography entries render correctly; the one uncited entry (`anonymized2026n127`) is defined but not referenced from the body, as expected. BibTeX cycle produced no errors after the inside-entry `% ANON:` comments were relocated above each entry's `@` opener.

**Honest state observation.** The anonymization pass is reviewable (every edit is a grepable marker) and reversible (the git diff is the audit trail), but it does not provide *strong* anonymity — a determined reviewer with access to arXiv or the Gradience GitHub public repo could still cross-reference the paper's specifics (Mistral-7B, $N = 45$ cross-task adapter pairs, family-pair partition, $S_{\mathrm{H1}}$ definition, committed $-0.533$ value) to associated prior work. TMLR's double-blind policy explicitly contemplates this — it asks that authors not make identification *trivial* (no names, no repo links, no tagged-release pointers), not that they make identification impossible. The present pass clears the trivial-identification bar; it does not clear the determined-cross-referencer bar, which is the policy's intent.

**Files touched.** `draft_v2_thesis_b.tex` (27 `% ANON:` marker regions), `references.bib` (13 `% ANON:` marker regions). No figures touched — the three included PDFs (`h1_decision.pdf`, `four_method_forest.pdf`, `layer_depth_trend.pdf`) carry no metadata leaks per a previous audit and are used as-is.

**Commit:** `[pending]`, to be tagged `v2-anonymized` after the TMLR template port (A3).

## RN-019 — A3 TMLR template port: `tmlr.sty` swap, bundled stylefiles, page-count delta 18 → 16

**Trigger:** Workstream A3 (2026-04-23). TMLR selected as primary venue per A1; A2 anonymization landed against the generic `\documentclass[11pt]{article}` preamble so that the template port could be done against a known-clean base. This RN records the preamble swap to TMLR's official stylefile (cloned from `github.com/JmlrOrg/tmlr-style-file`) and the page-count re-verification.

**Issue.** The generic preamble (`\documentclass[11pt]{article}` + `\usepackage[letterpaper,margin=1in]{geometry}` + `\usepackage{natbib}` + `\bibliographystyle{plainnat}`) was a neutral placeholder, not a submission preamble. TMLR requires the official `tmlr.sty` / `tmlr.bst` bundle, which (a) encodes TMLR's page dimensions (6.5in × 9.0in text area at 10pt), (b) auto-generates the "Under review as submission to TMLR" page header and the "Anonymous authors / Paper under double-blind review" title block when used without options, and (c) controls its own citation style via `\setcitestyle{authoryear,round,citesep={;},aysep={,},yysep={;}}`. Submitting under a generic article class would be an instant-desk-reject format violation.

**Resolution.**

Edits were applied with a new `% TMLR:` comment-marker convention (parallel to `% ANON:` but tracking template-port changes rather than anonymization). Rationale for a distinct marker: if the paper is ever redirected to a different venue (NeurIPS E&D fallback, AIES secondary), the TMLR-specific preamble changes need to be reversible independently of the anonymization edits. `grep '% TMLR:'` now enumerates exactly the lines that would need to be touched for a template re-port.

**Concrete preamble edits** (`draft_v2_thesis_b.tex`, lines ~20-70):

- `\documentclass[11pt]{article}` → `\documentclass[10pt]{article}` (TMLR convention; 10pt font shrinks the layout by ~11% vs. 11pt).
- `\usepackage[letterpaper,margin=1in]{geometry}` → removed. `tmlr.sty` sets `\textwidth 6.5in`, `\textheight 9.0in`, `\oddsidemargin 0in`, `\topmargin -0.625in` internally.
- `\usepackage[T1]{fontenc}` → removed. `tmlr.sty` includes `\usepackage[T1]{fontenc}` + `\usepackage{lmodern}`.
- `\usepackage{natbib}` → removed. `tmlr.sty` does `\RequirePackage{natbib}` with `\setcitestyle{authoryear,round,citesep={;},aysep={,},yysep={;}}`.
- Added `\usepackage{tmlr}` with commented-out option-swap hints: `[accepted]` for camera-ready, `[preprint]` for arXiv.
- `\bibliographystyle{plainnat}` → `\bibliographystyle{tmlr}`.
- `\author{Anonymized for review}` → TMLR-format placeholder using the package's `\name`/`\email`/`\addr` macros (`\author{\name Anonymized \email anonymous@anonymous \\ \addr Anonymized Institution}`). tmlr.sty auto-hides this under double-blind, so the placeholder never renders during review. A commented-out camera-ready author block (`\name John T. Nanney \email johntnanney@gmail.com \\ \addr TODO: affiliation string`) is included immediately above it for camera-ready restoration; the affiliation string is flagged as a `TODO:` because it has not been committed-to yet.
- `\date{April 2026}` → removed. TMLR's `\@maketitle` does not use `\@date`; dates are set via `\month`/`\year` macros at camera-ready only.

The header comment block at lines 1-20 of the .tex was updated to reference the TMLR stylefile and RN-019 rather than the generic-article-class placeholder.

**Bundled files.** The following files were copied from the upstream TMLR stylefile repo into `papers/n134_workshop/` so the submission tarball is self-contained (TMLR accepts a single-archive upload):
- `tmlr.sty` (6.6 kB) — the main style package.
- `tmlr.bst` (27 kB) — the bibliography style matching TMLR's `\setcitestyle`.
- `fancyhdr.sty` (17 kB) — required by `tmlr.sty` for the page-header machinery.
- `math_commands.tex` (12 kB) — optional math-macro library used in TMLR's `main.tex` template; not currently `\input` from our draft but bundled in case v3 revisions adopt its macros. Candidate cleanup if unused by camera-ready.

Upstream repo: `https://github.com/JmlrOrg/tmlr-style-file` (clone + copy; no modifications to any of the four upstream files).

**Compile verification.** Full cycle (`pdflatex → bibtex → pdflatex → pdflatex`) ran clean in the sandbox (TeX Live 2022 / Debian) with `microtype` disabled on a verification copy (same sandbox incompatibility as RN-018, unrelated to the template port; the canonical file retains `\usepackage{microtype}` since MacTeX handles it). BibTeX ran without errors against `tmlr.bst`; all 19 cited entries rendered correctly under TMLR's author-year citestyle. Zero undefined citations. One uncited entry (`anonymized2026n127`) defined but not referenced, as expected.

**Page-count delta:**
- Pre-port (11pt + 1" margins): **18 pages** total (13 pp main + ~2 pp refs + 3 pp appendix, per RN-018 state).
- Post-port (10pt + TMLR 6.5×9.0in): **16 pages** total (11 pp main-body §§1-9, pp 12-13 references, pp 13-16 appendices A-G).
- Delta: **-2 pages** (-11%). The page count drop is dominated by the 11pt → 10pt font shrink; TMLR's text area (6.5×9.0in) and our prior 1"-margin letter layout (6.5×9.0in) have nearly-identical usable area, so margin changes alone would not move the page count materially.

This places the main body comfortably within TMLR's soft-cap guidance (typical main-text target ≤12 pp before references and appendices), with 1 pp of slack if future revisions add framework content.

**Page-1 rendering sanity check.** `pdftotext` dump of page 1 confirms TMLR mode is active:
- Page header: `Under review as submission to TMLR` (expected string when `tmlr.sty` used without options).
- Title block: title correctly set with `\LARGE\bf\sffamily`.
- Author block: `Anonymous authors / Paper under double-blind review` (expected auto-replacement under anonymous mode).
- Abstract: centered "Abstract" heading with the `\large\bf\sffamily` treatment TMLR prescribes.

**Camera-ready restoration procedure additions** (extends RN-018's 7-step procedure):

Between steps 4 and 5 of RN-018's procedure, two TMLR-specific steps are inserted:
- **4a.** In `draft_v2_thesis_b.tex`, swap `\usepackage{tmlr}` → `\usepackage[accepted]{tmlr}` (or `[preprint]` for arXiv posting).
- **4b.** Uncomment the `% \author{\name John T. Nanney \email johntnanney@gmail.com \\ \addr TODO: affiliation string ...}` line immediately above the anonymized author block, fill in the TODO affiliation, and comment out or delete the `\author{\name Anonymized ...}` line.

All other `% TMLR:` comment lines (class size, package removals, bibliographystyle swap) should be *retained* at camera-ready — they document the template-port commit for future readers of the repo, and `%` lines do not render.

**Deferred from A3 to A4:**
- Actual TMLR OpenReview submission: pack submission bundle (tex + bib + sty + bst + figures + bbl), upload to OpenReview, fill in submission form.
- Pre-submission visual proof pass on a freshly-compiled PDF on John's MacTeX install (sandbox compile is a syntactic-correctness + layout-approximation check; the canonical rendering check needs the user's own TeX environment with `microtype` enabled).
- Supplementary-materials archive: decide whether to include the sidecar notes / analysis JSONs at submission time or only after acceptance. TMLR permits optional supplementary material but does not require it.

**Files touched in A3.** `draft_v2_thesis_b.tex` (preamble block, lines ~20-70; 11 new `% TMLR:` marker regions; 1 new `% ANON:` marker region for the camera-ready author block). No body-text edits; no bib edits. New files in the directory: `tmlr.sty`, `tmlr.bst`, `fancyhdr.sty`, `math_commands.tex` — all copied verbatim from upstream.

**Honest state observation.** The sandbox compile is a stronger check than pure syntactic correctness but weaker than a full visual proof — it validates that `tmlr.sty` loads, the bibliography compiles under `tmlr.bst`, page numbers are assigned correctly, and no package conflicts surface. What it does not check: that `microtype` font-expansion on John's MacTeX produces the same page breaks as the sandbox's non-microtype compile. Historically `microtype` can shift line-count by 1-3 lines per page, which at 16 pages could plausibly push the total to 15 or 17. John should re-run the full compile on his Mac once before packing the submission bundle and confirm the final page count matches what reviewers will see.

**Commit:** `[pending]`, to be included in the `v2-anonymized` tag (or tagged separately as `v2-tmlr` if A3 lands in a second commit).

