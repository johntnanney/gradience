# Revision Notes for N134 Paper

Each entry records a specific revision decision, its trigger (self-review, outside reader, reviewer comment), and its resolution. Entries are numbered sequentially and referenced from commit messages.

---

## RN-000 — draft_v1 is a skeleton, not prose

**Trigger:** T3 consolidation (2026-04-20).
**Issue:** The consolidation spec's T3 source assumed a paper draft existed at `/mnt/user-data/outputs/n134_paper_draft_v1.md` or similar. No such draft was produced. The only prose asset is `sidecar/notes/n134_report.md`, which is an internal findings report with full appendices (~350 lines), not a workshop paper.
**Resolution:** `draft_v1.md` committed as a structure-only skeleton that (a) points at the report as canonical source, (b) lists the intended section structure and target lengths, and (c) specifies the figure plan. `draft_current.md` initialized as an identical copy. The actual paper prose is the revision work that remains to be done.
**Commit:** [filled at commit time]

## RN-001 — Zhou et al. activation-dot-product citation unresolved

**Trigger:** T4 bibliography consolidation (2026-04-20).
**Issue:** The N134 report (§7.2, §7.3) references "Zhou et al.'s activation-dot-product result (r = 0.572)" as motivation for the activation-informed N135-alt follow-up. Web search on 2026-04-20 did not locate a distinct paper matching this description; the closest hit is Zhang & Zhou ACL 2025 OSRM (`zhang2025osrm`), which is a different paper and does not appear to report the quoted r=0.572 statistic.
**Resolution (pending):** Trace the citation through `sidecar/notes/n133_bp5_diagnostic.md` and earlier Gradience prior-work notes to find where this r=0.572 figure was first introduced. If it resolves to a real paper, add the bib entry and update in-text citations. If it does not resolve, drop the specific r=0.572 citation from paper prose and keep only the OSRM-framed activation-informed follow-up discussion. Note: draft_v1.md is a skeleton and does not currently contain this citation; the issue will only bind if draft_current.md prose re-introduces it.
**Commit:** [pending resolution]

## RN-002 — MergeBench attribution correction

**Trigger:** T4 bibliography consolidation (2026-04-20).
**Issue:** Prior Gradience positioning material (including `docs/positioning/gradience_differentiation.md`) attributes MergeBench to "Cocchieri et al. 2026." Web search established that the benchmark is actually He et al. 2025, arXiv:2505.10833. The report prose in `sidecar/notes/n134_report.md` mentions "Cocchieri et al." in §9 as part of the literature Gradience positions against; this was not a bug in the N134 report per se but it will be wrong if the paper draft uses that attribution.
**Resolution (pending):** When the paper draft (`draft_current.md`) enters revision and references MergeBench, use the `he2025mergebench` bib entry. Separately, consider a cleanup pass on `docs/positioning/gradience_differentiation.md` to correct the attribution there — this is outside T4's scope but flagged here for the record.
**Commit:** [pending resolution]

## RN-003 — Panahi et al. 2026 citation requires author verification

**Trigger:** T4 bibliography consolidation (2026-04-20).
**Issue:** Gradience positioning material references "Panahi et al. 2026" as an example of "more rigorous formal work" on LoRA merging theory. An April-2026 OpenReview paper ("LoRA Provably Reduces Forgetting and Enables Adapter Merging in Multiclass Linear Classification," OpenReview id `FSDxP3ZpAx`) matches the described character but the first author's surname was not confirmed as "Panahi" during the web search.
**Resolution (pending):** Check OpenReview authorship directly. If confirmed, add a proper bib entry. If the author is someone else, update the positioning material and any future paper-draft citation. Panahi is not currently cited in draft_v1.md (skeleton); this issue only binds if draft_current.md prose re-introduces it.
**Commit:** [pending resolution]

## RN-004 — [Title]

**Trigger:** [self-review | reader X | reviewer Y]
**Issue:** [what was identified]
**Resolution:** [what was changed]
**Commit:** [hash of the commit that implements the resolution]
