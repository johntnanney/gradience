# Revision Notes for N134 Paper

Each entry records a specific revision decision, its trigger (self-review, outside reader, reviewer comment), and its resolution. Entries are numbered sequentially and referenced from commit messages.

---

## RN-000 — draft_v1 is a skeleton, not prose

**Trigger:** T3 consolidation (2026-04-20).
**Issue:** The consolidation spec's T3 source assumed a paper draft existed at `/mnt/user-data/outputs/n134_paper_draft_v1.md` or similar. No such draft was produced. The only prose asset is `sidecar/notes/n134_report.md`, which is an internal findings report with full appendices (~350 lines), not a workshop paper.
**Resolution:** `draft_v1.md` committed as a structure-only skeleton that (a) points at the report as canonical source, (b) lists the intended section structure and target lengths, and (c) specifies the figure plan. `draft_current.md` initialized as an identical copy. The actual paper prose is the revision work that remains to be done.
**Commit:** [filled at commit time]

## RN-001 — [Title]

**Trigger:** [self-review | reader X | reviewer Y]
**Issue:** [what was identified]
**Resolution:** [what was changed]
**Commit:** [hash of the commit that implements the resolution]
