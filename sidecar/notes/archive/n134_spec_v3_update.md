# N134 Spec v3 Update — Instructions for Coding Agents (Archived)

**This document is archived.** It contains the edit instructions that were applied to `sidecar/notes/n134_spec.md` to produce spec v3 from spec v2.

**Status:** Applied April 19 2026. The active spec is `sidecar/notes/n134_spec.md` at v3.

**Scope of changes applied:**

1. Version header updated from v2 to v3.
2. §1 extended with two new paragraphs: "Position relative to the mergeability-prediction literature" (citing Rahamim et al. 2026, Zhou et al. 2026, Bolton et al. 2026) and "Why decoder-scale triage matters now" (citing arXiv:2511.21437).
3. §8 "H1 fails" extended from three named candidate explanations to four (added: intrinsic mergeability per Rahamim et al.).
4. Appendix A extended with a fourth declared unknown (weight-space sufficiency at decoder scale).
5. Appendix B heading renamed from "What v2 changed from v1" to "Change log"; v2→v3 subsection added.
6. Closing line updated to reflect v3 supersedes v2.

**What was NOT changed (verified):**

- §2 (confounds C1–C4): unchanged.
- §3 (experimental design): unchanged.
- §4 (H1 score, decision rule, confirmatory replications): unchanged.
- §5 (statistical protocol): unchanged.
- §6 (four-method comparison): unchanged.
- §7 (deviation policy): unchanged.
- §9 (execution plan, directory layout, dependencies, resource estimate, artifacts): unchanged.
- `scripts/n134/` (all 9 scripts): unchanged.
- `sidecar/data/n134/audit_v2_1.schema.json`: unchanged.
- Three original bullets in Appendix A: preserved; fourth bullet is appended.

**Rationale:** a parallel research program on "mergeability prediction" (Zhou et al. 2026, Rahamim et al. 2026, Bolton et al. 2026) emerged since v2 was committed, and an LLM-scale evaluation paper (Cocchieri et al. / arXiv:2511.21437) sharpened the case for why triage matters at decoder scale. N134 remains differentiated — pre-registered, decoder-scale, confound-controlled, pairwise — but the spec's positioning now cites this adjacent program explicitly.

---

*Archive retained for audit trail. See `sidecar/notes/n134_spec.md` for the active pre-registration.*
