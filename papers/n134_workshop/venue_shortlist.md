# Venue Shortlist — N134 Thesis B Paper

**Paper:** *Measurement Discipline for ML Diagnostics: A Psychometric Framework with a LoRA-Merging Case Study*

**Current state:** 13 pp main text, 18 pp total PDF (main + refs + appendices). Position/methods paper with framework in §§1–2 and §§6–8, and a worked empirical example in §§3–5.

**Compiled:** 2026-04-22 (Wed)

---

## Strategic summary

The paper's shape (13 pp main, positional-with-empirical-example, measurement-theoretic framing) rules out several venues whose CFPs have already closed for the 2026 cycle (ICML Position Papers track, FAccT 2026, the philosophy-of-ML workshops surveyed). Of the venues still open, two are strong thematic fits with very different tradeoffs: **TMLR**, which accepts papers at the paper's current length with no trim required and evaluates on technical correctness, and **NeurIPS 2026 Evaluations & Datasets Track**, whose renamed scope ("advancing the science of AI evaluation") reads like a direct description of this paper's thesis but whose 9-page main-text limit would require substantial compression.

**Tentative primary: TMLR.** Rationale: (a) length matches the paper's natural shape without forcing another compression pass that risks damaging the framework/case-study balance; (b) technical-correctness review emphasis is itself the venue analogue of the paper's measurement-discipline thesis; (c) rolling submissions eliminate calendar pressure and permit a careful final pass; (d) recent TMLR → NeurIPS/ICML/ICLR Journal-to-Conference Certification gives accepted TMLR papers a presentation path at a major conference without a resubmit. Cost: slower visibility than a NeurIPS acceptance announcement.

**Tentative backup: NeurIPS 2026 Evaluations & Datasets Track.** Rationale: thematic fit is nearly perfect — the renamed track explicitly welcomes "methodological analyses and formal treatments of metrics and evaluation design," which is the paper's §§1–2 and §7 framework content. Cost: a two-week compression from 13 pp → 9 pp main before the May 6 deadline, which is non-trivial and risks the very content the v2 revision pass just stabilized.

---

## Candidate venues

### 1. TMLR — Transactions on Machine Learning Research *(TENTATIVE PRIMARY)*

- **Deadline:** Rolling. Submissions resumed Jan 6, 2026; open continuously.
- **Page limit:** Flexible / variable manuscript length. Explicitly accommodates "shorter-format manuscripts" but accepts longer — no hard page cap.
- **Anonymization:** Double-blind.
- **Review criterion:** Technical correctness emphasized over subjective significance. Fast turnarounds, rolling review.
- **Fit notes:** The paper's position/methods/case-study structure and 13 pp length match TMLR's native format. The double-blind anonymization pass required by Workstream A2 satisfies this requirement. Venue's own epistemic stance (technical correctness over subjective novelty) aligns with paper's argument that measurement-disciplined reporting should be a field-level norm rather than a rare virtue.
- **Presentation path:** TMLR joined the NeurIPS/ICML/ICLR Journal-to-Conference Track (announced Oct 2025) — accepted TMLR papers earning the Journal-to-Conference Certification may be presented at one of those conferences.
- **URL:** https://jmlr.org/tmlr/

### 2. NeurIPS 2026 Evaluations & Datasets Track *(TENTATIVE BACKUP)*

- **Deadline:** Abstract May 4, 2026 AoE; full paper May 6, 2026 AoE. Conference Dec 2026.
- **Page limit:** 9 pp main (inherits from NeurIPS main-track template).
- **Anonymization:** Double-blind (new default in 2026 — previously single-blind for datasets; the default shifted with the track's rescoping). Authors with dataset-centric submissions may still opt for single-blind.
- **Review criterion:** "Advancing the science of AI evaluation" — explicitly welcomes work that analyzes strengths, limitations, or failure modes of existing benchmarks or evaluation practices; studies benchmark saturation or overfitting; and contributes "methodological analyses and formal treatments of metrics and evaluation design."
- **Fit notes:** Thematically the best match of all open venues. The track's renamed scope in 2026 reads like it was written to describe this paper. Cost: compressing 13 pp → 9 pp main is a two-week job and risks the §§1–2 framework / §5 worked example / §7 generalization balance that v2 just settled.
- **Code/data:** Code release not mandatory for "analytical, empirical, conceptual, or methodological" contributions that don't introduce reusable executable artifacts, provided paper contains sufficient detail. The N134 analysis scripts already exist under `sidecar/scripts/n134/` and would be straightforward to release, but per policy would not be required.
- **URL:** https://neurips.cc/Conferences/2026/CallForEvaluationsDatasets

### 3. NeurIPS 2026 Position Paper Track *(candidate — requires compression)*

- **Deadline:** Abstract May 4, 2026 AoE; full paper May 6, 2026 AoE.
- **Page limit:** 9 pp main (inherits from NeurIPS main-track template).
- **Anonymization:** Double-blind.
- **Structural requirements:** Title must state the position; abstract must begin with "This position paper argues that…"; introduction must state position in bold text; an "Alternative Views / Counterarguments and Objections" section in the main body is permitted (and near-mandatory in practice). A 250-word rationale for why the submission belongs in the position-paper track is required at submission.
- **LLM policy:** Stricter than main track — final paper must be "substantially written by human authors." Authors attest at submission.
- **Fit notes:** The paper's §§1–2 framework and §8 objections handling would translate naturally into this track's structural requirements — in fact, the §8 merge (executed in the v2 revision) already produces the "Alternative Views" section this track expects. Same 9-page compression cost as Option 2. Thematic fit slightly weaker than Option 2 because the paper includes substantial empirical content (§§3–5) that a pure position paper would not, which could complicate reviewer routing.
- **URL:** https://neurips.cc/Conferences/2026/CallForPositionPapers

### 4. AIES 2026 — Ninth AAAI/ACM Conference on AI, Ethics, and Society *(candidate — off-thesis)*

- **Deadline:** Abstract May 14, 2026 AoE; full paper May 21, 2026 AoE. Conference Oct 12–14, 2026, Malmö.
- **Page limit:** See CFP — typical AIES is 8 pp main + references.
- **Anonymization:** Double-blind per standard AAAI/ACM practice (verify on CFP page).
- **Fit notes:** Venue is methodologically sympathetic ("make methodological commitments explicit and speak meaningfully across disciplinary boundaries") but the paper's core thesis is measurement-methodological rather than ethics-policy-governance. AIES would be a better fit for the Diagnostic Reliability Report artifact under Workstream C, framed as a governance-relevant output, than for the current paper. Keep on the shortlist as an off-thesis option if the primary and backup both fall through.
- **URL:** https://www.aies-conference.com/2026/call-for-papers/

### 5. ICML 2026 Position Papers Track *(closed — reference only)*

- **Status:** DEADLINE PASSED. Abstract Jan 24, 2026; full paper Jan 29, 2026.
- **Note:** Listed here per the original brief so the reader can confirm the track exists and the cycle is closed for 2026. Worth holding for ICML 2027 if the second-domain paper (Workstream B) is ready by ~Jan 2027.
- **URL:** https://icml.cc/Conferences/2026/CallForPositionPapers

### 6. Philosophy-of-ML venues *(closed — 15-minute scan per brief)*

- Aarhus "Philosophy of Explainable AI: New Directions" workshop (July 2026): deadline **April 1, 2026 — PASSED**.
- 6ICPH (Philosophy of Mind: AI, Porto, May 4–8, 2026): conference imminent; submission cycle closed.
- MBR026_ROME (Model-Based Reasoning, June 2026): deadline **Feb 1, 2026 — PASSED**.
- *Synthese* special collections (ongoing, journal): a plausible home for a philosophy-pitched version of the framework chapters, but not for the current paper, which commits to an ML-methods reading audience.
- Conclusion: no open philosophy-of-ML venue matches the timing; the venue landscape in 2026 has the paper's natural home on the ML side rather than the philosophy side — a useful datum for the Workstream D institutional-home decision.

---

## Recommendation

**Submit to TMLR** as primary. The length match, the double-blind requirement, the rolling cycle, and the technical-correctness review criterion are all aligned with the paper's shape and thesis; the Journal-to-Conference Certification gives a path to conference presentation without paying the 9-page compression cost.

**Hold NeurIPS 2026 Evaluations & Datasets Track** as backup with a decision date of **April 28, 2026** — if by that date the TMLR submission has not begun (i.e., the anonymization audit in A2 revealed structural problems that TMLR's flexibility does not solve), redirect to the E&D track and commit to the two-week compression sprint. After April 28, the May 6 deadline becomes too tight.

**Off-thesis fallback:** AIES 2026 (May 21 deadline) if both TMLR and NeurIPS E&D routes close off.

---

## Sources
- [ICML 2026 Call for Position Papers](https://icml.cc/Conferences/2026/CallForPositionPapers)
- [NeurIPS 2026 Call for Papers](https://neurips.cc/Conferences/2026/CallForPapers)
- [NeurIPS 2026 Evaluations & Datasets Track](https://neurips.cc/Conferences/2026/CallForEvaluationsDatasets)
- [Introducing the Evaluations & Datasets Track at NeurIPS 2026 (blog)](https://blog.neurips.cc/2026/03/23/introducing-the-evaluations-datasets-track-at-neurips-2026/)
- [NeurIPS 2026 Call for Position Papers](https://neurips.cc/Conferences/2026/CallForPositionPapers)
- [NeurIPS 2026 Call for Workshops](https://neurips.cc/Conferences/2026/CallForWorkshops)
- [TMLR home](https://jmlr.org/tmlr/)
- [TMLR Journal-to-Conference Track announcement](https://medium.com/@TmlrOrg/tmlr-joins-neurips-icml-iclr-journal-to-conference-track-937a898eab3d)
- [FAccT 2026 CFP (closed)](https://facctconference.org/2026/cfp.html)
- [AIES 2026 CFP](https://www.aies-conference.com/2026/call-for-papers/)
- [Aarhus Philosophy of XAI workshop (closed)](https://projects.au.dk/treat/workshop-2026/open-call)
- [6ICPH Porto](https://philevents.org/event/show/143946)
- [MBR026_ROME (closed)](https://logicandknowledge.substack.com/p/mbr026_rome-model-based-reasoning)
