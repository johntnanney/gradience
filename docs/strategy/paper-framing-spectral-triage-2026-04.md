# Paper Framing: Spectral Triage System Paper

**Date:** April 2026
**Status:** Internal planning note — not for external distribution

---

This is worth thinking through carefully, because the program has shifted enough that the paper that made sense six months ago may not be the paper the evidence best supports now.

The existing draft — GRADIENCE_PAPER_DRAFT_REVISED.md, which centers on Llama-2-7B merge results — was written when the decoder-scale merge work was the leading edge. That's no longer where the strongest evidence lives. The Llama results (5 pairs, Study 16) are now better understood as bounded supporting evidence for the structural-behavioral separation finding, not as the main contribution.

The paper I think the evidence actually supports now is a **spectral triage system paper** — not a merge-outcome paper, not a spectral-theory paper, but a paper whose core claim is: *you can dramatically reduce the merge search space using structural pre-filtering, and here is a mechanistic account of why that works.*

Three reasons this is the right framing:

**First, it's where the validated evidence is densest.** The 90–93% candidate elimination, zero false positives on task boundaries, the conjunctive failure model, eligibility gating, the structural-behavioral separation — these all hang together as a single argument. The paper writes itself as: here's the problem (combinatorial merge search), here's the approach (spectral triage), here's the mechanism (conjunctive V-module pathology + readout incompatibility), here's the evidence (5 inventories, 3 backbones, 53+ pairs), and here are the explicit boundaries (small encoders, classification, rank ≤ 16). That's a complete paper without requiring a single additional experiment.

**Second, the mechanistic contribution is genuinely novel.** The conjunctive failure model — arrived at by eliminating five simpler hypotheses — is the kind of result that gives a system paper intellectual weight beyond "we built a tool and it works." Most merge papers in the literature are about strategies (TIES, DARE, Task Arithmetic). Almost none ask *why merges fail at the geometric level*. The fact that the answer is conjunctive (two independent conditions must co-occur) rather than scalar (some threshold on some metric) is a real finding, and it's the kind of finding that survives even if the specific thresholds turn out to need revision.

**Third, the boundary discipline is itself a contribution to the discourse.** The LoRA/merge literature is full of papers that test on one architecture, report a correlation, and imply generality. A paper that explicitly maps what is validated, what is suggestive, and what is untested — and treats those boundaries as part of the intellectual contribution rather than a reluctant concession in the limitations section — would be positioning itself against the prevailing norms of the field. That's a stronger rhetorical position than overclaiming.

The secondary results slot in naturally as supporting evidence rather than co-equal contributions. The spectral partitioning (N127) goes in as the generative theory for why the observables are predictive. The 86-adapter audit (Post 7) goes in as evidence that the spectral lens reads real structure at ecosystem scale. The decoder-scale merge evidence (Posts 3, Study 16) goes in as evidence of directional transfer, explicitly flagged as suggestive. The training dynamics (Posts 5, Study 12) can be mentioned in a future-work section or omitted entirely — they're interesting but belong to a different argument.

What I'd **leave out**: the telemetry/DFA strand, the curvature work, the philosophy, and most of the bounded companion diagnostics. These are real results but they dilute the paper's argument. A paper that tries to be "everything Gradience has done" will read like a technical report. A paper that says "here is one operational claim, here is the mechanism, here is the evidence, here is where it stops" will read like a contribution.

The one strategic question is **timing relative to DeBERTa**. If you can get ~3 hours of GPU before submission, the third-backbone result either strengthens the mechanism to "backbone-general" (much stronger paper) or bounds it to "backbone-contingent" (still publishable, different framing). If you can't, the paper is still writable — the two-backbone mechanism with explicit acknowledgment that a third is needed is honest and defensible, and the rest of the evidence (triage workflow, ecosystem audit, spectral partitioning) doesn't depend on the mechanism being general. But if DeBERTa is within reach, it's worth waiting for, because the difference between "validated on two backbones" and "validated on three" is disproportionate to the compute cost.

The venue question is secondary to getting the framing right, but this reads to me like a systems paper with a mechanistic contribution — which puts it somewhere between an ML systems venue and a methods-oriented ML venue. The boundary discipline and explicit scope-mapping might also make it a good fit for a workshop that values epistemological rigor over headline numbers.
