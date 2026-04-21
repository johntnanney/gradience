# What Gradience Is — Short Positioning Statement

*A standalone elevator pitch for Gradience. For the full five-axis
conceptual map, see THEORY.md §8. For program-by-program related-work
analysis, see `technical-report.md` §8.*

---

## The honest summary

Gradience is not best-in-class at execution. KnOTS, TSV, and SVC are
better at improving merge results — they are focused execution methods
from larger labs with more resources. Gradience is not best-in-class at
theoretical derivation. Xu 2026 and recent OpenReview theoretical
submissions (e.g., *LoRA Provably Reduces Forgetting and Enables
Adapter Merging in Multiclass Linear Classification*, OpenReview
FSDxP3ZpAx) are doing more rigorous formal work. Gradience is not best-in-class at benchmark
comparison. MergeBench and Hitit et al. 2026 cover more ground.

Gradience is uniquely positioned at the intersection of five properties:

1. **Weight-space substrate.** Analyze the LoRA $\Delta W$ itself, not
   activations, training trajectories, or behavioral outputs.

2. **Increasing commitment to pre-registered confirmatory
   epistemology.** N133 and N134 (decoder-scale merge triage) commit to
   hypotheses and decision rules before data collection; earlier
   Gradience work (Studies 14, 16, 17; the original technical report)
   was descriptive but shared the underlying measurement discipline.
   Pre-registration is being adopted progressively, not uniformly.

3. **Decoder-LLM scale with controlled confounds.** N133 and N134
   operate at Mistral-7B with explicit confound-defeat design, not at
   ViT-B/32 or small encoders. The N133 confound-cascade diagnostic —
   the kind of artifact that only accumulates when a program commits
   to pre-registered testing — is public.

4. **Triage purpose.** Decide *whether* to merge, not *how*. Most of
   the modular-adaptation literature works on execution (given a
   decision to merge, improve the merge) or prediction (estimate
   post-merge accuracy). Triage is the operationally most important
   question for a practitioner managing an adapter inventory but has
   the lowest academic reward-to-effort ratio.

5. **Measurement-instrument stance.** Treat spectral outputs (stable
   rank, energy rank, subspace overlap, SV-weighted alignment) as
   instruments with reliability, standard error of measurement,
   minimal detectable change, and construct validity — not as
   convenient scalars. See THEORY.md §5.3 "Toward a unified spectral
   measurement framework."

No other program in the current literature occupies that intersection.
Most programs occupy one of these, some two, a few three; none occupy
all five. This is not a claim of dominance — it is a claim of
distinctive positioning. Each axis of distinctiveness is independently
defensible, and the combination is defensible as a coherent research
program rather than an arbitrary assemblage of choices.

The durability of this positioning depends on whether the community
comes to value measurement-instrument stance and pre-registration, or
whether it remains the current norm to optimize for benchmark
comparison and novel-method publication. The former trend is slowly
gaining momentum (Hitit et al.'s LLM-scale benchmark is a symptom;
Zhou et al.'s interpretable-predictor framework is another; the AI
safety community's move toward pre-registration in interpretability
work is another). Gradience is well-placed to benefit if that trend
continues; less well-placed if it reverses.

---

## Paragraph for external communication

*Gradience is a measurement-first framework for LoRA adapter analysis.
Where most work in this space proposes new merge algorithms (KnOTS,
TSV, SVC, TARA-Merging) or new predictive models for merge success
(Zhou et al., Rahamim et al.), Gradience occupies a distinct cell:
pre-registered confirmatory testing of weight-space spectral
diagnostics at decoder-LLM scale, under explicit confound control,
treating spectral outputs as instruments with psychometric properties.
The program's empirical record (N127 through N134) commits to
hypotheses before data collection on its recent studies and reports
confound-cascade diagnostics when pre-registered predictions fail. The
methodological contribution is complementary to execution methods
(triage decides what to merge; execution methods improve retained
merges) and complementary to prediction methods (Gradience's
pre-registered tests at decoder scale sit alongside descriptive
predictive models at encoder or vision-classifier scale). The overall
research program aims at a mature scientific instrument for modular
adaptation, with measurement validity as the animating commitment.*

---

*Prepared April 20 2026. Consolidates positioning analysis from the
April 2026 literature review (four agent scans + direct verification).
Companion documents: THEORY.md §8 (conceptual map of modular adaptation
research), `technical-report.md` §8 (related programs), N134 spec v3.1,
RESEARCH_INVENTORY.md §9 (external reference literature).*
