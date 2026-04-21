# Figure Captions (staging area)

Captions are drafted and revised here during paper work, then pasted into the paper body only when finalized. This separation keeps caption edits out of draft diffs.

---

## Figure 1 — H1 decision plot

*Placeholder.* Scatter of S_H1 versus max_degradation for all 45 cross-task pairs, with family-pair coloring; raw and family-residualized regression lines; annotation of partial ρ, p, ΔR², and bootstrap 95% CI. Caption emphasizes that the wrong-signed significant partial correlation constitutes a null under the pre-registered decision rule.

## Figure 2 — DROPPED

Dropped at T2 pre-figure-work, 2026-04-20. N130 does not persist per-pair alignment records at the granularity needed for distributional plotting, and the three remaining studies (N132, N133, N134) use two different metric families. The cross-architecture claim is made in §5 prose instead. See RN-010 in `revision_notes.md`. Paragraph 5 of §5 draft candidate (suitable as starting prose for revision):

> The same/cross alignment ratio has been measured in four studies at two scales and across two metric families: approximately 5× on DistilBERT (N130, subspace principal-angle metric), 2.3× on DeBERTa (N132, same metric family), and 3.06× (N133) and 2.28× (N134) on Mistral-7B (SV-weighted cosine metric). The consistent same > cross separation across architectures, scales, and metric families indicates that task-boundary detection via spectral geometry is a robust property of LoRA adaptation rather than a consequence of any particular measurement choice.

## Figure 3 — Four-method forest plot

*Placeholder.* Horizontal forest plot of retained-set mean max_degradation for Gradience, KnOTS, TSV, SVC with bootstrap 95% CIs and the random baseline (3.14%) as a vertical reference line. Per-method Spearman ρ and p annotated inline. Caption emphasizes that no method achieves significance at α = 0.05.

## Figure 4 — Layer-depth trend in same/cross ratio

*Placeholder.* Line plot of same/cross alignment ratio by layer index on Mistral-7B N134; linear fit with r = 0.919 annotated. Caption emphasizes that the depth trend is robust and architecture-general but does not rescue the per-pair risk prediction; the motivation for S_H1's depth weighting survives, but S_H1 itself does not.
