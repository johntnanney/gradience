# Figure Captions (staging area)

Captions are drafted and revised here during paper work, then pasted into the paper body only when finalized. This separation keeps caption edits out of draft diffs.

**Paper-numbering convention:** figures are numbered 1, 2, 3 in the paper via `\begin{figure}` ordering. Script and output filenames are descriptive (no numeric prefix) so dropping or reordering figures does not force a rename. The mapping below is the paper's intended order.

---

## Figure 1 — H1 decision plot

**Script:** `scripts/n134/figures/fig_h1_decision.py`
**Output:** `papers/n134_workshop/figures/h1_decision.{pdf,png}`

**Caption draft (v1):**

> **Figure 1.** Scatter of O-module depth-weighted alignment ($S_{\mathrm{H1}}$) against max post-merge degradation for the 45 evaluated cross-task pairs. The grey dashed line is the raw OLS regression ($\rho = -0.180$, $p = 0.236$); the red solid line is the regression of max degradation on $S_{\mathrm{H1}}$ after both variables are OLS-residualized on FAMILY\_B task-family-pair dummies (partial $\rho = -0.533$, $p = 1.6 \times 10^{-4}$; bootstrap 95\% CI under family-pair block resampling: $[-0.825, -0.131]$). The partial regression line is plotted in raw coordinates with its intercept set to pass through the joint mean of the data; the slope reflects the same partial correlation reported in Table X. The partial correlation is large in magnitude but opposite in sign to the pre-registered prediction, and therefore fails the sign constraint of the pre-registered decision rule regardless of magnitude. Point colors encode family-pair identity (28 levels, `tab20` cycled); the informal purpose of the coloring is to make the family-level clustering visible to the eye — the clustering is the visual payload of the 88\% family-pair $R^2$ that dominates the regression baseline, and is what leaves approximately $\Delta R^2 = 0.003$ of residual variance for $S_{\mathrm{H1}}$ to explain.

## Figure 2 — Four-method forest plot

**Script:** `scripts/n134/figures/fig_four_method_forest.py`
**Output:** `papers/n134_workshop/figures/four_method_forest.{pdf,png}`

**Caption draft (v1):**

> **Figure 2.** Four-method comparison of pairwise merge-triage performance on the 45 evaluated cross-task pairs. Each method ranks pairs by its own risk score; the 22 lowest-risk pairs are retained and the mean max_degradation of the retained set is reported. Bootstrap 95\% CIs are computed with family-pair block resampling (5{,}000 resamples). Vertical dashed reference line: random-baseline mean max_degradation across all 45 pairs (3.14\%). Methods are ordered by the magnitude of their Spearman $\rho$ between risk score and max_degradation; per-method $\rho$ and $p$ are annotated on the right margin. No method achieves significance at $\alpha = 0.05$; three of the four methods produce wrong-signed rank correlations. KnOTS is the only method with both a right-signed $\rho$ and a positive improvement over random baseline, but its CI for retained-set degradation crosses zero. The null is a regime null under confound control at $N = 45$, not a Gradience-specific result.

## Figure 3 — Layer-depth trend in same/cross alignment ratio

**Script:** `scripts/n134/figures/fig_layer_depth_trend.py`
**Output:** `papers/n134_workshop/figures/layer_depth_trend.{pdf,png}`

**Caption draft (v1):**

> **Figure 3.** Layer-depth trend in the same/cross alignment ratio on Mistral-7B under N134. Per-layer same/cross alignment ratio plotted against layer index (0--31); linear fit in solid red with $r = 0.919$, slope $= 0.031$ per layer, $p = 1 \times 10^{-13}$. The ratio rises from approximately 1.66 at layer 0 to approximately 2.58 at layer 31. This depth trend is a property of the aggregate same/cross separation geometry — deeper layers separate same-task from cross-task pairs more sharply across the adapter population — and is what originally motivated $S_{\mathrm{H1}}$'s linear depth-weighting. It is not itself a predictor of per-pair merge outcomes: the motivated depth-weighting of $S_{\mathrm{H1}}$ does not translate into per-pair risk prediction in the present study (see Figure 1). The trend replicates the analogous layer-depth observations in N130, N132, and N133.

---

## Dropped figure

### Original F2 — cross-architecture same/cross alignment comparison — DROPPED

Dropped at T2 pre-figure-work, 2026-04-20. N130 does not persist per-pair alignment records at the granularity needed for distributional plotting, and the three remaining studies (N132, N133, N134) use two different metric families (subspace principal-angle vs. SV-weighted cosine). The cross-architecture claim is made in §5 prose instead. See RN-010 in `revision_notes.md`. Paragraph 5 of §5 draft candidate (suitable as starting prose for revision):

> The same/cross alignment ratio has been measured in four studies at two scales and across two metric families: approximately 5× on DistilBERT (N130, subspace principal-angle metric), 2.3× on DeBERTa (N132, same metric family), and 3.06× (N133) and 2.28× (N134) on Mistral-7B (SV-weighted cosine metric). The consistent same > cross separation across architectures, scales, and metric families indicates that task-boundary detection via spectral geometry is a robust property of LoRA adaptation rather than a consequence of any particular measurement choice.
