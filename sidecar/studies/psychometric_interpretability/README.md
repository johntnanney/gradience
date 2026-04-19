# Research thread: psychometric framework for feature-level interpretability

**Status:** outline v0 (April 19 2026). Not yet started.
**Relation to main research program:** **separate thread.** Independent of the
N133 / N134 merge-triage line. This thread is a methodology contribution to
interpretability research, motivated by Lindsey et al. 2026 and the broader
SAE-features-with-causal-claims literature.

## Thread overview

One-paragraph thesis (from the outline):

> Interpretability research increasingly makes claims of the form: *feature F
> represents concept C, and intervening on F causally affects behavior B.*
> This is structurally identical to the claim psychometric measurement theory
> has refined over roughly a century: *instrument I measures construct C, and
> C predicts outcome O.* We translate classical test theory and modern
> validity theory into the interpretability setting, demonstrate the framework
> empirically on emotion features in a small open model, and identify
> specific gaps in current practice that adopting the framework would close.

## Files in this directory

| File | Status | Purpose |
|------|--------|---------|
| `README.md` | current | thread index + entry point |
| `outline_v0.md` | committed | working outline, all §§1-7 + appendices |

## Open decision points (from outline v0)

Before drafting begins, the following need to be resolved:

1. Venue (NeurIPS D&B / ICLR methods / JMLR / bridge venue)
2. Model (Gemma-2-2B / Gemma-2-9B / Llama-3.2-3B)
3. Cross-linguistic DCF dimension — include or drop
4. Survey of existing interpretability papers for MDC headline — include or skip
5. Pre-registration recommendation framing (flagship vs soft)
6. Early feedback from interpretability researchers — pursue or not
7. Framework name

## Resource estimate

- Compute: $500-1500 on RunPod (SAE feature extraction, activation collection)
- Time: 4-6 months single-author, assuming ~60% of research time on this
- Compatible with N134/N135 consolidation only if strictly second-priority

## Why this lives in the sidecar

The sidecar is for research questions that do not (yet) belong in core Gradience.
A methodology paper on measurement validity for interpretability features is
squarely sidecar-appropriate: exploratory, mechanism-oriented, not obviously
promotable to the core product. If this thread produces a paper, the question
of whether any of the estimator code belongs in `gradience/` (e.g. as a
`gradience.psychometrics` submodule for adapter feature validity) is a
separate downstream decision.

## Picking up this thread later

The fastest re-entry path is:

1. Re-read `outline_v0.md` end to end (~20 minutes).
2. Resolve the seven decision points above.
3. Do the two-direction literature review flagged in "What's needed to start".
4. Curate the ~500-text stimulus set (the biggest scope risk; scoped down to
   300 English-only if cross-linguistic is dropped).
5. Begin §4 empirical work after decision points are resolved.

The outline deliberately includes `[DECIDE]`, `[SCOPE]`, and `[JUDGMENT]`
markers throughout so unresolved questions are easy to scan for.
