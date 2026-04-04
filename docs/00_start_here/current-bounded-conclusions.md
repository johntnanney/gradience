# Current Bounded Conclusions

**Audience:** maintainer, collaborator, strategy reviewer
**Status:** stable (decision index)
**Purpose:** single canonical status snapshot of bounded CPU-phase conclusions
**Canonical for:** what is settled, exploratory, kept, paused, or GPU-blocked right now
**Supersedes:** fragmented status reading across multiple memos
**See also:** [`../strategy/state-of-program-april-2026.md`](../strategy/state-of-program-april-2026.md), [`project-map.md`](project-map.md), [`stable-vs-experimental.md`](stable-vs-experimental.md), [`../strategy/cpu_phase_established_summary.md`](../strategy/cpu_phase_established_summary.md)

For the full project argument (theory, mechanism, validation), see the [Technical Report](../technical-report.md).
For the canonical full-program April 2026 snapshot, see [`../strategy/state-of-program-april-2026.md`](../strategy/state-of-program-april-2026.md).

---

## How to read this document

Each line below represents a research or development question that has been investigated and reached a bounded conclusion. "Bounded" means the conclusion holds within a specific, tested regime (small encoder models, classification tasks, LoRA rank ≤ 16) and should not be extrapolated beyond that regime without new evidence.

**Status labels:**

- `stable (bounded)` — settled for the current regime; safe to rely on operationally
- `resolved (bounded)` — question answered; no further work planned in current regime
- `bounded_keep` — useful but limited; keep available, do not promote or expand
- `keep_exploratory` — interesting structure found; not ready for operational use

---

## Decision Snapshot (As of April 3, 2026)

### Rank-proxy validation — `stable (bounded)`

**Question:** Can spectral analysis of singular value distributions guide rank budget allocation (how many dimensions each adapter layer should use)?

**Conclusion:** Spectral rank policies are competitive with gradient-based policies for rank budget allocation in the tested regime (small compressible encoders on SST-2 and IMDB). The `oht` policy (a spectral allocation method that identifies which dimensions carry the most energy) is the leading spectral variant. However, `proxy_gradient` (a gradient-based comparator that measures how similarly models respond to the same training signal) remains the operational default because it is more stable under resampling. Spectral policies align more closely with ablation-style importance measures than with gradient-style measures — suggesting the spectral signal captures structural importance rather than optimization-trajectory similarity.

**Guardrails:** Do not claim equivalence with adaptive-training methods. Do not generalize beyond encoder/classification regime. Saturated task families (tweet_eval, ag_news) are not informative for this comparison.

**Detail:** [`../strategy/rank_proxy_bounded_validation_summary.md`](../strategy/rank_proxy_bounded_validation_summary.md)

### Ablation proxy expansion — `resolved (bounded)`

**Question:** Which method of measuring component importance should be the operational standard — gradient-based comparison, ablation (removing or weakening components), or rank reduction (keeping only the top-k spectral dimensions)?

**Conclusion:** Gradient-based comparison (`proxy_gradient`) remains the operational standard because it is the most stable under resampling. Ablation-by-attenuation (`attenuate` — weakening components rather than removing them entirely) is retained as an explanatory companion: it reveals *why* structural differences exist, even though it shouldn't drive operational decisions. Rank-reduction soft-ablation (keeping only 75–85% of spectral dimensions) was tested and found degenerate in this regime — it consistently failed to discriminate between models and is paused indefinitely.

**Detail:** [`../strategy/ablation_proxy_resolution_summary.md`](../strategy/ablation_proxy_resolution_summary.md)

### HTSR / edge-gap add-on — `bounded_keep`

**Question:** Should two additional spectral observables — the HTSR-style tail exponent (how quickly singular values decay, adapted from Heavy-Tailed Spectral Rank theory) and the edge-gap ratio (the gap between the first and second singular values, indicating spectral concentration) — be promoted to front-line metrics?

**Conclusion:** Both probes show meaningful sensitivity within the tested regime but lack the robustness needed for front-line use. They detect regime boundaries (e.g., transitions between compressible and non-compressible adapter behavior) but do not consistently outperform the core observables (stable rank, energy concentration) across varied conditions. Keep both available in research outputs as secondary observables. Do not promote to product-facing metrics.

**Detail:** [`../strategy/phase_probe_addon_summary.md`](../strategy/phase_probe_addon_summary.md)

### Merge-aware training monitor — `bounded_keep`

**Question:** Can we monitor adapter compatibility drift *during* training — tracking whether an adapter is becoming more or less compatible with a reference adapter as training progresses?

**Conclusion:** An optional HuggingFace callback that computes merge-audit-compatible metrics during training is operational as a diagnostic prototype. It produces conservative end-of-run trend labels (`toward`, `away`, `mixed`, `inconclusive`) indicating compatibility drift direction. However, there is no evidence yet that this monitoring improves actual training outcomes. It is a visibility tool, not a steering tool — it does not adjust training, and should not be presented as doing so. Reference adapter selection follows the same task-relationship logic as the triage pipeline: same-task preferred, same-family as fallback, cross-task exploratory only.

**Detail:** [`../strategy/merge_aware_monitor_summary.md`](../strategy/merge_aware_monitor_summary.md)

### Over-accumulation line — `keep_exploratory`

**Question:** Can over-accumulation scoring (measuring whether weight updates accumulate in misaligned directions across training) predict merge compatibility?

**Conclusion:** A refined estimator (OA-v2) shows modest overall improvement in correlation with merge outcomes compared to the baseline (OA-v1): Spearman correlation rose from 0.18 to 0.25 across a 30-pair validation set. However, on the specific subset where OA-v2 was designed to excel (high-overlap, low-conflict pairs, n=9), correlation actually inverted to −0.09 — indicating instability in exactly the regime where it should work best. Promotion gates were explicitly failed on 3 of 4 criteria. Source-quality differences are a major confound that the current cohort cannot control for. OA-v1 remains authoritative for any operational merge policy. OA-v2 stays exploratory until a larger, quality-stratified cohort is available.

**Detail:** [`../strategy/over_accumulation_refinement_summary.md`](../strategy/over_accumulation_refinement_summary.md)

---

## Paused / Deprioritized CPU Branches

- **Over-accumulation escalation into policy/execution** (`paused`) — OA-v2 failed promotion gates; no path to operational use without larger validation cohort
- **Rank-reduction ablation expansion** (`paused`) — degenerate in current encoder/compressible regime; does not discriminate between models
- **Additional monitor feature branching** (`deprioritized`) — current monitor is diagnostic-only; no evidence base to justify expanding its scope

---

## Next Proving Grounds

- **GPU-return priority:** decoder-only spectral fingerprinting under controlled training conditions (causal follow-on to the completed [ecosystem census](../technical-report.md#72-decoder-only-ecosystem-census-completed))
  - spec: [`../plans/2026-04-03-decoder-only-spectral-fingerprinting-gpu-return-plan.md`](../plans/2026-04-03-decoder-only-spectral-fingerprinting-gpu-return-plan.md)
- **CPU-theory priority:** analytical spectral geometry of merge operations (extending the formal sketch in [Technical Report §2.3](../technical-report.md#23-the-observablecompatibility-link-a-formal-sketch))
  - spec: [`../plans/2026-04-03-analytical-spectral-geometry-of-merge-operations-plan.md`](../plans/2026-04-03-analytical-spectral-geometry-of-merge-operations-plan.md)
  - **New theorem target (Q7):** spectral partitioning convergence — can we prove that independently trained adapters on the same backbone converge in their high-SV directions? Motivated by independent training-side evidence (Tian, Ledent, & Sun, ICLR 2026) showing 89% high-SV inter-task alignment during multi-task co-training. If the convergence derives from pre-trained spectral gaps (Davis-Kahan), not co-training dynamics, it should hold for independent training too — and this would provide the generative explanation for why spectral triage works. See [Technical Report §2.3.1, §7.5](../technical-report.md) and [THEORY.md §6](../THEORY.md).
- **CPU phase synthesis:**
  - [`../strategy/cpu_phase_established_summary.md`](../strategy/cpu_phase_established_summary.md)
