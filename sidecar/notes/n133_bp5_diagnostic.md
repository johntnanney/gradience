# N133 B-P5 Diagnostic: Why Alignment-Only Triage Failed

**Date:** 2026-04-11
**Parent experiment:** N133 (Decoder-Scale Controlled Merge Triage, Mistral-7B-v0.3)
**Phase 4 status:** PARTIAL — elimination rate 70%, but 3/6 good merges missed (50% recall)

## The headline

**Within the 12 evaluated cross-task pairs, spectral alignment is a statistically significant _anti-predictor_ of merge quality.**

| Statistic | Value |
|---|---|
| Spearman ρ(alignment, max_degradation) | **+0.655** |
| p-value | **0.021** |
| n | 12 |

Higher alignment → *more* degradation. The top-6 alignment pairs (all MNLI-centered)
degraded 19.5%–60.4%. The bottom-3 alignment pairs (ranks 58–60) were the three
numerically best merges.

## The three missed good merges

| pair | alignment | align rank (of 60) | max_deg | deg breakdown |
|---|---|---|---|---|
| `code_s42_vs_gsm8k_s42`         | 0.0363 | **60 / 60** | **+0.000** | code 0 / gsm8k −1% |
| `code_s123_vs_gsm8k_s42`        | 0.0363 | 59 / 60     | +0.015     | code 0 / gsm8k +1.5% |
| `gsm8k_s42_vs_summarization_s123` | 0.0371 | 56 / 60   | +0.070     | gsm8k +7% / summ 0 |

The best merge in the entire eval set had the **lowest** alignment of any cross-task
pair tested.

## The six "retained" pairs (top 30% by alignment) — all BAD

| pair | alignment | rank | max_deg | deg breakdown |
|---|---|---|---|---|
| `mnli_s42_vs_sst2_s42`       | 0.0494 | 1 | +0.255 | mnli −25.5 / sst2 −3 |
| `mnli_s123_vs_sst2_s42`      | 0.0492 | 2 | +0.250 | mnli −25.0 / sst2 −3.5 |
| `mnli_s42_vs_sst2_s123`      | 0.0485 | 3 | +0.250 | mnli −25.0 / sst2 −4.5 |
| `mnli_s123_vs_sst2_s123`     | 0.0484 | 4 | +0.195 | mnli −19.5 / sst2 −4.5 |
| `mnli_s123_vs_squad_s42`     | 0.0447 | 5 | **+0.604** | mnli −26.5 / squad **−60.4** |
| `mnli_s123_vs_squad_s123`    | 0.0445 | 6 | +0.495 | mnli −18.5 / squad −49.5 |

## Why alignment-only triage fails here

### 1. Dynamic range is minuscule

Across all 60 cross-task pairs the mean-alignment values span
**[0.0363, 0.0494]** — a total range of 0.0131, coefficient of variation 8.3%.
There is essentially no signal in the alignment magnitude to rank from.

What little variance exists is organized by a confound: **classification-style
adapters (MNLI, SST2, squad, summarization) land higher in the alignment ranking
than generation-style adapters (code, gsm8k)**, because they share more
common-subspace structure (English semantic features) regardless of pair-specific
compatibility. Alignment is ranking by "both adapters look like LM classifiers,"
not by "these two adapters are compatible to merge."

### 2. The "good merge" criterion is contaminated by metric artifacts

The `max_deg < 0.10` threshold looks sensible but is extremely lenient when
either side has a degenerate baseline:

| task | source accuracy | interpretation |
|---|---|---|
| code | **1.000** | at ceiling — cannot degrade on this eval |
| summarization | **1.000** | saturated metric (same 100% on all merges, not a real signal) |
| gsm8k | 0.23–0.32 | floor — 10% rel drop = 2–3 absolute points, within sampling noise |
| mnli | 0.97 | real dynamic range — every percentage point is measurable |
| sst2 | 0.98–0.99 | near-ceiling but with room to drop |
| squad | 1.000 | at ceiling — unusual, suggests partial_match metric artifact |

**All three "missed good" merges are merges where gsm8k is one side** and the
other side is code or summarization (both pinned at 1.0). These aren't "good
merges" in the sense of preserving combined functionality — they're
**immovable-metric merges**: the 1.0 side literally cannot move, and gsm8k's
low baseline makes small absolute drops fall under the threshold.

By contrast, the MNLI merges actually retain 70%–78% MNLI accuracy (down from
97%) while preserving SST2 at ~95%. In absolute terms these are the pairs closest
to being usable blended adapters, but they cross the 10% relative threshold
because MNLI has the room to move.

### 3. The two failure modes interact

The confound above plus the metric artifact produces a perfect anti-correlation:

- **High-alignment pairs** are MNLI-heavy → MNLI has real dynamic range →
  merge degradation is measurable and large.
- **Low-alignment pairs** are code/gsm8k-heavy → one or both sides are at a
  ceiling or floor where degradation is unmeasurable → they "pass" the
  threshold by default.

Neither mechanism is about the geometric compatibility of the merge.

## What this means for the merge-triage program

1. **Alignment magnitude alone is not a decoder-scale triage metric.** The
   dynamic range is too small and the ranking is dominated by a task-style
   confound. This falsifies the simplest version of the N133 B-P5 prediction.

2. **We still don't know whether a _better_ geometric metric works**, because
   the eval-set design cannot distinguish "metric genuinely uninformative" from
   "metric swamped by source-task saturation". The 12 evaluated pairs include
   exactly 0 cases where both source baselines are in the 0.5–0.9 sweet spot
   where a merge could meaningfully preserve or destroy signal on both sides.
   B-P5 needs a re-run on a task set where every source baseline is in measurable
   range.

3. **The per-module finding from Phase 2 is unaffected.** O>V>Q>K same-task SNR
   separation (7.23× → 1.75×) was measured on adapter-pair geometry directly,
   not on merge evaluation. That result still stands and is what should seed
   the next triage-metric attempt.

4. **The B-P6 script tested the wrong quantity.** It compared Q/K vs V/O _erank_
   (and found no asymmetry). The Phase 2 finding was about per-module
   _alignment SNR_, not erank. Re-running B-P6 with the correct quantity is a
   separate follow-up.

## Suggested next metric for a re-run

Instead of mean same-task alignment, build a merge-risk score that:

- Uses **direction-aware** overlap between the two adapters (signed inner
  products of top-k singular directions), not magnitude-only alignment
- Weights by **per-module importance** (O ≫ V ≫ Q ≫ K from Phase 2) and
  **layer depth** (layer 31 has 1.83× the SNR of layer 0)
- Normalizes against **per-adapter erank** (low-erank adapters are the ones
  that get crushed in merges — MNLI's 6.04 erank is the lowest and it shows
  up as the loser in every MNLI-containing merge)

An intuition-check version: _"predict max_degradation from a weighted sum of
(module O direction overlap × depth weight × min erank across pair)"_. This
should recover the ordering the alignment-only metric inverted, because:

- MNLI pairs have low min-erank → flagged as high-risk
- code/gsm8k pairs have high erank on both sides → flagged as low-risk
- within MNLI pairs, the squad one has the worst O-module overlap → ranks worst
  (matches the +60% squad degradation)

All the data needed for this follow-up is in `sidecar/data/n133/pod_pull/` —
no GPU required.

## Composite risk score results (follow-up)

Ran `scripts/n133_bp5_composite_risk.py` on the pulled artifacts. Tested
10 candidate risk scores against `max_degradation` on the 12 evaluated
cross-task pairs.

### Spearman ρ (risk, max_degradation), n=12

| variant | ρ | p | triage recall (retain 6 lowest-risk) |
|---|---|---|---|
| **mean_alignment** | **+0.6550** | 0.021 | **3 / 3 good** |
| inv_min_erank | +0.3186 | 0.313 | **3 / 3 good** |
| OVmix_x_inv_erank | +0.2207 | 0.491 | **3 / 3 good** |
| OVmix_depth | +0.0736 | 0.820 | 2 / 3 good |
| O_depth_x_inv_erank | +0.1121 | 0.729 | 2 / 3 good |
| z_sum(O_depth, 1/min_erank) | +0.1051 | 0.745 | 2 / 3 good |
| O_quad_x_inv_erank | +0.1051 | 0.745 | 2 / 3 good |
| O_mean | −0.0911 | 0.778 | 1 / 3 good |
| O_depth | −0.1401 | 0.664 | 1 / 3 good |
| O_quad | −0.1821 | 0.571 | 1 / 3 good |

### The original B-P5 result had the sign wrong

The original B-P5 script retained the **top 30%** by alignment, assuming higher
alignment → more compatible merge. That was backwards on this data:
the top-6 aligned pairs were the six worst merges.

**Reversing the direction** (retain the _bottom_ 6 by alignment as the
"safe to merge" set) recovers **all 3 good merges perfectly.** So "alignment
as anti-risk" is a usable triage rule on this 12-pair selection, not a
failure mode. The original write-up describing alignment as a 50% recall
ranker is an artifact of the sign convention, not of alignment being
uninformative.

### But the apparent success is still a task-family confound

The 12 evaluated pairs split cleanly into:

- **6 MNLI-centered pairs** — all bad (deg 19.5%–60.4%), all highest-alignment,
  all with `min_erank ≈ 6.0` (MNLI has the lowest erank: 6.005–6.075)
- **6 code/gsm8k/summarization pairs** — all "good" by the threshold, all
  lowest-alignment, all with `min_erank ≥ 6.5`

Any metric that distinguishes "has MNLI on one side" from "doesn't" scores
perfectly on this sample. Both `mean_alignment` and `inv_min_erank`
achieve 3/3 recall for exactly this reason, and the "composite" combinations
don't improve on either alone because they're measuring the same binary
partition.

### The O-module-only variants don't transfer

Despite Phase 2 showing a clean O > V > Q > K per-module SNR gradient
(7.23× → 1.75×) on the adapter-pair geometry itself, conditioning the
alignment score on O-module layers only actually **hurts** triage performance
(1/3 good merges retained). Candidate explanations:

1. **Per-module SNR was a same-task-vs-cross-task signal**, not a
   within-cross-task ranking signal. Phase 2 compared the 6 same-task pairs
   (n=6) against the 60 cross-task pairs (n=60) and found the clearest
   separation in O-module. That separation can be strong while the
   _within-cross-task_ variance in O-module is uninformative about
   which cross-task pair merges best.

2. **The Phase 2 result is a property of source adapters**, not of their
   pairwise geometric interaction. The O-module carries more task-specific
   structure within each adapter — but that doesn't automatically make
   its pair-level alignment a better merge predictor.

3. **MNLI adapters happen to have unusually high O-module similarity with
   other adapters** (since they encode general English classification
   structure), making O-alignment inherit the MNLI-vs-not-MNLI confound
   rather than escape it.

### Interim verdict

- The composite scores do not meaningfully improve on alignment or
  min-erank alone.
- With a sign flip, alignment-only triage achieves perfect recall on
  the 12-pair selection — but this is a binary-partition artifact and
  does not generalize beyond MNLI-vs-not-MNLI.
- `inv_min_erank` is a cleaner mechanistic story (low-erank adapters
  get crushed in merges) and achieves the same 3/3 recall, but has
  the same confound.
- **The real test of any merge-risk metric at decoder scale needs a
  different task set** — one where pairs are not perfectly partitioned
  by task family, and where both source baselines are in measurable
  range (no code/summarization ceilings, no gsm8k floor).

Artifacts: `sidecar/data/n133/bp5_composite_risk.json`,
`sidecar/data/n133/bp5_composite_risk_plot.png`.

## Confound check: compressed-erank subset + full-scale ranking

Ran `scripts/n133_bp5_confound_check.py` to directly test whether the
apparent 3/3 triage wins of `mean_alignment` and `inv_min_erank` were
task-cluster artifacts.

### Test 1: inv_min_erank on a compressed-erank subset

Drop summarization (erank 9.09) and sst2 (erank 5.21), which are the two
outliers at the extremes. Remaining adapters span erank **6.005 to 7.653** —
a much tighter band. This filters the evaluated pairs down to 7
(2 good, 5 bad).

| metric | Spearman ρ | p | triage recall |
|---|---|---|---|
| **inv_min_erank** | **+0.0551** | **0.907** | **1 / 2** |
| **mean_alignment (corrected sign)** | **+0.8571** | **0.014** | **2 / 2** |

**`inv_min_erank` collapses to noise the moment the task-family outliers
are removed.** The previous 3/3 was pure MNLI-vs-rest detection: MNLI had
the lowest erank in the full 12-adapter set, so "low min_erank" meant
"has MNLI". Within the compressed subset, min_erank ranges from 6.005
(MNLI) through 6.825 (gsm8k) and back up to 7.65 (squad), but the pairs
that merge well are in the **middle** of the erank range (code+gsm8k),
so erank ordering is unrelated to merge quality.

`mean_alignment` actually gets _stronger_ in the compressed subset
(ρ +0.655 → +0.857). But look at what's driving that: the 4 `code↔gsm8k`
pairs are all tied at alignment **0.0363** (identical to 4 decimal
places). Alignment literally cannot distinguish among them — yet they
split 2-good / 2-bad on merge eval. The Spearman ρ is high because
those 4 tied pairs collectively have much lower max_deg than the 3
MNLI/squad pairs at higher alignment. It's a **two-cluster separation,
not a within-cluster ranking**.

Neither metric has sub-cluster resolution on this data.

### Test 2: corrected-sign alignment triage on all 60 cross-task pairs

If alignment-as-anti-risk generalizes, the 18 "safest" pairs it would
schedule next should reflect genuine compatibility, not a task-cluster
proxy. Result:

**Task appearances in the 18 safest pairs:**

| task | count in top-18 safest | coverage of that task's cross-task pairs |
|---|---|---|
| gsm8k | 12 | 12 / 20 → 60% |
| code | 10 | 10 / 20 → 50% |
| summarization | 6 | 6 / 20 → 30% |
| squad | 6 | 6 / 20 → 30% |
| sst2 | 2 | 2 / 20 → 10% |
| **mnli** | **0** | **0 / 20 → 0%** |

**Zero MNLI pairs in the top-18 safest.** Extending to top-30:

| task | count in safest 30 |
|---|---|
| gsm8k | **20 / 20 (100%)** |
| code | 14 / 20 (70%) |
| summarization | 8 / 20 (40%) |
| squad | 8 / 20 (40%) |
| sst2 | 6 / 20 (30%) |
| mnli | 4 / 20 (20%) |

**Every single gsm8k-containing cross-task pair lands in the safest
half.** The corrected-sign alignment ranking is essentially a
has-generation-task classifier. The four cross-task pairs involving
code are:
`code_*_vs_gsm8k_*` — all 4 in the top 4 positions with alignment
0.0363 (tied).

This would predict that `code_vs_summarization`, `code_vs_squad`,
`gsm8k_vs_summarization`, etc., are all "safe to merge" — and they
probably are, because one or both sides are at a metric ceiling (code,
summarization, squad) or a floor (gsm8k). Alignment is not detecting
merge compatibility; it is detecting which pairs have one or more
immovable metrics.

### Honest negative-result summary

1. **`inv_min_erank` was a task-family outlier detector, not a geometric
   merge-risk metric.** It fails completely (ρ ≈ 0) when the task set
   doesn't have a single adapter at the erank floor or ceiling.

2. **`mean_alignment` (corrected sign) has no within-cluster resolution.**
   Its apparent 3/3 win was a two-cluster separation (MNLI-heavy vs
   generation-heavy), and at full scale it's a has-generation-task
   classifier. It cannot distinguish pairs within the same cluster
   (four `code↔gsm8k` pairs all at 0.0363).

3. **The O-module-specific scoring doesn't help.** Phase 2's O>V>Q>K
   SNR finding is about same-task-vs-cross-task separation of source
   adapters, not within-cross-task pair ranking.

4. **We do not currently have a decoder-scale merge-risk metric.** The
   closest usable rule ("retain pairs with lowest pooled alignment")
   only works because the selection happens to match the metric
   artifact.

### What a real B-P5 re-run would need

- **Task set with no metric ceilings or floors.** Every source baseline
  in [0.50, 0.85]. This rules out code (1.0), summarization (1.0),
  squad (1.0), gsm8k (0.23). Candidates: cnn/dailymail ROUGE, hellaswag,
  piqa, boolq, winogrande, arc-challenge — all have meaningful dynamic
  range on Mistral-7B without ceiling.
- **Task set that is not binary-partitioned by family.** Include multiple
  tasks in each loose family so pair composition is not 1:1 with task
  membership.
- **More adapters per task.** n=2 seeds per task gives us no way to
  separate within-seed variance from across-task variance, which
  compounds the single-cluster problem.

Until those conditions are met, further work on the 12-pair selection
will continue producing cluster-proxy wins rather than metric validation.

Artifacts: `sidecar/data/n133/bp5_confound_check.json`,
`scripts/n133_bp5_confound_check.py`.

## Data used

- `audits/pair_alignment_summary.json` — all 66 pair alignments
- `merges/merge_eval_summary.json` — 18 evaluated merges with source + merged scores
- `evals/*_source_eval.json` — 12 source-adapter baselines
