<!-- ANON: project-number prefix stripped from title ("N134 —" removed). Original title retained in git history for camera-ready restoration. -->
# Decoder-Scale Controlled Merge Triage: Pre-Registration Document (v3.1)

**Status:** specification, v3.1. No data collected at time of writing. No analyses performed. Any deviation between this document and the final report is a null, not an opportunity.

<!-- ANON: version-history block rewritten: project-number identifiers stripped from each amendment entry, substance preserved. -->
**Version history.**
- v1 (drafted pre-literature-review): four-confound design, three-method scheduled comparison (this paper's score / KnOTS / TSV).
- v2 (April 18 2026): scheduled comparison expanded to four methods to include Singular Value Calibration (SVC, Li et al., arXiv:2602.05536). Audit-schema requirement made explicit. External positioning citations added (OSRM, Zhang & Zhou ACL 2025; TARA-Merging, Jeong et al. arXiv:2603.26299). No change to primary hypothesis, task set, training protocol, or decision rule.
- v3 (April 19 2026): positioning updated to reflect the adjacent mergeability-prediction research program (Rahamim et al. 2026, Zhou et al. 2026, Bolton et al. 2026) and the decoder-LLM-scale evaluation of existing merge methods (arXiv:2511.21437). §1 expanded. §8 "H1 fails" extended with a fourth candidate explanation. Appendix A extended with one additional declared unknown. No changes to H1 score, decision rule, confound constraints, task set, seeds, training protocol, statistical protocol, four-method comparison set, or execution plan.
- v3.1 (this document, April 19 2026): GPU platform specification changed from single H100 to single RTX 6000 Ada 48GB on commercial cloud. Resource estimate in §9.4 updated. External dependencies in §9.3 unchanged. No changes to experimental design, H1 score, decision rule, confound constraints, task set, seeds, training protocol, statistical protocol, comparison set, required artifacts, or execution plan.

<!-- ANON: supersession line rewritten per checklist §2.1(3): precursor study named descriptively, internal path reference removed. -->
**Supersedes:** nothing. Complements the precursor confound-diagnostic study on the same adapter-pair substrate (see paper §5.3 for that study's task-boundary detection results).

---

<!-- ANON: §1 heading renamed from "What N134 Is For" to "What This Study Is For" per checklist §2.1(4). -->
## 1. What This Study Is For

<!-- ANON: paragraph opens with project-number-free description of the precursor study's outcome; "Gradience's measurements" rewritten descriptively. -->
The precursor study established that spectral alignment separates same-task from cross-task adapter pairs at decoder scale (B-P1, B-P2 confirmed on Mistral-7B: ratio 3.06×, zero false positives on 66 pairs). It failed to establish that spectral alignment predicts *within-family* merge outcome: the pre-specified triage was indistinguishable from a task-family classifier under FAMILY_B residualization (ΔR² ≤ 0.015 for every tested aggregation, against a family-only R² of 0.97). The precursor's confound catalogue documents this as a four-stage cascade (sign error → metric-range saturation → task-family partition → composite overfitting).

The present study is the experiment the precursor confound catalogue specifies. Its purpose is to deliver a decision, not a discovery: either the pre-registered primary score clears the H1 gate under conditions that defeat C1–C4, or the spectral-triage-for-per-pair-prediction hypothesis is retired at decoder scale. The B-P1 / B-P2 task-boundary result is already strong; the present study is about whether there is any remaining geometric signal *inside* the cross-task region that the spectral measurement approach under test can read.

**Position in the 2025–2026 literature.** The spectral-methods-for-LoRA-merging literature has expanded substantially in the twelve months prior to this spec: KnOTS (Stoica et al. 2024), TSV (Gargiulo et al. 2025), SVC (Li et al. 2026), TARA-Merging (Jeong et al. 2026), Core Space Merging (2509.17786), OSRM (Zhang & Zhou ACL 2025), Iso-CTS, STAR, and others. All are merge-*execution* methods: they take the set of adapters-to-merge as given and improve the result. The present study tests a triage claim — whether spectral geometry predicts, in advance, which cross-task pairs *should* enter the execution pipeline. That question is orthogonal to every execution method cited above; each of them, if successful, operates on the retained set that a triage decision produces.

<!-- ANON: §1 positioning paragraph left substantively intact; project-number references replaced with descriptors (the present study / the spectral-triage-hypothesis). -->
**Position relative to the mergeability-prediction literature.** A distinct research program on mergeability prediction has emerged in parallel (Rahamim et al. 2026, arXiv:2601.06672; Zhou et al. 2026, arXiv:2601.22285; Bolton et al. 2026, arXiv:2601.09473). Rahamim et al. argue that mergeability is an intrinsic property of individual adapters, primarily governed by base-model prior knowledge, and not partner-dependent. Zhou et al. (GLADIA/Sapienza, same group as TSV) test 190 task pairs across four merge methods on ViT-B/32 vision classifiers and use linear optimization over interpretable pairwise metrics to predict post-merge accuracy; they find that subspace overlap and gradient alignment are "foundational, method-agnostic prerequisites" while individual weight-space metrics achieve |r| < 0.2 (effective rank) or |r| < 0.1 (task vector geometry), and that activation-based metrics achieve the strongest individual correlations (up to r = 0.572 for TSV). The present study differs from this program along three axes: (a) scale — existing work tests vision classifiers, the present study tests at decoder-LLM scale (Mistral-7B); (b) epistemology — Zhou et al. optimize linear combinations to describe post-merge accuracy, the present study tests a single pre-registered score against a sharp decision rule with confound-defeat built into the design; (c) framing — Rahamim et al. argue mergeability is intrinsic, the present study's pairwise H1 operationally tests the opposite hypothesis. The programs are complementary: Zhou et al.'s confirmation that subspace overlap is a method-agnostic prerequisite supports the premise of the spectral-triage design under test; the present study asks whether a specific pre-registered instantiation of that prerequisite clears a decoder-scale triage threshold.

**Why decoder-scale triage matters now.** Hitit et al. (arXiv:2511.21437, February 2026) evaluate six merge methods across four open-weight LLMs (Llama 3.2 3B, Qwen3 4B, Llama 3.1 8B, Qwen3 8B), twelve fine-tuned checkpoints each, sixteen standard benchmarks. Their finding is that subspace-based methods (TIES, Iso-C, TSV-M) produce large parameter-space displacements (L₂ often 100–300 for 3B–4B models, exceeding 300 for 8B) that correlate with catastrophic forgetting; only Task Arithmetic reliably produces constructive interference at decoder scale. The combination of this result with Zhou et al.'s "subspace overlap is foundational but individually weak on vision" suggests that triage — deciding whether to merge at all — may be the binding operational constraint at decoder scale in a way it is not at vision-classifier scale. The present study tests whether a weight-space spectral triage can meet that need under confound control.

## 2. Confounds Being Defeated

Each confound below corresponds to a concrete design choice. Failure to meet any specified threshold before the experiment begins is disqualifying; these are not targets to adjust toward after the fact.

<!-- ANON: each confound's "N134 constraint:" label changed to "Constraint:" per checklist §2.1(6); in-paragraph references to the precursor study generalized. -->
**C1 — Source-metric dynamic range.** In the precursor study, four of six source accuracies saturated at ceiling (≥ 0.97) and one sat near floor (GSM8K, 0.23–0.32). Degradation was bounded below by measurement-range effects independent of merge geometry. **Constraint:** every source adapter must achieve accuracy in the range **[0.70, 0.90]** on its own held-out validation set. Adapters falling outside this band are re-trained at adjusted rank or data fraction until they land inside it; adapters that cannot be brought into the band after two attempts are excluded and the task replaced from the reserve list (§3.2).

**C2 — Task-family non-partition.** In the precursor, six tasks partitioned into two coarse families (discriminative with headroom vs. generative near ceiling), and alignment-based scores operated as family classifiers. **Constraint:** ≥ 5 distinct task families represented, with no family containing more than ~25% of tasks. Tasks must be pilot-tested for adapter erank and required to span a continuous erank distribution (no binary high/low split). Family labels are fixed before training; no post-hoc regrouping.

**C3 — Within-task variance.** In the precursor, 2 seeds per task yielded 1 same-task pair per task — a 6-observation estimate of same-task alignment variance. **Constraint:** **≥ 3 seeds per task**, yielding C(3,2) = 3 same-task pairs per task as the minimum for within-task variance estimation.

**C4 — No post-hoc fitting.** In the precursor, ten composite risk scores were constructed and tested after the primary B-P5 failed. Three matched the 3/3 baseline recall by tiebreak; none survived diagnostic confound checks. **Constraint:** the primary H1 score is specified in §4 with a single decision rule. Any secondary or exploratory scores are labeled as such, reported with their own independent thresholds, and carry no evidential weight for the primary question. Scores not pre-registered here cannot be computed.

## 3. Experimental Design

### 3.1 Model

<!-- ANON: "same checkpoint as N133" → "same checkpoint as the precursor"; follow-up study identifier stripped. -->
**Mistral-7B-v0.3** (HuggingFace: `mistralai/Mistral-7B-v0.3`), same checkpoint as the precursor study. Same-backbone design is deliberate: the primary question is whether the precursor's null survives under unconfounded task selection, not whether it generalizes to a new decoder. A second decoder (Llama-3-8B) is scheduled as a follow-up study conditional on this study's H1 outcome.

### 3.2 Task set

Eight tasks selected for anticipated baseline in [0.70, 0.90] under r=16 LoRA on Mistral-7B-v0.3, with family diversity:

| # | Task | HuggingFace dataset | Family | Anticipated baseline |
|---|------|--------------------|--------|---------------------|
| 1 | ARC-Challenge | `allenai/ai2_arc` (challenge) | science QA | ~0.75 |
| 2 | HellaSwag (5k subset) | `Rowan/hellaswag` (5k seeded subsample) | commonsense completion | ~0.80 |
| 3 | WinoGrande | `winogrande` (winogrande_xl) | coreference/commonsense | ~0.70 |
| 4 | OpenBookQA | `openbookqa` (main) | knowledge QA | ~0.75 |
| 5 | CommonsenseQA | `tau/commonsense_qa` | commonsense QA | ~0.75 |
| 6 | PIQA | `piqa` | physical commonsense | ~0.80 |
| 7 | SIQA | `social_i_qa` | social commonsense | ~0.75 |
| 8 | BoolQ | `boolq` | reading comprehension | ~0.80 |

Five families represented; no family exceeds two tasks.

**Reserve list** (drawn from only if pilot replacement is triggered): LogiQA, ANLI-R1, Cosmos QA.

<!-- ANON: "Gradience audit" → "spectral audit" (the descriptor the paper uses). -->
**Pilot gate.** Before full training proceeds, a single-seed pilot per task confirms (a) baseline in [0.70, 0.90] on the task's held-out validation set and (b) adapter erank computed via the spectral audit falls in a continuous range across the 8 tasks with no bimodal cluster structure (assessed by Hartigan's dip test at $p > 0.10$ on the 8-point erank distribution). Any task failing (a) is re-trained with adjusted rank (try r=8, then r=32) or data fraction (try 50%, then 25%). After two retraining attempts, a failing task is replaced from the reserve list before Seeds 2–3 begin training.

### 3.3 Seeds

**Three seeds per task** (42, 123, 456), yielding:

- **24 adapters** total
- **24 same-task pairs** = 8 tasks × C(3,2) = 8 × 3
- **252 cross-task pairs** = C(24,2) − 24

Full spectral audit runs on all 276 pairs (cheap).

<!-- ANON: "N134_PAIR_SEED" env-var name kept (it is a technical identifier, not a project brand, and is referenced by scripts in this bundle); pair-budget comparison against precursor retained with descriptive reference. -->
**Merge evaluation sample.** 69 merges total:

- All **24 same-task pairs** evaluated (triage calibration).
- **45 cross-task pairs** evaluated: 3 pairs drawn from each of the C(8,2) = 28 cross-task-type cells. Because 3 × 28 = 84 > 45, not all cells get 3 observations; cells are sampled in order using the committed RNG (`PAIR_SEED = 134`) until 45 pairs are drawn, with a minimum of 1 pair per cell enforced (28 minimum), and the remaining 17 pairs distributed to cells selected by the same RNG without replacement.

This is roughly 4× the precursor's evaluated cross-task budget (12 pairs) and gives each cross-task-type cell ≥ 1 observation.

### 3.4 LoRA and training protocol

| Parameter | Value |
|-----------|-------|
| Rank | 16 (retry at 8 or 32 if pilot fails accuracy band) |
| Target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj` |
| Alpha | 32 (2× rank) |
| Optimizer | AdamW, lr=2e-4 |
| Warmup | 6% |
| Precision | bf16 |
| Checkpoint cadence | every 50 steps, first 500 steps + final |
| Training duration | scaled per task to achieve [0.70, 0.90] band; not a fixed step count |

Final checkpoint used for all structural audits.

### 3.5 Audit output schema (schema version v2.1)

<!-- ANON: reference to the precursor's Phase-2 limitation rewritten descriptively; schema-file path stripped to bundle-relative. -->
**This is a hard requirement.** The precursor's Phase-2 audit persisted only scalar SV-weighted alignment and per-layer singular values, which blocked the scheduled KnOTS / TSV / SVC comparison. The present study **must** persist per-layer U and V orthonormal factors for every LoRA layer.

Required fields in `audit_v2_1.schema.json` (committed with the bundle's analytical artifacts):

```
{
  "adapter_id": str,
  "base_model": "mistralai/Mistral-7B-v0.3",
  "layers": [
    {
      "layer_idx": int,
      "module": "q_proj" | "k_proj" | "v_proj" | "o_proj",
      "rank": int,                              # nominal rank (16)
      "singular_values": [float, ...],          # length = nominal rank
      "u_factor": [[float, ...], ...],          # shape (d_out, rank)
      "v_factor": [[float, ...], ...],          # shape (rank, d_in)
      "stable_rank": float,
      "energy_rank_90": int,
      "entropy_effective_rank": float,
      "frobenius_norm": float
    }
  ],
  "meta": {
    "task": str,
    "seed": int,
    "training_steps": int,
    "final_val_accuracy": float
  }
}
```

U and V factors are stored as float32 to keep file size manageable (each adapter ≈ 32 layers × 4 modules × (4096 × 16 + 16 × 4096) × 4 bytes ≈ 32 MB per adapter; 24 adapters ≈ 0.77 GB total).

## 4. Primary Hypothesis and Decision Rule

### H1 (primary)

**Score:** O-module depth-weighted alignment, defined as

$$
S_{\text{H1}}(\text{pair}) \;=\; \frac{\sum_{\ell=1}^{L} w_\ell \cdot \alpha^{(O)}_\ell}{\sum_{\ell=1}^{L} w_\ell}
$$

where $\alpha^{(O)}_\ell$ is the SV-weighted alignment on the O-projection of layer $\ell$, $L = 32$, and $w_\ell = \ell / L$ (linear depth weight, deeper layers weighted more heavily).

<!-- ANON: rationale paragraph rewritten to replace "N133" project-number references with descriptive "precursor" references; substantive numerical claims retained verbatim. -->
**Rationale.** The precursor's Phase-2 finding that O-projection alone gives 7.23× same/cross separation (vs. 3.06× aggregate) and the bonus layer-depth trend showing same/cross ratio rising from 2.32× at layer 0 to 4.24× at layer 31. The score encodes both findings. Crucially, the precursor's B-P5.b diagnostic showed this score **flipped sign** between raw and family-residualized correlations on the precursor task set — meaning its apparent signal was family-carried. If O-depth clears H1 on the present study's unconfounded task set, the flip disappears and the score is validated. If it does not, the hypothesis that per-module weighting recovers signal is retired.

**Outcome variable:** `max_degradation`, defined as $\max(s_A - m_A, s_B - m_B)$ where $s_A, s_B$ are source accuracies on their own tasks and $m_A, m_B$ are the merged model's accuracies evaluated on task A and task B respectively, under 0.5/0.5 linear merge. Consistent with the precursor's outcome definition.

**Decision rule.** H1 clears if **both**:

1. Spearman partial $\rho(S_{\text{H1}}, \text{max\_degradation})$, residualized on FAMILY_B task-family-pair dummies, is ≥ **0.50** with $p < 0.05$; and
2. OLS $\Delta R^2$ of $S_{\text{H1}}$ over a FAMILY_B dummy-only baseline is ≥ **0.10**.

Sign must be correct: higher O-depth alignment predicts higher degradation. Right magnitude with wrong sign is a null, not a partial confirmation.

**FAMILY_B partition** (fixed pre-experiment): each of the 8 tasks assigned to its family from the table in §3.2. Cross-task-pair family label is the unordered pair of task families (e.g. `{science_qa, commonsense_qa}`). With 5 families, there are C(5,2) + 5 = 15 possible pair-families; the 45 evaluated cross-task pairs will sample a subset of these.

**If H1 fails.** No composite score is constructed. The null is the result. Any composite analysis conducted after seeing H1's fate is reported as exploratory-non-evidential under the C4 constraint.

### Confirmatory replications (non-gating, reported alongside)

- **B-P1 replication.** 0/24 same-task pairs fall below the same/cross midpoint threshold; 0/252 cross-task pairs rise above. Confirmed if both counts are 0.
- **B-P2 replication.** Same/cross mean alignment ratio ≥ 2.0× with $p < 0.001$ by Welch's t-test.
- **B-P4 first-half replication.** ANOVA of per-adapter mean erank across tasks, $p < 10^{-6}$.

Failure of any of these is reported as a substantial finding requiring diagnosis — most likely pointing to task-set structural similarity not caught by the pilot.

### Secondary exploratory measures (non-evidential, logged only)

<!-- ANON: references to precursor project numbers (N132/N133) replaced with descriptive references. -->
These are *logged* but do not bear on H1. Any pattern observed is a hypothesis generator for a future study, not a confirmation.

- Per-module alignment heterogeneity (Q/K vs V/O erank and alignment asymmetry).
- Within-module C_k → alignment partial correlation (further replication of the precursor's within-module null result).
- Layer-depth trend in same/cross ratio (replication of the precursor's bonus finding).
- Dump of all 10 precursor composite risk scores evaluated on this study's data, for comparison purposes only.

## 5. Statistical Protocol

<!-- ANON: reference to "N133's Simpson's paradox" replaced descriptively. -->
- **Significance threshold:** $\alpha = 0.05$ for individual tests. No multiple-comparison correction applied to H1 (it is a single pre-registered test). Secondary measures are uncorrected but explicitly labeled as exploratory.
- **Bootstrap:** block bootstrap for confidence intervals on H1's partial $\rho$, with blocks defined by task-family-pair cell (respecting the dependence structure that created the precursor's Simpson's paradox). **5,000 resamples.**
- **Outlier policy:** no exclusions of evaluated pairs on statistical grounds. Pairs may be excluded only for infrastructure failure (training crash, evaluation timeout) and must be reported with cause.
- **Tied-pair handling.** The precursor found 38/60 pairs structurally tied at 4-decimal precision of mean_alignment; the metric's resolution is ≈ 2×10⁻³. H1 is computed at float64 precision; pairs with $|S_A - S_B| < 10^{-6}$ are treated as genuinely tied and ranked by bootstrap-median tiebreak. The number of tied clusters is reported.

## 6. Scheduled Comparison: Four-Method Head-to-Head

<!-- ANON: precursor reference rephrased; "blocked from N133" → "blocked in the precursor". -->
With U/V matrices persisted (§3.5), the comparison blocked in the precursor becomes tractable. **Four methods** are scored on the same 45 evaluated cross-task pairs:

<!-- ANON: "Gradience" method-name column entry replaced with "S_H1 (this paper)" per checklist §2.3(5). -->

| Method | Reference | Mechanism | What the method's triage signal is |
|--------|-----------|-----------|-----------------------------------|
| S_H1 (this paper) | this work | O-module depth-weighted SV-weighted alignment | $S_{\text{H1}}$ (§4) |
| KnOTS | Stoica et al. 2024 (arXiv:2410.19735) | joint SVD rotation to shared basis | per-pair norm of projected interference term |
| TSV | Gargiulo et al. 2025 (arXiv:2412.00081) | task-singular-vector whitening | per-pair whitened-interference magnitude |
| SVC | Li et al. 2026 (arXiv:2602.05536) | singular-value rescaling to counter over-accumulation | per-pair SV-inflation index (sum of shared-direction SVs / sum of all SVs) |

Each method is applied as a **triage** (not as an execution): rank the 45 cross-task pairs by the method's risk score, select the "safe" lowest-half (N=22 pairs), measure mean `max_degradation` in the retained set. Methods are compared on retained-set mean degradation with block-bootstrap CIs (same 5,000-resample, family-pair-blocked scheme as §5).

<!-- ANON: pre-registered direction sentence rewritten: "Gradience is at least as good as..." → "The present study's score is at least as good as..." -->
**Pre-registered direction:** "The present study's score is at least as good as KnOTS, TSV, and SVC on the triage objective on this unconfounded task set." This prediction is descriptive — it does not gate H1, and the primary H1 decision stands regardless of this comparison's outcome.

**SVC-specific note.** SVC is designed for portfolio-scale merging (k ≥ 3 adapters combined) where over-accumulation compounds; adapting it to pairwise triage requires computing its SV-inflation index on each 2-adapter pair and using it as a ranking signal. Coding agents must document any deviation from the paper's published procedure introduced by this adaptation.

<!-- ANON: "v2.0 integration roadmap" reference stripped (implementation-product detail). -->
**Not included:** TARA-Merging (requires preference signal / pseudo-loss evaluation; fundamentally different intervention class, not a fair triage comparison). Core Space Merging (operationally similar to KnOTS; not in scope for the present study).

<!-- ANON: report reference stripped. -->
**External citation:** OSRM (Zhang & Zhou, ACL 2025, arXiv:2505.22934) is cited in the paper as a training-time intervention that extends the KnOTS observation that weight-space subspace orthogonality does not guarantee non-interference. OSRM is not comparable as a triage method on pre-existing adapters. Its relevance is conceptual: it motivates a future *activation-informed* overlap diagnostic that is explicitly out of scope for the present study.

## 7. Deviations

Any methodological change made after this spec is committed:

- **Before any data collection begins:** version-controlled with rationale in this document's "Version history" header. Does not affect evidential weight.
- **After data collection begins:** disqualifying for the primary H1 test. The test's result under the modified protocol is reported as exploratory.

<!-- ANON: "the N133 post-hoc composite search" → descriptive. -->
This is strict, and deliberately so. The precursor's post-hoc composite search is the failure mode the present study is designed to make impossible.

## 8. What Each Outcome Means

<!-- ANON: outcome-interpretation blocks rewritten: project-identifier references to N134/N135/Gradience replaced with descriptive phrases. -->
**H1 clears, replications hold.** The spectral-triage-for-per-pair-prediction hypothesis is confirmed at decoder scale on an unconfounded task set. The spectral measurement approach under test generalizes beyond task-boundary detection. A cross-decoder replication on Llama-3-8B is licensed as a follow-up study. The four-method comparison in §6 situates the present study's score relative to the execution-method literature.

**H1 fails, replications hold.** The spectral pipeline's validated capability at decoder scale is task-boundary detection (B-P1 / B-P2), not per-pair risk prediction. The decoder-scale product surface for the spectral measurement approach narrows accordingly. The research question shifts: what *does* predict per-pair decoder merge outcome inside a family? Four candidates are pre-specified here so that post-hoc explanation is disciplined:

<!-- ANON: "Study 16 (Llama-2-7B, §9 of FINDINGS.md)" — internal report reference stripped; claim preserved descriptively with the public citation. -->
1. **Source-adapter behavioral quality.** Prior internal structural-compatibility work on Llama-2-7B showed that structural compatibility is necessary but not sufficient; Badirli et al. (arXiv:2602.12323) replicated this at HuggingFace-ecosystem scale. Quality may be the binding constraint at decoder scale rather than pairwise geometry.

2. **Activation-informed geometry.** OSRM (Zhang & Zhou, ACL 2025) established that weight-space subspace overlap can miss interference mediated by the base model's activation covariance. Zhou et al. (arXiv:2601.22285) independently found that activation-based pairwise metrics (activation dot product, cosine similarity) achieve substantially higher predictive correlations with merge outcome than any weight-space metric in their vision-classifier study (r = 0.572 for TSV vs. |r| < 0.2 for effective rank). If H1 fails, the activation-informed direction moves from roadmap to mandatory next step.

3. **Task-family identity as the dominant signal.** The precursor's diagnostic showed task-family identity alone explained R² = 0.97 of cross-task merge outcomes under FAMILY_B. If this holds under the present study's unconfounded task set, it would indicate that geometric pair-specificity is not reachable at this scale with weight-space measurement; operational triage would reduce to task-family lookup.

4. **Intrinsic mergeability (Rahamim et al. 2026).** If adapter mergeability is primarily a scalar property of each individual adapter (governed by base-model prior knowledge) rather than a pairwise property, then no pairwise score can succeed in principle at this scale. This hypothesis is directly testable by measuring within-adapter consistency of merge outcomes across partners on the present study's evaluated-pair sample, as a supplementary analysis.

Distinguishing these candidates is a follow-up-study question, not the present study's. The present study tells us whether the weight-space pairwise-spectral hypothesis survives; it does not adjudicate among these four alternatives in the event of failure.

**Replications fail.** Task-boundary detection does not replicate on this task set. This would be highly surprising and would require diagnosis before any further decoder-scale work proceeds. Most likely cause: the [0.70, 0.90] accuracy band selected tasks that are structurally similar in some way the pilot did not reveal (e.g. all multiple-choice, all commonsense-reasoning). Remediation would be a task-set redesign, not a re-run.

## 9. Execution Plan and Deliverables

### 9.1 Directory layout

<!-- ANON: "scripts/n134/" prefix stripped (internal project-number path); "sidecar/data/n134/" paths stripped (internal working-directory convention). The bundle-relative layout under which these same artifacts ship is described in supplementary/README.md. -->

```
analysis_scripts/
  00_pilot_train.py          # Phase 0: single-seed pilot per task
  01_pilot_gate.py           # Phase 0: verify [0.70, 0.90] band + erank continuity
  02_train_adapters.py       # Phase 1: full 24-adapter training (seeds 42, 123, 456)
  03_spectral_audit.py       # Phase 2: per-adapter + pairwise audit, schema v2.1
  04_sample_pairs.py         # Phase 3a: commit 45-pair cross-task sample using PAIR_SEED=134
  05_merge_eval.py           # Phase 3b: evaluate 69 merges (24 same-task + 45 cross-task)
  06_analysis_h1.py          # Phase 4: primary H1 + confirmatory replications
  07_analysis_secondary.py   # Phase 4: secondary exploratory measures
  08_compare_methods.py      # Phase 5: four-method scheduled comparison (§6)

raw_analytical_artifacts/
  pilot/                     # pilot adapter checkpoints + audit
  adapters/                  # 24 final adapters (Phase 1 output)
  audit_v2_1.schema.json     # schema definition
  audit/                     # 24 adapter audits + 276 pair audits (Phase 2 output)
  pair_sample.json           # the committed 45-pair sample (Phase 3a output)
  merges/                    # 69 merge-and-eval JSONs (Phase 3b output)
  analysis_h1.json           # H1 decision + replications (Phase 4 primary)
  analysis_secondary.json    # exploratory logs (Phase 4 secondary)
  method_comparison.json     # four-method comparison results (Phase 5)
  figures/                   # all diagnostic figures

pre_registration/
  preregistration_v3_1.md    # this document
  (the final report's corresponding prose is in the main manuscript)
```

### 9.2 Execution order (coding agent checklist)

<!-- ANON: "N134_PAIR_SEED" env-var reference rewritten to generic PAIR_SEED per earlier §3.3 edit. -->
1. **Phase 0 (pilot).** Run `00_pilot_train.py` for 8 tasks × 1 seed (42). Run `01_pilot_gate.py`. Halt and replace tasks from reserve list if gate fails per §3.2.
2. **Phase 1 (training).** Run `02_train_adapters.py` for 8 tasks × (seeds 123, 456). Pilot seed-42 adapters are retained. Total: 24 adapters.
3. **Phase 2 (audit).** Run `03_spectral_audit.py`. Validates output against schema v2.1. Halts if any adapter fails to produce all required fields.
4. **Phase 3a (pair sampling).** Run `04_sample_pairs.py` with `PAIR_SEED=134`. Commits `pair_sample.json` before any merge evaluation runs. This file must be committed before Phase 3b begins.
5. **Phase 3b (merge evaluation).** Run `05_merge_eval.py` on the 69 pairs (24 same-task + 45 cross-task as sampled). 0.5/0.5 linear merge, evaluate on each source's task, record `max_degradation`.
6. **Phase 4 (analysis).** Run `06_analysis_h1.py` for primary H1 + confirmatory replications. Run `07_analysis_secondary.py` for secondary exploratory measures. H1 decision is emitted here.
7. **Phase 5 (comparison).** Run `08_compare_methods.py` for the four-method scheduled comparison. Requires reference implementations of KnOTS, TSV, and SVC from their respective repositories (see §9.3).

<!-- ANON: "N133's resume-friendly convention" → descriptive. -->
**Idempotence requirement.** Every phase must skip its output if the output file exists and is valid (matches the schema version declared in `audit_v2_1.schema.json` for audit, or passes JSON validation for other outputs). This preserves the precursor study's resume-friendly convention.

### 9.3 External dependencies

<!-- ANON: "Gradience" package-name row rewritten descriptively: the spectral-audit dependency is internal to this project and ships with the bundle's analysis scripts rather than as an external package reviewers are expected to pip-install. -->

| Dependency | Source | Purpose |
|------------|--------|---------|
| Mistral-7B-v0.3 | HuggingFace `mistralai/Mistral-7B-v0.3` | base model |
| GLUE / commonsense datasets | HuggingFace `datasets` | training data (see §3.2 table) |
| PEFT | `peft>=0.11.0` | LoRA training |
| Spectral audit (this paper) | bundled with `analysis_scripts/` | spectral audit |
| KnOTS reference | `github.com/gstoica27/KnOTS` | method comparison (§6) |
| TSV reference | `github.com/AntoAndGar/task_singular_vectors` | method comparison (§6) |
| SVC reference | `github.com/lyymuwu/SVC` | method comparison (§6) |

### 9.4 Resource estimate

<!-- ANON: resource block rewritten per checklist §2.1(8): provider name "RunPod Secure Cloud" stripped from primary target; cost figures preserved (public pricing, no identity signal); substitute-GPU table left in place because all named cards are standard commercial SKUs with public pricing. -->
**Primary target: single RTX 6000 Ada 48GB on commercial cloud ($0.74/hr as of April 2026).**

Approximately **42 hours end-to-end**, $31 total cost:

- Phase 0 (pilot): ~10 h (8 adapters × ~75 min)
- Phase 1 (full training): ~20 h (16 additional adapters × ~75 min)
- Phase 2 (audit): ~2 h (mostly CPU-bound SVD, GPU near-idle)
- Phase 3 (merge eval): ~3.5 h (inference-bound)
- Phase 4–5 (analysis + comparison): ~1 h (CPU-bound)
- Buffer for pilot gate retries + setup: ~5 h

**VRAM footprint.** Peak training memory ~20–22 GB (14 GB frozen base in bf16 + activations for seq_len 512 batch 4 + optimizer state for LoRA trainables + CUDA overhead). 48 GB provides ~2× headroom; no gradient checkpointing or memory optimization required.

<!-- ANON: reference to "N133" in GPU-choice discussion generalized. -->
**Why 6000 Ada over A100 80GB or H100 80GB.** LoRA training on a 7B frozen backbone is dominated by memory-bandwidth cost of loading base weights each forward/backward pass; the LoRA math is negligible. The 80-GB-class premium buys VRAM the protocol does not use. H100 80GB completes in ~22 h at $1.99/hr = $44; A100 80GB completes in ~35 h at $1.19/hr = $42; RTX 6000 Ada 48GB completes in ~42 h at $0.74/hr = $31. Total cost range across the efficient frontier is only ~$13, so the choice is about wall clock and reliability, not money. RTX 6000 Ada is the cost-minimum reliable configuration for this workload.

**Acceptable substitutes (in priority order):**

1. **L40S 48GB** ($0.79/hr) — near-identical architecture and performance; use if 6000 Ada is unavailable at run time.
2. **RTX 6000 Pro 96GB** ($1.90/hr) — only if neither of the above is available at run time. Overspecified for this workload.
3. **A100 80GB PCIe** ($1.19/hr) — ~35% premium for unused VRAM; defensible if the coding agent has existing infrastructure tested against A100 from prior work.
4. **H100 PCIe 80GB** ($1.99/hr) — ~$13 premium over 6000 Ada for a 20-hour wall-clock reduction. Use this path if wall clock matters operationally.

**Unacceptable configurations** (any of these would constitute a deviation from the pre-registered protocol under §7):

- Any 40-GB GPU (A100 40GB, L40 40GB) — insufficient VRAM introduces fragility not accounted for in the protocol.
- Any 24-GB consumer GPU (RTX 4090, RTX 3090) — requires QLoRA or gradient checkpointing, which introduces uncontrolled training variables.
- Multi-GPU configurations — the protocol is embarrassingly serial (one adapter at a time, then one merge at a time). Parallelization would require restructuring the execution plan and would constitute a material deviation.
- H100 SXM — the SXM premium requires NVLink benefit from multi-GPU; wasted on single-GPU workloads.

**Phased execution option (recommended for first run).** Run Phase 0 as a standalone job first. Tear down the pod, verify pilot gate passes for ≥5 of 8 tasks per §3.2 before committing Phases 1–5. Pilot standalone cost: ~$7.50. This avoids paying for an idle pod if the protocol halts at the pilot gate.

<!-- ANON: "3× N133's compute" → "3× the precursor's compute"; "cost-per-experiment drops ~40% versus N133's H100 path" generalized similarly. -->
Scales roughly 3× the precursor's compute; cost-per-experiment drops ~40% versus the precursor's H100 path due to platform choice.

### 9.5 Required artifacts in the final report

<!-- ANON: report path "sidecar/notes/n134_report.md" → descriptive "the final manuscript". -->
The final manuscript must include:

1. Pilot gate results and any task replacements, with rationale.
2. Per-adapter accuracies (confirming [0.70, 0.90] band).
3. B-P1, B-P2, B-P4 replication statistics.
4. **H1 decision** (clears / null), with exact partial ρ, p-value, ΔR², and bootstrap CI.
5. Secondary exploratory results, explicitly labeled as non-evidential.
6. Four-method scheduled comparison table with retained-set mean degradation and CIs.
7. Any deviations from this spec, with timestamp of deviation relative to data collection start.

---

## Appendix A — Known unknowns declared in advance

These are things we don't know the answer to as of spec-commit time. Declaring them here prevents retroactive reframing.

<!-- ANON: references to "N134" in each bullet replaced with descriptive phrases. -->
- **Whether tasks 1–8 actually land in [0.70, 0.90].** Pilot will tell us. If fewer than 5 tasks land in-band after retries and substitutions, the study is halted and the task-set design is revised before any Phase 1 training.
- **Whether erank is continuously distributed across the 8 tasks.** Pilot tests via Hartigan's dip on the 8-point distribution. If bimodal, the study is halted and the task set revised.
- **Whether the four comparison methods admit clean triage-mode adaptation.** KnOTS and TSV are designed as execution methods; using them as triage requires choosing a per-pair scalar risk score from their per-pair transformations. SVC is designed for k≥3 merging; pair-wise adaptation documented in §6 but not guaranteed to match the paper's intended use. If a method's triage adaptation is judged unfaithful by the coding agent, report as "adaptation inadequate" rather than reporting a biased comparison.
- **Whether a weight-space score can succeed at decoder scale in principle.** Zhou et al. (2026) found on vision classifiers that weight-space metrics individually achieve |r| < 0.2 with post-merge accuracy while activation-based metrics achieve r > 0.5. Whether this encoder/vision-scale pattern holds at decoder-LLM scale is not known. The present study's H1 is a direct test on one specific weight-space score (O-depth alignment) in one specific regime (Mistral-7B, r=16, commonsense-reasoning tasks in [0.70, 0.90] accuracy band). A negative H1 result is consistent with either (i) this particular score being wrong for the regime, or (ii) weight-space pairwise prediction being fundamentally insufficient at decoder scale. Distinguishing (i) from (ii) requires a follow-up study with an activation-informed score variant and is out of scope for the present study.

---

## Appendix B — Change log

### v1 → v2 (April 18 2026)

<!-- ANON: change-log entries retained substantively; the "Gradience / KnOTS / TSV" method list rewritten. -->
For audit trail:

1. **§1 extended** with explicit positioning against the 2025–2026 spectral-merging literature. No change to scientific question.
2. **§3.5 strengthened** to make U/V persistence a hard schema requirement (v2.1). Necessary to enable SVC and faithful TSV comparisons.
3. **§6 expanded** from 3 methods (this paper's score / KnOTS / TSV) to 4 methods (+ SVC). Pre-registered direction updated accordingly.
4. **§6 end-notes added** clarifying why TARA-Merging and Core Space Merging are not in the comparison. OSRM cited as external conceptual reference.
5. **§8 "H1 fails" outcome updated** to reference activation-informed geometry (the OSRM direction) as a future-work candidate.
6. **§9 execution plan formalized** with explicit directory layout, phase-by-phase checklist, external dependency table, and required-artifacts list for the final report. This section was not in v1.
7. **Appendix A added** declaring pilot-stage unknowns in advance.

No changes to: task set, seeds, training protocol, primary H1 score, decision rule, confound constraints C1–C4, or statistical protocol.

### v2 → v3 (April 19 2026)

1. **§1 extended** with two additional paragraphs: "Position relative to the mergeability-prediction literature" (citing Rahamim et al. 2026, Zhou et al. 2026, Bolton et al. 2026) and "Why decoder-scale triage matters now" (citing arXiv:2511.21437). No change to scientific question.
2. **§8 "H1 fails" outcome** extended from three named candidate explanations to four. Added: intrinsic mergeability (Rahamim et al.). Expanded activation-informed geometry candidate to explicitly cite Zhou et al. metric magnitudes.
3. **Appendix A extended** with a fourth declared unknown: whether weight-space pairwise prediction is possible in principle at decoder scale, independent of the specific H1 score choice.
4. **Appendix B heading generalized** from "What v2 changed from v1" to "Change log" so future amendments can accumulate without renaming.

<!-- ANON: unchanged-items list: "Gradience / KnOTS / TSV / SVC" rewritten. -->
No changes to: task set, seeds, training protocol, primary H1 score definition (§4), decision rule thresholds (ρ ≥ 0.50, ΔR² ≥ 0.10), confound constraints C1–C4, statistical protocol (§5), four-method scheduled comparison set (this paper's score / KnOTS / TSV / SVC, §6), deviation policy (§7), or execution plan (§9). No new code required.

### v3 → v3.1 (April 19 2026)

1. **§9.4 resource estimate** rewritten. GPU target changed from single H100 (~18 h, ~$44) to single RTX 6000 Ada 48GB (~42 h, ~$31). Rationale: LoRA training on frozen 7B base is memory-bandwidth-bound, not FLOPS-bound; 48 GB is sufficient VRAM; RTX 6000 Ada is the cost-optimum reliable configuration.
2. **§9.4 extended** with acceptable substitute GPUs (L40S, RTX 6000 Pro, A100 80GB, H100 PCIe) in priority order, and with unacceptable configurations explicitly flagged as material deviations under §7.
3. **§9.4 extended** with recommended phased execution (Phase 0 standalone first, verify pilot gate, then commit to Phases 1–5).
4. **Appendix A unchanged.** The four declared unknowns remain accurate.

No changes to: scientific question, H1 score, decision rule, confound constraints, task set, seeds, LoRA hyperparameters (rank 16, targets q/k/v/o_proj, alpha 32, lr 2e-4, warmup 6%, bf16), audit schema v2.1, four-method comparison set, statistical protocol, deviation policy, directory layout, or required artifacts.

---

<!-- ANON: author byline stripped; original preserved in git history for camera-ready restoration. -->
*Pre-registration commit date: April 19 2026. Spec v3.1 supersedes v3 in all respects; v1, v2, and v3 are retained in git history for audit.*
