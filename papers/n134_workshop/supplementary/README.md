# Supplementary Materials

**Paper:** Measurement Discipline for ML Diagnostics: A Psychometric Framework with a LoRA-Merging Case Study

This bundle accompanies the submitted manuscript. It contains the pre-registration documents, the raw analytical artifacts, and the analysis scripts that produce the headline numbers reported in the paper. The purpose of the bundle is to make the paper's measurement-discipline arguments auditable in the sense the paper itself argues for: a reviewer should be able to verify that each reported number is the output of the pre-registered decision rule applied to the committed data, rather than a post-hoc artifact.

All files in this bundle have been reviewed for identity-revealing content consistent with TMLR's double-blind review policy. Replacements follow the same `% ANON:` / `# ANON:` marker convention used in the main manuscript: each edited region carries an adjacent comment naming what was changed, so a reviewer encountering a stripped passage knows the edit is intentional and has a signature that persists through the camera-ready restoration.

---

## What this bundle contains

```
supplementary/
├── README.md                                      (this file)
├── pre_registration/
│   ├── preregistration_v3_1.md                    (anonymized; the paper's canonical pre-reg)
│   └── icc_analysis_preregistration.md            (anonymized; the Appendix D pre-reg supplement)
├── raw_analytical_artifacts/
│   ├── analysis_h1.json                           (H1 outcome, all criteria)
│   ├── analysis_icc.json                          (Appendix D reliability analysis)
│   ├── analysis_secondary.json                    (§5.3 secondary measures)
│   ├── method_comparison.json                     (Appendix C four-method comparison)
│   ├── pair_alignment_full.json                   (per-pair alignments, full)
│   ├── pair_alignment_summary.json                (per-pair alignments, aggregated)
│   ├── adapter_profiles.json                      (per-adapter spectral profile)
│   ├── w0_properties.json                         (base-model reference spectrum)
│   └── per_adapter_audits/                        (24 per-task-per-seed raw audits)
│       ├── arc_challenge_s42_summary.json
│       ├── arc_challenge_s123_summary.json
│       └── ... (3 seeds x 8 tasks)
└── analysis_scripts/
    ├── compute_s_h1.py                            (produces analysis_h1.json)
    ├── compute_icc.py                             (produces analysis_icc.json)
    └── compare_methods.py                         (produces method_comparison.json)
```

Directory and filenames in this bundle are renamed versions of the corresponding files in the authors' working repository. The renaming strips project-internal identifiers (see the ANON comment block at the top of each file for the original filename, restored at camera-ready).

---

## Artifact-to-claim mapping

For each artifact, the paper section, the specific numerical claim it supports, and the verification path.

### Pre-registration documents

**`preregistration_v3_1.md`**
- Paper sections supported: §4 (full pre-registration design), §4.1 (construct articulation), §4.3 (confound decomposition), §4.4 (decision rule), Appendix F (deviations from pre-registration).
- Central claims this artifact warrants: that the four confounds named in §4.3 were identified and pre-specified *before* data collection began; that the three-criterion decision rule in §4.4 with its explicit thresholds (ρ_partial ≥ 0.5, Δ R² ≥ 0.1, sign-correctness) was committed-to rather than selected post-hoc; that the task set, seed protocol, training bounds, and four-method comparison set were all fixed in advance.
- Verification path: read §§1–11 of the document and compare against §§3–5 of the paper. The document supersedes nothing; its version history at the top records three minor amendments (v2, v3, v3.1), each dated before any data collection.

**`icc_analysis_preregistration.md`**
- Paper sections supported: §4.2 (reliability considerations at pre-registration time), §2.2 (framework treatment of reliability), Appendix D (reliability analysis).
- Central claim this artifact warrants: that ICC(2,1) absolute agreement, with its sample size (8 tasks × 3 seed pairs = 24 observations), bootstrap bounds, and the decision to report both parametric and bootstrap CIs, was specified before the reliability numbers were computed.
- Verification path: compare the ICC form, task list, and CI protocol named in this document against Appendix D's reported ICC = 0.566, SEM = 0.014, and the two reported confidence intervals.

### Raw analytical artifacts

**`analysis_h1.json`**
- Paper sections supported: §5.1 (H1 outcome under the pre-registered rule), §5.2 (informative null), §8 (objections: "the case study is weaker than the framework").
- Central number this artifact warrants: `h1_decision_rule.criterion_1_partial_spearman.rho_partial = -0.533` reported in §5.1 as the committed-value-against-a-sharp-rule; criterion 2's `delta_r2 = 0.0028`; criterion 3's sign-incorrectness.
- Verification path: the `h1_score.range` field `[0.015, 0.025]` matches the §5.1 text "S_H1 ∈ [0.015, 0.025] across the 45 cross-task pairs"; the `criterion_1_partial_spearman.rho_partial` field matches the -0.533 value reported against the committed threshold 0.5.

**`analysis_icc.json`**
- Paper sections supported: §2.2 (reliability clause), §4.2 (pre-registered commitment), Appendix D (full reliability discussion).
- Central number this artifact warrants: ICC(2,1) = 0.566, SEM = 0.014, 95% parametric CI [0.165, 0.874], 95% bootstrap CI [reported in Appendix D].
- Verification path: the `icc_point` and `sem` fields match the in-paper numbers; the `ci_parametric` and `ci_bootstrap` fields match the two CIs whose divergence Appendix D flags.

**`analysis_secondary.json`**
- Paper sections supported: §5.3 (three-architecture replication), §5.4 (four-method comparison).
- Central claim this artifact warrants: the per-architecture task-boundary detection ratios reported in §5.3 and the per-method null outcomes in §5.4 are derived from the committed secondary-analysis spec.

**`method_comparison.json`**
- Paper sections supported: §5.4 (four-method comparison as regime-scope test), Appendix C (pairwise triage adaptations of KnOTS, TSV, SVC).
- Central claim this artifact warrants: all four methods (the paper's S_H1 score plus KnOTS, TSV, SVC) fail the same decision rule on the same 45-pair evaluation set. The `n_cross_task_pairs_evaluated: 45` and bootstrap `n_bootstrap: 5000` fields document the evaluation protocol.
- Verification path: the per-method `rho_partial` values in this file are the numbers Table 2 of the paper reports.

**`pair_alignment_full.json`** and **`pair_alignment_summary.json`**
- Paper sections supported: §5 (headline result), §5.1 (H1 outcome), and — importantly for the rank-on-residuals finding of §6 — the residualized ranks that produce the intrinsic-precision observation.
- Central claim: these are the upstream inputs to `analysis_h1.json`'s reported numbers. Re-running `analysis_scripts/compute_s_h1.py` against these files reproduces `analysis_h1.json` exactly.

**`adapter_profiles.json`** and **`w0_properties.json`**
- Paper sections supported: §3 (case-study setup) and §4.3 confound C1 (source-metric dynamic range).
- Central claim: each of the 24 adapters has source-task accuracy in the pre-registered [0.70, 0.90] band. The `adapter_profiles.json` file contains per-adapter accuracy, effective rank (both @95 and @90 energy), and training-configuration metadata that demonstrates C1 was met.

**`per_adapter_audits/`**
- Paper sections supported: §5 (headline result), Appendix E (reproducibility-check trace).
- Central claim: the 24 per-task-per-seed summary JSONs (3 seeds × 8 tasks) are the raw audit outputs that pair-alignment and analysis scripts aggregate over. Their contents include per-layer principal angle cosines, energy-rank values, and spectral norms — i.e., the primary spectral measurements the paper is about. Attached in full so that reviewers can verify no per-layer outlier drives the aggregate numbers.

### Analysis scripts

**`compute_s_h1.py`**
- Reproduces: `analysis_h1.json`.
- Run: `python compute_s_h1.py --audit-dir per_adapter_audits/ --pair-alignments pair_alignment_full.json --out analysis_h1.json`
- Implements: the S_H1 formula from Appendix A, the partial-Spearman criterion, the ΔR² criterion, and the sign criterion. Comments in the script cite back to the relevant sections of `preregistration_v3_1.md`.

**`compute_icc.py`**
- Reproduces: `analysis_icc.json`.
- Run: `python compute_icc.py --audit-dir per_adapter_audits/ --spec icc_analysis_preregistration.md --out analysis_icc.json`
- Implements: ICC(2,1) absolute agreement for single-measurement, the SEM derivation from ICC and the same-task variance estimate, and both parametric and bootstrap CIs.

**`compare_methods.py`**
- Reproduces: `method_comparison.json`.
- Run: `python compare_methods.py --audit-dir per_adapter_audits/ --pair-alignments pair_alignment_full.json --out method_comparison.json`
- Implements: the four-method comparison of Appendix C. Includes the pairwise triage adaptations of KnOTS, TSV, and SVC documented in the appendix.

---

## How to reproduce the headline numbers

With the bundle extracted and a Python 3.10+ environment available:

```
cd supplementary/
python -m venv .venv && source .venv/bin/activate
pip install numpy scipy scikit-learn pandas
python analysis_scripts/compute_s_h1.py \
    --audit-dir raw_analytical_artifacts/per_adapter_audits/ \
    --pair-alignments raw_analytical_artifacts/pair_alignment_full.json \
    --out /tmp/analysis_h1_reproduced.json
diff <(python -m json.tool raw_analytical_artifacts/analysis_h1.json) \
     <(python -m json.tool /tmp/analysis_h1_reproduced.json)
```

An empty diff confirms the headline number (ρ_partial = -0.533) is deterministic from the committed data and the committed rule. Analogous recipes work for the ICC and four-method comparison JSONs.

Total reproduction cost: ~2 minutes of CPU time. No GPU required for re-running analysis from the committed audit JSONs; the audit data itself was collected on an RTX 6000 Ada 48GB (see `adapter_profiles.json` environment block) but the downstream statistics in this bundle are pure post-processing.

---

## Notes on the deviations-from-pre-registration

Appendix F of the paper enumerates all deviations between the pre-registration documents in this bundle and the reported analyses. Nothing in this supplementary changes that appendix; the deviations named there are the complete list. Readers who want to confirm this can diff the pre-registration documents in `pre_registration/` against the paper's §§3–5 and Appendix A directly.

---

## Scope of what is and is not attached

Attached, because directly warranting paper claims:
- The pre-registration documents (both the main v3.1 spec and the ICC analysis supplement)
- The four top-level analytical JSONs (H1, ICC, secondary, method comparison)
- The per-adapter raw audits (24 files)
- The four meta-JSONs (pair alignments, adapter profiles, base-model reference)
- The three analysis scripts that produce the headline numbers

Not attached, because not directly warranting paper claims:
- Figure-generation scripts (paper figures are reproducible from the analytical JSONs; figure scripts are stylistic)
- The authors' real-time incident log (the rank-on-residuals discovery narrative is distilled into §6 of the paper and Appendix E's reproducibility-check trace; the unabridged log contains working-session framing that does not add evidential weight)
- The companion technical report (heavily redundant with the paper's appendices; attaching it would be duplicative at best)
- Intermediate training checkpoints and model weights (24 adapters × Mistral-7B scale makes the weights too large to distribute; adapter training is specified in the pre-registration documents at the level of detail needed to retrain)

---

## Contact

Per TMLR double-blind policy, authors are anonymized during review. Correspondence may be directed through OpenReview.
