# Deeper Diagnostic Study Memo

## 1) Activation Audit Outcome
- Audited pairs: 189, pair advisories: {'none': 188, 'elevated': 1}.
- Layer bands: {'low': 2265, 'watch': 36, 'high': 7}. This quantifies whether current thresholds activate often enough for a meaningful validation slice.

## 2) Targeted Cohort Construction
- Cohort built from high-overlap/non-conflict/loadable/num-label-compatible slice: 30 pairs ({'overlap_gate': 0.14, 'conflict_gate': 0.1, 'n_high': 15, 'n_low': 15}).
- High-tail was selected to maximize advisory activation chance under current code; lower-tail comparators were selected from the same candidate regime.

## 3) Strict Naive Reruns
- Every rerun used `uniform_linear` with strict plan checks to confirm 0.5/0.5 linear layer configs.
- Source baselines were rerun on the exact evaluation dataset for each pair (dataset-matched source baselines).

## 4) Initial Reassessment
- High-tail mean delta vs best: -0.1047 (n=15).
- Lower-tail mean delta vs best: -0.1088 (n=15).
- Spearman(delta, OA max score): 0.1791.
- Subfactor decomposition (alignment/concentration/coefficient exposure) is reported in the explanatory analysis for benign-vs-inflation-prone overlap checks.
- Spearman(delta, triple-high layer fraction): 0.1533.
- At threshold 0.35: activated pairs=4, activated layers=43, rerun activated n=1.

## 5) Interpretation Guardrails
- This remains a bounded correlation study; not causal proof.
- No taxonomy/policy replacement is implied from this pass alone.
- Execution-side method escalation (e.g., SVC) should still depend on stronger replicated signal.
