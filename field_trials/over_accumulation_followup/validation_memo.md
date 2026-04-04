# Over-Accumulation Diagnostic Validation Memo (First Pass)

## 1. What Was Tested
- Cohort size: **21** pair evaluations from local field-trial artifacts.
- Naive-only filter active: **no**.
- Over-accumulation metrics were recomputed with current Gradience merge-audit code for each pair.
- Outcome metric focus: `merge_delta_vs_best_source` with evidence-based source baselines.

## 2. What the Advisory Explained
- Group 1 (high-overlap lower-score): n=4, mean delta vs best=-0.0190.
- Group 2 (high-overlap higher-score / official elevated when present): n=5, mean delta vs best=-0.0156.
- High-overlap union: n=9, Spearman(delta, max_oa)=0.2661.
- Official advisory distribution in high-overlap subset: {'none': 9}.

## 3. What Remained Bounded
- Task scope: classification-focused field-trial adapters.
- Artifact scope: LoRA/low-rank PEFT pairs already present in repo.
- Merge condition scope: retrospective; includes mixed historical strategies unless rerun with strict naive baseline.
- Explanatory status: correlational only; this pass does not establish causal mechanism.

## 4. What This Means for Gradience
- Official advisory did not activate in this cohort, so direct `none vs watch/elevated` validation remains pending.
- The score-proxy split provides bounded exploratory signal but is not strong enough to upgrade policy confidence by itself.

## 5. Whether to Proceed to Execution-Side Study
- Recommendation: **deeper diagnostic study first**, with strict naive reruns for a targeted high-overlap cohort and dataset-matched source baselines.
- Execution-side calibrated merge comparison (e.g., SVC-style) is reasonable only after strict-naive evidence is refreshed under current runtime constraints.
