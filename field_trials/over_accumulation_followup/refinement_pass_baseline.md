# Over-Accumulation Refinement Pass Baseline

## Current OA Diagnostic State
- Pair advisory counts: {'none': 188, 'elevated': 1}
- Layer band counts: {'low': 2265, 'watch': 36, 'high': 7}
- Cutpoint summary: {'layer_watch_rule': 'alignment >= 0.45 and score >= 0.35', 'layer_high_rule': 'alignment >= 0.65 and score >= 0.60', 'pair_watch_rule': 'high_count > 0 OR (high+watch fraction) >= 0.25', 'pair_elevated_rule': 'high_count > 0 AND (high_fraction >= 0.15 OR max_score >= 0.75)', 'pairs_with_max_score_ge_0_35': 4, 'pairs_with_max_score_ge_0_60': 1, 'pairs_with_max_score_ge_0_75': 0, 'layers_with_score_ge_0_35': 43, 'layers_with_score_ge_0_60': 7, 'layers_with_band_watch': 36, 'layers_with_band_high': 7}

## Activation Audit Snapshot
- Audited pairs: 189
- Layer entries: 2308
- Pair max score quantiles: {'q50': 0.0687, 'q75': 0.1204, 'q90': 0.1838, 'q95': 0.2401, 'q99': 0.38}
- Layer score quantiles: {'q50': 0.019095171689987183, 'q75': 0.0486310601234436, 'q90': 0.10769253815571848, 'q95': 0.16757805347442625, 'q99': 0.45749999999999996}
- Pair max score cutpoints: >=0.35=4, >=0.60=1, >=0.75=0

## Strict-Naive Cohort Snapshot
- Cohort count: 12
- Selection parameters: {'overlap_gate': 0.2, 'conflict_gate': 0.1, 'n_high': 6, 'n_low': 6}
- Candidate pools: total=13, preferred=13, working=13
- Task relationship mix: {'same_task': 11, 'same_family': 1}

## First-Pass + Deeper Findings Snapshot
- Strict reruns: ok=12, error=0
- Mean delta vs best: high-tail=-0.0183, lower-tail=-0.1243
- Spearman(delta, pair max OA): -0.0176
- Threshold 0.35 activation snapshot: pairs=4, layers=43, rerun_n=1

## Current Interpretation
- Pair-level activation is sparse under current thresholds.
- Pair-level OA signal appears weak/unstable under strict-naive outcomes.
- The line is not yet falsified but remains low-confidence without a sharper diagnostic signal.
