# Decoder Census Handoff (Canonical)

- Audience: strategy/research planning
- Status: bounded, canonical
- Purpose: handoff summary from CPU public decoder census into GPU-return controlled study framing
- Canonical for: decoder census interpretation and next-study framing
- Supersedes: [public_ecosystem_census_task_balanced_summary.md](/Users/john/code/gradience/docs/strategy/public_ecosystem_census_task_balanced_summary.md)
- See also: [task_balanced_extension_memo.md](/Users/john/code/gradience/field_trials/public_ecosystem_census/task_balanced_extension_memo.md), [decoder_census_robustness_note.md](/Users/john/code/gradience/field_trials/public_ecosystem_census/decoder_census_robustness_note.md), [2026-04-03-decoder-only-spectral-fingerprinting-gpu-return-plan.md](/Users/john/code/gradience/docs/plans/2026-04-03-decoder-only-spectral-fingerprinting-gpu-return-plan.md)

## Baseline Pilot-Plus (What It Found)

- Cohort: `n=26` fingerprints (Llama + Mistral only).
- Task mix was narrow (`chat_instruct`, `classification`, `general_unknown` only).
- Architecture and task structure were both visible in this slice:
  - mean architecture eta-squared: `0.1155`
  - mean task eta-squared: `0.2599`
- kNN purity was high in-slice:
  - architecture: `0.90`
  - task: `0.70`
- Confounding was already strong (`max R^2 ~ 0.66`), so residualization was required.

## Task-Balanced Extension (What Changed)

- Added targeted extension audits (`10` selected, all audited), yielding `n=36` total fingerprints.
- Architecture coverage improved to include Qwen (`llama=19, mistral=8, qwen=9`).
- Underrepresented task coverage improved most for math (`math_reasoning=8`), with minimal code/domain additions (`code=1`, `domain_specialist=1`).
- Effect-size movement after augmentation:
  - architecture eta-squared mean: `0.1155 -> 0.143` (`+0.0275`)
  - task eta-squared mean: `0.2599 -> 0.3359` (`+0.076`)
- Decision remained: `mixed_but_bounded`.

## What Remained Stable

- Decoder spectral structure is clearly non-random at ecosystem scale.
- Architecture remains a real signal once family coverage is broadened.
- Task-conditioned structure is also visible in this observational setting.
- The line remains bounded and research-facing; no product-policy implication.

## What Remained Confounded

- Confound pressure increased rather than disappeared (`max R^2 ~ 0.75`), so residualization remains mandatory.
- Found-artifact labels are noisy; task category confidence is uneven.
- Added non-chat task coverage is still asymmetric (math-heavy), so interaction claims stay exploratory.
- Purity shifts degraded after augmentation (`arch -0.1556`, `task -0.1333`), consistent with broader heterogeneity.

## What This Means for the GPU Study

The GPU-return decoder study should no longer ask whether any decoder-side structure exists.  
The census already supports that boundedly.

The GPU study should now ask:

1. How much observed decoder fingerprint variance is genuinely architecture-driven under matched controls?
2. How much is task-driven when architecture is fixed and confounds are explicitly held constant?
3. Which observables are architecture-led vs task-led after controlled confound removal?
4. Which census-era effects persist vs collapse in controlled training conditions?

## Practical Handoff

- Treat the census as ecological evidence of real structure with known confounds.
- Treat the GPU study as the causal disambiguation layer, not a signal-existence check.
- Keep claims bounded until controlled decoder results replicate the strongest census trends.
