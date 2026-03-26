# Core-Space Severity Replication — Study 01 Results

## Study setup

- Shared panel: 16 adapters, 4 tasks, 2 backbones, 56 total pairs, 48 cross-task
- Core-space computed for all 48 cross-task pairs
- Severity labels assigned from merge evaluation results

## Central result: core-space shared-basis does NOT reliably separate severity across backbones

### DistilBERT: promising but narrow

| Severity | n | Mean basis | Range |
|----------|---|-----------|-------|
| Catastrophic | 3 | **0.854** | [0.852-0.855] |
| Broad degradation | 5 | 0.900 | [0.848-0.952] |
| Asymmetric dilution | 9 | 0.922 | [0.885-0.947] |
| Mild degradation | 7 | 0.911 | [0.872-0.950] |

Correlation (basis vs max_delta): **r = -0.614** — moderate negative correlation (lower basis → worse outcome).

On DistilBERT alone, catastrophic pairs cluster at ~0.85 and non-catastrophic at ~0.90+. But ranges overlap.

### RoBERTa: signal collapses

| Severity | n | Mean basis | Range |
|----------|---|-----------|-------|
| Catastrophic | 5 | 0.885 | [0.849-0.932] |
| Broad degradation | 2 | 0.945 | [0.943-0.948] |
| Asymmetric dilution | 10 | 0.906 | [0.859-0.952] |
| Mild degradation | 3 | 0.890 | [0.875-0.909] |
| Near-safe | 4 | 0.877 | [0.854-0.913] |

Correlation (basis vs max_delta): **r = +0.273** — weak positive (opposite direction from DistilBERT).

On RoBERTa, catastrophic pairs have mean basis 0.885 with range up to 0.932. Non-catastrophic pairs have mean 0.902. The distributions overlap heavily and the correlation sign flips.

### Binary comparison: catastrophic vs non-catastrophic

| Backbone | Catastrophic basis | Non-catastrophic basis | Separation |
|----------|-------------------|----------------------|------------|
| DistilBERT | 0.854 [0.852-0.855] | 0.913 [0.848-0.952] | OVERLAP (non-cata min < cata max) |
| RoBERTa | 0.885 [0.849-0.932] | 0.902 [0.854-0.952] | OVERLAP (heavy) |

Neither backbone shows clean separation. DistilBERT has a suggestive cluster but the non-catastrophic range includes values below the catastrophic mean. RoBERTa shows no meaningful separation at all.

## Research questions answered

### RQ1: Do catastrophic pairs cluster at lower shared-basis?
**On DistilBERT: partially.** Catastrophic mean (0.854) is below non-catastrophic mean (0.913), but ranges overlap.
**On RoBERTa: no.** Catastrophic mean (0.885) is barely below non-catastrophic (0.902), and distributions overlap heavily.

### RQ2: Does separation hold across backbones?
**No.** The correlation between shared-basis and severity flips sign across backbones (r=-0.614 on DistilBERT, r=+0.273 on RoBERTa). This is a clear replication failure.

### RQ3: Does separation hold across task-pair instances?
**No.** Within the same task-pair family, shared-basis values are similar across backbones, but the severity labels they correspond to are different (e.g., QNLI×MRPC has basis ~0.85 on both backbones but is catastrophic on DistilBERT and mild on RoBERTa).

### RQ4: Does shared-basis add value beyond other signals?
**Not reliably.** On DistilBERT it appeared promising (r=-0.614), but this does not replicate. Advisory already catches all cross-task pairs. Shared-basis does not reliably grade within that boundary.

### RQ5: Is shared-basis stable enough for future product use?
**No.** The sign-flip across backbones disqualifies it as a reliable severity signal in its current form.

## Verdict

**`core_space_severity_signal_not_stable_enough`**

Core-space shared-basis is not stable enough across backbones to serve as a severity-grading signal inside cross-task pairs. The DistilBERT pattern (lower basis → worse outcome) does not replicate on RoBERTa. The correlation sign flips, distributions overlap heavily, and the same shared-basis value corresponds to different severity outcomes on different backbones.

## What this means for Gradience

1. **Core-space should not be promoted for severity grading.** The DistilBERT-only pattern was suggestive but backbone-specific.

2. **Core-space retains its current narrow role.** It is a legitimate advanced structural diagnostic. But its role should remain: selective use in genuinely ambiguous cases, not systematic severity prediction.

3. **The severity grading problem remains open.** Neither task-pair identity (backbone-dependent) nor core-space shared-basis (sign-flips across backbone) provides a reliable cross-backbone severity signal. Current Gradience catches the boundary but cannot grade within it.

4. **The advisory is confirmed as the right scope.** It catches same/different cleanly on both backbones. It should not be expected to grade severity — nothing currently available can do that reliably.
