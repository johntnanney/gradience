# Catastrophic Anchor Dossier: [PAIR] on [BACKBONE]

## Metadata

- **Anchor ID:** [e.g., CA-01]
- **Date:** [creation date]
- **Related studies:** [which studies produced the data]
- **Instability program role:** [what this anchor tests or demonstrates]

---

## Identity

- **Task pair:** [Task A × Task B]
- **Backbone:** [model name and layer count]
- **Taxonomy class:** [backbone reversal / unstable severe / stable asymmetric]
- **Instability score:** [composite score]

---

## Severity Profile

| Seed variant | Δ task A | Δ task B | Max Δ | Severity class |
|--------------|--------:|--------:|------:|----------------|
| s42 × s42 | | | | |
| s42 × s7 | | | | |
| s7 × s42 | | | | |
| s7 × s7 | | | | |

**Seed range:** [max - min worst Δ across variants]
**CV:** [coefficient of variation]
**Worst variant:** [which seed combo]
**Best variant:** [which seed combo]

---

## What Core Signals Said

| Signal | Value | Would it have predicted catastrophic? |
|--------|-------|--------------------------------------|
| pair_risk | | |
| dominant_issue | | |
| reconstruction_error | | |
| task_relationship_advisory | | |
| source QA eligibility (A) | | |
| source QA eligibility (B) | | |

**Verdict:** [Did any core signal distinguish this from non-catastrophic pairs?]

---

## Cross-Backbone Behavior

| Backbone | Worst Δ | Class | Seed range |
|----------|--------:|-------|----------:|
| DistilBERT | | | |
| RoBERTa | | | |
| DeBERTa | — | — | — |

**Backbone shift:** [absolute change in worst Δ between backbones]
**Reversal?** [yes/no — does the severity class change qualitatively?]

---

## Mechanistic Observations

### Victim pattern
[Which task collapses? Is the victim the stronger or weaker source?]

### Culprit pattern
[Is a specific adapter/seed implicated?]

### Architectural hypotheses
[Any backbone-specific structural explanations?]

---

## Open Questions

1. [Specific testable question about this anchor]
2. [What Workstream B could reveal]
3. [What DeBERTa adjudication will test]

---

## DeBERTa Prediction

**Expected behavior:** [what the instability program predicts for this pair on DeBERTa]
**Confidence:** [high / medium / low]
**What would be surprising:** [what outcome would challenge the program]
