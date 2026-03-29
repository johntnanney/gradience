# Panel: {PANEL_ID} — {Title}

## Purpose

{What experimental comparison this panel supports. 1–2 sentences.}

## Panel Type

{One of: catastrophic-anchor | severity-contrast | backbone-comparison | custom}

## Anchors

{Which adapter pairs or task pairs define this panel.}

| Pair ID | Task A | Task B | Expected severity | Notes |
|---------|--------|--------|-------------------|-------|
| {id}    | {task} | {task} | {catastrophic / broad-degradation / mild / asymmetric} | {source of expectation} |

## Conditions

### Backbones

{Which base models. E.g. `bert-base-uncased`, `roberta-base`, `deberta-v3-base`.}

### Seeds

{How many seeds per condition. Which seeds if fixed.}

### Training Configuration

{LoRA rank, learning rate, epochs, or reference to a config file.}

### Evaluation

{Which eval metrics, datasets, scripts.}

## Metrics Collected

{What is measured for each pair in the panel.}

- {metric 1 — description}
- {metric 2 — description}

## Rerun Protocol

{Exact steps to reproduce this panel from scratch.}

```bash
# Example commands
```

## Used By

{List of study IDs that reference this panel.}
