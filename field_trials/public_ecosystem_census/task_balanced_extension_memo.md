# Task-Balanced Extension Memo

## 1) What Changed

- Baseline fingerprints: 26
- Augmented fingerprints: 36
- New candidates discovered: 0
- New adapters selected for audit: 0

## 2) What Changed in Results

- Mean architecture eta-sq: 0.1155 -> 0.143
- Mean task eta-sq: 0.2599 -> 0.3359
- Architecture kNN purity: 0.9 -> 0.7444
- Task kNN purity: 0.7 -> 0.5667
- Module-type sign test (augmented): Attention < MLP in 2/13 adapters (15%)

## 3) What Remains Bounded

- Found-artifact observational setting only (no causal claim).
- Task labels remain metadata-driven and partially noisy.
- Architecture×task interactions remain exploratory unless per-cell counts are adequate.
- No product-policy implication from this extension pass.

## 4) Decoder Question Impact

This extension sharpens the architecture-vs-task interpretation for the public ecosystem slice and informs the later controlled decoder GPU study as either causal disambiguation or confirmation.

## 5) Decision

- **mixed_but_bounded**
