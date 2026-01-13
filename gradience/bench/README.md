# Gradience Bench (v0.1)

Bench is a minimal validation harness for Gradience recommendations.

**Goal:** turn "suggested rank" into a testable claim by running:

1) Train probe adapter (high-rank)
2) Audit -> get compression suggestions
3) Retrain with compressed ranks
4) Eval and compare
5) Emit a report (JSON + Markdown)

## Status

This directory is the **v0.1 scaffold** (layout + config + reporting utilities).
The actual train/audit/retrain protocol wiring lands in later commits.

## Layout

```
gradience/bench/
├── run_bench.py
├── configs/
│   └── distilbert_sst2.yaml
├── protocol.py
├── report.py
└── README.md
```