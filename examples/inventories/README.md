# Inventory Fixtures

This directory contains fixed inventory fixtures for neighborhood evaluation.

Each fixture includes:

- `qa/` — adapter QA artifacts (`gradience.adapter_qa/v1`)
- `reports/` — pair merge reports (`gradience.merge_qa_report/v1`)
- `expected.json` — expected grouping/exclusion/boundary outcomes

Run the evaluation harness:

```bash
python3 scripts/eval_neighborhoods.py
```

Results are written under:

`results/neighborhood_eval/<timestamp>/`
