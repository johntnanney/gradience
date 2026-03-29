# benchmarks/

Reusable scripts for running canonical experimental panels.

## Conventions

- Each benchmark must be executable with `python benchmarks/{name}.py`
- Document dependencies and expected runtime at the top of each script
- Produce output into `../results/`
- Scripts should be idempotent — safe to rerun without manual cleanup
- Use argparse for configuration; sensible defaults for everything

## Planned Benchmarks

- `run_catastrophic_anchor_panel.py` — Rerun the catastrophic anchor panel (Workstream A)
- `run_severity_contrast_panel.py` — Compare catastrophic vs mild vs asymmetric pairs (Workstream B)
- `run_backbone_comparison.py` — Same panel across multiple backbones
