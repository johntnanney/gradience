# Gradience vNext

This directory contains the canonical, versioned telemetry schema and shared data model used by:

- `gradience check` (config validator)
- `gradience audit` (efficiency auditor)
- `gradience monitor` (training monitor / flight recorder)

The first vNext feature is the **config validator** policy:

```python
from gradience.vnext import check_config, ConfigSnapshot

recs = check_config(config)
for r in recs:
    print(r.severity, r.action, r.message)
```

The guiding principle is the *restraint hypothesis*: training and fine-tuning generalize better when updates
are constrained (lower LR, lower adapter amplitude, fewer adapted modules), and Gradience should help you
detect when you’ve left that regime.

See `SCHEMA.md` for the frozen v1 telemetry contract, and `types.py` / `telemetry.py`
for the concrete implementations.
