# Rank Proxy Validation v2

This directory contains the canonicalized CPU-only external validation-target
artifact bundle for the adaptive-rank comparison line.

## Build

```bash
python3 field_trials/rank_proxy_validation_v2/build_v2_bundle.py
```

## Input Sources

The builder reads persisted v1 artifacts in:

- `field_trials/rank_proxy_validation/`

and writes the v2 deliverables in this directory.

## Notes

- This pass intentionally avoids heavy re-runs.
- v2 is a bounded canonicalization of existing CPU evidence.
- This is not a full layer-vector comparison archive.
- Per-layer allocation vectors were not persisted in v1 outputs; v2 preserves
  this limitation explicitly in `allocation_table.json`.
- Some structure-level claims in v2 therefore rely on already-produced
  comparison artifacts (for example allocation/proxy agreement tables), not a
  complete vector-preserving bundle.
