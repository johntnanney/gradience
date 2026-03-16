# API Reference

Gradience exposes a stable public API across multiple interfaces. This section documents every public class, function, and type.

## API tiers

| Tier | What's included | Stability guarantee |
|------|----------------|-------------------|
| **Stable** | `gradience.api`, CLI commands, telemetry schema, HF callback, exceptions | Backward compatible across minor releases |
| **Internal** | Everything else | May change without notice |

## Sections

- [Python API](python-api.md) — `gradience.api` module (programmatic wrappers)
- [Spectral Audit](audit.md) — `gradience.vnext.audit` (SVD-based analysis)
- [Merge Audit](merge.md) — `gradience.vnext.merge` (adapter compatibility)
- [Telemetry](telemetry.md) — `gradience.vnext.telemetry` (JSONL schema and reader/writer)
- [Rank Suggestion](rank-suggestion.md) — `gradience.vnext.rank_suggestion` (compression targets)
- [Types & Data Model](types.md) — `gradience.vnext.types` (enums, config snapshots, metrics)
- [Exceptions](exceptions.md) — `gradience.exceptions` (error hierarchy)
- [API Stability Policy](stability.md) — Versioning and compatibility guarantees
