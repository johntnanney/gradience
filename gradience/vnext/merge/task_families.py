"""Static task-family registry for advisory modulation.

Maps known eval_dataset names to family labels. When both adapters in a
pair belong to the same family but have different dataset names, the
cross-task advisory is downgraded from full caution to an informational
same-family note.

Design constraints (from Campaign A findings, 2026-03-29):
- Static mapping table, not a learned classifier
- Only validated families included
- Exact same-task remains its own class
- Family-equivalent is advisory-only, not automatic retention
- All family mappings documented here as the single source of truth
- Unknown datasets remain in singleton families (conservative default)

Evidence base:
- Campaign A confirmed SST-2 × IMDB merges are indistinguishable from
  same-task retained (avg Δ -0.022 vs -0.017, gap 0.005)
- Binary sentiment is the only empirically validated family so far
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Task-family registry
# ---------------------------------------------------------------------------
# Each key is a dataset name as it appears in eval_dataset fields.
# Each value is a canonical family label.
#
# Add entries ONLY for families with empirical merge-equivalence evidence.
# Do NOT add speculative families — keep this curated and narrow.

TASK_FAMILY_REGISTRY: dict[str, str] = {
    # Binary sentiment — validated by Campaign A (2026-03-29)
    # SST-2 × IMDB merges behave like same-task retained pairs.
    "sst2": "sentiment_binary",
    "imdb": "sentiment_binary",
    "yelp_polarity": "sentiment_binary",
    "amazon_polarity": "sentiment_binary",
}

# Reverse index: family label -> set of dataset names
_FAMILY_MEMBERS: dict[str, set[str]] = {}
for _ds, _family in TASK_FAMILY_REGISTRY.items():
    _FAMILY_MEMBERS.setdefault(_family, set()).add(_ds)


def lookup_family(eval_dataset: str) -> str | None:
    """Return the family label for a known dataset, or None if unknown."""
    return TASK_FAMILY_REGISTRY.get(eval_dataset)


def same_family(ds_a: str, ds_b: str) -> bool:
    """Return True if both datasets belong to the same validated family.

    Returns False if either dataset is unknown (conservative default).
    """
    fam_a = TASK_FAMILY_REGISTRY.get(ds_a)
    fam_b = TASK_FAMILY_REGISTRY.get(ds_b)
    if fam_a is None or fam_b is None:
        return False
    return fam_a == fam_b


def family_label(ds_a: str, ds_b: str) -> str | None:
    """Return the shared family label if both belong to the same family, else None."""
    fam_a = TASK_FAMILY_REGISTRY.get(ds_a)
    fam_b = TASK_FAMILY_REGISTRY.get(ds_b)
    if fam_a is not None and fam_a == fam_b:
        return fam_a
    return None
