"""N134 Phase 4: Deterministic pair sampling.

Produces the 69-pair evaluation set:
  - 24 same-task pairs (8 tasks x C(3,2) = 24)
  - 45 cross-task pairs sampled from 252 possible cross-task pairs

Sampling algorithm (committed RNG seed = 134):
  1. Enumerate all 252 cross-task pairs
  2. Group by cross-task-type cell (28 cells = C(8,2))
  3. Guarantee minimum 1 pair per cell (28 pairs)
  4. Remaining 17 pairs distributed to cells selected by RNG without replacement
  5. Within each cell, pairs selected by RNG

This script is CPU-only and fully deterministic.

Usage:
    python3 scripts/n134/04_sample_pairs.py

Output: /workspace/n134/pair_sample.json
"""

from __future__ import annotations

import json
import random
from itertools import combinations
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N134_PAIR_SEED = 134

TASKS = [
    "arc_challenge",
    "hellaswag",
    "winogrande",
    "openbookqa",
    "commonsenseqa",
    "piqa",
    "siqa",
    "boolq",
]

SEEDS = [42, 123, 456]

FAMILY_B_LABELS: dict[str, str] = {
    "arc_challenge": "science_qa",
    "hellaswag": "commonsense_completion",
    "winogrande": "coreference_commonsense",
    "openbookqa": "knowledge_qa",
    "commonsenseqa": "commonsense_qa",
    "piqa": "physical_commonsense",
    "siqa": "social_commonsense",
    "boolq": "reading_comprehension",
}

OUTPUT_DIR = Path("/workspace/n134")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def adapter_name(task: str, seed: int) -> str:
    return f"{task}_s{seed}"


def task_from_adapter(name: str) -> str:
    """Extract task from adapter name like 'arc_challenge_s42'."""
    # Strip the seed suffix (_s42, _s123, _s456)
    for seed in SEEDS:
        suffix = f"_s{seed}"
        if name.endswith(suffix):
            return name[: -len(suffix)]
    raise ValueError(f"Cannot parse task from adapter name: {name}")


def family_pair_label(task_a: str, task_b: str) -> str:
    """Canonical family-pair label (alphabetical order)."""
    fa = FAMILY_B_LABELS[task_a]
    fb = FAMILY_B_LABELS[task_b]
    pair = sorted([fa, fb])
    return f"{pair[0]}_x_{pair[1]}"


def cell_label(task_a: str, task_b: str) -> str:
    """Canonical cross-task cell label (alphabetical by task)."""
    pair = sorted([task_a, task_b])
    return f"{pair[0]}_x_{pair[1]}"


# ---------------------------------------------------------------------------
# Same-task pairs
# ---------------------------------------------------------------------------


def enumerate_same_task_pairs() -> list[dict]:
    """All 24 same-task pairs: 8 tasks x C(3,2)."""
    pairs = []
    for task in TASKS:
        for s_a, s_b in combinations(SEEDS, 2):
            a = adapter_name(task, s_a)
            b = adapter_name(task, s_b)
            pairs.append(
                {
                    "adapter_a": a,
                    "adapter_b": b,
                    "task_a": task,
                    "task_b": task,
                    "category": "same_task",
                }
            )
    return pairs


# ---------------------------------------------------------------------------
# Cross-task pair sampling
# ---------------------------------------------------------------------------


def enumerate_all_cross_task_pairs() -> list[dict]:
    """All 252 cross-task pairs from C(24,2) - 24 same-task."""
    all_adapters = [adapter_name(t, s) for t in TASKS for s in SEEDS]
    pairs = []
    for a, b in combinations(all_adapters, 2):
        ta = task_from_adapter(a)
        tb = task_from_adapter(b)
        if ta == tb:
            continue
        pairs.append(
            {
                "adapter_a": a,
                "adapter_b": b,
                "task_a": ta,
                "task_b": tb,
                "category": "cross_task",
                "family_pair": family_pair_label(ta, tb),
                "cell": cell_label(ta, tb),
            }
        )
    return pairs


def sample_cross_task_pairs(all_cross: list[dict], n_target: int = 45) -> list[dict]:
    """Sample n_target cross-task pairs with cell-coverage guarantee.

    Algorithm:
      1. Group all 252 pairs by cell (28 cells = C(8,2))
      2. Guarantee minimum 1 pair per cell (28 pairs) -- RNG selects which pair
      3. Remaining 17 pairs: RNG selects cells (without replacement from a
         pool of cell indices), then RNG selects a pair within that cell
    """
    rng = random.Random(N134_PAIR_SEED)

    # Group by cell
    cells: dict[str, list[dict]] = {}
    for pair in all_cross:
        c = pair["cell"]
        if c not in cells:
            cells[c] = []
        cells[c].append(pair)

    # Deterministic cell ordering (alphabetical)
    cell_names = sorted(cells.keys())
    assert len(cell_names) == 28, f"Expected 28 cells, got {len(cell_names)}"

    # Phase 1: one pair per cell (28 pairs)
    selected: list[dict] = []
    remaining_in_cell: dict[str, list[dict]] = {}

    for c in cell_names:
        pool = list(cells[c])
        rng.shuffle(pool)
        selected.append(pool[0])
        remaining_in_cell[c] = pool[1:]

    # Phase 2: distribute remaining 17 pairs
    n_remaining = n_target - len(selected)
    assert n_remaining == 17, f"Expected 17 remaining, got {n_remaining}"

    # Build a pool of cells that still have pairs available
    eligible_cells = [c for c in cell_names if remaining_in_cell[c]]
    # Select cells for the remaining 17 pairs (with replacement allowed for
    # cells, since a cell with 9 pairs can contribute more than 1 extra)
    # But the spec says "without replacement" for cell selection, meaning
    # we draw 17 cells from the eligible pool without replacement.
    # Since we have 28 cells each with at least 8 remaining pairs (9 total
    # per cell minus 1 already selected), and 17 < 28, this always works.
    rng.shuffle(eligible_cells)
    extra_cells = eligible_cells[:n_remaining]

    for c in extra_cells:
        pool = remaining_in_cell[c]
        # pool was already shuffled in phase 1 (it's the tail after removing index 0)
        # but let's re-shuffle for clarity -- RNG state is deterministic
        rng.shuffle(pool)
        selected.append(pool[0])
        remaining_in_cell[c] = pool[1:]

    assert len(selected) == n_target, f"Expected {n_target} pairs, got {len(selected)}"
    return selected


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    same_task_pairs = enumerate_same_task_pairs()
    all_cross = enumerate_all_cross_task_pairs()

    assert len(same_task_pairs) == 24, f"Expected 24 same-task pairs, got {len(same_task_pairs)}"
    assert len(all_cross) == 252, f"Expected 252 cross-task pairs, got {len(all_cross)}"

    cross_task_pairs = sample_cross_task_pairs(all_cross, n_target=45)

    # Cell counts
    cell_counts: dict[str, int] = {}
    for pair in cross_task_pairs:
        c = pair["cell"]
        cell_counts[c] = cell_counts.get(c, 0) + 1

    all_pairs = same_task_pairs + cross_task_pairs

    output = {
        "n134_pair_seed": N134_PAIR_SEED,
        "n_same_task": len(same_task_pairs),
        "n_cross_task": len(cross_task_pairs),
        "n_total": len(all_pairs),
        "family_b_labels": FAMILY_B_LABELS,
        "same_task_pairs": same_task_pairs,
        "cross_task_pairs": cross_task_pairs,
        "cell_counts": dict(sorted(cell_counts.items())),
        "all_pairs": all_pairs,
    }

    out_path = OUTPUT_DIR / "pair_sample.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print("=" * 60)
    print("  N134 Phase 4: Pair Sampling")
    print("=" * 60)
    print(f"  RNG seed:          {N134_PAIR_SEED}")
    print(f"  Same-task pairs:   {len(same_task_pairs)}")
    print(f"  Cross-task pairs:  {len(cross_task_pairs)}")
    print(f"  Total pairs:       {len(all_pairs)}")
    print(f"  Cross-task cells:  {len(cell_counts)}")
    print(f"  Cell counts:       min={min(cell_counts.values())} max={max(cell_counts.values())}")
    print(f"  Output:            {out_path}")
    print()

    # Verify determinism: re-run and check
    cross_task_pairs_2 = sample_cross_task_pairs(enumerate_all_cross_task_pairs(), n_target=45)
    for p1, p2 in zip(cross_task_pairs, cross_task_pairs_2):
        assert p1 == p2, "Determinism check failed"
    print("  Determinism check: PASSED")


if __name__ == "__main__":
    main()
