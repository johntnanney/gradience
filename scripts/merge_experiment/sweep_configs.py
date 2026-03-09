"""
Sweep configuration grid for merge parameter experiments.

Defines a set of experiment configurations varying:
- Merge coefficients (adapter weighting)
- Output rank (SVD truncation target)
- TIES trim fraction (magnitude trimming aggressiveness)
- Adapter source (Predibase vs. self-trained)

Usage::

    from sweep_configs import SWEEP_GRID, SweepConfig

    for config in SWEEP_GRID:
        print(f"{config.name}: {config.strategy}, rank={config.output_rank}")
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass
class SweepConfig:
    """One experiment configuration in the parameter sweep.

    Attributes
    ----------
    name : unique identifier for this config (used in directory names)
    strategy : merge planning strategy name
    coefficients : (coeff_a, coeff_b) adapter weighting
    output_rank : target rank for SVD refactoring
    trim_fraction : TIES magnitude trimming (0.0 for linear strategies)
    adapter_source : "predibase" or "trained" — which adapter pair to use
    output_alpha : LoRA alpha for merged adapter
    experiment : which sweep experiment this belongs to
    """

    name: str
    strategy: str
    coefficients: tuple[float, float]
    output_rank: int
    trim_fraction: float
    adapter_source: Literal["predibase", "trained"]
    output_alpha: float = 16.0
    experiment: str = ""

    def to_dict(self) -> dict:
        """Serialize to JSON-compatible dict."""
        return {
            "name": self.name,
            "strategy": self.strategy,
            "coefficients": list(self.coefficients),
            "output_rank": self.output_rank,
            "trim_fraction": self.trim_fraction,
            "adapter_source": self.adapter_source,
            "output_alpha": self.output_alpha,
            "experiment": self.experiment,
        }

    def plan_kwargs(self) -> dict:
        """Build kwargs dict for plan_from_audit()."""
        kwargs = {
            "output_rank": self.output_rank,
            "output_alpha": self.output_alpha,
        }
        if self.strategy == "uniform_linear":
            kwargs["coefficients"] = self.coefficients
        if self.strategy == "overlap_ties":
            kwargs["trim_fraction"] = self.trim_fraction
        return kwargs


# ---------------------------------------------------------------------------
# Sweep grid definition
# ---------------------------------------------------------------------------

SWEEP_GRID: list[SweepConfig] = []

# ---- Experiment 1: Coefficient sweep (uniform_linear, rank=8) ----
# Tests asymmetric weighting: does favoring one adapter help?
for ca, cb in [(0.3, 0.7), (0.4, 0.6), (0.5, 0.5), (0.6, 0.4), (0.7, 0.3)]:
    SWEEP_GRID.append(
        SweepConfig(
            name=f"coeff_{ca}_{cb}",
            strategy="uniform_linear",
            coefficients=(ca, cb),
            output_rank=8,
            trim_fraction=0.0,
            adapter_source="trained",
            experiment="coefficient_sweep",
        )
    )

# ---- Experiment 2: Rank sweep (uniform_linear + audit_aware) ----
# Tests rank truncation: does higher rank improve accuracy?
for rank in [4, 8, 12, 16]:
    for strat in ["uniform_linear", "audit_aware"]:
        SWEEP_GRID.append(
            SweepConfig(
                name=f"rank_{rank}_{strat}",
                strategy=strat,
                coefficients=(0.5, 0.5),
                output_rank=rank,
                trim_fraction=0.0,
                adapter_source="trained",
                output_alpha=float(rank * 2),  # maintain alpha/rank = 2.0
                experiment="rank_sweep",
            )
        )

# ---- Experiment 3: Trim fraction sweep (overlap_ties) ----
# Tests TIES trimming: is the underperformance due to wrong trim_fraction?
for trim in [0.05, 0.1, 0.2, 0.3, 0.5]:
    SWEEP_GRID.append(
        SweepConfig(
            name=f"trim_{trim}",
            strategy="overlap_ties",
            coefficients=(0.5, 0.5),
            output_rank=8,
            trim_fraction=trim,
            adapter_source="trained",
            experiment="trim_sweep",
        )
    )

# ---- Experiment 4: Cross-comparison (Predibase adapters, default params) ----
# Tests adapter source: how do Predibase merges compare to self-trained?
for strat in ["uniform_linear", "audit_aware", "overlap_ties"]:
    SWEEP_GRID.append(
        SweepConfig(
            name=f"predibase_{strat}",
            strategy=strat,
            coefficients=(0.5, 0.5),
            output_rank=8,
            trim_fraction=0.2 if strat == "overlap_ties" else 0.0,
            adapter_source="predibase",
            experiment="cross_comparison",
        )
    )


def get_grid_for_experiment(experiment_name: str) -> list[SweepConfig]:
    """Filter the grid to a specific experiment."""
    return [c for c in SWEEP_GRID if c.experiment == experiment_name]


def get_experiment_names() -> list[str]:
    """Return unique experiment names in grid order."""
    seen = set()
    names = []
    for c in SWEEP_GRID:
        if c.experiment not in seen:
            seen.add(c.experiment)
            names.append(c.experiment)
    return names
