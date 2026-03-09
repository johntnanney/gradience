"""
Gradience vNext types

Canonical data model for:
  - configuration snapshots (what we intended to run)
  - signal snapshots (what we observed)
  - recommendations/alerts (what we suggest doing next)

These are intentionally small and opinionated:
they encode the "restraint navigator" strategy, where we focus on
regime detection (generalization vs memorization/overwrite) and
efficiency auditing (unused adapter capacity).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

# ---------------------------------------------------------------------------
# Telemetry schema versioning
# ---------------------------------------------------------------------------

# Increment the trailing version when the *meaning* or *required fields* change.
TELEMETRY_SCHEMA_VERSION: str = "gradience.vnext.telemetry/v1"


# ---------------------------------------------------------------------------
# Core enums
# ---------------------------------------------------------------------------


class TaskFamily(str, Enum):
    """Coarse task family labels used for policy decisions."""

    EASY_CLASSIFICATION = "easy_classification"
    HARD_REASONING = "hard_reasoning"
    GENERATION = "generation"
    UNKNOWN = "unknown"


# Backward-compat alias (avoid breaking imports that use the old name).
TaskProfile = TaskFamily


class Severity(str, Enum):
    """Severity for alerts and recommendations."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class EventType(str, Enum):
    """Event types for telemetry (backward compatibility)."""

    AUDIT = "audit"
    TRAIN = "train"
    EVAL = "eval"
    ERROR = "error"


# ---------------------------------------------------------------------------
# Config snapshots
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LoRAConfigSnapshot:
    """A minimal, stable representation of a LoRA/PEFT adapter config."""

    r: int | None = None
    alpha: float | None = None
    target_modules: list[str] = field(default_factory=list)

    # Optional extras that vary by library/version.
    dropout: float | None = None
    bias: str | None = None

    # Room for future PEFT parameters without breaking schema.
    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def alpha_over_r(self) -> float | None:
        r = self.r
        if self.alpha is None or r is None or r == 0:
            return None
        return float(self.alpha) / float(r)

    def to_dict(self) -> dict[str, Any]:
        return {
            "r": self.r,
            "alpha": self.alpha,
            "alpha_over_r": self.alpha_over_r,
            "target_modules": list(self.target_modules),
            "dropout": self.dropout,
            "bias": self.bias,
            "extras": dict(self.extras),
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> LoRAConfigSnapshot:
        return LoRAConfigSnapshot(
            r=d.get("r"),
            alpha=d.get("alpha"),
            target_modules=list(d.get("target_modules") or []),
            dropout=d.get("dropout"),
            bias=d.get("bias"),
            extras=dict(d.get("extras") or {}),
        )


@dataclass(frozen=True)
class OptimizerConfigSnapshot:
    """Optimizer hyperparameters relevant to the restraint story."""

    name: str | None = None
    lr: float | None = None
    weight_decay: float | None = None

    betas: tuple[float, float] | None = None
    eps: float | None = None

    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "betas": list(self.betas) if self.betas is not None else None,
            "eps": self.eps,
            "extras": dict(self.extras),
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> OptimizerConfigSnapshot:
        betas = d.get("betas")
        betas_t = tuple(betas) if isinstance(betas, (list, tuple)) and len(betas) == 2 else None
        return OptimizerConfigSnapshot(
            name=d.get("name"),
            lr=d.get("lr"),
            weight_decay=d.get("weight_decay"),
            betas=betas_t,
            eps=d.get("eps"),
            extras=dict(d.get("extras") or {}),
        )


@dataclass(frozen=True)
class TrainingConfigSnapshot:
    """Training-loop parameters that affect dynamics and comparability."""

    seed: int | None = None
    batch_size: int | None = None
    gradient_accumulation: int | None = None
    max_steps: int | None = None
    epochs: int | None = None
    dtype: str | None = None

    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "batch_size": self.batch_size,
            "gradient_accumulation": self.gradient_accumulation,
            "max_steps": self.max_steps,
            "epochs": self.epochs,
            "dtype": self.dtype,
            "extras": dict(self.extras),
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> TrainingConfigSnapshot:
        return TrainingConfigSnapshot(
            seed=d.get("seed"),
            batch_size=d.get("batch_size"),
            gradient_accumulation=d.get("gradient_accumulation"),
            max_steps=d.get("max_steps"),
            epochs=d.get("epochs"),
            dtype=d.get("dtype"),
            extras=dict(d.get("extras") or {}),
        )


@dataclass(frozen=True)
class ConfigSnapshot:
    """Top-level run configuration snapshot."""

    # Identification
    model_name: str | None = None
    dataset_name: str | None = None

    # Coarse task family label (used by policy).
    task_profile: TaskFamily = TaskFamily.UNKNOWN

    # Nested config components
    optimizer: OptimizerConfigSnapshot = field(default_factory=OptimizerConfigSnapshot)
    lora: LoRAConfigSnapshot = field(default_factory=LoRAConfigSnapshot)
    training: TrainingConfigSnapshot = field(default_factory=TrainingConfigSnapshot)

    # Freeform notes + future extension point.
    notes: str | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "dataset_name": self.dataset_name,
            "task_profile": self.task_profile.value,
            "optimizer": self.optimizer.to_dict(),
            "lora": self.lora.to_dict(),
            "training": self.training.to_dict(),
            "notes": self.notes,
            "extras": dict(self.extras),
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> ConfigSnapshot:
        tp = d.get("task_profile", TaskFamily.UNKNOWN.value)
        try:
            task_profile = TaskFamily(tp)
        except Exception:
            task_profile = TaskFamily.UNKNOWN

        return ConfigSnapshot(
            model_name=d.get("model_name"),
            dataset_name=d.get("dataset_name"),
            task_profile=task_profile,
            optimizer=OptimizerConfigSnapshot.from_dict(d.get("optimizer") or {}),
            lora=LoRAConfigSnapshot.from_dict(d.get("lora") or {}),
            training=TrainingConfigSnapshot.from_dict(d.get("training") or {}),
            notes=d.get("notes"),
            extras=dict(d.get("extras") or {}),
        )


# ---------------------------------------------------------------------------
# Metric / signal snapshots
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EvalMetrics:
    """Common evaluation metrics for a given split."""

    loss: float | None = None
    ppl: float | None = None
    accuracy: float | None = None
    n: int | None = None  # number of examples used

    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "loss": self.loss,
            "ppl": self.ppl,
            "accuracy": self.accuracy,
            "n": self.n,
            "extras": dict(self.extras),
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> EvalMetrics:
        return EvalMetrics(
            loss=d.get("loss"),
            ppl=d.get("ppl"),
            accuracy=d.get("accuracy"),
            n=d.get("n"),
            extras=dict(d.get("extras") or {}),
        )


@dataclass(frozen=True)
class SignalSnapshot:
    """Cross-split signals used for regime detection (gap, dominance, etc.)."""

    # Generalization
    train: EvalMetrics = field(default_factory=EvalMetrics)
    test: EvalMetrics = field(default_factory=EvalMetrics)

    # Precomputed convenience scalars
    gap: float | None = None  # e.g., test_ppl / train_ppl

    # Efficiency / geometry (usually from LoRA delta auditing)
    stable_rank_mean: float | None = None
    utilization_mean: float | None = None  # stable_rank_mean / r (if available)

    # Overwrite / amplitude (activation-based dominance recommended)
    dominance_act_mean: float | None = None

    # Spectral (optional, mainly for grokking / long training)
    kappa_mean: float | None = None

    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "train": self.train.to_dict(),
            "test": self.test.to_dict(),
            "gap": self.gap,
            "stable_rank_mean": self.stable_rank_mean,
            "utilization_mean": self.utilization_mean,
            "dominance_act_mean": self.dominance_act_mean,
            "kappa_mean": self.kappa_mean,
            "extras": dict(self.extras),
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> SignalSnapshot:
        return SignalSnapshot(
            train=EvalMetrics.from_dict(d.get("train") or {}),
            test=EvalMetrics.from_dict(d.get("test") or {}),
            gap=d.get("gap"),
            stable_rank_mean=d.get("stable_rank_mean"),
            utilization_mean=d.get("utilization_mean"),
            dominance_act_mean=d.get("dominance_act_mean"),
            kappa_mean=d.get("kappa_mean"),
            extras=dict(d.get("extras") or {}),
        )


# ---------------------------------------------------------------------------
# Recommendations
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Recommendation:
    """A small, human-readable recommendation emitted by policies/monitors."""

    severity: Severity
    action: str  # intentionally a string: allows extension without breaking
    message: str

    # Optional: why we think this helps
    rationale: str | None = None

    # Optional: scope/caveats/confidence
    confidence: float | None = None  # 0..1
    scope: str | None = None

    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "severity": self.severity.value,
            "action": self.action,
            "message": self.message,
            "rationale": self.rationale,
            "confidence": self.confidence,
            "scope": self.scope,
            "evidence": dict(self.evidence),
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> Recommendation:
        sev = d.get("severity", Severity.INFO.value)
        try:
            severity = Severity(sev)
        except (ValueError, KeyError):
            severity = Severity.INFO

        return Recommendation(
            severity=severity,
            action=str(d.get("action") or ""),
            message=str(d.get("message") or ""),
            rationale=d.get("rationale"),
            confidence=d.get("confidence"),
            scope=d.get("scope"),
            evidence=dict(d.get("evidence") or {}),
        )
