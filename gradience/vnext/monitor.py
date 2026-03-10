"""
Real-time training monitor with heuristic rules.

Maintains a sliding window of training metrics and applies configurable
rules to detect saturation, plateaus, gradient anomalies, and convergence.
Framework-agnostic: receives metric dicts, emits alerts.

Can be driven by a live HF Trainer callback or by replaying existing
telemetry files for prototyping and threshold tuning.

Usage (live)::

    monitor = TrainingMonitor()
    alerts = monitor.update(step=100, metrics={"loss": 0.45, "grad_norm": 1.2})
    for a in alerts:
        print(a)

Usage (replay)::

    from gradience.vnext.monitor_replay import replay_telemetry
    result = replay_telemetry("run.jsonl")
    for a in result.alerts:
        print(f"Step {a.step}: {a.code}")
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Any

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class MonitorConfig:
    """Configuration for TrainingMonitor rules.

    All thresholds are intentionally conservative (low false-positive rate).
    Users can tighten them for their specific training setup.
    """

    # Sliding window size (number of metric updates to keep)
    window_size: int = 50

    # --- Loss plateau detection ---
    # Trigger if relative change in loss < delta for patience consecutive checks
    loss_plateau_patience: int = 5
    loss_plateau_delta: float = 0.01

    # --- Gradient norm plateau detection ---
    grad_plateau_patience: int = 5
    grad_plateau_delta: float = 0.02

    # --- Gradient spike detection ---
    # Trigger if grad_norm > multiplier × running mean
    grad_spike_multiplier: float = 5.0

    # --- Eval loss early stopping ---
    # Trigger if eval_loss hasn't improved for patience consecutive evals
    eval_patience: int = 5

    # --- Utilization check (optional, if utilization data is provided) ---
    utilization_warn_threshold: float = 0.25  # flag if utilization < 25%

    # --- Cooldown: suppress repeated alerts of the same type ---
    alert_cooldown_steps: int = 50

    # --- Enable/disable individual rules ---
    enable_loss_plateau: bool = True
    enable_grad_plateau: bool = True
    enable_grad_anomaly: bool = True
    enable_eval_plateau: bool = True
    enable_utilization_check: bool = True


# ---------------------------------------------------------------------------
# Alert data structure
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MonitorAlert:
    """A single alert from the training monitor."""

    step: int
    code: str  # "LOSS_PLATEAU" | "GRAD_PLATEAU" | "GRAD_SPIKE" | etc.
    severity: str  # "info" | "warning" | "critical"
    message: str  # human-readable
    details: dict[str, Any]  # numerical context
    recommendation: str  # actionable advice

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "code": self.code,
            "severity": self.severity,
            "message": self.message,
            "details": self.details,
            "recommendation": self.recommendation,
        }


# ---------------------------------------------------------------------------
# TrainingMonitor
# ---------------------------------------------------------------------------


class TrainingMonitor:
    """Sliding-window training monitor with heuristic rules.

    Framework-agnostic: receives metric dicts, emits MonitorAlert objects.
    """

    def __init__(self, config: MonitorConfig | None = None):
        self.config = config or MonitorConfig()

        # Metric histories (sliding windows)
        self._loss_history: deque[float] = deque(maxlen=self.config.window_size)
        self._grad_norm_history: deque[float] = deque(maxlen=self.config.window_size)
        self._step_history: deque[int] = deque(maxlen=self.config.window_size)

        # Eval history (separate, typically sparser)
        self._eval_loss_history: list[float] = []
        self._eval_step_history: list[int] = []

        # Cooldown tracking: code -> last_triggered_step
        self._cooldowns: dict[str, int] = {}

        # Consecutive counters for plateau detection
        self._loss_stale_count: int = 0
        self._grad_stale_count: int = 0
        self._eval_plateau_active: bool = False

    # -------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------

    def update(self, step: int, metrics: dict[str, Any]) -> list[MonitorAlert]:
        """Feed one training step's metrics. Returns any alerts triggered.

        Expected keys in metrics (all optional):
            loss, grad_norm, lr, utilization
        """
        loss = metrics.get("loss")
        grad_norm = metrics.get("grad_norm")
        utilization = metrics.get("utilization")

        if loss is not None and _is_finite(loss):
            self._loss_history.append(float(loss))
        if grad_norm is not None and _is_finite(grad_norm):
            self._grad_norm_history.append(float(grad_norm))
        self._step_history.append(step)

        alerts = []

        if self.config.enable_loss_plateau:
            a = self._check_loss_plateau(step)
            if a:
                alerts.append(a)

        if self.config.enable_grad_plateau:
            a = self._check_grad_plateau(step)
            if a:
                alerts.append(a)

        if self.config.enable_grad_anomaly and grad_norm is not None:
            a = self._check_grad_spike(step, float(grad_norm))
            if a:
                alerts.append(a)

        if self.config.enable_utilization_check and utilization is not None:
            a = self._check_utilization(step, float(utilization))
            if a:
                alerts.append(a)

        # Meta-rule: multiple plateau signals → high-confidence stop recommendation
        # Check both current alerts AND recent eval plateau state
        plateau_codes = {"LOSS_PLATEAU", "GRAD_PLATEAU", "EVAL_PLATEAU"}
        active_plateaus = {a.code for a in alerts} & plateau_codes
        # Also count EVAL_PLATEAU if it was recently active (within last eval)
        if self._eval_plateau_active:
            active_plateaus.add("EVAL_PLATEAU")
        if len(active_plateaus) >= 2:
            a = self._emit_if_cool(
                step,
                MonitorAlert(
                    step=step,
                    code="CONSIDER_STOPPING",
                    severity="warning",
                    message=("Both loss and gradient norm have plateaued. Training dynamics appear frozen."),
                    details={"converging_signals": sorted(active_plateaus)},
                    recommendation=(
                        "Check validation metrics. If they've also plateaued, "
                        "training is unlikely to improve further. "
                        "Consider stopping and saving compute."
                    ),
                ),
            )
            if a:
                alerts.append(a)

        return alerts

    def update_eval(self, step: int, metrics: dict[str, Any]) -> list[MonitorAlert]:
        """Feed one evaluation's metrics. Returns any alerts triggered.

        Expected keys: loss (or eval_loss), accuracy, etc.
        """
        eval_loss = metrics.get("loss") or metrics.get("eval_loss")

        if eval_loss is not None and _is_finite(eval_loss):
            self._eval_loss_history.append(float(eval_loss))
            self._eval_step_history.append(step)

        alerts = []
        if self.config.enable_eval_plateau and eval_loss is not None:
            a = self._check_eval_plateau(step)
            if a:
                alerts.append(a)

        # Meta-rule: check if eval plateau + any train plateau → stop
        if self._eval_plateau_active:
            train_plateaus = set()
            if self._loss_stale_count >= self.config.loss_plateau_patience:
                train_plateaus.add("LOSS_PLATEAU")
            if self._grad_stale_count >= self.config.grad_plateau_patience:
                train_plateaus.add("GRAD_PLATEAU")
            if train_plateaus:
                converging = train_plateaus | {"EVAL_PLATEAU"}
                a = self._emit_if_cool(
                    step,
                    MonitorAlert(
                        step=step,
                        code="CONSIDER_STOPPING",
                        severity="warning",
                        message=(
                            "Validation loss and training dynamics have both plateaued. "
                            "Training is unlikely to improve further."
                        ),
                        details={"converging_signals": sorted(converging)},
                        recommendation=(
                            "Both validation metrics and training signals indicate "
                            "convergence. Consider stopping to save compute."
                        ),
                    ),
                )
                if a:
                    alerts.append(a)

        return alerts

    @property
    def n_steps(self) -> int:
        """Number of training steps processed."""
        return len(self._step_history)

    @property
    def n_evals(self) -> int:
        """Number of evaluations processed."""
        return len(self._eval_loss_history)

    # -------------------------------------------------------------------
    # Rule implementations
    # -------------------------------------------------------------------

    def _check_loss_plateau(self, step: int) -> MonitorAlert | None:
        """Check if training loss has plateaued."""
        if len(self._loss_history) < 2:
            return None

        recent = self._loss_history[-1]
        prev = self._loss_history[-2]

        if abs(prev) < 1e-12:
            rel_change = abs(recent) if abs(recent) > self.config.loss_plateau_delta else 0.0
        else:
            rel_change = abs((recent - prev) / prev)

        if rel_change < self.config.loss_plateau_delta:
            self._loss_stale_count += 1
        else:
            self._loss_stale_count = 0

        if self._loss_stale_count >= self.config.loss_plateau_patience:
            return self._emit_if_cool(
                step,
                MonitorAlert(
                    step=step,
                    code="LOSS_PLATEAU",
                    severity="info",
                    message=(
                        f"Training loss has plateaued for "
                        f"{self._loss_stale_count} consecutive steps "
                        f"(relative change < {self.config.loss_plateau_delta:.1%})"
                    ),
                    details={
                        "current_loss": recent,
                        "stale_count": self._loss_stale_count,
                        "relative_change": rel_change,
                    },
                    recommendation=(
                        "Training loss is no longer decreasing meaningfully. "
                        "Check if validation metrics are still improving. "
                        "If not, consider early stopping or reducing the learning rate."
                    ),
                ),
            )
        return None

    def _check_grad_plateau(self, step: int) -> MonitorAlert | None:
        """Check if gradient norm has plateaued."""
        if len(self._grad_norm_history) < 2:
            return None

        recent = self._grad_norm_history[-1]
        prev = self._grad_norm_history[-2]

        if abs(prev) < 1e-12:
            rel_change = 0.0
        else:
            rel_change = abs((recent - prev) / prev)

        if rel_change < self.config.grad_plateau_delta:
            self._grad_stale_count += 1
        else:
            self._grad_stale_count = 0

        if self._grad_stale_count >= self.config.grad_plateau_patience:
            return self._emit_if_cool(
                step,
                MonitorAlert(
                    step=step,
                    code="GRAD_PLATEAU",
                    severity="info",
                    message=(
                        f"Gradient norm has plateaued for "
                        f"{self._grad_stale_count} consecutive steps "
                        f"(relative change < {self.config.grad_plateau_delta:.1%})"
                    ),
                    details={
                        "current_grad_norm": recent,
                        "stale_count": self._grad_stale_count,
                        "relative_change": rel_change,
                    },
                    recommendation=(
                        "Gradient dynamics appear frozen. The optimizer may be "
                        "stuck in a flat region. Consider increasing the learning "
                        "rate or checking if training has converged."
                    ),
                ),
            )
        return None

    def _check_grad_spike(self, step: int, grad_norm: float) -> MonitorAlert | None:
        """Check for gradient norm spikes."""
        if len(self._grad_norm_history) < 5:
            return None

        # Running mean of recent grad norms (excluding current)
        history = (
            list(self._grad_norm_history)[:-1] if len(self._grad_norm_history) > 1 else list(self._grad_norm_history)
        )
        running_mean = sum(history) / len(history)

        if running_mean < 1e-12:
            return None

        ratio = grad_norm / running_mean

        if ratio > self.config.grad_spike_multiplier:
            return self._emit_if_cool(
                step,
                MonitorAlert(
                    step=step,
                    code="GRAD_SPIKE",
                    severity="warning",
                    message=(
                        f"Gradient norm spike: {grad_norm:.4f} ({ratio:.1f}x the running mean of {running_mean:.4f})"
                    ),
                    details={
                        "grad_norm": grad_norm,
                        "running_mean": running_mean,
                        "spike_ratio": ratio,
                    },
                    recommendation=(
                        "A sudden gradient spike may indicate an unstable batch, "
                        "learning rate too high, or a data anomaly. "
                        "If this recurs, consider reducing the learning rate or "
                        "enabling gradient clipping."
                    ),
                ),
            )
        return None

    def _check_eval_plateau(self, step: int) -> MonitorAlert | None:
        """Check if eval loss has stopped improving."""
        if len(self._eval_loss_history) < self.config.eval_patience + 1:
            return None

        self._eval_plateau_active = False  # reset; re-set below if still plateaued
        best_loss = min(self._eval_loss_history[: -self.config.eval_patience])
        recent_losses = self._eval_loss_history[-self.config.eval_patience :]

        # Check if any recent eval improved over the best
        if all(loss_val >= best_loss - 1e-8 for loss_val in recent_losses):
            self._eval_plateau_active = True
            return self._emit_if_cool(
                step,
                MonitorAlert(
                    step=step,
                    code="EVAL_PLATEAU",
                    severity="warning",
                    message=(
                        f"Validation loss has not improved for "
                        f"{self.config.eval_patience} consecutive evaluations "
                        f"(best: {best_loss:.6f}, recent: {recent_losses[-1]:.6f})"
                    ),
                    details={
                        "best_eval_loss": best_loss,
                        "current_eval_loss": recent_losses[-1],
                        "patience": self.config.eval_patience,
                        "n_evals": len(self._eval_loss_history),
                    },
                    recommendation=(
                        "Validation loss is no longer improving. This is a strong "
                        "signal that training should stop. Continuing risks "
                        "overfitting to the training data."
                    ),
                ),
            )
        return None

    def _check_utilization(self, step: int, utilization: float) -> MonitorAlert | None:
        """Check if LoRA rank utilization is low."""
        if utilization >= self.config.utilization_warn_threshold:
            return None

        return self._emit_if_cool(
            step,
            MonitorAlert(
                step=step,
                code="UTILIZATION_LOW",
                severity="info",
                message=(
                    f"LoRA rank utilization is {utilization:.1%} "
                    f"(below {self.config.utilization_warn_threshold:.0%} threshold)"
                ),
                details={
                    "utilization": utilization,
                    "threshold": self.config.utilization_warn_threshold,
                },
                recommendation=(
                    "The adapter is using less than a quarter of its allocated rank. "
                    "Consider compressing with `gradience compress` to reduce "
                    "memory and improve merge compatibility."
                ),
            ),
        )

    # -------------------------------------------------------------------
    # Cooldown management
    # -------------------------------------------------------------------

    def _emit_if_cool(self, step: int, alert: MonitorAlert) -> MonitorAlert | None:
        """Only emit alert if cooldown has elapsed since last same-code alert."""
        last = self._cooldowns.get(alert.code)
        if last is not None and (step - last) < self.config.alert_cooldown_steps:
            return None
        self._cooldowns[alert.code] = step
        return alert


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_finite(value: Any) -> bool:
    """Check if a value is a finite number."""
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False
