"""
Telemetry replay through the training monitor.

Replays an existing JSONL telemetry file through TrainingMonitor,
returning all alerts that would have fired during live training.
Essential for prototyping heuristic rules and tuning thresholds
without needing GPU or live training runs.

Usage::

    from gradience.vnext.monitor_replay import replay_telemetry

    result = replay_telemetry("run.jsonl")
    print(f"Processed {result.n_steps_processed} steps")
    for alert in result.alerts:
        print(f"  Step {alert.step}: [{alert.code}] {alert.message}")
    if result.would_have_stopped:
        print(f"Would have stopped at step {result.would_have_stopped_step}")
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from gradience.vnext.monitor import MonitorAlert, MonitorConfig, TrainingMonitor


@dataclass
class ReplayResult:
    """Result of replaying telemetry through the monitor."""

    alerts: List[MonitorAlert]
    n_steps_processed: int
    n_evals_processed: int
    would_have_stopped: bool
    would_have_stopped_step: Optional[int]
    first_step: Optional[int]
    last_step: Optional[int]

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            f"Replay: {self.n_steps_processed} training steps, {self.n_evals_processed} evals",
            f"Steps: {self.first_step} → {self.last_step}",
            f"Alerts fired: {len(self.alerts)}",
        ]

        if self.alerts:
            # Group by code
            by_code: Dict[str, int] = {}
            for a in self.alerts:
                by_code[a.code] = by_code.get(a.code, 0) + 1
            for code, count in sorted(by_code.items()):
                first = next(a for a in self.alerts if a.code == code)
                lines.append(f"  {code}: {count}x (first at step {first.step})")

        if self.would_have_stopped:
            lines.append(f"Would have stopped at step {self.would_have_stopped_step}")
            if self.last_step and self.would_have_stopped_step:
                total = self.last_step - (self.first_step or 0)
                saved = self.last_step - self.would_have_stopped_step
                pct = (saved / total * 100) if total > 0 else 0
                lines.append(f"Training saved: {pct:.0f}% ({saved} steps)")
        else:
            lines.append("No stopping recommendation triggered.")

        return "\n".join(lines)


def replay_telemetry(
    telemetry_path: Union[str, Path],
    config: Optional[MonitorConfig] = None,
) -> ReplayResult:
    """Replay a JSONL telemetry file through the training monitor.

    Supports two telemetry formats:

    1. **Gradience v1 schema** (``schema: "gradience.vnext.telemetry/v1"``):
       Events have ``event`` field: ``"train_step"``, ``"eval"``, etc.

    2. **Study 7 flat format**: Each line is a flat dict with ``step``,
       ``train_loss``, ``val_loss``, ``grad_norm``, etc.

    Parameters
    ----------
    telemetry_path : path to JSONL file
    config : monitor configuration (uses defaults if None)

    Returns
    -------
    ReplayResult with all alerts, step counts, and stop recommendation.
    """
    path = Path(telemetry_path)
    monitor = TrainingMonitor(config)

    all_alerts: List[MonitorAlert] = []
    n_train = 0
    n_eval = 0
    first_step = None
    last_step = None

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            step = record.get("step")
            if step is None:
                continue
            step = int(step)

            if first_step is None:
                first_step = step
            last_step = step

            # Detect format and extract metrics
            event_type = record.get("event")

            if event_type == "train_step":
                # Gradience v1 schema
                metrics = {
                    "loss": record.get("loss"),
                    "grad_norm": record.get("grad_norm"),
                    "lr": record.get("lr"),
                }
                alerts = monitor.update(step, metrics)
                all_alerts.extend(alerts)
                n_train += 1

            elif event_type == "eval":
                # Gradience v1 eval event
                eval_metrics = record.get("metrics", {})
                alerts = monitor.update_eval(step, eval_metrics)
                all_alerts.extend(alerts)
                n_eval += 1

            elif event_type is None:
                # Flat format (Study 7 style)
                # Map to standard metric names
                metrics = {}
                for key_from, key_to in [
                    ("train_loss", "loss"),
                    ("loss", "loss"),
                    ("grad_norm", "grad_norm"),
                    ("weight_norm", "weight_norm"),
                ]:
                    if key_from in record:
                        metrics[key_to] = record[key_from]

                if metrics:
                    alerts = monitor.update(step, metrics)
                    all_alerts.extend(alerts)
                    n_train += 1

                # Also check eval-like fields
                eval_metrics = {}
                for key_from, key_to in [
                    ("val_loss", "loss"),
                    ("eval_loss", "loss"),
                    ("val_acc", "accuracy"),
                ]:
                    if key_from in record:
                        eval_metrics[key_to] = record[key_from]

                if eval_metrics:
                    alerts = monitor.update_eval(step, eval_metrics)
                    all_alerts.extend(alerts)
                    n_eval += 1

    # Determine if we would have stopped
    stop_alerts = [a for a in all_alerts if a.code == "CONSIDER_STOPPING"]
    would_stop = len(stop_alerts) > 0
    stop_step = stop_alerts[0].step if stop_alerts else None

    return ReplayResult(
        alerts=all_alerts,
        n_steps_processed=n_train,
        n_evals_processed=n_eval,
        would_have_stopped=would_stop,
        would_have_stopped_step=stop_step,
        first_step=first_step,
        last_step=last_step,
    )
