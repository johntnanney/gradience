from __future__ import annotations

from dataclasses import dataclass
from collections import deque
from typing import Any, Deque, Dict, Optional, Tuple
import time

import torch


@dataclass(frozen=True)
class Snapshot:
    """
    A lightweight snapshot of adapter state.

    adapter_state: CPU tensors only (to keep GPU memory stable).
    """
    step: int
    ts: float
    adapter_state: Dict[str, torch.Tensor]
    loss: Optional[float] = None


class LoRAGuard:
    """
    LoRA Guard: snapshot + rollback for adapter-only fine-tuning.

    Design:
    - weights-only snapshots (LoRA params only)
    - snapshots stored on CPU (cheap and stable)
    - anti-thrash: cooldown + max rollbacks per window
    - conservative rollback: restore without discarding by default
      (optionally prune snapshots newer than restored step)
    """

    def __init__(
        self,
        *,
        ring_size: int = 5,
        grad_threshold: float = 100.0,
        snapshot_on_cpu: bool = True,
        cooldown_steps: int = 20,
        max_rollbacks: int = 3,
        window_steps: int = 200,
        prune_newer_on_rollback: bool = True,
    ):
        if ring_size <= 0:
            raise ValueError("ring_size must be > 0")
        if cooldown_steps < 0:
            raise ValueError("cooldown_steps must be >= 0")
        if max_rollbacks <= 0:
            raise ValueError("max_rollbacks must be > 0")
        if window_steps <= 0:
            raise ValueError("window_steps must be > 0")

        self.ring_size = int(ring_size)
        self.grad_threshold = float(grad_threshold)
        self.snapshot_on_cpu = bool(snapshot_on_cpu)

        self.cooldown_steps = int(cooldown_steps)
        self.max_rollbacks = int(max_rollbacks)
        self.window_steps = int(window_steps)
        self.prune_newer_on_rollback = bool(prune_newer_on_rollback)

        self.snapshots: Deque[Snapshot] = deque(maxlen=self.ring_size)

        # Anti-thrash state
        self._last_rollback_step: Optional[int] = None
        self._rollback_events: Deque[int] = deque()  # steps where rollback occurred
        self.n_rollbacks: int = 0

    # -------------------------
    # Snapshot / rollback
    # -------------------------

    def snapshot(self, step: int, model: Any, loss: Optional[float] = None) -> None:
        """
        Save LoRA adapter weights only.

        Stored on CPU to avoid GPU memory churn and to make ring buffers cheap.
        """
        adapter_state: Dict[str, torch.Tensor] = {}

        for name, param in model.named_parameters():
            if "lora" in name.lower():
                t = param.detach()
                if self.snapshot_on_cpu:
                    t = t.to("cpu", non_blocking=False)
                adapter_state[name] = t.clone()

        if not adapter_state:
            # Not an error: model might not be LoRA, or naming differs.
            # Guard will just be inert.
            return

        self.snapshots.append(
            Snapshot(
                step=int(step),
                ts=time.time(),
                adapter_state=adapter_state,
                loss=float(loss) if loss is not None else None,
            )
        )

    def rollback(self, model: Any, *, steps_back: int = 1) -> Optional[int]:
        """
        Restore adapter weights from a previous snapshot.

        By default, restores without discarding the restored snapshot.
        If prune_newer_on_rollback=True, we discard snapshots that are newer than the restored step
        (prevents oscillations that re-trigger immediately).

        Returns the restored step, or None if insufficient snapshots exist.
        """
        if steps_back <= 0:
            raise ValueError("steps_back must be >= 1")

        if len(self.snapshots) < steps_back:
            return None

        idx = len(self.snapshots) - steps_back
        snap = list(self.snapshots)[idx]

        # Restore weights
        name_to_param = dict(model.named_parameters())
        for name, saved in snap.adapter_state.items():
            p = name_to_param.get(name)
            if p is None:
                continue
            # Copy to param device
            saved_t = saved.to(p.device)
            p.data.copy_(saved_t)

        self.n_rollbacks += 1
        self._last_rollback_step = snap.step
        self._rollback_events.append(int(snap.step))

        # Prune snapshots newer than restored step (optional)
        if self.prune_newer_on_rollback:
            self._prune_newer_than(snap.step)

        return snap.step

    def _prune_newer_than(self, step: int) -> None:
        """Remove snapshots with step > given step."""
        kept = [s for s in self.snapshots if s.step <= step]
        self.snapshots = deque(kept, maxlen=self.ring_size)

    # -------------------------
    # Trigger logic
    # -------------------------

    def check_triggers(
        self,
        *,
        loss: Optional[float] = None,
        grad_norm: Optional[float] = None,
    ) -> Optional[str]:
        """
        Return trigger name if activated; else None.

        Conservative triggers:
        - NaN/Inf loss
        - NaN/Inf grad_norm
        - grad_norm > threshold
        """
        if loss is not None:
            try:
                if not torch.isfinite(torch.tensor(loss)):
                    return "nan_loss"
            except Exception:
                # If loss can't be converted, ignore it.
                pass

        if grad_norm is not None:
            try:
                if not torch.isfinite(torch.tensor(grad_norm)):
                    return "nan_grad"
            except Exception:
                pass

            try:
                if float(grad_norm) > self.grad_threshold:
                    return "grad_explosion"
            except Exception:
                pass

        return None

    # -------------------------
    # Anti-thrash policy
    # -------------------------

    def can_attempt_rollback(self, step: int) -> bool:
        """
        Decide whether a rollback is allowed at this step.

        Rules:
        - cooldown: disallow rollback if too soon after last rollback
        - max_rollbacks per rolling window: disallow if too many rollbacks recently
        """
        step = int(step)

        # Cooldown check
        if self._last_rollback_step is not None:
            if step - self._last_rollback_step < self.cooldown_steps:
                return False

        # Windowed max rollbacks check
        # Keep only rollback steps within the last `window_steps`
        while self._rollback_events and (step - self._rollback_events[0] > self.window_steps):
            self._rollback_events.popleft()

        if len(self._rollback_events) >= self.max_rollbacks:
            return False

        return True

    # -------------------------
    # Convenience
    # -------------------------

    def snapshot_count(self) -> int:
        return len(self.snapshots)

    def memory_usage_mb(self) -> float:
        """Estimate CPU memory usage of stored snapshots."""
        total_bytes = 0
        for snap in self.snapshots:
            for t in snap.adapter_state.values():
                total_bytes += t.numel() * t.element_size()
        return total_bytes / (1024 * 1024)
