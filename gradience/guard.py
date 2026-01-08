"""
Gradience Guard - Training Integrity & Recovery

STABILITY: Experimental (single-GPU only)

Provides:
- Integrity monitoring (NaN/Inf detection in loss and weights)
- Automatic checkpoint/rollback on corruption
- Configurable recovery policies (LR dampening)
- Anti-thrash protection

Default: Shadow mode (detects and logs, but doesn't intervene)
Enable active mode with: Guard(shadow_mode=False)
"""

import os
import json
import time
import shutil
from enum import Enum
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Callable, Dict, Any, List
from collections import deque

import torch


class GuardEvent(Enum):
    """Events logged by Guard."""
    # Integrity events
    CORRUPTION_DETECTED = "corruption_detected"
    INTEGRITY_OK = "integrity_ok"
    
    # Shadow mode
    WOULD_ROLLBACK = "would_rollback"
    WOULD_DAMPEN_LR = "would_dampen_lr"
    
    # Active mode
    ROLLBACK_STARTED = "rollback_started"
    ROLLBACK_SUCCEEDED = "rollback_succeeded"
    ROLLBACK_FAILED = "rollback_failed"
    LR_DAMPENED = "lr_dampened"
    
    # Anti-thrash
    ABORTED_ANTI_THRASH = "aborted_anti_thrash"
    
    # Snapshots
    SNAPSHOT_CREATED = "snapshot_created"
    SNAPSHOT_PRUNED = "snapshot_pruned"
    SNAPSHOT_FAILED = "snapshot_failed"
    
    # Lifecycle
    GUARD_INITIALIZED = "guard_initialized"
    GUARD_DISABLED = "guard_disabled"


class CorruptionType(Enum):
    """Types of corruption detected."""
    NONFINITE_LOSS = "nonfinite_loss"
    NONFINITE_WEIGHTS = "nonfinite_weights"
    NONFINITE_GRADIENTS = "nonfinite_gradients"
    LOSS_EXPLOSION = "loss_explosion"


@dataclass
class GuardConfig:
    """
    Configuration for Guard behavior.
    
    Defaults are conservative (shadow mode, bounded storage).
    """
    # Operating mode
    shadow_mode: bool = True  # If True, log but don't intervene
    
    # Snapshot settings
    snapshot_interval: int = 100  # Steps between snapshots
    max_snapshots: int = 5  # Rolling window (bounded storage)
    snapshot_dir: str = "./gradience_snapshots"
    
    # Anti-thrash protection
    max_rollbacks_per_window: int = 3
    rollback_window_steps: int = 200
    
    # Recovery policy
    lr_dampen_factor: float = 0.5  # Multiply LR by this on rollback
    max_lr_dampens: int = 3  # Max times to dampen before abort
    
    # Detection thresholds
    loss_explosion_threshold: float = 1e6  # Loss above this = explosion
    check_weights: bool = True  # Check weights for NaN/Inf
    check_gradients: bool = False  # Check gradients (expensive)
    weight_check_interval: int = 10  # Check weights every N steps
    
    # === Muon Ratio (Structural Integrity) ===
    # ρ = weight_decay × σ_max
    # When ρ > threshold, expansion is winning over regularization
    check_muon_ratio: bool = True  # Enable muon ratio monitoring
    muon_ratio_threshold: float = 1.5  # Warn when ρ exceeds this
    muon_ratio_critical: float = 2.0  # Auto-dampen LR when ρ exceeds this
    muon_check_interval: int = 50  # Check muon ratio every N steps
    
    # Logging
    log_path: Optional[str] = None  # If set, write events to JSONL


@dataclass
class Snapshot:
    """A saved training state."""
    step: int
    path: str
    timestamp: float
    loss: Optional[float] = None


@dataclass 
class GuardState:
    """Internal state tracking."""
    step: int = 0
    rollback_count: int = 0
    lr_dampen_count: int = 0
    recent_rollback_steps: List[int] = field(default_factory=list)
    last_known_good_loss: Optional[float] = None
    is_aborted: bool = False
    abort_reason: Optional[str] = None


class Guard:
    """
    Training integrity monitor with automatic recovery.
    
    EXPERIMENTAL: Single-GPU only. Distributed training not validated.
    
    Usage (shadow mode - recommended for first use):
        guard = Guard(model, optimizer)
        
        for step, batch in enumerate(dataloader):
            loss = train_step(batch)
            
            status = guard.step(step, loss)
            if status.should_abort:
                print(f"Guard aborted: {status.reason}")
                break
    
    Usage (active mode):
        guard = Guard(model, optimizer, config=GuardConfig(shadow_mode=False))
        # Same loop - Guard will now intervene on corruption
    
    Events are logged to stderr and optionally to a JSONL file.
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        config: Optional[GuardConfig] = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.config = config or GuardConfig()
        
        self._state = GuardState()
        self._snapshots: deque = deque(maxlen=self.config.max_snapshots)
        self._log_file = None
        
        # Setup snapshot directory
        self._snapshot_dir = Path(self.config.snapshot_dir)
        if not self.config.shadow_mode:
            self._snapshot_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        if self.config.log_path:
            self._log_file = open(self.config.log_path, "a")
        
        self._log_event(GuardEvent.GUARD_INITIALIZED, {
            "shadow_mode": self.config.shadow_mode,
            "snapshot_interval": self.config.snapshot_interval,
            "max_snapshots": self.config.max_snapshots,
            "max_rollbacks_per_window": self.config.max_rollbacks_per_window,
        })
    
    def step(self, step: int, loss: float) -> "GuardStatus":
        """
        Check training integrity and take action if needed.
        
        Call this after each training step.
        
        Args:
            step: Current training step
            loss: Current loss value
            
        Returns:
            GuardStatus with should_abort, should_continue, and diagnostics
        """
        self._state.step = step
        
        # Check if already aborted
        if self._state.is_aborted:
            return GuardStatus(
                should_abort=True,
                should_continue=False,
                reason=self._state.abort_reason,
            )
        
        # Check integrity
        corruption = self._check_integrity(step, loss)
        
        if corruption:
            return self._handle_corruption(step, loss, corruption)
        
        # Update last known good
        if loss is not None and not self._is_nonfinite(loss):
            self._state.last_known_good_loss = loss
        
        # Check muon ratio (structural integrity)
        if self.config.check_muon_ratio and step % self.config.muon_check_interval == 0:
            muon_status = self._check_muon_ratio(step)
            if muon_status is not None:
                return muon_status
        
        # Maybe create snapshot
        if step > 0 and step % self.config.snapshot_interval == 0:
            self._maybe_create_snapshot(step, loss)
        
        return GuardStatus(
            should_abort=False,
            should_continue=True,
            integrity_ok=True,
        )
    
    def _check_integrity(self, step: int, loss: float) -> Optional[CorruptionType]:
        """Check for training corruption."""
        
        # Check loss
        if self._is_nonfinite(loss):
            return CorruptionType.NONFINITE_LOSS
        
        if loss > self.config.loss_explosion_threshold:
            return CorruptionType.LOSS_EXPLOSION
        
        # Check weights periodically
        if self.config.check_weights and step % self.config.weight_check_interval == 0:
            if self._has_nonfinite_weights():
                return CorruptionType.NONFINITE_WEIGHTS
        
        # Check gradients if enabled
        if self.config.check_gradients:
            if self._has_nonfinite_gradients():
                return CorruptionType.NONFINITE_GRADIENTS
        
        return None
    
    def _check_muon_ratio(self, step: int) -> Optional["GuardStatus"]:
        """
        Check muon ratio (structural integrity).
        
        ρ = weight_decay × σ_max
        
        When ρ > 1: Expansion is winning over regularization.
        When ρ > critical: Auto-dampen LR to restore balance.
        
        Returns:
            GuardStatus if action needed, None otherwise
        """
        # Get weight decay from optimizer
        weight_decay = self._get_weight_decay()
        if weight_decay <= 0:
            return None  # Can't compute muon ratio without weight decay
        
        # Compute σ_max (mean spectral norm across layers)
        sigma_max = self._compute_mean_sigma_max()
        if sigma_max <= 0:
            return None
        
        # Muon ratio
        muon_ratio = weight_decay * sigma_max
        
        # Log it
        self._log_event(GuardEvent.INTEGRITY_OK, {
            "step": step,
            "muon_ratio": muon_ratio,
            "weight_decay": weight_decay,
            "sigma_max": sigma_max,
        })
        
        # Check thresholds
        if muon_ratio > self.config.muon_ratio_critical:
            # Critical: auto-dampen LR
            if self.config.shadow_mode:
                self._log_event(GuardEvent.WOULD_DAMPEN_LR, {
                    "step": step,
                    "reason": "muon_ratio_critical",
                    "muon_ratio": muon_ratio,
                    "threshold": self.config.muon_ratio_critical,
                })
                return None  # Don't intervene in shadow mode
            else:
                # Active mode: dampen LR
                old_lr = self._dampen_lr()
                self._log_event(GuardEvent.LR_DAMPENED, {
                    "step": step,
                    "reason": "muon_ratio_critical",
                    "muon_ratio": muon_ratio,
                    "old_lr": old_lr,
                    "new_lr": old_lr * self.config.lr_dampen_factor,
                })
                return GuardStatus(
                    should_abort=False,
                    should_continue=True,
                    muon_ratio=muon_ratio,
                    action_taken="lr_dampened_muon_ratio",
                )
        
        elif muon_ratio > self.config.muon_ratio_threshold:
            # Warning level: log but don't intervene
            self._log_event(GuardEvent.INTEGRITY_OK, {
                "step": step,
                "warning": "muon_ratio_high",
                "muon_ratio": muon_ratio,
                "threshold": self.config.muon_ratio_threshold,
                "suggestion": f"Consider reducing LR by {muon_ratio:.1f}x",
            })
        
        return None
    
    def _get_weight_decay(self) -> float:
        """Get weight decay from optimizer."""
        if self.optimizer is None:
            return 0.0
        
        # Check defaults
        if hasattr(self.optimizer, 'defaults'):
            wd = self.optimizer.defaults.get('weight_decay', 0.0)
            if wd > 0:
                return wd
        
        # Check param groups
        for group in self.optimizer.param_groups:
            wd = group.get('weight_decay', 0.0)
            if wd > 0:
                return wd
        
        return 0.0
    
    def _compute_mean_sigma_max(self) -> float:
        """Compute mean spectral norm across weight matrices."""
        sigma_maxs = []
        
        for name, param in self.model.named_parameters():
            if param.dim() < 2:
                continue
            if min(param.shape) < 10:
                continue
            if 'weight' not in name:
                continue
            
            with torch.no_grad():
                W = param.float()
                if W.dim() > 2:
                    W = W.view(W.size(0), -1)
                
                try:
                    # Use power iteration for efficiency
                    sigma_max = self._approx_sigma_max(W)
                    sigma_maxs.append(sigma_max)
                except Exception:
                    continue
        
        if not sigma_maxs:
            return 0.0
        
        return sum(sigma_maxs) / len(sigma_maxs)
    
    def _approx_sigma_max(self, W: torch.Tensor, n_iters: int = 5) -> float:
        """Approximate spectral norm using power iteration."""
        v = torch.randn(W.size(1), device=W.device, dtype=W.dtype)
        v = v / v.norm()
        
        for _ in range(n_iters):
            u = W @ v
            u = u / (u.norm() + 1e-10)
            v = W.t() @ u
            v = v / (v.norm() + 1e-10)
        
        sigma = (u @ W @ v).item()
        return abs(sigma)
    
    def _handle_corruption(
        self, 
        step: int, 
        loss: float, 
        corruption: CorruptionType
    ) -> "GuardStatus":
        """Handle detected corruption."""
        
        self._log_event(GuardEvent.CORRUPTION_DETECTED, {
            "step": step,
            "loss": float(loss) if not self._is_nonfinite(loss) else "nonfinite",
            "corruption_type": corruption.value,
        })
        
        # Shadow mode: log but don't intervene
        if self.config.shadow_mode:
            self._log_event(GuardEvent.WOULD_ROLLBACK, {
                "step": step,
                "corruption_type": corruption.value,
                "snapshots_available": len(self._snapshots),
            })
            
            return GuardStatus(
                should_abort=False,
                should_continue=True,  # Continue in shadow mode
                integrity_ok=False,
                corruption_detected=corruption,
                action_taken="logged_only_shadow_mode",
            )
        
        # Active mode: attempt recovery
        return self._attempt_recovery(step, corruption)
    
    def _attempt_recovery(
        self, 
        step: int, 
        corruption: CorruptionType
    ) -> "GuardStatus":
        """Attempt to recover from corruption via rollback."""
        
        # Check anti-thrash
        self._prune_old_rollbacks(step)
        
        if len(self._state.recent_rollback_steps) >= self.config.max_rollbacks_per_window:
            self._state.is_aborted = True
            self._state.abort_reason = (
                f"Anti-thrash triggered: {len(self._state.recent_rollback_steps)} rollbacks "
                f"in {self.config.rollback_window_steps} steps. "
                f"Possible persistent instability or configuration error."
            )
            
            self._log_event(GuardEvent.ABORTED_ANTI_THRASH, {
                "step": step,
                "rollback_count": len(self._state.recent_rollback_steps),
                "window_steps": self.config.rollback_window_steps,
            })
            
            return GuardStatus(
                should_abort=True,
                should_continue=False,
                reason=self._state.abort_reason,
            )
        
        # Check if we have snapshots
        if not self._snapshots:
            self._state.is_aborted = True
            self._state.abort_reason = "Corruption detected but no snapshots available for rollback."
            
            return GuardStatus(
                should_abort=True,
                should_continue=False,
                reason=self._state.abort_reason,
            )
        
        # Attempt rollback
        snapshot = self._snapshots[-1]
        
        self._log_event(GuardEvent.ROLLBACK_STARTED, {
            "step": step,
            "rollback_to_step": snapshot.step,
            "corruption_type": corruption.value,
        })
        
        try:
            self._load_snapshot(snapshot)
            self._state.rollback_count += 1
            self._state.recent_rollback_steps.append(step)
            
            # Apply LR dampening
            if self._state.lr_dampen_count < self.config.max_lr_dampens:
                self._dampen_lr()
                self._state.lr_dampen_count += 1
            
            self._log_event(GuardEvent.ROLLBACK_SUCCEEDED, {
                "step": step,
                "rolled_back_to": snapshot.step,
                "total_rollbacks": self._state.rollback_count,
            })
            
            return GuardStatus(
                should_abort=False,
                should_continue=True,
                integrity_ok=False,
                corruption_detected=corruption,
                action_taken="rollback_succeeded",
                rolled_back_to_step=snapshot.step,
            )
            
        except Exception as e:
            self._state.is_aborted = True
            self._state.abort_reason = f"Rollback failed: {e}"
            
            self._log_event(GuardEvent.ROLLBACK_FAILED, {
                "step": step,
                "error": str(e),
            })
            
            return GuardStatus(
                should_abort=True,
                should_continue=False,
                reason=self._state.abort_reason,
            )
    
    def _maybe_create_snapshot(self, step: int, loss: float):
        """Create a snapshot if in active mode."""
        
        if self.config.shadow_mode:
            return  # No snapshots in shadow mode
        
        try:
            # Check disk space (rough estimate)
            snapshot_path = self._snapshot_dir / f"snapshot_step_{step}.pt"
            
            # Prune old snapshots first if at capacity
            while len(self._snapshots) >= self.config.max_snapshots:
                old = self._snapshots.popleft()
                self._delete_snapshot(old)
            
            # Save new snapshot
            torch.save({
                "step": step,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            }, snapshot_path)
            
            snapshot = Snapshot(
                step=step,
                path=str(snapshot_path),
                timestamp=time.time(),
                loss=loss,
            )
            self._snapshots.append(snapshot)
            
            self._log_event(GuardEvent.SNAPSHOT_CREATED, {
                "step": step,
                "path": str(snapshot_path),
                "total_snapshots": len(self._snapshots),
            })
            
        except Exception as e:
            self._log_event(GuardEvent.SNAPSHOT_FAILED, {
                "step": step,
                "error": str(e),
            })
    
    def _load_snapshot(self, snapshot: Snapshot):
        """Load model and optimizer state from snapshot."""
        checkpoint = torch.load(snapshot.path)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    def _delete_snapshot(self, snapshot: Snapshot):
        """Delete a snapshot file."""
        try:
            os.remove(snapshot.path)
            self._log_event(GuardEvent.SNAPSHOT_PRUNED, {
                "step": snapshot.step,
                "path": snapshot.path,
            })
        except:
            pass  # Best effort
    
    def _dampen_lr(self):
        """Reduce learning rate as recovery policy."""
        for param_group in self.optimizer.param_groups:
            old_lr = param_group['lr']
            param_group['lr'] *= self.config.lr_dampen_factor
            
            self._log_event(GuardEvent.LR_DAMPENED, {
                "old_lr": old_lr,
                "new_lr": param_group['lr'],
                "factor": self.config.lr_dampen_factor,
                "dampen_count": self._state.lr_dampen_count + 1,
            })
    
    def _prune_old_rollbacks(self, current_step: int):
        """Remove rollbacks outside the anti-thrash window."""
        cutoff = current_step - self.config.rollback_window_steps
        self._state.recent_rollback_steps = [
            s for s in self._state.recent_rollback_steps if s > cutoff
        ]
    
    def _is_nonfinite(self, value: float) -> bool:
        """Check if a value is NaN or Inf."""
        if value is None:
            return False
        try:
            return not (value == value) or abs(value) == float('inf')  # NaN != NaN
        except:
            return True
    
    def _has_nonfinite_weights(self) -> bool:
        """Check if any model weights are NaN or Inf."""
        for param in self.model.parameters():
            if torch.isnan(param).any() or torch.isinf(param).any():
                return True
        return False
    
    def _has_nonfinite_gradients(self) -> bool:
        """Check if any gradients are NaN or Inf."""
        for param in self.model.parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                    return True
        return False
    
    def _log_event(self, event: GuardEvent, data: Dict[str, Any]):
        """Log an event to stderr and optionally to file."""
        record = {
            "timestamp": time.time(),
            "event": event.value,
            **data,
        }
        
        # Always print significant events
        if event in (
            GuardEvent.CORRUPTION_DETECTED,
            GuardEvent.ROLLBACK_STARTED,
            GuardEvent.ROLLBACK_SUCCEEDED,
            GuardEvent.ROLLBACK_FAILED,
            GuardEvent.ABORTED_ANTI_THRASH,
            GuardEvent.WOULD_ROLLBACK,
        ):
            print(f"[Guard] {event.value}: {data}", flush=True)
        
        # Write to log file if configured
        if self._log_file:
            self._log_file.write(json.dumps(record) + "\n")
            self._log_file.flush()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current Guard statistics."""
        return {
            "step": self._state.step,
            "rollback_count": self._state.rollback_count,
            "lr_dampen_count": self._state.lr_dampen_count,
            "snapshots_held": len(self._snapshots),
            "is_aborted": self._state.is_aborted,
            "abort_reason": self._state.abort_reason,
            "shadow_mode": self.config.shadow_mode,
        }
    
    def cleanup(self):
        """Clean up resources. Call when training ends."""
        if self._log_file:
            self._log_file.close()
        
        # Optionally clean up snapshots
        # (leave them by default for debugging)
    
    def __del__(self):
        """Ensure cleanup on deletion."""
        try:
            self.cleanup()
        except:
            pass


@dataclass
class GuardStatus:
    """Status returned from Guard.step()."""
    should_abort: bool
    should_continue: bool
    integrity_ok: bool = True
    corruption_detected: Optional[CorruptionType] = None
    action_taken: Optional[str] = None
    rolled_back_to_step: Optional[int] = None
    reason: Optional[str] = None
    muon_ratio: Optional[float] = None  # ρ = λ × σ_max


# Convenience function for simple usage
def create_guard(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    shadow_mode: bool = True,
    log_path: Optional[str] = None,
    **kwargs,
) -> Guard:
    """
    Create a Guard with sensible defaults.
    
    Args:
        model: PyTorch model to monitor
        optimizer: Optimizer to checkpoint/restore
        shadow_mode: If True (default), detect but don't intervene
        log_path: Optional path for event log (JSONL)
        **kwargs: Additional GuardConfig options
        
    Returns:
        Configured Guard instance
        
    Example:
        guard = create_guard(model, optimizer)
        
        for step, batch in enumerate(dataloader):
            loss = train_step(batch)
            status = guard.step(step, loss)
            if status.should_abort:
                break
    """
    config = GuardConfig(
        shadow_mode=shadow_mode,
        log_path=log_path,
        **kwargs,
    )
    return Guard(model, optimizer, config)
# ---------------------------------------------------------------------
# === LEGACY API COMPAT (tests/test_core.py expects these names) ===
# Backwards-compatible export (tests + older code expect IntegrityStatus)
# ---------------------------------------------------------------------
if "IntegrityStatus" not in globals():
    # Try to alias an existing status/result class if present
    for _cand in (
        "GuardStatus",
        "GuardStepStatus",
        "GuardStepResult",
        "GuardStepOutcome",
        "StepStatus",
        "Status",
    ):
        if _cand in globals():
            IntegrityStatus = globals()[_cand]
            break
    else:
        # Minimal fallback so imports don't fail (fields used in tests are common)
        from dataclasses import dataclass
        from typing import Optional, Dict, Any

        @dataclass
        class IntegrityStatus:
            ok: bool = True
            corrupted: bool = False
            should_abort: bool = False
            reason: str = ""
            action: Optional[str] = None
            rollback_performed: bool = False
            lr_dampened: bool = False
            details: Optional[Dict[str, Any]] = None

# Ensure it is visible for star-import patterns if __all__ exists
try:
    __all__  # type: ignore
except NameError:
    __all__ = []  # type: ignore
if isinstance(__all__, list) and "IntegrityStatus" not in __all__:
    __all__.append("IntegrityStatus")  # type: ignore
# ---------------------------------------------------------------------
# Backwards-compatible export (tests + older code expect IntegrityTracker)
# ---------------------------------------------------------------------
if "IntegrityTracker" not in globals():
    # Try to alias an existing tracker/monitor class if present
    for _cand in (
        "Guard",                 # sometimes older tests used Guard as tracker
        "IntegrityMonitor",
        "IntegrityLogger",
        "IntegrityState",
        "IntegrityHistory",
        "GuardIntegrityTracker",
        "GuardTracker",
    ):
        if _cand in globals():
            IntegrityTracker = globals()[_cand]
            break
    else:
        # Minimal fallback tracker that records integrity events.
        from dataclasses import dataclass, field
        from typing import Any, Dict, List, Optional

        @dataclass
        class _IntegrityEvent:
            step: Optional[int] = None
            code: str = ""
            severity: str = "info"
            message: str = ""
            context: Dict[str, Any] = field(default_factory=dict)

        class IntegrityTracker:
            """
            Back-compat shim for older Guard APIs/tests.

            Stores events + counts and supports a few common method names
            (record/add_event/log_event/update/summary/reset).
            """
            def __init__(self) -> None:
                self.events: List[_IntegrityEvent] = []
                self.counts: Dict[str, int] = {}

            def record(
                self,
                code: str,
                severity: str = "info",
                message: str = "",
                step: Optional[int] = None,
                context: Optional[Dict[str, Any]] = None,
            ) -> None:
                self.events.append(_IntegrityEvent(
                    step=step, code=code, severity=severity, message=message, context=context or {}
                ))
                self.counts[code] = self.counts.get(code, 0) + 1

            # Common aliases
            def add_event(self, *args, **kwargs):  # type: ignore
                return self.record(*args, **kwargs)

            def log_event(self, *args, **kwargs):  # type: ignore
                return self.record(*args, **kwargs)

            def update(self, status: Any, step: Optional[int] = None) -> None:
                # If passed an IntegrityStatus-like object, log when it looks bad.
                corrupted = bool(getattr(status, "corrupted", False))
                should_abort = bool(getattr(status, "should_abort", False))
                if corrupted or should_abort:
                    code = getattr(status, "action", None) or "corruption"
                    reason = getattr(status, "reason", "") or "integrity issue"
                    details = getattr(status, "details", None)
                    self.record(code=str(code), severity="warning", message=str(reason), step=step, context=details or {})

            def summary(self) -> Dict[str, Any]:
                return {"n_events": len(self.events), "counts": dict(self.counts)}

            def reset(self) -> None:
                self.events.clear()
                self.counts.clear()

# Ensure it is visible for star-import patterns if __all__ exists
try:
    __all__  # type: ignore
except NameError:
    __all__ = []  # type: ignore
if isinstance(__all__, list) and "IntegrityTracker" not in __all__:
    __all__.append("IntegrityTracker")  # type: ignore
