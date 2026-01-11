"""
Controller Module: High-Level Orchestration

Coordinates spectral monitoring, guard system, and telemetry into
a unified interface. This is the main entry point for non-HuggingFace
integrations.

Usage
-----
```python
from gradience import GradienceController

controller = GradienceController(out_dir="./logs", guard_enabled=True)
controller.initialize(model, save_fn, load_fn)

for step in training_loop:
    loss = train_step()
    
    # Returns False if training should abort
    if not controller.step(step, loss, model):
        break

controller.finalize()
```
"""

from __future__ import annotations
import os
import time
from dataclasses import dataclass, field
from typing import Optional, Callable, Dict, Any

from .spectral import (
    SpectralAnalyzer,
    SpectralSnapshot,
    estimate_condition_proxy,
    RiskLevel,
    RISK_POLICY,
)
from .guard import (
    Guard,
    GuardConfig,
    IntegrityStatus,
    check_model_for_nonfinite,
    unwrap_model,
)
from .telemetry import TelemetryWriter, SummaryGenerator


@dataclass
class GradienceController:
    """
    High-level controller for Gradience monitoring.
    
    Parameters
    ----------
    out_dir : str
        Output directory for telemetry and snapshots
    
    Guard Settings
    --------------
    guard_enabled : bool
        Enable checkpoint/rollback
    guard_snapshot_interval : int
        Steps between snapshots
    guard_violation_threshold : float
        Loss explosion threshold
    guard_max_rollbacks : int
        Max rollbacks before abort
    guard_rollback_window : int
        Window for anti-thrash detection
    guard_lr_backoff : float
        LR multiplier on rollback
    
    Spectral Settings
    -----------------
    spectral_enabled : bool
        Enable condition number tracking
    spectral_interval : int
        Steps between spectral measurements
    
    Telemetry Settings
    ------------------
    telemetry_interval : int
        Steps between telemetry records
    """
    out_dir: str = "./gradience_logs"
    
    # Guard
    guard_enabled: bool = True
    guard_snapshot_interval: int = 100
    guard_violation_threshold: float = 1e6
    guard_max_rollbacks: int = 3
    guard_rollback_window: int = 200
    guard_lr_backoff: float = 0.5
    
    # Spectral
    spectral_enabled: bool = True
    spectral_interval: int = 50
    
    # Telemetry
    telemetry_interval: int = 10
    
    # Internal components
    _guard: Optional[Guard] = field(default=None, repr=False)
    _spectral: Optional[SpectralAnalyzer] = field(default=None, repr=False)
    _telemetry: Optional[TelemetryWriter] = field(default=None, repr=False)
    _summary: Optional[SummaryGenerator] = field(default=None, repr=False)
    
    # State
    _model = None
    _current_loss: Optional[float] = None
    _initialized: bool = False
    
    def __post_init__(self):
        os.makedirs(self.out_dir, exist_ok=True)
        
        # Initialize components
        if self.spectral_enabled:
            self._spectral = SpectralAnalyzer()
        
        if self.guard_enabled:
            guard_config = GuardConfig(
                snapshot_interval=self.guard_snapshot_interval,
                snapshot_dir=os.path.join(self.out_dir, "snapshots"),
                violation_threshold=self.guard_violation_threshold,
                max_rollbacks_per_window=self.guard_max_rollbacks,
                rollback_window=self.guard_rollback_window,
                lr_backoff_factor=self.guard_lr_backoff,
            )
            self._guard = Guard(config=guard_config)
        
        self._telemetry = TelemetryWriter(out_dir=self.out_dir)
        self._summary = SummaryGenerator()
    
    def initialize(
        self,
        model,
        save_fn: Callable[[str], None],
        load_fn: Callable[[str], None],
        get_lr_fn: Optional[Callable[[], float]] = None,
        set_lr_fn: Optional[Callable[[float], None]] = None,
    ) -> None:
        """
        Initialize controller with model and callbacks.
        
        Parameters
        ----------
        model : torch.nn.Module
            The model being trained
        save_fn : callable
            Function to save model state to path
        load_fn : callable
            Function to load model state from path
        get_lr_fn : callable, optional
            Function to get current LR
        set_lr_fn : callable, optional
            Function to set LR
        """
        self._model = model
        
        # Initialize guard
        if self._guard is not None:
            self._guard.initialize(save_fn, load_fn, get_lr_fn, set_lr_fn)
        
        # Open telemetry
        self._telemetry.open()
        
        # Log initialization
        self._telemetry.log_event("initialized", {
            "guard_enabled": self.guard_enabled,
            "spectral_enabled": self.spectral_enabled,
            "guard_snapshot_interval": self.guard_snapshot_interval,
            "spectral_interval": self.spectral_interval,
        })
        
        # Initial snapshot
        if self._guard is not None:
            self._guard.maybe_snapshot(0, force=True)
            self._telemetry.log_event("snapshot_saved", {"step": 0, "type": "initial"})
        
        self._initialized = True
    
    def step(
        self,
        step: int,
        loss: float,
        model = None,
    ) -> bool:
        """
        Process a training step.
        
        Parameters
        ----------
        step : int
            Current global step
        loss : float
            Current training loss
        model : optional
            Model (if different from initialized)
        
        Returns
        -------
        continue_training : bool
            True to continue, False to abort
        """
        if not self._initialized:
            raise RuntimeError("Controller not initialized. Call initialize() first.")
        
        model = model or self._model
        self._current_loss = loss
        self._summary.total_steps = step
        self._summary.final_loss = loss
        
        # Check integrity
        if self._guard is not None:
            has_nonfinite = check_model_for_nonfinite(unwrap_model(model))
            
            continue_training = self._guard.check_integrity(
                step=step,
                loss=loss,
                has_nonfinite_weights=has_nonfinite,
            )
            
            if not continue_training:
                self._telemetry.log_event("training_aborted", {
                    "step": step,
                    "reason": self._guard.integrity.last_rollback_reason,
                })
                return False
            
            # Log recovery if it happened
            if self._guard.integrity.status == IntegrityStatus.RECOVERED:
                self._telemetry.log_event("rollback_success", {
                    "step": step,
                    "reason": self._guard.integrity.last_rollback_reason,
                    "lr_mult": self._guard._lr_multiplier,
                })
        
        # Maybe snapshot
        if self._guard is not None:
            snapshot_path = self._guard.maybe_snapshot(step)
            if snapshot_path:
                self._telemetry.log_event("snapshot_saved", {
                    "step": step,
                    "path": snapshot_path,
                })
        
        # Update spectral
        if self._spectral is not None and step % self.spectral_interval == 0:
            self._update_spectral(model, step)
        
        # Log telemetry
        if step % self.telemetry_interval == 0:
            self._log_telemetry(step)
        
        return True
    
    def _update_spectral(self, model, step: int) -> None:
        """Compute and record spectral metrics."""
        try:
            unwrapped = unwrap_model(model)
            
            # Find representative weight matrix
            target_param = None
            for name, param in unwrapped.named_parameters():
                if "weight" in name and len(param.shape) == 2:
                    target_param = param
                    break
            
            if target_param is None:
                return
            
            sigma_max, sigma_min, kappa = estimate_condition_proxy(target_param)
            
            snapshot = SpectralSnapshot(
                step=step,
                timestamp=time.time(),
                sigma_max=sigma_max,
                sigma_min=sigma_min,
                kappa_tilde=kappa,
            )
            self._spectral.add_observation(snapshot)
            
        except Exception:
            pass
    
    def _log_telemetry(self, step: int) -> None:
        """Write telemetry record."""
        spectral_data = self._spectral.get_telemetry() if self._spectral else {}
        guard_data = self._guard.get_telemetry() if self._guard else {}
        
        # Update summary
        if spectral_data:
            self._summary.update_from_spectral(spectral_data)
        if guard_data:
            self._summary.update_from_guard(guard_data)
        
        self._telemetry.log_telemetry(
            step=step,
            loss=self._current_loss,
            spectral=spectral_data,
            guard=guard_data,
        )
    
    def finalize(self) -> Dict[str, Any]:
        """
        Finalize training and generate summary.
        
        Returns
        -------
        summary : dict
            Training summary
        """
        # Final telemetry
        if self._summary.total_steps > 0:
            self._log_telemetry(self._summary.total_steps)
        
        # Generate summary
        summary = self._summary.generate()
        
        # Write summary
        self._summary.write(self.out_dir)
        
        # Log completion
        self._telemetry.log_event("training_complete", summary)
        
        # Close telemetry
        self._telemetry.close()
        
        return summary
    
    def get_risk_level(self) -> RiskLevel:
        """Get current risk level."""
        if self._spectral is None:
            return RiskLevel.STABLE
        risk, _ = self._spectral.assess_risk()
        return risk
    
    def get_policy(self) -> Dict[str, Any]:
        """Get recommended policy for current risk level."""
        return RISK_POLICY.get(self.get_risk_level(), {})
