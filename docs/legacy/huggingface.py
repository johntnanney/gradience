"""
HuggingFace Trainer Integration

Provides GradienceCallback for seamless integration with HuggingFace Trainer.

Usage
-----
```python
from gradience.integrations.huggingface import GradienceCallback

callback = GradienceCallback(
    out_dir="./gradience_logs",
    guard_enabled=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    callbacks=[callback],
)
trainer.train()
```
"""

from __future__ import annotations
import os
from dataclasses import dataclass, field
from typing import Optional, Any

# Import from parent package
from ..controller import GradienceController
from ..guard import unwrap_model


@dataclass
class GradienceCallback:
    """
    HuggingFace Trainer callback for Gradience monitoring.
    
    All parameters are passed through to GradienceController.
    See GradienceController for full documentation.
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
    
    # Internal
    _controller: Optional[GradienceController] = field(default=None, repr=False)
    _trainer: Any = field(default=None, repr=False)
    _current_loss: Optional[float] = field(default=None, repr=False)
    
    def __post_init__(self):
        self._controller = GradienceController(
            out_dir=self.out_dir,
            guard_enabled=self.guard_enabled,
            guard_snapshot_interval=self.guard_snapshot_interval,
            guard_violation_threshold=self.guard_violation_threshold,
            guard_max_rollbacks=self.guard_max_rollbacks,
            guard_rollback_window=self.guard_rollback_window,
            guard_lr_backoff=self.guard_lr_backoff,
            spectral_enabled=self.spectral_enabled,
            spectral_interval=self.spectral_interval,
            telemetry_interval=self.telemetry_interval,
        )
    
    # =========================================================================
    # HuggingFace Trainer Callback Interface
    # =========================================================================
    
    def on_train_begin(self, args, state, control, model=None, **kwargs):
        """Initialize controller with model."""
        self._trainer = kwargs.get("trainer")
        
        if model is not None:
            self._controller.initialize(
                model=model,
                save_fn=self._make_save_fn(model),
                load_fn=self._make_load_fn(model),
                get_lr_fn=self._get_lr,
                set_lr_fn=self._set_lr,
            )
    
    def on_step_end(self, args, state, control, model=None, **kwargs):
        """Process each training step."""
        loss = self._extract_loss(state)
        if loss is not None:
            self._current_loss = loss
        
        continue_training = self._controller.step(
            step=state.global_step,
            loss=self._current_loss or 0.0,
            model=model,
        )
        
        if not continue_training:
            control.should_training_stop = True
        
        return control
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Capture loss from trainer logs."""
        if logs and "loss" in logs:
            self._current_loss = logs["loss"]
    
    def on_train_end(self, args, state, control, **kwargs):
        """Finalize and write summary."""
        self._controller.finalize()
    
    # =========================================================================
    # Helper Methods
    # =========================================================================
    
    def _make_save_fn(self, model):
        """Create save function for model."""
        def save_fn(path: str):
            os.makedirs(path, exist_ok=True)
            
            if self._trainer is not None and hasattr(self._trainer, "save_model"):
                self._trainer.save_model(path)
            else:
                try:
                    import torch
                    unwrapped = unwrap_model(model)
                    torch.save(
                        unwrapped.state_dict(),
                        os.path.join(path, "model.pt")
                    )
                except Exception:
                    pass
        
        return save_fn
    
    def _make_load_fn(self, model):
        """Create load function for model."""
        def load_fn(path: str):
            import torch
            
            unwrapped = unwrap_model(model)
            
            # Try different checkpoint formats
            for filename in ["pytorch_model.bin", "model.pt", "model.safetensors"]:
                filepath = os.path.join(path, filename)
                if os.path.exists(filepath):
                    if filename.endswith(".safetensors"):
                        from safetensors.torch import load_file
                        state_dict = load_file(filepath)
                    else:
                        state_dict = torch.load(filepath, map_location="cpu")
                    unwrapped.load_state_dict(state_dict)
                    return
        
        return load_fn
    
    def _get_lr(self) -> Optional[float]:
        """Get current learning rate."""
        if self._trainer and hasattr(self._trainer, "optimizer"):
            opt = self._trainer.optimizer
            if opt and len(opt.param_groups) > 0:
                return opt.param_groups[0]["lr"]
        return None
    
    def _set_lr(self, new_lr: float) -> None:
        """Set learning rate."""
        if self._trainer and hasattr(self._trainer, "optimizer"):
            opt = self._trainer.optimizer
            if opt:
                for pg in opt.param_groups:
                    pg["lr"] = new_lr
    
    def _extract_loss(self, state) -> Optional[float]:
        """Extract loss from trainer state."""
        if hasattr(state, "log_history") and state.log_history:
            for entry in reversed(state.log_history):
                if "loss" in entry:
                    return entry["loss"]
        return self._current_loss
