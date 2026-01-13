"""
Reusable CPU driver for HuggingFace callback testing.

This module provides a clean interface for testing HF callbacks without
requiring real Trainer infrastructure, model downloads, or GPU resources.
"""

import json
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from unittest.mock import Mock
from dataclasses import dataclass

import torch
import torch.nn as nn


# ============================================================================
# Minimal Mock Objects
# ============================================================================

class MockModel(nn.Module):
    """Minimal model with LoRA-like parameters for testing."""
    
    def __init__(self, num_params: int = 1280):
        super().__init__()
        # Create a mix of LoRA and base parameters
        self.lora_A = nn.Parameter(torch.randn(16, 8) * 0.01)
        self.lora_B = nn.Parameter(torch.randn(8, 16) * 0.01)
        self.base_weight = nn.Parameter(torch.randn(32, 32) * 0.01)
        
        # Add more parameters if needed
        remaining = num_params - sum(p.numel() for p in self.parameters())
        if remaining > 0:
            self.extra_weight = nn.Parameter(torch.randn(remaining) * 0.01)
    
    def __class__(self):
        # Mock for HF model detection
        return type("MockTransformerModel", (), {"__name__": "MockTransformerModel"})


@dataclass
class LogEvent:
    """Represents a single on_log event."""
    step: int
    loss: float
    grad_norm: float
    learning_rate: float = 5e-4
    additional_logs: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to logs dictionary for callback."""
        logs = {
            "loss": self.loss,
            "grad_norm": self.grad_norm,
            "learning_rate": self.learning_rate,
        }
        if self.additional_logs:
            logs.update(self.additional_logs)
        return logs


@dataclass 
class GuardScenario:
    """Defines a complete Guard testing scenario."""
    name: str
    description: str
    guard_config: Dict[str, Any]
    log_events: List[LogEvent]
    expected_alerts: List[str]  # Alert codes that should be present
    expected_rollbacks: int = 0
    expected_aborts: int = 0


# ============================================================================
# CPU Driver Implementation
# ============================================================================

class HFCallbackDriver:
    """
    CPU-friendly driver for testing HuggingFace callbacks without Trainer overhead.
    
    This class:
    - Creates minimal mock HF objects (TrainingArguments, TrainerState, etc.)
    - Runs callback lifecycle (on_train_begin → on_log events → on_train_end)  
    - Provides convenient assertion methods for telemetry validation
    - Handles temporary directories and cleanup automatically
    """
    
    def __init__(self, 
                 callback_class,
                 callback_config: Dict[str, Any],
                 model: Optional[nn.Module] = None,
                 temp_dir: Optional[str] = None):
        """
        Initialize the callback driver.
        
        Args:
            callback_class: The callback class to test (e.g., GradienceCallback)
            callback_config: Configuration dict for the callback
            model: Optional model to use (creates MockModel if None)
            temp_dir: Optional temporary directory (creates one if None)
        """
        self.callback_class = callback_class
        self.callback_config = callback_config
        self.model = model or MockModel()
        
        # Setup temporary directory
        if temp_dir:
            self.output_dir = Path(temp_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self._temp_dir = None
        else:
            self._temp_dir = tempfile.TemporaryDirectory()
            self.output_dir = Path(self._temp_dir.name) / "callback_test"
            self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Update callback config with output directory
        self.callback_config = dict(callback_config)
        self.callback_config.setdefault("output_dir", str(self.output_dir))
        self.callback_config.setdefault("filename", "test_run.jsonl")
        
        # Create callback with proper config object
        from gradience.vnext.integrations.hf import GradienceCallbackConfig
        config_obj = GradienceCallbackConfig(**self.callback_config)
        self.callback = self.callback_class(config_obj)
        
        # Setup mock HF objects
        self._setup_mock_objects()
        
        # Track events and results
        self.events: List[Dict[str, Any]] = []
        self.telemetry_file: Optional[Path] = None
        
    def _setup_mock_objects(self):
        """Create minimal mock HF training objects."""
        # Mock TrainingArguments
        self.mock_args = Mock()
        self.mock_args.output_dir = str(self.output_dir)
        self.mock_args.seed = 42
        self.mock_args.per_device_train_batch_size = 2
        self.mock_args.gradient_accumulation_steps = 1
        self.mock_args.max_steps = None
        self.mock_args.num_train_epochs = 1.0
        self.mock_args.learning_rate = self.callback_config.get("learning_rate", 5e-4)
        self.mock_args.weight_decay = 0.01
        self.mock_args.adam_beta1 = 0.9
        self.mock_args.adam_beta2 = 0.999
        self.mock_args.adam_epsilon = 1e-8
        self.mock_args.optim = "adamw_torch"
        self.mock_args.fp16 = False
        self.mock_args.bf16 = False
        
        # Mock TrainerState
        self.mock_state = Mock()
        self.mock_state.global_step = 0
        
        # Mock TrainerControl
        self.mock_control = Mock()
    
    def run_scenario(self, scenario: GuardScenario) -> 'CallbackResult':
        """
        Run a complete Guard scenario and return results.
        
        Args:
            scenario: The scenario to execute
            
        Returns:
            CallbackResult with telemetry and assertion helpers
        """
        return self.run_events(
            log_events=scenario.log_events,
            scenario_name=scenario.name
        )
    
    def run_events(self, 
                   log_events: List[LogEvent], 
                   scenario_name: str = "test") -> 'CallbackResult':
        """
        Run the callback lifecycle with specified log events.
        
        Args:
            log_events: List of LogEvent objects to simulate
            scenario_name: Name for this test run
            
        Returns:
            CallbackResult with telemetry and assertion helpers
        """
        # 1. Initialize training
        self.callback.on_train_begin(
            self.mock_args, 
            self.mock_state, 
            self.mock_control, 
            model=self.model
        )
        
        # 2. Run log events
        for log_event in log_events:
            self.mock_state.global_step = log_event.step
            logs = log_event.to_dict()
            
            self.callback.on_log(
                self.mock_args,
                self.mock_state, 
                self.mock_control,
                logs=logs,
                model=self.model
            )
        
        # 3. End training
        self.callback.on_train_end(
            self.mock_args,
            self.mock_state,
            self.mock_control
        )
        
        # 4. Read telemetry
        self.telemetry_file = self.output_dir / self.callback_config.get("filename", "test_run.jsonl")
        self._load_telemetry()
        
        return CallbackResult(
            events=self.events,
            telemetry_file=self.telemetry_file,
            callback=self.callback,
            scenario_name=scenario_name
        )
    
    def _load_telemetry(self):
        """Load telemetry events from the output file."""
        self.events = []
        if self.telemetry_file and self.telemetry_file.exists():
            with open(self.telemetry_file) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            self.events.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass  # Skip malformed lines
    
    def cleanup(self):
        """Clean up temporary resources."""
        if self._temp_dir:
            self._temp_dir.cleanup()


# ============================================================================
# Result Analysis Helper
# ============================================================================

class CallbackResult:
    """
    Helper class for analyzing callback test results and making assertions.
    """
    
    def __init__(self, 
                 events: List[Dict[str, Any]], 
                 telemetry_file: Path,
                 callback: Any,
                 scenario_name: str):
        self.events = events
        self.telemetry_file = telemetry_file
        self.callback = callback
        self.scenario_name = scenario_name
        
        # Pre-compute common event filters
        self.alerts = [e for e in events if e.get("event") == "alert"]
        self.guard_alerts = [e for e in self.alerts if e.get("code", "").startswith("GUARD_")]
        self.guard_metrics = [e for e in events if e.get("event") == "metrics" and e.get("kind") == "guard"]
    
    def assert_alert_present(self, alert_code: str, message: str = None) -> Dict[str, Any]:
        """
        Assert that a specific alert code is present in telemetry.
        
        Args:
            alert_code: The alert code to look for (e.g., "GUARD_ROLLBACK")
            message: Optional message substring to check
            
        Returns:
            The matching alert event
            
        Raises:
            AssertionError: If alert is not found
        """
        matching = [e for e in self.guard_alerts if e.get("code") == alert_code]
        assert matching, f"Alert '{alert_code}' not found in telemetry. Available: {[e.get('code') for e in self.guard_alerts]}"
        
        if message:
            msg_matching = [e for e in matching if message.lower() in e.get("message", "").lower()]
            assert msg_matching, f"Alert '{alert_code}' found but message doesn't contain '{message}'"
            return msg_matching[0]
        
        return matching[0]
    
    def assert_rollback_count(self, expected_count: int):
        """Assert the number of rollbacks performed."""
        if hasattr(self.callback, 'guard') and self.callback.guard:
            actual_count = self.callback.guard.n_rollbacks
            assert actual_count == expected_count, f"Expected {expected_count} rollbacks, got {actual_count}"
        else:
            # Fallback: count GUARD_ROLLBACK alerts
            rollback_alerts = [e for e in self.guard_alerts if e.get("code") == "GUARD_ROLLBACK"]
            assert len(rollback_alerts) == expected_count, f"Expected {expected_count} rollback alerts, got {len(rollback_alerts)}"
    
    def assert_sequence(self, expected_sequence: List[str]):
        """
        Assert that Guard alerts occur in a specific sequence.
        
        Args:
            expected_sequence: List of alert codes in expected order
        """
        actual_sequence = [e.get("code") for e in self.guard_alerts if e.get("code") in expected_sequence]
        assert actual_sequence == expected_sequence, f"Expected sequence {expected_sequence}, got {actual_sequence}"
    
    def get_alert_steps(self, alert_code: str) -> List[int]:
        """Get the steps at which a specific alert occurred."""
        steps = []
        for alert in self.guard_alerts:
            if alert.get("code") == alert_code:
                # Check both direct step and metadata.step
                step = alert.get("step") or alert.get("metadata", {}).get("step")
                if step is not None:
                    steps.append(step)
        return steps
    
    def get_metrics_by_action(self, action: str) -> List[Dict[str, Any]]:
        """Get Guard metrics events for a specific action."""
        return [e for e in self.guard_metrics if e.get("metrics", {}).get("action") == action]
    
    def print_summary(self):
        """Print a human-readable summary of the test results."""
        print(f"\n=== {self.scenario_name} Results ===")
        print(f"Total events: {len(self.events)}")
        print(f"Guard alerts: {len(self.guard_alerts)}")
        print(f"Guard metrics: {len(self.guard_metrics)}")
        
        if self.guard_alerts:
            print("\nGuard alert sequence:")
            for i, alert in enumerate(self.guard_alerts, 1):
                code = alert.get("code")
                step = alert.get("step") or alert.get("metadata", {}).get("step")
                message = alert.get("message", "")[:50] + "..."
                print(f"  {i}. {code} (step {step}): {message}")
        
        if hasattr(self.callback, 'guard') and self.callback.guard:
            print(f"\nGuard state:")
            print(f"  Rollbacks: {self.callback.guard.n_rollbacks}")
            print(f"  Snapshots: {self.callback.guard.snapshot_count()}")


# ============================================================================
# Common Scenarios
# ============================================================================

def create_grad_explosion_scenario(
    trigger_step: int = 10, 
    grad_norm: float = 500.0,
    guard_config_overrides: Optional[Dict[str, Any]] = None
) -> GuardScenario:
    """Create a basic gradient explosion scenario."""
    
    guard_config = {
        "enable_guard": True,
        "guard_snapshot_every": 5,
        "guard_grad_threshold": 100.0,
        "guard_cooldown_steps": 0,
        "guard_max_rollbacks": 3,
        "guard_window_steps": 50,
    }
    if guard_config_overrides:
        guard_config.update(guard_config_overrides)
    
    # Normal steps leading up to trigger
    normal_steps = [LogEvent(step=i, loss=2.5, grad_norm=1.0) for i in range(1, trigger_step)]
    
    # Trigger step
    trigger_event = LogEvent(step=trigger_step, loss=2.3, grad_norm=grad_norm)
    
    return GuardScenario(
        name="grad_explosion",
        description=f"Gradient explosion at step {trigger_step}",
        guard_config=guard_config,
        log_events=normal_steps + [trigger_event],
        expected_alerts=["GUARD_INIT", "GUARD_TRIGGERED", "GUARD_ROLLBACK"],
        expected_rollbacks=1
    )


def create_cooldown_scenario(
    first_step: int = 10,
    second_step: int = 15, 
    cooldown_steps: int = 20
) -> GuardScenario:
    """Create a cooldown lockout scenario."""
    
    guard_config = {
        "enable_guard": True,
        "guard_snapshot_every": 1,
        "guard_grad_threshold": 100.0,
        "guard_cooldown_steps": cooldown_steps,
        "guard_max_rollbacks": 999,
        "guard_window_steps": 100,
    }
    
    # Normal steps, first trigger, second trigger
    log_events = (
        [LogEvent(step=i, loss=2.5, grad_norm=1.0) for i in range(1, first_step)] +
        [LogEvent(step=first_step, loss=2.3, grad_norm=500.0)] +  # First trigger
        [LogEvent(step=second_step, loss=2.1, grad_norm=800.0)]   # Second trigger (should abort)
    )
    
    return GuardScenario(
        name="cooldown_lockout",
        description=f"Cooldown protection between steps {first_step} and {second_step}",
        guard_config=guard_config,
        log_events=log_events,
        expected_alerts=["GUARD_INIT", "GUARD_TRIGGERED", "GUARD_ROLLBACK", "GUARD_TRIGGERED", "GUARD_ABORT"],
        expected_rollbacks=1,
        expected_aborts=1
    )


def create_max_rollbacks_scenario(
    first_step: int = 10,
    second_step: int = 11,
    max_rollbacks: int = 1
) -> GuardScenario:
    """Create a max rollbacks lockout scenario."""
    
    guard_config = {
        "enable_guard": True,
        "guard_snapshot_every": 1,
        "guard_grad_threshold": 100.0,
        "guard_cooldown_steps": 0,  # Disabled
        "guard_max_rollbacks": max_rollbacks,
        "guard_window_steps": 99999,
    }
    
    # Normal steps, first trigger, second trigger  
    log_events = (
        [LogEvent(step=i, loss=2.5, grad_norm=1.0) for i in range(1, first_step)] +
        [LogEvent(step=first_step, loss=2.3, grad_norm=500.0)] +  # First trigger
        [LogEvent(step=second_step, loss=2.1, grad_norm=800.0)]   # Second trigger (should abort)
    )
    
    return GuardScenario(
        name="max_rollbacks_lockout", 
        description=f"Max rollbacks protection between steps {first_step} and {second_step}",
        guard_config=guard_config,
        log_events=log_events,
        expected_alerts=["GUARD_INIT", "GUARD_TRIGGERED", "GUARD_ROLLBACK", "GUARD_TRIGGERED", "GUARD_ABORT"],
        expected_rollbacks=1,
        expected_aborts=1
    )