"""
Enhanced stage context manager that combines heartbeat logging with watchdog monitoring.

Provides comprehensive monitoring for critical ML operations including:
- Progress heartbeat logging
- Hang detection and diagnostics
- Stage boundary logging
- Artifact tracking
"""

from pathlib import Path

from .heartbeat import start_heartbeat, stop_heartbeat
from .watchdog import WatchdogContext, setup_stack_trace_handler


class monitored_stage:
    """
    Enhanced context manager that combines heartbeat logging with hang detection.

    Automatically sets appropriate timeouts based on stage type and provides
    comprehensive diagnostics if hangs are detected.

    Usage:
        with monitored_stage("eval_probe", output_dir="/path", stage_type="evaluation") as stage:
            # Long-running operation
            results = evaluate_model()
            stage.add_artifact("eval.json")
            stage.progress()  # Reset watchdog timer
    """

    # Timeout configurations for different stage types (minutes)
    TIMEOUTS = {
        "evaluation": 45,  # Model evaluation can be slow
        "training": 120,  # Training is the longest operation
        "audit": 30,  # Audit should be relatively quick
        "file_io": 15,  # File operations should be fast
        "generation": 30,  # Config/report generation
        "default": 30,  # Default timeout
    }

    def __init__(
        self,
        stage_name: str,
        output_dir: str | Path | None = None,
        stage_type: str = "default",
        timeout_minutes: int | None = None,
        heartbeat_interval: int = 30,
        log_to_file: bool = True,
        seed: int | None = None,
        enable_watchdog: bool = True,
    ):
        """
        Initialize monitored stage.

        Args:
            stage_name: Name of the stage being monitored
            output_dir: Directory for logs and outputs
            stage_type: Type of stage (evaluation, training, audit, file_io, generation)
            timeout_minutes: Override default timeout for this stage type
            heartbeat_interval: Seconds between heartbeat messages
            log_to_file: Whether to write heartbeat logs to file
            seed: Random seed for the run
            enable_watchdog: Whether to enable hang detection (disable for testing)
        """
        self.stage_name = stage_name
        self.output_dir = Path(output_dir) if output_dir else None
        self.stage_type = stage_type
        self.heartbeat_interval = heartbeat_interval
        self.log_to_file = log_to_file
        self.seed = seed
        self.enable_watchdog = enable_watchdog

        # Determine timeout
        self.timeout_minutes = timeout_minutes or self.TIMEOUTS.get(stage_type, self.TIMEOUTS["default"])

        # Artifact tracking
        self.artifacts: list[str | Path] = []

        # Watchdog context
        self.watchdog: WatchdogContext | None = None

        # Log file for diagnostics
        self.log_file = None
        if self.output_dir and log_to_file:
            self.log_file = self.output_dir / "heartbeat.log"

    def add_artifact(self, artifact_path: str | Path) -> None:
        """Add an artifact that will be reported at stage end."""
        self.artifacts.append(artifact_path)

    def progress(self, message: str | None = None) -> None:
        """
        Signal that progress has been made (resets watchdog timer).

        Args:
            message: Optional progress message to log
        """
        if self.watchdog:
            self.watchdog.reset_progress()

        if message:
            print(f"[progress] {self.stage_name}: {message}")

    def __enter__(self):
        print(f"🚀 Starting monitored stage: {self.stage_name} (type: {self.stage_type})")

        # Start heartbeat logging
        start_heartbeat(self.stage_name, self.output_dir, self.heartbeat_interval, self.log_to_file, self.seed)

        # Start watchdog if enabled
        if self.enable_watchdog:
            self.watchdog = WatchdogContext(
                stage_name=self.stage_name,
                timeout_minutes=self.timeout_minutes,
                log_file=self.log_file,
                output_dir=self.output_dir,
            )
            self.watchdog.__enter__()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Stop watchdog first
        if self.watchdog:
            self.watchdog.__exit__(exc_type, exc_val, exc_tb)

        # Stop heartbeat logging
        stop_heartbeat(self.artifacts if self.artifacts else None)

        if exc_type is not None:
            print(f"💥 Stage '{self.stage_name}' failed with {exc_type.__name__}: {exc_val}")
        else:
            print(f"✅ Monitored stage completed: {self.stage_name}")


# Convenience functions for common stage types
def monitor_evaluation(stage_name: str, output_dir: str | Path | None = None, **kwargs):
    """Create monitored stage for evaluation operations."""
    return monitored_stage(stage_name, output_dir, stage_type="evaluation", **kwargs)


def monitor_training(stage_name: str, output_dir: str | Path | None = None, **kwargs):
    """Create monitored stage for training operations."""
    return monitored_stage(stage_name, output_dir, stage_type="training", **kwargs)


def monitor_file_operations(stage_name: str, output_dir: str | Path | None = None, **kwargs):
    """Create monitored stage for file I/O operations."""
    return monitored_stage(stage_name, output_dir, stage_type="file_io", **kwargs)


def monitor_audit(stage_name: str, output_dir: str | Path | None = None, **kwargs):
    """Create monitored stage for audit operations."""
    return monitored_stage(stage_name, output_dir, stage_type="audit", **kwargs)


def monitor_generation(stage_name: str, output_dir: str | Path | None = None, **kwargs):
    """Create monitored stage for config/report generation."""
    return monitored_stage(stage_name, output_dir, stage_type="generation", **kwargs)


def setup_global_monitoring():
    """
    Set up global monitoring infrastructure.

    Should be called once at the start of the bench process.
    """
    print("🔧 Setting up global monitoring infrastructure...")
    setup_stack_trace_handler()
    print("📡 Global monitoring ready")


# For backwards compatibility, provide an alias
heartbeat_stage_with_watchdog = monitored_stage
